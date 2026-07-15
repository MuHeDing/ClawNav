#!/usr/bin/env python
"""Calibrate Habitat discrete action effects from simulator pose deltas.

Example:
    PYTHONPATH=src python scripts/calibrate_habitat_action_scale.py \
        --habitat_config_path config/vln_r2r.yaml \
        --eval_split val_unseen \
        --episode_keys 2azQ1b91cZZ:11,2azQ1b91cZZ:16 \
        --repeat_count 12 \
        --output_path results/action_scale_calibration.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


ACTIONS2IDX = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}

DEFAULT_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure real Habitat pose deltas for discrete VLN actions."
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="config/vln_r2r.yaml",
    )
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument(
        "--episode_keys",
        type=str,
        default="",
        help="Comma-separated scene_id:episode_id keys. If omitted, uses max_episodes.",
    )
    parser.add_argument("--max_episodes", type=int, default=8)
    parser.add_argument("--repeat_count", type=int, default=20)
    parser.add_argument(
        "--actions",
        type=str,
        default=",".join(DEFAULT_ACTIONS),
        help="Comma-separated action names.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_threshold_m", type=float, default=0.05)
    parser.add_argument("--turn_threshold_deg", type=float, default=5.0)
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="JSONL output path. Defaults to results/action_scale_calibration_<timestamp>.jsonl.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    actions = parse_actions(args.actions)
    output_path = resolve_output_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    try:
        episodes = select_episodes(
            list(env.episodes),
            raw_keys=args.episode_keys,
            max_episodes=args.max_episodes,
        )
        summary_accumulator: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        with output_path.open("w", encoding="utf-8") as file:
            for episode in episodes:
                for action_name in actions:
                    records = calibrate_episode_action(
                        env=env,
                        episode=episode,
                        action_name=action_name,
                        repeat_count=args.repeat_count,
                        progress_threshold_m=args.progress_threshold_m,
                        turn_threshold_deg=args.turn_threshold_deg,
                    )
                    for record in records:
                        summary_accumulator[action_name].append(record)
                        file.write(json.dumps(record, sort_keys=True) + "\n")
        summary = summarize_records(summary_accumulator)
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote calibration records: {output_path}")
        print(f"Wrote calibration summary: {summary_path}")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


def make_env(args: argparse.Namespace):
    import habitat
    from habitat import Env
    from habitat_baselines.config.default import get_config as get_habitat_config
    from habitat.config.default_structured_configs import CollisionsMeasurementConfig
    from habitat_extensions import measures  # noqa: F401
    from habitat_extensions import sensors  # noqa: F401
    from habitat_extensions import task  # noqa: F401

    config = get_habitat_config(args.habitat_config_path)
    with habitat.config.read_write(config):
        if args.data_path:
            config.habitat.dataset.data_path = args.data_path
        if "{split}" in str(config.habitat.dataset.data_path):
            config.habitat.dataset.data_path = str(
                config.habitat.dataset.data_path
            ).format(split=args.eval_split)
        config.habitat.dataset.split = args.eval_split
        config.habitat.task.measurements.update(
            {"collisions": CollisionsMeasurementConfig()}
        )
    return Env(config=config)


def parse_actions(raw_actions: str) -> List[str]:
    actions = [
        action.strip().upper()
        for action in str(raw_actions or "").split(",")
        if action.strip()
    ]
    invalid = [action for action in actions if action not in ACTIONS2IDX]
    if invalid:
        raise ValueError(f"Unsupported actions: {', '.join(invalid)}")
    return actions or list(DEFAULT_ACTIONS)


def resolve_output_path(raw_path: str) -> Path:
    if raw_path:
        return Path(raw_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"action_scale_calibration_{timestamp}.jsonl"


def select_episodes(
    episodes: List[Any],
    raw_keys: str,
    max_episodes: int,
) -> List[Any]:
    requested_keys = [
        key.strip()
        for key in str(raw_keys or "").split(",")
        if key.strip()
    ]
    if requested_keys:
        episodes_by_key = {
            episode_key(episode): episode
            for episode in episodes
        }
        missing = [key for key in requested_keys if key not in episodes_by_key]
        if missing:
            raise ValueError(f"Episode keys not found: {', '.join(missing)}")
        return [episodes_by_key[key] for key in requested_keys]
    if max_episodes <= 0:
        return episodes
    return episodes[:max_episodes]


def episode_key(episode: Any) -> str:
    return (
        f"{canonical_scene_id(getattr(episode, 'scene_id', ''))}:"
        f"{getattr(episode, 'episode_id', '')}"
    )


def canonical_scene_id(scene_id: Any) -> str:
    raw = str(scene_id or "").rstrip("/").replace("\\", "/")
    parts = raw.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return Path(raw).stem


def calibrate_episode_action(
    env: Any,
    episode: Any,
    action_name: str,
    repeat_count: int,
    progress_threshold_m: float,
    turn_threshold_deg: float,
) -> List[Dict[str, Any]]:
    env.current_episode = episode
    env.reset()
    records: List[Dict[str, Any]] = []
    for step_index in range(max(0, repeat_count)):
        before = agent_pose(env)
        observations = env.step(ACTIONS2IDX[action_name])
        after = agent_pose(env)
        metrics = env.get_metrics()
        position_delta_m = position_delta(before.get("position"), after.get("position"))
        rotation_delta_deg = rotation_delta(
            before.get("rotation"),
            after.get("rotation"),
        )
        record = {
            "scene_id": canonical_scene_id(getattr(episode, "scene_id", "")),
            "episode_id": str(getattr(episode, "episode_id", "")),
            "action_name": action_name,
            "action_index": ACTIONS2IDX[action_name],
            "step_index": step_index,
            "before_position": before.get("position"),
            "after_position": after.get("position"),
            "before_rotation": before.get("rotation"),
            "after_rotation": after.get("rotation"),
            "position_delta_m": round_optional(position_delta_m),
            "rotation_delta_deg": round_optional(rotation_delta_deg),
            "collision": collision_from_metrics(metrics),
            "no_progress": no_progress(
                action_name,
                position_delta_m,
                rotation_delta_deg,
                progress_threshold_m=progress_threshold_m,
                turn_threshold_deg=turn_threshold_deg,
            ),
            "episode_over": bool(getattr(env, "episode_over", False)),
            "rgb_present": "rgb" in observations if isinstance(observations, dict) else False,
        }
        records.append(record)
        if getattr(env, "episode_over", False):
            break
    return records


def agent_pose(env: Any) -> Dict[str, Any]:
    state = env.sim.get_agent_state()
    return {
        "position": numeric_list(getattr(state, "position", None)),
        "rotation": numeric_list(getattr(state, "rotation", None)),
    }


def numeric_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if all(hasattr(value, attr) for attr in ("w", "x", "y", "z")):
        value = [value.w, value.x, value.y, value.z]
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def position_delta(
    before: Optional[Iterable[float]],
    after: Optional[Iterable[float]],
) -> Optional[float]:
    before_list = numeric_list(before)
    after_list = numeric_list(after)
    if before_list is None or after_list is None:
        return None
    dims = min(len(before_list), len(after_list))
    if dims == 0:
        return None
    return math.sqrt(
        sum((after_list[index] - before_list[index]) ** 2 for index in range(dims))
    )


def rotation_delta(before: Any, after: Any) -> Optional[float]:
    before_q = quaternion_wxyz(before)
    after_q = quaternion_wxyz(after)
    if before_q is None or after_q is None:
        return None
    dot = sum(before_q[index] * after_q[index] for index in range(4))
    dot = max(-1.0, min(1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def quaternion_wxyz(value: Any) -> Optional[List[float]]:
    values = numeric_list(value)
    if values is None or len(values) != 4:
        return None
    if abs(values[0]) >= abs(values[3]):
        ordered = [values[0], values[1], values[2], values[3]]
    else:
        ordered = [values[3], values[0], values[1], values[2]]
    norm = math.sqrt(sum(item * item for item in ordered))
    if norm <= 0:
        return None
    return [item / norm for item in ordered]


def collision_from_metrics(metrics: Dict[str, Any]) -> Optional[bool]:
    collisions = metrics.get("collisions") if isinstance(metrics, dict) else None
    if isinstance(collisions, dict) and "is_collision" in collisions:
        return bool(collisions["is_collision"])
    if isinstance(metrics, dict) and "collision" in metrics:
        return bool(metrics["collision"])
    return None


def no_progress(
    action_name: str,
    position_delta_m: Optional[float],
    rotation_delta_deg: Optional[float],
    *,
    progress_threshold_m: float,
    turn_threshold_deg: float,
) -> Optional[bool]:
    if action_name == "MOVE_FORWARD":
        if position_delta_m is None:
            return None
        return position_delta_m < progress_threshold_m
    if action_name in {"TURN_LEFT", "TURN_RIGHT"}:
        if rotation_delta_deg is None:
            return None
        return rotation_delta_deg < turn_threshold_deg
    return None


def summarize_records(
    records_by_action: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    summary = {}
    for action_name, records in records_by_action.items():
        position_values = numeric_record_values(records, "position_delta_m")
        rotation_values = numeric_record_values(records, "rotation_delta_deg")
        collisions = [record for record in records if record.get("collision") is True]
        no_progress_records = [
            record for record in records if record.get("no_progress") is True
        ]
        summary[action_name] = {
            "count": len(records),
            "position_delta_m_mean": mean(position_values),
            "position_delta_m_min": min(position_values) if position_values else None,
            "position_delta_m_max": max(position_values) if position_values else None,
            "rotation_delta_deg_mean": mean(rotation_values),
            "rotation_delta_deg_min": min(rotation_values) if rotation_values else None,
            "rotation_delta_deg_max": max(rotation_values) if rotation_values else None,
            "collision_count": len(collisions),
            "no_progress_count": len(no_progress_records),
        }
    return summary


def numeric_record_values(records: List[Dict[str, Any]], key: str) -> List[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def round_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


if __name__ == "__main__":
    main()
