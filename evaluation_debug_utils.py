from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _to_float_list(values: Optional[Iterable[Any]]) -> Optional[List[float]]:
    if values is None:
        return None
    return [float(v) for v in values]


def _sanitize_path_component(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._")
    return text or "unknown"


def canonical_scene_id(scene_id: str) -> str:
    scene_path = Path(str(scene_id))
    if scene_path.suffix:
        return scene_path.stem
    return scene_path.name


def build_episode_path_record(
    episode: Any,
    scene_id: str,
    episode_instruction: str,
) -> Dict[str, Any]:
    goal_position = None
    if getattr(episode, "goals", None):
        first_goal = episode.goals[0]
        goal_position = _to_float_list(getattr(first_goal, "position", None))

    reference_path = getattr(episode, "reference_path", None) or []
    return {
        "scene_id": scene_id,
        "episode_id": str(getattr(episode, "episode_id", "")),
        "episode_instruction": episode_instruction,
        "start_position": _to_float_list(getattr(episode, "start_position", None)),
        "goal_position": goal_position,
        "reference_path": [_to_float_list(point) for point in reference_path],
    }


def build_model_step_record(
    *,
    step_id: int,
    action_text: str,
    action_id: Optional[int],
    position: Optional[Iterable[Any]],
    rotation: Optional[Iterable[Any]],
    distance_to_goal: Optional[float],
    episode_over: bool,
) -> Dict[str, Any]:
    return {
        "step_id": int(step_id),
        "action_text": action_text,
        "action_id": None if action_id is None else int(action_id),
        "position": _to_float_list(position),
        "rotation": _to_float_list(rotation),
        "distance_to_goal": None if distance_to_goal is None else float(distance_to_goal),
        "episode_over": bool(episode_over),
    }


def build_episode_qualitative_record(
    *,
    episode: Any,
    scene_id: str,
    episode_instruction: str,
    model_path: List[Dict[str, Any]],
) -> Dict[str, Any]:
    record = build_episode_path_record(
        episode=episode,
        scene_id=scene_id,
        episode_instruction=episode_instruction,
    )
    record["model_path"] = model_path
    return record


def should_save_step_artifacts(
    *,
    save_step_artifacts: bool,
    should_save_video: bool,
    save_step_artifacts_with_video_only: bool,
) -> bool:
    if not save_step_artifacts:
        return False
    if save_step_artifacts_with_video_only:
        return bool(should_save_video)
    return True


def resolve_qualitative_output_path(
    *,
    output_path: Path,
    rank: int,
    disabled: bool,
) -> Optional[Path]:
    if disabled:
        return None
    return Path(output_path) / f"qualitative_trajectories_rank{int(rank)}.json"


def format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{float(value):.2f}"


def extract_multi_goal_positions(episode: Dict[str, Any]) -> List[List[float]]:
    positions: List[List[float]] = []
    for goal_item in episode.get("multi_goals", []):
        if not isinstance(goal_item, dict):
            continue
        for key, value in goal_item.items():
            if "position" not in str(key):
                continue
            if not isinstance(value, (list, tuple)) or len(value) < 3:
                continue
            positions.append([float(value[0]), float(value[1]), float(value[2])])
    return positions


def sanitize_json_text(raw: str) -> str:
    return re.sub(r",(\s*[\]}])", r"\1", raw)


def build_episode_multi_goal_lookup(raw: str) -> Dict[tuple, List[List[float]]]:
    payload = json.loads(sanitize_json_text(raw))
    lookup: Dict[tuple, List[List[float]]] = {}
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        return lookup

    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        positions = extract_multi_goal_positions(episode)
        if not positions:
            continue
        lookup[(canonical_scene_id(episode.get("scene_id", "")), str(episode.get("episode_id", "")))] = positions

    return lookup


def build_topdown_goal_display_settings(
    episode_multi_goals: Dict[tuple, List[List[float]]],
) -> Dict[str, bool]:
    has_multi_goals = bool(episode_multi_goals)
    return {
        "draw_goal_positions": not has_multi_goals,
        "draw_goal_aabbs": not has_multi_goals,
    }


def multi_goal_overlay_pad_meters() -> float:
    return 0.3


def resolve_step_image_output_path(
    *,
    output_path: Path,
    scene_id: str,
    episode_id: str,
    step_id: int,
) -> Path:
    episode_dir = Path(output_path) / "step_images" / (
        f"{_sanitize_path_component(scene_id)}_{_sanitize_path_component(episode_id)}"
    )
    return episode_dir / f"step_{int(step_id):04d}.png"


def resolve_step_map_output_path(
    *,
    output_path: Path,
    scene_id: str,
    episode_id: str,
    step_id: int,
) -> Path:
    episode_dir = Path(output_path) / "step_maps" / (
        f"{_sanitize_path_component(scene_id)}_{_sanitize_path_component(episode_id)}"
    )
    return episode_dir / f"step_{int(step_id):04d}.png"


def resolve_sanitized_vln_dataset_path(
    *,
    output_path: Path,
    dataset_path: str,
    rank: int,
) -> Path:
    dataset_name = Path(dataset_path).name
    sanitized_stub = dataset_name.replace(".json.gz", "").replace(".json", "")
    return Path(output_path) / f"{sanitized_stub}.sanitized.rank{int(rank)}.json.gz"


def normalize_vln_dataset_json_text(raw: str) -> str:
    sanitized = sanitize_json_text(raw)
    payload = json.loads(sanitized)

    instruction_vocab = payload.get("instruction_vocab")
    if not isinstance(instruction_vocab, dict):
        instruction_vocab = {}
        payload["instruction_vocab"] = instruction_vocab
    instruction_vocab.setdefault("word_list", [])

    episodes = payload.get("episodes")
    if isinstance(episodes, list):
        for episode in episodes:
            if isinstance(episode, dict):
                episode.pop("multi_goals", None)

    return json.dumps(payload, indent=2) + "\n"


def append_record_to_json_array_file(path: Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if not isinstance(existing, list):
            raise ValueError(f"Expected JSON array in {path}")
    else:
        existing = []

    existing.append(record)
    path.write_text(json.dumps(existing, indent=2) + "\n")
