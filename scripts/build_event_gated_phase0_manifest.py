#!/usr/bin/env python
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from harness.visual_readback.manifest import (
    EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION,
    compute_event_gated_phase0_manifest_sha256,
    load_event_gated_phase0_manifest_jsonl,
)


DEFAULT_THRESHOLDS = {
    "route_event_miss_rate_max": 0.0,
    "exact_duplicate_rate_max": 0.0,
    "adjacent_triplet_rate_max": 0.0,
    "ambiguous_evidence_rate_max": 0.0,
    "comparison_direction": "lower_is_better",
    "invalid_comparison_behavior": "mark_invalid",
}


def build_manifest_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    seed = _load_seed(args.seed_file) if args.seed_file else {}
    cases = list(seed.get("cases") or [])
    if not cases and args.episode_key:
        cases = [_case_from_episode_key(key, args.max_steps) for key in args.episode_key]
    if not cases:
        raise ValueError("provide --seed_file with cases or at least one --episode_key")

    route_event_labels = _load_json_argument(
        args.route_event_labels_json,
        seed.get("route_event_labels"),
        "route_event_labels",
    )
    shared_trigger_keys = _load_json_argument(
        args.shared_readback_trigger_keys_json,
        seed.get("shared_readback_trigger_keys"),
        "shared_readback_trigger_keys",
    )
    thresholds = _load_json_argument(
        args.phase0_stop_go_thresholds_json,
        seed.get("phase0_stop_go_thresholds") or DEFAULT_THRESHOLDS,
        "phase0_stop_go_thresholds",
    )
    source_paths = [Path(path) for path in args.source_file]
    source_hashes = dict(seed.get("source_file_hashes") or {})
    source_hashes.update({str(path): _sha256_file(path) for path in source_paths})
    if not source_hashes:
        raise ValueError("provide --source_file or seed source_file_hashes")

    rows: List[Dict[str, Any]] = []
    for case in cases:
        scene_id = str(case.get("scene_id") or "")
        episode_id = str(case.get("episode_id") or "")
        if not scene_id or not episode_id:
            raise ValueError("each case must include scene_id and episode_id")
        rows.append(
            {
                "manifest_schema_version": EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "max_steps": int(case.get("max_steps", args.max_steps)),
                "route_event_labels": _labels_for_case(route_event_labels, scene_id, episode_id),
                "accepted_step_tolerance": int(
                    case.get(
                        "accepted_step_tolerance",
                        seed.get("accepted_step_tolerance", args.accepted_step_tolerance),
                    )
                ),
                "shared_readback_trigger_keys": _keys_for_case(
                    shared_trigger_keys, scene_id, episode_id
                ),
                "trigger_key_source": str(
                    case.get("trigger_key_source")
                    or seed.get("trigger_key_source")
                    or args.trigger_key_source
                ),
                "phase0_stop_go_thresholds": thresholds,
                "selection_rules": seed.get("selection_rules") or args.selection_rules,
                "selection_seed": int(seed.get("selection_seed", args.selection_seed)),
                "source_dataset_id": str(
                    case.get("source_dataset_id")
                    or seed.get("source_dataset_id")
                    or args.source_dataset_id
                ),
                "source_run_id": str(
                    case.get("source_run_id")
                    or seed.get("source_run_id")
                    or args.source_run_id
                ),
                "source_file_hashes": source_hashes,
            }
        )
    _require_common_fields(rows)
    manifest_sha256 = compute_event_gated_phase0_manifest_sha256(rows)
    for row in rows:
        row["manifest_sha256"] = manifest_sha256
    return rows


def write_manifest(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    load_event_gated_phase0_manifest_jsonl(output_path)


def _load_seed(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("seed file must contain a JSON object")
    return data


def _case_from_episode_key(key: str, max_steps: int) -> Dict[str, Any]:
    parts = key.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("--episode_key must use scene_id:episode_id")
    return {"scene_id": parts[0], "episode_id": parts[1], "max_steps": max_steps}


def _load_json_argument(path: str, default: Any, name: str) -> Any:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        data = default
    if data is None:
        raise ValueError(f"provide {name}")
    return data


def _labels_for_case(labels: Any, scene_id: str, episode_id: str) -> List[Dict[str, Any]]:
    if isinstance(labels, list):
        return [dict(label) for label in labels]
    if isinstance(labels, dict):
        key = f"{scene_id}:{episode_id}"
        value = labels.get(key) or labels.get(episode_id)
        if isinstance(value, list):
            return [dict(label) for label in value]
    raise ValueError(f"missing route_event_labels for {scene_id}:{episode_id}")


def _keys_for_case(keys: Any, scene_id: str, episode_id: str) -> List[str]:
    if isinstance(keys, list):
        return [str(key) for key in keys]
    if isinstance(keys, dict):
        key = f"{scene_id}:{episode_id}"
        value = keys.get(key) or keys.get(episode_id)
        if isinstance(value, list):
            return [str(item) for item in value]
    raise ValueError(f"missing shared_readback_trigger_keys for {scene_id}:{episode_id}")


def _sha256_file(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"source file does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_common_fields(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        for key in ("trigger_key_source", "source_dataset_id", "source_run_id", "selection_rules"):
            if not str(row.get(key) or ""):
                raise ValueError(f"{key} is required")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed_file", default="")
    parser.add_argument("--episode_key", action="append", default=[])
    parser.add_argument("--route_event_labels_json", default="")
    parser.add_argument("--shared_readback_trigger_keys_json", default="")
    parser.add_argument("--phase0_stop_go_thresholds_json", default="")
    parser.add_argument("--accepted_step_tolerance", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--selection_seed", type=int, default=0)
    parser.add_argument("--selection_rules", default="predeclared_phase0_seed")
    parser.add_argument("--trigger_key_source", default="")
    parser.add_argument("--source_dataset_id", default="")
    parser.add_argument("--source_run_id", default="")
    parser.add_argument("--source_file", action="append", default=[])
    args = parser.parse_args(argv)

    rows = build_manifest_rows(args)
    write_manifest(rows, Path(args.output))
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"manifest_sha256={rows[0]['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
