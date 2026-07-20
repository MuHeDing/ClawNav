#!/usr/bin/env python
"""Capture a deterministic frozen Qwen instruction-stage manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

import requests

from harness.openclaw.instruction_stages import (
    StageFallbackCategory,
    parse_instruction_stage_plan,
)


PROMPT_SCHEMA_FINGERPRINT = hashlib.sha256(
    b"instruction_stages_v1:ordered:bounded:no_oracle:no_reasoning"
).hexdigest()


def generate_stage_manifest(
    episodes: Iterable[Dict[str, Any]],
    episode_keys: Sequence[str],
    segment_instruction: Callable[[str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    indexed = {_episode_key(row): row for row in episodes}
    requested = sorted(dict.fromkeys(str(key) for key in episode_keys if str(key)))
    missing = [key for key in requested if key not in indexed]
    if missing:
        raise ValueError("requested episode keys are missing: " + ", ".join(missing))
    rows = []
    for episode_key in requested:
        episode = indexed[episode_key]
        scene_id, episode_id = episode_key.rsplit(":", 1)
        instruction = _instruction_text(episode)
        response = segment_instruction(scene_id, episode_id, instruction)
        if not isinstance(response, dict):
            raise ValueError(f"segmentation response invalid for {episode_key}")
        metadata = response.get("runtime_metadata")
        stage_payload = response.get("stage_plan")
        if not isinstance(metadata, dict) or not isinstance(stage_payload, dict):
            raise ValueError(f"segmentation response invalid for {episode_key}")
        if str(metadata.get("fallback_category") or "none") != "none":
            raise ValueError(f"segmentation fallback is not allowed for {episode_key}")
        plan = parse_instruction_stage_plan(stage_payload, instruction)
        if plan.fallback_category is not StageFallbackCategory.NONE:
            raise ValueError(f"segmentation schema invalid for {episode_key}")
        rows.append(
            {
                "episode_key": episode_key,
                "instruction_sha256": plan.instruction_sha256,
                "stages": [stage.to_dict() for stage in plan.stages],
                "stage_plan_sha256": plan.stage_plan_sha256,
                "model_prompt_schema_fingerprint": PROMPT_SCHEMA_FINGERPRINT,
                "generation_status": "ok",
            }
        )
    return {"schema_version": "staged_stage_plan_manifest_v1", "rows": rows}


def _canonical_scene_id(value: Any) -> str:
    name = Path(str(value or "")).name
    return name[:-4] if name.endswith(".glb") else name


def _episode_key(row: Dict[str, Any]) -> str:
    return f"{_canonical_scene_id(row.get('scene_id'))}:{row.get('episode_id', '')}"


def _instruction_text(row: Dict[str, Any]) -> str:
    instruction = row.get("instruction")
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text") or instruction.get("text")
    text = str(instruction or row.get("instruction_text") or "").strip()
    if not text:
        raise ValueError(f"episode {_episode_key(row)} has no instruction")
    return text


def _load_episodes(path: Path) -> list[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("dataset must contain an episodes list")
    return [row for row in rows if isinstance(row, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--episode-keys", required=True)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8013")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False

    def segment(scene_id: str, episode_id: str, instruction: str) -> Dict[str, Any]:
        response = session.post(
            f"{args.gateway_url.rstrip('/')}/segment_instruction",
            json={
                "scene_id": scene_id,
                "episode_id": episode_id,
                "instruction": instruction,
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("gateway segmentation response must be an object")
        return value

    manifest = generate_stage_manifest(
        _load_episodes(args.data_path),
        args.episode_keys.split(","),
        segment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
