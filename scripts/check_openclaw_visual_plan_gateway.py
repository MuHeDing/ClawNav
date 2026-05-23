#!/usr/bin/env python
import argparse
import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import requests

from harness.openclaw.visual_analyzer import OpenClawVisualAnalyzer
from scripts.check_openclaw_plan_gateway import (
    validate_gateway_health,
    validate_plan_response,
)


PROBE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeElEQVR4nO3XMQ6AIAwF0JT//2Z3"
    "U2cTg4nCaEmwMmkNxQJHyYv0NwCgCkCqAKQKQKoApApAqgCkCkCqAKQKQKoApApAqgCk"
    "CkCqAKQKQKoApApAqgCkCkCqAKQKQKoApApAqgCkCkCqAKQKQKoApApAqgCk/gAqKQTO"
    "aHObXwAAAABJRU5ErkJggg=="
)


def ensure_probe_image(image_path: str = "") -> str:
    if image_path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"visual probe image does not exist: {image_path}")
        return str(path)

    probe_dir = Path(tempfile.gettempdir()) / "clawnav_openclaw_visual_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    path = probe_dir / "probe.png"
    if not path.exists():
        path.write_bytes(base64.b64decode(PROBE_PNG_BASE64))
    return str(path)


def build_visual_probe_payload(
    instruction: str,
    image_path: str,
    visual_observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "state": {
            "scene_id": "visual_gateway_check",
            "episode_id": "visual_gateway_check",
            "instruction": instruction,
            "step_id": 0,
            "last_action": None,
        },
        "runtime_context": {
            "policy_action": None,
            "recent_actions": [],
            "current_image_path": image_path,
            "recent_keyframe_paths": [image_path],
            "visual_observations": visual_observations,
        },
    }


def validate_visual_observations(
    observations: List[Dict[str, Any]],
    allow_empty_visual: bool = False,
) -> None:
    if not observations:
        raise ValueError("openclaw image capability returned no observations")
    first = observations[0]
    if first.get("error"):
        raise ValueError(f"openclaw image capability failed: {first['error']}")
    if allow_empty_visual:
        return
    if not (first.get("visual_observation") or first.get("caption")):
        raise ValueError("openclaw image capability returned an empty visual observation")


def validate_visual_plan_response(data: Dict[str, Any]) -> None:
    validate_plan_response(data)
    runtime_metadata = data.get("runtime_metadata")
    if not isinstance(runtime_metadata, dict):
        raise ValueError("visual plan response missing runtime_metadata.visual_analysis")
    visual_analysis = runtime_metadata.get("visual_analysis")
    if not isinstance(visual_analysis, dict) or not visual_analysis.get("ran"):
        raise ValueError("visual plan response missing runtime_metadata.visual_analysis.ran")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway_url", default="http://127.0.0.1:8011")
    parser.add_argument("--instruction", default="go to the kitchen and use visual memory")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--image_path", default="")
    parser.add_argument("--model", default="qwen/qwen3.5-flash")
    parser.add_argument("--allow_empty_visual", action="store_true")
    parser.add_argument("--require_service", default="openclaw_cli_plan_gateway")
    args = parser.parse_args()

    session = requests.Session()
    session.trust_env = False
    health = session.get(f"{args.gateway_url.rstrip('/')}/health", timeout=args.timeout)
    health.raise_for_status()
    validate_gateway_health(health.json(), require_service=args.require_service)

    image_path = ensure_probe_image(args.image_path)
    analyzer = OpenClawVisualAnalyzer(
        model=args.model,
        timeout_ms=int(args.timeout * 1000),
    )
    observations = analyzer.analyze([image_path])
    validate_visual_observations(observations, allow_empty_visual=args.allow_empty_visual)
    response = session.post(
        f"{args.gateway_url.rstrip('/')}/plan",
        json=build_visual_probe_payload(args.instruction, image_path, observations),
        timeout=args.timeout,
    )
    response.raise_for_status()
    data = response.json()
    validate_visual_plan_response(data)
    print(
        json.dumps(
            {
                "ok": True,
                "image_path": image_path,
                "visual_observation": observations[0],
                "response": data,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
