#!/usr/bin/env python
import argparse
import json
from typing import Any, Dict

import requests


ALLOWED_INTENTS = {"act", "recall_memory", "write_memory", "verify_progress", "replan"}
REQUIRED_KEYS = {"intent", "tool_name", "arguments", "reason"}


def build_probe_payload(instruction: str) -> Dict[str, Any]:
    return {
        "state": {
            "scene_id": "gateway_check",
            "episode_id": "gateway_check",
            "instruction": instruction,
            "step_id": 0,
            "last_action": None,
        },
        "runtime_context": {
            "policy_action": None,
            "recent_actions": [],
        },
    }


def validate_plan_response(data: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"missing required keys: {', '.join(sorted(missing))}")
    if data["intent"] not in ALLOWED_INTENTS:
        raise ValueError(f"unsupported intent: {data['intent']}")
    if not isinstance(data["arguments"], dict):
        raise ValueError("arguments must be an object")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway_url", default="http://127.0.0.1:8011")
    parser.add_argument("--instruction", default="go to kitchen")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    session = requests.Session()
    session.trust_env = False
    health = session.get(f"{args.gateway_url.rstrip('/')}/health", timeout=args.timeout)
    health.raise_for_status()
    response = session.post(
        f"{args.gateway_url.rstrip('/')}/plan",
        json=build_probe_payload(args.instruction),
        timeout=args.timeout,
    )
    response.raise_for_status()
    data = response.json()
    validate_plan_response(data)
    print(json.dumps({"ok": True, "response": data}, sort_keys=True))


if __name__ == "__main__":
    main()
