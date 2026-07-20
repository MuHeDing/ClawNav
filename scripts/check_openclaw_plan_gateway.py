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


def validate_gateway_health(
    data: Dict[str, Any],
    require_service: str = "",
    client_timeout_s: float = 0.0,
    enforce_timeout_budget: bool = False,
    required_stage_schema: str = "",
    required_action_schema: str = "",
) -> None:
    if not isinstance(data, dict):
        raise ValueError("gateway health response must be an object")
    if not data.get("ok"):
        raise ValueError("gateway health is not ok")
    if require_service:
        service = str(data.get("service") or "")
        if service != require_service:
            raise ValueError(
                f"gateway service must be {require_service}; got {service or '<missing>'}"
            )
    if enforce_timeout_budget:
        budget = (
            data.get("timeout_budget")
            if isinstance(data.get("timeout_budget"), dict)
            else {}
        )
        recommended = budget.get("recommended_gateway_timeout_s")
        if recommended is None:
            return
        try:
            recommended_timeout_s = float(recommended)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "timeout_budget.recommended_gateway_timeout_s must be numeric"
            ) from exc
        if client_timeout_s < recommended_timeout_s:
            raise ValueError(
                "OPENCLAW_GATEWAY_TIMEOUT is below the adapter timeout budget: "
                f"{client_timeout_s:g}s < recommended {recommended_timeout_s:g}s"
            )
    if required_stage_schema or required_action_schema:
        if data.get("instruction_segmentation") is not True:
            raise ValueError("gateway does not enable instruction segmentation")
    if required_stage_schema:
        supported = data.get("stage_schema_versions")
        if not isinstance(supported, list) or required_stage_schema not in supported:
            raise ValueError(
                f"gateway does not support stage schema {required_stage_schema}"
            )
        if data.get("active_stage_schema") != required_stage_schema:
            raise ValueError(
                f"gateway active stage schema is not {required_stage_schema}"
            )
    if required_action_schema:
        supported = data.get("action_schema_versions")
        if not isinstance(supported, list) or required_action_schema not in supported:
            raise ValueError(
                f"gateway does not support action schema {required_action_schema}"
            )
        if data.get("active_action_schema") != required_action_schema:
            raise ValueError(
                f"gateway active action schema is not {required_action_schema}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway_url", default="http://127.0.0.1:8011")
    parser.add_argument("--instruction", default="go to kitchen")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--require_service", default="")
    parser.add_argument("--enforce_timeout_budget", action="store_true")
    parser.add_argument("--required_stage_schema", default="")
    parser.add_argument("--required_action_schema", default="")
    args = parser.parse_args()

    session = requests.Session()
    session.trust_env = False
    health = session.get(f"{args.gateway_url.rstrip('/')}/health", timeout=args.timeout)
    health.raise_for_status()
    validate_gateway_health(
        health.json(),
        require_service=args.require_service,
        client_timeout_s=args.timeout,
        enforce_timeout_budget=args.enforce_timeout_budget,
        required_stage_schema=args.required_stage_schema,
        required_action_schema=args.required_action_schema,
    )
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
