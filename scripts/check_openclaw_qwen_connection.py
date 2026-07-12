#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from harness.openclaw.openclaw_cli_plan_gateway import QwenApiModelClient  # noqa: E402


ALLOWED_INTENTS = {"act", "recall_memory", "write_memory", "verify_progress", "replan"}
REQUIRED_PLAN_KEYS = {"intent", "tool_name", "arguments", "reason"}


def build_probe_payload(instruction: str) -> Dict[str, Any]:
    return {
        "state": {
            "scene_id": "openclaw_qwen_connection_check",
            "episode_id": "openclaw_qwen_connection_check",
            "instruction": instruction,
            "step_id": 0,
            "last_action": None,
        },
        "runtime_context": {
            "policy_action": None,
            "recent_actions": [],
            "keyframe_policy_mode": "event_gated_smoke",
        },
    }


def validate_plan_response(data: Dict[str, Any]) -> None:
    missing = REQUIRED_PLAN_KEYS - set(data)
    if missing:
        raise ValueError(f"missing plan response keys: {', '.join(sorted(missing))}")
    if data["intent"] not in ALLOWED_INTENTS:
        raise ValueError(f"unsupported intent: {data['intent']}")
    if not isinstance(data["arguments"], dict):
        raise ValueError("plan response arguments must be an object")


def check_qwen_api(args: argparse.Namespace) -> Dict[str, Any]:
    client = QwenApiModelClient(
        max_retries=args.retries,
        retry_backoff_s=args.retry_backoff_s,
    )
    client._ensure_config(args.timeout)
    if not client.api_key:
        raise RuntimeError(
            "Qwen API key was not resolved from env or OpenClaw auth profiles"
        )

    result = client.run(
        prompt=args.prompt,
        image_paths=[],
        model=args.model,
        timeout_s=args.timeout,
    )
    text = ""
    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            text = str(first.get("text") or "")
    return {
        "ok": bool(result.get("ok")),
        "check": "qwen_api_via_openclaw_config",
        "base_url": client.base_url,
        "api_key_resolved": bool(client.api_key),
        "model": args.model,
        "response_preview": text[:160],
        "usage": result.get("usage") or {},
        "request_attempts": result.get("request_attempts"),
        "retry_count": result.get("retry_count"),
    }


def check_gateway(args: argparse.Namespace) -> Dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    base = args.gateway_url.rstrip("/")
    health_response = session.get(f"{base}/health", timeout=args.timeout)
    health_response.raise_for_status()
    health = health_response.json()
    if not isinstance(health, dict) or not health.get("ok"):
        raise RuntimeError(f"gateway health is not ok: {health}")

    plan_response = session.post(
        f"{base}/plan",
        json=build_probe_payload(args.instruction),
        timeout=args.timeout,
    )
    plan_response.raise_for_status()
    plan = plan_response.json()
    if not isinstance(plan, dict):
        raise RuntimeError("gateway plan response must be a JSON object")
    validate_plan_response(plan)
    return {
        "ok": True,
        "check": "openclaw_gateway_plan",
        "gateway_url": base,
        "health": health,
        "plan": plan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Qwen/OpenClaw connectivity without running Habitat eval."
    )
    parser.add_argument(
        "--mode",
        choices=("api", "gateway", "both"),
        default="api",
        help="api checks Qwen through OpenClaw config; gateway checks a running /plan adapter.",
    )
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8013")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--prompt", default="Return exactly: OK")
    parser.add_argument("--instruction", default="go forward to the target")
    args = parser.parse_args()

    results = []
    try:
        if args.mode in {"api", "both"}:
            results.append(check_qwen_api(args))
        if args.mode in {"gateway", "both"}:
            results.append(check_gateway(args))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        raise SystemExit(1)

    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
