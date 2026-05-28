#!/usr/bin/env python
import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from harness.openclaw.openclaw_cli_plan_gateway import (
    OpenClawCliPlanPlanner,
    run_openclaw_command,
)
from scripts.summarize_openclaw_context_audit import (
    format_markdown,
    summarize_audit_file,
)


STEP0_MARKER = "STEP0_SECRET_MARKER_DO_NOT_LEAK"


class RecordingRunner:
    def __init__(
        self,
        mode: str,
        fake_fixed_overhead: int = 400,
        fake_growth_per_step: int = 0,
    ) -> None:
        self.mode = mode
        self.fake_fixed_overhead = int(fake_fixed_overhead)
        self.fake_growth_per_step = int(fake_growth_per_step)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, args: List[str], timeout_s: float) -> subprocess.CompletedProcess:
        message = _arg_after(args, "--message") or _arg_after(args, "--prompt")
        step_id = _step_id_from_message(message)
        self.calls.append(
            {
                "args": list(args),
                "timeout_s": timeout_s,
                "message": message,
                "step_id": step_id,
            }
        )
        if self.mode in {"agent", "model"}:
            return run_openclaw_command(args, timeout_s)
        input_tokens = max(1, len(message) // 4) + self.fake_fixed_overhead
        input_tokens += max(0, step_id) * self.fake_growth_per_step
        stdout = json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": f"stress_step_{step_id}",
                            }
                        )
                    }
                ],
                "usage": {
                    "input": input_tokens,
                    "output": 8,
                    "totalTokens": input_tokens + 8,
                },
            }
        )
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


def run_context_stress(
    output_dir: Path,
    steps: int = 100,
    mode: str = "fake",
    fake_fixed_overhead: int = 400,
    fake_growth_per_step: int = 0,
    openclaw_profile: str = "",
    agent_id: str = "main",
    agent_timeout_s: float = 120.0,
    agent_max_input_tokens: int = 50000,
    openclaw_model: str = "",
    openclaw_model_provider: str = "openclaw_cli",
    openclaw_model_max_images: int = 3,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = RecordingRunner(
        mode=mode,
        fake_fixed_overhead=fake_fixed_overhead,
        fake_growth_per_step=fake_growth_per_step,
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="model" if mode == "model" else "agent",
        agent_id=agent_id,
        agent_timeout_s=agent_timeout_s,
        agent_max_input_tokens=agent_max_input_tokens,
        openclaw_model=openclaw_model,
        openclaw_model_provider=openclaw_model_provider,
        openclaw_model_max_images=openclaw_model_max_images,
        agent_session_id="clawnav-context-stress",
        agent_session_timestamp="stress",
        openclaw_profile=openclaw_profile,
    )

    audit_path = output_dir / "context_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for step_id in range(int(steps)):
            instruction = "go to the target room"
            if step_id == 0:
                instruction = f"{instruction} {STEP0_MARKER}"
            decision = planner.plan_payload(
                {
                    "state": {
                        "scene_id": "stress_scene",
                        "episode_id": "stress_episode",
                        "instruction": instruction,
                        "step_id": step_id,
                    },
                    "runtime_context": {
                        "run_id": str(output_dir),
                        "policy_action": "MOVE_FORWARD",
                        "current_image_path": f"/tmp/openclaw_context_stress/step_{step_id:06d}.png",
                        "recent_keyframe_paths": [
                            f"/tmp/openclaw_context_stress/step_{max(0, step_id - 1):06d}.png"
                        ],
                    },
                }
            )
            metadata = decision.get("runtime_metadata") or {}
            audit = dict(metadata.get("context_audit") or {})
            call = runner.calls[-1] if runner.calls else {}
            message = str(call.get("message") or "")
            audit.update(
                {
                    "step_id": step_id,
                    "prompt_contains_step0_marker": STEP0_MARKER in message,
                    "decision_reason": decision.get("reason"),
                }
            )
            handle.write(json.dumps(audit, ensure_ascii=True, sort_keys=True) + "\n")

    summary = summarize_audit_file(audit_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "context_audit_report.md").write_text(
        format_markdown(summary) + "\n",
        encoding="utf-8",
    )
    return summary


def _arg_after(args: List[str], key: str) -> str:
    try:
        index = args.index(key)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    return str(args[index + 1])


def _step_id_from_message(message: str) -> int:
    try:
        payload = json.loads(message.split("Payload:\n", 1)[1])
    except (IndexError, json.JSONDecodeError):
        return 0
    state = payload.get("state") if isinstance(payload, dict) else {}
    try:
        return int((state or {}).get("step_id") or 0)
    except (TypeError, ValueError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--mode", choices=("fake", "agent", "model"), default="fake")
    parser.add_argument("--fake_fixed_overhead", type=int, default=400)
    parser.add_argument("--fake_growth_per_step", type=int, default=0)
    parser.add_argument("--openclaw_profile", default="")
    parser.add_argument("--agent_id", default="main")
    parser.add_argument("--agent_timeout", type=float, default=120.0)
    parser.add_argument("--agent_max_input_tokens", type=int, default=50000)
    parser.add_argument("--openclaw_model", default="")
    parser.add_argument(
        "--openclaw_model_provider",
        choices=("openclaw_cli", "qwen_api"),
        default="openclaw_cli",
    )
    parser.add_argument("--openclaw_model_max_images", type=int, default=3)
    args = parser.parse_args()

    summary = run_context_stress(
        output_dir=Path(args.output_dir),
        steps=args.steps,
        mode=args.mode,
        fake_fixed_overhead=args.fake_fixed_overhead,
        fake_growth_per_step=args.fake_growth_per_step,
        openclaw_profile=args.openclaw_profile,
        agent_id=args.agent_id,
        agent_timeout_s=args.agent_timeout,
        agent_max_input_tokens=args.agent_max_input_tokens,
        openclaw_model=args.openclaw_model,
        openclaw_model_provider=args.openclaw_model_provider,
        openclaw_model_max_images=args.openclaw_model_max_images,
    )
    print(format_markdown(summary))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
