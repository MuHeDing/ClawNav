import json
import subprocess

from harness.openclaw.openclaw_cli_plan_gateway import OpenClawCliPlanPlanner


class FakeOpenClawRunner:
    def __init__(self, returncode=0, stdout=None, stderr=""):
        self.returncode = returncode
        self.stdout = stdout or json.dumps({"ok": True})
        self.stderr = stderr
        self.calls = []

    def __call__(self, args, timeout_s):
        self.calls.append((args, timeout_s))
        return subprocess.CompletedProcess(
            args,
            self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def test_cli_plan_gateway_health_uses_openclaw_gateway_call():
    runner = FakeOpenClawRunner(stdout=json.dumps({"ok": True, "defaultAgentId": "main"}))
    planner = OpenClawCliPlanPlanner(run_openclaw=runner)

    payload = planner.health_payload()

    assert payload["ok"] is True
    assert payload["service"] == "openclaw_cli_plan_gateway"
    assert runner.calls[0][0][:4] == ["openclaw", "gateway", "call", "health"]


def test_cli_plan_gateway_plan_requires_openclaw_health_and_returns_plan_intent():
    runner = FakeOpenClawRunner()
    planner = OpenClawCliPlanPlanner(run_openclaw=runner, recall_interval_steps=5)

    decision = planner.plan_payload(
        {
            "state": {
                "instruction": "go to kitchen",
                "step_id": 0,
            }
        }
    )

    assert decision["intent"] == "recall_memory"
    assert decision["tool_name"] == "MemoryQuerySkill"
    assert decision["arguments"]["text"] == "go to kitchen"
    assert runner.calls[0][0][:4] == ["openclaw", "gateway", "call", "health"]


def test_cli_plan_gateway_raises_when_openclaw_gateway_is_unavailable():
    runner = FakeOpenClawRunner(returncode=1, stderr="connection refused")
    planner = OpenClawCliPlanPlanner(run_openclaw=runner)

    try:
        planner.plan_payload({"state": {"instruction": "go", "step_id": 1}})
    except RuntimeError as exc:
        assert "connection refused" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_cli_plan_gateway_agent_mode_calls_openclaw_agent_and_parses_json_text():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "write_memory",
                                "tool_name": "MemoryWriteSkill",
                                "arguments": {"summary": "saw a kitchen"},
                                "reason": "landmark_keyframe",
                            }
                        )
                    }
                ],
                "meta": {"agentMeta": {"provider": "qwen", "model": "qwen3.5-plus"}},
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_id="main",
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go to kitchen", "step_id": 3},
            "runtime_context": {"keyframe_candidate": {"image_path": "frame.png"}},
        }
    )

    assert decision["intent"] == "write_memory"
    assert decision["tool_name"] == "MemoryWriteSkill"
    assert decision["arguments"]["summary"] == "saw a kitchen"
    assert runner.calls[0][0][:5] == ["openclaw", "agent", "--agent", "main", "--json"]


def test_cli_plan_gateway_agent_mode_falls_back_to_heuristic_when_agent_fails():
    runner = FakeOpenClawRunner(returncode=1, stderr="qwen unavailable")
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        recall_interval_steps=5,
    )

    decision = planner.plan_payload(
        {
            "state": {
                "instruction": "go to kitchen",
                "step_id": 5,
            }
        }
    )

    assert decision["intent"] == "recall_memory"
    assert decision["tool_name"] == "MemoryQuerySkill"
    assert decision["reason"] == "openclaw_cli_agent_fallback:openclaw_cli_interval_recall"
    assert "qwen unavailable" in decision["arguments"]["planner_error"]
