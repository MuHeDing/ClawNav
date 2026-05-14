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
