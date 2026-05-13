from harness.openclaw.gateway import (
    FakeOpenClawGatewayClient,
    OpenClawGatewayClient,
    OpenClawGatewayError,
)
from harness.types import VLNState


def make_state():
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=0,
        current_image=None,
    )


def test_fake_gateway_returns_plan_decision_without_oracle_fields():
    client = FakeOpenClawGatewayClient(
        response={
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {"distance_to_goal": 1.0, "text": "go"},
            "reason": "fake",
        }
    )

    decision = client.plan(make_state(), runtime_context={"success": True})

    assert decision.intent == "act"
    assert decision.tool_name == "NavigationPolicySkill"
    assert decision.arguments == {"text": "go"}
    assert decision.planner_backend == "gateway"


def test_gateway_client_builds_request_payload_without_oracle_fields():
    captured = {}

    def post_json(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {
            "intent": "recall_memory",
            "tool_name": "MemoryQuerySkill",
            "arguments": {"text": "go"},
            "reason": "gateway",
        }

    client = OpenClawGatewayClient(
        base_url="http://gateway",
        post_json=post_json,
        timeout_s=1.5,
    )

    decision = client.plan(make_state(), {"success": True, "policy_action": "STOP"})

    assert captured["url"] == "http://gateway/plan"
    assert captured["timeout"] == 1.5
    assert "success" not in captured["payload"]["runtime_context"]
    assert captured["payload"]["runtime_context"]["policy_action"] == "STOP"
    assert decision.reason == "gateway"


def test_gateway_client_raises_typed_error_on_bad_response():
    client = FakeOpenClawGatewayClient(response={"intent": "act"})

    try:
        client.plan(make_state(), {})
    except OpenClawGatewayError as exc:
        assert "tool_name" in str(exc)
    else:
        raise AssertionError("expected OpenClawGatewayError")
