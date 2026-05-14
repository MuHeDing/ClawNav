import harness.openclaw.gateway as gateway_module
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


def test_gateway_client_drops_non_json_runtime_context_fields():
    captured = {}

    def post_json(url, payload, timeout):
        captured["payload"] = payload
        return {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {},
        }

    client = OpenClawGatewayClient(
        base_url="http://gateway",
        post_json=post_json,
    )

    client.plan(
        make_state(),
        {
            "policy_action": "TURN_LEFT",
            "recent_frames": [object()],
            "nested": {"ok": True, "bad": object()},
        },
    )

    assert captured["payload"]["runtime_context"] == {
        "policy_action": "TURN_LEFT",
        "nested": {"ok": True},
    }


def test_gateway_client_raises_typed_error_on_bad_response():
    client = FakeOpenClawGatewayClient(response={"intent": "act"})

    try:
        client.plan(make_state(), {})
    except OpenClawGatewayError as exc:
        assert "tool_name" in str(exc)
    else:
        raise AssertionError("expected OpenClawGatewayError")


def test_gateway_client_bypasses_system_proxy_env(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "intent": "act",
                "tool_name": "NavigationPolicySkill",
                "arguments": {},
            }

    class Session:
        def __init__(self):
            self.trust_env = True

        def post(self, url, json, timeout):
            captured["trust_env"] = self.trust_env
            captured["url"] = url
            return Response()

    monkeypatch.setattr(gateway_module.requests, "Session", Session)

    client = OpenClawGatewayClient(base_url="http://127.0.0.1:8011")
    decision = client.plan(make_state(), {})

    assert decision.intent == "act"
    assert captured["url"] == "http://127.0.0.1:8011/plan"
    assert captured["trust_env"] is False
