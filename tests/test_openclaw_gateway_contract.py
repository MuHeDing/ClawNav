from harness.openclaw.gateway import (
    OpenClawGatewayError,
    gateway_response_to_decision,
    json_safe_dict,
    strip_oracle_fields,
)


def test_plan_request_context_is_json_safe_and_non_oracle():
    context = {
        "policy_action": "TURN_LEFT",
        "success": True,
        "distance_to_goal": 1.0,
        "recent_frames": [object()],
        "recent_actions": ["TURN_LEFT"],
        "nested": {"ok": True, "SPL": 0.5, "bad": object()},
    }

    payload = json_safe_dict(strip_oracle_fields(context))

    assert payload == {
        "policy_action": "TURN_LEFT",
        "recent_actions": ["TURN_LEFT"],
        "nested": {"ok": True},
    }


def test_gateway_contract_accepts_all_runtime_intents():
    for intent, tool in [
        ("act", "NavigationPolicySkill"),
        ("recall_memory", "MemoryQuerySkill"),
        ("write_memory", "MemoryWriteSkill"),
        ("verify_progress", "ProgressCriticSkill"),
        ("replan", "ReplannerSkill"),
    ]:
        decision = gateway_response_to_decision(
            {
                "intent": intent,
                "tool_name": tool,
                "arguments": {},
                "reason": "contract_test",
            }
        )
        assert decision.intent == intent
        assert decision.tool_name == tool


def test_gateway_contract_rejects_missing_tool_name():
    try:
        gateway_response_to_decision({"intent": "act"})
    except OpenClawGatewayError as exc:
        assert "tool_name" in str(exc)
    else:
        raise AssertionError("expected OpenClawGatewayError")
