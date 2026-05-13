from harness.openclaw.subagents import (
    FakeSubagentClient,
    SubagentRequest,
    sanitize_subagent_context,
)


def test_sanitize_subagent_context_removes_oracle_fields():
    sanitized = sanitize_subagent_context(
        {"distance_to_goal": 1.0, "success": True, "instruction": "go"}
    )

    assert sanitized == {"instruction": "go"}


def test_fake_subagent_client_returns_structured_response():
    client = FakeSubagentClient({"intent": "act", "reason": "ok"})
    request = SubagentRequest(
        role="planner",
        instruction="go",
        context={"success": True, "step_id": 1},
    )

    response = client.call(request)

    assert response["intent"] == "act"
    assert client.requests[0].context == {"step_id": 1}
