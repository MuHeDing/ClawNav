from scripts.check_openclaw_plan_gateway import build_probe_payload, validate_plan_response


def test_build_probe_payload_contains_no_oracle_fields():
    payload = build_probe_payload("go to kitchen")

    assert payload["state"]["instruction"] == "go to kitchen"
    assert "success" not in payload["runtime_context"]
    assert "distance_to_goal" not in payload["runtime_context"]


def test_validate_plan_response_accepts_valid_response():
    validate_plan_response(
        {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {},
            "reason": "ok",
        }
    )


def test_validate_plan_response_rejects_bad_response():
    try:
        validate_plan_response({"intent": "act"})
    except ValueError as exc:
        assert "tool_name" in str(exc)
    else:
        raise AssertionError("expected ValueError")
