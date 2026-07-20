from scripts.check_openclaw_plan_gateway import (
    build_probe_payload,
    validate_gateway_health,
    validate_plan_response,
)
from scripts.check_openclaw_visual_plan_gateway import (
    build_visual_probe_payload,
    validate_visual_plan_response,
    validate_visual_observations,
)


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


def test_validate_gateway_health_rejects_wrong_adapter_service_when_required():
    try:
        validate_gateway_health(
            {"ok": True, "service": "clawnav_openclaw_gateway"},
            require_service="openclaw_cli_plan_gateway",
        )
    except ValueError as exc:
        assert "openclaw_cli_plan_gateway" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_gateway_health_rejects_low_timeout_budget_when_enforced():
    try:
        validate_gateway_health(
            {
                "ok": True,
                "service": "openclaw_cli_plan_gateway",
                "timeout_budget": {"recommended_gateway_timeout_s": 242},
            },
            require_service="openclaw_cli_plan_gateway",
            client_timeout_s=120,
            enforce_timeout_budget=True,
        )
    except ValueError as exc:
        assert "OPENCLAW_GATEWAY_TIMEOUT" in str(exc)
        assert "242" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_gateway_health_requires_supported_and_active_staged_schemas():
    valid = {
        "ok": True,
        "instruction_segmentation": True,
        "stage_schema_versions": ["instruction_stages_v1"],
        "action_schema_versions": ["route_v2", "route_v3_staged"],
        "active_stage_schema": "instruction_stages_v1",
        "active_action_schema": "route_v3_staged",
    }
    validate_gateway_health(
        valid,
        required_stage_schema="instruction_stages_v1",
        required_action_schema="route_v3_staged",
    )

    for override in (
        {"instruction_segmentation": False},
        {"active_stage_schema": "legacy"},
        {"active_action_schema": "route_v2"},
        {"action_schema_versions": ["route_v2"]},
    ):
        try:
            validate_gateway_health(
                {**valid, **override},
                required_stage_schema="instruction_stages_v1",
                required_action_schema="route_v3_staged",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("expected staged capability validation failure")


def test_build_visual_probe_payload_passes_image_content_summary_not_oracle_metrics():
    observations = [
        {
            "image_path": "/tmp/frame.png",
            "caption": "hallway",
            "visual_observation": "A hallway with a doorway.",
        }
    ]

    payload = build_visual_probe_payload(
        "go to kitchen", "/tmp/frame.png", observations
    )

    assert payload["runtime_context"]["current_image_path"] == "/tmp/frame.png"
    assert payload["runtime_context"]["recent_keyframe_paths"] == ["/tmp/frame.png"]
    assert payload["runtime_context"]["visual_observations"] == observations
    assert "success" not in payload["runtime_context"]
    assert "distance_to_goal" not in payload["runtime_context"]


def test_validate_visual_observations_rejects_empty_or_failed_observation():
    validate_visual_observations(
        [{"image_path": "/tmp/frame.png", "visual_observation": "hallway"}]
    )

    for observations in (
        [],
        [{"image_path": "/tmp/frame.png", "error": "qwen unavailable"}],
        [{"image_path": "/tmp/frame.png", "visual_observation": ""}],
    ):
        try:
            validate_visual_observations(observations)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_validate_visual_plan_response_requires_adapter_visual_metadata():
    response = {
        "intent": "act",
        "tool_name": "NavigationPolicySkill",
        "arguments": {},
        "reason": "ok",
    }

    try:
        validate_visual_plan_response(response)
    except ValueError as exc:
        assert "visual_analysis" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    validate_visual_plan_response(
        {
            **response,
            "runtime_metadata": {"visual_analysis": {"ran": True, "failures": 0}},
        }
    )
