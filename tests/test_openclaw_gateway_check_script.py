from scripts.check_openclaw_plan_gateway import build_probe_payload, validate_plan_response
from scripts.check_openclaw_visual_plan_gateway import (
    build_visual_probe_payload,
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


def test_build_visual_probe_payload_passes_image_content_summary_not_oracle_metrics():
    observations = [
        {
            "image_path": "/tmp/frame.png",
            "caption": "hallway",
            "visual_observation": "A hallway with a doorway.",
        }
    ]

    payload = build_visual_probe_payload("go to kitchen", "/tmp/frame.png", observations)

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
