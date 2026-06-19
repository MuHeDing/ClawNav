from harness.openclaw.keyframe_gate import (
    EventGatedKeyframeGate,
    candidate_action_status_from_tool_result,
)


def gate(**overrides):
    params = {
        "min_gap_steps": 2,
        "episode_cap": 4,
        "coverage_gap_steps": 10,
        "debug_save_all_eligible": False,
    }
    params.update(overrides)
    return EventGatedKeyframeGate(**params)


def evaluate(gate_obj, step_id, action, status="ok"):
    return gate_obj.evaluate(
        scene_id="s1",
        episode_id="e1",
        step_id=step_id,
        raw_candidate_action=action,
        candidate_action_status=status,
        current_image_path=f"/tmp/current/step_{step_id:06d}.png",
        keyframe_target_path=f"/tmp/keyframes/step_{step_id:06d}.png",
    )


def test_initial_context_saves_even_without_valid_action():
    result = evaluate(gate(), 0, "", status="empty")

    assert result["save_decision"] == "save"
    assert result["save_reason"] == "initial_context"
    assert result["candidate_action_status"] == "empty"
    assert result["candidate_event_type"] == "initial_context"
    assert result["keyframe_cap_validation_failed"] is False


def test_missing_image_blocks_initial_context():
    result = gate().evaluate(
        scene_id="s1",
        episode_id="e1",
        step_id=0,
        raw_candidate_action="TURN_LEFT",
        candidate_action_status="ok",
        current_image_path="",
        keyframe_target_path="/tmp/keyframes/step_000000.png",
    )

    assert result["save_decision"] == "skip"
    assert result["skip_reason"] == "missing_current_image"


def test_candidate_action_status_from_navigation_result():
    assert candidate_action_status_from_tool_result({"ok": False, "payload": {}}) == "failed"
    assert candidate_action_status_from_tool_result({"ok": True, "payload": {}}) == "empty"
    assert (
        candidate_action_status_from_tool_result(
            {"ok": True, "payload": {"action_text": "dance"}}
        )
        == "unrecognized"
    )
    assert (
        candidate_action_status_from_tool_result(
            {"ok": True, "payload": {"action_text": "turn left"}}
        )
        == "ok"
    )


def test_turn_and_stop_candidates_save_once_per_action_segment():
    gate_obj = gate()

    first = evaluate(gate_obj, 1, "TURN_LEFT")
    duplicate = evaluate(gate_obj, 2, "TURN_LEFT")
    reset = evaluate(gate_obj, 3, "MOVE_FORWARD")
    second = evaluate(gate_obj, 4, "TURN_LEFT")

    assert first["save_decision"] == "save"
    assert first["save_reason"] == "candidate_decision_point"
    assert duplicate["save_decision"] == "skip"
    assert duplicate["skip_reason"] == "action_segment_duplicate"
    assert reset["save_decision"] == "skip"
    assert reset["skip_reason"] == "no_direct_save_event"
    assert second["save_decision"] == "save"
    assert second["candidate_segment_id"] != first["candidate_segment_id"]


def test_failed_empty_and_unrecognized_actions_reset_segment_without_saving():
    gate_obj = gate()

    assert evaluate(gate_obj, 1, "STOP")["save_decision"] == "save"
    failed = evaluate(gate_obj, 2, "STOP", status="failed")
    empty = evaluate(gate_obj, 3, "", status="empty")
    unrecognized = evaluate(gate_obj, 4, "DANCE", status="unrecognized")
    new_stop = evaluate(gate_obj, 5, "STOP")

    assert failed["skip_reason"] == "candidate_action_failed"
    assert empty["skip_reason"] == "candidate_action_empty"
    assert unrecognized["skip_reason"] == "candidate_action_unrecognized"
    assert new_stop["save_decision"] == "save"


def test_cooldown_blocks_new_decision_before_min_gap():
    gate_obj = gate(min_gap_steps=5)

    assert evaluate(gate_obj, 1, "TURN_RIGHT")["save_decision"] == "save"
    blocked = evaluate(gate_obj, 3, "STOP")

    assert blocked["save_decision"] == "skip"
    assert blocked["skip_reason"] == "cooldown"


def test_cap_blocks_late_direct_events_and_marks_validation_failure():
    gate_obj = gate(episode_cap=1)

    assert evaluate(gate_obj, 0, "", status="empty")["save_decision"] == "save"
    blocked = evaluate(gate_obj, 10, "STOP")

    assert blocked["save_decision"] == "skip"
    assert blocked["skip_reason"] == "episode_cap_reached"
    assert blocked["keyframe_cap_validation_failed"] is True


def test_coverage_gap_saves_when_no_recent_keyframe_exists():
    gate_obj = gate(coverage_gap_steps=5)

    assert evaluate(gate_obj, 0, "", status="empty")["save_decision"] == "save"
    coverage = evaluate(gate_obj, 6, "MOVE_FORWARD")

    assert coverage["save_decision"] == "save"
    assert coverage["save_reason"] == "coverage_gap"
