from harness.openclaw.motion_feedback import build_motion_feedback


def test_motion_feedback_marks_effective_forward():
    feedback = build_motion_feedback(
        {
            "available": True,
            "previous_action": "MOVE_FORWARD",
            "last_forward_delta_m": 0.25,
            "last_action_had_progress": True,
            "collision": False,
            "consecutive_no_progress_forward": 0,
            "progress_threshold_m": 0.05,
        }
    )

    assert feedback["last_action"] == "MOVE_FORWARD"
    assert feedback["expected_effect"] == "advance about 0.25m"
    assert feedback["actual_effect"] == "effective_forward"
    assert feedback["last_forward_delta_m"] == 0.25
    assert feedback["recommended_constraint"] == "forward_allowed"


def test_motion_feedback_marks_blocked_forward_without_raw_pose():
    feedback = build_motion_feedback(
        {
            "available": True,
            "previous_action": "MOVE_FORWARD",
            "last_forward_delta_m": 0.02,
            "last_action_had_progress": False,
            "collision": True,
            "consecutive_no_progress_forward": 2,
            "progress_threshold_m": 0.05,
            "sim_position": [1.0, 2.0, 3.0],
            "sim_rotation": [1.0, 0.0, 0.0, 0.0],
        }
    )

    assert feedback["actual_effect"] == "blocked"
    assert feedback["collision_recent"] is True
    assert feedback["consecutive_no_progress_forward"] == 2
    assert feedback["recommended_constraint"] == "avoid_forward"
    assert feedback["distance_source"] == "rounded_local_odometry"
    assert "sim_position" not in feedback
    assert "sim_rotation" not in feedback


def test_motion_feedback_marks_effective_turn():
    feedback = build_motion_feedback(
        {
            "available": True,
            "previous_action": "TURN_RIGHT",
            "last_turn_delta_deg": 15.0,
            "last_action_had_progress": True,
            "collision": False,
            "turn_progress_threshold_deg": 5.0,
        }
    )

    assert feedback["last_action"] == "TURN_RIGHT"
    assert feedback["expected_effect"] == "rotate about 15 degrees"
    assert feedback["actual_effect"] == "effective_turn"
    assert feedback["last_turn_delta_deg"] == 15.0
    assert feedback["recommended_constraint"] == "turn_allowed"
