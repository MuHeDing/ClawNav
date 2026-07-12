from harness.memory.working_memory import WorkingMemory


def test_recent_frames_are_bounded():
    wm = WorkingMemory(max_recent_frames=2)
    wm.append_frame("f0")
    wm.append_frame("f1")
    wm.append_frame("f2")
    assert wm.get_recent_frames(10) == ["f1", "f2"]


def test_repeated_turns_mark_possible_stuck():
    wm = WorkingMemory()
    for _ in range(5):
        wm.append_action("TURN_LEFT")
    assert wm.has_action_oscillation()


def test_low_displacement_is_non_oracle():
    wm = WorkingMemory()
    wm.append_pose([0.0, 0.0, 0.0])
    wm.append_pose([0.01, 0.0, 0.0])
    wm.append_pose([0.02, 0.0, 0.0])
    assert wm.has_low_displacement(threshold=0.05)


def test_oracle_metrics_are_not_used_for_decision():
    wm = WorkingMemory()
    wm.append_diagnostics({"distance_to_goal": 1.0, "success": True})
    assert wm.decision_metrics() == {}
    assert wm.diagnostic_metrics() == {"distance_to_goal": 1.0, "success": True}


def test_reset_clears_episode_local_state():
    wm = WorkingMemory()
    wm.append_frame("frame")
    wm.append_action("MOVE_FORWARD")
    wm.append_pose([1.0, 2.0, 3.0])
    wm.append_online_metrics({"collision": True})
    wm.append_diagnostics({"distance_to_goal": 1.0})

    wm.reset()

    assert wm.get_recent_frames() == []
    assert list(wm.actions) == []
    assert list(wm.poses) == []
    assert wm.decision_metrics() == {}
    assert wm.diagnostic_metrics() == {}
