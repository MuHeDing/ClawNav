from harness.memory.task_memory import TaskMemory


def test_single_subgoal_mode_uses_full_instruction():
    tm = TaskMemory()
    tm.reset("ep1", "go to the kitchen", mode="single")
    assert tm.current_subgoal.text == "go to the kitchen"
    assert tm.task_state.global_instruction == "go to the kitchen"


def test_rule_based_subgoal_mode_splits_instruction():
    tm = TaskMemory()
    tm.reset(
        "ep1",
        "Walk down the hallway and turn left at the sofa, then enter the kitchen and stop near the sink.",
        mode="rule",
    )
    texts = [sg.text for sg in tm.task_state.pending_subgoals]
    assert any("hallway" in text for text in texts)
    assert any("kitchen" in text for text in texts)


def test_task_memory_advances_subgoals():
    tm = TaskMemory()
    tm.reset("ep1", "go to kitchen", subgoals=["find hallway", "enter kitchen"])
    assert tm.current_subgoal.text == "find hallway"
    tm.mark_current_complete(reason="hallway visible")
    assert tm.current_subgoal.text == "enter kitchen"
    assert len(tm.task_state.completed_subgoals) == 1


def test_task_memory_records_failure_and_recovery():
    tm = TaskMemory()
    tm.reset("ep1", "go to kitchen", subgoals=["enter kitchen"])
    tm.mark_current_failed("stuck")
    tm.record_recovery_attempt("recall_memory")
    assert tm.task_state.failure_reason == "stuck"
    assert tm.task_state.recovery_attempts == ["recall_memory"]
