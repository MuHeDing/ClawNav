from harness.memory.task_memory import TaskMemory
from harness.memory.working_memory import WorkingMemory
from harness.skills.progress_critic import ProgressCriticSkill
from harness.skills.replanner import ReplannerSkill
from harness.types import VLNState


def make_state(last_action="MOVE_FORWARD"):
    return VLNState(
        scene_id="s",
        episode_id="e",
        instruction="go to kitchen",
        step_id=5,
        current_image="cur",
        online_metrics={},
        diagnostics={"distance_to_goal": 0.1, "success": True, "SPL": 1.0},
        last_action=last_action,
    )


def test_repeated_turns_produce_possible_stuck():
    wm = WorkingMemory()
    for _ in range(5):
        wm.append_action("TURN_LEFT")
    result = ProgressCriticSkill().run(make_state(), {"working_memory": wm})
    assert result.ok
    assert result.payload["possible_stuck"] is True
    assert "possible_stuck" in result.payload["signals"]


def test_low_displacement_produces_signal():
    wm = WorkingMemory()
    wm.append_pose([0.0, 0.0, 0.0])
    wm.append_pose([0.01, 0.0, 0.0])
    wm.append_pose([0.02, 0.0, 0.0])
    result = ProgressCriticSkill().run(make_state(), {"working_memory": wm})
    assert result.payload["low_displacement"] is True
    assert "low_displacement" in result.payload["signals"]


def test_stop_with_poor_memory_consistency_is_risky_stop():
    result = ProgressCriticSkill().run(
        make_state(last_action="STOP"),
        {
            "policy_action": "STOP",
            "semantic_alignment": 0.1,
            "memory_consistency": 0.1,
        },
    )
    assert result.payload["risky_stop"] is True
    assert "risky_stop" in result.payload["signals"]


def test_critic_ignores_distance_to_goal():
    result = ProgressCriticSkill().run(
        make_state(last_action="STOP"),
        {
            "policy_action": "STOP",
            "semantic_alignment": 1.0,
            "memory_consistency": 1.0,
        },
    )
    assert result.payload["risky_stop"] is False
    assert result.payload["used_oracle_metrics"] is False
    assert "distance_to_goal" not in result.payload["decision_inputs"]


def test_replanner_creates_recovery_subgoal_from_failure_reason():
    tm = TaskMemory()
    tm.reset("ep1", "go to kitchen", subgoals=["enter kitchen"])
    result = ReplannerSkill().run(
        make_state(),
        {
            "task_memory": tm,
            "failure_reason": "possible_stuck",
            "memory_hits": [{"name": "kitchen entrance"}],
        },
    )
    assert result.ok
    assert "Recover from possible_stuck" in result.payload["active_subgoal"]
    assert result.payload["memory_query_hint"] == "enter kitchen"
    assert result.payload["reason"] == "possible_stuck"
