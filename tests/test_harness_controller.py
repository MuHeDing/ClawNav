from harness.config import HarnessConfig
from harness.controller import HarnessController
from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.types import SkillResult, VLNState


class RecordingSkill(Skill):
    def __init__(self, name, result=None):
        self.name = name
        self.calls = []
        self.result = result or SkillResult.ok_result(
            "noop",
            {"skill": name},
        )

    def run(self, state, payload):
        self.calls.append(payload)
        return self.result


class FailingSkill(RecordingSkill):
    def run(self, state, payload):
        self.calls.append(payload)
        raise RuntimeError("boom")


def make_state(last_action="MOVE_FORWARD"):
    return VLNState(
        scene_id="s",
        episode_id="e",
        instruction="go to kitchen",
        step_id=0,
        current_image="cur",
        online_metrics={},
        diagnostics={"distance_to_goal": 1.0, "success": False, "SPL": 0.0},
        last_action=last_action,
    )


def make_registry(*skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def nav_skill():
    return RecordingSkill(
        "NavigationPolicySkill",
        SkillResult.ok_result("action", {"action_text": "MOVE_FORWARD"}),
    )


def test_act_only_calls_only_navigation_policy_skill():
    nav = nav_skill()
    memory = RecordingSkill("MemoryQuerySkill")
    controller = HarnessController(
        make_registry(nav, memory),
        HarnessConfig(harness_mode="act_only"),
    )
    result = controller.run_step(make_state(), {"recent_frames": ["r1"]})
    assert result.payload["action_text"] == "MOVE_FORWARD"
    assert len(nav.calls) == 1
    assert memory.calls == []


def test_first_step_in_memory_recall_recalls_before_acting():
    nav = nav_skill()
    memory = RecordingSkill(
        "MemoryQuerySkill",
        SkillResult.ok_result(
            "memory_query",
            {
                "policy_context": {"memory_context_text": "kitchen entrance"},
                "control_context": {"hits": []},
                "executor_context": {},
                "memory_hits": [],
            },
        ),
    )
    controller = HarnessController(
        make_registry(nav, memory),
        HarnessConfig(harness_mode="memory_recall"),
    )
    controller.run_step(make_state(), {"active_subgoal": "enter kitchen"})
    assert len(memory.calls) == 1
    assert len(nav.calls) == 1
    assert "kitchen entrance" in nav.calls[0]["memory_context_text"]


def test_controller_obeys_max_internal_calls_per_step():
    nav = nav_skill()
    memory = RecordingSkill("MemoryQuerySkill")
    critic = RecordingSkill(
        "ProgressCriticSkill",
        SkillResult.ok_result("progress_critic", {"risky_stop": False, "signals": []}),
    )
    controller = HarnessController(
        make_registry(nav, memory, critic),
        HarnessConfig(harness_mode="full", max_internal_calls_per_step=1),
    )
    result = controller.run_step(make_state(), {})
    assert result.payload["action_text"] == "MOVE_FORWARD"
    assert controller.last_trace["fallback"] is True
    assert controller.last_trace["fallback_reason"] == "internal_call_budget_exceeded"


def test_skill_exception_falls_back_to_baseline_navigation_action():
    nav = nav_skill()
    memory = FailingSkill("MemoryQuerySkill")
    controller = HarnessController(
        make_registry(nav, memory),
        HarnessConfig(harness_mode="memory_recall"),
    )
    result = controller.run_step(make_state(), {})
    assert result.payload["action_text"] == "MOVE_FORWARD"
    assert len(nav.calls) == 1
    assert controller.last_trace["fallback"] is True


def test_controller_never_passes_distance_to_goal_to_critic_payload():
    nav = nav_skill()
    critic = RecordingSkill(
        "ProgressCriticSkill",
        SkillResult.ok_result("progress_critic", {"risky_stop": False, "signals": []}),
    )
    controller = HarnessController(
        make_registry(nav, critic),
        HarnessConfig(harness_mode="memory_critic"),
    )
    controller.run_step(make_state(), {})
    assert "distance_to_goal" not in critic.calls[0]
    assert "diagnostics" not in critic.calls[0]


def test_controller_does_not_pass_diagnostics_to_decision_skills():
    nav = nav_skill()
    memory = RecordingSkill("MemoryQuerySkill")
    critic = RecordingSkill(
        "ProgressCriticSkill",
        SkillResult.ok_result(
            "progress_critic",
            {"risky_stop": True, "possible_stuck": True, "signals": ["risky_stop"]},
        ),
    )
    replanner = RecordingSkill("ReplannerSkill")
    controller = HarnessController(
        make_registry(nav, memory, critic, replanner),
        HarnessConfig(harness_mode="full", max_internal_calls_per_step=5),
    )
    controller.run_step(make_state(last_action="STOP"), {"policy_action": "STOP"})
    for skill in (memory, critic, replanner):
        assert skill.calls
        assert "diagnostics" not in skill.calls[0]
        assert "distance_to_goal" not in skill.calls[0]
        assert "success" not in skill.calls[0]


def test_risky_stop_triggers_recall_and_replan_without_oracle_metric():
    nav = nav_skill()
    memory = RecordingSkill(
        "MemoryQuerySkill",
        SkillResult.ok_result("memory_query", {"memory_hits": [], "policy_context": {}}),
    )
    critic = RecordingSkill(
        "ProgressCriticSkill",
        SkillResult.ok_result(
            "progress_critic",
            {"risky_stop": True, "signals": ["risky_stop"]},
        ),
    )
    replanner = RecordingSkill("ReplannerSkill")
    controller = HarnessController(
        make_registry(nav, memory, critic, replanner),
        HarnessConfig(harness_mode="full", max_internal_calls_per_step=5),
    )
    result = controller.run_step(make_state(last_action="STOP"), {"policy_action": "STOP"})
    assert result.payload["action_text"] == "MOVE_FORWARD"
    assert memory.calls
    assert replanner.calls
    assert "distance_to_goal" not in memory.calls[0]
    assert "distance_to_goal" not in replanner.calls[0]


def test_controller_trace_records_skill_runtime_metadata():
    nav = nav_skill()
    controller = HarnessController(
        make_registry(nav),
        HarnessConfig(harness_mode="act_only"),
    )

    controller.run_step(make_state(), {})

    assert "skill_runtime" in controller.last_trace
    assert controller.last_trace["skill_runtime"][0]["skill"] == "NavigationPolicySkill"
    assert "latency_ms" in controller.last_trace["skill_runtime"][0]
    assert controller.last_trace["skill_runtime"][0]["runtime_status"] == "completed"
    assert controller.last_trace["skill_runtime"][0]["result_type"] == "action"
