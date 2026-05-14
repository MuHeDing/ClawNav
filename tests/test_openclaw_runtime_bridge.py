from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.gateway import FakeOpenClawGatewayClient, OpenClawGatewayError
from harness.openclaw.planner import OpenClawPlanDecision, RuleOpenClawPlanner
from harness.openclaw.runtime import OpenClawVLNRuntime
from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.types import SkillResult, VLNState


class EchoNavigationSkill(Skill):
    name = "NavigationPolicySkill"
    description = "Returns a fixed action."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("action", {"action_text": "TURN_LEFT"})


class EchoMemorySkill(Skill):
    name = "MemoryQuerySkill"
    description = "Returns fake memory."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result(
            "memory",
            {"policy_context": {"memory_context_text": "kitchen"}},
        )


class FailingGatewayPlanner:
    def plan(self, state, runtime_context):
        raise OpenClawGatewayError("502 Server Error: Bad Gateway for url: http://gateway/plan")


class StaticPlanner:
    def __init__(self, decision):
        self.decision = decision

    def plan(self, state, runtime_context):
        return self.decision


class EchoWriteSkill(Skill):
    name = "MemoryWriteSkill"
    description = "Writes fake memory."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("memory_write", {"stored": True, **payload})


class EchoCriticSkill(Skill):
    name = "ProgressCriticSkill"
    description = "Returns fake critic result."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("critic", {"possible_stuck": False})


class EchoReplannerSkill(Skill):
    name = "ReplannerSkill"
    description = "Returns fake subgoal."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("replan", {"active_subgoal": "recover hallway"})


def make_state(step_id=1):
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=step_id,
        current_image=None,
    )


def make_runtime():
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(EchoMemorySkill())
    return OpenClawVLNRuntime(
        tool_registry=registry,
        planner=RuleOpenClawPlanner(recall_interval_steps=5),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )


def make_full_runtime(decision):
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(EchoMemorySkill())
    registry.register(EchoWriteSkill())
    registry.register(EchoCriticSkill())
    registry.register(EchoReplannerSkill())
    return OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )


def test_runtime_lists_tools():
    runtime = make_runtime()

    tools = runtime.list_tools()

    assert [tool["name"] for tool in tools] == [
        "MemoryQuerySkill",
        "NavigationPolicySkill",
    ]


def test_runtime_step_calls_planned_tool_and_executor():
    runtime = make_runtime()

    result = runtime.step(make_state(step_id=1), payload={})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.executor_command["action_index"] == 2
    assert result.runtime_metadata["planner_backend"] == "rule"
    assert result.runtime_metadata["runtime_mode"] == "openclaw_bridge"


def test_runtime_initial_step_can_recall_then_act():
    runtime = make_runtime()

    result = runtime.step(make_state(step_id=0), payload={})

    assert result.ok is True
    assert result.runtime_metadata["planned_intent"] == "recall_memory"
    assert "MemoryQuerySkill" in result.runtime_metadata["tool_calls"][0]["tool_name"]


def test_runtime_can_use_gateway_planner_client():
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=FakeOpenClawGatewayClient(
            {
                "intent": "act",
                "tool_name": "NavigationPolicySkill",
                "arguments": {},
                "reason": "gateway_test",
            }
        ),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=1), payload={})

    assert result.ok is True
    assert result.runtime_metadata["planner_backend"] == "gateway"
    assert result.runtime_metadata["planner_reason"] == "gateway_test"


def test_runtime_falls_back_to_rule_planner_when_gateway_fails():
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(EchoMemorySkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=FailingGatewayPlanner(),
        fallback_planner=RuleOpenClawPlanner(recall_interval_steps=5),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=1), payload={})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["planner_backend"] == "rule"
    assert result.runtime_metadata["planner_fallback"] is True
    assert "502 Server Error" in result.runtime_metadata["planner_error"]


def test_runtime_executes_write_memory_intent_before_action():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"step_id": 3, "note": "landmark"},
        reason="curator",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(make_state(step_id=3), payload={})

    assert result.ok is True
    assert result.runtime_metadata["planned_intent"] == "write_memory"
    assert result.runtime_metadata["tool_calls"][0]["tool_name"] == "MemoryWriteSkill"
    assert result.runtime_metadata["tool_calls"][-1]["tool_name"] == "NavigationPolicySkill"


def test_runtime_executes_critic_and_replan_intents_before_action():
    for intent, tool in [
        ("verify_progress", "ProgressCriticSkill"),
        ("replan", "ReplannerSkill"),
    ]:
        decision = OpenClawPlanDecision(
            intent=intent,
            tool_name=tool,
            arguments={"reason": "planner"},
            reason="planner",
            planner_backend="gateway",
        )
        runtime = make_full_runtime(decision)

        result = runtime.step(make_state(step_id=4), payload={})

        assert result.ok is True
        assert result.runtime_metadata["tool_calls"][0]["tool_name"] == tool
        assert result.runtime_metadata["tool_calls"][-1]["tool_name"] == "NavigationPolicySkill"


def test_runtime_enriches_write_memory_with_keyframe_candidate():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"note": "landmark"},
        reason="curator",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(
        make_state(step_id=7),
        payload={
            "keyframe_candidate": {
                "step_id": 7,
                "image_path": "/tmp/keyframe.png",
                "reason": "interval",
            }
        },
    )

    first_call = result.runtime_metadata["tool_calls"][0]
    assert first_call["tool_name"] == "MemoryWriteSkill"
    assert "image_path" in first_call["payload_summary"]["keys"]
    assert "write_type" in first_call["payload_summary"]["keys"]
