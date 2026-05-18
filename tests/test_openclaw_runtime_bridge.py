from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.gateway import FakeOpenClawGatewayClient, OpenClawGatewayError
from harness.openclaw.planner import OpenClawPlanDecision, RuleOpenClawPlanner
from harness.openclaw.runtime import OpenClawVLNRuntime
from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.skills.memory_write import MemoryWriteSkill
from harness.skills.visual_memory_curator import VisualMemoryCuratorSkill
from harness.types import SkillResult, VLNState


class EchoNavigationSkill(Skill):
    name = "NavigationPolicySkill"
    description = "Returns a fixed action."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("action", {"action_text": "TURN_LEFT"})


class RecordingNavigationSkill(Skill):
    name = "NavigationPolicySkill"
    description = "Records payload and returns a fixed action."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self):
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
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


class RecordingMemorySkill(Skill):
    name = "MemoryQuerySkill"
    description = "Records memory query payload."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self):
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
        return SkillResult.ok_result(
            "memory",
            {"policy_context": {"memory_context_text": "doorway"}},
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


def test_runtime_passes_replan_subgoal_to_navigation_policy():
    decision = OpenClawPlanDecision(
        intent="replan",
        tool_name="ReplannerSkill",
        arguments={"failure_reason": "rotation_loop"},
        reason="planner",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(EchoReplannerSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=4), payload={})

    assert result.ok is True
    assert navigation.calls[0]["active_subgoal"] == "recover hallway"


def test_runtime_uses_planner_action_override_without_policy_action():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "MOVE_FORWARD"},
        reason="planner chose next discrete action",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=4), payload={})

    assert result.ok is True
    assert result.action_text == "MOVE_FORWARD"
    assert result.executor_command["action_index"] == 1
    assert navigation.calls == []
    assert result.runtime_metadata["planner_action_override"] == "MOVE_FORWARD"


def test_runtime_records_image_paths_used_in_metadata():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "MOVE_FORWARD"},
        reason="planner chose next discrete action",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(RecordingNavigationSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={
            "current_image_path": "/tmp/current.png",
            "recent_keyframe_paths": ["/tmp/key0.png", "/tmp/current.png"],
        },
    )

    assert result.ok is True
    assert result.runtime_metadata["image_paths_used"] == [
        "/tmp/current.png",
        "/tmp/key0.png",
    ]


def test_runtime_stops_on_openclaw_cli_agent_fallback_without_policy_call():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"planner_error": "qwen unavailable"},
        reason="openclaw_cli_agent_fallback:openclaw_cli_default_act",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=4), payload={})

    assert result.ok is False
    assert result.action_text == "STOP"
    assert result.error == "qwen unavailable"
    assert navigation.calls == []
    assert result.runtime_metadata["planner_agent_fallback"] is True
    assert result.runtime_metadata["planner_reason"] == (
        "openclaw_cli_agent_fallback:openclaw_cli_default_act"
    )
    assert result.runtime_metadata["tool_calls"] == []


def test_runtime_passes_memory_context_to_navigation_policy():
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "go to kitchen"},
        reason="recall",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(EchoMemorySkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=0), payload={})

    assert result.ok is True
    assert navigation.calls[0]["memory_context_text"] == "kitchen"


def test_runtime_records_recall_usage_metadata():
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "kitchen", "step_id": 0},
        reason="initial",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(make_state(step_id=0), payload={})

    assert result.ok is True
    assert result.runtime_metadata["recall_usage"][0]["event_type"] == "memory_recall"
    assert result.runtime_metadata["recall_usage"][0]["used_by_policy"] is True


def test_runtime_records_recall_causality_from_context_and_action_change():
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={
            "text": "kitchen",
            "step_id": 0,
            "allowed_scopes": ["episode"],
            "memory_namespace": "episode:s1:e1",
        },
        reason="initial",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(
        make_state(step_id=0),
        payload={"policy_action": "MOVE_FORWARD"},
    )

    event = result.runtime_metadata["recall_usage"][0]
    assert event["step_id"] == 0
    assert event["allowed_scopes"] == ["episode"]
    assert event["selected_namespace"] == "episode:s1:e1"
    assert event["used_by_planner"] is True
    assert event["action_before_recall"] == "MOVE_FORWARD"
    assert event["action_after_recall"] == "TURN_LEFT"
    assert event["action_changed_after_recall"] is True


def test_runtime_enriches_recall_memory_with_visual_observation_and_episode_namespace():
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "kitchen", "step_id": 5},
        reason="uncertain",
        planner_backend="gateway",
    )
    memory = RecordingMemorySkill()
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(memory)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(
        make_state(step_id=5),
        payload={
            "current_image_path": "/tmp/current.png",
            "visual_observations": [
                {
                    "image_path": "/tmp/current.png",
                    "visual_observation": "Doorway ahead with sofa on right.",
                }
            ],
        },
    )

    assert result.ok is True
    assert memory.calls[0]["visual_observation"] == "Doorway ahead with sofa on right."
    assert memory.calls[0]["allowed_scopes"] == ["episode"]
    assert memory.calls[0]["memory_namespace"] == "episode:s1:e1"


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
    assert first_call["payload_summary"]["image_path"] == "/tmp/keyframe.png"
    assert "write_type" in first_call["payload_summary"]["keys"]


def test_runtime_enriches_write_memory_with_current_image_path():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"note": "current visual landmark"},
        reason="curator",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(
        make_state(step_id=8),
        payload={"current_image_path": "/tmp/current.png"},
    )

    first_call = result.runtime_metadata["tool_calls"][0]
    assert first_call["tool_name"] == "MemoryWriteSkill"
    assert first_call["payload_summary"]["image_path"] == "/tmp/current.png"


def test_runtime_write_memory_intent_stores_current_image_path():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"note": "current visual landmark"},
        reason="curator",
        planner_backend="gateway",
    )
    store = []
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(MemoryWriteSkill(store=store))
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(
        make_state(step_id=9),
        payload={"current_image_path": "/tmp/current.png"},
    )

    assert result.ok is True
    assert store[0]["image_path"] == "/tmp/current.png"


def test_runtime_enriches_write_memory_with_visual_observation():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"note": "visual landmark"},
        reason="curator",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(
        make_state(step_id=10),
        payload={
            "current_image_path": "/tmp/current.png",
            "visual_observations": [
                {
                    "image_path": "/tmp/current.png",
                    "caption": "A hallway with a doorway ahead.",
                    "visual_observation": "Doorway ahead can anchor navigation.",
                    "objects": ["doorway"],
                    "landmarks": ["doorway"],
                    "spatial_cues": ["doorway ahead"],
                    "navigation_relevance": "Useful route anchor.",
                    "confidence": 0.8,
                }
            ],
        },
    )

    first_call = result.runtime_metadata["tool_calls"][0]
    assert first_call["tool_name"] == "MemoryWriteSkill"
    assert first_call["payload_summary"]["caption"] == "A hallway with a doorway ahead."
    assert first_call["payload_summary"]["visual_observation"] == (
        "Doorway ahead can anchor navigation."
    )
    assert result.runtime_metadata["visual_analysis"]["ran"] is True


def test_runtime_adds_default_write_gate_for_visual_memory_writes():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"note": "visual landmark"},
        reason="curator",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(
        make_state(step_id=11),
        payload={
            "current_image_path": "/tmp/current.png",
            "visual_observations": [
                {
                    "image_path": "/tmp/current.png",
                    "caption": "A hallway with a doorway ahead.",
                    "landmarks": ["doorway"],
                    "confidence": 0.8,
                }
            ],
        },
    )

    first_call = result.runtime_metadata["tool_calls"][0]
    assert first_call["payload_summary"]["write_gate"]["curator_decision"] == "write"
    assert first_call["payload_summary"]["write_gate"]["candidate_reason"] == "planner_request"


def test_runtime_applies_visual_memory_curator_before_write_memory():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"note": "generic wall"},
        reason="curator",
        planner_backend="gateway",
    )
    store = []
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(VisualMemoryCuratorSkill())
    registry.register(MemoryWriteSkill(store=store))
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(
        make_state(step_id=12),
        payload={
            "current_image_path": "/tmp/current.png",
            "visual_observations": [
                {
                    "image_path": "/tmp/current.png",
                    "caption": "A blank wall.",
                    "visual_observation": "Generic wall with no navigation cue.",
                    "confidence": 0.2,
                }
            ],
        },
    )

    assert result.ok is True
    assert result.runtime_metadata["tool_calls"][0]["tool_name"] == "VisualMemoryCuratorSkill"
    assert result.runtime_metadata["tool_calls"][1]["tool_name"] == "MemoryWriteSkill"
    assert result.runtime_metadata["memory_writes"][0]["skipped"] is True
    assert result.runtime_metadata["memory_writes"][0]["write_gate"]["curator_decision"] == "skip"
    assert result.runtime_metadata["visual_analysis"]["ran"] is True
    assert result.runtime_metadata["visual_analysis"]["visual_observation"] == (
        "Generic wall with no navigation cue."
    )
    assert store == []
