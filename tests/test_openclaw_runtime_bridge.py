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


class SequenceNavigationSkill(Skill):
    name = "NavigationPolicySkill"
    description = "Records payloads and returns actions in order."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
        action = self.actions.pop(0) if self.actions else "TURN_LEFT"
        return SkillResult.ok_result("action", {"action_text": action})


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


class MemoryContextSkill(Skill):
    name = "MemoryQuerySkill"
    description = "Returns policy memory context."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self):
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
        return SkillResult.ok_result(
            "memory",
            {
                "memory_hits": [],
                "query": payload.get("text", ""),
                "step_id": payload.get("step_id", state.step_id),
                "policy_context": {
                    "memory_context_text": "remember the bright doorway",
                    "memory_images": ["/tmp/doorway.png"],
                },
            },
        )


class FailingGatewayPlanner:
    def plan(self, state, runtime_context):
        raise OpenClawGatewayError("502 Server Error: Bad Gateway for url: http://gateway/plan")


class StaticPlanner:
    def __init__(self, decision):
        self.decision = decision

    def plan(self, state, runtime_context):
        return self.decision


class RecordingPlanner:
    def __init__(self, decision):
        self.decision = decision
        self.payloads = []

    def plan(self, state, runtime_context):
        self.payloads.append(dict(runtime_context))
        return self.decision


class SequencePlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.payloads = []

    def plan(self, state, runtime_context):
        self.payloads.append(dict(runtime_context))
        if self.decisions:
            return self.decisions.pop(0)
        return OpenClawPlanDecision(
            intent="act",
            tool_name="NavigationPolicySkill",
            arguments={"action_text": "TURN_LEFT"},
            reason="default",
            planner_backend="gateway",
        )


class RecallThenReplanPlanner:
    def __init__(self):
        self.payloads = []

    def plan(self, state, runtime_context):
        self.payloads.append(dict(runtime_context))
        if runtime_context.get("memory_context_text"):
            return OpenClawPlanDecision(
                intent="replan",
                tool_name="ReplannerSkill",
                arguments={
                    "action_text": "MOVE_FORWARD",
                    "active_subgoal": "follow recalled doorway",
                },
                reason="memory_changed_plan",
                planner_backend="gateway",
            )
        return OpenClawPlanDecision(
            intent="recall_memory",
            tool_name="MemoryQuerySkill",
            arguments={"text": "doorway", "allowed_scopes": ["episode"]},
            reason="need_memory",
            planner_backend="gateway",
        )


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


def test_runtime_preserves_gateway_context_audit_metadata():
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
                "runtime_metadata": {
                    "context_audit": {
                        "openclaw_session_mode": "fresh_per_step",
                        "provider_input_tokens": 900,
                    }
                },
            }
        ),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=1), payload={})

    assert result.runtime_metadata["context_audit"]["openclaw_session_mode"] == (
        "fresh_per_step"
    )
    assert result.runtime_metadata["context_audit"]["provider_input_tokens"] == 900


def test_runtime_injects_bounded_context_engine_summary_on_next_step(tmp_path):
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "TURN_LEFT"},
        reason="planner chose visible doorway",
        planner_backend="gateway",
    )
    planner = RecordingPlanner(decision)
    registry = SkillRegistry()
    registry.register(RecordingNavigationSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    first = runtime.step(
        make_state(step_id=0),
        payload={"run_id": str(tmp_path), "current_image_path": "/tmp/step0.png"},
    )
    second = runtime.step(make_state(step_id=1), payload={"run_id": str(tmp_path)})

    assert first.runtime_metadata["context_engine"]["recorded"] is True
    assert second.runtime_metadata["context_engine"]["recorded"] is True
    assert planner.payloads[0]["task_state"]["current_step_id"] == 0
    assert "recent_step_summary" not in planner.payloads[0]
    assert planner.payloads[1]["task_state"]["last_action_text"] == "TURN_LEFT"
    assert "Step 0: TURN_LEFT" in planner.payloads[1]["recent_step_summary"]
    assert (tmp_path / "openclaw_context_engine" / "running_summary.md").exists()


def test_runtime_mirrors_memory_writes_into_context_engine_retrieval(tmp_path):
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="write_memory",
                tool_name="MemoryWriteSkill",
                arguments={
                    "step_id": 0,
                    "note": "blue doorway landmark",
                    "caption": "blue doorway ahead",
                },
                reason="store landmark",
                planner_backend="gateway",
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="NavigationPolicySkill",
                arguments={"action_text": "MOVE_FORWARD"},
                reason="use recalled landmark",
                planner_backend="gateway",
            ),
        ]
    )
    registry = SkillRegistry()
    registry.register(RecordingNavigationSkill())
    registry.register(MemoryWriteSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    first = runtime.step(make_state(step_id=0), payload={"run_id": str(tmp_path)})
    second = runtime.step(
        make_state(step_id=1),
        payload={"run_id": str(tmp_path), "memory_query": "blue doorway"},
    )

    assert first.runtime_metadata["context_engine"]["mirrored_memory_ids"] == [
        "mem_000001"
    ]
    assert second.runtime_metadata["context_engine"]["recorded"] is True
    assert planner.payloads[1]["retrieved_memory_ids"] == ["mem_000001"]
    assert "blue doorway ahead" in planner.payloads[1]["memory_context_text"]


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


def test_runtime_can_treat_planner_action_as_guidance_without_skipping_policy():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "MOVE_FORWARD", "active_subgoal": "follow the archway"},
        reason="planner guidance",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
    )

    result = runtime.step(make_state(step_id=4), payload={})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert navigation.calls[0]["active_subgoal"] == "follow the archway"
    assert "planner_action_override" not in result.runtime_metadata
    assert result.runtime_metadata["planner_action_guidance"] == "MOVE_FORWARD"


def test_runtime_does_not_promote_planner_action_to_policy_subgoal():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "MOVE_FORWARD"},
        reason="turn toward the doorway",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
    )

    result = runtime.step(make_state(step_id=4), payload={})

    assert result.ok is True
    assert "active_subgoal" not in navigation.calls[0]
    assert result.runtime_metadata["planner_action_guidance"] == "MOVE_FORWARD"


def test_runtime_filters_planner_stop_guidance_without_skipping_policy():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "STOP"},
        reason="goal reached",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
    )

    result = runtime.step(make_state(step_id=4), payload={})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert "active_subgoal" not in navigation.calls[0]
    assert "planner_action_override" not in result.runtime_metadata
    assert "policy_skipped" not in result.runtime_metadata
    assert result.runtime_metadata["planner_action_guidance"] == "STOP"


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


def test_runtime_stops_on_openclaw_cli_model_fallback_without_policy_call():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"planner_error": "qwen request timed out"},
        reason="openclaw_cli_model_fallback:openclaw_cli_default_act",
        planner_backend="gateway",
        runtime_metadata={
            "context_audit": {
                "planner_step_mode": "fast_text",
                "qwen_api_called": True,
            }
        },
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=22), payload={})

    assert result.ok is False
    assert result.action_text == "STOP"
    assert result.error == "qwen request timed out"
    assert navigation.calls == []
    assert result.runtime_metadata["planner_model_fallback"] is True
    assert result.runtime_metadata["planner_fallback"] is True
    assert result.runtime_metadata["planner_reason"] == (
        "openclaw_cli_model_fallback:openclaw_cli_default_act"
    )
    assert result.runtime_metadata["context_audit"]["planner_step_mode"] == "fast_text"
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


def test_runtime_clean_off_suppresses_navigation_policy_memory_payload():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={
            "active_subgoal": "continue toward doorway",
            "memory_context_text": "remember the doorway",
            "memory_images": ["/tmp/doorway.png"],
        },
        reason="memory fast path",
        planner_backend="gateway",
        runtime_metadata={"memory_gate": {"mode": "safe_cue", "policy_context_used": True}},
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_memory_context_enabled=False,
        allow_planner_action_override=False,
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={
            "active_subgoal": "payload subgoal",
            "memory_context_text": "payload memory",
            "recent_frames": ["frame0"],
        },
    )

    assert result.ok is True
    assert navigation.calls == [{"recent_frames": ["frame0"]}]
    assert result.runtime_metadata["policy_memory_context_enabled"] is False
    assert result.runtime_metadata["memory_gate"]["mode"] == "safe_cue"


def test_runtime_clean_off_suppresses_memory_query_policy_context():
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "doorway"},
        reason="recall",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(MemoryContextSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_memory_context_enabled=False,
    )

    result = runtime.step(make_state(step_id=0), payload={"recent_frames": ["frame0"]})

    assert result.ok is True
    assert navigation.calls == [{"recent_frames": ["frame0"]}]


def test_runtime_audit_clean_prompt_stop_verification_keeps_memory_stop():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"memory_context_text": "Navigation cue: doorway", "active_subgoal": "doorway"},
        reason="memory stop risk",
        planner_backend="gateway",
        runtime_metadata={
            "memory_gate": {
                "mode": "safe_cue",
                "policy_context_used": True,
                "control_context": {"requires_stop_verification": True},
            }
        },
    )
    navigation = SequenceNavigationSkill(["STOP", "MOVE_FORWARD"])
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        stop_verification_mode="audit_clean_prompt",
        allow_planner_action_override=False,
    )

    result = runtime.step(make_state(step_id=8), payload={"recent_frames": ["frame0"]})

    assert result.ok is True
    assert result.action_text == "STOP"
    assert len(navigation.calls) == 2
    assert navigation.calls[0]["memory_context_text"] == "Navigation cue: doorway"
    assert navigation.calls[1] == {"recent_frames": ["frame0"]}
    verifier = result.runtime_metadata["stop_verifier"]
    assert verifier["mode"] == "audit_clean_prompt"
    assert verifier["triggered"] is True
    assert verifier["memory_action"] == "STOP"
    assert verifier["clean_action"] == "MOVE_FORWARD"
    assert verifier["disagreement"] is True
    assert verifier["blocked"] is False


def test_runtime_clean_prompt_block_replaces_memory_stop_with_clean_action():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"memory_context_text": "Navigation cue: doorway", "active_subgoal": "doorway"},
        reason="memory stop risk",
        planner_backend="gateway",
        runtime_metadata={
            "memory_gate": {
                "mode": "safe_cue",
                "policy_context_used": True,
                "control_context": {"requires_stop_verification": True},
            }
        },
    )
    navigation = SequenceNavigationSkill(["STOP", "MOVE_FORWARD"])
    registry = SkillRegistry()
    registry.register(navigation)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        stop_verification_mode="clean_prompt_block",
        allow_planner_action_override=False,
    )

    result = runtime.step(make_state(step_id=8), payload={"recent_frames": ["frame0"]})

    assert result.ok is True
    assert result.action_text == "MOVE_FORWARD"
    assert result.executor_command["action_index"] == 1
    assert result.runtime_metadata["stop_verifier"]["blocked"] is True
    assert result.runtime_metadata["stop_verifier"]["executed_action"] == "MOVE_FORWARD"


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


def test_runtime_records_planner_intent_change_after_recall():
    planner = RecallThenReplanPlanner()
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(EchoMemorySkill())
    registry.register(EchoReplannerSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(
        make_state(step_id=1),
        payload={"policy_action": "TURN_LEFT"},
    )

    assert result.action_text == "MOVE_FORWARD"
    assert len(planner.payloads) == 2
    assert planner.payloads[1]["memory_context_text"] == "kitchen"
    event = result.runtime_metadata["recall_usage"][0]
    assert event["planner_intent_before_recall"] == "recall_memory"
    assert event["planner_intent_after_recall"] == "replan"
    assert event["replan_created_after_recall"] is True
    assert event["used_by_planner"] is True
    assert event["action_before_recall"] == "TURN_LEFT"
    assert event["action_after_recall"] == "MOVE_FORWARD"
    assert event["action_changed_after_recall"] is True


def test_runtime_uses_planner_visual_analysis_latency_metadata():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "MOVE_FORWARD"},
        reason="visual metadata",
        planner_backend="gateway",
        runtime_metadata={
            "visual_analysis": {
                "ran": True,
                "vlm_latency_ms": 37.5,
                "num_images": 1,
            }
        },
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(make_state(step_id=2), payload={})

    assert result.runtime_metadata["visual_analysis"]["vlm_latency_ms"] == 37.5


def test_runtime_auto_writes_and_recalls_planner_visual_analysis_for_policy():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "MOVE_FORWARD"},
        reason="visual doorway guidance",
        planner_backend="gateway",
        runtime_metadata={
            "visual_analysis": {
                "ran": True,
                "image_paths": ["/tmp/current.png"],
                "observations": [
                    {
                        "image_path": "/tmp/current.png",
                        "caption": "A bright doorway next to the hall.",
                        "visual_observation": "The doorway is a useful navigation landmark.",
                        "landmarks": ["bright doorway"],
                        "confidence": 0.9,
                    }
                ],
            }
        },
    )
    store = []
    navigation = RecordingNavigationSkill()
    memory = MemoryContextSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(memory)
    registry.register(VisualMemoryCuratorSkill())
    registry.register(MemoryWriteSkill(store=store))
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
    )

    result = runtime.step(
        make_state(step_id=0),
        payload={"current_image_path": "/tmp/current.png"},
    )

    assert result.ok is True
    assert store[0]["image_path"] == "/tmp/current.png"
    assert result.runtime_metadata["tool_calls"][0]["tool_name"] == "VisualMemoryCuratorSkill"
    assert result.runtime_metadata["tool_calls"][1]["tool_name"] == "MemoryWriteSkill"
    assert result.runtime_metadata["tool_calls"][2]["tool_name"] == "MemoryQuerySkill"
    assert result.runtime_metadata["memory_writes"][0]["written"] is True
    assert result.runtime_metadata["recall_usage"][0]["used_by_policy"] is True
    assert navigation.calls[0]["memory_context_text"] == "remember the bright doorway"
    assert "memory_images" not in navigation.calls[0]


def test_runtime_supplies_auto_recalled_visual_memory_to_next_planner_call():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={"action_text": "MOVE_FORWARD"},
        reason="visual doorway guidance",
        planner_backend="gateway",
        runtime_metadata={
            "visual_analysis": {
                "ran": True,
                "image_paths": ["/tmp/current.png"],
                "observations": [
                    {
                        "image_path": "/tmp/current.png",
                        "caption": "A bright doorway next to the hall.",
                        "visual_observation": "The doorway is a useful navigation landmark.",
                        "landmarks": ["bright doorway"],
                        "confidence": 0.9,
                    }
                ],
            }
        },
    )
    store = []
    planner = RecordingPlanner(decision)
    navigation = RecordingNavigationSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(MemoryContextSkill())
    registry.register(VisualMemoryCuratorSkill())
    registry.register(MemoryWriteSkill(store=store))
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
    )

    runtime.step(make_state(step_id=0), payload={"current_image_path": "/tmp/current.png"})
    runtime.step(make_state(step_id=1), payload={"current_image_path": "/tmp/next.png"})

    assert planner.payloads[0].get("memory_context_text") is None
    assert planner.payloads[1]["memory_context_text"] == "remember the bright doorway"
    assert planner.payloads[1]["memory_images"] == ["/tmp/doorway.png"]
    assert navigation.calls[1]["memory_context_text"] == "remember the bright doorway"
    assert "memory_images" not in navigation.calls[1]


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


def test_runtime_supplies_recent_visual_memories_to_curator_for_duplicate_skip():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"note": "red door landmark"},
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
    payload = {
        "current_image_path": "/tmp/red-door.png",
        "visual_observations": [
            {
                "image_path": "/tmp/red-door.png",
                "caption": "A red door next to a sofa.",
                "objects": ["sofa"],
                "landmarks": ["red door"],
                "spatial_cues": ["sofa on right"],
                "confidence": 0.9,
            }
        ],
    }

    first = runtime.step(make_state(step_id=13), payload=payload)
    second = runtime.step(make_state(step_id=14), payload=payload)

    assert first.runtime_metadata["memory_writes"][0]["written"] is True
    second_write = second.runtime_metadata["memory_writes"][0]
    assert second_write["skipped"] is True
    assert second_write["write_gate"]["curator_decision"] == "skip"
    assert second_write["write_gate"]["duplicate_of_memory_id"] == "/tmp/red-door.png"
