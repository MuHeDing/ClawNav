from pathlib import Path

from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.gateway import FakeOpenClawGatewayClient, OpenClawGatewayError
from harness.openclaw.planner import OpenClawPlanDecision, RuleOpenClawPlanner
from harness.openclaw.runtime import OpenClawVLNRuntime
from harness.memory.memory_manager import MemoryManager
from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.skills.memory_query import MemoryQuerySkill
from harness.skills.memory_write import MemoryWriteSkill
from harness.skills.visual_memory_curator import VisualMemoryCuratorSkill
from harness.types import MemoryHit
from harness.types import SkillResult, VLNState
from harness.config import HarnessConfig
from harness.visual_readback.memory_smoke import ImageBackedLocalMemoryClient


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
    description = "Returns configured actions in order."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
        action = self.actions.pop(0) if self.actions else "STOP"
        return SkillResult.ok_result("action", {"action_text": action})


class FailingNavigationSkill(Skill):
    name = "NavigationPolicySkill"
    description = "Fails without returning an action."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        del state, payload
        return SkillResult.error_result("navigation_failed", result_type="action")


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


class ImageMemoryContextSkill(Skill):
    name = "MemoryQuerySkill"
    description = "Returns image-backed policy memory context."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self, image_path):
        self.image_path = image_path
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
        hit = MemoryHit(
            memory_id="m1",
            memory_type="semantic_frame",
            name="left opening",
            confidence=0.8,
            image_path=self.image_path,
            evidence_text="left opening near landmark",
        )
        return SkillResult.ok_result(
            "memory_query",
            {
                "memory_hits": [hit],
                "query": payload.get("text", ""),
                "policy_context": {
                    "memory_context_text": "remember the left opening",
                    "memory_images": [self.image_path],
                },
                "control_context": {"confidence": 0.8, "recall_confidence": 0.8},
            },
            confidence=0.8,
        )


class RecordingVisualReadSkill(Skill):
    name = "VisualMemoryReadSkill"
    description = "Records visual readback payload."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self):
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
        return SkillResult.ok_result(
            "visual_memory_read",
            {
                "read_status": "completed",
                "trigger_rule": payload.get("trigger_rule"),
                "candidate_action": payload.get("candidate_action"),
                "retrieved_image_paths": [hit.image_path for hit in payload["memory_hits"]],
                "actually_read_image_paths": [
                    payload["current_image_path"],
                    *[hit.image_path for hit in payload["memory_hits"]],
                ],
                "model_image_count": 1 + len(payload["memory_hits"]),
                "attached_memory_ids": [hit.memory_id for hit in payload["memory_hits"]],
                "matched_memory_ids": [hit.memory_id for hit in payload["memory_hits"]],
                "verifier_labels": ["route_conflict"],
                "readback_confidence": 0.9,
                "verifier_confidence": 0.9,
            },
            confidence=0.9,
        )


class StopBlockedVisualReadSkill(Skill):
    name = "VisualMemoryReadSkill"
    description = "Returns goal-not-visible STOP evidence."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result(
            "visual_memory_read",
            {
                "read_status": "completed",
                "trigger_rule": payload.get("trigger_rule"),
                "candidate_action": payload.get("candidate_action"),
                "retrieved_image_paths": [hit.image_path for hit in payload["memory_hits"]],
                "actually_read_image_paths": [
                    payload["current_image_path"],
                    *[hit.image_path for hit in payload["memory_hits"]],
                ],
                "model_image_count": 1 + len(payload["memory_hits"]),
                "attached_memory_ids": [hit.memory_id for hit in payload["memory_hits"]],
                "matched_memory_ids": [hit.memory_id for hit in payload["memory_hits"]],
                "verifier_labels": ["goal_not_visible"],
                "readback_confidence": 0.95,
                "verifier_confidence": 0.95,
            },
            confidence=0.95,
        )


class ActionHintVisualReadSkill(Skill):
    name = "VisualMemoryReadSkill"
    description = "Returns an executable action recommendation."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(
        self,
        audit_action_hint="TURN_LEFT",
        labels=None,
        confidence=0.9,
        recommended_action="TURN_LEFT",
        decision_scope="immediate_next_action",
        action_confidence=None,
        candidate_action_valid=False,
        should_override=True,
        invalid_reason="route_conflict",
    ):
        self.audit_action_hint = audit_action_hint
        self.labels = list(labels or ["route_conflict"])
        self.confidence = confidence
        self.recommended_action = recommended_action
        self.decision_scope = decision_scope
        self.action_confidence = action_confidence
        self.candidate_action_valid = candidate_action_valid
        self.should_override = should_override
        self.invalid_reason = invalid_reason
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
        return SkillResult.ok_result(
            "visual_memory_read",
            {
                "read_status": "completed",
                "trigger_rule": payload.get("trigger_rule"),
                "candidate_action": payload.get("candidate_action"),
                "retrieved_image_paths": [hit.image_path for hit in payload["memory_hits"]],
                "actually_read_image_paths": [
                    payload["current_image_path"],
                    *[hit.image_path for hit in payload["memory_hits"]],
                ],
                "model_image_count": 1 + len(payload["memory_hits"]),
                "attached_memory_ids": [hit.memory_id for hit in payload["memory_hits"]],
                "matched_memory_ids": [hit.memory_id for hit in payload["memory_hits"]],
                "verifier_labels": self.labels,
                "visual_evidence": ["memory view conflicts with current heading"],
                "audit_action_hint": self.audit_action_hint,
                "candidate_action_valid": self.candidate_action_valid,
                "should_override": self.should_override,
                "invalid_reason": self.invalid_reason,
                "recommended_action": self.recommended_action,
                "decision_scope": self.decision_scope,
                "action_confidence": (
                    self.confidence
                    if self.action_confidence is None
                    else self.action_confidence
                ),
                "readback_confidence": self.confidence,
                "verifier_confidence": self.confidence,
            },
            confidence=self.confidence,
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


def test_v4_controller_readback_keeps_policy_payload_clean(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "left opening", "step_id": 4},
        reason="decision_point",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    memory_skill = ImageMemoryContextSkill(str(memory))
    visual_skill = RecordingVisualReadSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(memory_skill)
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_controller",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.ok is True
    assert "memory_context_text" not in navigation.calls[0]
    assert "memory_images" not in navigation.calls[0]
    assert visual_skill.calls[0]["current_image_path"] == str(current)
    assert visual_skill.calls[0]["candidate_action"] == "TURN_LEFT"
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["read_status"] == "completed"
    assert visual_readback["trigger_rule"] == "decision_point"
    assert visual_readback["policy_context_available"] is True
    assert visual_readback["actual_policy_payload_merge"] is False
    assert visual_readback["used_by_policy"] is False
    assert visual_readback["readback_state_used_by_policy"] is False
    assert visual_readback["final_policy_payload_has_memory"] is False
    assert result.runtime_metadata["recall_usage"][0]["used_by_policy"] is False
    assert result.action_text == "TURN_LEFT"


def test_v4_smoke_seed_forces_image_hit_before_readback(tmp_path):
    current = tmp_path / "current.png"
    current.write_text("current", encoding="utf-8")
    config = HarnessConfig(
        memory_backend="image_backed_local",
        visual_readback_mode="image_read_controller",
    )
    config.visual_readback_smoke_seed_memory = True
    memory_client = ImageBackedLocalMemoryClient(memory_source=config.memory_source)
    memory_manager = MemoryManager(memory_client, config)
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={},
        reason="ordinary_gateway_act",
        planner_backend="gateway",
    )
    navigation = RecordingNavigationSkill()
    visual_skill = RecordingVisualReadSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(MemoryWriteSkill(client=memory_client))
    registry.register(MemoryQuerySkill(memory_manager))
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=config,
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    tool_names = [call["tool_name"] for call in result.runtime_metadata["tool_calls"]]
    assert "MemoryWriteSkill" in tool_names
    assert "MemoryQuerySkill" in tool_names
    assert tool_names.index("NavigationPolicySkill") < tool_names.index("MemoryQuerySkill")
    assert tool_names.index("MemoryQuerySkill") < tool_names.index("VisualMemoryReadSkill")
    assert len(memory_client.records) == 1
    assert visual_skill.calls[0]["current_image_path"] == str(current)
    assert visual_skill.calls[0]["memory_hits"][0].image_path == str(current)
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["trigger_source"] == "controlled_smoke_seed"
    assert visual_readback["read_status"] == "completed"
    assert visual_readback["actually_read_image_paths"] == [str(current), str(current)]
    assert visual_readback["matched_memory_ids"]
    assert result.runtime_metadata["recall_usage"][0]["num_hits"] == 1
    assert result.runtime_metadata["recall_usage"][0]["used_by_policy"] is False


def test_event_gated_smoke_gate_promotes_turn_candidate_with_queryable_memory_write(tmp_path):
    current = tmp_path / "current.png"
    target = tmp_path / "keyframes" / "s1" / "e1" / "step_000004.png"
    current.write_text("current-frame", encoding="utf-8")
    memory_client = ImageBackedLocalMemoryClient()
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={},
        reason="ordinary_gateway_act",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["TURN_RIGHT"]))
    registry.register(MemoryWriteSkill(client=memory_client))
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            keyframe_policy_mode="event_gated_smoke",
            visual_readback_mode="off",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={
            "run_id": str(tmp_path / "run-1"),
            "current_image_path": str(current),
            "keyframe_target_path": str(target),
        },
    )

    assert result.ok is True
    keyframe_gate = result.runtime_metadata["keyframe_gate"]
    assert keyframe_gate["save_decision"] == "save"
    assert keyframe_gate["save_reason"] == "candidate_decision_point"
    assert keyframe_gate["candidate_event_type"] == "TURN_RIGHT"
    assert keyframe_gate["promotion_status"] == "promoted"
    assert keyframe_gate["promoted_image_path"] == str(target)
    assert keyframe_gate["write_status"] == "written"
    assert keyframe_gate["memory_id"] == "image-backed-local-0"
    assert keyframe_gate["memory_id_link_status"] == "linked"
    assert keyframe_gate["controller_event_status"] == "confirmed_executed"
    assert target.read_text(encoding="utf-8") == "current-frame"
    assert "MemoryWriteSkill" in [
        call["tool_name"] for call in result.runtime_metadata["tool_calls"]
    ]
    assert len(memory_client.records) == 1
    record = memory_client.records[0]
    assert record["source_image_role"] == "event_gated_keyframe"
    assert record["metadata"]["keyframe_gate"]["candidate_event_type"] == "TURN_RIGHT"
    assert record["metadata"]["run_id"] == str(tmp_path / "run-1")


def test_event_gated_smoke_gate_memory_write_failure_does_not_abort(tmp_path):
    class FailingClient:
        def ingest_semantic(self, payload):
            del payload
            raise RuntimeError("backend down")

    current = tmp_path / "current.png"
    target = tmp_path / "keyframes" / "s1" / "e1" / "step_000004.png"
    current.write_text("current-frame", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={},
        reason="ordinary_gateway_act",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["TURN_RIGHT"]))
    registry.register(MemoryWriteSkill(client=FailingClient()))
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            keyframe_policy_mode="event_gated_smoke",
            visual_readback_mode="off",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={
            "run_id": str(tmp_path / "run-1"),
            "current_image_path": str(current),
            "keyframe_target_path": str(target),
        },
    )

    assert result.ok is True
    keyframe_gate = result.runtime_metadata["keyframe_gate"]
    assert keyframe_gate["promotion_status"] == "promoted"
    assert keyframe_gate["write_status"] == "memory_write_failed"
    assert keyframe_gate["memory_id_link_status"] == "not_linked"
    assert "backend down" in keyframe_gate["failure_reason"]


def test_event_gated_smoke_gate_failed_action_does_not_create_fallback_stop_keyframe(tmp_path):
    current = tmp_path / "current.png"
    target = tmp_path / "keyframes" / "s1" / "e1" / "step_000004.png"
    current.write_text("current-frame", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={},
        reason="ordinary_gateway_act",
        planner_backend="gateway",
    )
    visual_skill = RecordingVisualReadSkill()
    registry = SkillRegistry()
    registry.register(FailingNavigationSkill())
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            keyframe_policy_mode="event_gated_smoke",
            visual_readback_mode="image_read_controller",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={
            "run_id": str(tmp_path / "run-1"),
            "current_image_path": str(current),
            "keyframe_target_path": str(target),
        },
    )

    assert result.ok is False
    keyframe_gate = result.runtime_metadata["keyframe_gate"]
    assert keyframe_gate["save_decision"] == "skip"
    assert keyframe_gate["skip_reason"] == "candidate_action_failed"
    assert not target.exists()
    assert result.runtime_metadata["visual_readback"]["read_status"] == "skipped"
    assert result.runtime_metadata["visual_readback"]["skip_reason"] == "no_trigger"
    assert visual_skill.calls == []


def test_event_gated_smoke_bypasses_legacy_smoke_seed_write(tmp_path):
    current = tmp_path / "current.png"
    target = tmp_path / "keyframes" / "s1" / "e1" / "step_000004.png"
    current.write_text("current-frame", encoding="utf-8")
    config = HarnessConfig(
        memory_backend="image_backed_local",
        keyframe_policy_mode="event_gated_smoke",
        visual_readback_mode="image_read_controller",
    )
    config.visual_readback_smoke_seed_memory = True
    memory_client = ImageBackedLocalMemoryClient(memory_source=config.memory_source)
    memory_manager = MemoryManager(memory_client, config)
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={},
        reason="ordinary_gateway_act",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["TURN_LEFT"]))
    registry.register(MemoryWriteSkill(client=memory_client))
    registry.register(MemoryQuerySkill(memory_manager))
    registry.register(RecordingVisualReadSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=config,
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={
            "run_id": str(tmp_path / "run-1"),
            "current_image_path": str(current),
            "keyframe_target_path": str(target),
        },
    )

    tool_names = [call["tool_name"] for call in result.runtime_metadata["tool_calls"]]
    assert tool_names.count("MemoryWriteSkill") == 1
    assert len(memory_client.records) == 1
    assert memory_client.records[0]["source_image_role"] == "event_gated_keyframe"
    assert memory_client.records[0]["metadata"]["keyframe_gate"]["save_reason"] == (
        "candidate_decision_point"
    )
    assert result.runtime_metadata["keyframe_gate"]["promotion_status"] == "promoted"


def test_event_gated_smoke_readback_uses_only_prior_eligible_keyframes(tmp_path):
    config = HarnessConfig(
        memory_backend="image_backed_local",
        keyframe_policy_mode="event_gated_smoke",
        visual_readback_mode="image_read_controller",
    )
    memory_client = ImageBackedLocalMemoryClient(memory_source=config.memory_source)
    memory_manager = MemoryManager(memory_client, config)
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="act",
                tool_name="NavigationPolicySkill",
                arguments={},
                reason="first turn",
                planner_backend="gateway",
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="NavigationPolicySkill",
                arguments={},
                reason="stop check",
                planner_backend="gateway",
            ),
        ]
    )
    visual_skill = RecordingVisualReadSkill()
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["TURN_LEFT", "STOP"]))
    registry.register(MemoryWriteSkill(client=memory_client))
    registry.register(MemoryQuerySkill(memory_manager))
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=config,
    )
    current1 = tmp_path / "current1.png"
    current2 = tmp_path / "current2.png"
    target1 = tmp_path / "keyframes" / "s1" / "e1" / "step_000001.png"
    target2 = tmp_path / "keyframes" / "s1" / "e1" / "step_000002.png"
    current1.write_text("first", encoding="utf-8")
    current2.write_text("second", encoding="utf-8")

    first = runtime.step(
        make_state(step_id=1),
        payload={
            "run_id": str(tmp_path / "run-1"),
            "current_image_path": str(current1),
            "keyframe_target_path": str(target1),
        },
    )
    second = runtime.step(
        make_state(step_id=2),
        payload={
            "run_id": str(tmp_path / "run-1"),
            "current_image_path": str(current2),
            "keyframe_target_path": str(target2),
        },
    )

    assert first.runtime_metadata["visual_readback"]["read_status"] == "skipped"
    assert first.runtime_metadata["visual_readback"]["eligible_prior_keyframe_count"] == 0
    assert second.runtime_metadata["visual_readback"]["read_status"] == "completed"
    assert second.runtime_metadata["visual_readback"]["eligible_prior_keyframe_count"] == 1
    assert second.runtime_metadata["visual_readback"]["candidate_pool_backend_returned_count"] == 1
    assert second.runtime_metadata["visual_readback"]["candidate_pool_attached_count"] == 1
    assert second.runtime_metadata["visual_readback"]["candidate_pool_exact_duplicate_drop_count"] == 0
    assert visual_skill.calls[-1]["memory_hits"][0].image_path == str(target1)
    assert visual_skill.calls[-1]["memory_hits"][0].metadata["step_id"] == 1
    assert visual_skill.calls[-1]["memory_hits"][0].metadata["source_image_role"] == (
        "event_gated_keyframe"
    )


def test_v4_stop_block_defaults_to_shadow_log_only(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "goal", "step_id": 4},
        reason="risky_stop",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["STOP"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(StopBlockedVisualReadSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_controller",
            visual_readback_stop_fallback_policy="log_only",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "STOP"
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["controller_decision"] == "block_stop_shadow"
    assert visual_readback["fallback_policy"] == "log_only"
    assert visual_readback["fallback_denominator"] == "shadow_primary"
    assert visual_readback["fallback_source"] == "log_only"
    assert visual_readback["final_action"] == "STOP"
    assert visual_readback["executed_action_changed_after_visual_read"] is False


def test_v4_stop_block_executed_pilot_uses_previous_non_stop(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="act",
                tool_name="NavigationPolicySkill",
                arguments={},
                reason="move",
                planner_backend="gateway",
            ),
            OpenClawPlanDecision(
                intent="recall_memory",
                tool_name="MemoryQuerySkill",
                arguments={"text": "goal", "step_id": 5},
                reason="risky_stop",
                planner_backend="gateway",
            ),
        ]
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["MOVE_FORWARD", "STOP"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(StopBlockedVisualReadSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_controller",
            visual_readback_stop_fallback_policy="previous_non_stop_else_move_forward",
        ),
    )

    first = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )
    second = runtime.step(
        make_state(step_id=5),
        payload={"current_image_path": str(current)},
    )

    assert first.action_text == "MOVE_FORWARD"
    assert second.action_text == "MOVE_FORWARD"
    visual_readback = second.runtime_metadata["visual_readback"]
    assert visual_readback["controller_decision"] == "block_stop_executed"
    assert visual_readback["fallback_denominator"] == "executed_pilot"
    assert visual_readback["fallback_source"] == "previous_non_stop"
    assert visual_readback["final_action"] == "MOVE_FORWARD"
    assert visual_readback["executed_action_changed_after_visual_read"] is True


def test_image_read_replan_prompt_executes_clean_policy_reassessment(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="decision_point",
        planner_backend="gateway",
    )
    navigation = SequenceNavigationSkill(["TURN_LEFT", "MOVE_FORWARD"])
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(RecordingVisualReadSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_replan_prompt",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "MOVE_FORWARD"
    assert len(navigation.calls) == 2
    first_payload, replan_payload = navigation.calls
    assert "active_subgoal" not in first_payload
    assert "memory_context_text" not in first_payload
    assert "memory_images" not in first_payload
    assert replan_payload["active_subgoal"].startswith("Reassess the route")
    assert "Verifier labels: route_conflict." in replan_payload["memory_context_text"]
    assert "memory_images" not in replan_payload
    tool_names = [call["tool_name"] for call in result.runtime_metadata["tool_calls"]]
    assert tool_names.count("NavigationPolicySkill") == 2
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["controller_decision"] == "execute_replan"
    assert visual_readback["replan_request_logged_after_visual_read"] is True
    assert visual_readback["replan_executed_after_visual_read"] is True
    assert visual_readback["readback_state_used_by_policy"] is True
    assert visual_readback["used_by_policy"] is True
    assert visual_readback["replan_action_text"] == "MOVE_FORWARD"
    assert visual_readback["final_action"] == "MOVE_FORWARD"
    assert visual_readback["executed_action_changed_after_visual_read"] is True
    assert visual_readback["controller_private_replan_state"]["route_stage"] == (
        "visual_route_conflict_reassess"
    )


def test_image_read_action_override_executes_recommended_action(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="motion_consistency_check",
        planner_backend="gateway",
    )
    navigation = SequenceNavigationSkill(["MOVE_FORWARD"])
    visual_skill = ActionHintVisualReadSkill(audit_action_hint="TURN_LEFT")
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "TURN_LEFT"
    assert len(navigation.calls) == 1
    assert visual_skill.calls[0]["candidate_action"] == "MOVE_FORWARD"
    assert visual_skill.calls[0]["trigger_rule"] == "motion_consistency_check"
    tool_names = [call["tool_name"] for call in result.runtime_metadata["tool_calls"]]
    assert tool_names.count("NavigationPolicySkill") == 1
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["controller_decision"] == "execute_action_hint"
    assert visual_readback["action_hint_override_attempted_after_visual_read"] is True
    assert visual_readback["action_hint_executed_after_visual_read"] is True
    assert visual_readback["action_hint_used_as_executable_action"] is True
    assert visual_readback["action_hint_action_text"] == "TURN_LEFT"
    assert visual_readback["readback_state_used_by_controller"] is True
    assert visual_readback["readback_state_used_by_policy"] is False
    assert visual_readback["final_action"] == "TURN_LEFT"
    assert visual_readback["executed_action_changed_after_visual_read"] is True


def test_image_read_action_override_keeps_valid_candidate_action(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="motion_consistency_check",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["MOVE_FORWARD"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(
        ActionHintVisualReadSkill(
            candidate_action_valid=True,
            should_override=False,
            invalid_reason="",
            recommended_action="TURN_LEFT",
        )
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "MOVE_FORWARD"
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["candidate_action_valid"] is True
    assert visual_readback["should_override"] is False
    assert visual_readback["action_hint_executed_after_visual_read"] is False
    assert visual_readback["action_hint_override_failure_reason"] == (
        "candidate_action_valid"
    )


def test_image_read_action_override_rejects_payload_without_judge_decision(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="motion_consistency_check",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["MOVE_FORWARD"]))
    registry.register(ImageMemoryContextSkill(str(memory)))

    class LegacyRecommendedActionSkill(Skill):
        name = "VisualMemoryReadSkill"
        description = "Returns the old recommendation-only payload."
        input_schema = {"type": "object"}
        output_schema = {"type": "object"}

        def run(self, state, payload):
            del state
            return SkillResult.ok_result(
                "visual_memory_read",
                {
                    "read_status": "completed",
                    "trigger_rule": payload.get("trigger_rule"),
                    "candidate_action": payload.get("candidate_action"),
                    "verifier_labels": ["route_conflict"],
                    "audit_action_hint": "TURN_LEFT",
                    "recommended_action": "TURN_LEFT",
                    "decision_scope": "immediate_next_action",
                    "action_confidence": 0.9,
                    "verifier_confidence": 0.9,
                },
            )

    registry.register(LegacyRecommendedActionSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "MOVE_FORWARD"
    assert result.runtime_metadata["visual_readback"][
        "action_hint_override_failure_reason"
    ] == "candidate_action_valid"


def test_image_read_action_override_ignores_legacy_text_direction_without_recommended_action(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="motion_consistency_check",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["TURN_LEFT"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(
        ActionHintVisualReadSkill(
            audit_action_hint=(
                "The target doorway is to the right of the fireplace, but this "
                "describes a landmark relation rather than an immediate action."
            ),
            recommended_action="",
            decision_scope="landmark_relation",
        )
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "TURN_LEFT"
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["action_hint_executed_after_visual_read"] is False
    assert visual_readback["action_hint_override_failure_reason"] == (
        "not_immediate_next_action"
    )


def test_image_read_action_override_does_not_execute_goal_not_visible_action(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "goal", "step_id": 4},
        reason="risky_stop",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["STOP"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(
        ActionHintVisualReadSkill(
            labels=["goal_not_visible"],
            recommended_action="MOVE_FORWARD",
            decision_scope="immediate_next_action",
            action_confidence=0.95,
        )
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "STOP"
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["action_hint_executed_after_visual_read"] is False
    assert visual_readback["action_hint_override_failure_reason"] == (
        "unsupported_verifier_labels"
    )


def test_image_read_action_override_respects_episode_override_limit(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="motion_consistency_check",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["MOVE_FORWARD", "MOVE_FORWARD"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(ActionHintVisualReadSkill(recommended_action="TURN_LEFT"))
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
            visual_readback_max_action_overrides_per_episode=1,
        ),
    )

    first = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )
    second = runtime.step(
        make_state(step_id=5),
        payload={"current_image_path": str(current)},
    )

    assert first.action_text == "TURN_LEFT"
    assert second.action_text == "MOVE_FORWARD"
    visual_readback = second.runtime_metadata["visual_readback"]
    assert visual_readback["action_hint_executed_after_visual_read"] is False
    assert visual_readback["action_hint_override_failure_reason"] == (
        "max_action_overrides_per_episode_reached"
    )


def test_image_read_action_override_adaptive_judges_each_action_without_step_budget(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="motion_consistency_check",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["MOVE_FORWARD", "MOVE_FORWARD"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(ActionHintVisualReadSkill(recommended_action="TURN_LEFT"))
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
            visual_readback_max_action_overrides_per_episode="adaptive",
        ),
    )

    first = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )
    second = runtime.step(
        make_state(step_id=5),
        payload={"current_image_path": str(current)},
    )

    assert first.action_text == "TURN_LEFT"
    assert second.action_text == "TURN_LEFT"
    visual_readback = second.runtime_metadata["visual_readback"]
    assert visual_readback["action_hint_executed_after_visual_read"] is True
    assert visual_readback["executed_action_changed_after_visual_read"] is True
    assert visual_readback["action_override_budget_policy"] == "adaptive"
    assert visual_readback["action_override_budget_limit"] == "per_decision"


def test_same_action_recommendation_does_not_consume_override_budget(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="motion_consistency_check",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["TURN_LEFT", "MOVE_FORWARD"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    visual_skill = ActionHintVisualReadSkill(recommended_action="TURN_LEFT")
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
            visual_readback_max_action_overrides_per_episode=1,
        ),
    )

    first = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )
    visual_skill.recommended_action = "TURN_RIGHT"
    second = runtime.step(
        make_state(step_id=5),
        payload={"current_image_path": str(current)},
    )

    assert first.action_text == "TURN_LEFT"
    assert second.action_text == "TURN_RIGHT"
    assert first.runtime_metadata["visual_readback"][
        "executed_action_changed_after_visual_read"
    ] is False
    assert second.runtime_metadata["visual_readback"][
        "executed_action_changed_after_visual_read"
    ] is True


def test_sparse_action_override_skips_plain_move_forward_readback(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="ordinary_forward",
        planner_backend="gateway",
    )
    navigation = SequenceNavigationSkill(["MOVE_FORWARD"])
    visual_skill = ActionHintVisualReadSkill(audit_action_hint="TURN_LEFT")
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "MOVE_FORWARD"
    assert visual_skill.calls == []
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["read_status"] == "skipped"
    assert visual_readback["skip_reason"] == "no_trigger"
    assert visual_readback["trigger_rule"] == ""
    assert visual_readback["trigger_policy"] == "sparse_action_override"
    assert visual_readback["action_hint_override_failure_reason"] == "readback_not_completed"
    assert result.runtime_metadata["visual_readback_config"][
        "visual_readback_trigger_policy"
    ] == "sparse_action_override"


def test_sparse_action_override_reads_saved_keyframe_turn_event(tmp_path):
    current = tmp_path / "current.png"
    target = tmp_path / "keyframes" / "s1" / "e1" / "step_000010.png"
    prior = tmp_path / "keyframes" / "s1" / "e1" / "step_000000.png"
    current.write_text("current", encoding="utf-8")
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text("prior", encoding="utf-8")
    run_id = str(tmp_path / "run-1")
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={},
        reason="ordinary_gateway_act",
        planner_backend="gateway",
    )
    navigation = SequenceNavigationSkill(["TURN_RIGHT"])
    visual_skill = ActionHintVisualReadSkill(audit_action_hint="TURN_LEFT")
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            keyframe_policy_mode="event_gated_smoke",
            keyframe_min_gap_steps=5,
        ),
    )
    runtime.event_gated_keyframe_ledger.append(
        {
            "run_id": run_id,
            "scene_id": "s1",
            "episode_id": "e1",
            "step_id": 0,
            "memory_id": "image-backed-local-0",
            "memory_type": "semantic_frame",
            "memory_namespace": "episode:s1:e1",
            "source_image_role": "event_gated_keyframe",
            "image_path": str(prior),
            "retrieval_text": "prior hallway keyframe",
            "confidence": 1.0,
        }
    )

    result = runtime.step(
        make_state(step_id=10),
        payload={
            "run_id": run_id,
            "current_image_path": str(current),
            "keyframe_target_path": str(target),
        },
    )

    assert result.action_text == "TURN_LEFT"
    assert visual_skill.calls[0]["candidate_action"] == "TURN_RIGHT"
    assert visual_skill.calls[0]["trigger_rule"] == "decision_point"
    assert visual_skill.calls[0]["memory_hits"][0].image_path == str(prior)
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["trigger_policy"] == "sparse_action_override"
    assert visual_readback["candidate_pool_attached_count"] == 1
    assert visual_readback["action_hint_executed_after_visual_read"] is True
    assert result.runtime_metadata["keyframe_gate"]["save_decision"] == "save"


def test_image_read_action_override_rejects_invalid_recommended_action(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    decision = OpenClawPlanDecision(
        intent="recall_memory",
        tool_name="MemoryQuerySkill",
        arguments={"text": "route", "step_id": 4},
        reason="motion_consistency_check",
        planner_backend="gateway",
    )
    registry = SkillRegistry()
    registry.register(SequenceNavigationSkill(["MOVE_FORWARD"]))
    registry.register(ImageMemoryContextSkill(str(memory)))
    registry.register(
        ActionHintVisualReadSkill(
            audit_action_hint="fly upward",
            recommended_action="fly upward",
        )
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=HarnessConfig(
            memory_backend="image_backed_local",
            visual_readback_mode="image_read_action_override",
            visual_readback_trigger_policy="dense_action_override",
        ),
    )

    result = runtime.step(
        make_state(step_id=4),
        payload={"current_image_path": str(current)},
    )

    assert result.action_text == "MOVE_FORWARD"
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["action_hint_executed_after_visual_read"] is False
    assert visual_readback["action_hint_override_failure_reason"] == (
        "invalid_recommended_action"
    )


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


def test_runtime_bridges_gateway_visual_update_audit_into_image_memory_readback(tmp_path):
    current = tmp_path / "current.png"
    current.write_text("current", encoding="utf-8")
    config = HarnessConfig(
        memory_backend="image_backed_local",
        visual_readback_mode="image_read_controller",
    )
    memory_client = ImageBackedLocalMemoryClient(memory_source=config.memory_source)
    memory_manager = MemoryManager(memory_client, config)
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="NavigationPolicySkill",
        arguments={},
        reason=(
            "The doorway is visible on the left side of the image, so turn left "
            "toward the exit."
        ),
        planner_backend="gateway",
        runtime_metadata={
            "context_audit": {
                "planner_step_mode": "visual_update",
                "visual_memory_update_status": "updated",
                "model_image_paths": [str(current)],
            }
        },
    )
    navigation = RecordingNavigationSkill()
    visual_skill = RecordingVisualReadSkill()
    registry = SkillRegistry()
    registry.register(navigation)
    registry.register(VisualMemoryCuratorSkill())
    registry.register(MemoryWriteSkill(client=memory_client))
    registry.register(MemoryQuerySkill(memory_manager))
    registry.register(visual_skill)
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=config,
    )

    result = runtime.step(
        make_state(step_id=0),
        payload={"current_image_path": str(current)},
    )

    tool_names = [call["tool_name"] for call in result.runtime_metadata["tool_calls"]]
    assert "MemoryWriteSkill" in tool_names
    assert "MemoryQuerySkill" in tool_names
    assert "VisualMemoryReadSkill" in tool_names
    assert tool_names.index("MemoryWriteSkill") < tool_names.index("MemoryQuerySkill")
    assert tool_names.index("MemoryQuerySkill") < tool_names.index("VisualMemoryReadSkill")
    assert len(memory_client.records) == 1
    assert memory_client.records[0]["image_path"] == str(current)
    assert memory_client.records[0]["source_image_role"] == "gateway_visual_update"
    assert memory_client.records[0]["metadata"]["gateway_visual_memory_update_status"] == (
        "updated"
    )
    assert visual_skill.calls[0]["memory_hits"][0].image_path == str(current)
    visual_readback = result.runtime_metadata["visual_readback"]
    assert visual_readback["trigger_source"] == "online_controller"
    assert visual_readback["read_status"] == "completed"
    assert visual_readback["actually_read_image_paths"] == [str(current), str(current)]
    assert visual_readback["matched_memory_ids"]
    assert result.runtime_metadata["recall_usage"][0]["num_hits"] == 1
    assert result.runtime_metadata["recall_usage"][0]["used_by_policy"] is False


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
