import pytest

from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.memory.episode_visual_store import EpisodeVisualMemoryStore
from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.gateway import (
    FakeOpenClawGatewayClient,
    InstructionSegmentationResult,
    OpenClawGatewayError,
)
from harness.openclaw.instruction_stages import parse_instruction_stage_plan
from harness.openclaw.planner import OpenClawPlanDecision, RuleOpenClawPlanner
from harness.openclaw.runtime import OpenClawVLNRuntime
from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.skills.memory_write import MemoryWriteSkill
from harness.skills.visual_memory_curator import VisualMemoryCuratorSkill
from harness.types import MemoryHit, SkillResult, VLNState


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


class ImageBackedMemorySkill(Skill):
    name = "MemoryQuerySkill"
    description = "Returns one image-backed memory and records calls."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self, image_path="/tmp/history.png", fail=False):
        self.image_path = image_path
        self.fail = fail
        self.calls = []

    def run(self, state, payload):
        self.calls.append(dict(payload))
        if self.fail:
            return SkillResult.error_result("query unavailable")
        return SkillResult.ok_result(
            "memory_query",
            {
                "memory_hits": [
                    MemoryHit(
                        memory_id="query_mem_1",
                        memory_type="episode_visual",
                        name="hallway",
                        confidence=0.9,
                        image_path=self.image_path,
                        metadata={"stage_id": "stage_00", "image_backed": True},
                    )
                ],
                "policy_context": {"memory_images": [self.image_path]},
            },
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
        raise OpenClawGatewayError(
            "502 Server Error: Bad Gateway for url: http://gateway/plan"
        )


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


class StagedRecordingPlanner(RecordingPlanner):
    def __init__(self, decision, stages=None, plan_errors=None):
        super().__init__(decision)
        self.stages = stages or [
            {
                "order": 0,
                "route_clause": "go to the kitchen",
                "transition_type": "final_arrival",
                "expected_landmarks": ["kitchen"],
                "completion_cues": ["inside the kitchen"],
                "final_stage": True,
            }
        ]
        self.segment_calls = []
        self.plan_errors = list(plan_errors or [])

    def segment_instruction(self, scene_id, episode_id, instruction):
        self.segment_calls.append((scene_id, episode_id, instruction))
        plan = parse_instruction_stage_plan(
            {"schema_version": "instruction_stages_v1", "stages": self.stages},
            instruction,
        )
        return InstructionSegmentationResult(
            stage_plan=plan,
            runtime_metadata={"segmentation_source": "qwen"},
        )

    def plan(self, state, runtime_context):
        self.payloads.append(dict(runtime_context))
        if self.plan_errors:
            error = self.plan_errors.pop(0)
            if error:
                raise error
        return self.decision


class StagedSequencePlanner(StagedRecordingPlanner):
    def __init__(self, decisions, stages=None):
        super().__init__(decisions[-1], stages=stages)
        self.decisions = list(decisions)

    def plan(self, state, runtime_context):
        self.payloads.append(dict(runtime_context))
        if self.decisions:
            return self.decisions.pop(0)
        return self.decision


def route_v2_decision(**overrides):
    arguments = {
        "action_text": "TURN_LEFT",
        "confidence": 0.8,
        "visual_summary": "The billiard table is clearly visible on the left.",
        "progress_state": "following route",
        "route_stage": "intermediate_landmark",
        "confirmed_landmarks": ["billiard table"],
        "current_target": "window",
        "target_relation": "ahead",
        "stop_evidence": "none",
        "semantic_stop_state": "not_ready",
        "reason": "continue",
    }
    arguments.update(overrides)
    return OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments=arguments,
        reason=str(arguments["reason"]),
        planner_backend="gateway",
        runtime_metadata={"context_audit": {"qwen_output_json_valid": True}},
    )


def route_v3_decision(**overrides):
    arguments = {
        "active_stage_id": "stage_00",
        "stage_complete_candidate": False,
        "stage_relation": "unknown",
        "stage_evidence_refs": [],
    }
    arguments.update(overrides)
    return route_v2_decision(**arguments)


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


def make_state(step_id=1, instruction="go to kitchen"):
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction=instruction,
        step_id=step_id,
        current_image=None,
    )


def make_pose_state(
    step_id,
    position,
    rotation=(1.0, 0.0, 0.0, 0.0),
    last_action=None,
    online_metrics=None,
    instruction="go to kitchen",
):
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction=instruction,
        step_id=step_id,
        current_image=None,
        online_metrics=online_metrics or {},
        diagnostic_pose={
            "position": list(position),
            "rotation": list(rotation),
        },
        last_action=last_action,
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


def test_qwen_direct_runtime_records_local_odometry_without_changing_action():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={"action_text": "MOVE_FORWARD", "confidence": 0.9},
        reason="continue forward",
        planner_backend="gateway",
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    first = runtime.step(
        make_pose_state(step_id=0, position=(0.0, 0.0, 0.0)),
        payload={},
    )
    second = runtime.step(
        make_pose_state(
            step_id=1,
            position=(0.0, 0.0, 0.25),
            last_action="MOVE_FORWARD",
            online_metrics={"collision": False},
        ),
        payload={},
    )

    assert first.action_text == "MOVE_FORWARD"
    assert second.action_text == "MOVE_FORWARD"
    assert second.runtime_metadata["odometry_available"] is True
    assert second.runtime_metadata["odometry_previous_action"] == "MOVE_FORWARD"
    assert second.runtime_metadata["odometry_last_forward_delta_m"] == 0.25
    assert second.runtime_metadata["odometry_last_action_had_progress"] is True
    assert second.runtime_metadata["odometry_consecutive_no_progress_forward"] == 0
    assert (
        second.runtime_metadata["local_control_context"]["odometry"]["available"]
        is True
    )


def test_qwen_direct_runtime_counts_forward_no_progress_and_collision():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={"action_text": "TURN_LEFT", "confidence": 0.9},
        reason="recover",
        planner_backend="gateway",
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    runtime.step(
        make_pose_state(step_id=0, position=(0.0, 0.0, 0.0)),
        payload={},
    )
    first_blocked = runtime.step(
        make_pose_state(
            step_id=1,
            position=(0.0, 0.0, 0.01),
            last_action="MOVE_FORWARD",
            online_metrics={"collision": False},
        ),
        payload={},
    )
    second_blocked = runtime.step(
        make_pose_state(
            step_id=2,
            position=(0.0, 0.0, 0.015),
            last_action="MOVE_FORWARD",
            online_metrics={"collision": True},
        ),
        payload={},
    )

    assert first_blocked.runtime_metadata["odometry_last_action_had_progress"] is False
    assert (
        first_blocked.runtime_metadata["odometry_consecutive_no_progress_forward"] == 1
    )
    assert (
        second_blocked.runtime_metadata["odometry_consecutive_no_progress_forward"] == 2
    )
    assert second_blocked.runtime_metadata["odometry_collision"] is True
    assert second_blocked.runtime_metadata["odometry_consecutive_collision"] == 1


def test_qwen_direct_runtime_tracks_turn_round_and_negated_waypoint_context():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={
            "action_text": "TURN_LEFT",
            "confidence": 0.9,
            "visual_summary": "The billiard table is not visible; it must be located.",
            "progress_state": "reorienting",
            "stop_evidence": "none",
        },
        reason="continue turning",
        planner_backend="gateway",
    )
    planner = RecordingPlanner(decision)
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )
    instruction = (
        "turn round and walk past the billiard table, walk straight on, "
        "stop by the window."
    )

    runtime.step(
        make_pose_state(
            step_id=0,
            position=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            instruction=instruction,
        ),
        payload={"instruction": instruction},
    )
    result = runtime.step(
        make_pose_state(
            step_id=1,
            position=(0.0, 0.0, 0.0),
            rotation=(0.9238795, 0.0, 0.3826834, 0.0),
            last_action="TURN_LEFT",
            instruction=instruction,
        ),
        payload={"instruction": instruction},
    )

    route_progress = planner.payloads[1]["route_progress"]
    assert route_progress["turn_round_required"] is True
    assert route_progress["heading_change_from_start_deg"] == pytest.approx(45.0)
    assert route_progress["turn_round_completed"] is False
    assert route_progress["positively_seen_waypoints"] == []
    assert route_progress["passed_waypoints"] == []
    assert route_progress["negated_waypoint_mentions"] == ["billiard table"]
    final_route_progress = result.runtime_metadata["route_progress"]
    assert final_route_progress["turn_round_completed"] is False
    assert final_route_progress["negated_waypoint_mentions"] == ["billiard table"]
    assert (
        final_route_progress["waypoint_states"]["billiard table"]["negated_mentions"]
        == 2
    )


def test_qwen_direct_runtime_marks_waypoint_passed_after_two_effective_forwards():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={
            "action_text": "MOVE_FORWARD",
            "confidence": 0.9,
            "visual_summary": "The billiard table is clearly visible on the left.",
            "progress_state": "following route",
            "stop_evidence": "none",
        },
        reason="continue past the table",
        planner_backend="gateway",
    )
    planner = RecordingPlanner(decision)
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )
    instruction = "walk past the billiard table, walk straight on, stop by the window."

    runtime.step(
        make_pose_state(
            step_id=0,
            position=(0.0, 0.0, 0.0),
            instruction=instruction,
        ),
        payload={"instruction": instruction},
    )
    runtime.step(
        make_pose_state(
            step_id=1,
            position=(0.0, 0.0, 0.25),
            last_action="MOVE_FORWARD",
            instruction=instruction,
        ),
        payload={"instruction": instruction},
    )
    result = runtime.step(
        make_pose_state(
            step_id=2,
            position=(0.0, 0.0, 0.5),
            last_action="MOVE_FORWARD",
            instruction=instruction,
        ),
        payload={"instruction": instruction},
    )

    route_progress = planner.payloads[2]["route_progress"]
    assert route_progress["positively_seen_waypoints"] == ["billiard table"]
    assert route_progress["passed_waypoints"] == ["billiard table"]
    assert (
        route_progress["waypoint_states"]["billiard table"][
            "effective_forward_after_seen"
        ]
        == 2
    )
    assert result.runtime_metadata["route_progress"] == route_progress


def test_dynamic_visual_registry_records_previous_executed_action_and_resets():
    planner = RecordingPlanner(route_v2_decision())
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )

    runtime.step(
        make_pose_state(step_id=0, position=(0, 0, 0)),
        payload={"current_image_path": "/tmp/frame0.png"},
    )
    runtime.step(
        make_pose_state(
            step_id=1,
            position=(0, 0, 0),
            rotation=(0.9238795, 0.0, 0.3826834, 0.0),
            last_action="TURN_LEFT",
        ),
        payload={"current_image_path": "/tmp/frame1.png"},
    )

    records = planner.payloads[1]["visual_evidence_registry"]
    assert records[-1]["step_id"] == 1
    assert records[-1]["previous_executed_action"] == "TURN_LEFT"
    assert records[-1]["heading_deg"] == pytest.approx(45.0)

    runtime.reset_episode("scene", "episode-2")
    planner.payloads.clear()
    new_episode_state = make_pose_state(step_id=0, position=(0, 0, 0))
    new_episode_state.episode_id = "episode-2"
    runtime.step(
        new_episode_state,
        payload={"current_image_path": "/tmp/new-frame.png"},
    )

    assert [
        record["image_path"]
        for record in planner.payloads[0]["visual_evidence_registry"]
    ] == ["/tmp/new-frame.png"]


def test_dynamic_visual_registry_assigns_stall_anchor_and_signed_scan_roles():
    planner = RecordingPlanner(route_v2_decision(action_text="TURN_LEFT"))
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )

    states = [
        make_pose_state(step_id=0, position=(0, 0, 0)),
        make_pose_state(
            step_id=1,
            position=(0, 0, 0),
            last_action="MOVE_FORWARD",
        ),
        make_pose_state(
            step_id=2,
            position=(0, 0, 0),
            rotation=(0.9238795, 0.0, 0.3826834, 0.0),
            last_action="TURN_LEFT",
        ),
        make_pose_state(
            step_id=3,
            position=(0, 0, 0),
            rotation=(0.9238795, 0.0, -0.3826834, 0.0),
            last_action="TURN_RIGHT",
        ),
    ]
    for index, state in enumerate(states):
        runtime.step(
            state,
            payload={
                "current_image_path": f"/tmp/frame{index}.png",
                "motion_feedback_enabled": True,
            },
        )

    records = planner.payloads[-1]["visual_evidence_registry"]
    roles_by_path = {record["image_path"]: record["roles"] for record in records}
    assert "stuck_before_keyframe" in roles_by_path["/tmp/frame0.png"]
    assert "center_scan" in roles_by_path["/tmp/frame1.png"]
    assert "left_scan" in roles_by_path["/tmp/frame2.png"]
    assert "right_scan" in roles_by_path["/tmp/frame3.png"]


def test_dynamic_visual_turn_loop_recovery_collects_scans_and_clears_after_forward_progress():
    planner = RecordingPlanner(route_v2_decision(action_text="TURN_LEFT"))
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )
    actions = [
        "TURN_LEFT",
        "TURN_RIGHT",
        "TURN_LEFT",
        "TURN_RIGHT",
        "TURN_LEFT",
        "TURN_RIGHT",
        "TURN_LEFT",
        "TURN_RIGHT",
    ]
    rotations = [
        (1.0, 0.0, 0.0, 0.0),
        (0.9238795, 0.0, 0.3826834, 0.0),
        (0.9238795, 0.0, -0.3826834, 0.0),
        (0.9238795, 0.0, 0.3826834, 0.0),
        (0.9238795, 0.0, -0.3826834, 0.0),
        (0.9238795, 0.0, 0.3826834, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.9238795, 0.0, 0.3826834, 0.0),
        (0.9238795, 0.0, -0.3826834, 0.0),
    ]

    for step_id, rotation in enumerate(rotations):
        runtime.step(
            make_pose_state(
                step_id=step_id,
                position=(0.0, 0.0, 0.0),
                rotation=rotation,
                last_action=actions[step_id - 1] if step_id else None,
            ),
            payload={
                "current_image_path": f"/tmp/turn-loop-{step_id}.png",
                "recent_actions": actions[:step_id],
                "forward_stall_odometry_enabled": True,
            },
        )

    recovery_payload = planner.payloads[-1]
    assert recovery_payload["turn_loop_recovery_active"] is True
    assert recovery_payload["visual_recovery_reason"] == "turn_loop"
    assert recovery_payload["visual_recovery_phase"] == "choose_escape"
    assert recovery_payload["turn_loop_feedback"]["translation_span_m"] == 0.0
    assert recovery_payload["turn_loop_feedback"]["turn_count"] >= 6
    roles_by_path = {
        record["image_path"]: record["roles"]
        for record in recovery_payload["visual_evidence_registry"]
    }
    assert any("stuck_before_keyframe" in roles for roles in roles_by_path.values())
    assert any("left_scan" in roles for roles in roles_by_path.values())
    assert any("right_scan" in roles for roles in roles_by_path.values())

    runtime.step(
        make_pose_state(
            step_id=9,
            position=(0.0, 0.0, 0.25),
            rotation=rotations[-1],
            last_action="MOVE_FORWARD",
        ),
        payload={
            "current_image_path": "/tmp/turn-loop-9.png",
            "recent_actions": actions + ["MOVE_FORWARD"],
            "forward_stall_odometry_enabled": True,
        },
    )

    cleared_payload = planner.payloads[-1]
    assert cleared_payload["turn_loop_recovery_active"] is False
    assert cleared_payload["visual_recovery_active"] is False
    assert "turn_loop_feedback" not in cleared_payload


def test_turn_loop_recovery_does_not_force_missing_directional_scans():
    planner = RecordingPlanner(route_v2_decision(action_text="TURN_LEFT"))
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )
    state = make_pose_state(step_id=8, position=(0.0, 0.0, 0.0))
    episode_state = runtime._qwen_direct_state_for_episode(state)
    episode_state["visual_evidence_registry"] = [
        {
            "step_id": 8,
            "image_path": "/tmp/center.png",
            "heading_deg": 0.0,
            "roles": [],
        }
    ]
    episode_state["visual_recovery"] = {
        "reason": "turn_loop",
        "stuck_before_path": "/tmp/before.png",
        "center_path": "/tmp/center.png",
        "anchor_heading_deg": 0.0,
        "anchor_step_id": 8,
        "turn_count": 7,
        "left_turn_count": 5,
        "right_turn_count": 2,
        "forward_count": 0,
        "translation_span_m": 0.0,
    }
    payload = {"current_image_path": "/tmp/center.png"}

    runtime._attach_visual_evidence_registry(state, payload)

    assert payload["turn_loop_recovery_active"] is True
    assert payload["visual_recovery_phase"] == "choose_escape"
    assert payload["turn_loop_feedback"]["missing_scan_roles"] == []
    assert "committed_turn" not in payload["turn_loop_feedback"]


def test_dynamic_visual_turn_loop_recovery_ignores_required_same_direction_turn_round():
    planner = RecordingPlanner(route_v2_decision(action_text="TURN_LEFT"))
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )
    actions = ["TURN_LEFT"] * 8
    instruction = "Turn around and walk past the table."

    for step_id in range(9):
        runtime.step(
            make_pose_state(
                step_id=step_id,
                position=(0.0, 0.0, 0.0),
                last_action=actions[step_id - 1] if step_id else None,
                instruction=instruction,
            ),
            payload={
                "current_image_path": f"/tmp/turn-round-{step_id}.png",
                "recent_actions": actions[:step_id],
                "forward_stall_odometry_enabled": True,
            },
        )

    assert planner.payloads[-1]["route_progress"]["turn_round_required"] is True
    assert planner.payloads[-1]["turn_loop_recovery_active"] is False
    assert planner.payloads[-1]["visual_recovery_active"] is False


def test_dynamic_visual_turn_loop_recovery_ignores_turns_with_translation():
    planner = RecordingPlanner(route_v2_decision(action_text="TURN_LEFT"))
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )
    actions = [
        "TURN_LEFT",
        "TURN_RIGHT",
        "TURN_LEFT",
        "TURN_RIGHT",
        "TURN_LEFT",
        "TURN_RIGHT",
    ]

    for step_id in range(7):
        runtime.step(
            make_pose_state(
                step_id=step_id,
                position=(0.0, 0.0, step_id * 0.1),
                last_action=actions[step_id - 1] if step_id else None,
            ),
            payload={
                "current_image_path": f"/tmp/moving-turn-{step_id}.png",
                "recent_actions": actions[:step_id],
                "forward_stall_odometry_enabled": True,
            },
        )

    assert planner.payloads[-1]["turn_loop_recovery_active"] is False
    assert planner.payloads[-1]["visual_recovery_active"] is False


def test_dynamic_visual_registry_promotes_only_schema_valid_positive_semantics():
    valid = route_v2_decision()
    invalid = route_v2_decision(
        visual_summary="The billiard table is not visible.",
        confirmed_landmarks=["billiard table"],
    )
    invalid.runtime_metadata = {"context_audit": {"qwen_output_json_valid": False}}
    planner = SequencePlanner([valid, invalid, valid])
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )
    instruction = "walk past the billiard table and stop by the window"

    for step_id in range(3):
        runtime.step(
            make_pose_state(
                step_id=step_id,
                position=(0, 0, step_id * 0.25),
                instruction=instruction,
            ),
            payload={"current_image_path": f"/tmp/semantic{step_id}.png"},
        )

    second_call_records = planner.payloads[1]["visual_evidence_registry"]
    first = next(record for record in second_call_records if record["step_id"] == 0)
    assert "confirmed_landmark" in first["roles"]
    assert first["confirmed_landmarks"] == ["billiard table"]
    assert first["capture_route_stage"] == "intermediate_landmark"
    assert first["capture_current_target"] == "window"
    third_call_records = planner.payloads[2]["visual_evidence_registry"]
    second = next(record for record in third_call_records if record["step_id"] == 1)
    assert "confirmed_landmark" not in second["roles"]


def test_dynamic_visual_registry_is_bounded_to_32_records():
    planner = RecordingPlanner(route_v2_decision(confirmed_landmarks=[]))
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )

    for step_id in range(35):
        runtime.step(
            make_pose_state(step_id=step_id, position=(0, 0, step_id * 0.25)),
            payload={"current_image_path": f"/tmp/cap{step_id}.png"},
        )

    records = planner.payloads[-1]["visual_evidence_registry"]
    assert len(records) == 32
    assert records[-1]["image_path"] == "/tmp/cap34.png"


def test_dynamic_visual_registry_tracks_promoted_keyframe_path_with_semantic_provenance():
    planner = RecordingPlanner(route_v2_decision())
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
    )
    state = make_pose_state(step_id=0, position=(0, 0, 0))
    payload = {"current_image_path": "/tmp/current.png"}

    runtime._record_visual_evidence_frame(state, payload)
    runtime._promote_visual_semantic_evidence(state, payload, route_v2_decision())
    runtime._record_promoted_keyframe_evidence(
        state,
        payload,
        {
            "promotion_status": "promoted",
            "promoted_image_path": "/tmp/promoted.png",
        },
    )
    runtime._attach_visual_evidence_registry(state, payload)

    promoted = next(
        record
        for record in payload["visual_evidence_registry"]
        if record["image_path"] == "/tmp/promoted.png"
    )
    assert "keyframe" in promoted["roles"]
    assert promoted["capture_route_stage"] == "intermediate_landmark"
    assert promoted["capture_current_target"] == "window"
    assert promoted["confirmed_landmarks"] == ["billiard table"]


def test_dynamic_visual_registry_mirrors_into_shared_episode_store():
    planner = RecordingPlanner(route_v2_decision())
    episode_store = EpisodeVisualMemoryStore()
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
        staged_visual_memory_enabled=True,
        episode_visual_store=episode_store,
    )
    state = make_pose_state(step_id=0, position=(0, 0, 0))
    payload = {
        "current_image_path": "/tmp/current-shared-store.png",
        "active_stage_id": "stage_00",
    }

    runtime._record_visual_evidence_frame(state, payload)
    runtime._promote_visual_semantic_evidence(state, payload, route_v2_decision())

    assert episode_store.episode_key == "s1:e1"
    assert len(episode_store.records) == 1
    record = episode_store.records[0]
    assert record.stage_id == "stage_00"
    assert record.visual_summary.startswith("The billiard table")
    assert "billiard table" in record.landmarks
    assert "confirmed_landmark" in record.image_roles

    runtime.reset_episode("s1", "e2")
    assert episode_store.records == ()


def test_signed_heading_delta_handles_wraparound():
    assert OpenClawVLNRuntime._signed_heading_delta(170.0, -170.0) == pytest.approx(
        20.0
    )
    assert OpenClawVLNRuntime._signed_heading_delta(-170.0, 170.0) == pytest.approx(
        -20.0
    )


def test_qwen_direct_runtime_keeps_turn_round_completion_after_further_rotation():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={
            "action_text": "TURN_LEFT",
            "confidence": 0.9,
            "visual_summary": "Continuing to reorient in the room.",
            "progress_state": "reorienting",
            "stop_evidence": "none",
        },
        reason="continue turning",
        planner_backend="gateway",
    )
    planner = RecordingPlanner(decision)
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )
    instruction = "turn round and stop by the window."

    for step_id, rotation in enumerate(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        )
    ):
        runtime.step(
            make_pose_state(
                step_id=step_id,
                position=(0.0, 0.0, 0.0),
                rotation=rotation,
                last_action="TURN_LEFT" if step_id else None,
                instruction=instruction,
            ),
            payload={"instruction": instruction},
        )

    assert (
        planner.payloads[1]["route_progress"]["heading_change_from_start_deg"] == 180.0
    )
    assert planner.payloads[1]["route_progress"]["turn_round_completed"] is True
    assert planner.payloads[2]["route_progress"]["heading_change_from_start_deg"] == 0.0
    assert planner.payloads[2]["route_progress"]["turn_round_completed"] is True


def test_qwen_direct_runtime_sends_motion_feedback_before_planning():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={"action_text": "TURN_LEFT", "confidence": 0.9},
        reason="recover",
        planner_backend="gateway",
    )
    planner = RecordingPlanner(decision)
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    runtime.step(
        make_pose_state(step_id=0, position=(0.0, 0.0, 0.0)),
        payload={"motion_feedback_enabled": True},
    )
    result = runtime.step(
        make_pose_state(
            step_id=1,
            position=(0.0, 0.0, 0.02),
            last_action="MOVE_FORWARD",
            online_metrics={"collision": True},
        ),
        payload={"motion_feedback_enabled": True},
    )

    feedback = planner.payloads[1]["motion_feedback"]
    assert feedback["last_action"] == "MOVE_FORWARD"
    assert feedback["actual_effect"] == "blocked"
    assert feedback["last_forward_delta_m"] == 0.02
    assert feedback["recommended_constraint"] == "avoid_forward"
    assert "sim_position" not in feedback
    assert result.runtime_metadata["motion_feedback"]["actual_effect"] == "blocked"


def test_qwen_direct_odometry_forward_stall_requeries_for_non_forward_action():
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={"action_text": "TURN_LEFT", "confidence": 0.9},
                reason="initial",
                planner_backend="gateway",
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={"action_text": "TURN_LEFT", "confidence": 0.9},
                reason="first blocked forward observed",
                planner_backend="gateway",
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "MOVE_FORWARD",
                    "confidence": 0.9,
                    "progress_state": "path appears ahead",
                    "stop_evidence": "none",
                },
                reason="try forward again",
                planner_backend="gateway",
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "TURN_RIGHT",
                    "confidence": 0.8,
                    "progress_state": "turning after blocked forward",
                    "stop_evidence": "none",
                },
                reason="odometry recovery",
                planner_backend="gateway",
            ),
        ]
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    runtime.step(
        make_pose_state(step_id=0, position=(0.0, 0.0, 0.0)),
        payload={
            "motion_feedback_enabled": True,
            "forward_stall_odometry_enabled": True,
        },
    )
    runtime.step(
        make_pose_state(
            step_id=1,
            position=(0.0, 0.0, 0.01),
            last_action="MOVE_FORWARD",
            online_metrics={"collision": True},
        ),
        payload={
            "motion_feedback_enabled": True,
            "forward_stall_odometry_enabled": True,
        },
    )
    result = runtime.step(
        make_pose_state(
            step_id=2,
            position=(0.0, 0.0, 0.015),
            last_action="MOVE_FORWARD",
            online_metrics={"collision": True},
        ),
        payload={
            "motion_feedback_enabled": True,
            "forward_stall_odometry_enabled": True,
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_RIGHT"
    assert len(planner.payloads) == 4
    feedback = planner.payloads[3]["forward_stall_feedback"]
    assert feedback["blocked_action"] == "MOVE_FORWARD"
    assert feedback["allowed_actions"] == ["TURN_LEFT", "TURN_RIGHT"]
    assert feedback["motion_feedback"]["actual_effect"] == "blocked"
    assert planner.payloads[3]["control_context"]["force_non_forward_action"] is True
    assert result.runtime_metadata["forward_stall_gate_decision"] == "blocked"
    assert result.runtime_metadata["forward_stall_evidence_source"] == "odometry"
    assert result.runtime_metadata["blocked_forward_by_odometry_gate"] is True
    assert result.runtime_metadata["qwen_direct_requery_reason"] == "forward_stall_gate"
    assert result.runtime_metadata["final_action_source"] == "qwen"


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


def test_runtime_seeds_saved_keyframes_into_context_engine_retrieval(tmp_path):
    keyframe = tmp_path / "step_000000.png"
    current = tmp_path / "step_000001.png"
    keyframe.write_bytes(b"keyframe")
    current.write_bytes(b"current")
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "MOVE_FORWARD",
                    "confidence": 0.9,
                    "progress_state": "approaching archway",
                    "stop_evidence": "none",
                },
                reason="archway remains ahead",
                planner_backend="gateway",
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "MOVE_FORWARD",
                    "confidence": 0.9,
                    "progress_state": "approaching archway",
                    "stop_evidence": "none",
                },
                reason="use retrieved archway keyframe",
                planner_backend="gateway",
            ),
        ]
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    first = runtime.step(
        make_state(step_id=0),
        payload={
            "run_id": str(tmp_path),
            "current_image_path": str(keyframe),
            "keyframe_candidate": {
                "step_id": 0,
                "reason": "central archway keyframe",
                "image_path": str(keyframe),
            },
        },
    )
    second = runtime.step(
        make_state(step_id=1),
        payload={
            "run_id": str(tmp_path),
            "current_image_path": str(current),
            "memory_query": "central archway",
        },
    )

    assert first.runtime_metadata["context_engine"]["seeded_keyframe_memory_ids"] == [
        "mem_000001"
    ]
    assert second.runtime_metadata["context_engine"]["memory_context_used"] is True
    assert planner.payloads[1]["retrieved_memory_ids"] == ["mem_000001"]
    assert planner.payloads[1]["retrieved_memory_image_paths"] == [str(keyframe)]
    assert planner.payloads[1]["retrieved_memory_images"] == [
        {
            "memory_id": "mem_000001",
            "image_path": str(keyframe),
            "source": "openclaw_retrieved_memory",
            "step_id": 0,
        }
    ]


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
    assert (
        result.runtime_metadata["tool_calls"][-1]["tool_name"]
        == "NavigationPolicySkill"
    )


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
        assert (
            result.runtime_metadata["tool_calls"][-1]["tool_name"]
            == "NavigationPolicySkill"
        )


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
        arguments={
            "action_text": "MOVE_FORWARD",
            "active_subgoal": "follow the archway",
        },
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


def test_qwen_direct_runtime_uses_context_audit_for_forward_stall_gate():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={
            "action_text": "MOVE_FORWARD",
            "confidence": 0.9,
            "progress_state": "unknown",
            "stop_evidence": "none",
        },
        reason="qwen direct",
        planner_backend="gateway",
        runtime_metadata={
            "context_audit": {
                "policy_backend": "qwen_direct",
                "planner_step_mode": "fast_text",
                "planner_authority": "qwen",
            }
        },
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    result = runtime.step(
        make_state(step_id=12),
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ]
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["forward_stall_gate_decision"] == "blocked"
    assert result.runtime_metadata["final_action_source"] == "forward_stall_gate"


def test_qwen_direct_blocked_stop_requeries_for_non_stop_action():
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "STOP",
                    "confidence": 0.9,
                    "visual_summary": "Archway target reached environment.",
                    "progress_state": "arrival assumed from step count",
                    "stop_evidence": "instruction_complete",
                    "reason": "The archway destination is assumed reached.",
                },
                reason="weak stop",
                planner_backend="gateway",
                runtime_metadata={
                    "context_audit": {
                        "policy_backend": "qwen_direct",
                        "planner_step_mode": "visual_update",
                        "planner_authority": "qwen",
                    }
                },
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "TURN_RIGHT",
                    "confidence": 0.72,
                    "visual_summary": "Archway is not centered in the current view.",
                    "progress_state": "needs_reorientation",
                    "stop_evidence": "none",
                    "reason": "STOP was rejected, so reorient toward the target.",
                },
                reason="blocked stop recovery",
                planner_backend="gateway",
                runtime_metadata={
                    "context_audit": {
                        "policy_backend": "qwen_direct",
                        "planner_step_mode": "visual_update",
                        "planner_authority": "qwen",
                    }
                },
            ),
        ]
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    result = runtime.step(
        make_state(step_id=12),
        payload={"recent_actions": ["MOVE_FORWARD", "MOVE_FORWARD", "TURN_LEFT"]},
    )

    assert result.ok is True
    assert result.action_text == "TURN_RIGHT"
    assert len(planner.payloads) == 2
    feedback = planner.payloads[1]["blocked_stop_feedback"]
    assert feedback["blocked_action"] == "STOP"
    assert feedback["allowed_actions"] == [
        "MOVE_FORWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
    ]
    assert feedback["stop_evidence"] == "instruction_complete"
    assert "MOVE_FORWARD only" in feedback["reason"]
    assert planner.payloads[1]["control_context"]["force_non_stop_action"] is True
    assert result.runtime_metadata["qwen_direct_requery_triggered"] is True
    assert result.runtime_metadata["qwen_direct_requery_reason"] == "blocked_stop_gate"
    assert result.runtime_metadata["qwen_direct_initial_candidate_action"] == "STOP"
    assert result.runtime_metadata["candidate_action"] == "TURN_RIGHT"
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_structural_stop_uses_same_step_non_translating_verification():
    stop_decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={
            "action_text": "STOP",
            "confidence": 0.9,
            "visual_summary": "At the archway threshold, framed by the wooden archway.",
            "progress_state": "Target reached.",
            "stop_evidence": "visible_goal",
            "current_target": "central archway",
            "target_relation": "at_threshold",
            "semantic_stop_state": "at_or_inside_target",
            "reason": "The current image shows the agent at the threshold.",
        },
        reason="threshold stop",
        planner_backend="gateway",
        runtime_metadata={
            "context_audit": {
                "policy_backend": "qwen_direct",
                "planner_step_mode": "visual_update",
                "planner_authority": "qwen",
            }
        },
    )
    planner = SequencePlanner([stop_decision, stop_decision])
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    result = runtime.step(
        make_state(
            step_id=15,
            instruction="Walk across the floor and wait the archway. ",
        ),
        payload={"recent_actions": ["MOVE_FORWARD"] * 11},
    )

    assert result.ok is True
    assert result.action_text == "STOP"
    assert len(planner.payloads) == 2
    feedback = planner.payloads[1]["stop_verification_feedback"]
    assert feedback["allowed_actions"] == ["STOP", "TURN_LEFT", "TURN_RIGHT"]
    assert planner.payloads[1]["control_context"]["force_non_forward_action"] is True
    assert planner.payloads[1]["control_context"]["force_stop_verification"] is True
    assert result.runtime_metadata["qwen_direct_requery_triggered"] is True
    assert result.runtime_metadata["qwen_direct_requery_reason"] == (
        "structural_stop_verification"
    )
    assert result.runtime_metadata["stop_gate_decision"] == "passed"
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_structural_stop_verification_never_executes_forward():
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "STOP",
                    "confidence": 0.9,
                    "visual_summary": "At the archway threshold, framed by the archway.",
                    "progress_state": "Target reached.",
                    "stop_evidence": "visible_goal",
                    "current_target": "archway",
                    "target_relation": "at_threshold",
                    "semantic_stop_state": "at_or_inside_target",
                    "reason": "The agent is at the threshold.",
                },
                reason="threshold stop",
                planner_backend="gateway",
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "MOVE_FORWARD",
                    "confidence": 0.8,
                    "visual_summary": "The archway remains centered.",
                    "progress_state": "approaching_target",
                    "stop_evidence": "none",
                    "reason": "Move forward to confirm.",
                },
                reason="move forward to confirm",
                planner_backend="gateway",
            ),
        ]
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    result = runtime.step(
        make_state(
            step_id=15, instruction="Walk across the floor and wait the archway."
        ),
        payload={"recent_actions": ["MOVE_FORWARD"] * 11},
    )

    assert result.action_text in {"TURN_LEFT", "TURN_RIGHT"}
    assert result.action_text != "MOVE_FORWARD"
    assert result.runtime_metadata["stop_verification_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "MOVE_FORWARD"


def test_qwen_direct_structural_stop_reverifies_without_cross_step_forward_progress():
    stop_decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={
            "action_text": "STOP",
            "confidence": 0.9,
            "visual_summary": "At the archway threshold, framed by the wooden archway.",
            "progress_state": "Target reached.",
            "stop_evidence": "visible_goal",
            "current_target": "central archway",
            "target_relation": "at_threshold",
            "semantic_stop_state": "at_or_inside_target",
            "reason": "The current image shows the agent at the threshold.",
        },
        reason="threshold stop",
        planner_backend="gateway",
        runtime_metadata={
            "context_audit": {
                "policy_backend": "qwen_direct",
                "planner_step_mode": "visual_update",
                "planner_authority": "qwen",
            }
        },
    )
    planner = SequencePlanner([stop_decision] * 4)
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    first = runtime.step(
        make_state(
            step_id=15,
            instruction="Walk across the floor and wait the archway. ",
        ),
        payload={"recent_actions": ["MOVE_FORWARD"] * 11},
    )
    second = runtime.step(
        make_state(
            step_id=24,
            instruction="Walk across the floor and wait the archway. ",
        ),
        payload={"recent_actions": ["MOVE_FORWARD"] * 19},
    )

    assert first.action_text == "STOP"
    assert first.runtime_metadata["qwen_direct_requery_reason"] == (
        "structural_stop_verification"
    )
    assert second.ok is True
    assert second.action_text == "STOP"
    assert second.runtime_metadata["stop_gate_decision"] == "passed"
    assert second.runtime_metadata["qwen_direct_requery_reason"] == (
        "structural_stop_verification"
    )


def test_qwen_direct_structural_stop_ignores_capped_forward_history_for_verification():
    stop_decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={
            "action_text": "STOP",
            "confidence": 0.9,
            "visual_summary": "At the archway threshold, framed by the wooden archway.",
            "progress_state": "Target reached.",
            "stop_evidence": "visible_goal",
            "current_target": "central archway",
            "target_relation": "at_threshold",
            "semantic_stop_state": "at_or_inside_target",
            "reason": "The current image shows the agent at the threshold.",
        },
        reason="threshold stop",
        planner_backend="gateway",
        runtime_metadata={
            "context_audit": {
                "policy_backend": "qwen_direct",
                "planner_step_mode": "visual_update",
                "planner_authority": "qwen",
            }
        },
    )
    planner = SequencePlanner([stop_decision] * 4)
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    capped_recent_actions = ["MOVE_FORWARD"] * 20
    first = runtime.step(
        make_state(
            step_id=59,
            instruction="Walk across the floor and wait the archway. ",
        ),
        payload={"recent_actions": capped_recent_actions},
    )
    second = runtime.step(
        make_state(
            step_id=60,
            instruction="Walk across the floor and wait the archway. ",
        ),
        payload={"recent_actions": capped_recent_actions},
    )

    assert first.action_text == "STOP"
    assert first.runtime_metadata["stop_gate_decision"] == "passed"
    assert second.action_text == "STOP"
    assert second.runtime_metadata["stop_gate_decision"] == "passed"
    assert second.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_multi_waypoint_stop_requery_preserves_block_reason():
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "STOP",
                    "confidence": 0.9,
                    "visual_summary": (
                        "The view shows an opening leading directly to an outdoor area, "
                        "indicating the agent is at the exit threshold."
                    ),
                    "progress_state": "The final destination area has been reached.",
                    "stop_evidence": "instruction_complete",
                    "current_target": "outdoor foyer",
                    "target_relation": "at the outdoor foyer entrance",
                    "semantic_stop_state": "instruction_complete_at_target",
                    "reason": "The exit is directly ahead, so the instruction is complete.",
                },
                reason="early outdoor stop",
                planner_backend="gateway",
                runtime_metadata={
                    "context_audit": {
                        "policy_backend": "qwen_direct",
                        "planner_step_mode": "visual_update",
                        "planner_authority": "qwen",
                    }
                },
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "MOVE_FORWARD",
                    "confidence": 0.72,
                    "visual_summary": "Doorway remains visible.",
                    "progress_state": "Continuing because the waypoint chain is incomplete.",
                    "stop_evidence": "none",
                    "reason": "STOP was rejected, so continue toward the route.",
                },
                reason="blocked stop recovery",
                planner_backend="gateway",
                runtime_metadata={
                    "context_audit": {
                        "policy_backend": "qwen_direct",
                        "planner_step_mode": "visual_update",
                        "planner_authority": "qwen",
                    }
                },
            ),
        ]
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    result = runtime.step(
        make_state(
            step_id=11,
            instruction=(
                "Go through living room, through the door on the to the right, "
                "through the den, through the dining to, to the outdoor foyer. "
                "Stop before going outside. "
            ),
        ),
        payload={"recent_actions": ["MOVE_FORWARD", "TURN_RIGHT", "TURN_RIGHT"]},
    )

    assert result.ok is True
    assert result.action_text == "MOVE_FORWARD"
    assert result.runtime_metadata["qwen_direct_requery_triggered"] is True
    assert result.runtime_metadata["qwen_direct_requery_reason"] == "blocked_stop_gate"
    assert result.runtime_metadata["stop_gate_block_reason"] == (
        "missing_multi_waypoint_progress"
    )
    assert result.runtime_metadata["stop_gate_missing_waypoints"] == [
        "den",
        "dining",
        "outdoor foyer",
    ]
    assert result.runtime_metadata["candidate_action"] == "MOVE_FORWARD"
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_forward_stall_requeries_for_non_forward_action():
    planner = SequencePlanner(
        [
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "MOVE_FORWARD",
                    "confidence": 0.86,
                    "visual_summary": "Still approaching the central archway.",
                    "progress_state": "Destination not yet reached.",
                    "stop_evidence": "none",
                    "reason": "The route remains ahead, so continue forward.",
                },
                reason="forward stall",
                planner_backend="gateway",
                runtime_metadata={
                    "context_audit": {
                        "policy_backend": "qwen_direct",
                        "planner_step_mode": "visual_update",
                        "planner_authority": "qwen",
                    }
                },
            ),
            OpenClawPlanDecision(
                intent="act",
                tool_name="QwenDirectPolicy",
                arguments={
                    "action_text": "TURN_RIGHT",
                    "confidence": 0.74,
                    "visual_summary": "The archway is no longer centered after repeated forward motion.",
                    "progress_state": "reorienting toward target after no progress",
                    "stop_evidence": "none",
                    "reason": "Forward was rejected due to no progress, so reorient right.",
                },
                reason="forward stall recovery",
                planner_backend="gateway",
                runtime_metadata={
                    "context_audit": {
                        "policy_backend": "qwen_direct",
                        "planner_step_mode": "visual_update",
                        "planner_authority": "qwen",
                    }
                },
            ),
        ]
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    result = runtime.step(
        make_state(step_id=12),
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ]
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_RIGHT"
    assert len(planner.payloads) == 2
    feedback = planner.payloads[1]["forward_stall_feedback"]
    assert feedback["blocked_action"] == "MOVE_FORWARD"
    assert feedback["allowed_actions"] == ["TURN_LEFT", "TURN_RIGHT"]
    assert feedback["progress_state"] == "Destination not yet reached."
    assert "fallback_action" not in feedback
    assert planner.payloads[1]["control_context"]["force_non_forward_action"] is True
    assert result.runtime_metadata["qwen_direct_requery_triggered"] is True
    assert result.runtime_metadata["qwen_direct_requery_reason"] == "forward_stall_gate"
    assert (
        result.runtime_metadata["qwen_direct_initial_candidate_action"]
        == "MOVE_FORWARD"
    )
    assert result.runtime_metadata["qwen_direct_initial_fallback_action"] == "TURN_LEFT"
    assert result.runtime_metadata["forward_stall_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "MOVE_FORWARD"
    assert result.runtime_metadata["candidate_action"] == "TURN_RIGHT"
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_runtime_allows_forward_when_context_audit_has_visual_update():
    decision = OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments={
            "action_text": "MOVE_FORWARD",
            "confidence": 0.9,
            "progress_state": "unknown",
            "stop_evidence": "none",
        },
        reason="qwen direct",
        planner_backend="gateway",
        runtime_metadata={
            "context_audit": {
                "policy_backend": "qwen_direct",
                "planner_step_mode": "visual_update",
                "planner_authority": "qwen",
            }
        },
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )

    result = runtime.step(
        make_state(step_id=12),
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ]
        },
    )

    assert result.ok is True
    assert result.action_text == "MOVE_FORWARD"
    assert "forward_stall_gate_decision" not in result.runtime_metadata
    assert result.runtime_metadata["final_action_source"] == "qwen"


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
    assert (
        result.runtime_metadata["tool_calls"][0]["tool_name"]
        == "VisualMemoryCuratorSkill"
    )
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

    runtime.step(
        make_state(step_id=0), payload={"current_image_path": "/tmp/current.png"}
    )
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
    assert (
        first_call["payload_summary"]["write_gate"]["candidate_reason"]
        == "planner_request"
    )


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
    assert (
        result.runtime_metadata["tool_calls"][0]["tool_name"]
        == "VisualMemoryCuratorSkill"
    )
    assert result.runtime_metadata["tool_calls"][1]["tool_name"] == "MemoryWriteSkill"
    assert result.runtime_metadata["memory_writes"][0]["skipped"] is True
    assert (
        result.runtime_metadata["memory_writes"][0]["write_gate"]["curator_decision"]
        == "skip"
    )
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


def _two_stage_manifest():
    return [
        {
            "order": 0,
            "route_clause": "cross the hallway",
            "transition_type": "traverse",
            "expected_landmarks": ["hallway"],
            "completion_cues": ["end of hallway"],
            "final_stage": False,
        },
        {
            "order": 1,
            "route_clause": "enter the kitchen",
            "transition_type": "final_arrival",
            "expected_landmarks": ["kitchen"],
            "completion_cues": ["inside the kitchen"],
            "final_stage": True,
        },
    ]


def _staged_runtime(planner, **overrides):
    overrides.setdefault("staged_memory_treatment", "off_ablation")
    return OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        staged_visual_memory_enabled=True,
        **overrides,
    )


def test_staged_runtime_segments_once_and_exposes_controller_stage_context():
    planner = StagedRecordingPlanner(route_v3_decision(), stages=_two_stage_manifest())
    runtime = _staged_runtime(planner)

    runtime.step(make_state(step_id=0), {"current_image_path": "/tmp/0.png"})
    runtime.step(make_state(step_id=1), {"current_image_path": "/tmp/1.png"})

    assert len(planner.segment_calls) == 1
    first = planner.payloads[0]
    assert first["active_stage_id"] == "stage_00"
    assert first["stage_state"]["original_instruction"] == "go to kitchen"
    assert first["stage_state"]["completed_stage_ids"] == []
    assert [stage["stage_id"] for stage in first["stage_state"]["pending_stages"]] == [
        "stage_01"
    ]
    assert first["trigger_reasons"] == ["stage_entry"]


def test_staged_runtime_installs_single_stage_fallback_on_segmentation_error():
    class SegmentationFailurePlanner(StagedRecordingPlanner):
        def segment_instruction(self, scene_id, episode_id, instruction):
            self.segment_calls.append((scene_id, episode_id, instruction))
            raise OpenClawGatewayError("provider secret detail")

    planner = SegmentationFailurePlanner(route_v3_decision())
    runtime = _staged_runtime(planner)

    result = runtime.step(make_state(step_id=0), {})

    assert result.ok is True
    assert len(planner.segment_calls) == 1
    assert planner.payloads[0]["stage_state"]["fallback_category"] == "transport_error"
    assert planner.payloads[0]["stage_state"]["segmentation_source"] == "fallback"
    assert "provider secret detail" not in str(planner.payloads[0])


def test_staged_runtime_does_not_resegment_after_planner_error():
    planner = StagedRecordingPlanner(
        route_v3_decision(),
        stages=_two_stage_manifest(),
        plan_errors=[OpenClawGatewayError("plan failed"), None],
    )
    runtime = _staged_runtime(planner)

    first = runtime.step(make_state(step_id=0), {})
    second = runtime.step(make_state(step_id=1), {})

    assert first.ok is False
    assert second.ok is True
    assert len(planner.segment_calls) == 1


def test_staged_runtime_rejects_unattached_evidence_without_advancing():
    planner = StagedRecordingPlanner(
        route_v3_decision(
            stage_complete_candidate=True,
            stage_relation="past",
            stage_evidence_refs=["unknown"],
            visual_summary="end of hallway",
        ),
        stages=_two_stage_manifest(),
    )
    runtime = _staged_runtime(planner)

    result = runtime.step(
        make_pose_state(0, [0.0, 0.0, 0.0]),
        {"current_image_path": "/tmp/0.png"},
    )

    assert runtime.staged_episode_state["s1::e1"]["stage_state"].active_stage_index == 0
    assert result.runtime_metadata["stage_transition"]["decision"] == "rejected"
    assert result.runtime_metadata["stage_transition"]["rule_ids"] == [
        "evidence_ref_not_attached"
    ]


def test_staged_runtime_controller_advances_once_and_emits_next_entry_edge():
    planner = StagedRecordingPlanner(
        route_v3_decision(
            stage_complete_candidate=True,
            stage_relation="past",
            stage_evidence_refs=["current"],
            visual_summary="We reached the end of hallway.",
        ),
        stages=_two_stage_manifest(),
    )
    runtime = _staged_runtime(planner, stage_min_translation_m=0.25)

    first = runtime.step(
        make_pose_state(0, [0.0, 0.0, 0.0]),
        {"current_image_path": "/tmp/0.png"},
    )
    second = runtime.step(
        make_pose_state(1, [0.5, 0.0, 0.0], last_action="MOVE_FORWARD"),
        {"current_image_path": "/tmp/1.png"},
    )
    third = runtime.step(
        make_pose_state(2, [0.6, 0.0, 0.0], last_action="MOVE_FORWARD"),
        {"current_image_path": "/tmp/2.png"},
    )

    assert (
        first.runtime_metadata["stage_transition"]["decision"] == "needs_verification"
    )
    assert second.runtime_metadata["stage_transition"]["decision"] == "accepted"
    assert planner.payloads[4]["active_stage_id"] == "stage_01"
    assert "stage_entry" in planner.payloads[4]["trigger_reasons"]
    stage_state = runtime.staged_episode_state["s1::e1"]["stage_state"]
    assert stage_state.active_stage_index == 1
    assert stage_state.completed_stage_ids == ["stage_00"]
    assert len(stage_state.transition_audit) == 1
    assert third.ok is True


def test_staged_runtime_reset_clears_controller_state_and_resegments():
    planner = StagedRecordingPlanner(route_v3_decision(), stages=_two_stage_manifest())
    runtime = _staged_runtime(planner)
    runtime.step(make_state(step_id=0), {})

    runtime.reset_episode("s1", "e2")
    other = VLNState("s1", "e2", "go upstairs", 0, None)
    runtime.step(other, {})

    assert len(planner.segment_calls) == 2
    assert set(runtime.staged_episode_state) == {"s1::e2"}


def test_disabled_staged_runtime_preserves_legacy_payload_and_never_segments():
    planner = StagedRecordingPlanner(route_v2_decision())
    runtime = OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        staged_visual_memory_enabled=False,
    )

    runtime.step(make_state(step_id=0), {})

    assert planner.segment_calls == []
    assert "active_stage_id" not in planner.payloads[0]
    assert "stage_state" not in planner.payloads[0]


def test_stage_entry_forces_both_memory_layers_before_primary_qwen_call():
    memory_skill = ImageBackedMemorySkill()
    registry = SkillRegistry()
    registry.register(memory_skill)
    store = EpisodeVisualMemoryStore(capacity=8)
    store.start_episode("s1", "e1")
    store.add_observation(
        image_path="/tmp/registry-history.png",
        step_id=0,
        stage_id="stage_00",
        visual_summary="end of hallway",
        image_roles=["keyframe"],
    )
    planner = StagedRecordingPlanner(route_v3_decision(), stages=_two_stage_manifest())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        staged_visual_memory_enabled=True,
        episode_visual_store=store,
    )

    result = runtime.step(
        make_state(step_id=1), {"current_image_path": "/tmp/current.png"}
    )

    assert len(memory_skill.calls) == 1
    assert memory_skill.calls[0]["active_stage_id"] == "stage_00"
    assert "cross the hallway" in memory_skill.calls[0]["text"]
    assert planner.payloads[0]["retrieved_memory_image_paths"] == [
        "/tmp/registry-history.png",
        "/tmp/history.png",
    ]
    event = result.runtime_metadata["staged_memory_event"]
    assert event["registry_status"] == "hit"
    assert event["query_status"] == "hit"
    assert event["trigger_reasons"] == ["stage_entry"]


def test_completion_and_stop_candidates_coalesce_and_requery_only_once():
    registry = SkillRegistry()
    memory_skill = ImageBackedMemorySkill()
    registry.register(memory_skill)
    planner = StagedRecordingPlanner(
        route_v3_decision(
            action_text="STOP",
            stage_complete_candidate=True,
            stage_relation="past",
            stage_evidence_refs=["current"],
            visual_summary="end of hallway",
            semantic_stop_state="not_ready",
        ),
        stages=_two_stage_manifest(),
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        staged_visual_memory_enabled=True,
    )

    result = runtime.step(
        make_pose_state(0, [0.0, 0.0, 0.0]),
        {"current_image_path": "/tmp/current.png"},
    )

    assert len(planner.payloads) == 2
    event = result.runtime_metadata["staged_memory_event"]
    assert event["trigger_reasons"] == [
        "stop_candidate",
        "stage_completion_candidate",
        "stage_entry",
    ]
    assert event["requery_performed"] is True
    assert result.action_text != "STOP"


def test_memory_requery_candidate_controls_gates_but_not_controller_stage_state():
    registry = SkillRegistry()
    registry.register(ImageBackedMemorySkill())
    planner = StagedSequencePlanner(
        [
            route_v3_decision(
                action_text="STOP",
                stage_complete_candidate=True,
                stage_relation="past",
                stage_evidence_refs=["current"],
                visual_summary="end of hallway",
                semantic_stop_state="not_ready",
            ),
            route_v3_decision(
                action_text="TURN_RIGHT",
                stage_complete_candidate=False,
                stage_relation="unknown",
                stage_evidence_refs=[],
                visual_summary="memory suggests checking right",
            ),
        ],
        stages=_two_stage_manifest(),
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        staged_visual_memory_enabled=True,
    )

    result = runtime.step(
        make_pose_state(0, [0.0, 0.0, 0.0]),
        {"current_image_path": "/tmp/current.png"},
    )

    assert result.action_text == "TURN_RIGHT"
    assert result.runtime_metadata["stage_transition"]["decision"] == "rejected"
    assert result.runtime_metadata["stage_transition"]["rule_ids"] == [
        "completion_candidate_required"
    ]
    assert runtime.staged_episode_state["s1::e1"]["stage_state"].active_stage_index == 0


def test_failed_forced_recall_cannot_advance_stage_even_when_candidate_matches():
    registry = SkillRegistry()
    memory_skill = ImageBackedMemorySkill(fail=True)
    registry.register(memory_skill)
    planner = StagedRecordingPlanner(
        route_v3_decision(
            stage_complete_candidate=True,
            stage_relation="past",
            stage_evidence_refs=["current"],
            visual_summary="end of hallway",
        ),
        stages=_two_stage_manifest(),
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        staged_visual_memory_enabled=True,
    )
    runtime.step(
        make_pose_state(0, [0.0, 0.0, 0.0]),
        {"current_image_path": "/tmp/current.png"},
    )
    result = runtime.step(
        make_pose_state(1, [0.5, 0.0, 0.0], last_action="MOVE_FORWARD"),
        {"current_image_path": "/tmp/current-1.png"},
    )

    assert (
        result.runtime_metadata["stage_transition"]["decision"] == "needs_verification"
    )
    assert result.runtime_metadata["stage_transition"]["rule_ids"] == [
        "forced_recall_unavailable"
    ]
    assert runtime.staged_episode_state["s1::e1"]["stage_state"].active_stage_index == 0


def test_off_ablation_emits_same_event_without_attaching_history():
    memory_skill = ImageBackedMemorySkill()
    registry = SkillRegistry()
    registry.register(memory_skill)
    planner = StagedRecordingPlanner(route_v3_decision(), stages=_two_stage_manifest())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=planner,
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
        staged_visual_memory_enabled=True,
        staged_memory_treatment="off_ablation",
    )

    result = runtime.step(make_state(step_id=0), {})

    assert memory_skill.calls == []
    assert "retrieved_memory_image_paths" not in planner.payloads[0]
    assert result.runtime_metadata["staged_memory_event"]["registry_status"] == (
        "disabled_ablation"
    )
    assert result.runtime_metadata["staged_memory_event"]["query_status"] == (
        "disabled_ablation"
    )
