from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.planner import OpenClawPlanDecision
from harness.openclaw.runtime import OpenClawVLNRuntime
from harness.skill_registry import SkillRegistry
from harness.types import VLNState


class StaticPlanner:
    def __init__(self, decision):
        self.decision = decision

    def plan(self, state, runtime_context):
        return self.decision


def make_state(step_id=1, instruction="go to kitchen"):
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction=instruction,
        step_id=step_id,
        current_image=None,
    )


def make_runtime(decision):
    return OpenClawVLNRuntime(
        tool_registry=SkillRegistry(),
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        policy_backend="qwen_direct",
    )


def direct_decision(arguments):
    return OpenClawPlanDecision(
        intent="act",
        tool_name="QwenDirectPolicy",
        arguments=arguments,
        reason="qwen direct",
        planner_backend="gateway",
        runtime_metadata={
            "context_audit": {
                "policy_backend": "qwen_direct",
                "planner_authority": "qwen",
            }
        },
    )


def test_qwen_direct_runtime_executes_qwen_action_without_navigation_skill():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "MOVE_FORWARD",
                "confidence": 0.9,
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(make_state(), payload={})

    assert result.ok is True
    assert result.action_text == "MOVE_FORWARD"
    assert result.executor_command["action_index"] == 1
    assert result.runtime_metadata["candidate_action"] == "MOVE_FORWARD"
    assert result.runtime_metadata["final_action"] == "MOVE_FORWARD"
    assert result.runtime_metadata["final_action_source"] == "qwen"
    assert result.runtime_metadata["navigation_policy_skill_called"] is False
    assert result.runtime_metadata["tool_calls"] == []


def test_qwen_direct_runtime_records_qwen_schema_fields():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "MOVE_FORWARD",
                "confidence": 0.66,
                "visual_summary": "archway visible on the left",
                "progress_state": "needs_reorientation",
                "stop_evidence": "none",
                "current_target": "archway",
                "target_relation": "archway left of center",
                "semantic_stop_state": "not_ready",
                "reason": "route is not centered yet",
            }
        )
    )

    result = runtime.step(make_state(), payload={})

    assert result.runtime_metadata["qwen_confidence"] == 0.66
    assert result.runtime_metadata["qwen_progress_state"] == "needs_reorientation"
    assert result.runtime_metadata["qwen_stop_evidence"] == "none"
    assert result.runtime_metadata["qwen_current_target"] == "archway"
    assert result.runtime_metadata["qwen_target_relation"] == "archway left of center"
    assert result.runtime_metadata["qwen_semantic_stop_state"] == "not_ready"
    assert result.runtime_metadata["qwen_visual_summary"] == "archway visible on the left"
    assert result.runtime_metadata["qwen_visual_summary_present"] is True
    assert result.runtime_metadata["qwen_reason"] == "route is not centered yet"


def test_qwen_direct_stop_without_evidence_is_blocked_before_executor():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.8,
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(make_state(), payload={"recent_actions": []})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.executor_command["action_index"] == 2
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "STOP"
    assert result.runtime_metadata["replacement_action"] == "TURN_LEFT"
    assert result.runtime_metadata["fallback_policy"] == "blocked_stop_turn"


def test_qwen_direct_stop_with_instruction_complete_needs_current_arrival_evidence():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.85,
                "visual_summary": "Floor crossed, archway approach complete",
                "progress_state": "Waiting at destination",
                "stop_evidence": "instruction_complete",
                "reason": "Three steps traversed floor; archway target reached",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": ["MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"],
            "planner_step_mode": "visual_update",
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "STOP"
    assert result.runtime_metadata["replacement_action"] == "TURN_LEFT"
    assert result.runtime_metadata["fallback_policy"] == "blocked_stop_turn"


def test_qwen_direct_stop_with_visible_goal_claim_needs_visual_evidence():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.85,
                "visual_summary": "Floor walk completed, archway vicinity assumed.",
                "progress_state": "Transitioning to wait phase.",
                "stop_evidence": "visible_goal",
                "reason": "Halting to satisfy wait instruction phase.",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": ["MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"],
            "planner_step_mode": "visual_update",
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "STOP"
    assert result.runtime_metadata["replacement_action"] == "TURN_LEFT"
    assert result.runtime_metadata["fallback_policy"] == "blocked_stop_turn"


def test_qwen_direct_stop_with_assumed_arrival_reason_is_blocked():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.9,
                "visual_summary": "Reached target floor zone for archway wait instruction.",
                "progress_state": "Walking segment complete; entering stationary wait state.",
                "stop_evidence": "instruction_complete",
                "reason": (
                    "Completed three consecutive forward moves per plan; arrival at "
                    "archway destination assumed based on task structure and movement limits."
                ),
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": ["MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"],
            "planner_step_mode": "visual_update",
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "STOP"
    assert result.runtime_metadata["replacement_action"] == "TURN_LEFT"
    assert result.runtime_metadata["fallback_policy"] == "blocked_stop_turn"


def test_qwen_direct_stop_with_generic_reached_environment_is_blocked():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "Archway target reached environment; orientation aligned.",
                "progress_state": "Travel phase complete, initiating wait protocol.",
                "stop_evidence": "instruction_complete",
                "reason": (
                    "Context confirms archway reached; instruction requires waiting "
                    "rather than further traversal. Navigation objective satisfied."
                ),
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": ["MOVE_FORWARD", "MOVE_FORWARD", "TURN_LEFT"],
            "planner_step_mode": "visual_update",
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "STOP"
    assert result.runtime_metadata["replacement_action"] == "TURN_LEFT"
    assert result.runtime_metadata["fallback_policy"] == "blocked_stop_turn"


def test_qwen_direct_stop_with_at_archway_visual_summary_passes():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "At archway, aligned.",
                "progress_state": "Navigation complete, entering wait state.",
                "stop_evidence": "visible_goal",
                "current_target": "archway",
                "target_relation": "at the archway",
                "semantic_stop_state": "at_or_inside_target",
                "reason": "Current view places the agent at the archway.",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={"structural_stop_confirmation_forward_count": 1},
    )

    assert result.ok is True
    assert result.action_text == "STOP"
    assert result.runtime_metadata["stop_gate_decision"] == "passed"
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_structural_stop_passes_when_forward_would_overshoot():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "At archway, aligned with the entrance threshold.",
                "progress_state": "Waiting at destination.",
                "stop_evidence": "visible_goal",
                "current_target": "archway",
                "target_relation": "at the archway",
                "semantic_stop_state": "at_or_inside_target",
                "reason": (
                    "Moving forward would place the agent inside the room rather "
                    "than waiting in the entrance."
                ),
            }
        )
    )

    result = runtime.step(
        make_state(instruction="Walk across the floor and wait the archway. "),
        payload={},
    )

    assert result.ok is True
    assert result.action_text == "STOP"
    assert result.runtime_metadata["stop_gate_decision"] == "passed"
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_stop_with_goal_evidence_passes():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.9,
                "visual_summary": "goal object is visible ahead",
                "stop_evidence": "visible_goal",
                "current_target": "goal object",
                "target_relation": "beside the goal object",
                "semantic_stop_state": "beside_target",
            }
        )
    )

    result = runtime.step(make_state(), payload={})

    assert result.ok is True
    assert result.action_text == "STOP"
    assert result.runtime_metadata["stop_gate_decision"] == "passed"
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_stop_accepts_underscore_threshold_after_same_step_verification():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "At the archway threshold, framed by the wooden archway.",
                "progress_state": "Navigation complete, entering wait state.",
                "stop_evidence": "visible_goal",
                "current_target": "archway",
                "target_relation": "at_threshold",
                "semantic_stop_state": "at_or_inside_target",
                "reason": "Current view places the agent at the archway threshold.",
            }
        )
    )

    result = runtime.step(
        make_state(instruction="Walk across the floor and wait the archway. "),
        payload={},
    )

    assert result.ok is True
    assert result.action_text == "STOP"
    assert result.runtime_metadata["stop_gate_decision"] == "passed"
    assert result.runtime_metadata["qwen_direct_requery_reason"] == (
        "structural_stop_verification"
    )


def test_qwen_direct_stop_accepts_underscore_threshold_after_confirmation_forward():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "At the archway threshold, framed by the wooden archway.",
                "progress_state": "Navigation complete, entering wait state.",
                "stop_evidence": "visible_goal",
                "current_target": "archway",
                "target_relation": "at_threshold",
                "semantic_stop_state": "at_or_inside_target",
                "reason": "Current view places the agent at the archway threshold.",
            }
        )
    )

    result = runtime.step(
        make_state(instruction="Walk across the floor and wait the archway. "),
        payload={"structural_stop_confirmation_forward_count": 1},
    )

    assert result.ok is True
    assert result.action_text == "STOP"
    assert result.runtime_metadata["stop_gate_decision"] == "passed"


def test_qwen_direct_stop_accepts_framed_by_after_same_step_verification():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "The agent is framed by the large wooden archway.",
                "progress_state": "At the target archway.",
                "stop_evidence": "visible_goal",
                "current_target": "archway",
                "target_relation": "framed_by",
                "semantic_stop_state": "at_or_inside_target",
                "reason": "The current image shows the agent framed by the archway.",
            }
        )
    )

    result = runtime.step(
        make_state(instruction="Walk across the floor and wait the archway. "),
        payload={},
    )

    assert result.ok is True
    assert result.action_text == "STOP"
    assert result.runtime_metadata["stop_gate_decision"] == "passed"
    assert result.runtime_metadata["qwen_direct_requery_reason"] == (
        "structural_stop_verification"
    )


def test_qwen_direct_route_stop_requires_non_oracle_progress_before_window():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": (
                    "The agent is positioned directly beside a large window "
                    "with a wooden frame, looking out onto a garden."
                ),
                "progress_state": "Arrived at the final destination.",
                "stop_evidence": "instruction_complete",
                "current_target": "window",
                "target_relation": "agent is beside the window",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": (
                    "The agent is adjacent to the window, satisfying the final "
                    "instruction 'stop by the window'."
                ),
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=1,
            instruction=(
                "turn round and walk past the billiard table, walk straight on, "
                "stop by the window. "
            ),
        ),
        payload={"recent_actions": ["TURN_LEFT"]},
    )

    assert result.ok is True
    assert result.action_text in {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"}
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_block_reason"] == "missing_route_progress"
    assert result.runtime_metadata["stop_gate_min_forward_actions"] == 4
    assert "distance_to_goal" not in result.runtime_metadata


def test_qwen_direct_route_stop_still_blocks_after_three_forward_actions():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": (
                    "The agent is positioned directly beside a large window "
                    "with a wooden frame."
                ),
                "progress_state": (
                    "Turned around, walked past the billiard table, and walked "
                    "straight to the final window."
                ),
                "stop_evidence": "instruction_complete",
                "current_target": "window",
                "target_relation": "agent is beside the window",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": "The route past the billiard table is complete.",
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=7,
            instruction=(
                "turn round and walk past the billiard table, walk straight on, "
                "stop by the window. "
            ),
        ),
        payload={
            "recent_actions": [
                "TURN_LEFT",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ]
        },
    )

    assert result.ok is True
    assert result.action_text in {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"}
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_block_reason"] == "missing_route_progress"
    assert result.runtime_metadata["stop_gate_forward_action_count"] == 3
    assert result.runtime_metadata["stop_gate_min_forward_actions"] == 4


def test_qwen_direct_route_stop_does_not_treat_visible_landmark_as_passed():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": (
                    "The agent is positioned directly beside a large window "
                    "with a wooden frame."
                ),
                "progress_state": (
                    "Turned around, walked past the billiard table, and walked "
                    "straight to the final window."
                ),
                "stop_evidence": "instruction_complete",
                "current_target": "window",
                "target_relation": "agent is beside the window",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": "The route past the billiard table is complete.",
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=7,
            instruction=(
                "turn round and walk past the billiard table, walk straight on, "
                "stop by the window. "
            ),
        ),
        payload={
            "recent_actions": [
                "TURN_LEFT",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ],
            "observed_visual_summaries": [
                "Doorway into the billiard table room is visible.",
            ],
            "route_progress": {
                "turn_round_required": True,
                "turn_round_completed": True,
                "heading_change_from_start_deg": 180.0,
                "required_waypoints": ["billiard table"],
                "positively_seen_waypoints": ["billiard table"],
                "passed_waypoints": [],
                "negated_waypoint_mentions": [],
            },
        },
    )

    assert result.ok is True
    assert result.action_text in {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"}
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_route_missing_reasons"] == [
        "intermediate_waypoint_passed"
    ]
    assert result.runtime_metadata["stop_gate_positively_seen_waypoints"] == [
        "billiard table"
    ]


def test_qwen_direct_route_stop_passes_after_turn_round_and_waypoint_passed():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "The agent is beside the final wooden-framed window.",
                "progress_state": "Route complete.",
                "stop_evidence": "instruction_complete",
                "current_target": "window",
                "target_relation": "beside the window",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": "The intermediate route is complete.",
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=12,
            instruction=(
                "turn round and walk past the billiard table, walk straight on, "
                "stop by the window."
            ),
        ),
        payload={
            "recent_actions": ["MOVE_FORWARD"] * 4,
            "route_progress": {
                "turn_round_required": True,
                "turn_round_completed": True,
                "heading_change_from_start_deg": 180.0,
                "required_waypoints": ["billiard table"],
                "positively_seen_waypoints": ["billiard table"],
                "passed_waypoints": ["billiard table"],
                "negated_waypoint_mentions": [],
            },
        },
    )

    assert result.action_text == "STOP"
    assert result.runtime_metadata["stop_gate_decision"] == "passed"
    assert result.runtime_metadata["stop_permission"] is True


def test_qwen_direct_route_stop_rejects_negated_waypoint_mentions():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "The agent is beside the final wooden-framed window.",
                "progress_state": "Route complete.",
                "stop_evidence": "instruction_complete",
                "current_target": "window",
                "target_relation": "beside the window",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": "The intermediate route is complete.",
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=12,
            instruction=(
                "turn round and walk past the billiard table, walk straight on, "
                "stop by the window."
            ),
        ),
        payload={
            "recent_actions": ["MOVE_FORWARD"] * 4,
            "observed_visual_summaries": [
                "The billiard table is not visible; the agent needs to locate it.",
            ],
            "route_progress": {
                "turn_round_required": True,
                "turn_round_completed": True,
                "heading_change_from_start_deg": 180.0,
                "required_waypoints": ["billiard table"],
                "positively_seen_waypoints": [],
                "passed_waypoints": [],
                "negated_waypoint_mentions": ["billiard table"],
            },
        },
    )

    assert result.action_text != "STOP"
    assert result.runtime_metadata["stop_gate_missing_route_waypoints"] == [
        "billiard table"
    ]
    assert result.runtime_metadata["stop_gate_negated_waypoint_mentions"] == [
        "billiard table"
    ]


def test_qwen_direct_route_stop_requires_turn_round_completion():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": "The agent is beside the final wooden-framed window.",
                "progress_state": "Route complete.",
                "stop_evidence": "instruction_complete",
                "current_target": "window",
                "target_relation": "beside the window",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": "The intermediate route is complete.",
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=12,
            instruction=(
                "turn round and walk past the billiard table, walk straight on, "
                "stop by the window."
            ),
        ),
        payload={
            "recent_actions": ["MOVE_FORWARD"] * 4,
            "route_progress": {
                "turn_round_required": True,
                "turn_round_completed": False,
                "heading_change_from_start_deg": 45.0,
                "required_waypoints": ["billiard table"],
                "positively_seen_waypoints": ["billiard table"],
                "passed_waypoints": ["billiard table"],
                "negated_waypoint_mentions": [],
            },
        },
    )

    assert result.action_text != "STOP"
    assert "turn_round_progress" in result.runtime_metadata[
        "stop_gate_route_missing_reasons"
    ]
    assert result.runtime_metadata["stop_gate_heading_change_from_start_deg"] == 45.0
    assert result.runtime_metadata["stop_gate_turn_round_completed"] is False
    assert result.runtime_metadata["stop_permission"] is False


def test_qwen_direct_route_stop_ignores_unobserved_waypoint_claims():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.95,
                "visual_summary": (
                    "The agent is positioned directly beside a large window "
                    "with a wooden frame."
                ),
                "progress_state": (
                    "Turned around, walked past the billiard table, and walked "
                    "straight to the final window."
                ),
                "stop_evidence": "instruction_complete",
                "current_target": "window",
                "target_relation": "agent is beside the window",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": "The route past the billiard table is complete.",
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=7,
            instruction=(
                "turn round and walk past the billiard table, walk straight on, "
                "stop by the window. "
            ),
        ),
        payload={
            "recent_actions": [
                "TURN_LEFT",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ]
        },
    )

    assert result.ok is True
    assert result.action_text in {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"}
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_block_reason"] == "missing_route_progress"
    assert result.runtime_metadata["stop_gate_missing_route_waypoints"] == [
        "billiard table"
    ]


def test_qwen_direct_stop_beside_archway_relation_is_too_loose():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.9,
                "visual_summary": (
                    "The agent is positioned beside the large wooden archway, "
                    "facing the interior hallway."
                ),
                "progress_state": "Arrived at target.",
                "stop_evidence": "visible_goal",
                "current_target": "archway",
                "target_relation": "beside_target",
                "semantic_stop_state": "beside_target",
                "reason": (
                    "The agent has successfully navigated to the archway and "
                    "reoriented beside it as planned."
                ),
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=25,
            instruction="Walk across the floor and wait the archway. ",
        ),
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "TURN_RIGHT",
                "TURN_RIGHT",
                "MOVE_FORWARD",
            ]
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_block_reason"] == (
        "structural_target_relation_too_loose"
    )


def test_qwen_direct_stop_missing_semantic_arrival_state_is_blocked():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.9,
                "visual_summary": "At archway, aligned.",
                "progress_state": "Navigation complete, entering wait state.",
                "stop_evidence": "visible_goal",
                "reason": "Current view places the agent at the archway.",
            }
        )
    )

    result = runtime.step(make_state(), payload={})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_block_reason"] == (
        "missing_semantic_stop_state"
    )


def test_qwen_direct_stop_with_visible_ahead_relation_is_blocked():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.9,
                "visual_summary": "The archway is visible directly ahead.",
                "progress_state": "Approaching the target.",
                "stop_evidence": "visible_goal",
                "current_target": "archway",
                "target_relation": "archway visible directly ahead",
                "semantic_stop_state": "visible_not_reached",
                "reason": "The target is visible but not yet reached.",
            }
        )
    )

    result = runtime.step(make_state(), payload={})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_block_reason"] == (
        "semantic_stop_state_not_arrived"
    )


def test_qwen_direct_stop_with_edge_or_vicinity_relation_is_blocked():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.9,
                "visual_summary": "The archway is visible on the left edge.",
                "progress_state": "Near the archway vicinity.",
                "stop_evidence": "instruction_complete",
                "current_target": "archway",
                "target_relation": "archway left edge vicinity",
                "semantic_stop_state": "at_or_inside_target",
                "reason": "The archway is close enough to wait.",
            }
        )
    )

    result = runtime.step(make_state(), payload={})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_block_reason"] == (
        "semantic_target_relation_not_arrived"
    )


def test_qwen_direct_stop_for_multi_waypoint_route_requires_recent_waypoint_progress():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.9,
                "visual_summary": (
                    "The view shows an opening leading directly to an outdoor area "
                    "with greenery, indicating the agent is at the exit threshold."
                ),
                "progress_state": (
                    "The agent has successfully navigated to the final destination area."
                ),
                "stop_evidence": "instruction_complete",
                "current_target": "outdoor foyer",
                "target_relation": "at the outdoor foyer entrance",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": (
                    "The instruction explicitly requires the agent to stop before "
                    "going outside. The current visual evidence shows the exit to "
                    "the outdoors is directly ahead, so moving forward would violate "
                    "the constraint."
                ),
            }
        )
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
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "TURN_RIGHT",
                "TURN_RIGHT",
            ],
            "planner_step_mode": "visual_update",
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["stop_gate_decision"] == "blocked"
    assert result.runtime_metadata["stop_gate_block_reason"] == (
        "missing_multi_waypoint_progress"
    )
    assert result.runtime_metadata["stop_gate_missing_waypoints"] == [
        "den",
        "dining",
        "outdoor foyer",
    ]
    assert result.runtime_metadata["final_action_source"] == "blocked_stop_gate"


def test_qwen_direct_stop_for_multi_waypoint_route_passes_with_waypoint_progress():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "confidence": 0.9,
                "visual_summary": (
                    "At the outdoor foyer entrance with the outside visible directly ahead."
                ),
                "progress_state": (
                    "Passed through the den and dining area; now at the outdoor foyer."
                ),
                "stop_evidence": "instruction_complete",
                "current_target": "outdoor foyer",
                "target_relation": "at the outdoor foyer entrance",
                "semantic_stop_state": "instruction_complete_at_target",
                "reason": (
                    "The route through the den and dining area is complete, and the "
                    "agent is at the outdoor foyer before going outside."
                ),
            }
        )
    )

    result = runtime.step(
        make_state(
            step_id=28,
            instruction=(
                "Go through living room, through the door on the to the right, "
                "through the den, through the dining to, to the outdoor foyer. "
                "Stop before going outside. "
            ),
        ),
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "TURN_RIGHT",
            ],
            "planner_step_mode": "visual_update",
            "observed_visual_summaries": [
                "Passed through the den.",
                "Dining area is behind the agent.",
                "At the outdoor foyer entrance.",
            ],
        },
    )

    assert result.ok is True
    assert result.action_text == "STOP"
    assert result.runtime_metadata["stop_gate_decision"] == "passed"
    assert "stop_gate_block_reason" not in result.runtime_metadata
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_hard_failure_stops_without_other_gates():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "STOP",
                "qwen_failure": True,
                "qwen_failure_reason": "invalid json",
                "fallback_policy": "hard_failure_stop",
            }
        )
    )

    result = runtime.step(make_state(), payload={})

    assert result.ok is True
    assert result.action_text == "STOP"
    assert result.runtime_metadata["qwen_failure"] is True
    assert result.runtime_metadata["qwen_failure_reason"] == "invalid json"
    assert result.runtime_metadata["final_action_source"] == "qwen_failure_stop"
    assert result.runtime_metadata["fallback_policy"] == "hard_failure_stop"
    assert "stop_gate_decision" not in result.runtime_metadata


def test_qwen_direct_repeated_turn_loop_chooses_opposite_turn():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "TURN_LEFT",
                "confidence": 0.8,
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={"recent_actions": ["TURN_LEFT", "TURN_LEFT", "TURN_LEFT"]},
    )

    assert result.ok is True
    assert result.action_text == "TURN_RIGHT"
    assert result.runtime_metadata["loop_gate_decision"] == "blocked"
    assert result.runtime_metadata["loop_pattern"] == "repeated_turn"
    assert result.runtime_metadata["replacement_action"] == "TURN_RIGHT"
    assert result.runtime_metadata["fallback_policy"] == "loop_break_turn"


def test_qwen_direct_turn_oscillation_is_audit_only():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "TURN_LEFT",
                "confidence": 0.8,
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "forward_stall_odometry_enabled": True,
            "recent_actions": [
                "TURN_LEFT",
                "TURN_RIGHT",
                "TURN_LEFT",
                "TURN_RIGHT",
                "TURN_LEFT",
                "TURN_RIGHT",
                "TURN_LEFT",
                "TURN_RIGHT",
            ],
            "local_control_context": {
                "odometry": {
                    "available": True,
                    "previous_action": "TURN_RIGHT",
                    "collision": False,
                    "consecutive_no_progress_forward": 0,
                }
            },
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["turn_oscillation_gate_decision"] == "audit_only"
    assert result.runtime_metadata["turn_oscillation_detected"] is True
    assert result.runtime_metadata["loop_pattern"] == "turn_oscillation"
    assert "replacement_action" not in result.runtime_metadata
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_repeated_forward_without_evidence_uses_forward_stall_gate():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "MOVE_FORWARD",
                "confidence": 0.82,
                "progress_state": "unknown",
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ],
            "planner_step_mode": "fast_text",
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["forward_stall_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "MOVE_FORWARD"
    assert result.runtime_metadata["replacement_action"] == "TURN_LEFT"
    assert result.runtime_metadata["fallback_policy"] == "forward_stall_turn"
    assert result.runtime_metadata["final_action_source"] == "forward_stall_gate"


def test_qwen_direct_repeated_forward_with_fresh_visual_but_no_progress_turns():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "MOVE_FORWARD",
                "confidence": 0.82,
                "visual_summary": "The archway remains ahead from a similar view.",
                "progress_state": "Approaching the archway; destination not yet reached.",
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ],
            "planner_step_mode": "visual_update",
        },
    )

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.runtime_metadata["forward_stall_gate_decision"] == "blocked"
    assert result.runtime_metadata["blocked_action"] == "MOVE_FORWARD"
    assert result.runtime_metadata["final_action_source"] == "forward_stall_gate"


def test_qwen_direct_effective_forward_odometry_overrides_text_stall_signal():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "MOVE_FORWARD",
                "confidence": 0.82,
                "visual_summary": "The archway remains centered ahead.",
                "progress_state": "approaching_target",
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": ["MOVE_FORWARD"] * 4,
            "planner_step_mode": "visual_update",
            "forward_stall_odometry_enabled": True,
            "local_control_context": {
                "odometry": {
                    "available": True,
                    "previous_action": "MOVE_FORWARD",
                    "last_forward_delta_m": 0.25,
                    "last_action_had_progress": True,
                    "consecutive_no_progress_forward": 0,
                    "collision": False,
                    "progress_threshold_m": 0.05,
                }
            },
        },
    )

    assert result.ok is True
    assert result.action_text == "MOVE_FORWARD"
    assert "forward_stall_gate_decision" not in result.runtime_metadata
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_odometry_mode_ignores_text_stall_when_odometry_is_unavailable():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "MOVE_FORWARD",
                "confidence": 0.82,
                "visual_summary": "The archway remains ahead from a similar view.",
                "progress_state": "Approaching the archway; destination not yet reached.",
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": ["MOVE_FORWARD"] * 4,
            "planner_step_mode": "visual_update",
            "forward_stall_odometry_enabled": True,
            "local_control_context": {"odometry": {"available": False}},
        },
    )

    assert result.ok is True
    assert result.action_text == "MOVE_FORWARD"
    assert "forward_stall_gate_decision" not in result.runtime_metadata
    assert result.runtime_metadata["final_action_source"] == "qwen"


def test_qwen_direct_repeated_forward_with_fresh_visual_can_pass():
    runtime = make_runtime(
        direct_decision(
            {
                "action_text": "MOVE_FORWARD",
                "confidence": 0.82,
                "progress_state": "progress_visible",
                "stop_evidence": "none",
            }
        )
    )

    result = runtime.step(
        make_state(),
        payload={
            "recent_actions": [
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
                "MOVE_FORWARD",
            ],
            "planner_step_mode": "visual_update",
        },
    )

    assert result.ok is True
    assert result.action_text == "MOVE_FORWARD"
    assert "forward_stall_gate_decision" not in result.runtime_metadata
    assert result.runtime_metadata["final_action_source"] == "qwen"
