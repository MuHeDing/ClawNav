from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


ACTION_TEXTS = {
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
}

ACTION_ALIASES = {
    "FORWARD": "MOVE_FORWARD",
    "MOVE": "MOVE_FORWARD",
    "LEFT": "TURN_LEFT",
    "RIGHT": "TURN_RIGHT",
}

NO_STOP_EVIDENCE = {"", "NONE", "NO", "FALSE", "UNKNOWN", "NULL"}
INSTRUCTION_COMPLETE_STOP_EVIDENCE = {
    "INSTRUCTION_COMPLETE",
    "ARRIVAL_VIEW",
    "ARRIVED",
    "AT_GOAL",
}
STOP_VISUAL_NEGATIVE_MARKERS = (
    "approach",
    "approaching",
    "toward",
    "continue",
    "not visible",
    "not visually",
    "not yet",
    "not at",
    "cannot see",
    "unclear",
    "unknown",
    "assumed",
    "inferred",
)
STOP_VISUAL_POSITIVE_MARKERS = (
    "visible",
    "in view",
    "directly ahead",
    "at the",
    "at archway",
    "at doorway",
    "at entrance",
    "at target",
    "at goal",
    "inside",
    "within",
    "beside",
    "next to",
    "framed by",
    "at destination",
    "goal object",
    "target is",
)
STOP_TERMINAL_VISUAL_MARKERS = (
    "outdoor",
    "outside",
    "exit",
    "foyer",
    "greenery",
    "directly ahead",
)
ARRIVED_SEMANTIC_STOP_STATES = {
    "AT_OR_INSIDE_TARGET",
    "BESIDE_TARGET",
    "INSTRUCTION_COMPLETE_AT_TARGET",
}
SEMANTIC_TARGET_RELATION_NEGATIVE_MARKERS = (
    "visible ahead",
    "directly ahead",
    "ahead",
    "left edge",
    "right edge",
    "edge",
    "vicinity",
    "nearby",
    "near",
    "approach",
    "approaching",
    "not reached",
    "not arrived",
    "not inside",
    "not beside",
    "not at",
    "off center",
    "off-center",
    "path clear",
)
SEMANTIC_TARGET_RELATION_POSITIVE_MARKERS = (
    "at ",
    "at the ",
    "at target",
    "inside",
    "in the ",
    "within",
    "standing in",
    "framed by",
    "threshold",
    "beside",
    "next to",
    "by ",
    "by the ",
)
ROUTE_WAYPOINT_PATTERNS = (
    ("outdoor foyer", "outdoor foyer"),
    ("living room", "living room"),
    ("dining room", "dining"),
    ("dining", "dining"),
    ("den", "den"),
    ("kitchen", "kitchen"),
    ("hallway", "hallway"),
    ("corridor", "corridor"),
    ("bedroom", "bedroom"),
    ("bathroom", "bathroom"),
    ("office", "office"),
    ("foyer", "foyer"),
    ("entryway", "entryway"),
    ("stairway", "stairway"),
    ("stairs", "stairs"),
    ("patio", "patio"),
    ("porch", "porch"),
    ("balcony", "balcony"),
    ("garage", "garage"),
)
FORWARD_STALL_NO_PROGRESS_MARKERS = (
    "approach",
    "approaching",
    "not yet",
    "not reached",
    "not arrived",
    "still",
    "same",
    "similar",
    "unchanged",
    "main room",
    "not at",
)
ROUTE_STOP_HOLDOFF_MARKERS = (
    "turn round",
    "turn around",
    "walk past",
    "go past",
    "move past",
    "keep walking straight",
    "walk straight on",
)
ROUTE_STOP_MIN_STEP_ID = 4
ROUTE_STOP_MIN_FORWARD_ACTIONS = 4
ROUTE_WAYPOINT_PASS_FORWARD_ACTIONS = 2
TURN_ROUND_COMPLETION_THRESHOLD_DEG = 150.0
ROUTE_WAYPOINT_NEGATIVE_MARKERS = (
    "not visible",
    "not seen",
    "not in view",
    "cannot see",
    "can't see",
    "could not see",
    "unable to see",
    "need to locate",
    "needs to locate",
    "looking for",
    "searching for",
    "without seeing",
    "before reaching",
    "not reached",
    "missing",
)
STRUCTURAL_STOP_CONFIRMATION_FORWARD_ACTIONS = 1
INTERMEDIATE_ROUTE_WAYPOINT_PATTERNS = (
    ("billiard table", "billiard table"),
    ("pool table", "billiard table"),
)
STRUCTURAL_STOP_TARGETS = (
    "archway",
    "doorway",
    "entrance",
)
STRUCTURAL_TARGET_LOOSE_RELATION_MARKERS = (
    "beside_target",
    "beside target",
    "beside",
    "next to",
    "adjacent",
)
STRUCTURAL_TARGET_ARRIVAL_MARKERS = (
    "inside",
    "within",
    "framed by",
    "threshold",
    "in the archway",
    "at the archway",
    "at archway",
    "in the doorway",
    "at the doorway",
    "at doorway",
    "in the entrance",
    "at the entrance",
    "at entrance",
)
STRUCTURAL_FORWARD_OVERRUN_MARKERS = (
    "further movement would",
    "further forward movement would",
    "moving forward would",
    "move forward would",
    "would place the agent inside",
    "would place it inside",
    "would take the agent deeper",
    "would take it deeper",
    "deeper into",
    "rather than waiting",
    "rather than keeping",
    "violates the task",
    "violate the task",
    "overshoot",
    "go past",
    "too far",
)


@dataclass
class QwenDirectGateResult:
    final_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class QwenDirectControlGates:
    def __init__(self, low_confidence_threshold: float = 0.35) -> None:
        self.low_confidence_threshold = float(low_confidence_threshold)

    def apply(
        self,
        arguments: Dict[str, Any],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> QwenDirectGateResult:
        runtime_context = runtime_context or {}
        candidate_action = self._normalize_action(arguments.get("action_text"))
        metadata = {
            "candidate_action": candidate_action,
            "navigation_policy_skill_called": False,
        }

        if bool(arguments.get("qwen_failure")):
            return self._hard_failure(arguments, metadata)

        if not candidate_action:
            failure_args = {
                "qwen_failure_reason": "missing_or_unsupported_action_text",
                "fallback_policy": "hard_failure_stop",
            }
            return self._hard_failure(failure_args, metadata)

        if (
            candidate_action == "MOVE_FORWARD"
            and runtime_context.get("structural_stop_verification_active") is True
        ):
            replacement = self._turn_fallback(runtime_context)
            metadata.update(
                {
                    "stop_verification_gate_decision": "blocked",
                    "blocked_action": "MOVE_FORWARD",
                    "replacement_action": replacement,
                    "final_action": replacement,
                    "final_action_source": "stop_verification_gate",
                    "fallback_policy": "stop_verification_turn",
                }
            )
            return QwenDirectGateResult(replacement, metadata)

        if candidate_action == "STOP":
            stop_block_metadata = self._stop_block_metadata(arguments, runtime_context)
            if not stop_block_metadata:
                passed_metadata = {
                    "stop_gate_decision": "passed",
                    "stop_permission": True,
                    "final_action": "STOP",
                    "final_action_source": "qwen",
                }
                passed_metadata.update(
                    self._structural_confirmation_metadata(runtime_context)
                )
                metadata.update(passed_metadata)
                return QwenDirectGateResult("STOP", metadata)
            replacement, fallback_policy = self._blocked_stop_replacement(
                stop_block_metadata,
                arguments,
                runtime_context,
            )
            metadata.update(
                {
                    "stop_gate_decision": "blocked",
                    "stop_permission": False,
                    "blocked_action": "STOP",
                    "replacement_action": replacement,
                    "final_action": replacement,
                    "final_action_source": "blocked_stop_gate",
                    "fallback_policy": fallback_policy,
                }
            )
            metadata.update(stop_block_metadata)
            return QwenDirectGateResult(replacement, metadata)

        if self._repeated_same_direction_turn(candidate_action, runtime_context):
            metadata.update(
                {
                    "loop_gate_decision": "audit_only",
                    "loop_pattern": "repeated_turn",
                }
            )

        turn_oscillation_detected = self._turn_oscillation_detected(
            candidate_action,
            runtime_context,
        )
        if turn_oscillation_detected:
            metadata.update(
                {
                    "turn_oscillation_gate_decision": "audit_only",
                    "turn_oscillation_detected": True,
                    "loop_pattern": "turn_oscillation",
                }
            )

        forward_replacement, forward_metadata = self._forward_stall_replacement(
            candidate_action,
            arguments,
            runtime_context,
        )
        if forward_replacement:
            metadata.update(
                {
                    "forward_stall_gate_decision": "blocked",
                    "blocked_action": "MOVE_FORWARD",
                    "replacement_action": forward_replacement,
                    "final_action": forward_replacement,
                    "final_action_source": "forward_stall_gate",
                    "fallback_policy": "forward_stall_turn",
                }
            )
            metadata.update(forward_metadata)
            return QwenDirectGateResult(forward_replacement, metadata)

        confidence = self._confidence(arguments.get("confidence"))
        if confidence is not None and confidence < self.low_confidence_threshold:
            replacement = self._turn_fallback(runtime_context)
            metadata.update(
                {
                    "uncertainty_gate_decision": "fallback_turn",
                    "replacement_action": replacement,
                    "final_action": replacement,
                    "final_action_source": "uncertainty_gate",
                    "fallback_policy": "uncertainty_turn",
                }
            )
            return QwenDirectGateResult(replacement, metadata)

        metadata.update(
            {
                "final_action": candidate_action,
                "final_action_source": "qwen",
            }
        )
        return QwenDirectGateResult(candidate_action, metadata)

    def _hard_failure(
        self,
        arguments: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> QwenDirectGateResult:
        reason = str(arguments.get("qwen_failure_reason") or "qwen_direct_failure")
        episode_invalid = bool(arguments.get("episode_invalid"))
        metadata.update(
            {
                "qwen_failure": True,
                "qwen_failure_reason": reason,
                "episode_invalid": episode_invalid,
                "final_action": "STOP",
                "final_action_source": (
                    "qwen_invalid_episode_abort"
                    if episode_invalid
                    else "qwen_failure_stop"
                ),
                "fallback_policy": str(
                    arguments.get("fallback_policy") or "hard_failure_stop"
                ),
            }
        )
        return QwenDirectGateResult("STOP", metadata)

    def _stop_block_metadata(
        self,
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence = str(arguments.get("stop_evidence") or "").strip().upper()
        if evidence in NO_STOP_EVIDENCE:
            return {"stop_gate_block_reason": "missing_stop_evidence"}
        if evidence not in {"VISIBLE_GOAL", *INSTRUCTION_COMPLETE_STOP_EVIDENCE}:
            return {"stop_gate_block_reason": "unsupported_stop_evidence"}
        semantic_block = self._semantic_stop_block_metadata(arguments)
        if semantic_block:
            return semantic_block
        if not self._has_current_stop_visual_evidence(arguments):
            return {"stop_gate_block_reason": "insufficient_visual_evidence"}
        structural_block = self._structural_target_block_metadata(
            arguments,
            runtime_context,
        )
        if structural_block:
            return structural_block
        route_block = self._route_stop_block_metadata(arguments, runtime_context)
        if route_block:
            return route_block
        missing_waypoints = self._missing_multi_waypoint_progress(
            arguments,
            runtime_context,
        )
        if missing_waypoints:
            return {
                "stop_gate_block_reason": "missing_multi_waypoint_progress",
                "stop_gate_missing_waypoints": missing_waypoints,
            }
        return {}

    def _semantic_stop_block_metadata(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        semantic_state = self._normalize_semantic_stop_state(
            arguments.get("semantic_stop_state")
        )
        if not semantic_state:
            return {"stop_gate_block_reason": "missing_semantic_stop_state"}
        if semantic_state not in ARRIVED_SEMANTIC_STOP_STATES:
            return {
                "stop_gate_block_reason": "semantic_stop_state_not_arrived",
                "semantic_stop_state": semantic_state,
            }
        target_relation = self._normalize_target_relation(
            arguments.get("target_relation")
        )
        if not target_relation:
            return {
                "stop_gate_block_reason": "missing_target_relation",
                "semantic_stop_state": semantic_state,
            }
        if self._target_relation_is_not_arrived(target_relation):
            return {
                "stop_gate_block_reason": "semantic_target_relation_not_arrived",
                "semantic_stop_state": semantic_state,
                "target_relation": target_relation,
            }
        if not self._target_relation_has_arrival_anchor(target_relation):
            return {
                "stop_gate_block_reason": "semantic_target_relation_not_arrived",
                "semantic_stop_state": semantic_state,
                "target_relation": target_relation,
            }
        return {}

    def _normalize_semantic_stop_state(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        return normalized

    def _normalize_target_relation(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.strip().lower().replace("_", " ").replace("-", " ")
        return " ".join(normalized.split())

    def _target_relation_is_not_arrived(self, target_relation: str) -> bool:
        return any(
            marker in target_relation
            for marker in SEMANTIC_TARGET_RELATION_NEGATIVE_MARKERS
        )

    def _target_relation_has_arrival_anchor(self, target_relation: str) -> bool:
        return any(
            marker in target_relation
            for marker in SEMANTIC_TARGET_RELATION_POSITIVE_MARKERS
        )

    def _has_current_stop_visual_evidence(self, arguments: Dict[str, Any]) -> bool:
        visual_summary = str(arguments.get("visual_summary") or "").strip().lower()
        if not visual_summary:
            return False
        stop_explanation = " ".join(
            [
                visual_summary,
                str(arguments.get("reason") or "").strip().lower(),
                str(arguments.get("progress_state") or "").strip().lower(),
            ]
        )
        if any(marker in stop_explanation for marker in STOP_VISUAL_NEGATIVE_MARKERS):
            return False
        return any(
            marker in visual_summary for marker in STOP_VISUAL_POSITIVE_MARKERS
        )

    def _structural_target_block_metadata(
        self,
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        current_target = self._normalize_target_relation(
            arguments.get("current_target")
        )
        if not any(target in current_target for target in STRUCTURAL_STOP_TARGETS):
            return {}
        target_relation = self._normalize_target_relation(
            arguments.get("target_relation")
        )
        semantic_state = self._normalize_semantic_stop_state(
            arguments.get("semantic_stop_state")
        )
        loose_relation = semantic_state == "BESIDE_TARGET" or any(
            marker in target_relation
            for marker in STRUCTURAL_TARGET_LOOSE_RELATION_MARKERS
        )
        arrival_text = " ".join(
            [
                target_relation,
                str(arguments.get("visual_summary") or "").strip().lower(),
                str(arguments.get("reason") or "").strip().lower(),
            ]
        )
        has_arrival_anchor = any(
            marker in arrival_text for marker in STRUCTURAL_TARGET_ARRIVAL_MARKERS
        )
        if loose_relation and not has_arrival_anchor:
            return {
                "stop_gate_block_reason": "structural_target_relation_too_loose",
                "semantic_stop_state": semantic_state,
                "target_relation": target_relation,
            }
        if has_arrival_anchor:
            if self._structural_forward_would_overrun(arrival_text):
                return {}
            if runtime_context.get("structural_stop_verification_active") is True:
                return {}
            confirmation_count = self._structural_confirmation_forward_count(
                runtime_context
            )
            if confirmation_count >= STRUCTURAL_STOP_CONFIRMATION_FORWARD_ACTIONS:
                return {}
            return {
                "stop_gate_block_reason": "structural_stop_confirmation_required",
                "semantic_stop_state": semantic_state,
                "target_relation": target_relation,
                "stop_gate_structural_confirmation_forward_count": confirmation_count,
                "stop_gate_structural_confirmation_min_forward_actions": (
                    STRUCTURAL_STOP_CONFIRMATION_FORWARD_ACTIONS
                ),
            }
        return {}

    def _route_stop_block_metadata(
        self,
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        instruction = str(runtime_context.get("instruction") or "").strip().lower()
        if not self._instruction_needs_route_stop_holdoff(instruction):
            return {}
        current_step_id = self._int_or_none(runtime_context.get("current_step_id"))
        recent_actions = self._recent_actions(runtime_context)
        forward_count = sum(1 for action in recent_actions if action == "MOVE_FORWARD")
        missing_reasons: List[str] = []
        if current_step_id is None or current_step_id < ROUTE_STOP_MIN_STEP_ID:
            missing_reasons.append("min_step")
        if forward_count < ROUTE_STOP_MIN_FORWARD_ACTIONS:
            missing_reasons.append("min_forward_actions")

        required_waypoints = self._required_intermediate_route_waypoints(instruction)
        route_progress = runtime_context.get("route_progress")
        if not isinstance(route_progress, dict):
            route_progress = {}
        observations = self._route_visual_observations(arguments, runtime_context)
        positively_seen = self._normalized_waypoint_list(
            route_progress.get("positively_seen_waypoints")
        )
        passed_waypoints = self._normalized_waypoint_list(
            route_progress.get("passed_waypoints")
        )
        negated_mentions = self._normalized_waypoint_list(
            route_progress.get("negated_waypoint_mentions")
        )
        for waypoint in required_waypoints:
            observation = self._waypoint_observation_status(observations, waypoint)
            if observation == "positive" and waypoint not in positively_seen:
                positively_seen.append(waypoint)
            elif observation == "negated" and waypoint not in negated_mentions:
                negated_mentions.append(waypoint)

        turn_round_required = self._instruction_requires_turn_round(instruction)
        turn_round_completed = (
            route_progress.get("turn_round_completed") is True
            if turn_round_required
            else True
        )
        heading_change = self._float_or_none(
            route_progress.get("heading_change_from_start_deg")
        )
        if turn_round_required and not turn_round_completed:
            missing_reasons.append("turn_round_progress")

        missing_waypoints = [
            waypoint for waypoint in required_waypoints if waypoint not in positively_seen
        ]
        unpassed_waypoints = [
            waypoint
            for waypoint in required_waypoints
            if waypoint in positively_seen and waypoint not in passed_waypoints
        ]
        if missing_waypoints:
            missing_reasons.append("intermediate_waypoint_seen")
        elif unpassed_waypoints:
            missing_reasons.append("intermediate_waypoint_passed")

        if not missing_reasons:
            return {}
        metadata: Dict[str, Any] = {
            "stop_gate_block_reason": "missing_route_progress",
            "stop_gate_route_missing_reasons": missing_reasons,
            "stop_gate_min_step_id": ROUTE_STOP_MIN_STEP_ID,
            "stop_gate_current_step_id": current_step_id,
            "stop_gate_min_forward_actions": ROUTE_STOP_MIN_FORWARD_ACTIONS,
            "stop_gate_forward_action_count": forward_count,
            "stop_gate_required_route_waypoints": required_waypoints,
            "stop_gate_positively_seen_waypoints": positively_seen,
            "stop_gate_passed_waypoints": passed_waypoints,
            "stop_gate_negated_waypoint_mentions": negated_mentions,
            "stop_gate_turn_round_required": turn_round_required,
            "stop_gate_turn_round_completed": turn_round_completed,
            "stop_gate_heading_change_from_start_deg": heading_change,
        }
        if missing_waypoints:
            metadata["stop_gate_missing_route_waypoints"] = missing_waypoints
        if unpassed_waypoints:
            metadata["stop_gate_unpassed_route_waypoints"] = unpassed_waypoints
        return metadata

    def _instruction_needs_route_stop_holdoff(self, instruction: str) -> bool:
        if not instruction or "stop" not in instruction:
            return False
        return any(marker in instruction for marker in ROUTE_STOP_HOLDOFF_MARKERS)

    def _required_intermediate_route_waypoints(self, instruction: str) -> List[str]:
        matches = []
        for phrase, waypoint in INTERMEDIATE_ROUTE_WAYPOINT_PATTERNS:
            match = re.search(r"\b" + re.escape(phrase) + r"\b", instruction)
            if match is not None:
                matches.append((match.start(), waypoint))
        waypoints = []
        seen = set()
        for _, waypoint in sorted(matches, key=lambda item: item[0]):
            if waypoint in seen:
                continue
            waypoints.append(waypoint)
            seen.add(waypoint)
        return waypoints

    def _missing_multi_waypoint_progress(
        self,
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> List[str]:
        if not self._looks_like_terminal_visual_stop(arguments):
            return []
        required_waypoints = self._required_route_waypoints(runtime_context)
        if len(required_waypoints) < 2:
            return []
        observations = self._route_visual_observations(arguments, runtime_context)
        return [
            waypoint
            for waypoint in required_waypoints
            if self._waypoint_observation_status(observations, waypoint) != "positive"
        ]

    def _looks_like_terminal_visual_stop(self, arguments: Dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(arguments.get("visual_summary") or "").strip().lower(),
                str(arguments.get("reason") or "").strip().lower(),
                str(arguments.get("progress_state") or "").strip().lower(),
            ]
        )
        return any(marker in text for marker in STOP_TERMINAL_VISUAL_MARKERS)

    def _required_route_waypoints(
        self,
        runtime_context: Dict[str, Any],
    ) -> List[str]:
        instruction = str(runtime_context.get("instruction") or "").strip().lower()
        if not instruction:
            return []
        waypoints = self._extract_route_waypoints(instruction)
        if len(waypoints) >= 4:
            waypoints = waypoints[1:]
        return waypoints

    def _extract_route_waypoints(self, instruction: str) -> List[str]:
        matches = []
        for phrase, waypoint in ROUTE_WAYPOINT_PATTERNS:
            match = re.search(r"\b" + re.escape(phrase) + r"\b", instruction)
            if match is not None:
                matches.append((match.start(), waypoint))
        waypoints = []
        seen = set()
        for _, waypoint in sorted(matches, key=lambda item: item[0]):
            if waypoint in seen:
                continue
            if any(self._contains_waypoint(existing, waypoint) for existing in waypoints):
                continue
            waypoints.append(waypoint)
            seen.add(waypoint)
        return waypoints

    def _route_visual_observations(
        self,
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> List[str]:
        values: List[Any] = [
            arguments.get("visual_summary"),
            runtime_context.get("current_visual_summary"),
            runtime_context.get("recent_visual_summary"),
        ]
        observed = runtime_context.get("observed_visual_summaries")
        if isinstance(observed, list):
            values.extend(observed)
        elif isinstance(observed, str):
            values.append(observed)
        return [str(value).strip().lower() for value in values if str(value or "").strip()]

    def _waypoint_observation_status(
        self,
        observations: List[str],
        waypoint: str,
    ) -> str:
        saw_negated = False
        pattern = re.compile(r"\b" + re.escape(waypoint) + r"\b")
        for observation in observations:
            clauses = re.split(r"[.!?;\n]|\bbut\b|\bhowever\b", observation.lower())
            for clause in clauses:
                if pattern.search(clause) is None:
                    continue
                if any(marker in clause for marker in ROUTE_WAYPOINT_NEGATIVE_MARKERS):
                    saw_negated = True
                    continue
                return "positive"
        return "negated" if saw_negated else ""

    def classify_waypoint_observation(self, summary: Any, waypoint: str) -> str:
        text = str(summary or "").strip().lower()
        if not text:
            return ""
        return self._waypoint_observation_status([text], waypoint)

    def required_intermediate_route_waypoints(self, instruction: str) -> List[str]:
        return self._required_intermediate_route_waypoints(
            str(instruction or "").strip().lower()
        )

    def route_requires_turn_round(self, instruction: str) -> bool:
        return self._instruction_requires_turn_round(
            str(instruction or "").strip().lower()
        )

    def _instruction_requires_turn_round(self, instruction: str) -> bool:
        return "turn round" in instruction or "turn around" in instruction

    def _normalized_waypoint_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        for item in value:
            waypoint = str(item or "").strip().lower()
            if waypoint and waypoint not in normalized:
                normalized.append(waypoint)
        return normalized

    def _structural_confirmation_forward_count(
        self,
        runtime_context: Dict[str, Any],
    ) -> int:
        value = self._int_or_none(
            runtime_context.get("structural_stop_confirmation_forward_count")
        )
        return max(0, value or 0)

    def _structural_confirmation_metadata(
        self,
        runtime_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "structural_stop_confirmation_forward_count" not in runtime_context:
            return {}
        return {
            "stop_gate_structural_confirmation_forward_count": (
                self._structural_confirmation_forward_count(runtime_context)
            ),
            "stop_gate_structural_confirmation_min_forward_actions": (
                STRUCTURAL_STOP_CONFIRMATION_FORWARD_ACTIONS
            ),
        }

    def _blocked_stop_replacement(
        self,
        stop_block_metadata: Dict[str, Any],
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> tuple[str, str]:
        return self._turn_fallback(runtime_context), "blocked_stop_turn"

    def _structural_forward_would_overrun(self, text: str) -> bool:
        return any(marker in text for marker in STRUCTURAL_FORWARD_OVERRUN_MARKERS)

    def _contains_waypoint(self, text: str, waypoint: str) -> bool:
        if not text or not waypoint:
            return False
        return re.search(r"\b" + re.escape(waypoint) + r"\b", text) is not None

    def _repeated_same_direction_turn(
        self,
        candidate_action: str,
        runtime_context: Dict[str, Any],
    ) -> bool:
        if candidate_action not in {"TURN_LEFT", "TURN_RIGHT"}:
            return False
        if runtime_context.get("turn_loop_recovery_active") is True:
            return False
        recent = self._recent_actions(runtime_context)
        if len(recent) < 3:
            return False
        return recent[-3:] == [candidate_action, candidate_action, candidate_action]

    def _forward_stall_replacement(
        self,
        candidate_action: str,
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        if candidate_action != "MOVE_FORWARD":
            return "", {}
        odometry_metadata = self._odometry_forward_stall_metadata(runtime_context)
        if odometry_metadata:
            return self._turn_fallback(runtime_context), odometry_metadata
        if runtime_context.get("forward_stall_odometry_enabled") is True:
            return "", {}
        recent = self._recent_actions(runtime_context)
        if len(recent) < 4 or recent[-4:] != ["MOVE_FORWARD"] * 4:
            return "", {}
        if self._has_no_progress_evidence(arguments):
            return self._turn_fallback(runtime_context), {
                "forward_stall_evidence_source": "qwen_text",
                "forward_stall_gate_reason": "no_progress_evidence",
            }
        if self._has_fresh_visual_evidence(runtime_context):
            return "", {}
        if self._has_progress_evidence(arguments):
            return "", {}
        return self._turn_fallback(runtime_context), {
            "forward_stall_evidence_source": "recent_actions",
            "forward_stall_gate_reason": "repeated_forward_without_fresh_progress",
        }

    def _odometry_forward_stall_metadata(
        self,
        runtime_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if runtime_context.get("forward_stall_odometry_enabled") is not True:
            return {}
        odometry = self._odometry_context(runtime_context)
        if not odometry:
            return {}
        if self._normalize_action(odometry.get("previous_action")) != "MOVE_FORWARD":
            return {}
        consecutive_no_progress = self._int_or_none(
            odometry.get("consecutive_no_progress_forward")
        )
        consecutive_no_progress = consecutive_no_progress or 0
        last_forward_delta = self._float_or_none(odometry.get("last_forward_delta_m"))
        progress_threshold = self._float_or_none(odometry.get("progress_threshold_m"))
        if progress_threshold is None:
            progress_threshold = 0.05
        collision = bool(odometry.get("collision"))
        tiny_forward = (
            last_forward_delta is not None
            and last_forward_delta < progress_threshold
        )
        if consecutive_no_progress < 2 and not (collision and tiny_forward):
            return {}
        reason = (
            "odometry_consecutive_no_progress"
            if consecutive_no_progress >= 2
            else "odometry_collision_tiny_forward_delta"
        )
        return {
            "forward_stall_evidence_source": "odometry",
            "forward_stall_gate_reason": reason,
            "blocked_forward_by_odometry_gate": True,
            "forward_stall_odometry_consecutive_no_progress_forward": (
                consecutive_no_progress
            ),
            "forward_stall_odometry_last_forward_delta_m": last_forward_delta,
            "forward_stall_odometry_collision": collision,
        }

    def _turn_oscillation_detected(
        self,
        candidate_action: str,
        runtime_context: Dict[str, Any],
    ) -> bool:
        if candidate_action not in {"TURN_LEFT", "TURN_RIGHT"}:
            return False
        if runtime_context.get("forward_stall_odometry_enabled") is not True:
            return False
        recent = self._recent_actions(runtime_context)
        tail = recent[-8:]
        if len(tail) < 6:
            return False
        turn_count = sum(1 for action in tail if action in {"TURN_LEFT", "TURN_RIGHT"})
        forward_count = sum(1 for action in tail if action == "MOVE_FORWARD")
        if turn_count < 6 or forward_count > 1:
            return False
        odometry = self._odometry_context(runtime_context)
        if not odometry:
            return False
        if bool(odometry.get("collision")):
            return False
        consecutive_no_progress = self._int_or_none(
            odometry.get("consecutive_no_progress_forward")
        )
        if consecutive_no_progress and consecutive_no_progress > 0:
            return False
        return True

    def _odometry_context(self, runtime_context: Dict[str, Any]) -> Dict[str, Any]:
        local_control_context = runtime_context.get("local_control_context")
        if not isinstance(local_control_context, dict):
            return {}
        odometry = local_control_context.get("odometry")
        return odometry if isinstance(odometry, dict) else {}

    def _has_fresh_visual_evidence(self, runtime_context: Dict[str, Any]) -> bool:
        planner_step_mode = str(runtime_context.get("planner_step_mode") or "")
        if planner_step_mode == "visual_update":
            return True
        if runtime_context.get("current_visual_evidence") is True:
            return True
        return runtime_context.get("used_current_image") is True

    def _has_progress_evidence(self, arguments: Dict[str, Any]) -> bool:
        progress_state = str(arguments.get("progress_state") or "").strip().lower()
        if not progress_state:
            return False
        negative_markers = ("unknown", "stuck", "stall", "no_progress", "not sure")
        if any(marker in progress_state for marker in negative_markers):
            return False
        positive_markers = (
            "progress",
            "advance",
            "advancing",
            "centered",
            "following",
            "visible",
            "aligned",
        )
        return any(marker in progress_state for marker in positive_markers)

    def _has_no_progress_evidence(self, arguments: Dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(arguments.get("progress_state") or "").strip().lower(),
                str(arguments.get("visual_summary") or "").strip().lower(),
                str(arguments.get("reason") or "").strip().lower(),
            ]
        )
        return any(marker in text for marker in FORWARD_STALL_NO_PROGRESS_MARKERS)

    def _turn_fallback(self, runtime_context: Dict[str, Any]) -> str:
        recent = self._recent_actions(runtime_context)
        if len(recent) >= 2 and recent[-2:] == ["TURN_LEFT", "TURN_LEFT"]:
            return "TURN_RIGHT"
        return "TURN_LEFT"

    def _recent_actions(self, runtime_context: Dict[str, Any]) -> List[str]:
        raw_actions = runtime_context.get("recent_actions") or []
        if not isinstance(raw_actions, list):
            return []
        return [
            action
            for action in (self._normalize_action(value) for value in raw_actions)
            if action
        ]

    def _normalize_action(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        normalized = ACTION_ALIASES.get(normalized, normalized)
        return normalized if normalized in ACTION_TEXTS else ""

    def _confidence(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _float_or_none(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _int_or_none(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
