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
STRUCTURAL_CONFIRM_FORWARD_NEGATIVE_MARKERS = (
    "blocked",
    "obstruct",
    "wall",
    "off center",
    "off-center",
    "not aligned",
    "not centered",
    "not visible",
    "left edge",
    "right edge",
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

        if candidate_action == "STOP":
            stop_block_metadata = self._stop_block_metadata(arguments, runtime_context)
            if not stop_block_metadata:
                passed_metadata = {
                    "stop_gate_decision": "passed",
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
                    "blocked_action": "STOP",
                    "replacement_action": replacement,
                    "final_action": replacement,
                    "final_action_source": "blocked_stop_gate",
                    "fallback_policy": fallback_policy,
                }
            )
            metadata.update(stop_block_metadata)
            return QwenDirectGateResult(replacement, metadata)

        loop_replacement = self._loop_replacement(candidate_action, runtime_context)
        if loop_replacement:
            metadata.update(
                {
                    "loop_gate_decision": "blocked",
                    "loop_pattern": "repeated_turn",
                    "replacement_action": loop_replacement,
                    "final_action": loop_replacement,
                    "final_action_source": "loop_gate",
                    "fallback_policy": "loop_break_turn",
                }
            )
            return QwenDirectGateResult(loop_replacement, metadata)

        forward_replacement = self._forward_stall_replacement(
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
        metadata.update(
            {
                "qwen_failure": True,
                "qwen_failure_reason": reason,
                "final_action": "STOP",
                "final_action_source": "qwen_failure_stop",
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

        progress_text = self._observed_route_progress_text(arguments, runtime_context)
        missing_waypoints = [
            waypoint
            for waypoint in self._required_intermediate_route_waypoints(instruction)
            if not self._contains_waypoint(progress_text, waypoint)
        ]
        if missing_waypoints:
            missing_reasons.append("intermediate_waypoint_progress")

        if not missing_reasons:
            return {}
        metadata: Dict[str, Any] = {
            "stop_gate_block_reason": "missing_route_progress",
            "stop_gate_route_missing_reasons": missing_reasons,
            "stop_gate_min_step_id": ROUTE_STOP_MIN_STEP_ID,
            "stop_gate_current_step_id": current_step_id,
            "stop_gate_min_forward_actions": ROUTE_STOP_MIN_FORWARD_ACTIONS,
            "stop_gate_forward_action_count": forward_count,
        }
        if missing_waypoints:
            metadata["stop_gate_missing_route_waypoints"] = missing_waypoints
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
        progress_text = self._observed_route_progress_text(arguments, runtime_context)
        return [
            waypoint
            for waypoint in required_waypoints
            if not self._contains_waypoint(progress_text, waypoint)
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

    def _observed_route_progress_text(
        self,
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> str:
        values = [
            arguments.get("visual_summary"),
            runtime_context.get("current_visual_summary"),
            runtime_context.get("recent_visual_summary"),
        ]
        for key in ("observed_visual_summaries", "observed_route_waypoints"):
            value = runtime_context.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif isinstance(value, str):
                values.append(value)
        return " ".join(str(value or "").strip().lower() for value in values)

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
        if stop_block_metadata.get("stop_gate_block_reason") == (
            "structural_stop_confirmation_required"
        ):
            if self._can_confirm_structural_stop_by_forward(arguments):
                return "MOVE_FORWARD", "structural_stop_confirm_forward"
        return self._turn_fallback(runtime_context), "blocked_stop_turn"

    def _can_confirm_structural_stop_by_forward(
        self,
        arguments: Dict[str, Any],
    ) -> bool:
        text = " ".join(
            [
                str(arguments.get("visual_summary") or "").strip().lower(),
                str(arguments.get("target_relation") or "").strip().lower(),
                str(arguments.get("reason") or "").strip().lower(),
            ]
        )
        return not any(
            marker in text for marker in STRUCTURAL_CONFIRM_FORWARD_NEGATIVE_MARKERS
        )

    def _structural_forward_would_overrun(self, text: str) -> bool:
        return any(marker in text for marker in STRUCTURAL_FORWARD_OVERRUN_MARKERS)

    def _contains_waypoint(self, text: str, waypoint: str) -> bool:
        if not text or not waypoint:
            return False
        return re.search(r"\b" + re.escape(waypoint) + r"\b", text) is not None

    def _loop_replacement(
        self,
        candidate_action: str,
        runtime_context: Dict[str, Any],
    ) -> str:
        if candidate_action not in {"TURN_LEFT", "TURN_RIGHT"}:
            return ""
        recent = self._recent_actions(runtime_context)
        if len(recent) < 3:
            return ""
        if recent[-3:] == [candidate_action, candidate_action, candidate_action]:
            return "TURN_RIGHT" if candidate_action == "TURN_LEFT" else "TURN_LEFT"
        return ""

    def _forward_stall_replacement(
        self,
        candidate_action: str,
        arguments: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> str:
        if candidate_action != "MOVE_FORWARD":
            return ""
        recent = self._recent_actions(runtime_context)
        if len(recent) < 4 or recent[-4:] != ["MOVE_FORWARD"] * 4:
            return ""
        if self._has_no_progress_evidence(arguments):
            return self._turn_fallback(runtime_context)
        if self._has_fresh_visual_evidence(runtime_context):
            return ""
        if self._has_progress_evidence(arguments):
            return ""
        return self._turn_fallback(runtime_context)

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

    def _int_or_none(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
