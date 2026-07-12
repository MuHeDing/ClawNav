from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


DIRECT_SAVE_ACTIONS = {"STOP", "TURN_LEFT", "TURN_RIGHT"}
VALID_ACTIONS = {"STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"}


def normalize_candidate_action(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    return text if text in VALID_ACTIONS else ""


def candidate_action_status_from_tool_result(tool_result: Dict[str, Any]) -> str:
    if not tool_result.get("ok"):
        return "failed"
    payload = tool_result.get("payload")
    if not isinstance(payload, dict):
        return "failed"
    raw_action = str(payload.get("action_text") or "")
    if not raw_action.strip():
        return "empty"
    if not normalize_candidate_action(raw_action):
        return "unrecognized"
    return "ok"


@dataclass
class EventGatedKeyframeGate:
    min_gap_steps: int
    episode_cap: int
    coverage_gap_steps: int
    debug_save_all_eligible: bool = False

    def __post_init__(self) -> None:
        self._episode_key: Tuple[str, str] = ("", "")
        self._saved_count = 0
        self._last_saved_step: Optional[int] = None
        self._current_candidate_segment_action = ""
        self._current_candidate_segment_id = 0
        self._saved_candidate_segments: set[int] = set()

    def evaluate(
        self,
        scene_id: str,
        episode_id: str,
        step_id: int,
        raw_candidate_action: str,
        candidate_action_status: str,
        current_image_path: str,
        keyframe_target_path: str,
    ) -> Dict[str, Any]:
        self._reset_if_episode_changed(str(scene_id or ""), str(episode_id or ""))
        normalized_action = normalize_candidate_action(raw_candidate_action)
        result = self._base_result(
            step_id=step_id,
            raw_candidate_action=raw_candidate_action,
            normalized_action=normalized_action,
            candidate_action_status=candidate_action_status,
            current_image_path=current_image_path,
            keyframe_target_path=keyframe_target_path,
        )
        if not current_image_path:
            return self._skip(result, "missing_current_image")
        if not keyframe_target_path:
            return self._skip(result, "missing_keyframe_target_path")

        save_reason = self._eligible_save_reason(
            step_id=step_id,
            normalized_action=normalized_action,
            candidate_action_status=candidate_action_status,
            result=result,
        )
        if not save_reason:
            return result

        if save_reason == "candidate_decision_point":
            if self._segment_already_saved(normalized_action):
                return self._skip(result, "action_segment_duplicate")
            result["candidate_segment_id"] = self._current_candidate_segment_id

        if self._cooldown_active(step_id, save_reason):
            return self._skip(result, "cooldown")

        if self._saved_count >= self.episode_cap:
            blocked = self._skip(result, "episode_cap_reached")
            blocked["keyframe_cap_validation_failed"] = save_reason in {
                "candidate_decision_point",
                "coverage_gap",
            }
            return blocked

        result["save_decision"] = "save"
        result["save_reason"] = save_reason
        result["skip_reason"] = ""
        result["gate_eligible"] = True
        if save_reason == "initial_context":
            result["candidate_event_type"] = "initial_context"
        elif save_reason == "coverage_gap":
            result["candidate_event_type"] = "coverage_gap"
        else:
            result["candidate_event_type"] = normalized_action
        self._record_save(step_id, save_reason, normalized_action, result)
        return result

    def _eligible_save_reason(
        self,
        step_id: int,
        normalized_action: str,
        candidate_action_status: str,
        result: Dict[str, Any],
    ) -> str:
        if step_id == 0:
            return "initial_context"
        if candidate_action_status != "ok":
            self._reset_candidate_segment()
            self._skip(result, f"candidate_action_{candidate_action_status}")
            return ""
        if normalized_action in DIRECT_SAVE_ACTIONS:
            self._advance_candidate_segment(normalized_action)
            return "candidate_decision_point"
        self._reset_candidate_segment()
        if self._coverage_gap_active(step_id):
            return "coverage_gap"
        self._skip(result, "no_direct_save_event")
        return ""

    def _base_result(
        self,
        step_id: int,
        raw_candidate_action: str,
        normalized_action: str,
        candidate_action_status: str,
        current_image_path: str,
        keyframe_target_path: str,
    ) -> Dict[str, Any]:
        return {
            "gate_phase": "pre_readback",
            "keyframe_policy_mode": "event_gated_smoke",
            "step_id": step_id,
            "raw_candidate_action": str(raw_candidate_action or ""),
            "normalized_candidate_action": normalized_action,
            "candidate_action_status": candidate_action_status,
            "candidate_event_type": "",
            "candidate_segment_id": self._current_candidate_segment_id,
            "confirmed_event_type": "",
            "controller_event_status": "unconfirmed",
            "current_image_path": str(current_image_path or ""),
            "keyframe_target_path": str(keyframe_target_path or ""),
            "save_decision": "skip",
            "save_reason": "",
            "skip_reason": "",
            "gate_eligible": False,
            "promotion_status": "not_attempted",
            "write_status": "not_attempted",
            "memory_id_link_status": "not_attempted",
            "saved_count_before": self._saved_count,
            "saved_count_after": self._saved_count,
            "episode_cap": self.episode_cap,
            "cooldown_active": False,
            "keyframe_cap_validation_failed": False,
            "failure_reason": "",
        }

    def _skip(self, result: Dict[str, Any], reason: str) -> Dict[str, Any]:
        result["save_decision"] = "skip"
        result["skip_reason"] = reason
        result["failure_reason"] = reason
        if reason == "cooldown":
            result["cooldown_active"] = True
        return result

    def _reset_if_episode_changed(self, scene_id: str, episode_id: str) -> None:
        episode_key = (scene_id, episode_id)
        if episode_key == self._episode_key:
            return
        self._episode_key = episode_key
        self._saved_count = 0
        self._last_saved_step = None
        self._current_candidate_segment_action = ""
        self._current_candidate_segment_id = 0
        self._saved_candidate_segments = set()

    def _advance_candidate_segment(self, normalized_action: str) -> None:
        if normalized_action == self._current_candidate_segment_action:
            return
        self._current_candidate_segment_action = normalized_action
        self._current_candidate_segment_id += 1

    def _reset_candidate_segment(self) -> None:
        self._current_candidate_segment_action = ""

    def _segment_already_saved(self, normalized_action: str) -> bool:
        if normalized_action != self._current_candidate_segment_action:
            return False
        return self._current_candidate_segment_id in self._saved_candidate_segments

    def _cooldown_active(self, step_id: int, save_reason: str) -> bool:
        if save_reason == "initial_context":
            return False
        if self._last_saved_step is None:
            return False
        return step_id - self._last_saved_step < self.min_gap_steps

    def _coverage_gap_active(self, step_id: int) -> bool:
        if self.debug_save_all_eligible:
            return True
        if self._last_saved_step is None:
            return False
        return step_id - self._last_saved_step >= self.coverage_gap_steps

    def _record_save(
        self,
        step_id: int,
        save_reason: str,
        normalized_action: str,
        result: Dict[str, Any],
    ) -> None:
        self._saved_count += 1
        self._last_saved_step = step_id
        if save_reason == "candidate_decision_point":
            self._saved_candidate_segments.add(self._current_candidate_segment_id)
            result["candidate_segment_id"] = self._current_candidate_segment_id
        result["saved_count_after"] = self._saved_count
