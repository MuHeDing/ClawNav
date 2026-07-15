from typing import Any, Dict, Mapping, Optional


ACTION_EXPECTED_EFFECTS = {
    "MOVE_FORWARD": "advance about 0.25m",
    "TURN_LEFT": "rotate about 15 degrees",
    "TURN_RIGHT": "rotate about 15 degrees",
    "STOP": "stop in place",
}


def build_motion_feedback(odometry: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(odometry, Mapping) or not odometry.get("available"):
        return {}
    last_action = _normalize_action(odometry.get("previous_action"))
    if not last_action:
        return {}

    feedback: Dict[str, Any] = {
        "last_action": last_action,
        "expected_effect": ACTION_EXPECTED_EFFECTS.get(last_action, "unknown"),
        "actual_effect": _actual_effect(last_action, odometry),
        "collision_recent": bool(odometry.get("collision")),
        "consecutive_no_progress_forward": _nonnegative_int(
            odometry.get("consecutive_no_progress_forward")
        ),
    }
    if last_action == "MOVE_FORWARD":
        feedback["last_forward_delta_m"] = _rounded_number(
            odometry.get("last_forward_delta_m")
        )
        feedback["distance_source"] = "rounded_local_odometry"
    elif last_action in {"TURN_LEFT", "TURN_RIGHT"}:
        feedback["last_turn_delta_deg"] = _rounded_number(
            odometry.get("last_turn_delta_deg")
        )
        feedback["rotation_source"] = "rounded_local_odometry"

    feedback["recommended_constraint"] = _recommended_constraint(feedback)
    return {key: value for key, value in feedback.items() if value is not None}


def _actual_effect(action: str, odometry: Mapping[str, Any]) -> str:
    had_progress = odometry.get("last_action_had_progress")
    if action == "MOVE_FORWARD":
        delta = _float_or_none(odometry.get("last_forward_delta_m"))
        threshold = _float_or_none(odometry.get("progress_threshold_m"))
        threshold = 0.05 if threshold is None else threshold
        consecutive_no_progress = _nonnegative_int(
            odometry.get("consecutive_no_progress_forward")
        )
        if (
            had_progress is False
            or consecutive_no_progress > 0
            or (delta is not None and delta < threshold)
        ):
            return "blocked" if bool(odometry.get("collision")) else "partial_forward"
        if delta is not None and delta >= 0.2:
            return "effective_forward"
        if delta is not None and delta >= threshold:
            return "partial_forward"
        return "unknown"
    if action in {"TURN_LEFT", "TURN_RIGHT"}:
        if had_progress is True:
            return "effective_turn"
        if had_progress is False:
            return "ineffective_turn"
        return "unknown"
    if action == "STOP":
        return "stopped"
    return "unknown"


def _recommended_constraint(feedback: Mapping[str, Any]) -> str:
    action = str(feedback.get("last_action") or "")
    effect = str(feedback.get("actual_effect") or "")
    if action == "MOVE_FORWARD":
        if effect == "blocked":
            return "avoid_forward"
        if effect == "partial_forward":
            return "prefer_realign_before_forward"
        if effect == "effective_forward":
            return "forward_allowed"
    if action in {"TURN_LEFT", "TURN_RIGHT"}:
        if effect == "effective_turn":
            return "turn_allowed"
        if effect == "ineffective_turn":
            return "avoid_repeating_same_turn"
    return "none"


def _normalize_action(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "FORWARD": "MOVE_FORWARD",
        "MOVE": "MOVE_FORWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in ACTION_EXPECTED_EFFECTS else ""


def _rounded_number(value: Any) -> Optional[float]:
    number = _float_or_none(value)
    if number is None:
        return None
    return round(number, 6)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
