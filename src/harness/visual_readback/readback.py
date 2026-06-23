import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def normalize_visual_readback_response(
    response: Any,
    request: Dict[str, Any],
) -> Dict[str, Any]:
    attached_memory_ids = list(request.get("attached_memory_ids") or [])
    actually_read_image_paths = list(request.get("actually_read_image_paths") or [])
    try:
        data = _response_dict(response)
    except ValueError:
        return _failed_payload(
            request,
            error_type="parse_failure",
            actually_read_image_paths=actually_read_image_paths,
        )

    matched_memory_ids = [
        memory_id
        for memory_id in _string_list(data.get("matched_memory_ids"))
        if memory_id in attached_memory_ids
    ]
    verifier_labels = _string_list(data.get("verifier_labels")) or [
        "insufficient_evidence"
    ]
    evidence_audit = _evidence_source_audit(data, request)
    return {
        "read_status": "completed",
        "trigger_rule": str(request.get("trigger_rule") or ""),
        "candidate_action": str(request.get("candidate_action") or ""),
        "retrieved_image_paths": list(request.get("retrieved_image_paths") or []),
        "actually_read_image_paths": actually_read_image_paths,
        "model_image_count": len(actually_read_image_paths),
        "original_retrieved_memory_ids": list(
            request.get("original_retrieved_memory_ids") or []
        ),
        "attached_memory_ids": attached_memory_ids,
        "matched_memory_ids": matched_memory_ids,
        "matched_memory_ids_from_attached_only": True,
        "required_images": str(data.get("required_images") or "memory"),
        "verifier_labels": verifier_labels,
        "visual_evidence": str(data.get("visual_evidence") or ""),
        "audit_action_hint": str(data.get("audit_action_hint") or ""),
        "candidate_action_valid": _bool(data.get("candidate_action_valid"), True),
        "should_override": _bool(data.get("should_override"), False),
        "invalid_reason": str(data.get("invalid_reason") or ""),
        "recommended_action": str(data.get("recommended_action") or ""),
        "decision_scope": str(data.get("decision_scope") or ""),
        "action_confidence": _float(data.get("action_confidence"), 0.0),
        "audit_relative_direction_hint": str(
            data.get("audit_relative_direction_hint") or ""
        ),
        "readback_confidence": _float(data.get("readback_confidence"), 0.0),
        "retrieval_confidence": _float(data.get("retrieval_confidence"), 0.0),
        "verifier_confidence": _float(data.get("verifier_confidence"), 0.0),
        "visual_grounding_status": normalize_visual_grounding_status(
            data.get("visual_grounding_status")
        ),
        "grounding_eval_protocol": data.get("grounding_eval_protocol") or {},
        **evidence_audit,
    }


def build_readback_request(
    current_image_path: str,
    memory_hits: Iterable[Any],
    candidate_action: str,
    trigger_rule: str,
    instruction: str,
) -> Dict[str, Any]:
    attached_memory_ids: List[str] = []
    retrieved_image_paths: List[str] = []
    retrieval_confidences: List[float] = []
    for hit in memory_hits:
        memory_id = _hit_value(hit, "memory_id")
        image_path = _hit_value(hit, "image_path")
        if not image_path:
            continue
        attached_memory_ids.append(str(memory_id))
        retrieved_image_paths.append(str(image_path))
        retrieval_confidences.append(_float(_hit_value(hit, "confidence"), 0.0))
    actually_read_image_paths = [str(current_image_path)] + retrieved_image_paths
    return {
        "current_image_path": str(current_image_path),
        "retrieved_image_paths": retrieved_image_paths,
        "actually_read_image_paths": actually_read_image_paths,
        "original_retrieved_memory_ids": list(attached_memory_ids),
        "attached_memory_ids": attached_memory_ids,
        "retrieval_confidence": max(retrieval_confidences) if retrieval_confidences else 0.0,
        "candidate_action": str(candidate_action or ""),
        "trigger_rule": str(trigger_rule or ""),
        "instruction": str(instruction or ""),
    }


def validate_readback_images(request: Dict[str, Any]) -> str:
    current_image_path = str(request.get("current_image_path") or "")
    if not current_image_path or not Path(current_image_path).exists():
        return "missing_current_image"
    if not request.get("retrieved_image_paths"):
        return "missing_memory_image"
    missing_memory = [
        path for path in request["retrieved_image_paths"] if not Path(path).exists()
    ]
    if missing_memory:
        return "missing_memory_image"
    return ""


def normalize_visual_grounding_status(value: Any) -> str:
    status = str(value or "unverified").strip().lower()
    if status in {
        "grounded",
        "verified",
        "successful",
        "success",
        "clear",
        "complete",
        "confirmed",
        "memory_image_readable",
    }:
        return "grounded"
    if status in {
        "mismatch",
        "conflict",
        "conflict_detected",
        "different",
        "difference",
    }:
        return "grounded"
    if status in {"ungrounded", "not_grounded", "not grounded"}:
        return "ungrounded"
    return status or "unverified"


def _response_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("response JSON must be an object")
        return data
    raise ValueError(f"unsupported response type: {type(response).__name__}")


def _failed_payload(
    request: Dict[str, Any],
    error_type: str,
    actually_read_image_paths: List[str],
) -> Dict[str, Any]:
    return {
        "read_status": "failed",
        "error_type": error_type,
        "trigger_rule": str(request.get("trigger_rule") or ""),
        "candidate_action": str(request.get("candidate_action") or ""),
        "retrieved_image_paths": list(request.get("retrieved_image_paths") or []),
        "actually_read_image_paths": actually_read_image_paths,
        "model_image_count": len(actually_read_image_paths),
        "attached_memory_ids": list(request.get("attached_memory_ids") or []),
        "matched_memory_ids": [],
        "matched_memory_ids_from_attached_only": True,
        "verifier_labels": ["insufficient_evidence"],
        "visual_evidence": "",
        "audit_action_hint": "",
        "candidate_action_valid": True,
        "should_override": False,
        "invalid_reason": "",
        "recommended_action": "",
        "decision_scope": "",
        "action_confidence": 0.0,
        "readback_confidence": 0.0,
        "retrieval_confidence": _float(request.get("retrieval_confidence"), 0.0),
        "verifier_confidence": 0.0,
        "visual_grounding_status": "unverified",
        "evidence_sources": [],
        "memory_evidence_used_count": 0,
        "current_only_evidence_count": 0,
        "ambiguous_evidence_count": 0,
        "invalid_evidence_source_count": 0,
        "no_evidence_count": 1,
    }


def _evidence_source_audit(
    data: Dict[str, Any],
    request: Dict[str, Any],
) -> Dict[str, Any]:
    raw_sources = data.get("evidence_sources")
    attached_memory_ids = set(_string_list(request.get("attached_memory_ids")))
    retrieved_paths = set(_string_list(request.get("retrieved_image_paths")))
    sources = raw_sources if isinstance(raw_sources, list) else []
    if not sources:
        has_legacy_text = any(
            str(data.get(key) or "").strip()
            for key in (
                "visual_evidence",
                "audit_action_hint",
                "audit_relative_direction_hint",
            )
        )
        return {
            "evidence_sources": [],
            "memory_evidence_used_count": 0,
            "current_only_evidence_count": 0,
            "ambiguous_evidence_count": 1 if has_legacy_text else 0,
            "invalid_evidence_source_count": 0,
            "no_evidence_count": 0 if has_legacy_text else 1,
        }

    normalized_sources: List[Dict[str, Any]] = []
    seen_identities = set()
    memory_count = 0
    current_count = 0
    invalid_count = 0
    for source in sources:
        if not isinstance(source, dict):
            invalid_count += 1
            continue
        normalized = dict(source)
        source_type = str(normalized.get("evidence_source_type") or "")
        identity = _evidence_source_identity(normalized)
        if not source_type or identity is None:
            invalid_count += 1
            normalized_sources.append(normalized)
            continue
        dedupe_key = (source_type, identity)
        if dedupe_key in seen_identities:
            invalid_count += 1
            normalized_sources.append(normalized)
            continue
        seen_identities.add(dedupe_key)
        if source_type == "memory":
            if _source_maps_to_attached_memory(normalized, attached_memory_ids, retrieved_paths):
                memory_count += 1
            else:
                invalid_count += 1
        elif source_type == "current":
            if _source_maps_to_current(normalized):
                current_count += 1
            else:
                invalid_count += 1
        else:
            invalid_count += 1
        normalized_sources.append(normalized)
    return {
        "evidence_sources": normalized_sources,
        "memory_evidence_used_count": memory_count,
        "current_only_evidence_count": current_count,
        "ambiguous_evidence_count": 0,
        "invalid_evidence_source_count": invalid_count,
        "no_evidence_count": 0
        if memory_count or current_count or invalid_count or normalized_sources
        else 1,
    }


def _evidence_source_identity(source: Dict[str, Any]) -> Any:
    for key in ("memory_id", "retrieved_image_path", "step_id", "readback_slot_id"):
        value = source.get(key)
        if value is not None and str(value) != "":
            return (key, str(value))
    return None


def _source_maps_to_attached_memory(
    source: Dict[str, Any],
    attached_memory_ids: set[str],
    retrieved_paths: set[str],
) -> bool:
    memory_id = str(source.get("memory_id") or "")
    if memory_id and memory_id in attached_memory_ids:
        return True
    image_path = str(source.get("retrieved_image_path") or "")
    if image_path and image_path in retrieved_paths:
        return True
    try:
        slot_id = int(source.get("readback_slot_id"))
    except (TypeError, ValueError):
        return False
    return 1 <= slot_id <= len(retrieved_paths)


def _source_maps_to_current(source: Dict[str, Any]) -> bool:
    try:
        return int(source.get("readback_slot_id")) == 0
    except (TypeError, ValueError):
        return bool(source.get("retrieved_image_path") == source.get("current_image_path"))


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _hit_value(hit: Any, name: str) -> Any:
    if isinstance(hit, dict):
        return hit.get(name)
    return getattr(hit, name, None)
