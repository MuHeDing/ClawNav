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
        "readback_confidence": 0.0,
        "retrieval_confidence": _float(request.get("retrieval_confidence"), 0.0),
        "verifier_confidence": 0.0,
        "visual_grounding_status": "unverified",
    }


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


def _hit_value(hit: Any, name: str) -> Any:
    if isinstance(hit, dict):
        return hit.get(name)
    return getattr(hit, name, None)
