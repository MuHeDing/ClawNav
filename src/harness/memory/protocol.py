from typing import Any, Dict, Optional


SPATIAL_MEMORY_PROTOCOL_VERSION = "phase2.spatial_memory.v1"

ORACLE_MEMORY_KEYS = {
    "distance_to_goal",
    "success",
    "SPL",
    "spl",
    "softspl",
    "oracle_success",
    "oracle_path",
    "oracle_shortest_path",
    "oracle_shortest_path_action",
    "oracle_action",
    "future_observations",
    "future_trajectory_frames",
}


def build_query_payload(
    query_type: str,
    text: Optional[str] = None,
    pose: Optional[Dict[str, Any]] = None,
    n_results: int = 5,
    memory_source: str = "episode-local",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "protocol_version": SPATIAL_MEMORY_PROTOCOL_VERSION,
        "query_type": query_type,
        "n_results": n_results,
        "memory_source": memory_source,
    }
    if text is not None:
        payload["text"] = text
    if pose is not None:
        payload["pose"] = pose
    return payload


def normalize_memory_result(
    item: Dict[str, Any],
    memory_source: str,
) -> Dict[str, Any]:
    evidence = item.get("evidence") or {}
    metadata = dict(item.get("metadata") or {})
    metadata.update(
        {
            key: value
            for key, value in item.items()
            if key
            not in {
                "evidence",
                "metadata",
                "id",
                "memory_id",
                "memory_type",
                "name",
                "text",
                "confidence",
                "target_pose",
                "evidence_text",
                "image_path",
                "note",
                "timestamp",
                "source",
                "memory_source",
            }
        }
    )
    return {
        "memory_id": str(item.get("id") or item.get("memory_id") or ""),
        "memory_type": str(item.get("memory_type") or "semantic"),
        "name": str(item.get("name") or item.get("text") or ""),
        "confidence": float(item.get("confidence", 0.0)),
        "target_pose": item.get("target_pose"),
        "evidence_text": str(
            item.get("evidence_text")
            or evidence.get("text")
            or evidence.get("note")
            or ""
        ),
        "image_path": item.get("image_path") or evidence.get("image_path"),
        "note": str(item.get("note") or evidence.get("note") or ""),
        "timestamp": item.get("timestamp"),
        "memory_source": memory_source or item.get("memory_source") or item.get("source") or "episode-local",
        "metadata": strip_oracle_fields(metadata),
    }


def strip_oracle_fields(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in ORACLE_MEMORY_KEYS
    }
