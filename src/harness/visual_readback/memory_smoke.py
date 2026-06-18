from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.memory.spatial_memory_client import BaseSpatialMemoryClient
from harness.types import MemoryHit


class ImageBackedLocalMemoryClient(BaseSpatialMemoryClient):
    """Small in-memory client for Phase 0 image-path round-trip smoke tests."""

    def __init__(self, memory_source: str = "episode-local") -> None:
        super().__init__(memory_source=memory_source)
        self.records: List[Dict[str, Any]] = []

    def ingest_semantic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(payload)
        record.setdefault("memory_id", f"image-backed-local-{len(self.records)}")
        record.setdefault("memory_source", self.memory_source)
        self.records.append(record)
        return {"ok": True, "memory_id": record["memory_id"]}

    def query_semantic(
        self,
        text: str,
        n_results: int = 5,
        allowed_scopes: Optional[List[str]] = None,
        memory_namespace: str = "",
        memory_source: str = "",
    ) -> List[MemoryHit]:
        del text
        hits: List[MemoryHit] = []
        requested_source = memory_source or self.memory_source
        for record in self.records:
            record_namespace = str(record.get("memory_namespace") or "")
            record_source = str(record.get("memory_source") or self.memory_source)
            if memory_namespace and record_namespace != memory_namespace:
                continue
            if requested_source and record_source != requested_source:
                continue
            scope = str(record.get("memory_scope") or "episode")
            if allowed_scopes and scope not in allowed_scopes:
                continue
            hits.append(_memory_hit_from_record(record))
        return hits[:n_results]


def run_image_backed_memory_smoke(
    client: BaseSpatialMemoryClient,
    image_path: Path,
    memory_namespace: str,
    query_text: str = "visual readback smoke",
) -> Dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return {
            "passed": False,
            "reason": f"image_path does not exist: {path}",
            "written_image_path": str(path),
            "retrieved_image_path": "",
            "retrieved_memory_id": "",
        }

    payload = {
        "memory_id": "visual-readback-smoke",
        "memory_type": "semantic_frame",
        "name": "visual readback smoke keyframe",
        "image_path": str(path),
        "retrieval_text": query_text,
        "evidence_text": "visual readback smoke keyframe",
        "memory_scope": "episode",
        "memory_namespace": memory_namespace,
        "memory_source": client.memory_source,
    }
    client.ingest_semantic(payload)
    hits = client.query_semantic(
        query_text,
        n_results=3,
        allowed_scopes=["episode"],
        memory_namespace=memory_namespace,
        memory_source=client.memory_source,
    )
    for hit in hits:
        retrieved_path = str(hit.image_path or "")
        if retrieved_path and Path(retrieved_path).exists():
            return {
                "passed": True,
                "reason": "ok",
                "written_image_path": str(path),
                "retrieved_image_path": retrieved_path,
                "retrieved_memory_id": hit.memory_id,
            }
    return {
        "passed": False,
        "reason": "query did not return a readable image_path",
        "written_image_path": str(path),
        "retrieved_image_path": "",
        "retrieved_memory_id": "",
    }


def _memory_hit_from_record(record: Dict[str, Any]) -> MemoryHit:
    return MemoryHit(
        memory_id=str(record.get("memory_id") or ""),
        memory_type=str(record.get("memory_type") or "semantic_frame"),
        name=str(record.get("name") or record.get("retrieval_text") or "memory"),
        confidence=float(record.get("confidence") or 1.0),
        target_pose=record.get("target_pose"),
        evidence_text=str(record.get("evidence_text") or record.get("retrieval_text") or ""),
        image_path=str(record.get("image_path") or ""),
        note=str(record.get("note") or ""),
        timestamp=record.get("timestamp"),
        memory_source=str(record.get("memory_source") or "episode-local"),
        metadata={
            "memory_scope": str(record.get("memory_scope") or "episode"),
            "memory_namespace": str(record.get("memory_namespace") or ""),
        },
    )
