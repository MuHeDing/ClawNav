from typing import Any, Dict, List, Optional

from harness.config import HarnessConfig
from harness.memory.spatial_memory_client import BaseSpatialMemoryClient
from harness.types import MemoryHit, MemoryRecallResult


class MemoryManager:
    def __init__(
        self,
        client: BaseSpatialMemoryClient,
        config: Optional[HarnessConfig] = None,
    ) -> None:
        self.client = client
        self.config = config or HarnessConfig()
        self.last_recall_step: Optional[int] = None

    def should_recall(self, step_id: int, reason: str = "") -> bool:
        if reason in {"initial", "risky_stop", "stuck", "replan"}:
            return True
        if self.last_recall_step is None:
            return True
        return (step_id - self.last_recall_step) >= self.config.recall_interval_steps

    def mark_recalled(self, step_id: int) -> None:
        self.last_recall_step = step_id

    def recall(
        self,
        text: str,
        step_id: int,
        reason: str = "",
        n_results: int = 5,
    ) -> MemoryRecallResult:
        hits = self.client.query_semantic(text, n_results=n_results)
        self.mark_recalled(step_id)
        return MemoryRecallResult(
            hits=hits,
            query=text,
            backend=self.config.memory_backend,
            policy_context=self._build_policy_context(hits),
            control_context=self._build_control_context(hits, reason),
            executor_context=self._build_executor_context(hits),
        )

    def propose_write(
        self,
        step_id: int,
        image_path: Optional[str] = None,
        note: str = "",
        write_type: str = "episodic_keyframe",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "should_write": True,
            "write_type": write_type,
            "step_id": step_id,
            "image_path": image_path,
            "note": note,
            "memory_source": self.config.memory_source,
            "metadata": metadata or {},
        }

    def _build_policy_context(self, hits: List[MemoryHit]) -> Dict[str, Any]:
        texts = []
        image_paths = []
        for hit in hits:
            evidence = hit.evidence_text or hit.note or hit.name
            if evidence:
                texts.append(f"- {hit.name}: {evidence} (confidence={hit.confidence:.2f})")
            if hit.image_path:
                image_paths.append(hit.image_path)
        memory_context_text = "\n".join(texts)
        if len(memory_context_text) > self.config.max_prompt_context_chars:
            memory_context_text = memory_context_text[: self.config.max_prompt_context_chars]
        return {
            "memory_context_text": memory_context_text,
            "memory_images": image_paths[: self.config.max_memory_images],
        }

    def _build_control_context(self, hits: List[MemoryHit], reason: str) -> Dict[str, Any]:
        return {
            "hits": hits,
            "reason": reason,
            "best_hit": hits[0] if hits else None,
            "confidence": hits[0].confidence if hits else 0.0,
        }

    def _build_executor_context(self, hits: List[MemoryHit]) -> Dict[str, Any]:
        target_poses = [hit.target_pose for hit in hits if hit.target_pose is not None]
        return {
            "target_poses": target_poses,
            "landmark_names": [hit.name for hit in hits],
        }
