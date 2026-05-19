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
        active_subgoal: str = "",
        visual_observation: str = "",
        planner_reason: str = "",
        critic_signal: str = "",
        allowed_scopes: Optional[List[str]] = None,
        memory_namespace: str = "",
    ) -> MemoryRecallResult:
        query_text = self._build_query_text(
            text=text,
            active_subgoal=active_subgoal,
            visual_observation=visual_observation,
            planner_reason=planner_reason,
            critic_signal=critic_signal,
        )
        hits = self.client.query_semantic(
            query_text,
            n_results=n_results,
            allowed_scopes=allowed_scopes,
            memory_namespace=memory_namespace,
        )
        hits = self._filter_hits(
            hits,
            allowed_scopes=allowed_scopes,
            memory_namespace=memory_namespace,
        )
        self.mark_recalled(step_id)
        return MemoryRecallResult(
            hits=hits,
            query=query_text,
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
            visual_evidence = self._visual_evidence_text(hit)
            evidence = visual_evidence or hit.evidence_text or hit.note or hit.name
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
        best_hit = hits[0] if hits else None
        return {
            "hits": hits,
            "reason": reason,
            "best_hit": best_hit,
            "confidence": best_hit.confidence if best_hit else 0.0,
            "best_landmark": self._best_landmark(best_hit),
            "recall_confidence": best_hit.confidence if best_hit else 0.0,
        }

    def _build_executor_context(self, hits: List[MemoryHit]) -> Dict[str, Any]:
        target_poses = [hit.target_pose for hit in hits if hit.target_pose is not None]
        topological_anchor = ""
        if hits:
            topological_anchor = self._best_landmark(hits[0]) or hits[0].name
        return {
            "target_poses": target_poses,
            "landmark_names": [hit.name for hit in hits],
            "topological_anchor": topological_anchor,
        }

    def _build_query_text(
        self,
        text: str,
        active_subgoal: str = "",
        visual_observation: str = "",
        planner_reason: str = "",
        critic_signal: str = "",
    ) -> str:
        parts = [
            text,
            active_subgoal,
            visual_observation,
            planner_reason,
            critic_signal,
        ]
        return "\n".join(str(part) for part in parts if part)

    def _filter_hits(
        self,
        hits: List[MemoryHit],
        allowed_scopes: Optional[List[str]] = None,
        memory_namespace: str = "",
    ) -> List[MemoryHit]:
        if not allowed_scopes and not memory_namespace:
            return hits
        allowed = set(allowed_scopes or [])
        filtered: List[MemoryHit] = []
        for hit in hits:
            metadata = hit.metadata or {}
            scope = metadata.get("memory_scope")
            namespace = metadata.get("memory_namespace")
            if allowed and scope not in allowed:
                continue
            if memory_namespace and namespace != memory_namespace:
                continue
            filtered.append(hit)
        return filtered

    def _visual_evidence_text(self, hit: MemoryHit) -> str:
        metadata = hit.metadata or {}
        parts = []
        for key in (
            "caption",
            "visual_observation",
            "place_category",
            "navigation_relevance",
        ):
            value = metadata.get(key)
            if value:
                parts.append(str(value))
        for key in ("objects", "landmarks", "spatial_cues"):
            values = metadata.get(key)
            if isinstance(values, list) and values:
                parts.append(", ".join(str(value) for value in values if str(value)))
        return " | ".join(parts)

    def _best_landmark(self, hit: Optional[MemoryHit]) -> str:
        if hit is None:
            return ""
        landmarks = (hit.metadata or {}).get("landmarks")
        if isinstance(landmarks, list) and landmarks:
            return str(landmarks[0])
        return hit.name
