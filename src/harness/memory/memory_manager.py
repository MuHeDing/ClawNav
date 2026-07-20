from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set

from harness.config import HarnessConfig
from harness.memory.episode_visual_store import EpisodeVisualMemoryStore
from harness.memory.spatial_memory_client import BaseSpatialMemoryClient
from harness.types import MemoryHit, MemoryRecallResult


MAX_POLICY_EVIDENCE_CHARS = 180
MAX_POLICY_EVIDENCE_SEGMENTS = 2
META_EVIDENCE_PREFIXES = (
    "the user wants",
    "i need to",
    "drafting the description",
    "based on the visual evidence",
    "on the visual evidence",
    "here is a description",
)
NAVIGATION_TERMS = {
    "arch",
    "archway",
    "corridor",
    "door",
    "doorway",
    "forward",
    "hallway",
    "left",
    "obstacle",
    "path",
    "right",
    "route",
    "turn",
}


class MemoryManager:
    def __init__(
        self,
        client: BaseSpatialMemoryClient,
        config: Optional[HarnessConfig] = None,
        episode_visual_store: Optional[EpisodeVisualMemoryStore] = None,
    ) -> None:
        self.client = client
        self.config = config or HarnessConfig()
        self.episode_visual_store = episode_visual_store
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
        active_stage_id: str = "",
        expected_landmarks: Optional[List[str]] = None,
        trigger_reasons: Optional[List[str]] = None,
        use_episode_visual_store: Optional[bool] = None,
    ) -> MemoryRecallResult:
        query_text = self._build_query_text(
            text=text,
            active_subgoal=active_subgoal,
            visual_observation=visual_observation,
            planner_reason=planner_reason,
            critic_signal=critic_signal,
        )
        external_hits = self.client.query_semantic(
            query_text,
            n_results=n_results,
            allowed_scopes=allowed_scopes,
            memory_namespace=memory_namespace,
        )
        external_hits = self._filter_hits(
            external_hits,
            allowed_scopes=allowed_scopes,
            memory_namespace=memory_namespace,
        )
        external_hits = self._rerank_hits(
            external_hits, query_text=query_text, step_id=step_id
        )
        include_episode_store = (
            self.config.staged_visual_memory_enabled
            if use_episode_visual_store is None
            else bool(use_episode_visual_store)
        )
        episode_query_status = "not_attempted"
        episode_hits: List[MemoryHit] = []
        if include_episode_store and self.episode_visual_store is not None:
            episode_result = self.episode_visual_store.query_semantic(
                query_text,
                active_stage_id=active_stage_id,
                expected_landmarks=expected_landmarks or [],
                trigger_reasons=trigger_reasons or ([reason] if reason else []),
                limit=min(n_results, self.config.staged_semantic_query_max_records),
            )
            episode_query_status = episode_result.status.value
            episode_hits = episode_result.to_memory_hits()
        hits = self._merge_hits(episode_hits, external_hits)
        self.mark_recalled(step_id)
        visual_memory_hit_count = sum(1 for hit in hits if hit.image_path)
        return MemoryRecallResult(
            hits=hits,
            query=query_text,
            backend=self.config.memory_backend,
            policy_context=self._build_policy_context(hits),
            control_context=self._build_control_context(
                hits,
                reason,
                episode_store_query_status=episode_query_status,
                episode_hit_count=len(episode_hits),
                external_hit_count=len(external_hits),
                visual_memory_hit_count=visual_memory_hit_count,
            ),
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
                compact_evidence = self._compact_policy_evidence(evidence)
                if compact_evidence:
                    hit_name = self._policy_hit_name(hit)
                    texts.append(
                        f"- {hit_name}: {compact_evidence} "
                        f"(confidence={hit.confidence:.2f})"
                    )
            if hit.image_path:
                image_paths.append(hit.image_path)
        memory_context_text = "\n".join(texts)
        if len(memory_context_text) > self.config.max_prompt_context_chars:
            memory_context_text = memory_context_text[: self.config.max_prompt_context_chars]
        return {
            "memory_context_text": memory_context_text,
            "memory_images": image_paths[: self.config.max_memory_images],
        }

    def _build_control_context(
        self,
        hits: List[MemoryHit],
        reason: str,
        *,
        episode_store_query_status: str = "not_attempted",
        episode_hit_count: int = 0,
        external_hit_count: int = 0,
        visual_memory_hit_count: int = 0,
    ) -> Dict[str, Any]:
        best_hit = hits[0] if hits else None
        return {
            "hits": hits,
            "reason": reason,
            "best_hit": best_hit,
            "confidence": best_hit.confidence if best_hit else 0.0,
            "best_landmark": self._best_landmark(best_hit),
            "recall_confidence": best_hit.confidence if best_hit else 0.0,
            "episode_store_query_status": episode_store_query_status,
            "episode_hit_count": episode_hit_count,
            "external_hit_count": external_hit_count,
            "visual_memory_hit_count": visual_memory_hit_count,
            "image_backed_recall": visual_memory_hit_count > 0,
        }

    def _merge_hits(
        self,
        episode_hits: List[MemoryHit],
        external_hits: List[MemoryHit],
    ) -> List[MemoryHit]:
        merged: List[MemoryHit] = []
        path_to_index: Dict[str, int] = {}
        for hit in [*episode_hits, *external_hits]:
            canonical_path = self._canonical_hit_path(hit.image_path)
            if canonical_path and canonical_path in path_to_index:
                index = path_to_index[canonical_path]
                existing = merged[index]
                metadata = dict(existing.metadata or {})
                incoming = dict(hit.metadata or {})
                for key in ("image_roles", "provenance", "stage_ids"):
                    values = list(metadata.get(key) or [])
                    for value in incoming.get(key) or []:
                        if value not in values:
                            values.append(value)
                    if values:
                        metadata[key] = values
                merged_ids = list(metadata.get("merged_memory_ids") or [])
                for memory_id in (existing.memory_id, hit.memory_id):
                    if memory_id and memory_id not in merged_ids:
                        merged_ids.append(memory_id)
                metadata["merged_memory_ids"] = merged_ids
                merged[index] = replace(existing, metadata=metadata)
                continue
            if canonical_path:
                path_to_index[canonical_path] = len(merged)
            merged.append(hit)
        return merged

    @staticmethod
    def _canonical_hit_path(image_path: Optional[str]) -> str:
        if not isinstance(image_path, str) or not image_path:
            return ""
        return str(Path(image_path).expanduser().resolve(strict=False))

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
            if allowed and (not scope or scope not in allowed):
                continue
            if memory_namespace and (not namespace or namespace != memory_namespace):
                continue
            filtered.append(hit)
        return filtered

    def _rerank_hits(
        self,
        hits: List[MemoryHit],
        query_text: str,
        step_id: int,
    ) -> List[MemoryHit]:
        query_terms = self._term_set(query_text)
        return sorted(
            hits,
            key=lambda hit: self._rerank_score(hit, query_terms, step_id),
            reverse=True,
        )

    def _rerank_score(
        self,
        hit: MemoryHit,
        query_terms: Set[str],
        step_id: int,
    ) -> float:
        metadata = hit.metadata or {}
        overlap = len(query_terms.intersection(self._hit_terms(hit)))
        confidence = float(hit.confidence or 0.0)
        recency = self._recency_score(metadata.get("step_id"), step_id)
        scope_bonus = {
            "episode": 0.3,
            "scene": 0.15,
            "task": 0.05,
        }.get(str(metadata.get("memory_scope") or ""), 0.0)
        return confidence + (0.35 * overlap) + (0.2 * recency) + scope_bonus

    def _recency_score(self, hit_step: Any, step_id: int) -> float:
        if not isinstance(hit_step, int) or step_id < hit_step:
            return 0.0
        return max(0.0, 1.0 - min(step_id - hit_step, 50) / 50.0)

    def _hit_terms(self, hit: MemoryHit) -> Set[str]:
        metadata = hit.metadata or {}
        terms = set()
        for value in (
            hit.name,
            hit.evidence_text,
            hit.note,
            metadata.get("caption", ""),
            metadata.get("visual_observation", ""),
            metadata.get("place_category", ""),
            metadata.get("navigation_relevance", ""),
        ):
            terms.update(self._term_set(str(value or "")))
        for key in ("objects", "landmarks", "spatial_cues"):
            values = metadata.get(key)
            if isinstance(values, list):
                for value in values:
                    terms.update(self._term_set(str(value)))
        return terms

    def _term_set(self, text: str) -> Set[str]:
        return {
            term
            for term in re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if len(term) > 2
        }

    def _visual_evidence_text(self, hit: MemoryHit) -> str:
        metadata = hit.metadata or {}
        parts = []
        for key in (
            "navigation_relevance",
            "visual_observation",
            "caption",
            "place_category",
        ):
            value = metadata.get(key)
            if value:
                parts.append(str(value))
        for key in ("objects", "landmarks", "spatial_cues"):
            values = metadata.get(key)
            if isinstance(values, list) and values:
                parts.append(", ".join(str(value) for value in values if str(value)))
        return " | ".join(parts)

    def _compact_policy_evidence(self, evidence: str) -> str:
        segments: List[str] = []
        for segment in self._evidence_segments(evidence):
            cleaned = self._clean_policy_segment(segment)
            if not cleaned:
                continue
            if self._looks_like_meta_reasoning(cleaned):
                continue
            segments.append(cleaned)
            if len(segments) >= MAX_POLICY_EVIDENCE_SEGMENTS:
                break
        compact = "; ".join(segments)
        if not compact:
            compact = self._clean_policy_segment(evidence)
        if len(compact) > MAX_POLICY_EVIDENCE_CHARS:
            compact = compact[:MAX_POLICY_EVIDENCE_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:.")
        return compact

    def _policy_hit_name(self, hit: MemoryHit) -> str:
        for candidate in (
            self._best_landmark(hit),
            hit.name,
            hit.memory_type,
            hit.memory_id,
        ):
            cleaned = self._clean_policy_segment(candidate)
            if not cleaned:
                continue
            if len(cleaned) > 80:
                continue
            if self._looks_like_meta_reasoning(cleaned):
                continue
            return cleaned
        return "memory"

    def _evidence_segments(self, evidence: str) -> List[str]:
        parts = []
        for line in str(evidence).splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"^[*\-\d.)\s]+", "", cleaned).strip()
            cleaned = cleaned.strip("|")
            if cleaned:
                parts.extend(
                    segment.strip()
                    for segment in re.split(r"(?<=[.!?])\s+|\s+\|\s+", cleaned)
                    if segment.strip()
                )
        nav_segments = [
            segment
            for segment in parts
            if self._contains_navigation_term(segment)
            and not self._looks_like_meta_reasoning(segment)
            and not self._looks_like_heading(segment)
        ]
        if nav_segments:
            return nav_segments
        return [
            segment
            for segment in parts
            if not self._looks_like_meta_reasoning(segment)
            and not self._looks_like_heading(segment)
        ]

    def _clean_policy_segment(self, segment: str) -> str:
        cleaned = re.sub(r"\*\*", "", str(segment))
        cleaned = re.sub(r"`", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;")
        return cleaned

    def _looks_like_meta_reasoning(self, segment: str) -> bool:
        lowered = segment.lower().strip()
        return any(lowered.startswith(prefix) for prefix in META_EVIDENCE_PREFIXES)

    def _looks_like_heading(self, segment: str) -> bool:
        cleaned = self._clean_policy_segment(segment)
        words = self._term_set(cleaned)
        if len(words) > 5:
            return False
        if any(term in words for term in NAVIGATION_TERMS):
            return False
        return ":" not in cleaned and not cleaned.endswith((".", "!", "?"))

    def _contains_navigation_term(self, segment: str) -> bool:
        terms = self._term_set(segment)
        return bool(terms.intersection(NAVIGATION_TERMS))

    def _best_landmark(self, hit: Optional[MemoryHit]) -> str:
        if hit is None:
            return ""
        landmarks = (hit.metadata or {}).get("landmarks")
        if isinstance(landmarks, list) and landmarks:
            return str(landmarks[0])
        return hit.name
