from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from harness.openclaw.instruction_stages import assert_no_forbidden_oracle_keys
from harness.types import MemoryHit


class VisualMemoryOperationStatus(str, Enum):
    SELECTED = "selected"
    EMPTY = "empty"
    FAILED = "failed"
    DISABLED_ABLATION = "disabled_ablation"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class EpisodeVisualMemoryRecord:
    memory_id: str
    image_path: str
    scene_id: str
    episode_id: str
    step_id: int
    stage_id: str
    stage_ids: Tuple[str, ...]
    visual_summary: str
    landmarks: Tuple[str, ...]
    action_before_capture: str
    action_after_capture: str
    odometry_snapshot: Mapping[str, Any]
    image_roles: Tuple[str, ...]
    provenance: Tuple[str, ...]
    importance: float = 0.0

    def to_memory_hit(self) -> MemoryHit:
        return MemoryHit(
            memory_id=self.memory_id,
            memory_type="episode_visual",
            name=self.landmarks[0] if self.landmarks else "visual observation",
            confidence=max(0.0, min(1.0, self.importance)),
            evidence_text=self.visual_summary,
            image_path=self.image_path,
            note=self.visual_summary,
            memory_source="episode-visual-store",
            metadata={
                "scene_id": self.scene_id,
                "episode_id": self.episode_id,
                "step_id": self.step_id,
                "stage_id": self.stage_id,
                "stage_ids": list(self.stage_ids),
                "landmarks": list(self.landmarks),
                "image_roles": list(self.image_roles),
                "provenance": list(self.provenance),
                "action_before_capture": self.action_before_capture,
                "action_after_capture": self.action_after_capture,
                "odometry_snapshot": dict(self.odometry_snapshot),
                "memory_scope": "episode",
                "memory_namespace": f"episode:{self.scene_id}:{self.episode_id}",
                "image_backed": True,
            },
        )


@dataclass(frozen=True)
class VisualMemoryOperationResult:
    operation: str
    status: VisualMemoryOperationStatus
    records: Tuple[EpisodeVisualMemoryRecord, ...] = ()
    attempted: bool = True
    reason: str = ""

    @property
    def image_backed_hit_count(self) -> int:
        return sum(1 for record in self.records if record.image_path)

    def to_memory_hits(self) -> list[MemoryHit]:
        return [record.to_memory_hit() for record in self.records]


class EpisodeVisualMemoryStore:
    def __init__(self, capacity: int = 64) -> None:
        if int(capacity) <= 0:
            raise ValueError("episode visual memory capacity must be positive")
        self.capacity = int(capacity)
        self._scene_id = ""
        self._episode_id = ""
        self._records: list[EpisodeVisualMemoryRecord] = []
        self._next_memory_index = 1

    @property
    def episode_key(self) -> str:
        if not self._scene_id and not self._episode_id:
            return ""
        return f"{self._scene_id}:{self._episode_id}"

    @property
    def records(self) -> Tuple[EpisodeVisualMemoryRecord, ...]:
        return tuple(self._records)

    def start_episode(self, scene_id: str, episode_id: str) -> None:
        normalized_scene = str(scene_id or "")
        normalized_episode = str(episode_id or "")
        if (
            normalized_scene == self._scene_id
            and normalized_episode == self._episode_id
        ):
            return
        self.reset_episode()
        self._scene_id = normalized_scene
        self._episode_id = normalized_episode

    def reset_episode(self) -> None:
        self._scene_id = ""
        self._episode_id = ""
        self._records.clear()
        self._next_memory_index = 1

    def add_observation(
        self,
        *,
        image_path: str,
        step_id: int,
        stage_id: str = "",
        visual_summary: str = "",
        landmarks: Optional[Sequence[str]] = None,
        action_before_capture: str = "",
        action_after_capture: str = "",
        odometry_snapshot: Optional[Mapping[str, Any]] = None,
        image_roles: Optional[Sequence[str]] = None,
        provenance: Optional[Sequence[str]] = None,
        importance: float = 0.0,
    ) -> EpisodeVisualMemoryRecord:
        if not self.episode_key:
            raise ValueError("start_episode must be called before adding observations")
        canonical_path = self._canonical_path(image_path)
        safe_odometry = dict(odometry_snapshot or {})
        assert_no_forbidden_oracle_keys(safe_odometry)
        normalized_landmarks = self._bounded_values(landmarks or (), 16, 96)
        normalized_roles = self._bounded_values(image_roles or (), 16, 64)
        normalized_provenance = self._bounded_values(provenance or (), 24, 120)
        normalized_stage_id = self._bounded_text(stage_id, 64, allow_empty=True)
        normalized_summary = self._bounded_text(
            visual_summary, 512, allow_empty=True
        )
        normalized_before = self._bounded_text(
            action_before_capture, 64, allow_empty=True
        )
        normalized_after = self._bounded_text(
            action_after_capture, 64, allow_empty=True
        )
        try:
            normalized_step = max(0, int(step_id))
            normalized_importance = max(0.0, min(1.0, float(importance)))
        except (TypeError, ValueError) as exc:
            raise ValueError("step_id and importance must be numeric") from exc

        existing_index = next(
            (
                index
                for index, record in enumerate(self._records)
                if record.image_path == canonical_path
            ),
            None,
        )
        if existing_index is not None:
            existing = self._records[existing_index]
            merged = replace(
                existing,
                step_id=max(existing.step_id, normalized_step),
                stage_id=normalized_stage_id or existing.stage_id,
                stage_ids=self._merge_values(
                    existing.stage_ids,
                    (normalized_stage_id,) if normalized_stage_id else (),
                ),
                visual_summary=normalized_summary or existing.visual_summary,
                landmarks=self._merge_values(existing.landmarks, normalized_landmarks),
                action_before_capture=(
                    normalized_before or existing.action_before_capture
                ),
                action_after_capture=normalized_after or existing.action_after_capture,
                odometry_snapshot=safe_odometry or existing.odometry_snapshot,
                image_roles=self._merge_values(existing.image_roles, normalized_roles),
                provenance=self._merge_values(
                    existing.provenance, normalized_provenance
                ),
                importance=max(existing.importance, normalized_importance),
            )
            self._records[existing_index] = merged
            return merged

        memory_id = f"mem_{self._next_memory_index:06d}"
        self._next_memory_index += 1
        record = EpisodeVisualMemoryRecord(
            memory_id=memory_id,
            image_path=canonical_path,
            scene_id=self._scene_id,
            episode_id=self._episode_id,
            step_id=normalized_step,
            stage_id=normalized_stage_id,
            stage_ids=(normalized_stage_id,) if normalized_stage_id else (),
            visual_summary=normalized_summary,
            landmarks=normalized_landmarks,
            action_before_capture=normalized_before,
            action_after_capture=normalized_after,
            odometry_snapshot=safe_odometry,
            image_roles=normalized_roles,
            provenance=normalized_provenance,
            importance=normalized_importance,
        )
        self._records.append(record)
        del self._records[: max(0, len(self._records) - self.capacity)]
        return record

    def record_action_after_capture(self, step_id: int, action_text: str) -> int:
        normalized_action = self._bounded_text(
            action_text, 64, allow_empty=False
        )
        normalized_step = int(step_id)
        updated = 0
        for index, record in enumerate(self._records):
            if record.step_id != normalized_step:
                continue
            self._records[index] = replace(
                record,
                action_after_capture=normalized_action,
            )
            updated += 1
        return updated

    def select_registry_evidence(
        self,
        *,
        active_stage_id: str = "",
        expected_landmarks: Optional[Sequence[str]] = None,
        trigger_reasons: Optional[Sequence[str]] = None,
        limit: int = 2,
    ) -> VisualMemoryOperationResult:
        return self._select(
            operation="registry",
            query="",
            active_stage_id=active_stage_id,
            expected_landmarks=expected_landmarks or (),
            trigger_reasons=trigger_reasons or (),
            limit=limit,
        )

    def query_semantic(
        self,
        query: str,
        *,
        active_stage_id: str = "",
        expected_landmarks: Optional[Sequence[str]] = None,
        trigger_reasons: Optional[Sequence[str]] = None,
        limit: int = 4,
    ) -> VisualMemoryOperationResult:
        assert_no_forbidden_oracle_keys(
            {
                "active_stage_id": active_stage_id,
                "expected_landmarks": list(expected_landmarks or ()),
                "trigger_reasons": list(trigger_reasons or ()),
            }
        )
        return self._select(
            operation="semantic_query",
            query=str(query or ""),
            active_stage_id=active_stage_id,
            expected_landmarks=expected_landmarks or (),
            trigger_reasons=trigger_reasons or (),
            limit=limit,
        )

    def _select(
        self,
        *,
        operation: str,
        query: str,
        active_stage_id: str,
        expected_landmarks: Sequence[str],
        trigger_reasons: Sequence[str],
        limit: int,
    ) -> VisualMemoryOperationResult:
        bounded_limit = max(0, int(limit))
        eligible = [
            record
            for record in self._records
            if record.image_path
            and (
                record.visual_summary
                or record.landmarks
                or set(record.image_roles) - {"current"}
            )
        ]
        query_terms = self._terms(query)
        landmark_terms = self._terms(" ".join(str(v) for v in expected_landmarks))
        reasons = {str(reason) for reason in trigger_reasons}
        ranked = sorted(
            eligible,
            key=lambda record: (
                self._score(
                    record,
                    query_terms=query_terms,
                    landmark_terms=landmark_terms,
                    active_stage_id=str(active_stage_id or ""),
                    trigger_reasons=reasons,
                ),
                record.step_id,
                record.memory_id,
            ),
            reverse=True,
        )[:bounded_limit]
        status = (
            VisualMemoryOperationStatus.SELECTED
            if ranked
            else VisualMemoryOperationStatus.EMPTY
        )
        return VisualMemoryOperationResult(
            operation=operation,
            status=status,
            records=tuple(ranked),
        )

    def _score(
        self,
        record: EpisodeVisualMemoryRecord,
        *,
        query_terms: set[str],
        landmark_terms: set[str],
        active_stage_id: str,
        trigger_reasons: set[str],
    ) -> float:
        record_terms = self._terms(
            " ".join((record.visual_summary, *record.landmarks, *record.image_roles))
        )
        score = record.importance * 2.0
        if active_stage_id and active_stage_id in record.stage_ids:
            score += 100.0
        score += 20.0 * len(landmark_terms & record_terms)
        score += 2.0 * len(query_terms & record_terms)
        roles = set(record.image_roles)
        if trigger_reasons & {"stage_completion_candidate", "stop_candidate"}:
            if roles & {"confirmed_landmark", "target_candidate", "keyframe"}:
                score += 8.0
        if trigger_reasons & {"no_progress", "turn_loop"}:
            if roles & {
                "stuck_before_keyframe",
                "center_scan",
                "left_scan",
                "right_scan",
                "keyframe",
            }:
                score += 8.0
        if "stage_entry" in trigger_reasons and "keyframe" in roles:
            score += 4.0
        return score

    @staticmethod
    def _canonical_path(image_path: str) -> str:
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("image_path must be a non-empty string")
        return str(Path(image_path).expanduser().resolve(strict=False))

    @staticmethod
    def _bounded_text(value: Any, maximum: int, *, allow_empty: bool) -> str:
        if not isinstance(value, str):
            raise ValueError("visual memory text fields must be strings")
        normalized = value.strip()
        if not normalized and not allow_empty:
            raise ValueError("visual memory text field must not be empty")
        if len(normalized) > maximum:
            raise ValueError("visual memory text field exceeds limit")
        return normalized

    @classmethod
    def _bounded_values(
        cls, values: Iterable[Any], maximum_items: int, maximum_chars: int
    ) -> Tuple[str, ...]:
        if isinstance(values, (str, bytes, bytearray)):
            raise ValueError("visual memory list fields must be sequences")
        raw = list(values)
        if len(raw) > maximum_items:
            raise ValueError("visual memory list field exceeds item limit")
        normalized = tuple(
            cls._bounded_text(value, maximum_chars, allow_empty=False)
            for value in raw
        )
        return cls._merge_values((), normalized)

    @staticmethod
    def _merge_values(left: Sequence[str], right: Sequence[str]) -> Tuple[str, ...]:
        merged: list[str] = []
        seen: set[str] = set()
        for value in (*left, *right):
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
        return tuple(merged)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-zA-Z0-9_]+", str(text).lower())
            if len(term) > 2
        }
