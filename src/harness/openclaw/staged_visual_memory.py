from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from harness.openclaw.instruction_stages import InstructionStage


class StagedMemoryTrigger(str, Enum):
    STAGE_ENTRY = "stage_entry"
    STAGE_COMPLETION_CANDIDATE = "stage_completion_candidate"
    NO_PROGRESS = "no_progress"
    TURN_LOOP = "turn_loop"
    STOP_CANDIDATE = "stop_candidate"


_TRIGGER_PRIORITY = {
    StagedMemoryTrigger.STOP_CANDIDATE: 0,
    StagedMemoryTrigger.STAGE_COMPLETION_CANDIDATE: 1,
    StagedMemoryTrigger.NO_PROGRESS: 2,
    StagedMemoryTrigger.TURN_LOOP: 3,
    StagedMemoryTrigger.STAGE_ENTRY: 4,
}


@dataclass
class StagedMemoryEvent:
    event_id: str
    episode_key: str
    step_id: int
    ordinal: int
    selected: bool = True
    status: str = "selected"
    _reasons: set[StagedMemoryTrigger] = field(default_factory=set)
    _operation_reasons: Tuple[StagedMemoryTrigger, ...] = ()
    operations: Optional["ForcedMemoryOperationsResult"] = None
    requery_performed: bool = False
    query_text: str = ""
    candidate_action_before: str = ""
    candidate_action_after: str = ""
    stage_candidate_before: bool = False
    stage_candidate_after: bool = False
    provider_call_ids: list[str] = field(default_factory=list)

    @property
    def trigger_reasons(self) -> Tuple[StagedMemoryTrigger, ...]:
        return tuple(sorted(self._reasons, key=_TRIGGER_PRIORITY.__getitem__))

    def add_reasons(self, reasons: Iterable[StagedMemoryTrigger]) -> None:
        self._reasons.update(reasons)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "memory_event_id": self.event_id,
            "step_id": self.step_id,
            "trigger_reasons": [reason.value for reason in self.trigger_reasons],
            "selected": self.selected,
            "status": self.status,
            "requery_performed": self.requery_performed,
            "query_text": self.query_text,
            "provider_call_ids": list(self.provider_call_ids),
            "candidate_action_before": self.candidate_action_before,
            "candidate_action_after": self.candidate_action_after,
            "stage_candidate_before": self.stage_candidate_before,
            "stage_candidate_after": self.stage_candidate_after,
        }
        if self.operations is not None:
            payload.update(self.operations.to_dict())
        return payload


class StagedVisualMemoryCoordinator:
    def __init__(self, event_cap: int = 64, recovery_retrigger_steps: int = 3) -> None:
        if int(event_cap) <= 0 or int(recovery_retrigger_steps) <= 0:
            raise ValueError("event cap and recovery retrigger steps must be positive")
        self.event_cap = int(event_cap)
        self.recovery_retrigger_steps = int(recovery_retrigger_steps)
        self._episodes: Dict[str, Dict[str, Any]] = {}

    def reset(self) -> None:
        self._episodes.clear()

    def observe(
        self,
        episode_key: str,
        step_id: int,
        *,
        stage_entry: bool = False,
        stage_completion_candidate: bool = False,
        no_progress: bool = False,
        turn_loop: bool = False,
        stop_candidate: bool = False,
    ) -> Optional[StagedMemoryEvent]:
        normalized_step = max(0, int(step_id))
        episode = self._episodes.setdefault(
            str(episode_key),
            {
                "events": {},
                "selected_count": 0,
                "ordinal": 0,
                "recovery": {},
            },
        )
        reasons: set[StagedMemoryTrigger] = set()
        if stage_entry:
            reasons.add(StagedMemoryTrigger.STAGE_ENTRY)
        if stage_completion_candidate:
            reasons.add(StagedMemoryTrigger.STAGE_COMPLETION_CANDIDATE)
        if stop_candidate:
            reasons.add(StagedMemoryTrigger.STOP_CANDIDATE)
        for trigger, active in (
            (StagedMemoryTrigger.NO_PROGRESS, no_progress),
            (StagedMemoryTrigger.TURN_LOOP, turn_loop),
        ):
            if self._recovery_trigger_due(episode, trigger, active, normalized_step):
                reasons.add(trigger)

        existing = episode["events"].get(normalized_step)
        if existing is not None:
            existing.add_reasons(reasons)
            return existing
        if not reasons:
            return None

        episode["ordinal"] += 1
        selected = episode["selected_count"] < self.event_cap
        if selected:
            episode["selected_count"] += 1
        event = StagedMemoryEvent(
            event_id=(f"{episode_key}:{normalized_step}:{episode['ordinal']}"),
            episode_key=str(episode_key),
            step_id=normalized_step,
            ordinal=episode["ordinal"],
            selected=selected,
            status="selected" if selected else "suppressed:event_cap",
            _reasons=reasons,
        )
        episode["events"][normalized_step] = event
        return event

    def _recovery_trigger_due(
        self,
        episode: Dict[str, Any],
        trigger: StagedMemoryTrigger,
        active: bool,
        step_id: int,
    ) -> bool:
        recovery = episode["recovery"].setdefault(
            trigger.value,
            {"active": False, "last_event_step": None},
        )
        if not active:
            recovery["active"] = False
            return False
        last_step = recovery.get("last_event_step")
        due = not recovery["active"] or (
            last_step is not None
            and step_id - int(last_step) >= self.recovery_retrigger_steps
        )
        recovery["active"] = True
        if due:
            recovery["last_event_step"] = step_id
        return due


@dataclass(frozen=True)
class ForcedMemoryOperationsResult:
    registry_status: str
    query_status: str
    registry_attempts: int
    query_attempts: int
    selected_evidence: Tuple[Dict[str, Any], ...] = ()

    @property
    def image_backed_hit_count(self) -> int:
        return len(self.selected_evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_status": self.registry_status,
            "query_status": self.query_status,
            "registry_attempts": self.registry_attempts,
            "query_attempts": self.query_attempts,
            "selected_memory_ids": [
                item.get("memory_id", "") for item in self.selected_evidence
            ],
            "attached_image_roles": [
                item.get("image_role", "retrieved_memory")
                for item in self.selected_evidence
            ],
            "image_backed_hit_count": self.image_backed_hit_count,
        }


def build_stage_memory_query(
    stage: InstructionStage,
    trigger_reasons: Sequence[StagedMemoryTrigger],
    *,
    visual_summary: str = "",
    stage_relation: str = "unknown",
    limit: int = 800,
) -> str:
    parts = [
        f"route clause: {stage.route_clause}",
        "expected landmarks: " + ", ".join(stage.expected_landmarks),
        "completion cues: " + ", ".join(stage.completion_cues),
        "trigger reasons: " + ", ".join(reason.value for reason in trigger_reasons),
        f"current relation: {str(stage_relation or 'unknown')}",
        f"current visual summary: {str(visual_summary or '')}",
    ]
    return " | ".join(parts)[: max(1, int(limit))]


def run_forced_memory_operations(
    *,
    registry_call: Callable[[], Any],
    query_call: Callable[[], Any],
    treatment: str,
    exclude_image_path: str = "",
) -> ForcedMemoryOperationsResult:
    if treatment == "off_ablation":
        return ForcedMemoryOperationsResult(
            registry_status="disabled_ablation",
            query_status="disabled_ablation",
            registry_attempts=0,
            query_attempts=0,
        )
    if treatment != "on":
        raise ValueError("treatment must be on or off_ablation")

    registry_attempts = 1
    registry_status = "empty"
    registry_evidence: list[Dict[str, Any]] = []
    try:
        registry_result = registry_call()
        registry_status, registry_evidence = _normalize_registry_result(
            registry_result,
            exclude_image_path,
        )
    except Exception:
        registry_status = "failed"

    query_attempts = 0
    query_status = "failed"
    query_evidence: list[Dict[str, Any]] = []
    for attempt in range(2):
        query_attempts += 1
        try:
            query_result = query_call()
            query_status, query_evidence, retryable = _normalize_query_result(
                query_result,
                exclude_image_path,
            )
        except Exception:
            retryable = True
            query_status = "failed"
            query_evidence = []
        if not retryable or attempt == 1:
            break

    selected = _dedupe_evidence([*registry_evidence, *query_evidence])
    return ForcedMemoryOperationsResult(
        registry_status=registry_status,
        query_status=query_status,
        registry_attempts=registry_attempts,
        query_attempts=query_attempts,
        selected_evidence=tuple(selected),
    )


def _normalize_registry_result(
    result: Any,
    exclude_image_path: str,
) -> tuple[str, list[Dict[str, Any]]]:
    if result is None:
        return "empty", []
    status = getattr(getattr(result, "status", None), "value", None)
    records = getattr(result, "records", ()) or ()
    evidence = [
        _record_evidence(record, "registry")
        for record in records
        if str(getattr(record, "image_path", "")) != exclude_image_path
    ]
    evidence = [item for item in evidence if item.get("image_path")]
    if evidence:
        return "hit", evidence
    if status == "failed":
        return "failed", []
    return "empty", []


def _normalize_query_result(
    result: Any,
    exclude_image_path: str,
) -> tuple[str, list[Dict[str, Any]], bool]:
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        return "failed", [], True
    payload = result.get("payload")
    if not isinstance(payload, Mapping):
        return "failed", [], True
    hits = payload.get("memory_hits")
    if not isinstance(hits, list):
        return "failed", [], True
    evidence = []
    for hit in hits:
        image_path = (
            hit.get("image_path")
            if isinstance(hit, Mapping)
            else getattr(hit, "image_path", "")
        )
        if not image_path or str(image_path) == exclude_image_path:
            continue
        memory_id = (
            hit.get("memory_id")
            if isinstance(hit, Mapping)
            else getattr(hit, "memory_id", "")
        )
        metadata = (
            hit.get("metadata", {})
            if isinstance(hit, Mapping)
            else getattr(hit, "metadata", {})
        )
        evidence.append(
            {
                "memory_id": str(memory_id or ""),
                "image_path": str(image_path),
                "image_role": "retrieved_memory",
                "stage_id": str(metadata.get("stage_id") or "")
                if isinstance(metadata, Mapping)
                else "",
                "source": "explicit_query",
            }
        )
    return ("hit" if evidence else "empty"), evidence, False


def _record_evidence(record: Any, source: str) -> Dict[str, Any]:
    return {
        "memory_id": str(getattr(record, "memory_id", "") or ""),
        "image_path": str(getattr(record, "image_path", "") or ""),
        "image_role": "retrieved_memory",
        "stage_id": str(getattr(record, "stage_id", "") or ""),
        "source": source,
    }


def _dedupe_evidence(items: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    deduped: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        path = str(item.get("image_path") or "")
        if not path or path in seen:
            continue
        seen.add(path)
        deduped.append(dict(item))
    return deduped
