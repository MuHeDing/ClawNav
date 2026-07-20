from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple


INSTRUCTION_STAGE_SCHEMA_VERSION = "instruction_stages_v1"
MAX_INSTRUCTION_CHARS = 4096
MAX_STAGES = 12
MAX_ROUTE_CLAUSE_CHARS = 240
MAX_LANDMARKS = 4
MAX_COMPLETION_CUES = 3
MAX_EVIDENCE_TEXT_CHARS = 96

FORBIDDEN_ORACLE_KEYS = frozenset(
    {
        "distance_to_goal",
        "success",
        "spl",
        "softspl",
        "oracle_success",
        "oracle_path",
        "oracle_shortest_path",
        "oracle_shortest_path_action",
        "oracle_action",
        "target_coordinate",
        "target_coordinates",
        "target_coords",
        "target_position",
        "goal_coordinates",
        "goal_position",
        "reference_path",
        "shortest_path",
    }
)


class TransitionType(str, Enum):
    TURN = "turn"
    APPROACH = "approach"
    PASS = "pass"
    ENTER = "enter"
    EXIT = "exit"
    TRAVERSE = "traverse"
    FINAL_ARRIVAL = "final_arrival"


class StageFallbackCategory(str, Enum):
    NONE = "none"
    TRANSPORT_ERROR = "transport_error"
    PARSE_ERROR = "parse_error"
    SCHEMA_ERROR = "schema_error"
    PROVIDER_FAILURE = "provider_failure"
    MANIFEST_ERROR = "manifest_error"


@dataclass(frozen=True)
class InstructionStage:
    stage_id: str
    order: int
    route_clause: str
    transition_type: TransitionType
    expected_landmarks: Tuple[str, ...] = ()
    completion_cues: Tuple[str, ...] = ()
    final_stage: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "order": self.order,
            "route_clause": self.route_clause,
            "transition_type": self.transition_type.value,
            "expected_landmarks": list(self.expected_landmarks),
            "completion_cues": list(self.completion_cues),
            "final_stage": self.final_stage,
        }


@dataclass
class EpisodeStageState:
    original_instruction: str
    stages: Tuple[InstructionStage, ...]
    instruction_sha256: str
    stage_plan_sha256: str
    segmentation_source: str
    fallback_category: StageFallbackCategory = StageFallbackCategory.NONE
    active_stage_index: int = 0
    completed_stage_ids: list[str] = field(default_factory=list)
    transition_audit: list[Dict[str, Any]] = field(default_factory=list)

    @property
    def active_stage(self) -> InstructionStage:
        return self.stages[self.active_stage_index]

    @property
    def pending_stages(self) -> Tuple[InstructionStage, ...]:
        return self.stages[self.active_stage_index + 1 :]

    @property
    def completed_stages(self) -> Tuple[InstructionStage, ...]:
        completed = set(self.completed_stage_ids)
        return tuple(stage for stage in self.stages if stage.stage_id in completed)

    def advance(self, authority: str) -> bool:
        if not str(authority).startswith("controller"):
            raise ValueError("only controller authority may advance stage state")
        if self.active_stage_index >= len(self.stages) - 1:
            return False
        previous = self.active_stage
        self.completed_stage_ids.append(previous.stage_id)
        self.active_stage_index += 1
        self.transition_audit.append(
            {
                "from_stage_id": previous.stage_id,
                "to_stage_id": self.active_stage.stage_id,
                "authority": str(authority),
            }
        )
        return True


@dataclass(frozen=True)
class InstructionStagePlan:
    original_instruction: str
    stages: Tuple[InstructionStage, ...]
    instruction_sha256: str
    stage_plan_sha256: str
    segmentation_source: str
    fallback_category: StageFallbackCategory = StageFallbackCategory.NONE

    def new_episode_state(self) -> EpisodeStageState:
        return EpisodeStageState(
            original_instruction=self.original_instruction,
            stages=self.stages,
            instruction_sha256=self.instruction_sha256,
            stage_plan_sha256=self.stage_plan_sha256,
            segmentation_source=self.segmentation_source,
            fallback_category=self.fallback_category,
        )


class InstructionStageSchemaError(ValueError):
    pass


def parse_instruction_stage_plan(
    payload: Any,
    original_instruction: str,
    *,
    fallback_category: StageFallbackCategory = StageFallbackCategory.SCHEMA_ERROR,
) -> InstructionStagePlan:
    instruction = _validate_instruction(original_instruction)
    try:
        stages = _validate_payload(payload)
    except (InstructionStageSchemaError, TypeError, ValueError):
        return _fallback_plan(instruction, fallback_category)
    return _build_plan(
        instruction,
        stages,
        segmentation_source="qwen",
        fallback_category=StageFallbackCategory.NONE,
    )


def single_stage_fallback(
    original_instruction: str,
    category: StageFallbackCategory,
) -> InstructionStagePlan:
    return _fallback_plan(_validate_instruction(original_instruction), category)


def assert_no_forbidden_oracle_keys(value: Any) -> None:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized = str(key).strip().lower()
                if normalized in FORBIDDEN_ORACLE_KEYS:
                    found.add(str(key))
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for nested in item:
                visit(nested)

    visit(value)
    if found:
        raise ValueError("Forbidden oracle keys: " + ", ".join(sorted(found)))


def _validate_instruction(instruction: Any) -> str:
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a string")
    normalized = instruction.strip()
    if not normalized:
        raise ValueError("instruction must not be empty")
    if len(normalized) > MAX_INSTRUCTION_CHARS:
        raise ValueError("instruction exceeds maximum length")
    return normalized


def _validate_payload(payload: Any) -> Tuple[InstructionStage, ...]:
    if not isinstance(payload, Mapping):
        raise InstructionStageSchemaError("response must be an object")
    if set(payload) != {"schema_version", "stages"}:
        raise InstructionStageSchemaError("response fields mismatch")
    if payload.get("schema_version") != INSTRUCTION_STAGE_SCHEMA_VERSION:
        raise InstructionStageSchemaError("unsupported stage schema")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not 1 <= len(raw_stages) <= MAX_STAGES:
        raise InstructionStageSchemaError("stage count is out of bounds")

    stages = tuple(_validate_stage(raw, index) for index, raw in enumerate(raw_stages))
    normalized_clauses = [stage.route_clause.casefold() for stage in stages]
    if len(set(normalized_clauses)) != len(normalized_clauses):
        raise InstructionStageSchemaError("route clauses must be unique")
    final_indices = [stage.order for stage in stages if stage.final_stage]
    if final_indices != [len(stages) - 1]:
        raise InstructionStageSchemaError("only the last stage may be final")
    return stages


def _validate_stage(raw: Any, expected_order: int) -> InstructionStage:
    if not isinstance(raw, Mapping):
        raise InstructionStageSchemaError("stage must be an object")
    required = {
        "order",
        "route_clause",
        "transition_type",
        "expected_landmarks",
        "completion_cues",
        "final_stage",
    }
    if set(raw) != required:
        raise InstructionStageSchemaError("stage fields mismatch")
    order = raw.get("order")
    if isinstance(order, bool) or not isinstance(order, int) or order != expected_order:
        raise InstructionStageSchemaError("stage order must be contiguous and zero-based")
    route_clause = _bounded_text(raw.get("route_clause"), MAX_ROUTE_CLAUSE_CHARS)
    try:
        transition_type = TransitionType(raw.get("transition_type"))
    except (TypeError, ValueError) as exc:
        raise InstructionStageSchemaError("unsupported transition type") from exc
    landmarks = _bounded_text_list(
        raw.get("expected_landmarks"), MAX_LANDMARKS, "expected_landmarks"
    )
    cues = _bounded_text_list(
        raw.get("completion_cues"), MAX_COMPLETION_CUES, "completion_cues"
    )
    final_stage = raw.get("final_stage")
    if not isinstance(final_stage, bool):
        raise InstructionStageSchemaError("final_stage must be boolean")
    return InstructionStage(
        stage_id=f"stage_{expected_order:02d}",
        order=expected_order,
        route_clause=route_clause,
        transition_type=transition_type,
        expected_landmarks=landmarks,
        completion_cues=cues,
        final_stage=final_stage,
    )


def _bounded_text(value: Any, maximum: int) -> str:
    if not isinstance(value, str):
        raise InstructionStageSchemaError("text field must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise InstructionStageSchemaError("text field is empty or overlong")
    return normalized


def _bounded_text_list(value: Any, maximum: int, field_name: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise InstructionStageSchemaError(f"{field_name} must be a bounded list")
    normalized = tuple(_bounded_text(item, MAX_EVIDENCE_TEXT_CHARS) for item in value)
    if len({item.casefold() for item in normalized}) != len(normalized):
        raise InstructionStageSchemaError(f"{field_name} entries must be unique")
    return normalized


def _fallback_plan(
    instruction: str,
    category: StageFallbackCategory,
) -> InstructionStagePlan:
    stage = InstructionStage(
        stage_id="stage_00",
        order=0,
        route_clause=instruction,
        transition_type=TransitionType.TRAVERSE,
        expected_landmarks=(),
        completion_cues=(),
        final_stage=True,
    )
    return _build_plan(
        instruction,
        (stage,),
        segmentation_source="fallback",
        fallback_category=category,
    )


def _build_plan(
    instruction: str,
    stages: Tuple[InstructionStage, ...],
    *,
    segmentation_source: str,
    fallback_category: StageFallbackCategory,
) -> InstructionStagePlan:
    instruction_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    canonical_stages = [stage.to_dict() for stage in stages]
    stage_hash = hashlib.sha256(
        json.dumps(
            canonical_stages,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return InstructionStagePlan(
        original_instruction=instruction,
        stages=stages,
        instruction_sha256=instruction_hash,
        stage_plan_sha256=stage_hash,
        segmentation_source=segmentation_source,
        fallback_category=fallback_category,
    )
