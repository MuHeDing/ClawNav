from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Tuple

from harness.openclaw.instruction_stages import InstructionStage, TransitionType


class StageRelation(str, Enum):
    BEFORE = "before"
    AT = "at"
    INSIDE = "inside"
    PAST = "past"
    OUTSIDE = "outside"
    UNKNOWN = "unknown"


class StageTransitionDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_VERIFICATION = "needs_verification"


@dataclass(frozen=True)
class StageTransitionEvidence:
    stage_complete_candidate: bool
    relation: StageRelation = StageRelation.UNKNOWN
    evidence_refs: Tuple[str, ...] = ()
    attachment_manifest: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    grounded_landmarks: Tuple[str, ...] = ()
    grounded_completion_cues: Tuple[str, ...] = ()
    current_landmark_grounded: bool = False
    translation_since_stage_entry_m: float = 0.0
    heading_change_since_stage_entry_deg: float = 0.0
    translation_after_grounding_m: float = 0.0
    semantic_stop_gate_passed: bool = False
    structural_stop_gate_passed: bool = False


@dataclass(frozen=True)
class StageTransitionResult:
    decision: StageTransitionDecision
    rule_ids: Tuple[str, ...]
    evidence_refs: Tuple[str, ...] = ()


class StageTransitionValidator:
    def __init__(
        self,
        min_translation_m: float = 0.25,
        min_heading_change_deg: float = 15.0,
    ) -> None:
        if min_translation_m < 0 or min_heading_change_deg < 0:
            raise ValueError("stage transition thresholds must be non-negative")
        self.min_translation_m = float(min_translation_m)
        self.min_heading_change_deg = float(min_heading_change_deg)

    def validate(
        self,
        stage: InstructionStage,
        evidence: StageTransitionEvidence,
    ) -> StageTransitionResult:
        if not evidence.stage_complete_candidate:
            return self._result(
                StageTransitionDecision.REJECTED,
                "completion_candidate_required",
                evidence,
            )
        attachment_error = self._attachment_error(stage, evidence)
        if attachment_error:
            return self._result(
                StageTransitionDecision.REJECTED, attachment_error, evidence
            )

        validators = {
            TransitionType.TURN: self._validate_turn,
            TransitionType.APPROACH: self._validate_approach,
            TransitionType.PASS: self._validate_pass,
            TransitionType.ENTER: self._validate_enter,
            TransitionType.EXIT: self._validate_exit,
            TransitionType.TRAVERSE: self._validate_traverse,
            TransitionType.FINAL_ARRIVAL: self._validate_final_arrival,
        }
        return validators[stage.transition_type](stage, evidence)

    def _attachment_error(
        self,
        stage: InstructionStage,
        evidence: StageTransitionEvidence,
    ) -> str:
        for reference in evidence.evidence_refs:
            attachment = evidence.attachment_manifest.get(reference)
            if attachment is None:
                return "evidence_ref_not_attached"
            attachment_stage = str(attachment.get("stage_id") or "")
            image_role = str(attachment.get("image_role") or "")
            if (
                image_role not in {"current", "floorplan"}
                and attachment_stage
                and attachment_stage != stage.stage_id
            ):
                return "evidence_stage_mismatch"
        return ""

    def _validate_turn(
        self, stage: InstructionStage, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        if evidence.relation is StageRelation.UNKNOWN:
            return self._needs("turn_relation_unverified", evidence)
        if evidence.relation is StageRelation.BEFORE:
            return self._reject("turn_relation_before", evidence)
        if (
            abs(evidence.heading_change_since_stage_entry_deg)
            < self.min_heading_change_deg
        ):
            return self._needs("turn_heading_change_insufficient", evidence)
        return self._accept("turn_heading_and_relation_confirmed", evidence)

    def _validate_approach(
        self, stage: InstructionStage, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        grounded = (
            evidence.current_landmark_grounded
            and evidence.relation in {StageRelation.AT, StageRelation.INSIDE}
        ) or bool(evidence.grounded_completion_cues)
        if not grounded:
            return self._needs("approach_grounding_unverified", evidence)
        if evidence.translation_since_stage_entry_m < self.min_translation_m:
            return self._needs("approach_progress_insufficient", evidence)
        return self._accept("approach_grounding_and_progress_confirmed", evidence)

    def _validate_pass(
        self, stage: InstructionStage, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        if not evidence.grounded_landmarks and not evidence.current_landmark_grounded:
            return self._needs("pass_landmark_unverified", evidence)
        if evidence.relation is not StageRelation.PAST:
            return self._needs("pass_relation_unverified", evidence)
        if evidence.translation_after_grounding_m < self.min_translation_m:
            return self._needs("pass_post_grounding_motion_insufficient", evidence)
        return self._accept("pass_relation_and_motion_confirmed", evidence)

    def _validate_enter(
        self, stage: InstructionStage, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        if not evidence.grounded_landmarks and not evidence.current_landmark_grounded:
            return self._needs("enter_threshold_unverified", evidence)
        if evidence.relation is not StageRelation.INSIDE:
            return self._needs("enter_relation_unverified", evidence)
        if evidence.translation_after_grounding_m < self.min_translation_m:
            return self._needs("enter_post_threshold_motion_insufficient", evidence)
        return self._accept("enter_relation_and_motion_confirmed", evidence)

    def _validate_exit(
        self, stage: InstructionStage, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        if not evidence.grounded_landmarks and not evidence.current_landmark_grounded:
            return self._needs("exit_threshold_unverified", evidence)
        if evidence.relation is not StageRelation.OUTSIDE:
            return self._needs("exit_relation_unverified", evidence)
        if evidence.translation_after_grounding_m < self.min_translation_m:
            return self._needs("exit_post_threshold_motion_insufficient", evidence)
        return self._accept("exit_relation_and_motion_confirmed", evidence)

    def _validate_traverse(
        self, stage: InstructionStage, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        if not evidence.grounded_completion_cues:
            return self._needs("traverse_completion_cue_unverified", evidence)
        if evidence.translation_since_stage_entry_m < self.min_translation_m:
            return self._needs("traverse_progress_insufficient", evidence)
        return self._accept("traverse_cue_and_progress_confirmed", evidence)

    def _validate_final_arrival(
        self, stage: InstructionStage, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        if not stage.final_stage:
            return self._reject("final_arrival_requires_final_stage", evidence)
        if not evidence.semantic_stop_gate_passed:
            return self._needs("semantic_stop_gate_unverified", evidence)
        if not evidence.structural_stop_gate_passed:
            return self._needs("structural_stop_gate_unverified", evidence)
        return self._accept("final_arrival_gates_confirmed", evidence)

    @staticmethod
    def _result(
        decision: StageTransitionDecision,
        rule_id: str,
        evidence: StageTransitionEvidence,
    ) -> StageTransitionResult:
        return StageTransitionResult(decision, (rule_id,), evidence.evidence_refs)

    def _accept(
        self, rule_id: str, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        return self._result(StageTransitionDecision.ACCEPTED, rule_id, evidence)

    def _reject(
        self, rule_id: str, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        return self._result(StageTransitionDecision.REJECTED, rule_id, evidence)

    def _needs(
        self, rule_id: str, evidence: StageTransitionEvidence
    ) -> StageTransitionResult:
        return self._result(
            StageTransitionDecision.NEEDS_VERIFICATION, rule_id, evidence
        )
