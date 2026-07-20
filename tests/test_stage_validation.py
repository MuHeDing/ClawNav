import pytest

from harness.openclaw.instruction_stages import InstructionStage, TransitionType
from harness.openclaw.stage_validation import (
    StageRelation,
    StageTransitionDecision,
    StageTransitionEvidence,
    StageTransitionValidator,
)


def make_stage(transition_type, *, final_stage=False):
    return InstructionStage(
        stage_id="stage_00",
        order=0,
        route_clause="cross the named transition",
        transition_type=TransitionType(transition_type),
        expected_landmarks=("doorway",),
        completion_cues=("threshold crossed",),
        final_stage=final_stage,
    )


def make_evidence(**overrides):
    data = {
        "stage_complete_candidate": True,
        "relation": StageRelation.AT,
        "evidence_refs": ("current",),
        "attachment_manifest": {"current": {"image_role": "current", "stage_id": "stage_00"}},
        "grounded_landmarks": ("doorway",),
        "grounded_completion_cues": ("threshold crossed",),
        "current_landmark_grounded": True,
        "translation_since_stage_entry_m": 0.5,
        "heading_change_since_stage_entry_deg": 30.0,
        "translation_after_grounding_m": 0.5,
        "semantic_stop_gate_passed": True,
        "structural_stop_gate_passed": True,
    }
    data.update(overrides)
    return StageTransitionEvidence(**data)


@pytest.mark.parametrize(
    ("transition_type", "relation", "final_stage"),
    [
        ("turn", StageRelation.AT, False),
        ("approach", StageRelation.AT, False),
        ("pass", StageRelation.PAST, False),
        ("enter", StageRelation.INSIDE, False),
        ("exit", StageRelation.OUTSIDE, False),
        ("traverse", StageRelation.AT, False),
        ("final_arrival", StageRelation.AT, True),
    ],
)
def test_transition_validator_accepts_each_supported_transition(
    transition_type, relation, final_stage
):
    validator = StageTransitionValidator(
        min_translation_m=0.25,
        min_heading_change_deg=15.0,
    )

    result = validator.validate(
        make_stage(transition_type, final_stage=final_stage),
        make_evidence(relation=relation),
    )

    assert result.decision is StageTransitionDecision.ACCEPTED
    assert result.rule_ids
    assert result.evidence_refs == ("current",)


def test_transition_validator_rejects_missing_candidate_and_unattached_reference():
    validator = StageTransitionValidator()
    stage = make_stage("traverse")

    no_candidate = validator.validate(
        stage, make_evidence(stage_complete_candidate=False)
    )
    unattached = validator.validate(
        stage,
        make_evidence(
            evidence_refs=("memory:mem_1",),
            attachment_manifest={"current": {"image_role": "current"}},
        ),
    )

    assert no_candidate.decision is StageTransitionDecision.REJECTED
    assert "completion_candidate_required" in no_candidate.rule_ids
    assert unattached.decision is StageTransitionDecision.REJECTED
    assert "evidence_ref_not_attached" in unattached.rule_ids


@pytest.mark.parametrize(
    ("transition_type", "evidence"),
    [
        ("turn", {"heading_change_since_stage_entry_deg": 14.99}),
        ("approach", {"translation_since_stage_entry_m": 0.24}),
        ("pass", {"translation_after_grounding_m": 0.24}),
        ("enter", {"relation": StageRelation.UNKNOWN}),
        ("exit", {"current_landmark_grounded": False, "grounded_landmarks": ()}),
        ("traverse", {"grounded_completion_cues": ()}),
        ("final_arrival", {"structural_stop_gate_passed": False}),
    ],
)
def test_transition_validator_defers_incomplete_evidence(transition_type, evidence):
    validator = StageTransitionValidator()
    stage = make_stage(
        transition_type,
        final_stage=transition_type == "final_arrival",
    )

    result = validator.validate(stage, make_evidence(**evidence))

    assert result.decision is StageTransitionDecision.NEEDS_VERIFICATION


def test_transition_validator_rejects_cross_stage_historical_evidence():
    validator = StageTransitionValidator()
    result = validator.validate(
        make_stage("pass"),
        make_evidence(
            relation=StageRelation.PAST,
            evidence_refs=("registry:obs_1",),
            attachment_manifest={
                "registry:obs_1": {
                    "image_role": "history",
                    "stage_id": "stage_02",
                }
            },
        ),
    )

    assert result.decision is StageTransitionDecision.REJECTED
    assert "evidence_stage_mismatch" in result.rule_ids

