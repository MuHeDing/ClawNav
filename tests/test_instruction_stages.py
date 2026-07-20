import pytest

from harness.openclaw.instruction_stages import (
    StageFallbackCategory,
    TransitionType,
    assert_no_forbidden_oracle_keys,
    parse_instruction_stage_plan,
)


def stage_payload(order=0, **overrides):
    payload = {
        "order": order,
        "route_clause": f"follow route clause {order}",
        "transition_type": "traverse",
        "expected_landmarks": [f"landmark {order}"],
        "completion_cues": [f"cue {order}"],
        "final_stage": True,
    }
    payload.update(overrides)
    return payload


def plan_payload(stages):
    return {"schema_version": "instruction_stages_v1", "stages": stages}


@pytest.mark.parametrize("count", [1, 12])
def test_stage_plan_accepts_supported_boundaries(count):
    stages = [
        stage_payload(index, final_stage=index == count - 1)
        for index in range(count)
    ]

    plan = parse_instruction_stage_plan(plan_payload(stages), "follow the route")

    assert plan.fallback_category is StageFallbackCategory.NONE
    assert len(plan.stages) == count
    assert [stage.stage_id for stage in plan.stages] == [
        f"stage_{index:02d}" for index in range(count)
    ]
    assert plan.stages[-1].final_stage is True


@pytest.mark.parametrize("count", [0, 13])
def test_stage_plan_falls_back_for_unsupported_stage_count(count):
    stages = [
        stage_payload(index, final_stage=index == count - 1)
        for index in range(count)
    ]

    plan = parse_instruction_stage_plan(plan_payload(stages), "Keep this exact instruction")

    assert plan.fallback_category is StageFallbackCategory.SCHEMA_ERROR
    assert len(plan.stages) == 1
    assert plan.stages[0].route_clause == "Keep this exact instruction"
    assert plan.stages[0].transition_type is TransitionType.TRAVERSE
    assert plan.stages[0].final_stage is True


@pytest.mark.parametrize(
    "stages",
    [
        [stage_payload(1)],
        [stage_payload(0, final_stage=False), stage_payload(1, final_stage=False)],
        [stage_payload(0, final_stage=True), stage_payload(1, final_stage=True)],
        [
            stage_payload(0, route_clause="same", final_stage=False),
            stage_payload(1, route_clause=" SAME "),
        ],
        [stage_payload(0, transition_type="teleport")],
        [stage_payload(0, route_clause="x" * 241)],
        [stage_payload(0, expected_landmarks=["x" * 97])],
        [stage_payload(0, completion_cues=["x" * 97])],
    ],
)
def test_stage_plan_rejects_semantically_invalid_provider_output(stages):
    plan = parse_instruction_stage_plan(plan_payload(stages), "original")

    assert plan.fallback_category is StageFallbackCategory.SCHEMA_ERROR
    assert [stage.route_clause for stage in plan.stages] == ["original"]


def test_stage_plan_requires_exact_schema_and_fields():
    extra_top_level = plan_payload([stage_payload()]) | {"reasoning": "hidden"}
    missing_stage_field = stage_payload()
    missing_stage_field.pop("completion_cues")

    assert (
        parse_instruction_stage_plan(extra_top_level, "original").fallback_category
        is StageFallbackCategory.SCHEMA_ERROR
    )
    assert (
        parse_instruction_stage_plan(
            plan_payload([missing_stage_field]), "original"
        ).fallback_category
        is StageFallbackCategory.SCHEMA_ERROR
    )


def test_stage_state_keeps_original_instruction_and_controller_progress():
    plan = parse_instruction_stage_plan(
        plan_payload(
            [
                stage_payload(0, final_stage=False),
                stage_payload(1, transition_type="final_arrival"),
            ]
        ),
        "immutable full instruction",
    )
    state = plan.new_episode_state()

    assert state.original_instruction == "immutable full instruction"
    assert state.active_stage.stage_id == "stage_00"
    assert state.completed_stage_ids == []
    assert [stage.stage_id for stage in state.pending_stages] == ["stage_01"]

    assert state.advance("controller:test") is True
    assert state.active_stage.stage_id == "stage_01"
    assert state.completed_stage_ids == ["stage_00"]
    assert state.transition_audit[-1]["authority"] == "controller:test"


@pytest.mark.parametrize(
    "payload",
    [
        {"distance_to_goal": 1.0},
        {"nested": {"success": True}},
        {"items": [{"SPL": 0.5}]},
        {"reference_path": [[0, 0], [1, 1]]},
        {"target_coordinates": [1, 2, 3]},
    ],
)
def test_oracle_guard_rejects_nested_forbidden_keys(payload):
    with pytest.raises(ValueError, match="Forbidden oracle keys"):
        assert_no_forbidden_oracle_keys(payload)


def test_oracle_guard_allows_non_target_odometry_and_visual_evidence():
    assert_no_forbidden_oracle_keys(
        {
            "odometry": {"translation_since_stage_entry_m": 0.5},
            "current_relation": "inside",
            "evidence_refs": ["current", "registry:obs_0001"],
        }
    )

