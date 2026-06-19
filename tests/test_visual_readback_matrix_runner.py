import json

import pytest

from harness.visual_readback.manifest import (
    EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION,
    compute_event_gated_phase0_manifest_sha256,
)
from scripts.run_openclaw_visual_readback_matrix import (
    build_phase1b_matrix,
    build_runner_plan,
    validate_runner_request,
)


def gate_artifact(tmp_path, status="candidate_set_ready"):
    path = tmp_path / "phase0_gate.json"
    path.write_text(
        json.dumps(
            {
                "phase0_status": status,
                "candidate_set_gate_passed": status == "candidate_set_ready",
                "fixed_manifest_gate_passed": status == "fixed_manifest_ready",
            }
        ),
        encoding="utf-8",
    )
    return path


def manifest_path(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"manifest_version": "visual_readback_fixed_v1"}), encoding="utf-8")
    return path


def phase0_manifest_path(tmp_path):
    path = tmp_path / "phase0_manifest.jsonl"
    row = {
        "manifest_schema_version": EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION,
        "scene_id": "s1",
        "episode_id": "e1",
        "max_steps": 100,
        "route_event_labels": [{"event_type": "TURN_RIGHT", "step_id": 10}],
        "accepted_step_tolerance": 2,
        "shared_readback_trigger_keys": ["s1:e1:10:decision_point"],
        "trigger_key_source": "fixture",
        "phase0_stop_go_thresholds": {
            "route_event_miss_rate_max": 0.0,
            "exact_duplicate_rate_max": 0.0,
            "adjacent_triplet_rate_max": 1.0,
            "ambiguous_evidence_rate_max": 0.0,
            "comparison_direction": "lower_is_better",
            "invalid_comparison_behavior": "mark_invalid",
        },
        "selection_rules": "unit_test_seed",
        "selection_seed": 11,
        "source_dataset_id": "dataset-v1",
        "source_run_id": "run-1",
        "source_file_hashes": {"trace.jsonl": "abc123"},
    }
    row["manifest_sha256"] = compute_event_gated_phase0_manifest_sha256([row])
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_phase1b_matrix_contains_supported_source_spec_modes_and_phase_a_control(tmp_path):
    matrix = build_phase1b_matrix(
        output_root=tmp_path / "runs",
        manifest_path=manifest_path(tmp_path),
    )

    source_modes = [
        item["visual_readback_mode"]
        for item in matrix
        if item["claim_scope"] == "full_fixed_manifest"
    ]
    assert source_modes == [
        "off",
        "text_only_prompt",
        "path_only",
        "image_read_prompt",
        "image_read_controller",
        "current_only_controller",
        "shuffled_image_read_controller",
    ]
    assert "direct_policy_images" not in source_modes
    control = next(item for item in matrix if item["name"] == "interval_10_smoke_audit_control")
    assert control["visual_readback_mode"] == "image_read_controller"
    assert control["effective_config"]["keyframe_policy_mode"] == "interval"
    assert control["effective_config"]["keyframe_interval_step"] == "10"
    assert control["effective_config"]["source_image_role"] == "interval_smoke_audit_keyframe"
    assert control["effective_config"]["candidate_pool_diagnostics"] == "smoke_audit_compatible"
    assert control["claim_scope"] == "phase_a_retrieval_audit_control"


def test_phase1b_runner_requires_gate_artifact_and_manifest(tmp_path):
    with pytest.raises(ValueError, match="phase0 gate"):
        validate_runner_request("phase1b", gate_artifact_path="", manifest_path="manifest.json")

    with pytest.raises(ValueError, match="manifest"):
        validate_runner_request("phase1b", gate_artifact_path=str(gate_artifact(tmp_path)), manifest_path="")


def test_phase1b_runner_refuses_before_fixed_manifest_gate(tmp_path):
    with pytest.raises(ValueError, match="Fixed-manifest gate"):
        validate_runner_request(
            "phase1b",
            gate_artifact_path=str(gate_artifact(tmp_path, status="candidate_set_ready")),
            manifest_path=str(manifest_path(tmp_path)),
        )


def test_runner_plan_records_effective_configs_and_output_paths(tmp_path):
    gate = gate_artifact(tmp_path, status="fixed_manifest_ready")
    manifest = manifest_path(tmp_path)

    plan = build_runner_plan(
        phase="phase1b",
        output_root=tmp_path / "runs",
        gate_artifact_path=str(gate),
        manifest_path=str(manifest),
        dry_run=True,
    )

    assert len(plan) == 8
    v4c = next(item for item in plan if item["name"] == "V4c_current_only_controller")
    assert v4c["effective_config"]["visual_readback_mode"] == "current_only_controller"
    assert v4c["effective_config"]["visual_readback_control_only"] == "1"
    assert str(tmp_path / "runs" / "V4c_current_only_controller") in v4c["command"][-1]
    control = next(item for item in plan if item["name"] == "interval_10_smoke_audit_control")
    assert "--keyframe_policy_mode interval" in control["command"][-1]


def test_phase0b_runner_emits_interval_cleanup_reporting_arm(tmp_path):
    manifest = phase0_manifest_path(tmp_path)

    plan = build_runner_plan(
        phase="phase0b",
        output_root=tmp_path / "runs",
        manifest_path=str(manifest),
        dry_run=True,
    )

    assert [item["name"] for item in plan] == ["interval_10", "interval_10_exact_dedupe"]
    assert plan[0]["effective_config"]["keyframe_policy_mode"] == "interval"
    assert plan[1]["effective_config"]["interval_cleanup_exact_dedupe"] == "summary_only"
    assert plan[1]["claim_scope"] == "interval_cleanup_reporting_only"
    assert "--event_gated_phase0_manifest" in plan[1]["command"][-1]


def test_downscope_plan_cannot_emit_full_matrix_claims(tmp_path):
    plan = build_runner_plan(
        phase="stop_only",
        output_root=tmp_path / "runs",
        gate_artifact_path=str(gate_artifact(tmp_path, status="downscope_stop_only")),
        manifest_path="",
        dry_run=True,
    )

    assert [item["name"] for item in plan] == ["Phase1a_V4_stop_only"]
    assert plan[0]["claim_scope"] == "narrow_stop_only"
