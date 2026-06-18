import json

import pytest

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


def test_phase1b_matrix_contains_only_supported_source_spec_modes(tmp_path):
    matrix = build_phase1b_matrix(
        output_root=tmp_path / "runs",
        manifest_path=manifest_path(tmp_path),
    )

    modes = [item["visual_readback_mode"] for item in matrix]
    assert modes == [
        "off",
        "text_only_prompt",
        "path_only",
        "image_read_prompt",
        "image_read_controller",
        "current_only_controller",
        "shuffled_image_read_controller",
    ]
    assert "direct_policy_images" not in modes


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

    assert len(plan) == 7
    v4c = next(item for item in plan if item["name"] == "V4c_current_only_controller")
    assert v4c["effective_config"]["visual_readback_mode"] == "current_only_controller"
    assert v4c["effective_config"]["visual_readback_control_only"] == "1"
    assert str(tmp_path / "runs" / "V4c_current_only_controller") in v4c["command"][-1]


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
