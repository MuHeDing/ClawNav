#!/usr/bin/env python
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

from harness.visual_readback.manifest import load_event_gated_phase0_manifest_jsonl


LOWMEM_ARGS = [
    "--num_history 8",
    "--max_pixels 401408",
    "--kv_start_size 8",
    "--kv_recent_size 24",
]
COMMON_ENV = [
    "PYTHONPATH=.:src",
]
COMMON_ARGS = [
    "/ssd/dingmuhe/anaconda3/envs/janusvln/bin/python",
    "src/evaluation_harness.py",
    "--model_path ${MODEL_PATH}",
    "--habitat_config_path config/vln_r2r.yaml",
    "--harness_runtime openclaw_bridge",
    "--openclaw_planner_backend gateway",
    "--openclaw_gateway_url ${OPENCLAW_GATEWAY_URL}",
    "--harness_memory_backend ${HARNESS_MEMORY_BACKEND:-image_backed_local}",
    "--visual_readback_control_only true",
    "--visual_readback_stop_fallback_policy log_only",
    *LOWMEM_ARGS,
]
PHASE1B_MODES = [
    ("V0_off", "off"),
    ("V1_text_only_prompt", "text_only_prompt"),
    ("V2_path_only", "path_only"),
    ("V3_image_read_prompt", "image_read_prompt"),
    ("V4_image_read_controller", "image_read_controller"),
    ("V4c_current_only_controller", "current_only_controller"),
    ("V5_shuffled_image_read_controller", "shuffled_image_read_controller"),
]


def build_phase1b_matrix(output_root: Path, manifest_path: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for name, mode in PHASE1B_MODES:
        effective_config = {
            "visual_readback_mode": mode,
            "visual_readback_control_only": "1"
            if mode in {"image_read_controller", "current_only_controller", "shuffled_image_read_controller"}
            else "0",
            "visual_readback_fixed_case_manifest": str(manifest_path),
        }
        if mode == "shuffled_image_read_controller":
            effective_config["visual_readback_shuffle_seed"] = "${OPENCLAW_VISUAL_READBACK_SHUFFLE_SEED:-123}"
        args = [
            *COMMON_ENV,
            *COMMON_ARGS,
            "--visual_readback_mode",
            mode,
            "--visual_readback_fixed_case_manifest",
            str(manifest_path),
        ]
        if mode == "shuffled_image_read_controller":
            args.extend(["--visual_readback_shuffle_seed", "${OPENCLAW_VISUAL_READBACK_SHUFFLE_SEED:-123}"])
        args.extend(["--output_path", str(output_root / name)])
        runs.append(
            {
                "name": name,
                "visual_readback_mode": mode,
                "effective_config": effective_config,
                "command": ["bash", "-lc", " ".join(args)],
                "claim_scope": "full_fixed_manifest",
            }
        )
    runs.append(_phase_a_smoke_audit_control_run(output_root, manifest_path))
    return runs


def _phase_a_smoke_audit_control_run(
    output_root: Path,
    manifest_path: Path,
) -> Dict[str, object]:
    name = "interval_10_smoke_audit_control"
    args = [
        *COMMON_ENV,
        *COMMON_ARGS,
        "--visual_readback_mode",
        "image_read_controller",
        "--visual_readback_fixed_case_manifest",
        str(manifest_path),
        "--visual_readback_control_only",
        "true",
        "--keyframe_policy_mode",
        "interval",
        "--output_path",
        str(output_root / name),
    ]
    return {
        "name": name,
        "visual_readback_mode": "image_read_controller",
        "effective_config": {
            "keyframe_policy_mode": "interval",
            "keyframe_interval_step": "10",
            "visual_readback_mode": "image_read_controller",
            "visual_readback_control_only": "1",
            "visual_readback_fixed_case_manifest": str(manifest_path),
            "memory_scope": "episode",
            "source_image_role": "interval_smoke_audit_keyframe",
            "memory_namespace_role": "interval_smoke_audit_control",
            "readback_query_predicate": "prior_step_same_run_episode_namespace",
            "exact_dedupe": "enabled",
            "attachment_audit": "enabled",
            "evidence_diagnostics": "enabled",
            "candidate_pool_diagnostics": "smoke_audit_compatible",
            "legacy_auto_write_bypass_scope": "interval_smoke_audit_namespace",
        },
        "command": ["bash", "-lc", " ".join(args)],
        "claim_scope": "phase_a_retrieval_audit_control",
    }


def build_phase0b_interval_cleanup_matrix(
    output_root: Path,
    manifest_path: Path,
) -> List[Dict[str, object]]:
    manifest = load_event_gated_phase0_manifest_jsonl(manifest_path)
    thresholds = dict(manifest.rows[0].get("phase0_stop_go_thresholds") or {})
    interval_run_dir = output_root / "interval_10"
    interval_args = [
        *COMMON_ENV,
        *COMMON_ARGS,
        "--visual_readback_mode",
        "image_read_controller",
        "--output_path",
        str(interval_run_dir),
    ]
    summary_args = [
        *COMMON_ENV,
        "/ssd/dingmuhe/anaconda3/envs/janusvln/bin/python",
        "scripts/summarize_openclaw_visual_readback.py",
        str(interval_run_dir),
        "--format",
        "json",
        "--event_gated_phase0_manifest",
        str(manifest_path),
        "--expected_manifest_sha256",
        manifest.manifest_sha256,
        "--output",
        str(output_root / "interval_10_exact_dedupe" / "summary.json"),
    ]
    return [
        {
            "name": "interval_10",
            "visual_readback_mode": "image_read_controller",
            "effective_config": {
                "keyframe_policy_mode": "interval",
                "keyframe_interval_step": "10",
                "visual_readback_mode": "image_read_controller",
                "event_gated_phase0_manifest_path": str(manifest_path),
                "event_gated_phase0_manifest_sha256": manifest.manifest_sha256,
                "phase0_stop_go_thresholds": thresholds,
            },
            "command": ["bash", "-lc", " ".join(interval_args)],
            "claim_scope": "phase0_interval_baseline",
        },
        {
            "name": "interval_10_exact_dedupe",
            "visual_readback_mode": "image_read_controller",
            "effective_config": {
                "keyframe_policy_mode": "interval",
                "keyframe_interval_step": "10",
                "visual_readback_mode": "image_read_controller",
                "event_gated_phase0_manifest_path": str(manifest_path),
                "event_gated_phase0_manifest_sha256": manifest.manifest_sha256,
                "phase0_stop_go_thresholds": thresholds,
                "interval_cleanup_exact_dedupe": "summary_only",
            },
            "command": ["bash", "-lc", " ".join(summary_args)],
            "claim_scope": "interval_cleanup_reporting_only",
        },
    ]


def build_runner_plan(
    phase: str,
    output_root: Path,
    gate_artifact_path: str = "",
    manifest_path: str = "",
    dry_run: bool = False,
) -> List[Dict[str, object]]:
    del dry_run
    validate_runner_request(phase, gate_artifact_path, manifest_path)
    if phase == "phase0":
        return [
            {
                "name": "Phase0_audit",
                "visual_readback_mode": "off",
                "effective_config": {"visual_readback_mode": "off"},
                "command": [
                    "bash",
                    "-lc",
                    " ".join([*COMMON_ENV, *COMMON_ARGS, "--visual_readback_mode", "off", "--output_path", str(output_root / "Phase0_audit")]),
                ],
                "claim_scope": "phase0_audit_only",
            }
        ]
    if phase == "phase0b":
        return build_phase0b_interval_cleanup_matrix(Path(output_root), Path(manifest_path))
    if phase == "phase1a":
        return [_phase1a_run(output_root, "Phase1a_V4", "mechanism_validation")]
    if phase == "stop_only":
        return [_phase1a_run(output_root, "Phase1a_V4_stop_only", "narrow_stop_only")]
    if phase == "turn_only":
        return [_phase1a_run(output_root, "Phase1a_V4_turn_only", "narrow_turn_only")]
    if phase == "phase1b":
        return build_phase1b_matrix(Path(output_root), Path(manifest_path))
    raise ValueError(f"unsupported visual readback runner phase: {phase}")


def validate_runner_request(
    phase: str,
    gate_artifact_path: str = "",
    manifest_path: str = "",
) -> None:
    if phase not in {"phase0", "phase0b", "phase1a", "phase1b", "stop_only", "turn_only"}:
        raise ValueError(f"unsupported visual readback runner phase: {phase}")
    if phase == "phase0b" and not manifest_path:
        raise ValueError("phase0b requires an event-gated Phase 0 manifest")
    if phase == "phase1b":
        if not gate_artifact_path:
            raise ValueError("phase1b requires a phase0 gate artifact")
        if not manifest_path:
            raise ValueError("phase1b requires a fixed replay manifest")
        gate = _read_gate_artifact(gate_artifact_path)
        if not gate.get("fixed_manifest_gate_passed"):
            raise ValueError("Fixed-manifest gate must pass before phase1b matrix")
    if phase in {"stop_only", "turn_only"} and gate_artifact_path:
        gate = _read_gate_artifact(gate_artifact_path)
        allowed = {
            "stop_only": "downscope_stop_only",
            "turn_only": "downscope_turn_only",
        }[phase]
        if gate.get("phase0_status") not in {allowed, "candidate_set_ready", "fixed_manifest_ready"}:
            raise ValueError(f"{phase} requires compatible Phase 0 downscope gate")


def _phase1a_run(output_root: Path, name: str, claim_scope: str) -> Dict[str, object]:
    args = [
        *COMMON_ENV,
        *COMMON_ARGS,
        "--visual_readback_mode",
        "image_read_controller",
        "--output_path",
        str(output_root / name),
    ]
    return {
        "name": name,
        "visual_readback_mode": "image_read_controller",
        "effective_config": {
            "visual_readback_mode": "image_read_controller",
            "visual_readback_control_only": "1",
            "visual_readback_stop_fallback_policy": "log_only",
        },
        "command": ["bash", "-lc", " ".join(args)],
        "claim_scope": claim_scope,
    }


def _read_gate_artifact(path: str) -> Dict[str, object]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ValueError(f"phase0 gate artifact does not exist: {path}")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["phase0", "phase0b", "phase1a", "phase1b", "stop_only", "turn_only"],
        required=True,
    )
    parser.add_argument("--output_root", default="results/openclaw_visual_readback")
    parser.add_argument("--phase0_gate_artifact", default="")
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    plan = build_runner_plan(
        phase=args.phase,
        output_root=Path(args.output_root),
        gate_artifact_path=args.phase0_gate_artifact,
        manifest_path=args.manifest_path,
        dry_run=args.dry_run,
    )
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for item in plan:
        print(item["name"], item["command"])
        if not args.dry_run:
            subprocess.run(item["command"], check=True)


if __name__ == "__main__":
    main()
