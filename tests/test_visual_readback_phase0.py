import json

from evaluation_harness import build_memory_client
from harness.config import HarnessConfig
from harness.memory.spatial_memory_client import FakeSpatialMemoryClient
from harness.visual_readback.manifest import (
    EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION,
    compute_event_gated_phase0_manifest_sha256,
)
from harness.visual_readback.audit import build_phase0_audit_report
from harness.visual_readback.memory_smoke import (
    ImageBackedLocalMemoryClient,
    run_image_backed_memory_smoke,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_phase0_manifest(path, rows):
    manifest_hash = compute_event_gated_phase0_manifest_sha256(rows)
    rows = [{**row, "manifest_sha256": manifest_hash} for row in rows]
    write_jsonl(path, rows)
    return manifest_hash


def test_phase0_audit_separates_primary_offline_and_oracle_rows(tmp_path):
    trace_path = tmp_path / "harness_traces" / "harness_trace_rank0.jsonl"
    write_jsonl(
        trace_path,
        [
            {
                "scene_id": "s1",
                "episode_id": "e1",
                "visual_readback": {
                    "trigger_source": "online_controller",
                    "trigger_rule": "risky_stop",
                    "oracle_selection_fields": [],
                },
            },
            {
                "scene_id": "s1",
                "episode_id": "e1",
                "visual_readback": {
                    "trigger_source": "offline_fixture",
                    "trigger_rule": "near_miss_recovery",
                    "oracle_selection_fields": ["success"],
                },
            },
            {
                "scene_id": "s2",
                "episode_id": "e2",
                "visual_readback": {
                    "trigger_source": "static_pre_run",
                    "trigger_rule": "decision_point",
                    "oracle_selection_fields": [],
                },
            },
            {
                "scene_id": "s3",
                "episode_id": "e3",
                "visual_readback": {
                    "trigger_source": "online_controller",
                    "trigger_rule": "route_uncertainty",
                    "oracle_selection_fields": ["distance_to_goal"],
                },
            },
        ],
    )

    report = build_phase0_audit_report([trace_path])

    assert report["parent_episode_count"] == 3
    assert report["parent_online_trigger_count"] == 2
    assert report["parent_static_pre_run_count"] == 1
    assert report["parent_eligible_trigger_count"] == 2
    assert report["primary_quantitative_set_count"] == 2
    assert report["candidate_case_set_count"] == 2
    assert report["offline_stress_set_count"] == 1
    assert report["oracle_rejected_count"] == 1
    assert report["trigger_type_distribution"] == {
        "decision_point": 1,
        "near_miss_recovery": 1,
        "risky_stop": 1,
        "route_uncertainty": 1,
    }
    assert [case["trigger_rule"] for case in report["candidate_case_set"]] == [
        "risky_stop",
        "decision_point",
    ]


def test_phase0_audit_reports_manifest_thresholds_route_misses_and_interval_cleanup(tmp_path):
    trace_path = tmp_path / "harness_traces" / "harness_trace_rank0.jsonl"
    write_jsonl(
        trace_path,
        [
            {
                "scene_id": "s1",
                "episode_id": "e1",
                "step_id": 4,
                "visual_readback": {
                    "trigger_key": "s1:e1:4:decision_point",
                    "trigger_source": "online_controller",
                    "trigger_rule": "decision_point",
                    "read_status": "completed",
                    "retrieved_image_paths": [
                        "/tmp/scene/keyframe_000000.png",
                        "/tmp/scene/keyframe_000010.png",
                        "/tmp/scene/keyframe_000010.png",
                        "/tmp/scene/keyframe_000020.png",
                    ],
                    "matched_memory_ids": ["m1"],
                },
            },
            {
                "scene_id": "s1",
                "episode_id": "e1",
                "step_id": 26,
                "visual_readback": {
                    "trigger_source": "online_controller",
                    "trigger_rule": "risky_stop",
                    "read_status": "completed",
                    "retrieved_image_paths": [],
                    "matched_memory_ids": [],
                },
            },
        ],
    )
    manifest_path = tmp_path / "phase0_manifest.jsonl"
    thresholds = {
        "route_event_miss_rate_max": 0.25,
        "exact_duplicate_rate_max": 0.25,
        "adjacent_triplet_rate_max": 0.5,
        "ambiguous_evidence_rate_max": 0.5,
        "comparison_direction": "lower_is_better",
        "invalid_comparison_behavior": "mark_invalid",
    }
    manifest_hash = write_phase0_manifest(
        manifest_path,
        [
            {
                "manifest_schema_version": EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION,
                "scene_id": "s1",
                "episode_id": "e1",
                "max_steps": 30,
                "route_event_labels": [
                    {"event_type": "TURN_RIGHT", "step_id": 10},
                    {"event_type": "STOP", "step_id": 26},
                ],
                "accepted_step_tolerance": 2,
                "shared_readback_trigger_keys": [
                    "s1:e1:4:decision_point",
                    "s1:e1:26:risky_stop",
                ],
                "trigger_key_source": "fixture",
                "phase0_stop_go_thresholds": thresholds,
                "selection_rules": "unit_test_seed",
                "selection_seed": 7,
                "source_dataset_id": "dataset-v1",
                "source_run_id": "run-1",
                "source_file_hashes": {"trace.jsonl": "abc123"},
            }
        ],
    )

    report = build_phase0_audit_report([trace_path], manifest_path=manifest_path)

    assert report["manifest_schema_version"] == EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION
    assert report["manifest_sha256"] == manifest_hash
    assert report["phase0_stop_go_thresholds"] == thresholds
    assert report["route_event_label_count"] == 2
    assert report["missed_route_event_count"] == 1
    assert report["route_event_miss_rate"] == 0.5
    assert report["exact_duplicate_drop_count"] == 1
    assert report["exact_duplicate_rate"] == 0.25
    assert report["adjacent_triplet_count"] == 1
    assert report["adjacent_triplet_rate"] == 1.0
    assert report["legacy_stable_evidence_count"] == 1
    assert report["legacy_ambiguous_evidence_count"] == 1
    assert report["unmatched_trigger_count"] == 0
    assert report["phase0_threshold_results"]["route_event_miss_rate_max"]["passed"] is False
    assert report["phase0_threshold_results"]["exact_duplicate_rate_max"]["passed"] is True
    assert report["phase0_stop_go_decision"] == "go_event_gated_smoke"


def test_image_backed_memory_smoke_fails_for_text_only_fake_backend(tmp_path):
    image_path = tmp_path / "keyframe.png"
    image_path.write_text("fake image bytes", encoding="utf-8")

    result = run_image_backed_memory_smoke(
        FakeSpatialMemoryClient(),
        image_path=image_path,
        memory_namespace="episode:s1:e1",
    )

    assert result["passed"] is False
    assert result["retrieved_image_path"] == ""
    assert "image_path" in result["reason"]


def test_image_backed_memory_smoke_passes_when_backend_round_trips_path(tmp_path):
    image_path = tmp_path / "keyframe.png"
    image_path.write_text("fake image bytes", encoding="utf-8")
    client = ImageBackedLocalMemoryClient()

    result = run_image_backed_memory_smoke(
        client,
        image_path=image_path,
        memory_namespace="episode:s1:e1",
    )

    assert result["passed"] is True
    assert result["written_image_path"] == str(image_path)
    assert result["retrieved_image_path"] == str(image_path)
    assert result["retrieved_memory_id"]


def test_build_memory_client_can_create_image_backed_local_client():
    client = build_memory_client(HarnessConfig(memory_backend="image_backed_local"))

    assert isinstance(client, ImageBackedLocalMemoryClient)
