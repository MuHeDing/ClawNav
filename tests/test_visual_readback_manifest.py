import json

import pytest

from scripts.build_event_gated_phase0_manifest import (
    build_manifest_rows,
    write_manifest as write_phase0_manifest,
)
from harness.visual_readback.manifest import (
    VisualReadbackManifestError,
    compute_event_gated_phase0_manifest_sha256,
    load_event_gated_phase0_manifest_jsonl,
    load_visual_readback_manifest,
    select_v5_replacement_hits,
)
from harness.visual_readback.replay import (
    assert_live_hits_match_manifest,
    build_replay_payload,
)


def manifest_doc(tmp_path, **overrides):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    replacement = tmp_path / "replacement.png"
    for path in (current, memory, replacement):
        path.write_text(path.stem, encoding="utf-8")
    doc = {
        "manifest_version": "visual_readback_fixed_v1",
        "generation_run_id": "run-1",
        "primary_set_name": "phase1b_primary",
        "selection_basis": "candidate_case_set",
        "oracle_selection_fields": [],
        "shuffle_scope": "same_scene_other_episode",
        "shuffle_seed": 123,
        "negative_control_pool": [
            {
                "memory_id": "wrong-1",
                "image_path": str(replacement),
                "negative_control_label": "wrong",
                "scene_id": "s1",
                "episode_id": "other",
            }
        ],
        "cases": [
            {
                "case_id": "case-1",
                "scene_id": "s1",
                "episode_id": "e1",
                "step_id": 4,
                "trigger_rule": "decision_point",
                "trigger_source": "online_controller",
                "candidate_action_from_v0": "TURN_RIGHT",
                "current_image_path": str(current),
                "query_text": "left opening",
                "memory_hits": [
                    {
                        "memory_id": "m1",
                        "image_path": str(memory),
                        "confidence": 0.88,
                    }
                ],
                "frozen_policy_inputs": {
                    "instruction": "go to the kitchen",
                    "recent_frames": ["frame0"],
                },
                "required_verifier_labels": ["route_conflict"],
                "required_images": "current_and_memory",
                "adjudication": {
                    "route_conflict": "correct",
                },
            }
        ],
    }
    doc.update(overrides)
    return doc


def write_manifest(tmp_path, doc):
    path = tmp_path / "visual_readback_fixed_manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def phase0_seed(tmp_path):
    source = tmp_path / "trace.jsonl"
    source.write_text('{"ok": true}\n', encoding="utf-8")
    return {
        "cases": [{"scene_id": "s1", "episode_id": "e1", "max_steps": 100}],
        "route_event_labels": {
            "s1:e1": [{"event_type": "TURN_RIGHT", "step_id": 4}]
        },
        "shared_readback_trigger_keys": {"s1:e1": ["s1:e1:4:decision_point"]},
        "accepted_step_tolerance": 2,
        "phase0_stop_go_thresholds": {
            "route_event_miss_rate_max": 0.0,
            "exact_duplicate_rate_max": 0.1,
            "adjacent_triplet_rate_max": 0.2,
            "ambiguous_evidence_rate_max": 0.0,
            "comparison_direction": "lower_is_better",
            "invalid_comparison_behavior": "mark_invalid",
        },
        "selection_rules": "predeclared_test_seed",
        "selection_seed": 123,
        "trigger_key_source": "fixture",
        "source_dataset_id": "dataset-v1",
        "source_run_id": "run-1",
        "source_file_hashes": {"trace.jsonl": "abc123"},
    }


def write_phase0_jsonl(tmp_path, rows):
    path = tmp_path / "phase0.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_valid_manifest_loads_and_preserves_case_order(tmp_path):
    doc = manifest_doc(tmp_path)
    doc["cases"].append({**doc["cases"][0], "case_id": "case-2", "step_id": 5})

    manifest = load_visual_readback_manifest(write_manifest(tmp_path, doc))

    assert manifest.manifest_version == "visual_readback_fixed_v1"
    assert [case["case_id"] for case in manifest.cases] == ["case-1", "case-2"]


def test_event_gated_phase0_manifest_jsonl_validates_and_recomputes_hash(tmp_path):
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps(phase0_seed(tmp_path)), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "seed_file": str(seed_path),
            "episode_key": [],
            "route_event_labels_json": "",
            "shared_readback_trigger_keys_json": "",
            "phase0_stop_go_thresholds_json": "",
            "source_file": [],
            "max_steps": 100,
            "accepted_step_tolerance": 2,
            "selection_seed": 0,
            "selection_rules": "",
            "trigger_key_source": "",
            "source_dataset_id": "",
            "source_run_id": "",
        },
    )()
    rows = build_manifest_rows(args)
    output_path = tmp_path / "phase0_manifest.jsonl"

    write_phase0_manifest(rows, output_path)
    manifest = load_event_gated_phase0_manifest_jsonl(output_path)

    assert manifest.manifest_schema_version == "event_gated_phase0_v1"
    assert manifest.manifest_sha256 == rows[0]["manifest_sha256"]
    assert compute_event_gated_phase0_manifest_sha256(rows) == rows[0]["manifest_sha256"]
    assert manifest.cases[0]["phase0_stop_go_thresholds"]["comparison_direction"] == "lower_is_better"


def test_event_gated_phase0_manifest_rejects_missing_thresholds(tmp_path):
    row = phase0_seed(tmp_path)
    case = row["cases"][0]
    del row["phase0_stop_go_thresholds"]
    rows = [
        {
            "manifest_schema_version": "event_gated_phase0_v1",
            "scene_id": case["scene_id"],
            "episode_id": case["episode_id"],
            "max_steps": case["max_steps"],
            "route_event_labels": row["route_event_labels"]["s1:e1"],
            "accepted_step_tolerance": 2,
            "shared_readback_trigger_keys": row["shared_readback_trigger_keys"]["s1:e1"],
            "trigger_key_source": "fixture",
            "selection_rules": "predeclared_test_seed",
            "selection_seed": 123,
            "source_dataset_id": "dataset-v1",
            "source_run_id": "run-1",
            "source_file_hashes": {"trace.jsonl": "abc123"},
        }
    ]

    with pytest.raises(VisualReadbackManifestError, match="phase0_stop_go_thresholds"):
        load_event_gated_phase0_manifest_jsonl(write_phase0_jsonl(tmp_path, rows))


def test_event_gated_phase0_manifest_hash_changes_with_canonical_rows(tmp_path):
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps(phase0_seed(tmp_path)), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "seed_file": str(seed_path),
            "episode_key": [],
            "route_event_labels_json": "",
            "shared_readback_trigger_keys_json": "",
            "phase0_stop_go_thresholds_json": "",
            "source_file": [],
            "max_steps": 100,
            "accepted_step_tolerance": 2,
            "selection_seed": 0,
            "selection_rules": "",
            "trigger_key_source": "",
            "source_dataset_id": "",
            "source_run_id": "",
        },
    )()
    rows = build_manifest_rows(args)
    original = rows[0]["manifest_sha256"]
    rows[0]["accepted_step_tolerance"] = 3

    assert compute_event_gated_phase0_manifest_sha256(rows) != original


def test_manifest_rejects_missing_required_case_fields(tmp_path):
    doc = manifest_doc(tmp_path)
    del doc["cases"][0]["frozen_policy_inputs"]

    with pytest.raises(VisualReadbackManifestError, match="frozen_policy_inputs"):
        load_visual_readback_manifest(write_manifest(tmp_path, doc))


def test_v5_replacement_selection_is_deterministic_and_marks_unjudgeable(tmp_path):
    manifest = load_visual_readback_manifest(write_manifest(tmp_path, manifest_doc(tmp_path)))
    case = manifest.cases[0]

    first = select_v5_replacement_hits(manifest, case)
    second = select_v5_replacement_hits(manifest, case)

    assert first == second
    assert first[0]["memory_id"] == "wrong-1"

    no_pool = load_visual_readback_manifest(
        write_manifest(tmp_path, manifest_doc(tmp_path, negative_control_pool=[]))
    )
    assert select_v5_replacement_hits(no_pool, no_pool.cases[0]) == []
    assert no_pool.cases[0]["negative_control_label"] == "negative_control_unjudgeable"


def test_live_query_mismatch_is_reported_for_replay_denominators(tmp_path):
    manifest = load_visual_readback_manifest(write_manifest(tmp_path, manifest_doc(tmp_path)))
    case = manifest.cases[0]

    result = assert_live_hits_match_manifest(
        case,
        [{"memory_id": "different", "image_path": "/tmp/different.png"}],
    )

    assert result["status"] == "replay_mismatch"
    assert result["expected_memory_ids"] == ["m1"]
    assert result["actual_memory_ids"] == ["different"]


def test_replay_payloads_do_not_reuse_v0_action_and_v4c_is_current_only(tmp_path):
    manifest = load_visual_readback_manifest(write_manifest(tmp_path, manifest_doc(tmp_path)))
    case = manifest.cases[0]

    v1 = build_replay_payload(case, "text_only_prompt")
    v4 = build_replay_payload(case, "image_read_controller")
    v4c = build_replay_payload(case, "current_only_controller")

    assert "action_text" not in v1["policy_inputs"]
    assert v1["candidate_action_from_v0"] == "TURN_RIGHT"
    assert v4["memory_hits"][0]["memory_id"] == "m1"
    assert v4c["memory_hits"] == []
    assert v4c["current_image_path"] == case["current_image_path"]
    assert v4c["current_only"] is True
