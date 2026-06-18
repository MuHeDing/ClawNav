import json

import pytest

from harness.visual_readback.manifest import (
    VisualReadbackManifestError,
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


def test_valid_manifest_loads_and_preserves_case_order(tmp_path):
    doc = manifest_doc(tmp_path)
    doc["cases"].append({**doc["cases"][0], "case_id": "case-2", "step_id": 5})

    manifest = load_visual_readback_manifest(write_manifest(tmp_path, doc))

    assert manifest.manifest_version == "visual_readback_fixed_v1"
    assert [case["case_id"] for case in manifest.cases] == ["case-1", "case-2"]


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
