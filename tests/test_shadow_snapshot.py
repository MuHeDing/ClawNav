import json

import pytest

from harness.openclaw.shadow_snapshot import (
    ShadowSnapshotWriter,
    build_image_role_manifests,
    sanitize_shadow_value,
)


def test_image_role_manifest_masks_only_memory_and_is_deterministic():
    context = {
        "map_context": {"map_image_path": "/tmp/map.png"},
        "retrieved_memory_images": [{"image_path": "/tmp/mem.png", "memory_id": "m1"}],
        "current_image_path": "/tmp/current.png",
    }

    full, masked = build_image_role_manifests(context)

    assert [item["image_role"] for item in full] == [
        "map_view",
        "retrieved_memory",
        "current",
    ]
    assert [item["image_role"] for item in masked] == ["map_view", "current"]
    assert build_image_role_manifests(context) == (full, masked)


def test_shadow_writer_selects_predeclared_trigger_before_outcome(tmp_path):
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps({"episode_key": "scene:episode", "trigger_classes": ["stage_entry"]})
        + "\n",
        encoding="utf-8",
    )
    writer = ShadowSnapshotWriter(tmp_path / "run", selection)

    result = writer.consider(
        scene_id="scene",
        episode_id="episode",
        step_id=3,
        memory_event_id="scene::episode:3:1",
        trigger_reasons=["stage_entry"],
        provider_call_id="primary",
        stage_state={"active_stage_id": "stage_00", "instruction_sha256": "abc"},
        runtime_context={
            "current_image_path": "/tmp/current.png",
            "retrieved_memory_images": [
                {"image_path": "/tmp/mem.png", "memory_id": "m1"}
            ],
            "distance_to_goal": 0.2,
        },
    )

    assert result.selected is True
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
    assert payload["provider_call_id"] == "primary"
    assert payload["selection_basis"]["trigger_reasons"] == ["stage_entry"]
    assert "executed_action" not in payload
    assert "distance_to_goal" not in json.dumps(payload)
    assert payload["non_memory_image_role_manifest"] == [
        {
            "image_path": "/tmp/current.png",
            "image_role": "current",
            "memory_id": "",
            "source": "current",
        }
    ]


def test_shadow_writer_records_not_selected_reason_and_first_event_only(tmp_path):
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {"episode_key": "scene:episode", "trigger_classes": ["stop_candidate"]}
        )
        + "\n",
        encoding="utf-8",
    )
    writer = ShadowSnapshotWriter(tmp_path / "run", selection)

    unrelated = writer.consider(
        scene_id="scene",
        episode_id="episode",
        step_id=1,
        memory_event_id="e1",
        trigger_reasons=["stage_entry"],
        provider_call_id="primary",
        stage_state={},
        runtime_context={},
    )
    selected = writer.consider(
        scene_id="scene",
        episode_id="episode",
        step_id=2,
        memory_event_id="e2",
        trigger_reasons=["stop_candidate"],
        provider_call_id="memory_requery",
        stage_state={},
        runtime_context={},
    )
    repeated = writer.consider(
        scene_id="scene",
        episode_id="episode",
        step_id=3,
        memory_event_id="e3",
        trigger_reasons=["stop_candidate"],
        provider_call_id="memory_requery",
        stage_state={},
        runtime_context={},
    )

    assert unrelated.reason == "trigger_not_predeclared"
    assert selected.selected is True
    assert repeated.reason == "trigger_already_selected"
    rows = [
        json.loads(line)
        for line in (tmp_path / "run/harness_traces/shadow_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 3
    assert sum(row["selected"] for row in rows) == 1


def test_shadow_sanitizer_rejects_sensitive_oracle_reasoning_and_image_bytes():
    value = {
        "token": "secret",
        "authorization": "Bearer secret",
        "reasoning_content": "hidden chain",
        "raw_error": "provider internals",
        "distance_to_goal": 0.1,
        "image_base64": "AAAA",
        "safe": {"active_stage_id": "stage_00"},
    }

    sanitized = sanitize_shadow_value(value)

    assert sanitized == {"safe": {"active_stage_id": "stage_00"}}
    with pytest.raises(ValueError):
        sanitize_shadow_value({"safe": "data:image/png;base64,AAAA"})
