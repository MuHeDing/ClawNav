import json

import pytest

from scripts.replay_staged_visual_memory_shadow import replay_shadow_snapshots


def test_replay_runs_two_replicas_per_arm_and_reports_stability(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": "staged_shadow_input_v1",
                "scene_id": "s1",
                "episode_id": "e1",
                "step_id": 3,
                "memory_event_id": "event-1",
                "stage_state": {
                    "full_instruction": "go",
                    "active_stage_id": "stage_00",
                },
                "input_context": {
                    "active_stage_id": "stage_00",
                    "retrieved_memory_images": [
                        {"image_path": "/tmp/mem.png", "memory_id": "m1"}
                    ],
                },
                "full_image_role_manifest": [],
                "non_memory_image_role_manifest": [],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def post(payload):
        calls.append(payload)
        return {"intent": "act", "arguments": {"action_text": "TURN_LEFT"}}

    report = replay_shadow_snapshots([snapshot], post, replicas=2)

    assert len(calls) == 4
    assert report["row_count"] == 4
    assert report["event_count"] == 1
    assert report["arms"]["memory_on"]["stable_event_count"] == 1
    assert report["arms"]["memory_off"]["stable_event_count"] == 1
    assert "retrieved_memory_images" in calls[0]["runtime_context"]
    assert "retrieved_memory_images" not in calls[2]["runtime_context"]


def test_replay_rejects_output_inside_primary_result_path(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()

    with pytest.raises(ValueError):
        replay_shadow_snapshots(
            [],
            lambda _: {},
            replicas=2,
            output_dir=primary / "shadow",
            primary_result_path=primary,
        )


def test_replay_distinguishes_unstable_provider_replicas(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": "staged_shadow_input_v1",
                "scene_id": "s1",
                "episode_id": "e1",
                "step_id": 1,
                "memory_event_id": "event-unstable",
                "stage_state": {"full_instruction": "go"},
                "input_context": {},
            }
        ),
        encoding="utf-8",
    )
    counter = {"value": 0}

    def post(_):
        counter["value"] += 1
        return {"arguments": {"action_text": f"TURN_{counter['value']}"}}

    report = replay_shadow_snapshots([snapshot], post, replicas=2)

    assert report["arms"]["memory_on"]["unstable_event_count"] == 1
    assert report["arms"]["memory_off"]["unstable_event_count"] == 1
