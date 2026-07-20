from pathlib import Path

import pytest

from harness.memory.episode_visual_store import (
    EpisodeVisualMemoryStore,
    VisualMemoryOperationStatus,
)


def add_record(store, image_path, *, step_id, stage_id="stage_00", **overrides):
    data = {
        "image_path": str(image_path),
        "step_id": step_id,
        "stage_id": stage_id,
        "visual_summary": "A bright doorway leads into the kitchen.",
        "landmarks": ["doorway", "kitchen"],
        "action_before_capture": "MOVE_FORWARD",
        "action_after_capture": "TURN_LEFT",
        "odometry_snapshot": {"translation_since_stage_entry_m": 0.5},
        "image_roles": ["keyframe"],
        "provenance": ["runtime:current_rgb"],
        "importance": 0.7,
    }
    data.update(overrides)
    return store.add_observation(**data)


def test_episode_store_resets_between_episodes_and_evicts_oldest(tmp_path):
    store = EpisodeVisualMemoryStore(capacity=2)
    store.start_episode("scene-a", "episode-1")
    first = add_record(store, tmp_path / "one.png", step_id=1)
    add_record(store, tmp_path / "two.png", step_id=2)
    third = add_record(store, tmp_path / "three.png", step_id=3)

    assert [record.memory_id for record in store.records] == [
        "mem_000002",
        third.memory_id,
    ]
    assert first.memory_id not in {record.memory_id for record in store.records}

    store.start_episode("scene-a", "episode-2")
    assert store.records == ()
    assert store.episode_key == "scene-a:episode-2"


def test_same_canonical_path_merges_roles_and_provenance(tmp_path):
    store = EpisodeVisualMemoryStore(capacity=4)
    store.start_episode("scene", "episode")
    image_path = tmp_path / "frame.png"
    first = add_record(store, image_path, step_id=1)
    merged = add_record(
        store,
        Path(str(tmp_path / "." / "frame.png")),
        step_id=2,
        image_roles=["confirmed_landmark"],
        provenance=["runtime:qwen_semantic_promotion"],
    )

    assert merged.memory_id == first.memory_id
    assert len(store.records) == 1
    assert set(merged.image_roles) == {"keyframe", "confirmed_landmark"}
    assert set(merged.provenance) == {
        "runtime:current_rgb",
        "runtime:qwen_semantic_promotion",
    }


def test_store_links_next_observed_executed_action_to_capture(tmp_path):
    store = EpisodeVisualMemoryStore()
    store.start_episode("scene", "episode")
    record = add_record(
        store, tmp_path / "frame.png", step_id=4, action_after_capture=""
    )

    updated = store.record_action_after_capture(4, "MOVE_FORWARD")

    assert updated == 1
    assert store.records[0].memory_id == record.memory_id
    assert store.records[0].action_after_capture == "MOVE_FORWARD"


def test_registry_and_semantic_query_are_independent_and_stage_aware(tmp_path):
    store = EpisodeVisualMemoryStore(capacity=8)
    store.start_episode("scene", "episode")
    stage_match = add_record(
        store,
        tmp_path / "stage0.png",
        step_id=1,
        stage_id="stage_00",
        landmarks=["stairs"],
        visual_summary="Stairs beside the turning point.",
    )
    add_record(
        store,
        tmp_path / "recent.png",
        step_id=9,
        stage_id="stage_01",
        landmarks=["sofa"],
        visual_summary="A sofa in the next room.",
        importance=1.0,
    )

    registry = store.select_registry_evidence(
        active_stage_id="stage_00",
        expected_landmarks=["stairs"],
        trigger_reasons=["stage_completion_candidate"],
        limit=1,
    )
    semantic = store.query_semantic(
        "Did we pass the stairs?",
        active_stage_id="stage_00",
        expected_landmarks=["stairs"],
        trigger_reasons=["stage_completion_candidate"],
        limit=1,
    )

    assert registry.operation == "registry"
    assert semantic.operation == "semantic_query"
    assert registry.status is VisualMemoryOperationStatus.SELECTED
    assert semantic.status is VisualMemoryOperationStatus.SELECTED
    assert registry.records[0].memory_id == stage_match.memory_id
    assert semantic.records[0].memory_id == stage_match.memory_id
    assert semantic.to_memory_hits()[0].image_path == str(
        (tmp_path / "stage0.png").resolve()
    )


def test_store_reports_empty_attempt_and_rejects_oracle_fields(tmp_path):
    store = EpisodeVisualMemoryStore()
    store.start_episode("scene", "episode")

    result = store.query_semantic("doorway", active_stage_id="stage_00")

    assert result.status is VisualMemoryOperationStatus.EMPTY
    assert result.attempted is True
    assert result.image_backed_hit_count == 0

    with pytest.raises(ValueError, match="Forbidden oracle keys"):
        add_record(
            store,
            tmp_path / "oracle.png",
            step_id=1,
            odometry_snapshot={"distance_to_goal": 0.1},
        )
