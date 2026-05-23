import json

from harness.memory.context_engine import MemoryAwareContextEngine


def test_context_engine_records_task_state_and_recent_step_summary(tmp_path):
    engine = MemoryAwareContextEngine(tmp_path)

    first = engine.prepare_plan_context(
        run_id=str(tmp_path),
        scene_id="s1",
        episode_id="e1",
        instruction="go to the kitchen",
        step_id=0,
        payload={"current_image_path": "/tmp/step0.png"},
    )
    assert first["task_state"]["current_step_id"] == 0
    assert "recent_step_summary" not in first

    metadata = engine.record_step(
        run_id=str(tmp_path),
        scene_id="s1",
        episode_id="e1",
        instruction="go to the kitchen",
        step_id=0,
        payload={"current_image_path": "/tmp/step0.png"},
        action_text="TURN_LEFT",
        planner_reason="look for kitchen doorway",
        ok=True,
    )

    assert metadata["recorded"] is True
    assert (tmp_path / "task_state.json").exists()
    assert (tmp_path / "running_summary.md").exists()
    task_state = json.loads((tmp_path / "task_state.json").read_text())
    assert task_state["current_step_id"] == 0
    assert task_state["last_action_text"] == "TURN_LEFT"
    assert "Step 0: TURN_LEFT" in (tmp_path / "running_summary.md").read_text()

    second = engine.prepare_plan_context(
        run_id=str(tmp_path),
        scene_id="s1",
        episode_id="e1",
        instruction="go to the kitchen",
        step_id=1,
        payload={},
    )
    assert "Step 0: TURN_LEFT" in second["recent_step_summary"]
    assert second["task_state"]["last_action_text"] == "TURN_LEFT"


def test_context_engine_retrieves_episode_memory_with_hybrid_score(tmp_path):
    engine = MemoryAwareContextEngine(tmp_path)
    matching_id = engine.add_memory(
        text="bright doorway leads toward the kitchen",
        scene_id="s1",
        episode_id="e1",
        step_id=2,
        image_path="/tmp/doorway.png",
        tags=["doorway"],
        importance=0.4,
    )
    engine.add_memory(
        text="same scene but wrong episode kitchen note",
        scene_id="s1",
        episode_id="other",
        step_id=1,
        importance=1.0,
    )

    context = engine.prepare_plan_context(
        run_id=str(tmp_path),
        scene_id="s1",
        episode_id="e1",
        instruction="find kitchen doorway",
        step_id=3,
        payload={"memory_query": "kitchen doorway"},
    )

    assert context["retrieved_memory_ids"] == [matching_id]
    assert "bright doorway leads toward the kitchen" in context["memory_context_text"]
    assert "wrong episode" not in context["memory_context_text"]
    assert (tmp_path / "memory" / "episode_e1.md").exists()


def test_context_engine_review_compacts_duplicates_and_writes_dreams(tmp_path):
    engine = MemoryAwareContextEngine(tmp_path)
    keep_id = engine.add_memory(
        text="landmark: blue sofa near hallway",
        scene_id="s1",
        episode_id="e1",
        step_id=1,
        image_path="/tmp/sofa.png",
        importance=0.7,
    )
    engine.add_memory(
        text="landmark: blue sofa near hallway",
        scene_id="s1",
        episode_id="e1",
        step_id=2,
        image_path="/tmp/sofa.png",
        importance=0.6,
    )
    engine.add_memory(
        text="low value transient turn",
        scene_id="s1",
        episode_id="e1",
        step_id=3,
        importance=0.05,
    )

    review = engine.review_and_compact(current_step_id=25)

    assert review["kept_memory_ids"] == [keep_id]
    assert review["removed_count"] == 2
    assert "removed=2" in (tmp_path / "review_log.md").read_text()
    assert "Review unresolved navigation evidence" in (tmp_path / "DREAMS.md").read_text()
