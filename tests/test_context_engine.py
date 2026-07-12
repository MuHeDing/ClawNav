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


def test_context_engine_decision_log_records_memory_and_control_usage(tmp_path):
    engine = MemoryAwareContextEngine(tmp_path)

    record = engine.record_step(
        run_id=str(tmp_path),
        scene_id="s1",
        episode_id="e1",
        instruction="go to the kitchen",
        step_id=3,
        payload={
            "current_image_path": "/tmp/step3.png",
            "memory_context_text": "remember doorway",
            "memory_images": ["/tmp/memory0.png"],
            "control_context": {"recent_forward_count": 4},
            "evidence_context": {"has_current_image": True},
        },
        action_text="TURN_LEFT",
        planner_reason="forward stall gate",
        ok=True,
    )

    row = json.loads((tmp_path / "decision_log.jsonl").read_text(encoding="utf-8"))
    assert record["memory_context_used"] is True
    assert record["control_context_used"] is True
    assert row["memory_context_used"] is True
    assert row["control_context_used"] is True
    assert row["evidence_context_used"] is True


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


def test_context_engine_returns_retrieved_memory_image_paths(tmp_path):
    engine = MemoryAwareContextEngine(tmp_path)
    first_id = engine.add_memory(
        text="bright doorway leads toward the kitchen",
        scene_id="s1",
        episode_id="e1",
        step_id=2,
        image_path="/tmp/memory0.png",
        tags=["doorway"],
        importance=0.8,
    )
    second_id = engine.add_memory(
        text="kitchen doorway remains visible after the turn",
        scene_id="s1",
        episode_id="e1",
        step_id=4,
        image_path="/tmp/memory1.png",
        tags=["kitchen"],
        importance=0.6,
    )
    engine.add_memory(
        text="matching text without an image path",
        scene_id="s1",
        episode_id="e1",
        step_id=5,
        tags=["kitchen"],
        importance=0.1,
    )

    context = engine.prepare_plan_context(
        run_id=str(tmp_path),
        scene_id="s1",
        episode_id="e1",
        instruction="find kitchen doorway",
        step_id=6,
        payload={"memory_query": "kitchen doorway"},
    )

    assert context["retrieved_memory_ids"][:2] == [first_id, second_id]
    assert context["memory_images"] == ["/tmp/memory0.png", "/tmp/memory1.png"]
    assert context["retrieved_memory_image_paths"] == [
        "/tmp/memory0.png",
        "/tmp/memory1.png",
    ]
    assert context["retrieved_memory_images"] == [
        {
            "memory_id": first_id,
            "image_path": "/tmp/memory0.png",
            "source": "openclaw_retrieved_memory",
            "step_id": 2,
        },
        {
            "memory_id": second_id,
            "image_path": "/tmp/memory1.png",
            "source": "openclaw_retrieved_memory",
            "step_id": 4,
        },
    ]


def test_context_engine_returns_up_to_eight_retrieved_memory_images(tmp_path):
    engine = MemoryAwareContextEngine(tmp_path)
    for index in range(9):
        engine.add_memory(
            text=f"central archway keyframe {index}",
            scene_id="s1",
            episode_id="e1",
            step_id=index,
            image_path=f"/tmp/keyframe{index}.png",
            tags=["archway"],
            importance=0.8,
        )

    context = engine.prepare_plan_context(
        run_id=str(tmp_path),
        scene_id="s1",
        episode_id="e1",
        instruction="walk to the archway",
        step_id=10,
        payload={"memory_query": "central archway"},
    )

    assert len(context["retrieved_memory_image_paths"]) == 8
    assert context["retrieved_memory_image_paths"][0] == "/tmp/keyframe8.png"
    assert context["retrieved_memory_image_paths"][-1] == "/tmp/keyframe1.png"


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
