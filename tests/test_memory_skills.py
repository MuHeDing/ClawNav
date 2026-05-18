from harness.config import HarnessConfig
from harness.memory.memory_manager import MemoryManager
from harness.memory.spatial_memory_client import FakeSpatialMemoryClient
from harness.skills.memory_query import MemoryQuerySkill
from harness.skills.memory_write import MemoryWriteSkill


class InMemoryStore:
    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(record)


def test_memory_query_skill_returns_memory_hits():
    manager = MemoryManager(FakeSpatialMemoryClient(), HarnessConfig())
    skill = MemoryQuerySkill(manager)
    result = skill.run(None, {"text": "kitchen", "step_id": 0, "reason": "initial"})
    assert result.ok is True
    assert result.result_type == "memory_query"
    assert result.payload["memory_hits"]
    assert result.payload["policy_context"]["memory_context_text"]


def test_memory_query_skill_forwards_visual_recall_context():
    class RecordingManager:
        def __init__(self):
            self.kwargs = None

        def recall(self, **kwargs):
            self.kwargs = kwargs
            return type(
                "Recall",
                (),
                {
                    "hits": [],
                    "query": kwargs["text"],
                    "backend": "fake",
                    "policy_context": {},
                    "control_context": {"confidence": 0.0},
                    "executor_context": {},
                },
            )()

    manager = RecordingManager()
    skill = MemoryQuerySkill(manager)

    result = skill.run(
        None,
        {
            "text": "go to kitchen",
            "step_id": 3,
            "active_subgoal": "find doorway",
            "visual_observation": "sofa on right",
            "planner_reason": "uncertain",
            "critic_signal": "oscillation",
            "allowed_scopes": ["episode"],
            "memory_namespace": "episode:s1:e1",
        },
    )

    assert result.ok is True
    assert manager.kwargs["active_subgoal"] == "find doorway"
    assert manager.kwargs["visual_observation"] == "sofa on right"
    assert manager.kwargs["allowed_scopes"] == ["episode"]
    assert manager.kwargs["memory_namespace"] == "episode:s1:e1"


def test_memory_write_skill_stores_episodic_records():
    store = InMemoryStore()
    skill = MemoryWriteSkill(store=store, allowed_sources={"episode-local"})
    result = skill.run(
        None,
        {
            "should_write": True,
            "write_type": "episodic_keyframe",
            "step_id": 0,
            "image_path": "frame.jpg",
            "note": "start",
            "memory_source": "episode-local",
        },
    )
    assert result.ok is True
    assert len(store.records) == 1
    assert store.records[0]["image_path"] == "frame.jpg"


def test_memory_write_skill_accepts_manager_proposed_payload():
    store = InMemoryStore()
    manager = MemoryManager(FakeSpatialMemoryClient(), HarnessConfig())
    payload = manager.propose_write(step_id=2, image_path="frame2.jpg", note="turn")
    result = MemoryWriteSkill(store=store).run(None, payload)
    assert result.ok is True
    assert store.records[0]["step_id"] == 2


def test_memory_write_skill_validates_memory_source():
    store = InMemoryStore()
    skill = MemoryWriteSkill(store=store, allowed_sources={"episode-local"})
    result = skill.run(
        None,
        {
            "should_write": True,
            "write_type": "episodic_keyframe",
            "step_id": 0,
            "memory_source": "oracle",
        },
    )
    assert result.ok is False
    assert "memory_source" in result.error
    assert store.records == []


def test_memory_write_skill_stores_visual_memory_fields_and_retrieval_text():
    store = InMemoryStore()
    skill = MemoryWriteSkill(store=store, allowed_sources={"episode-local"})

    result = skill.run(
        None,
        {
            "should_write": True,
            "step_id": 4,
            "scene_id": "s1",
            "episode_id": "e1",
            "image_path": "frame.jpg",
            "caption": "A hallway with a doorway ahead.",
            "visual_observation": "Doorway ahead can anchor navigation.",
            "objects": ["doorway", "sofa"],
            "landmarks": ["doorway"],
            "place_category": "hallway",
            "spatial_cues": ["doorway ahead", "sofa on right"],
            "navigation_relevance": "Useful route anchor.",
            "source_image_role": "keyframe",
            "memory_scope": "episode",
            "write_gate": {
                "candidate_reason": "landmark",
                "curator_decision": "write",
            },
            "memory_source": "episode-local",
        },
    )

    assert result.ok is True
    record = store.records[0]
    assert record["caption"] == "A hallway with a doorway ahead."
    assert record["objects"] == ["doorway", "sofa"]
    assert record["landmarks"] == ["doorway"]
    assert record["memory_namespace"] == "episode:s1:e1"
    assert "Doorway ahead can anchor navigation." in record["retrieval_text"]
    assert "sofa" in record["retrieval_text"]
    assert record["write_gate"]["candidate_reason"] == "landmark"


def test_memory_write_skill_skips_when_write_gate_rejects_candidate():
    store = InMemoryStore()
    skill = MemoryWriteSkill(store=store, allowed_sources={"episode-local"})

    result = skill.run(
        None,
        {
            "should_write": True,
            "image_path": "frame.jpg",
            "write_gate": {
                "curator_decision": "skip",
                "curator_reason": "duplicate generic hallway",
            },
            "memory_source": "episode-local",
        },
    )

    assert result.ok is True
    assert result.payload["written"] is False
    assert result.payload["skipped"] is True
    assert result.payload["skip_reason"] == "duplicate generic hallway"
    assert store.records == []
