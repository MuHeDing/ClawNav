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
