from harness.config import HarnessConfig
from harness.memory.memory_manager import MemoryManager
from harness.memory.spatial_memory_client import FakeSpatialMemoryClient


def test_memory_manager_splits_contexts():
    manager = MemoryManager(FakeSpatialMemoryClient(), HarnessConfig())
    result = manager.recall(text="kitchen", step_id=0, reason="initial")
    assert result.policy_context["memory_context_text"]
    assert "hits" in result.control_context
    assert "target_poses" in result.executor_context


def test_recall_interval_blocks_overcalling():
    cfg = HarnessConfig(recall_interval_steps=5)
    manager = MemoryManager(FakeSpatialMemoryClient(), cfg)
    assert manager.should_recall(step_id=0, reason="initial") is True
    manager.mark_recalled(step_id=0)
    assert manager.should_recall(step_id=2, reason="periodic") is False


def test_memory_manager_proposes_write_without_side_effect():
    manager = MemoryManager(FakeSpatialMemoryClient(), HarnessConfig())
    decision = manager.propose_write(step_id=0, image_path="frame.jpg", note="start")
    assert decision["should_write"] is True
    assert decision["write_type"] == "episodic_keyframe"
