from harness.config import HarnessConfig
from harness.memory.spatial_memory_client import BaseSpatialMemoryClient
from harness.memory.memory_manager import MemoryManager
from harness.memory.spatial_memory_client import FakeSpatialMemoryClient
from harness.types import MemoryHit


class RecordingMemoryClient(BaseSpatialMemoryClient):
    def __init__(self, hits):
        super().__init__(memory_source="episode-local")
        self.hits = hits
        self.queries = []

    def query_semantic(
        self,
        text: str,
        n_results: int = 5,
        allowed_scopes=None,
        memory_namespace: str = "",
        memory_source: str = "",
    ):
        self.queries.append(
            {
                "text": text,
                "n_results": n_results,
                "allowed_scopes": allowed_scopes,
                "memory_namespace": memory_namespace,
                "memory_source": memory_source,
            }
        )
        return self.hits[:n_results]


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


def test_memory_manager_recall_uses_visual_fields_in_policy_context():
    hit = MemoryHit(
        memory_id="m1",
        memory_type="place",
        name="hallway",
        confidence=0.9,
        evidence_text="remembered hallway",
        image_path="/tmp/keyframe.png",
        metadata={
            "caption": "A hallway with a doorway ahead.",
            "objects": ["doorway"],
            "landmarks": ["doorway"],
            "place_category": "hallway",
            "spatial_cues": ["doorway ahead"],
            "navigation_relevance": "Route anchor toward kitchen.",
            "memory_scope": "episode",
            "memory_namespace": "episode:s1:e1",
        },
    )
    manager = MemoryManager(RecordingMemoryClient([hit]), HarnessConfig())

    result = manager.recall(text="go to kitchen", step_id=3, reason="initial")

    assert "A hallway with a doorway ahead." in result.policy_context["memory_context_text"]
    assert "doorway ahead" in result.policy_context["memory_context_text"]
    assert result.policy_context["memory_images"] == ["/tmp/keyframe.png"]
    assert result.control_context["best_landmark"] == "doorway"
    assert result.control_context["recall_confidence"] == 0.9
    assert result.executor_context["topological_anchor"] == "doorway"


def test_memory_manager_compacts_visual_evidence_for_policy_context():
    verbose_visual_memory = """
The user wants a description of the image focused on navigation-relevant visual evidence.
I need to break down the image into key components:
1. Place Category: It is a large formal interior room.
2. Spatial Layout: A clear central corridor leads directly to the central archway.
3. Stable Landmarks: Wooden balcony above the archway and wall sconces flank the target.
Drafting the description:
The open floor in the middle is the primary forward path with no obstacle.
"""
    hit = MemoryHit(
        memory_id="m1",
        memory_type="place",
        name=verbose_visual_memory,
        confidence=0.82,
        image_path="/tmp/keyframe.png",
        metadata={
            "visual_observation": verbose_visual_memory,
            "memory_scope": "episode",
            "memory_namespace": "episode:s1:e1",
        },
    )
    manager = MemoryManager(
        RecordingMemoryClient([hit]),
        HarnessConfig(max_prompt_context_chars=1000),
    )

    result = manager.recall(text="wait at the archway", step_id=4)

    context = result.policy_context["memory_context_text"]
    assert "The user wants" not in context
    assert "I need to" not in context
    assert "Drafting the description" not in context
    assert "central corridor leads directly to the central archway" in context
    assert context.startswith("- place:")
    assert len(context) < 260


def test_memory_manager_recall_builds_query_from_visual_context_and_filters_namespace():
    matching = MemoryHit(
        memory_id="m1",
        memory_type="place",
        name="doorway",
        confidence=0.9,
        metadata={"memory_scope": "episode", "memory_namespace": "episode:s1:e1"},
    )
    other_episode = MemoryHit(
        memory_id="m2",
        memory_type="place",
        name="doorway",
        confidence=0.8,
        metadata={"memory_scope": "episode", "memory_namespace": "episode:s1:e2"},
    )
    client = RecordingMemoryClient([matching, other_episode])
    manager = MemoryManager(client, HarnessConfig())

    result = manager.recall(
        text="go to kitchen",
        step_id=4,
        reason="uncertain",
        active_subgoal="find doorway",
        visual_observation="sofa on right",
        planner_reason="planner uncertain",
        critic_signal="oscillation",
        allowed_scopes=["episode"],
        memory_namespace="episode:s1:e1",
    )

    query_text = client.queries[0]["text"]
    assert "go to kitchen" in query_text
    assert "find doorway" in query_text
    assert "sofa on right" in query_text
    assert "oscillation" in query_text
    assert [hit.memory_id for hit in result.hits] == ["m1"]


def test_memory_manager_strict_scope_drops_hits_without_scope_metadata():
    external_hit = MemoryHit(
        memory_id="m1",
        memory_type="semantic_frame",
        name="doorway",
        confidence=0.9,
        evidence_text="A doorway beside the kitchen.",
        metadata={},
    )
    manager = MemoryManager(RecordingMemoryClient([external_hit]), HarnessConfig())

    result = manager.recall(
        text="go to kitchen",
        step_id=4,
        allowed_scopes=["episode"],
        memory_namespace="episode:s1:e1",
    )

    assert result.hits == []


def test_memory_manager_passes_scope_filters_before_semantic_query():
    client = RecordingMemoryClient([])
    manager = MemoryManager(client, HarnessConfig())

    manager.recall(
        text="find doorway",
        step_id=4,
        allowed_scopes=["episode"],
        memory_namespace="episode:s1:e1",
    )

    assert client.queries[0]["allowed_scopes"] == ["episode"]
    assert client.queries[0]["memory_namespace"] == "episode:s1:e1"


def test_memory_manager_reranks_by_visual_overlap_and_confidence():
    generic = MemoryHit(
        memory_id="generic",
        memory_type="semantic",
        name="generic hallway",
        confidence=0.95,
        metadata={
            "landmarks": ["window"],
            "objects": ["chair"],
            "memory_scope": "episode",
        },
    )
    doorway = MemoryHit(
        memory_id="door",
        memory_type="semantic",
        name="doorway memory",
        confidence=0.70,
        metadata={
            "landmarks": ["red door"],
            "objects": ["sofa"],
            "memory_scope": "episode",
        },
    )
    manager = MemoryManager(RecordingMemoryClient([generic, doorway]), HarnessConfig())

    recall = manager.recall(
        text="go toward the red door near the sofa",
        step_id=3,
        visual_observation="red door ahead with sofa on right",
        allowed_scopes=["episode"],
    )

    assert [hit.memory_id for hit in recall.hits] == ["door", "generic"]


def test_memory_manager_reranks_recent_hit_when_scores_tie():
    older = MemoryHit(
        memory_id="old",
        memory_type="semantic",
        name="doorway",
        confidence=0.8,
        metadata={"step_id": 2},
    )
    newer = MemoryHit(
        memory_id="new",
        memory_type="semantic",
        name="doorway",
        confidence=0.8,
        metadata={"step_id": 9},
    )
    manager = MemoryManager(RecordingMemoryClient([older, newer]), HarnessConfig())

    recall = manager.recall("doorway", step_id=10)

    assert [hit.memory_id for hit in recall.hits] == ["new", "old"]
