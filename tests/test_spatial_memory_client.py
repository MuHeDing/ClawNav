from harness.memory.spatial_memory_client import FakeSpatialMemoryClient, SpatialMemoryHttpClient


def test_fake_spatial_memory_returns_episode_local_hits():
    client = FakeSpatialMemoryClient(memory_source="episode-local")
    hits = client.query_semantic("kitchen", n_results=2)
    assert len(hits) <= 2
    assert hits[0].memory_source == "episode-local"


def test_http_client_maps_results(monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "m1",
                        "memory_type": "place",
                        "name": "kitchen",
                        "target_pose": {"x": 1.0, "y": 2.0},
                        "confidence": 0.9,
                        "source": "test",
                        "timestamp": 1.0,
                        "evidence": {"note": "kitchen entrance", "image_path": "/tmp/a.jpg"},
                    }
                ]
            }

    def fake_post(url, json, timeout):
        return Resp()

    monkeypatch.setattr("requests.post", fake_post)
    client = SpatialMemoryHttpClient("http://memory", memory_source="scene-prior")
    hits = client.query_semantic("kitchen")
    assert hits[0].memory_id == "m1"
    assert hits[0].target_pose["x"] == 1.0
    assert hits[0].memory_source == "scene-prior"


def test_spatial_http_semantic_query_sends_scope_filters(monkeypatch):
    captured = {}

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    monkeypatch.setattr("requests.post", fake_post)
    client = SpatialMemoryHttpClient("http://memory", memory_source="episode-local")

    client.query_semantic(
        "doorway",
        n_results=3,
        allowed_scopes=["episode"],
        memory_namespace="episode:s1:e1",
    )

    assert captured["url"] == "http://memory/query/semantic/text"
    assert captured["json"]["allowed_scopes"] == ["episode"]
    assert captured["json"]["memory_namespace"] == "episode:s1:e1"


def test_spatial_http_ingest_semantic_maps_clawnav_record_to_service_schema(monkeypatch, tmp_path):
    captured = {}
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"png")

    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "id": "sem_1"}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    monkeypatch.setattr("requests.post", fake_post)
    client = SpatialMemoryHttpClient("http://memory", memory_source="episode-local")

    result = client.ingest_semantic(
        {
            "image_path": str(image_path),
            "scene_id": "scene-a",
            "episode_id": "episode-1",
            "step_id": 5,
            "caption": "A hallway.",
            "visual_observation": "Doorway ahead.",
            "navigation_relevance": "Use the doorway as a route anchor.",
            "objects": ["door"],
            "landmarks": ["doorway"],
            "spatial_cues": ["ahead"],
            "memory_source": "episode-local",
            "retrieval_text": "Doorway ahead.",
        }
    )

    assert result["ok"] is True
    assert captured["url"] == "http://memory/memory/semantic/ingest"
    assert captured["json"]["image"] == str(image_path.resolve())
    assert captured["json"]["robot_id"] == "clawnav"
    assert captured["json"]["robot_type"] == "habitat_vln"
    assert captured["json"]["source"] == "episode-local"
    assert captured["json"]["robot_pose"] == {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "frame_id": "map",
    }
    assert "scene-a" in captured["json"]["note"]
    assert "Doorway ahead." in captured["json"]["note"]
    assert "doorway" in captured["json"]["tags"]
