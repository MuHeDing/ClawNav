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
