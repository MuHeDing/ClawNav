from pathlib import Path

from harness.memory.protocol import (
    SPATIAL_MEMORY_PROTOCOL_VERSION,
    build_query_payload,
    normalize_memory_result,
)


def test_build_query_payload_includes_protocol_fields():
    payload = build_query_payload(
        query_type="semantic",
        text="kitchen",
        n_results=3,
        memory_source="episode-local",
    )

    assert payload["protocol_version"] == SPATIAL_MEMORY_PROTOCOL_VERSION
    assert payload["query_type"] == "semantic"
    assert payload["text"] == "kitchen"
    assert payload["n_results"] == 3
    assert payload["memory_source"] == "episode-local"


def test_normalize_memory_result_maps_evidence_fields():
    result = normalize_memory_result(
        {
            "id": "m1",
            "memory_type": "place",
            "name": "kitchen",
            "confidence": 0.9,
            "target_pose": {"x": 1.0},
            "evidence": {"text": "kitchen entrance", "image_path": "/tmp/a.jpg"},
        },
        memory_source="scene-prior",
    )

    assert result["memory_id"] == "m1"
    assert result["evidence_text"] == "kitchen entrance"
    assert result["image_path"] == "/tmp/a.jpg"
    assert result["memory_source"] == "scene-prior"


def test_normalize_memory_result_rejects_oracle_metadata():
    result = normalize_memory_result(
        {
            "id": "m1",
            "metadata": {"distance_to_goal": 1.0},
        },
        memory_source="episode-local",
    )

    assert "distance_to_goal" not in result["metadata"]


def test_spatial_memory_protocol_doc_exists_and_lists_endpoints():
    doc = Path("docs/protocols/spatial-memory-protocol.md")
    text = doc.read_text(encoding="utf-8")

    for endpoint in [
        "GET /health",
        "POST /query/object",
        "POST /query/place",
        "POST /query/position",
        "POST /query/semantic/text",
        "POST /query/unified",
        "POST /memory/semantic/ingest",
    ]:
        assert endpoint in text

    assert "memory_source" in text
    assert "No Oracle Fields" in text
