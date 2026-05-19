from harness.skills.visual_memory_curator import VisualMemoryCuratorSkill


def test_visual_memory_curator_skips_duplicate_recent_memory():
    skill = VisualMemoryCuratorSkill()
    payload = {
        "caption": "A red door next to a sofa.",
        "objects": ["sofa"],
        "landmarks": ["red door"],
        "spatial_cues": ["sofa on right"],
        "confidence": 0.9,
        "recent_visual_memories": [
            {
                "memory_id": "mem-1",
                "caption": "A red door next to a sofa.",
                "objects": ["sofa"],
                "landmarks": ["red door"],
                "spatial_cues": ["sofa on right"],
            }
        ],
    }

    result = skill.run(state=None, payload=payload)

    gate = result.payload["write_gate"]
    assert result.payload["should_write"] is False
    assert gate["curator_decision"] == "skip"
    assert gate["duplicate_of_memory_id"] == "mem-1"
    assert gate["novelty_score"] < 0.25


def test_visual_memory_curator_writes_novel_landmark():
    skill = VisualMemoryCuratorSkill()
    payload = {
        "caption": "A stairway descends at the end of the corridor.",
        "landmarks": ["stairway"],
        "spatial_cues": ["corridor ends at stairs"],
        "confidence": 0.86,
        "recent_visual_memories": [
            {"memory_id": "mem-1", "landmarks": ["red door"], "objects": ["sofa"]}
        ],
    }

    result = skill.run(state=None, payload=payload)

    assert result.payload["should_write"] is True
    assert result.payload["write_gate"]["novelty_score"] > 0.5
    assert result.payload["write_gate"]["duplicate_of_memory_id"] is None
