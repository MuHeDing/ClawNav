import json

from scripts.generate_qwen_stage_plan_manifest import generate_stage_manifest


def test_generate_stage_manifest_is_deterministic_and_hashes_valid_plans():
    episodes = [
        {"scene_id": "/data/s1/s1.glb", "episode_id": "2", "instruction": "go b"},
        {"scene_id": "/data/s1/s1.glb", "episode_id": "1", "instruction": "go a"},
    ]

    def segment(scene_id, episode_id, instruction):
        return {
            "stage_plan": {
                "schema_version": "instruction_stages_v1",
                "stages": [
                    {
                        "order": 0,
                        "route_clause": instruction,
                        "transition_type": "final_arrival",
                        "expected_landmarks": [],
                        "completion_cues": ["arrived"],
                        "final_stage": True,
                    }
                ],
            },
            "runtime_metadata": {"fallback_category": "none"},
        }

    first = generate_stage_manifest(episodes, ["s1:2", "s1:1"], segment)
    second = generate_stage_manifest(episodes, ["s1:2", "s1:1"], segment)

    assert first == second
    assert first["schema_version"] == "staged_stage_plan_manifest_v1"
    assert [row["episode_key"] for row in first["rows"]] == ["s1:1", "s1:2"]
    assert all(row["generation_status"] == "ok" for row in first["rows"])
    assert len(first["rows"][0]["instruction_sha256"]) == 64
    assert len(first["rows"][0]["stage_plan_sha256"]) == 64
    json.dumps(first)


def test_generate_stage_manifest_rejects_missing_key_and_fallback():
    episodes = [{"scene_id": "s1.glb", "episode_id": "1", "instruction": "go"}]

    try:
        generate_stage_manifest(episodes, ["s1:2"], lambda *_: {})
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected missing-key failure")

    def fallback(*_):
        return {
            "stage_plan": {"schema_version": "instruction_stages_v1", "stages": []},
            "runtime_metadata": {"fallback_category": "provider_failure"},
        }

    try:
        generate_stage_manifest(episodes, ["s1:1"], fallback)
    except ValueError as exc:
        assert "fallback" in str(exc)
    else:
        raise AssertionError("expected fallback failure")
