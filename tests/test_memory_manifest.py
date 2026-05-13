import json

from harness.memory.manifest import (
    MemoryManifest,
    MemoryManifestError,
    load_memory_manifest,
)


def test_episode_local_manifest_is_valid_without_files(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"memory_source": "episode-local", "entries": []}),
        encoding="utf-8",
    )

    manifest = load_memory_manifest(path)

    assert isinstance(manifest, MemoryManifest)
    assert manifest.memory_source == "episode-local"
    assert manifest.entries == []


def test_scene_prior_requires_exploration_split_and_no_target_instruction(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "memory_source": "scene-prior",
                "entries": [
                    {
                        "scene_id": "s1",
                        "split": "exploration",
                        "uses_target_instruction": False,
                        "uses_oracle_path": False,
                        "uses_future_observations": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_memory_manifest(path)

    assert manifest.memory_source == "scene-prior"


def test_manifest_rejects_oracle_or_future_leakage(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "memory_source": "scene-prior",
                "entries": [
                    {
                        "scene_id": "s1",
                        "split": "val_unseen",
                        "uses_target_instruction": True,
                        "uses_oracle_path": True,
                        "uses_future_observations": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_memory_manifest(path)
    except MemoryManifestError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("expected MemoryManifestError")
