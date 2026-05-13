from pathlib import Path


def test_scene_prior_script_uses_manifest_and_spatial_http():
    text = Path("scripts/evaluation_openclaw_scene_prior.sh").read_text(encoding="utf-8")

    assert "--harness_runtime openclaw_bridge" in text
    assert "--harness_memory_backend spatial_http" in text
    assert "--harness_memory_source scene-prior" in text
    assert "--memory_manifest_path" in text
    assert "--max_pixels 401408" in text
    assert "--use_llm_adaptive_sparse_attention" not in text


def test_train_scene_script_uses_train_scene_memory_source():
    text = Path("scripts/evaluation_openclaw_train_scene_memory.sh").read_text(
        encoding="utf-8"
    )

    assert "--harness_memory_source train-scene-only" in text
    assert "--memory_manifest_path" in text
    assert "--enable_slow_fast" not in text


def test_manifest_builder_mentions_no_leakage_fields():
    text = Path("scripts/build_scene_prior_memory_manifest.py").read_text(
        encoding="utf-8"
    )

    assert "uses_target_instruction" in text
    assert "uses_oracle_path" in text
    assert "uses_future_observations" in text
