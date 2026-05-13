from scripts.run_openclaw_vln_ablation_matrix import build_matrix


def test_ablation_matrix_contains_required_variants():
    names = [item["name"] for item in build_matrix()]

    assert "baseline_lowmem" in names
    assert "phase2_memory_recall" in names
    assert "phase3_openclaw_bridge" in names
    assert "scene_prior_memory" in names
    assert "train_scene_memory" in names
    assert "subagent_planner" in names


def test_ablation_matrix_preserves_lowmem_settings():
    for item in build_matrix():
        args = " ".join(item["args"])
        assert "--max_pixels 401408" in args
        assert "--kv_start_size 8" in args
        assert "--kv_recent_size 24" in args
        assert "--num_history 8" in args
        assert "--use_llm_adaptive_sparse_attention" not in args
