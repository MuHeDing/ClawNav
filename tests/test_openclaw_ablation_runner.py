from scripts.run_openclaw_vln_ablation_matrix import build_matrix


def test_ablation_matrix_contains_required_variants():
    names = [item["name"] for item in build_matrix()]

    assert "baseline_lowmem" in names
    assert "phase2_memory_recall" in names
    assert "phase3_openclaw_bridge" in names
    assert "scene_prior_memory" in names
    assert "train_scene_memory" in names
    assert "subagent_planner" in names
    assert "openclaw_visual_observations" in names
    assert "openclaw_visual_memory_write" in names
    assert "openclaw_visual_memory_recall" in names
    assert "openclaw_full_visual_memory_system" in names


def test_ablation_matrix_preserves_lowmem_settings():
    for item in build_matrix():
        args = " ".join(item["args"])
        assert "--max_pixels 401408" in args
        assert "--kv_start_size 8" in args
        assert "--kv_recent_size 24" in args
        assert "--num_history 8" in args
        assert "--use_llm_adaptive_sparse_attention" not in args


def test_visual_ablation_variants_declare_qwen_visual_gateway_configuration():
    for item in build_matrix():
        if "visual" not in item["name"]:
            continue
        args = " ".join(item["args"])
        assert "OPENCLAW_VISUAL_MODE=describe" in args
        assert "OPENCLAW_VISUAL_MODEL=${OPENCLAW_VISUAL_MODEL:-qwen/qwen3.5-vl}" in args
        assert "OPENCLAW_VISUAL_TIMEOUT_MS=30000" in args
        assert "--openclaw_planner_backend gateway" in args


def test_ablation_matrix_declares_a0_to_a7_ladder():
    ladder = {item.get("ladder_id") for item in build_matrix()}

    assert {"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"}.issubset(ladder)


def test_full_visual_memory_system_enables_curator_and_critic():
    full = next(
        item for item in build_matrix() if item["name"] == "openclaw_full_visual_memory_system"
    )
    args = " ".join(full["args"])

    assert full["ladder_id"] == "A7"
    assert "OPENCLAW_ENABLE_SUBAGENT_CRITIC=1" in args
    assert "OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR=1" in args
