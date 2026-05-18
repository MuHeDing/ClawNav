from pathlib import Path


def test_lowmem_sparse_scripts_disable_visual_prune_eval_profile():
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "scripts" / "evaluation_lowmem_sparse.sh",
        repo_root / "scripts" / "evaluation_lowmem_sparse2.sh",
    ]

    for script in scripts:
        contents = script.read_text(encoding="utf-8")

        assert "--disable_visual_prune_eval_profile" in contents


def test_lowmem_harness_script_uses_no_sparse_lowmem_config():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "evaluation_lowmem_harness.sh"
    contents = script.read_text(encoding="utf-8")

    assert "src/evaluation_harness.py" in contents
    assert "max_pixels=401408" in contents
    assert "kv_start_size=8" in contents
    assert "kv_recent_size=24" in contents
    assert "num_history=8" in contents
    assert "--use_llm_adaptive_sparse_attention" not in contents
    assert "--enable_slow_fast" not in contents


def test_openclaw_gateway_script_defaults_to_multi_episode_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "evaluation_openclaw_gateway.sh"
    contents = script.read_text(encoding="utf-8")

    assert "HARNESS_DEBUG_MAX_EPISODES=${HARNESS_DEBUG_MAX_EPISODES:-20}" in contents
    assert "--harness_debug_max_episodes" in contents


def test_openclaw_gateway_script_allows_runtime_fallback_after_preflight_failure():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "evaluation_openclaw_gateway.sh"
    contents = script.read_text(encoding="utf-8")

    assert "REQUIRE_GATEWAY=${REQUIRE_GATEWAY:-0}" in contents
    assert "continuing so runtime fallback can handle planner errors" in contents


def test_openclaw_cli_plan_gateway_start_script_uses_adapter_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "start_openclaw_cli_plan_gateway.sh"
    contents = script.read_text(encoding="utf-8")

    assert "harness.openclaw.openclaw_cli_plan_gateway" in contents
    assert "OPENCLAW_GATEWAY_WS_URL" in contents
    assert "OPENCLAW_VISUAL_MODE" in contents
    assert "--openclaw_visual_mode" in contents
    assert "--openclaw_visual_model" in contents


def test_openclaw_visual_memory_script_preflights_qwen_visual_gateway():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "evaluation_openclaw_visual_memory.sh"
    contents = script.read_text(encoding="utf-8")

    assert "OPENCLAW_VISUAL_MODE=${OPENCLAW_VISUAL_MODE:-describe}" in contents
    assert "OPENCLAW_VISUAL_MODEL=${OPENCLAW_VISUAL_MODEL:-qwen/qwen3.5-vl}" in contents
    assert "scripts/check_openclaw_visual_plan_gateway.py" in contents
    assert "REQUIRE_VISUAL_GATEWAY=${REQUIRE_VISUAL_GATEWAY:-1}" in contents
    assert "./scripts/evaluation_openclaw_gateway.sh" in contents


def test_openclaw_gateway_script_can_enable_visual_memory_curator_and_critic():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "evaluation_openclaw_gateway.sh"
    contents = script.read_text(encoding="utf-8")

    assert "OPENCLAW_ENABLE_SUBAGENT_CRITIC=${OPENCLAW_ENABLE_SUBAGENT_CRITIC:-0}" in contents
    assert (
        "OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR=${OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR:-0}"
        in contents
    )
    assert "--openclaw_enable_subagent_critic" in contents
    assert "--openclaw_enable_subagent_memory_curator" in contents
