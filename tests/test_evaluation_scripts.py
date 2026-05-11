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
