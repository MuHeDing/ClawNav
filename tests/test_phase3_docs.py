from pathlib import Path


def test_openclaw_vln_runtime_doc_exists():
    text = Path("docs/protocols/openclaw-vln-runtime.md").read_text(encoding="utf-8")

    assert "OpenClaw VLN Runtime" in text
    assert "openclaw_bridge" in text
    assert "No Required OpenClaw Python Dependency" in text
    assert "No Oracle Decision Inputs" in text
    assert "SpatialMemory" in text


def test_openclaw_runtime_script_exists_and_preserves_lowmem_settings():
    text = Path("scripts/evaluation_openclaw_vln_runtime.sh").read_text(encoding="utf-8")

    assert "src/evaluation_harness.py" in text
    assert "--harness_runtime openclaw_bridge" in text
    assert "--max_pixels 401408" in text
    assert "--kv_start_size 8" in text
    assert "--kv_recent_size 24" in text
    assert "--num_history 8" in text
    assert "--use_llm_adaptive_sparse_attention" not in text
    assert "--enable_slow_fast" not in text
