from pathlib import Path


def test_phase4_research_system_doc_exists():
    text = Path("docs/protocols/openclaw-vln-research-system.md").read_text(encoding="utf-8")

    assert "External OpenClaw Gateway" in text
    assert "Robot Executor" in text
    assert "Scene-Prior Memory" in text
    assert "Train-Scene-Only Memory" in text
    assert "LLM/VLM Subagents" in text
    assert "Ablation Matrix" in text
    assert "No Oracle Decision Inputs" in text
