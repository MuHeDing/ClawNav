from pathlib import Path


def test_openclaw_tool_adapter_doc_exists():
    doc = Path("docs/protocols/openclaw-tool-adapter.md")
    text = doc.read_text(encoding="utf-8")

    assert "OpenClaw Tool Adapter" in text
    assert "list_tools" in text
    assert "call_tool" in text
    assert "No OpenClaw Runtime Dependency" in text
    assert "No Oracle Decision Inputs" in text
