import json

from scripts.smoke_visual_readback_skill import (
    run_runtime_readback_smoke,
    run_visual_readback_skill_smoke,
)


class FakeReadbackAdapter:
    def __init__(self):
        self.calls = []

    def read(self, request):
        self.calls.append(dict(request))
        return {
            "verifier_labels": ["route_conflict"],
            "matched_memory_ids": list(request.get("attached_memory_ids") or []),
            "visual_evidence": "The current and memory images are both attached.",
            "readback_confidence": 0.91,
            "verifier_confidence": 0.9,
            "visual_grounding_status": "grounded",
        }


def test_skill_smoke_writes_fixed_summary_and_payload(tmp_path):
    adapter = FakeReadbackAdapter()

    summary = run_visual_readback_skill_smoke(
        output_root=tmp_path,
        adapter=adapter,
        model="qwen/qwen3.5-flash",
    )

    payload = json.loads((tmp_path / "skill_payload.json").read_text(encoding="utf-8"))
    persisted_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert persisted_summary == summary
    assert summary["model"] == "qwen/qwen3.5-flash"
    assert summary["read_status"] == "completed"
    assert summary["parse_success_rate"] == 1.0
    assert summary["image_attach_rate"] == 1.0
    assert summary["timeout_rate"] == 0.0
    assert payload["actually_read_image_paths"] == [
        str(tmp_path / "images" / "current.png"),
        str(tmp_path / "images" / "memory.png"),
    ]
    assert payload["matched_memory_ids"] == ["memory-flash-default-1"]


def test_runtime_smoke_records_readback_trace_with_memory_hit(tmp_path):
    adapter = FakeReadbackAdapter()

    summary = run_runtime_readback_smoke(
        output_root=tmp_path,
        adapter=adapter,
        model="qwen/qwen3.5-flash",
    )

    metadata = json.loads((tmp_path / "runtime_metadata.json").read_text(encoding="utf-8"))
    runtime_summary = json.loads((tmp_path / "runtime_summary.json").read_text(encoding="utf-8"))
    assert runtime_summary == summary
    assert summary["runtime_ok"] is True
    assert summary["read_status"] == "completed"
    assert summary["matched_memory_ids"] == ["memory-flash-default-1"]
    assert summary["has_current_and_memory_images"] is True
    assert summary["memory_query_tool_call_count"] >= 1
    assert summary["visual_read_tool_call_count"] == 1
    assert metadata["visual_readback"]["actually_read_image_paths"] == [
        str(tmp_path / "images" / "current.png"),
        str(tmp_path / "images" / "memory.png"),
    ]
    assert metadata["visual_readback"]["matched_memory_ids"] == [
        "memory-flash-default-1"
    ]
