from pathlib import Path


RUNBOOK = Path("docs/runbooks/openclaw-visual-readback.md")


def test_visual_readback_runbook_documents_reproducible_gates_and_claim_limits():
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "PYTHONPATH=.:src pytest" in text
    assert "scripts/run_openclaw_visual_readback_matrix.py --phase phase0 --dry_run" in text
    assert "scripts/smoke_visual_readback_skill.py" in text
    assert "scripts/summarize_openclaw_visual_readback.py" in text
    assert "scripts/run_visual_readback_runtime_smoke.sh" in text
    assert "scripts/check_visual_readback_runtime_smoke.py" in text
    assert "QwenDirectVisualReadbackAdapter + QwenApiModelClient" in text
    assert "openclaw capability image describe-many" in text
    assert "diagnostic_status=natural_image_hit" in text
    assert "natural_image_hit_ready=true" in text

    for gate_status in (
        "no_go",
        "downscope_stop_only",
        "downscope_turn_only",
        "candidate_set_ready",
        "fixed_manifest_ready",
    ):
        assert gate_status in text

    for smoke_metric in (
        "parse_success_rate",
        "image_attach_rate",
        "latency_ms",
        "timeout_rate",
    ):
        assert smoke_metric in text

    assert "Phase 1 is not an SR/SPL improvement claim" in text
