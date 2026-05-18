import json

from scripts.build_openclaw_visual_memory_report import build_report, format_report_markdown


def write_run(
    root,
    name,
    success=0.0,
    spl=0.0,
    trace_records=None,
):
    run = root / name
    traces = run / "harness_traces"
    traces.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"sucs_all": success, "spls_all": spl, "length": 2}),
        encoding="utf-8",
    )
    records = trace_records or [{"runtime_mode": "openclaw_bridge"}]
    (traces / "harness_trace_rank0.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return run


def test_build_report_passes_when_visual_memory_evidence_is_present(tmp_path):
    baseline = write_run(tmp_path, "baseline_lowmem", success=0.25, spl=0.1)
    visual = write_run(
        tmp_path,
        "openclaw_full_visual_memory_system",
        success=0.5,
        spl=0.25,
        trace_records=[
            {
                "runtime_mode": "openclaw_bridge",
                "visual_analysis": {"ran": True, "latency_ms": 12.0},
                "memory_writes": [
                    {
                        "written": True,
                        "memory_scope": "episode",
                        "memory_namespace": "episode:s1:e1",
                        "write_gate": {"curator_decision": "write"},
                    }
                ],
                "recall_usage": [
                    {
                        "num_hits": 1,
                        "used_by_policy": True,
                        "selected_namespace": "episode:s1:e1",
                    }
                ],
                "oracle_metrics_used_for_decision": False,
            }
        ],
    )

    report = build_report([baseline, visual], require_visual=True)

    assert report["status"] == "pass"
    assert report["runs"][1]["success_delta_vs_baseline"] == 0.25
    assert report["checks"]["no_oracle_leakage"]["passed"] is True
    assert report["checks"]["visual_analysis_present"]["passed"] is True
    assert report["checks"]["visual_memory_written"]["passed"] is True
    assert report["checks"]["useful_recall_present"]["passed"] is True


def test_build_report_fails_on_oracle_leakage_or_missing_visual_evidence(tmp_path):
    baseline = write_run(tmp_path, "baseline_lowmem", success=0.25, spl=0.1)
    visual = write_run(
        tmp_path,
        "openclaw_full_visual_memory_system",
        success=0.5,
        spl=0.25,
        trace_records=[
            {
                "runtime_mode": "openclaw_bridge",
                "oracle_metrics_used_for_decision": True,
            }
        ],
    )

    report = build_report([baseline, visual], require_visual=True)

    assert report["status"] == "fail"
    assert report["checks"]["no_oracle_leakage"]["passed"] is False
    assert report["checks"]["visual_analysis_present"]["passed"] is False
    assert report["checks"]["visual_memory_written"]["passed"] is False
    assert report["checks"]["useful_recall_present"]["passed"] is False


def test_format_report_markdown_includes_status_checks_and_metrics(tmp_path):
    baseline = write_run(tmp_path, "baseline_lowmem", success=0.25, spl=0.1)
    visual = write_run(
        tmp_path,
        "openclaw_full_visual_memory_system",
        success=0.5,
        spl=0.25,
        trace_records=[
            {
                "visual_analysis": {"ran": True, "latency_ms": 12.0},
                "memory_writes": [
                    {
                        "written": True,
                        "memory_scope": "episode",
                        "memory_namespace": "episode:s1:e1",
                        "write_gate": {"curator_decision": "write"},
                    }
                ],
                "recall_usage": [{"num_hits": 1, "used_by_policy": True}],
            }
        ],
    )

    markdown = format_report_markdown(build_report([baseline, visual], require_visual=True))

    assert "# OpenClaw Visual Memory Evidence Report" in markdown
    assert "**Status:** pass" in markdown
    assert "| visual_analysis_present | pass |" in markdown
    assert "success_delta_vs_baseline" in markdown
    assert "openclaw_full_visual_memory_system" in markdown
