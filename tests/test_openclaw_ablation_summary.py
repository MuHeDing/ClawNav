import json

from scripts.summarize_openclaw_vln_ablation import summarize_run


def test_summarize_run_reads_navigation_and_harness_metrics(tmp_path):
    run = tmp_path / "run"
    traces = run / "harness_traces"
    traces.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"sucs_all": 0.5, "spls_all": 0.25, "length": 2}),
        encoding="utf-8",
    )
    (traces / "harness_trace_rank0.jsonl").write_text(
        json.dumps(
            {
                "runtime_mode": "openclaw_bridge",
                "planned_intent": "recall_memory",
                "tool_calls": [{"tool_name": "MemoryQuerySkill"}],
                "oracle_metrics_used_for_decision": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(run)

    assert summary["success"] == 0.5
    assert summary["spl"] == 0.25
    assert summary["trace_steps"] == 1
    assert summary["memory_recall_steps"] == 1
    assert summary["oracle_leakage_steps"] == 0
