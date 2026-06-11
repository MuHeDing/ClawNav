import json

from scripts.summarize_openclaw_vln_ablation import format_markdown, summarize_run


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
                "visual_analysis": {"ran": True, "vlm_latency_ms": 12.0},
                "memory_writes": [
                    {
                        "written": True,
                        "skipped": False,
                        "write_gate": {
                            "curator_decision": "write",
                            "novelty_score": 0.71,
                        },
                        "memory_scope": "episode",
                        "memory_namespace": "episode:s1:e1",
                    }
                ],
                "recall_usage": [
                    {
                        "used_by_planner": True,
                        "action_changed_after_recall": True,
                        "planner_intent_before_recall": "recall_memory",
                        "planner_intent_after_recall": "replan",
                        "selected_namespace": "episode:s1:e1",
                        "num_hits": 1,
                    }
                ],
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
    assert summary["visual_analysis_steps"] == 1
    assert summary["memory_write_steps"] == 1
    assert summary["action_changed_after_recall_steps"] == 1
    assert summary["episode_namespace_hits"] == 1
    assert summary["memory_write_attempts"] == 1
    assert summary["memory_write_written"] == 1
    assert summary["write_acceptance_rate"] == 1.0
    assert summary["useful_recall_rate"] == 1.0
    assert summary["visual_analysis_latency_ms_avg"] == 12.0
    assert summary["avg_vlm_latency_ms"] == 12.0
    assert summary["avg_write_novelty_score"] == 0.71
    assert summary["planner_intent_changed_after_recall_events"] == 1
    assert summary["memory_scope_counts"] == {"episode": 1}
    assert summary["memory_namespace_counts"] == {"episode:s1:e1": 1}


def test_summarize_run_reports_write_gate_skips_and_namespace_distribution(tmp_path):
    run = tmp_path / "run"
    traces = run / "harness_traces"
    traces.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"sucs_all": 0.25, "spls_all": 0.1, "length": 3}),
        encoding="utf-8",
    )
    (traces / "harness_trace_rank0.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "memory_writes": [
                            {
                                "written": False,
                                "skipped": True,
                                "skip_reason": "duplicate visual keyframe",
                                "memory_scope": "scene",
                                "memory_namespace": "scene:s1:train-scene-only",
                                "write_gate": {
                                    "curator_decision": "skip",
                                    "duplicate_of_memory_id": "m1",
                                },
                            }
                        ],
                        "recall_usage": [
                            {
                                "selected_namespace": "scene:s1:train-scene-only",
                                "num_hits": 2,
                                "used_by_policy": False,
                                "used_by_critic": False,
                                "action_changed_after_recall": False,
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "visual_analysis": {
                            "ran": True,
                            "latency_ms": 18.0,
                            "error": "vlm timeout",
                        }
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(run)

    assert summary["memory_write_attempts"] == 1
    assert summary["memory_write_written"] == 0
    assert summary["memory_write_skipped"] == 1
    assert summary["duplicate_skip_count"] == 1
    assert summary["write_acceptance_rate"] == 0.0
    assert summary["memory_pollution_rate"] == 1.0
    assert summary["write_gate_decisions"] == {"skip": 1}
    assert summary["scene_namespace_hits"] == 1
    assert summary["recall_events_with_hits"] == 1
    assert summary["useful_recall_rate"] == 0.0
    assert summary["visual_analysis_failures"] == 1


def test_summarize_run_flags_all_one_step_results_as_invalid(tmp_path):
    run = tmp_path / "run"
    traces = run / "harness_traces"
    traces.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"sucs_all": 0.0, "spls_all": 0.0, "length": 2}),
        encoding="utf-8",
    )
    (run / "result.json").write_text(
        "\n".join(
            [
                json.dumps({"episode_id": "1", "success": 0.0, "steps": 1}),
                json.dumps({"episode_id": "2", "success": 0.0, "steps": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (traces / "harness_trace_rank0.jsonl").write_text("", encoding="utf-8")

    summary = summarize_run(run)

    assert summary["all_episodes_one_step"] is True
    assert summary["invalid_run_reason"] == "all_episodes_ended_after_one_step"


def test_ablation_summary_tolerates_missing_memory_gate_fields(tmp_path):
    run = tmp_path / "run"
    traces = run / "harness_traces"
    traces.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"sucs_all": 0.5, "spls_all": 0.2, "length": 1}),
        encoding="utf-8",
    )
    (traces / "harness_trace_rank0.jsonl").write_text(
        json.dumps({"planned_intent": "act"}) + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(run)

    assert summary["memory_gate_policy_context_used_count"] == 0
    assert summary["memory_gate_stop_semantics_filtered_count"] == 0
    assert summary["stop_verifier_trigger_count"] == 0


def test_ablation_summary_surfaces_memory_gate_and_stop_metrics(tmp_path):
    run = tmp_path / "run"
    traces = run / "harness_traces"
    traces.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"sucs_all": 0.5, "spls_all": 0.2, "length": 1}),
        encoding="utf-8",
    )
    (traces / "harness_trace_rank0.jsonl").write_text(
        json.dumps(
            {
                "memory_gate": {
                    "policy_context_used": True,
                    "stop_semantics_filtered": True,
                    "raw_reason_included": False,
                },
                "stop_verifier": {
                    "triggered": True,
                    "blocked": True,
                    "disagreement": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(run)

    assert summary["memory_gate_policy_context_used_count"] == 1
    assert summary["memory_gate_stop_semantics_filtered_count"] == 1
    assert summary["memory_gate_raw_reason_included_count"] == 0
    assert summary["stop_verifier_trigger_count"] == 1
    assert summary["stop_verifier_block_count"] == 1
    assert summary["stop_verifier_disagreement_count"] == 1


def test_format_markdown_reports_core_ablation_metrics():
    markdown = format_markdown(
        [
            {
                "run": "openclaw_full_visual_memory_system",
                "success": 0.5,
                "spl": 0.25,
                "visual_analysis_steps": 8,
                "memory_write_attempts": 4,
                "write_acceptance_rate": 0.75,
                "memory_recall_events": 3,
                "useful_recall_rate": 0.666666,
                "action_changed_after_recall_events": 1,
                "oracle_leakage_steps": 0,
                "memory_gate_policy_context_used_count": 2,
                "stop_verifier_trigger_count": 1,
            }
        ]
    )

    assert "| run | success | spl |" in markdown
    assert "openclaw_full_visual_memory_system" in markdown
    assert "0.7500" in markdown
    assert "0.6667" in markdown
    assert "memory_gate_policy_context_used_count" in markdown
    assert "stop_verifier_trigger_count" in markdown
