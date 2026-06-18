import json

from scripts.summarize_openclaw_visual_readback import main as summarize_main
from harness.visual_readback.metrics import (
    format_visual_readback_markdown,
    summarize_visual_readback_run,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def row(case_id, mode, **visual_readback):
    block = {
        "case_id": case_id,
        "mode": mode,
        "trigger_source": "online_controller",
        "trigger_rule": "decision_point",
        "read_status": "completed",
        "replay_status": "ok",
        "retrieved_image_paths": [f"/tmp/{case_id}-memory.png"],
        "actually_read_image_paths": [f"/tmp/{case_id}-current.png", f"/tmp/{case_id}-memory.png"],
        "matched_memory_ids": [f"{case_id}-memory"],
        "required_verifier_labels": ["route_conflict"],
        "verifier_labels": ["route_conflict"],
        "visual_grounding_status": "grounded",
        "adjudication_status": "correct",
        "controller_decision_changed_after_visual_read": mode == "image_read_controller",
        "replan_request_logged_after_visual_read": mode == "image_read_controller",
    }
    block.update(visual_readback)
    return {
        "scene_id": "scene-1",
        "episode_id": f"episode-{case_id}",
        "step_id": 3,
        "visual_readback": block,
    }


def test_visual_readback_summary_counts_primary_offline_and_readback_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "sucs_all": 0.25,
                "spls_all": 0.2,
                "ndtw": 0.4,
                "visual_readback_config": {
                    "visual_readback_mode": "image_read_controller",
                    "visual_readback_control_only": True,
                },
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            row("A", "image_read_controller", current_only_sufficient=False),
            row("A", "text_only_prompt", visual_grounding_status="ungrounded", adjudication_status="wrong"),
            row("A", "path_only", visual_grounding_status="ungrounded"),
            row("A", "image_read_prompt", visual_grounding_status="ungrounded", adjudication_status="wrong"),
            row(
                "A",
                "current_only_controller",
                visual_grounding_status="ungrounded",
                adjudication_status="wrong",
                current_only_sufficient=False,
            ),
            row(
                "A",
                "shuffled_image_read_controller",
                adjudication_status="insufficient_evidence",
                visual_grounding_status="ungrounded",
                negative_control_label="wrong",
            ),
            row("B", "image_read_controller", current_only_sufficient=True),
            row("B", "current_only_controller", current_only_sufficient=True),
            row("C", "image_read_controller"),
            row(
                "C",
                "shuffled_image_read_controller",
                adjudication_status="unjudgeable",
                negative_control_label="negative_control_unjudgeable",
            ),
            row(
                "OFF",
                "image_read_controller",
                trigger_source="offline_fixture",
                visual_grounding_status="grounded",
                adjudication_status="wrong",
            ),
        ],
    )

    summary = summarize_visual_readback_run(run_dir)

    assert summary["trace_rows"] == 11
    assert summary["primary_quantitative_set_count"] == 3
    assert summary["offline_stress_set_count"] == 1
    assert summary["readback_trigger_count"] == 11
    assert summary["readback_completed_count"] == 11
    assert summary["retrieved_memory_image_count"] == 11
    assert summary["actually_read_image_count"] == 22
    assert summary["matched_memory_count"] == 11
    assert summary["visual_grounded_readback_count"] == 6
    assert summary["adjudication_correct_readback_count"] == 5
    assert summary["controller_decision_changed_after_visual_read_count"] == 4
    assert summary["replan_request_logged_after_visual_read_count"] == 4
    assert summary["grounded_but_wrong_label_count"] == 1
    assert summary["visual_readback_config"]["visual_readback_mode"] == "image_read_controller"

    primary_claim_metrics = summary["primary_phase1_claim_metrics"]
    assert "sucs_all" not in primary_claim_metrics
    assert summary["secondary_navigation_metrics"] == {
        "sucs_all": 0.25,
        "spls_all": 0.2,
        "ndtw": 0.4,
    }


def test_visual_readback_endpoint_report_uses_spec_denominators(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            row("A", "image_read_controller", current_only_sufficient=False),
            row("A", "text_only_prompt", adjudication_status="wrong"),
            row("A", "path_only", visual_grounding_status="ungrounded"),
            row("A", "image_read_prompt", adjudication_status="wrong"),
            row("A", "current_only_controller", adjudication_status="wrong", current_only_sufficient=False),
            row(
                "A",
                "shuffled_image_read_controller",
                adjudication_status="insufficient_evidence",
                visual_grounding_status="ungrounded",
                negative_control_label="wrong",
            ),
            row("B", "image_read_controller", current_only_sufficient=True),
            row("B", "current_only_controller", current_only_sufficient=True),
            row("C", "image_read_controller"),
            row(
                "C",
                "shuffled_image_read_controller",
                negative_control_label="negative_control_unjudgeable",
            ),
            row(
                "MISMATCH",
                "image_read_controller",
                replay_status="replay_mismatch",
            ),
            row(
                "MISMATCH",
                "text_only_prompt",
                replay_status="replay_mismatch",
            ),
        ],
    )

    endpoint_rows = {
        item["claim"]: item for item in summarize_visual_readback_run(run_dir)["endpoint_report"]
    }

    assert endpoint_rows["V4 > V1"]["endpoint"] == "adjudicated_controller_support_rate"
    assert endpoint_rows["V4 > V1"]["denominator"] == 1
    assert endpoint_rows["V4 > V1"]["excluded_counts"]["replay_mismatch"] == 1
    assert endpoint_rows["V4 > V1"]["effect_count"] == 1

    assert endpoint_rows["V4 > V2"]["endpoint"] == "visual_grounded_readback_rate"
    assert endpoint_rows["V4 > V2"]["denominator"] == 1
    assert endpoint_rows["V4 > V2"]["v4_support_count"] == 1
    assert endpoint_rows["V4 > V2"]["baseline_support_count"] == 0

    assert endpoint_rows["V4 > V4c"]["endpoint"] == "historical_memory_incremental_effect_rate"
    assert endpoint_rows["V4 > V4c"]["denominator"] == 1
    assert endpoint_rows["V4 > V4c"]["excluded_counts"]["current_only_sufficient"] == 1

    assert endpoint_rows["V4 > V5"]["endpoint"] == "historical_memory_dependent_effect_rate"
    assert endpoint_rows["V4 > V5"]["denominator"] == 1
    assert endpoint_rows["V4 > V5"]["excluded_counts"]["negative_control_unjudgeable"] == 1

    for row_data in endpoint_rows.values():
        assert row_data["minimum_effect_size"]
        assert row_data["paired_test_or_confidence_interval"]
        assert row_data["downgrade_to_descriptive_rule"]
        assert row_data["interpretation_scope"] == "mechanism_validation_only"


def test_visual_readback_markdown_keeps_phase1_claim_language_narrow(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [row("A", "image_read_controller")],
    )

    markdown = format_visual_readback_markdown(summarize_visual_readback_run(run_dir))

    assert "mechanism validation only" in markdown
    assert "SR/SPL improvement" not in markdown
    assert "| V4 > V1 | adjudicated_controller_support_rate |" in markdown


def test_visual_readback_summary_script_writes_json_or_markdown(tmp_path):
    run_dir = tmp_path / "run"
    output_path = tmp_path / "report.json"
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [row("A", "image_read_controller")],
    )

    code = summarize_main(
        [
            str(run_dir),
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    )

    assert code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["phase1_interpretation"] == "mechanism_validation_only"
