import json

from scripts.summarize_openclaw_visual_readback import main as summarize_main
from harness.visual_readback.manifest import (
    EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION,
    compute_event_gated_phase0_manifest_sha256,
)
from harness.visual_readback.metrics import (
    build_phase_a_gate_report,
    format_visual_readback_markdown,
    summarize_visual_readback_run,
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_phase0_manifest(path, rows):
    manifest_hash = compute_event_gated_phase0_manifest_sha256(rows)
    rows = [{**row, "manifest_sha256": manifest_hash} for row in rows]
    write_jsonl(path, rows)
    return manifest_hash


def phase0_manifest_row(**overrides):
    row_data = {
        "manifest_schema_version": EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION,
        "scene_id": "scene-1",
        "episode_id": "episode-A",
        "max_steps": 20,
        "route_event_labels": [{"event_type": "TURN_RIGHT", "step_id": 10}],
        "accepted_step_tolerance": 2,
        "shared_readback_trigger_keys": ["scene-1:episode-A:3:decision_point"],
        "trigger_key_source": "fixture",
        "phase0_stop_go_thresholds": {
            "route_event_miss_rate_max": 0.0,
            "exact_duplicate_rate_max": 0.0,
            "adjacent_triplet_rate_max": 1.0,
            "ambiguous_evidence_rate_max": 0.0,
            "comparison_direction": "lower_is_better",
            "invalid_comparison_behavior": "mark_invalid",
        },
        "selection_rules": "unit_test_seed",
        "selection_seed": 3,
        "source_dataset_id": "dataset-v1",
        "source_run_id": "run-1",
        "source_file_hashes": {"trace.jsonl": "abc123"},
    }
    row_data.update(overrides)
    return row_data


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


def gate_row(step_id, keyframe_gate, **visual_readback):
    block = {
        "mode": "image_read_controller",
        "trigger_source": "online_controller",
        "trigger_rule": "decision_point",
        "read_status": "skipped",
        "skip_reason": "no_trigger",
    }
    block.update(visual_readback)
    return {
        "scene_id": "scene-1",
        "episode_id": "episode-A",
        "step_id": step_id,
        "keyframe_gate": keyframe_gate,
        "visual_readback": block,
    }


def phase_a_summary(**overrides):
    summary = {
        "manifest_sha256": "manifest-1",
        "phase0_stop_go_thresholds": {
            "route_event_miss_rate_max": 0.0,
            "exact_duplicate_rate_max": 0.0,
            "adjacent_triplet_rate_max": 1.0,
            "ambiguous_evidence_rate_max": 0.0,
            "comparison_direction": "lower_is_better",
            "invalid_comparison_behavior": "mark_invalid",
        },
        "shared_trigger_key_count": 2,
        "missing_shared_trigger_key_count": 0,
        "extra_trigger_key_count": 0,
        "unmatched_trigger_count": 0,
        "route_event_miss_rate": 0.0,
        "exact_duplicate_rate": 0.0,
        "adjacent_triplet_rate": 0.0,
        "memory_evidence_used_count": 1,
        "candidate_pool_attached_total": 1,
        "keyframe_cap_validation_failed_count": 0,
        "keyframe_write_failed_count": 0,
    }
    summary.update(overrides)
    return summary


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
            row(
                "C",
                "image_read_controller",
                candidate_action_valid=False,
                should_override=True,
                invalid_reason="route_conflict",
                replan_executed_after_visual_read=True,
                readback_state_used_by_policy=True,
                action_hint_override_attempted_after_visual_read=True,
                action_hint_executed_after_visual_read=True,
                readback_state_used_by_controller=True,
                executed_action_changed_after_visual_read=True,
            ),
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
    assert summary["deduped_retrieved_memory_image_count"] == 11
    assert summary["exact_duplicate_drop_count"] == 0
    assert summary["adjacent_triplet_count"] == 0
    assert summary["actually_read_image_count"] == 22
    assert summary["matched_memory_count"] == 11
    assert summary["visual_grounded_readback_count"] == 6
    assert summary["adjudication_correct_readback_count"] == 5
    assert summary["controller_decision_changed_after_visual_read_count"] == 4
    assert summary["replan_request_logged_after_visual_read_count"] == 4
    assert summary["replan_executed_after_visual_read_count"] == 1
    assert summary["readback_state_used_by_policy_count"] == 1
    assert summary["action_hint_override_attempted_after_visual_read_count"] == 1
    assert summary["action_hint_executed_after_visual_read_count"] == 1
    assert summary["readback_state_used_by_controller_count"] == 1
    assert summary["executed_action_changed_after_visual_read_count"] == 1
    assert summary["candidate_action_valid_counts"] == {"false": 1, "missing": 10}
    assert summary["should_override_counts"] == {"missing": 10, "true": 1}
    assert summary["invalid_reason_counts"] == {"route_conflict": 1}
    assert summary["grounded_but_wrong_label_count"] == 1
    assert summary["memory_evidence_used_count"] == 0
    assert summary["current_only_evidence_count"] == 0
    assert summary["ambiguous_evidence_count"] == 0
    assert summary["invalid_evidence_source_count"] == 0
    assert summary["no_evidence_count"] == 0
    assert summary["visual_readback_config"]["visual_readback_mode"] == "image_read_controller"

    primary_claim_metrics = summary["primary_phase1_claim_metrics"]
    assert "sucs_all" not in primary_claim_metrics
    assert summary["secondary_navigation_metrics"] == {
        "sucs_all": 0.25,
        "spls_all": 0.2,
        "ndtw": 0.4,
    }


def test_visual_readback_summary_counts_action_override_shadow_reasons(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            row(
                "A",
                "image_read_action_override",
                action_hint_override_failure_reason="missing_current_action_evidence",
            ),
            row(
                "B",
                "image_read_action_override",
                action_hint_override_failure_reason="stop_hint_shadowed",
            ),
            row(
                "C",
                "image_read_action_override",
                action_hint_override_failure_reason="unstable_action_hint_shadowed",
            ),
            row(
                "D",
                "image_read_action_override",
                action_hint_override_failure_reason="unstable_action_hint_shadowed",
            ),
        ],
    )

    summary = summarize_visual_readback_run(run_dir)

    assert summary["action_hint_override_failure_reason_counts"] == {
        "missing_current_action_evidence": 1,
        "stop_hint_shadowed": 1,
        "unstable_action_hint_shadowed": 2,
    }
    assert summary["missing_current_action_evidence_count"] == 1
    assert summary["stop_hint_shadowed_count"] == 1
    assert summary["unstable_action_hint_shadowed_count"] == 2


def test_visual_readback_summary_reports_interval_exact_dedupe_metrics(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            row(
                "A",
                "image_read_controller",
                retrieved_image_paths=[
                    "/tmp/keyframe_000000.png",
                    "/tmp/keyframe_000010.png",
                    "/tmp/keyframe_000010.png",
                    "/tmp/keyframe_000020.png",
                ],
            )
        ],
    )

    summary = summarize_visual_readback_run(run_dir)

    assert summary["retrieved_memory_image_count"] == 4
    assert summary["deduped_retrieved_memory_image_count"] == 3
    assert summary["exact_duplicate_drop_count"] == 1
    assert summary["exact_duplicate_rate"] == 0.25
    assert summary["adjacent_triplet_count"] == 1
    assert summary["adjacent_triplet_rate"] == 1.0
    assert summary["interval_cleanup_attachment_metrics"]["interval_cleanup_mode"] == (
        "interval_10_exact_dedupe_summary"
    )


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


def test_visual_readback_summary_aggregates_evidence_source_counters(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            row(
                "A",
                "image_read_controller",
                memory_evidence_used_count=2,
                current_only_evidence_count=1,
                ambiguous_evidence_count=0,
                invalid_evidence_source_count=1,
                no_evidence_count=0,
            ),
            row(
                "B",
                "image_read_controller",
                memory_evidence_used_count=0,
                current_only_evidence_count=0,
                ambiguous_evidence_count=1,
                invalid_evidence_source_count=0,
                no_evidence_count=1,
            ),
        ],
    )

    summary = summarize_visual_readback_run(run_dir)

    assert summary["memory_evidence_used_count"] == 2
    assert summary["current_only_evidence_count"] == 1
    assert summary["ambiguous_evidence_count"] == 1
    assert summary["invalid_evidence_source_count"] == 1
    assert summary["no_evidence_count"] == 1


def test_visual_readback_summary_aggregates_event_gated_keyframe_gate_metrics(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            gate_row(
                0,
                {
                    "keyframe_policy_mode": "event_gated_smoke",
                    "save_decision": "save",
                    "save_reason": "initial_context",
                    "candidate_event_type": "initial_context",
                    "candidate_action_status": "ok",
                    "promotion_status": "promoted",
                    "write_status": "written",
                    "memory_id_link_status": "linked",
                    "controller_event_status": "not_applicable",
                },
            ),
            gate_row(
                3,
                {
                    "keyframe_policy_mode": "event_gated_smoke",
                    "save_decision": "save",
                    "save_reason": "candidate_decision_point",
                    "candidate_segment_id": 1,
                    "candidate_event_type": "TURN_RIGHT",
                    "confirmed_event_type": "TURN_RIGHT",
                    "candidate_action_status": "ok",
                    "promotion_status": "promoted",
                    "write_status": "memory_write_failed",
                    "memory_id_link_status": "not_linked",
                    "controller_event_status": "confirmed_executed",
                    "failure_reason": "backend down",
                },
            ),
            gate_row(
                4,
                {
                    "keyframe_policy_mode": "event_gated_smoke",
                    "save_decision": "skip",
                    "skip_reason": "cooldown",
                    "candidate_action_status": "ok",
                    "candidate_event_type": "STOP",
                    "cooldown_active": True,
                    "write_status": "not_attempted",
                    "memory_id_link_status": "not_attempted",
                    "controller_event_status": "candidate_only",
                    "confirmed_event_type": "MOVE_FORWARD",
                },
            ),
            gate_row(
                5,
                {
                    "keyframe_policy_mode": "event_gated_smoke",
                    "save_decision": "skip",
                    "skip_reason": "action_segment_duplicate",
                    "candidate_action_status": "ok",
                    "candidate_event_type": "TURN_RIGHT",
                    "candidate_segment_id": 1,
                    "write_status": "not_attempted",
                    "memory_id_link_status": "not_attempted",
                    "controller_event_status": "candidate_only",
                },
            ),
            gate_row(
                9,
                {
                    "keyframe_policy_mode": "event_gated_smoke",
                    "save_decision": "skip",
                    "skip_reason": "episode_cap_reached",
                    "candidate_action_status": "ok",
                    "candidate_event_type": "coverage_gap",
                    "write_status": "not_attempted",
                    "memory_id_link_status": "not_attempted",
                    "controller_event_status": "not_applicable",
                    "keyframe_cap_validation_failed": True,
                },
            ),
            gate_row(
                10,
                {
                    "keyframe_policy_mode": "event_gated_smoke",
                    "save_decision": "skip",
                    "skip_reason": "candidate_action_failed",
                    "candidate_action_status": "failed",
                    "write_status": "failed",
                    "memory_id_link_status": "failed",
                    "controller_event_status": "unconfirmed",
                    "failure_reason": "missing required event_gated_keyframe fields: retrieval_text_or_action_context",
                },
            ),
        ],
    )

    summary = summarize_visual_readback_run(run_dir)

    assert summary["keyframe_policy_mode_counts"] == {"event_gated_smoke": 6}
    assert summary["keyframe_gate_trace_count"] == 6
    assert summary["keyframe_saved_count"] == 2
    assert summary["keyframe_skipped_count"] == 4
    assert summary["keyframe_saved_by_reason"] == {
        "candidate_decision_point": 1,
        "initial_context": 1,
    }
    assert summary["keyframe_skipped_by_reason"]["cooldown"] == 1
    assert summary["keyframe_skipped_by_reason"]["action_segment_duplicate"] == 1
    assert summary["initial_context_keyframe_count"] == 1
    assert summary["direct_save_segment_count"] == 1
    assert summary["cooldown_skip_count"] == 1
    assert summary["action_segment_duplicate_skip_count"] == 1
    assert summary["episode_cap_reached_skip_count"] == 1
    assert summary["keyframe_cap_validation_failed_count"] == 1
    assert summary["keyframe_write_failed_count"] == 2
    assert summary["retrieval_text_failure_count"] == 1
    assert summary["memory_id_link_status_counts"]["linked"] == 1
    assert summary["memory_id_link_status_counts"]["not_linked"] == 1
    assert summary["candidate_event_type_counts"]["TURN_RIGHT"] == 2
    assert summary["confirmed_event_type_counts"]["TURN_RIGHT"] == 1
    assert summary["controller_event_status_counts"]["candidate_only"] == 2
    assert summary["candidate_action_status_counts"]["failed"] == 1


def test_visual_readback_summary_aggregates_smoke_candidate_pool_metrics(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            row(
                "A",
                "image_read_controller",
                candidate_pool_source="local_event_gated_keyframe_ledger",
                candidate_pool_requested_count=3,
                eligible_prior_keyframe_count=4,
                candidate_pool_backend_returned_count=4,
                candidate_pool_exact_duplicate_drop_count=1,
                candidate_pool_attached_count=3,
            ),
            row(
                "B",
                "image_read_controller",
                read_status="skipped",
                skip_reason="no_eligible_prior_keyframe",
                candidate_pool_source="local_event_gated_keyframe_ledger",
                candidate_pool_requested_count=3,
                eligible_prior_keyframe_count=0,
                candidate_pool_backend_returned_count=0,
                candidate_pool_exact_duplicate_drop_count=0,
                candidate_pool_attached_count=0,
            ),
            row("C", "image_read_controller", read_status="skipped", skip_reason="memory_query_failed"),
            row("D", "image_read_controller", read_status="skipped", skip_reason="no_memory_hit"),
        ],
    )

    summary = summarize_visual_readback_run(run_dir)

    assert summary["candidate_pool_attempt_count"] == 2
    assert summary["candidate_pool_requested_total"] == 6
    assert summary["eligible_prior_keyframe_total"] == 4
    assert summary["candidate_pool_backend_returned_total"] == 4
    assert summary["candidate_pool_exact_duplicate_drop_total"] == 1
    assert summary["candidate_pool_attached_total"] == 3
    assert summary["candidate_pool_source_counts"] == {"local_event_gated_keyframe_ledger": 2}
    assert summary["no_eligible_prior_keyframe_count"] == 1
    assert summary["memory_query_failed_count"] == 1
    assert summary["no_memory_hit_count"] == 1


def test_phase_a_gate_report_invalidates_manifest_mismatch():
    summaries = {
        "interval_10": phase_a_summary(manifest_sha256="manifest-1"),
        "interval_10_exact_dedupe": phase_a_summary(manifest_sha256="manifest-1"),
        "interval_10_smoke_audit_control": phase_a_summary(manifest_sha256="manifest-2"),
        "event_gated_smoke": phase_a_summary(
            keyframe_policy_mode_counts={"event_gated_smoke": 2},
            keyframe_gate_trace_count=2,
        ),
    }

    report = build_phase_a_gate_report(summaries)

    assert report["phase_a_comparison_valid"] is False
    assert "manifest_sha256_mismatch" in report["phase_a_invalid_reasons"]
    assert report["phase_a_claim_scope"] == "invalid_comparison"
    assert report["phase_a_allowed_interpretation"] == "no_event_gated_claim"


def test_phase_a_gate_report_keeps_traceability_only_when_control_explains_gain():
    summaries = {
        "interval_10": phase_a_summary(exact_duplicate_rate=0.5, memory_evidence_used_count=0),
        "interval_10_exact_dedupe": phase_a_summary(exact_duplicate_rate=0.0, memory_evidence_used_count=0),
        "interval_10_smoke_audit_control": phase_a_summary(
            exact_duplicate_rate=0.0,
            memory_evidence_used_count=2,
            candidate_pool_attached_total=2,
        ),
        "event_gated_smoke": phase_a_summary(
            keyframe_policy_mode_counts={"event_gated_smoke": 2},
            keyframe_gate_trace_count=2,
            exact_duplicate_rate=0.0,
            memory_evidence_used_count=2,
            candidate_pool_attached_total=2,
        ),
    }

    report = build_phase_a_gate_report(summaries)

    assert report["phase_a_comparison_valid"] is True
    assert report["phase_a_claim_scope"] == "traceability_only"
    assert report["phase_a_allowed_interpretation"] == "traceability_and_evidence_quality_only"
    assert report["retrieval_audit_control_explains_gain"] is True
    assert "exact_duplicate_rate" in report["gate_attributable_deltas"]
    assert "exact_duplicate_rate" in report["retrieval_audit_control_deltas"]


def test_summarize_script_builds_phase_a_report_from_summary_jsons(tmp_path, capsys):
    paths = {}
    for name, summary in {
        "interval_10": phase_a_summary(exact_duplicate_rate=0.5),
        "interval_10_exact_dedupe": phase_a_summary(exact_duplicate_rate=0.0),
        "interval_10_smoke_audit_control": phase_a_summary(exact_duplicate_rate=0.0),
        "event_gated_smoke": phase_a_summary(
            keyframe_policy_mode_counts={"event_gated_smoke": 1},
            keyframe_gate_trace_count=1,
        ),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths[name] = path

    rc = summarize_main(
        [
            "--phase_a_summary",
            f"interval_10={paths['interval_10']}",
            "--phase_a_summary",
            f"interval_10_exact_dedupe={paths['interval_10_exact_dedupe']}",
            "--phase_a_summary",
            f"interval_10_smoke_audit_control={paths['interval_10_smoke_audit_control']}",
            "--phase_a_summary",
            f"event_gated_smoke={paths['event_gated_smoke']}",
            "--format",
            "json",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert output["phase_a_comparison_valid"] is True
    assert output["phase_a_required_arms_present"]["event_gated_smoke"] is True


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


def test_visual_readback_summary_script_records_phase0_manifest_contract(tmp_path):
    run_dir = tmp_path / "run"
    output_path = tmp_path / "report.json"
    manifest_path = tmp_path / "phase0_manifest.jsonl"
    manifest_hash = write_phase0_manifest(
        manifest_path,
        [phase0_manifest_row()],
    )
    write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [row("A", "image_read_controller", trigger_key="scene-1:episode-A:3:decision_point")],
    )

    code = summarize_main(
        [
            str(run_dir),
            "--format",
            "json",
            "--event_gated_phase0_manifest",
            str(manifest_path),
            "--expected_manifest_sha256",
            manifest_hash,
            "--output",
            str(output_path),
        ]
    )

    assert code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["manifest_sha256"] == manifest_hash
    assert payload["phase0_comparison_valid"] is True
    assert payload["phase0_stop_go_decision"] == "no_go_interval_cleanup"
