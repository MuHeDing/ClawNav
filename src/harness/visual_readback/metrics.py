import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from harness.visual_readback.readback import normalize_visual_grounding_status


PRIMARY_TRIGGER_SOURCES = {"online_controller", "static_pre_run"}
OFFLINE_TRIGGER_SOURCES = {"offline_fixture"}
V4_MODE = "image_read_controller"
ENDPOINT_DEFINITIONS = [
    {
        "claim": "V4 > V1",
        "baseline_mode": "text_only_prompt",
        "endpoint": "adjudicated_controller_support_rate",
        "denominator_description": "fixed_replay_manifest judgeable cases",
        "pass_rule": (
            "V4 produces grounded, adjudication-correct controller evidence more "
            "often than text-only prompt produces the same adjudicated support"
        ),
        "failure_handling": (
            "replay_state_mismatch / replay_mismatch / unjudgeable labels excluded and counted"
        ),
        "allowed_interpretation": "controller evidence has value beyond text-only summary",
    },
    {
        "claim": "V4 > V2",
        "baseline_mode": "path_only",
        "endpoint": "visual_grounded_readback_rate",
        "denominator_description": "fixed_replay_manifest judgeable cases with image_path",
        "pass_rule": "V4 grounded rate > path-only grounded rate",
        "failure_handling": "missing image_path cases excluded and counted",
        "allowed_interpretation": "gain is not from image path strings",
    },
    {
        "claim": "V4 > V3",
        "baseline_mode": "image_read_prompt",
        "endpoint": "paired_adjudicated_support_rate",
        "denominator_description": "fixed_replay_manifest judgeable cases",
        "pass_rule": (
            "V4 and V3 are compared on the same adjudicated support/conflict target"
        ),
        "failure_handling": (
            "replay_state_mismatch, prompt-contaminated, or missing shared labels reported separately"
        ),
        "allowed_interpretation": "controller-side arbitration preserves attribution",
    },
    {
        "claim": "V4 > V4c",
        "baseline_mode": "current_only_controller",
        "endpoint": "historical_memory_incremental_effect_rate",
        "denominator_description": "fixed_replay_manifest cases where current_only is insufficient",
        "pass_rule": (
            "V4 produces adjudication-correct memory-dependent evidence that "
            "V4c_current_only_controller cannot produce"
        ),
        "failure_handling": "current_only-sufficient cases excluded and counted separately",
        "allowed_interpretation": "effect depends on historical memory rather than current-view verification",
    },
    {
        "claim": "V4 > V5",
        "baseline_mode": "shuffled_image_read_controller",
        "endpoint": "historical_memory_dependent_effect_rate",
        "denominator_description": "fixed_replay_manifest cases with pre-labeled wrong V5 replacements",
        "pass_rule": (
            "V4 effect survives and V5 wrong-image effect disappears or becomes insufficient_evidence"
        ),
        "failure_handling": "negative_control_unjudgeable excluded and counted",
        "allowed_interpretation": "effect depends on the correct retrieved memory image",
    },
]
EXCLUDED_COUNT_KEYS = (
    "replay_state_mismatch",
    "replay_mismatch",
    "unjudgeable",
    "missing_image_path",
    "prompt_contaminated",
    "missing_shared_labels",
    "current_only_sufficient",
    "negative_control_unjudgeable",
)
NAVIGATION_METRIC_KEYS = (
    "sucs_all",
    "spls_all",
    "ne",
    "nav_error",
    "ndtw",
    "sdtw",
    "length",
)
PHASE_A_REQUIRED_ARMS = (
    "interval_10",
    "interval_10_exact_dedupe",
    "interval_10_smoke_audit_control",
    "event_gated_smoke",
)
KEYFRAME_WRITE_FAILURE_STATUSES = {"failed", "memory_write_failed", "write_failed"}
CANDIDATE_POOL_SUM_KEYS = (
    "candidate_pool_requested_count",
    "eligible_prior_keyframe_count",
    "candidate_pool_backend_returned_count",
    "candidate_pool_exact_duplicate_drop_count",
    "candidate_pool_attached_count",
)
PHASE_A_DELTA_METRICS = (
    "route_event_miss_rate",
    "exact_duplicate_rate",
    "adjacent_triplet_rate",
    "memory_evidence_used_count",
    "current_only_evidence_count",
    "ambiguous_evidence_count",
    "invalid_evidence_source_count",
    "eligible_prior_keyframe_total",
    "candidate_pool_backend_returned_total",
    "candidate_pool_exact_duplicate_drop_total",
    "candidate_pool_attached_total",
)


def summarize_visual_readback_run(
    run_dir: Path,
    phase0_manifest_path: Optional[Path] = None,
    expected_manifest_sha256: str = "",
    expected_phase0_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    trace_paths = _trace_paths(run_path)
    loaded_rows = _load_trace_rows(trace_paths)
    blocks = [_visual_readback_block(row) for row in loaded_rows]
    summary_json = _load_summary_json(run_path)
    endpoint_report = build_endpoint_report(blocks)
    interval_cleanup = _interval_cleanup_attachment_metrics(blocks)
    keyframe_gate = _keyframe_gate_summary(blocks)
    candidate_pool = _candidate_pool_summary(blocks)
    action_hint_failure_counts = Counter(
        str(block.get("action_hint_override_failure_reason") or "")
        for block in blocks
        if str(block.get("action_hint_override_failure_reason") or "")
    )
    summary = {
        "run_dir": str(run_path),
        "trace_rows": len(loaded_rows),
        "primary_quantitative_set_count": len(_unique_cases(_primary_blocks(blocks))),
        "offline_stress_set_count": len(_unique_cases(_offline_blocks(blocks))),
        "trigger_type_distribution": dict(
            sorted(Counter(str(block.get("trigger_rule") or "") for block in blocks if block.get("trigger_rule")).items())
        ),
        "readback_mode_counts": dict(
            sorted(Counter(str(block.get("mode") or "") for block in blocks if block.get("mode")).items())
        ),
        "readback_trigger_count": sum(1 for block in blocks if _has_trigger(block)),
        "readback_completed_count": sum(1 for block in blocks if block.get("read_status") == "completed"),
        "retrieved_memory_image_count": sum(len(_list_value(block.get("retrieved_image_paths"))) for block in blocks),
        "deduped_retrieved_memory_image_count": interval_cleanup["deduped_retrieved_memory_image_count"],
        "exact_duplicate_drop_count": interval_cleanup["exact_duplicate_drop_count"],
        "exact_duplicate_rate": interval_cleanup["exact_duplicate_rate"],
        "adjacent_triplet_count": interval_cleanup["adjacent_triplet_count"],
        "adjacent_triplet_denominator": interval_cleanup["adjacent_triplet_denominator"],
        "adjacent_triplet_rate": interval_cleanup["adjacent_triplet_rate"],
        "interval_cleanup_attachment_metrics": interval_cleanup,
        **keyframe_gate,
        **candidate_pool,
        "actually_read_image_count": sum(len(_list_value(block.get("actually_read_image_paths"))) for block in blocks),
        "matched_memory_count": sum(len(_stable_memory_ids(block)) for block in blocks),
        "visual_grounded_readback_count": sum(1 for block in blocks if _is_grounded(block)),
        "adjudication_correct_readback_count": sum(1 for block in blocks if _is_adjudication_correct(block)),
        "controller_decision_changed_after_visual_read_count": sum(
            1 for block in blocks if bool(block.get("controller_decision_changed_after_visual_read"))
        ),
        "replan_request_logged_after_visual_read_count": sum(
            1 for block in blocks if bool(block.get("replan_request_logged_after_visual_read"))
        ),
        "replan_executed_after_visual_read_count": sum(
            1 for block in blocks if bool(block.get("replan_executed_after_visual_read"))
        ),
        "readback_state_used_by_policy_count": sum(
            1 for block in blocks if bool(block.get("readback_state_used_by_policy"))
        ),
        "executed_action_changed_after_visual_read_count": sum(
            1 for block in blocks if bool(block.get("executed_action_changed_after_visual_read"))
        ),
        "action_hint_override_attempted_after_visual_read_count": sum(
            1
            for block in blocks
            if bool(block.get("action_hint_override_attempted_after_visual_read"))
        ),
        "action_hint_executed_after_visual_read_count": sum(
            1 for block in blocks if bool(block.get("action_hint_executed_after_visual_read"))
        ),
        "action_hint_override_failure_reason_counts": dict(
            sorted(action_hint_failure_counts.items())
        ),
        "missing_current_action_evidence_count": action_hint_failure_counts.get(
            "missing_current_action_evidence",
            0,
        ),
        "stop_hint_shadowed_count": action_hint_failure_counts.get(
            "stop_hint_shadowed",
            0,
        ),
        "unstable_action_hint_shadowed_count": action_hint_failure_counts.get(
            "unstable_action_hint_shadowed",
            0,
        ),
        "candidate_action_valid_counts": dict(
            sorted(Counter(_bool_status(block, "candidate_action_valid") for block in blocks).items())
        ),
        "should_override_counts": dict(
            sorted(Counter(_bool_status(block, "should_override") for block in blocks).items())
        ),
        "invalid_reason_counts": dict(
            sorted(
                Counter(
                    str(block.get("invalid_reason") or "")
                    for block in blocks
                    if str(block.get("invalid_reason") or "")
                ).items()
            )
        ),
        "readback_state_used_by_controller_count": sum(
            1 for block in blocks if bool(block.get("readback_state_used_by_controller"))
        ),
        "grounded_but_wrong_label_count": sum(1 for block in blocks if _is_grounded_but_wrong(block)),
        "memory_evidence_used_count": sum(_int_value(block.get("memory_evidence_used_count")) for block in blocks),
        "current_only_evidence_count": sum(_int_value(block.get("current_only_evidence_count")) for block in blocks),
        "ambiguous_evidence_count": sum(_int_value(block.get("ambiguous_evidence_count")) for block in blocks),
        "invalid_evidence_source_count": sum(_int_value(block.get("invalid_evidence_source_count")) for block in blocks),
        "no_evidence_count": sum(_int_value(block.get("no_evidence_count")) for block in blocks),
        "verifier_label_confusion": _verifier_label_confusion(blocks),
        "endpoint_report": endpoint_report,
        "primary_phase1_claim_metrics": {
            row["endpoint"]: {
                "claim": row["claim"],
                "denominator": row["denominator"],
                "v4_rate": row["v4_rate"],
                "baseline_rate": row["baseline_rate"],
                "effect_count": row["effect_count"],
            }
            for row in endpoint_report
        },
        "secondary_navigation_metrics": _secondary_navigation_metrics(summary_json),
        "visual_readback_config": _visual_readback_config(summary_json, blocks),
        "phase1_interpretation": "mechanism_validation_only",
    }
    if phase0_manifest_path:
        summary.update(
            _phase0_summary_fields(
                trace_paths=trace_paths,
                phase0_manifest_path=Path(phase0_manifest_path),
                expected_manifest_sha256=expected_manifest_sha256,
                expected_phase0_thresholds=expected_phase0_thresholds,
            )
        )
    return summary


def build_phase_a_gate_report(arm_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    arms = {
        str(name): summary
        for name, summary in arm_summaries.items()
        if isinstance(summary, dict)
    }
    present = {name: name in arms for name in PHASE_A_REQUIRED_ARMS}
    invalid_reasons: List[str] = []
    if not all(present.values()):
        invalid_reasons.append("missing_required_arm")

    compared = {name: arms.get(name, {}) for name in PHASE_A_REQUIRED_ARMS}
    if all(present.values()):
        manifest_values = {
            name: str(summary.get("manifest_sha256") or "")
            for name, summary in compared.items()
        }
        nonempty_manifest_values = {value for value in manifest_values.values() if value}
        if len(nonempty_manifest_values) != 1 or "" in manifest_values.values():
            invalid_reasons.append("manifest_sha256_mismatch")

        threshold_values = {
            name: _canonical_json(summary.get("phase0_stop_go_thresholds"))
            for name, summary in compared.items()
        }
        nonempty_threshold_values = {value for value in threshold_values.values() if value != "null"}
        if len(nonempty_threshold_values) != 1 or "null" in threshold_values.values():
            invalid_reasons.append("phase0_stop_go_thresholds_mismatch")

        trigger_counts = {
            _int_value(summary.get("shared_trigger_key_count"))
            for summary in compared.values()
        }
        if len(trigger_counts) > 1:
            invalid_reasons.append("shared_trigger_key_count_mismatch")
        if any(_int_value(summary.get("unmatched_trigger_count")) for summary in compared.values()):
            invalid_reasons.append("shared_trigger_key_mismatch")

        event_gated = compared["event_gated_smoke"]
        policy_counts = event_gated.get("keyframe_policy_mode_counts")
        policy_counts = policy_counts if isinstance(policy_counts, dict) else {}
        if not (
            _int_value(event_gated.get("keyframe_gate_trace_count")) > 0
            and _int_value(policy_counts.get("event_gated_smoke")) > 0
        ):
            invalid_reasons.append("event_gated_smoke_not_validated")
        if _int_value(event_gated.get("keyframe_cap_validation_failed_count")):
            invalid_reasons.append("event_gated_keyframe_cap_validation_failed")
        if _int_value(event_gated.get("keyframe_write_failed_count")):
            invalid_reasons.append("event_gated_keyframe_write_failed")

    event_gated = compared.get("event_gated_smoke", {})
    audit_control = compared.get("interval_10_smoke_audit_control", {})
    interval = compared.get("interval_10", {})
    gate_deltas = _metric_deltas(event_gated, audit_control, PHASE_A_DELTA_METRICS)
    audit_control_deltas = _metric_deltas(audit_control, interval, PHASE_A_DELTA_METRICS)
    retrieval_audit_control_explains_gain = _retrieval_audit_control_explains_gain(
        interval,
        audit_control,
        event_gated,
    )
    route_event_coverage_regressed = _route_event_coverage_regressed(
        audit_control,
        event_gated,
    )
    valid = not invalid_reasons
    if not valid:
        claim_scope = "invalid_comparison"
        allowed_interpretation = "no_event_gated_claim"
    elif retrieval_audit_control_explains_gain or route_event_coverage_regressed:
        claim_scope = "traceability_only"
        allowed_interpretation = "traceability_and_evidence_quality_only"
    else:
        claim_scope = "gate_attributable_traceability_candidate"
        allowed_interpretation = "traceability_and_evidence_quality_only"

    manifest_sha256 = ""
    thresholds: Dict[str, Any] = {}
    shared_trigger_key_count = 0
    if all(present.values()):
        manifest_sha256 = str(compared["interval_10"].get("manifest_sha256") or "")
        raw_thresholds = compared["interval_10"].get("phase0_stop_go_thresholds")
        thresholds = dict(raw_thresholds) if isinstance(raw_thresholds, dict) else {}
        shared_trigger_key_count = _int_value(compared["interval_10"].get("shared_trigger_key_count"))
    return {
        "phase_a_required_arms": list(PHASE_A_REQUIRED_ARMS),
        "phase_a_required_arms_present": present,
        "phase_a_comparison_valid": valid,
        "phase_a_invalid_reasons": sorted(set(invalid_reasons)),
        "phase_a_manifest_sha256": manifest_sha256,
        "phase_a_stop_go_thresholds": thresholds,
        "phase_a_shared_trigger_key_count": shared_trigger_key_count,
        "phase_a_claim_scope": claim_scope,
        "phase_a_allowed_interpretation": allowed_interpretation,
        "retrieval_audit_control_explains_gain": retrieval_audit_control_explains_gain,
        "route_event_coverage_regressed": route_event_coverage_regressed,
        "gate_attributable_deltas": gate_deltas,
        "retrieval_audit_control_deltas": audit_control_deltas,
        "phase_a_no_speed_claim": True,
        "phase_a_no_navigation_success_claim": True,
    }


def build_endpoint_report(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_by_case_and_mode(_primary_blocks(list(blocks)))
    rows: List[Dict[str, Any]] = []
    for definition in ENDPOINT_DEFINITIONS:
        counts = _endpoint_counts(grouped, definition)
        rows.append(
            {
                "claim": definition["claim"],
                "endpoint": definition["endpoint"],
                "baseline_mode": definition["baseline_mode"],
                "denominator_description": definition["denominator_description"],
                "denominator": counts["denominator"],
                "pass_rule": definition["pass_rule"],
                "failure_handling": definition["failure_handling"],
                "minimum_effect_size": "pre_registered_positive_effect",
                "paired_test_or_confidence_interval": (
                    "paired confidence interval when denominator >= 30; otherwise descriptive"
                ),
                "downgrade_to_descriptive_rule": (
                    "downgrade when denominator < 30, confidence interval is unavailable, or effect is non-positive"
                ),
                "allowed_interpretation": definition["allowed_interpretation"],
                "interpretation_scope": "mechanism_validation_only",
                "v4_support_count": counts["v4_support_count"],
                "baseline_support_count": counts["baseline_support_count"],
                "effect_count": counts["effect_count"],
                "v4_rate": _rate(counts["v4_support_count"], counts["denominator"]),
                "baseline_rate": _rate(counts["baseline_support_count"], counts["denominator"]),
                "excluded_counts": counts["excluded_counts"],
            }
        )
    return rows


def format_visual_readback_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# OpenClaw Visual Readback Report",
        "",
        "Phase 1 interpretation: mechanism validation only.",
        "",
        "Smoke keyframe metrics are traceability and evidence-quality diagnostics only.",
        "",
        "Navigation metrics, when present, are secondary diagnostics for this phase.",
        "",
        "## Counts",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "trace_rows",
        "primary_quantitative_set_count",
        "offline_stress_set_count",
        "readback_trigger_count",
        "readback_completed_count",
        "visual_grounded_readback_count",
        "adjudication_correct_readback_count",
        "grounded_but_wrong_label_count",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")

    if summary.get("keyframe_gate_trace_count"):
        lines.extend(
            [
                "",
                "## Smoke Keyframes",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        for key in (
            "keyframe_gate_trace_count",
            "keyframe_saved_count",
            "keyframe_skipped_count",
            "initial_context_keyframe_count",
            "direct_save_segment_count",
            "cooldown_skip_count",
            "action_segment_duplicate_skip_count",
            "episode_cap_reached_skip_count",
            "keyframe_cap_validation_failed_count",
            "keyframe_write_failed_count",
            "retrieval_text_failure_count",
        ):
            lines.append(f"| {key} | {summary.get(key, 0)} |")

    if summary.get("candidate_pool_attempt_count"):
        lines.extend(
            [
                "",
                "## Candidate Pool",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        for key in (
            "candidate_pool_attempt_count",
            "eligible_prior_keyframe_total",
            "candidate_pool_backend_returned_total",
            "candidate_pool_exact_duplicate_drop_total",
            "candidate_pool_attached_total",
            "memory_query_unavailable_count",
            "memory_query_failed_count",
            "memory_query_timeout_count",
            "no_eligible_prior_keyframe_count",
            "no_memory_hit_count",
        ):
            lines.append(f"| {key} | {summary.get(key, 0)} |")

    lines.extend(
        [
            "",
            "## Endpoints",
            "",
            "| claim | endpoint | denominator | V4 support | baseline support | exclusions |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary.get("endpoint_report") or []:
        exclusions = json.dumps(row.get("excluded_counts") or {}, sort_keys=True)
        lines.append(
            "| {claim} | {endpoint} | {denominator} | {v4_support_count} | "
            "{baseline_support_count} | `{exclusions}` |".format(
                exclusions=exclusions,
                **row,
            )
        )
    return "\n".join(lines)


def format_phase_a_gate_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# OpenClaw Visual Readback Phase A Report",
        "",
        f"Comparison valid: {report.get('phase_a_comparison_valid')}",
        f"Claim scope: {report.get('phase_a_claim_scope')}",
        f"Allowed interpretation: {report.get('phase_a_allowed_interpretation')}",
        "",
        "This report does not support speed, SR/SPL, or executed-policy improvement claims.",
        "",
        "## Required Arms",
        "",
        "| arm | present |",
        "| --- | ---: |",
    ]
    present = report.get("phase_a_required_arms_present")
    present = present if isinstance(present, dict) else {}
    for arm in report.get("phase_a_required_arms") or PHASE_A_REQUIRED_ARMS:
        lines.append(f"| {arm} | {present.get(arm, False)} |")
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| metric | gate vs audit-control | audit-control vs interval |",
            "| --- | ---: | ---: |",
        ]
    )
    gate_deltas = report.get("gate_attributable_deltas")
    audit_deltas = report.get("retrieval_audit_control_deltas")
    gate_deltas = gate_deltas if isinstance(gate_deltas, dict) else {}
    audit_deltas = audit_deltas if isinstance(audit_deltas, dict) else {}
    for metric in PHASE_A_DELTA_METRICS:
        lines.append(
            f"| {metric} | {gate_deltas.get(metric)} | {audit_deltas.get(metric)} |"
        )
    return "\n".join(lines)


def _phase0_summary_fields(
    trace_paths: Iterable[Path],
    phase0_manifest_path: Path,
    expected_manifest_sha256: str = "",
    expected_phase0_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from harness.visual_readback.audit import build_phase0_audit_report

    audit = build_phase0_audit_report(trace_paths, manifest_path=phase0_manifest_path)
    valid = bool(audit.get("phase0_comparison_valid", True))
    invalid_reasons: List[str] = []
    if expected_manifest_sha256 and expected_manifest_sha256 != audit.get("manifest_sha256"):
        valid = False
        invalid_reasons.append("manifest_sha256_mismatch")
    if expected_phase0_thresholds is not None and expected_phase0_thresholds != audit.get(
        "phase0_stop_go_thresholds"
    ):
        valid = False
        invalid_reasons.append("phase0_stop_go_thresholds_mismatch")
    fields = {
        "phase0_manifest_path": str(phase0_manifest_path),
        "manifest_schema_version": audit.get("manifest_schema_version", ""),
        "manifest_sha256": audit.get("manifest_sha256", ""),
        "phase0_stop_go_thresholds": audit.get("phase0_stop_go_thresholds", {}),
        "phase0_threshold_results": audit.get("phase0_threshold_results", {}),
        "phase0_comparison_valid": valid,
        "phase0_invalid_reasons": invalid_reasons,
        "phase0_stop_go_decision": (
            audit.get("phase0_stop_go_decision") if valid else "invalid_comparison"
        ),
        "phase0_audit": audit,
    }
    return fields


def _keyframe_gate_summary(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    gates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for block in blocks:
        gate = block.get("keyframe_gate")
        if isinstance(gate, dict):
            gates.append((block, gate))

    policy_counts: Counter[str] = Counter()
    saved_by_reason: Counter[str] = Counter()
    skipped_by_reason: Counter[str] = Counter()
    memory_link_counts: Counter[str] = Counter()
    candidate_event_counts: Counter[str] = Counter()
    confirmed_event_counts: Counter[str] = Counter()
    controller_status_counts: Counter[str] = Counter()
    candidate_action_status_counts: Counter[str] = Counter()
    write_status_counts: Counter[str] = Counter()
    promotion_status_counts: Counter[str] = Counter()
    direct_save_segments = set()
    saved_count = 0
    skipped_count = 0
    initial_context_count = 0
    cooldown_skip_count = 0
    action_segment_duplicate_skip_count = 0
    episode_cap_reached_skip_count = 0
    cap_validation_failed_count = 0
    write_failed_count = 0
    retrieval_text_failure_count = 0

    for block, gate in gates:
        policy = str(gate.get("keyframe_policy_mode") or "")
        if policy:
            policy_counts[policy] += 1
        decision = str(gate.get("save_decision") or "")
        save_reason = str(gate.get("save_reason") or "")
        skip_reason = str(gate.get("skip_reason") or "")
        if decision == "save":
            saved_count += 1
            if save_reason:
                saved_by_reason[save_reason] += 1
            if save_reason == "initial_context" or gate.get("candidate_event_type") == "initial_context":
                initial_context_count += 1
            if save_reason == "candidate_decision_point":
                direct_save_segments.add(
                    (
                        str(block.get("scene_id") or ""),
                        str(block.get("episode_id") or ""),
                        str(gate.get("candidate_segment_id") or block.get("step_id") or ""),
                    )
                )
        elif decision == "skip":
            skipped_count += 1
            if skip_reason:
                skipped_by_reason[skip_reason] += 1

        if skip_reason == "cooldown" or _truthy(gate.get("cooldown_active")):
            cooldown_skip_count += 1
        if skip_reason == "action_segment_duplicate":
            action_segment_duplicate_skip_count += 1
        if skip_reason == "episode_cap_reached":
            episode_cap_reached_skip_count += 1
        if _truthy(gate.get("keyframe_cap_validation_failed")):
            cap_validation_failed_count += 1

        write_status = str(gate.get("write_status") or "")
        if write_status:
            write_status_counts[write_status] += 1
        if write_status in KEYFRAME_WRITE_FAILURE_STATUSES or (
            write_status.endswith("_failed") and write_status != "not_attempted"
        ):
            write_failed_count += 1

        link_status = str(gate.get("memory_id_link_status") or "")
        if link_status:
            memory_link_counts[link_status] += 1
        promotion_status = str(gate.get("promotion_status") or "")
        if promotion_status:
            promotion_status_counts[promotion_status] += 1
        candidate_event = str(gate.get("candidate_event_type") or "")
        if candidate_event:
            candidate_event_counts[candidate_event] += 1
        confirmed_event = str(gate.get("confirmed_event_type") or "")
        if confirmed_event:
            confirmed_event_counts[confirmed_event] += 1
        controller_status = str(gate.get("controller_event_status") or "")
        if controller_status:
            controller_status_counts[controller_status] += 1
        action_status = str(gate.get("candidate_action_status") or "")
        if action_status:
            candidate_action_status_counts[action_status] += 1
        failure_reason = str(gate.get("failure_reason") or "")
        if "retrieval_text" in failure_reason or "action_context" in failure_reason:
            retrieval_text_failure_count += 1

    return {
        "keyframe_policy_mode_counts": dict(sorted(policy_counts.items())),
        "keyframe_gate_trace_count": len(gates),
        "keyframe_saved_count": saved_count,
        "keyframe_skipped_count": skipped_count,
        "keyframe_saved_by_reason": dict(sorted(saved_by_reason.items())),
        "keyframe_skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "initial_context_keyframe_count": initial_context_count,
        "direct_save_segment_count": len(direct_save_segments),
        "cooldown_skip_count": cooldown_skip_count,
        "action_segment_duplicate_skip_count": action_segment_duplicate_skip_count,
        "episode_cap_reached_skip_count": episode_cap_reached_skip_count,
        "keyframe_cap_validation_failed_count": cap_validation_failed_count,
        "keyframe_write_failed_count": write_failed_count,
        "retrieval_text_failure_count": retrieval_text_failure_count,
        "memory_id_link_status_counts": dict(sorted(memory_link_counts.items())),
        "candidate_event_type_counts": dict(sorted(candidate_event_counts.items())),
        "confirmed_event_type_counts": dict(sorted(confirmed_event_counts.items())),
        "controller_event_status_counts": dict(sorted(controller_status_counts.items())),
        "candidate_action_status_counts": dict(sorted(candidate_action_status_counts.items())),
        "keyframe_write_status_counts": dict(sorted(write_status_counts.items())),
        "keyframe_promotion_status_counts": dict(sorted(promotion_status_counts.items())),
    }


def _candidate_pool_summary(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    attempts = [
        block
        for block in blocks
        if block.get("candidate_pool_source")
        or any(key in block for key in CANDIDATE_POOL_SUM_KEYS)
    ]
    source_counts = Counter(
        str(block.get("candidate_pool_source") or "")
        for block in attempts
        if block.get("candidate_pool_source")
    )
    skip_counts = Counter(str(block.get("skip_reason") or "") for block in blocks)
    return {
        "candidate_pool_attempt_count": len(attempts),
        "candidate_pool_requested_total": sum(
            _int_value(block.get("candidate_pool_requested_count"))
            for block in attempts
        ),
        "eligible_prior_keyframe_total": sum(
            _int_value(block.get("eligible_prior_keyframe_count"))
            for block in attempts
        ),
        "candidate_pool_backend_returned_total": sum(
            _int_value(block.get("candidate_pool_backend_returned_count"))
            for block in attempts
        ),
        "candidate_pool_exact_duplicate_drop_total": sum(
            _int_value(block.get("candidate_pool_exact_duplicate_drop_count"))
            for block in attempts
        ),
        "candidate_pool_attached_total": sum(
            _int_value(block.get("candidate_pool_attached_count"))
            for block in attempts
        ),
        "candidate_pool_source_counts": dict(sorted(source_counts.items())),
        "memory_query_unavailable_count": skip_counts.get("memory_query_unavailable", 0),
        "memory_query_failed_count": skip_counts.get("memory_query_failed", 0),
        "memory_query_timeout_count": skip_counts.get("memory_query_timeout", 0),
        "no_eligible_prior_keyframe_count": skip_counts.get("no_eligible_prior_keyframe", 0),
        "no_memory_hit_count": skip_counts.get("no_memory_hit", 0),
        "retrieval_failure_counts": {
            "memory_query_unavailable": skip_counts.get("memory_query_unavailable", 0),
            "memory_query_failed": skip_counts.get("memory_query_failed", 0),
            "memory_query_timeout": skip_counts.get("memory_query_timeout", 0),
            "no_eligible_prior_keyframe": skip_counts.get("no_eligible_prior_keyframe", 0),
            "no_memory_hit": skip_counts.get("no_memory_hit", 0),
        },
    }


def _interval_cleanup_attachment_metrics(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    retrieved_total = 0
    deduped_total = 0
    duplicate_drops = 0
    adjacent_triplet_count = 0
    adjacent_denominator = 0
    for block in blocks:
        paths = [str(path) for path in _list_value(block.get("retrieved_image_paths")) if str(path)]
        if not paths:
            continue
        retrieved_total += len(paths)
        unique_paths = list(dict.fromkeys(paths))
        deduped_total += len(unique_paths)
        duplicate_drops += len(paths) - len(unique_paths)
        steps = [_step_from_path(path) for path in unique_paths]
        step_values = [step for step in steps if step is not None]
        if len(step_values) >= 3:
            adjacent_denominator += 1
            if _contains_adjacent_triplet(step_values):
                adjacent_triplet_count += 1
    return {
        "interval_cleanup_mode": "interval_10_exact_dedupe_summary",
        "retrieved_memory_image_count": retrieved_total,
        "deduped_retrieved_memory_image_count": deduped_total,
        "exact_duplicate_drop_count": duplicate_drops,
        "exact_duplicate_rate": _rate(duplicate_drops, retrieved_total),
        "adjacent_triplet_count": adjacent_triplet_count,
        "adjacent_triplet_denominator": adjacent_denominator,
        "adjacent_triplet_rate": _rate(adjacent_triplet_count, adjacent_denominator),
    }


def _trace_paths(run_path: Path) -> List[Path]:
    if run_path.is_file():
        return [run_path]
    trace_dir = run_path / "harness_traces"
    if trace_dir.exists():
        return sorted(trace_dir.glob("harness_trace_rank*.jsonl"))
    return sorted(run_path.glob("**/harness_trace_rank*.jsonl"))


def _load_trace_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_summary_json(run_path: Path) -> Dict[str, Any]:
    path = run_path / "summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _visual_readback_block(row: Dict[str, Any]) -> Dict[str, Any]:
    block = row.get("visual_readback") or row.get("visual_memory_read") or {}
    merged = dict(row)
    if isinstance(block, dict):
        merged.update(block)
    if "scene_id" not in merged and row.get("scene_id"):
        merged["scene_id"] = row.get("scene_id")
    if "episode_id" not in merged and row.get("episode_id"):
        merged["episode_id"] = row.get("episode_id")
    if "step_id" not in merged and row.get("step_id") is not None:
        merged["step_id"] = row.get("step_id")
    return merged


def _primary_blocks(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        block
        for block in blocks
        if str(block.get("trigger_source") or "") in PRIMARY_TRIGGER_SOURCES
    ]


def _offline_blocks(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        block
        for block in blocks
        if str(block.get("trigger_source") or "") in OFFLINE_TRIGGER_SOURCES
    ]


def _unique_cases(blocks: Iterable[Dict[str, Any]]) -> set[Tuple[str, str, str, str]]:
    return {_case_key(block) for block in blocks}


def _case_key(block: Dict[str, Any]) -> Tuple[str, str, str, str]:
    case_id = str(block.get("case_id") or "")
    if case_id:
        return (case_id, "", "", "")
    return (
        str(block.get("scene_id") or ""),
        str(block.get("episode_id") or ""),
        str(block.get("step_id") or ""),
        str(block.get("trigger_rule") or ""),
    )


def _has_trigger(block: Dict[str, Any]) -> bool:
    return bool(block.get("trigger_source") or block.get("trigger_rule") or block.get("mode"))


def _group_by_case_and_mode(blocks: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for block in blocks:
        mode = str(block.get("mode") or block.get("visual_readback_mode") or "")
        if not mode:
            continue
        grouped[_case_key(block)][mode] = block
    return grouped


def _endpoint_counts(
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]],
    definition: Dict[str, str],
) -> Dict[str, Any]:
    excluded_counts = _empty_excluded_counts()
    denominator = 0
    v4_support_count = 0
    baseline_support_count = 0
    effect_count = 0
    baseline_mode = definition["baseline_mode"]
    endpoint = definition["endpoint"]
    for modes in grouped.values():
        v4 = modes.get(V4_MODE)
        baseline = modes.get(baseline_mode)
        if not v4 or not baseline:
            continue
        exclusion = _endpoint_exclusion(endpoint, v4, baseline)
        if exclusion:
            excluded_counts[exclusion] += 1
            continue
        denominator += 1
        v4_support = _support_for_endpoint(endpoint, v4)
        baseline_support = _support_for_endpoint(endpoint, baseline)
        if v4_support:
            v4_support_count += 1
        if baseline_support:
            baseline_support_count += 1
        if v4_support and not baseline_support:
            effect_count += 1
    return {
        "denominator": denominator,
        "v4_support_count": v4_support_count,
        "baseline_support_count": baseline_support_count,
        "effect_count": effect_count,
        "excluded_counts": dict(excluded_counts),
    }


def _endpoint_exclusion(endpoint: str, v4: Dict[str, Any], baseline: Dict[str, Any]) -> Optional[str]:
    replay_exclusion = _replay_or_judgeable_exclusion(v4, baseline)
    if replay_exclusion:
        return replay_exclusion
    if endpoint == "visual_grounded_readback_rate" and (
        not _has_image_path(v4) or not _has_image_path(baseline)
    ):
        return "missing_image_path"
    if endpoint == "paired_adjudicated_support_rate":
        if _is_prompt_contaminated(v4) or _is_prompt_contaminated(baseline):
            return "prompt_contaminated"
        if not _has_shared_labels(v4, baseline):
            return "missing_shared_labels"
    if endpoint == "historical_memory_incremental_effect_rate" and (
        _truthy(v4.get("current_only_sufficient")) or _truthy(baseline.get("current_only_sufficient"))
    ):
        return "current_only_sufficient"
    if endpoint == "historical_memory_dependent_effect_rate" and str(
        baseline.get("negative_control_label") or ""
    ) == "negative_control_unjudgeable":
        return "negative_control_unjudgeable"
    return None


def _replay_or_judgeable_exclusion(v4: Dict[str, Any], baseline: Dict[str, Any]) -> Optional[str]:
    for block in (v4, baseline):
        status = str(block.get("replay_status") or block.get("status") or "")
        if status in {"replay_state_mismatch", "replay_mismatch"}:
            return status
    for block in (v4, baseline):
        if str(block.get("adjudication_status") or "") in {"unjudgeable", "negative_control_unjudgeable"}:
            return "unjudgeable"
    return None


def _support_for_endpoint(endpoint: str, block: Dict[str, Any]) -> bool:
    if endpoint == "visual_grounded_readback_rate":
        return _is_grounded(block)
    return _is_grounded(block) and _is_adjudication_correct(block)


def _is_grounded(block: Dict[str, Any]) -> bool:
    return normalize_visual_grounding_status(
        block.get("visual_grounding_status")
    ) == "grounded"


def _is_adjudication_correct(block: Dict[str, Any]) -> bool:
    return str(block.get("adjudication_status") or "") in {"correct", "adjudication_correct", "supported"}


def _is_grounded_but_wrong(block: Dict[str, Any]) -> bool:
    return _is_grounded(block) and str(block.get("adjudication_status") or "") in {
        "wrong",
        "incorrect",
        "grounded_but_wrong",
    }


def _has_image_path(block: Dict[str, Any]) -> bool:
    return bool(
        _list_value(block.get("actually_read_image_paths"))
        or _list_value(block.get("retrieved_image_paths"))
        or block.get("current_image_path")
        or block.get("image_path")
    )


def _is_prompt_contaminated(block: Dict[str, Any]) -> bool:
    return any(
        _truthy(block.get(key))
        for key in (
            "prompt_contaminated",
            "actual_policy_payload_merge",
            "readback_state_used_by_policy",
            "final_policy_payload_has_memory",
        )
    )


def _has_shared_labels(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_labels = set(_list_value(left.get("required_verifier_labels")) or _list_value(left.get("verifier_labels")))
    right_labels = set(_list_value(right.get("required_verifier_labels")) or _list_value(right.get("verifier_labels")))
    return bool(left_labels.intersection(right_labels))


def _verifier_label_confusion(blocks: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    per_label: Dict[str, Counter[str]] = defaultdict(Counter)
    for block in blocks:
        required = set(_list_value(block.get("required_verifier_labels")))
        predicted = set(_list_value(block.get("verifier_labels")))
        for label in required.intersection(predicted):
            per_label[label]["true_positive"] += 1
        for label in required.difference(predicted):
            per_label[label]["false_negative"] += 1
        for label in predicted.difference(required):
            per_label[label]["false_positive"] += 1
    return {
        label: {
            "true_positive": counts.get("true_positive", 0),
            "false_negative": counts.get("false_negative", 0),
            "false_positive": counts.get("false_positive", 0),
        }
        for label, counts in sorted(per_label.items())
    }


def _secondary_navigation_metrics(summary_json: Dict[str, Any]) -> Dict[str, Any]:
    return {key: summary_json[key] for key in NAVIGATION_METRIC_KEYS if key in summary_json}


def _visual_readback_config(summary_json: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    config = summary_json.get("visual_readback_config")
    if isinstance(config, dict):
        return dict(config)
    for block in blocks:
        config = block.get("visual_readback_config")
        if isinstance(config, dict):
            return dict(config)
    return {}


def _empty_excluded_counts() -> Counter[str]:
    counter: Counter[str] = Counter()
    for key in EXCLUDED_COUNT_KEYS:
        counter[key] = 0
    return counter


def _stable_memory_ids(block: Dict[str, Any]) -> List[Any]:
    ids = _list_value(block.get("matched_memory_ids"))
    if ids:
        return ids
    return _list_value(block.get("attached_memory_ids"))


def _step_from_path(path: str) -> Optional[int]:
    match = re.search(r"(?:step_|keyframe_)?(\d+)(?:\.[A-Za-z0-9]+)?$", path)
    if not match:
        return None
    return int(match.group(1))


def _contains_adjacent_triplet(steps: List[int]) -> bool:
    unique_steps = sorted(set(steps))
    for first, second, third in zip(unique_steps, unique_steps[1:], unique_steps[2:]):
        if second - first == third - second:
            return True
    return False


def _list_value(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value == "":
        return []
    return [value]


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _bool_status(block: Dict[str, Any], key: str) -> str:
    if key not in block:
        return "missing"
    value = block.get(key)
    if isinstance(value, bool):
        return "true" if value else "false"
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return "true"
    if normalized in {"0", "false", "no", "n", "off"}:
        return "false"
    return "missing"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _number_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_deltas(
    left: Dict[str, Any],
    right: Dict[str, Any],
    metric_names: Iterable[str],
) -> Dict[str, Optional[float]]:
    deltas: Dict[str, Optional[float]] = {}
    for metric in metric_names:
        left_value = _number_value(left.get(metric))
        right_value = _number_value(right.get(metric))
        deltas[metric] = None if left_value is None or right_value is None else left_value - right_value
    return deltas


def _lower_than(left: Dict[str, Any], right: Dict[str, Any], metric: str) -> bool:
    left_value = _number_value(left.get(metric))
    right_value = _number_value(right.get(metric))
    return left_value is not None and right_value is not None and left_value < right_value


def _greater_than(left: Dict[str, Any], right: Dict[str, Any], metric: str) -> bool:
    left_value = _number_value(left.get(metric))
    right_value = _number_value(right.get(metric))
    return left_value is not None and right_value is not None and left_value > right_value


def _retrieval_audit_control_explains_gain(
    interval: Dict[str, Any],
    audit_control: Dict[str, Any],
    event_gated: Dict[str, Any],
) -> bool:
    control_improved = (
        _lower_than(audit_control, interval, "exact_duplicate_rate")
        or _lower_than(audit_control, interval, "adjacent_triplet_rate")
        or _greater_than(audit_control, interval, "memory_evidence_used_count")
        or _greater_than(audit_control, interval, "candidate_pool_attached_total")
    )
    event_gated_outperforms_control = (
        _lower_than(event_gated, audit_control, "exact_duplicate_rate")
        or _lower_than(event_gated, audit_control, "adjacent_triplet_rate")
        or _greater_than(event_gated, audit_control, "memory_evidence_used_count")
        or _greater_than(event_gated, audit_control, "candidate_pool_attached_total")
    )
    return control_improved and not event_gated_outperforms_control


def _route_event_coverage_regressed(
    audit_control: Dict[str, Any],
    event_gated: Dict[str, Any],
) -> bool:
    control_miss_rate = _number_value(audit_control.get("route_event_miss_rate"))
    event_miss_rate = _number_value(event_gated.get("route_event_miss_rate"))
    if control_miss_rate is None or event_miss_rate is None:
        return False
    return event_miss_rate > control_miss_rate


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _rate(count: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return count / denominator
