#!/usr/bin/env python
"""Compare qwen_direct_policy results against a Janus baseline on shared keys."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ABLATION_ARMS = ("fixed_off", "dynamic_off", "fixed_on", "dynamic_on")
PINNED_QWEN_MODEL = "qwen3.5-flash-2026-02-23"


def compare_result_files(
    qwen_result_path: Path | str,
    janus_result_path: Path | str,
    *,
    qwen_trace_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    qwen_rows, qwen_duplicates = _load_episode_rows(Path(qwen_result_path))
    janus_rows, janus_duplicates = _load_episode_rows(Path(janus_result_path))

    qwen_keys = set(qwen_rows)
    janus_keys = set(janus_rows)
    common_keys = sorted(qwen_keys & janus_keys)

    qwen_common = [qwen_rows[key] for key in common_keys]
    janus_common = [janus_rows[key] for key in common_keys]

    report = {
        "qwen_result_path": str(qwen_result_path),
        "janus_result_path": str(janus_result_path),
        "common_key_count": len(common_keys),
        "qwen_count": len(qwen_rows),
        "janus_count": len(janus_rows),
        "duplicate_qwen_keys": sorted(qwen_duplicates),
        "duplicate_janus_keys": sorted(janus_duplicates),
        "missing_in_qwen": sorted(janus_keys - qwen_keys),
        "missing_in_janus": sorted(qwen_keys - janus_keys),
        "qwen_metrics": _metrics(qwen_common),
        "janus_metrics": _metrics(janus_common),
        "overlap_buckets": _overlap_buckets(qwen_common, janus_common),
    }
    if qwen_trace_path is not None:
        report["trace_preflight"] = validate_qwen_direct_trace(Path(qwen_trace_path))
    return report


def validate_qwen_direct_trace(trace_path: Path | str) -> Dict[str, Any]:
    rows = _load_jsonl(Path(trace_path))
    janus_loaded_count = 0
    navigation_policy_skill_called_count = 0
    planned_navigation_policy_skill_count = 0
    non_direct_backend_count = 0
    qwen_failure_count = 0
    action_counts: Counter[str] = Counter()
    candidate_action_counts: Counter[str] = Counter()
    final_action_source_counts: Counter[str] = Counter()
    planner_step_mode_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()

    for row in rows:
        audit = row.get("context_audit") if isinstance(row.get("context_audit"), dict) else {}
        policy_backend = row.get("policy_backend") or audit.get("policy_backend")
        if policy_backend and policy_backend != "qwen_direct":
            non_direct_backend_count += 1
        if row.get("janus_loaded") is True or audit.get("janus_loaded") is True:
            janus_loaded_count += 1
        if row.get("navigation_policy_skill_called") is True:
            navigation_policy_skill_called_count += 1
        if row.get("planned_tool") == "NavigationPolicySkill":
            planned_navigation_policy_skill_count += 1
        final_action = str(row.get("final_action") or row.get("action_text") or "")
        if final_action:
            action_counts[final_action] += 1
        candidate_action = str(row.get("candidate_action") or "")
        if candidate_action:
            candidate_action_counts[candidate_action] += 1
        final_action_source = str(row.get("final_action_source") or "")
        if final_action_source:
            final_action_source_counts[final_action_source] += 1
        planner_step_mode = str(row.get("planner_step_mode") or audit.get("planner_step_mode") or "")
        if planner_step_mode:
            planner_step_mode_counts[planner_step_mode] += 1
        for gate_key in (
            "stop_gate_decision",
            "loop_gate_decision",
            "uncertainty_gate_decision",
            "forward_stall_gate_decision",
        ):
            if gate_key in row:
                gate_counts[gate_key] += 1
        if row.get("qwen_failure") is True or audit.get("qwen_failure") is True:
            qwen_failure_count += 1

    valid = not any(
        (
            janus_loaded_count,
            navigation_policy_skill_called_count,
            planned_navigation_policy_skill_count,
            non_direct_backend_count,
        )
    )
    behavior_warnings: List[Dict[str, Any]] = []
    if rows and action_counts.get("MOVE_FORWARD") == len(rows):
        behavior_warnings.append(
            {"code": "all_forward_actions", "count": len(rows)}
        )
    return {
        "trace_path": str(trace_path),
        "valid": valid,
        "trace_rows": len(rows),
        "janus_loaded_count": janus_loaded_count,
        "navigation_policy_skill_called_count": navigation_policy_skill_called_count,
        "planned_navigation_policy_skill_count": planned_navigation_policy_skill_count,
        "non_direct_backend_count": non_direct_backend_count,
        "action_counts": dict(sorted(action_counts.items())),
        "candidate_action_counts": dict(sorted(candidate_action_counts.items())),
        "final_action_source_counts": dict(sorted(final_action_source_counts.items())),
        "planner_step_mode_counts": dict(sorted(planner_step_mode_counts.items())),
        "gate_counts": dict(sorted(gate_counts.items())),
        "qwen_failure_count": qwen_failure_count,
        "behaviorally_suspicious": bool(behavior_warnings),
        "behavior_warnings": behavior_warnings,
    }


def compare_ablation_2x2(arms: Dict[str, Dict[str, Path | str]]) -> Dict[str, Any]:
    missing_arms = [name for name in ABLATION_ARMS if name not in arms]
    if missing_arms:
        raise ValueError(f"missing 2x2 arms: {', '.join(missing_arms)}")
    report_arms: Dict[str, Dict[str, Any]] = {}
    key_sets: Dict[str, set[str]] = {}
    for name in ABLATION_ARMS:
        spec = arms[name]
        result_path = Path(spec["result"])
        trace_path = Path(spec["trace"])
        rows, duplicates = _load_episode_rows(result_path)
        keys = set(rows)
        key_sets[name] = keys
        trace_audit = _ablation_trace_audit(
            trace_path,
            expected_dynamic=name.startswith("dynamic"),
            expected_thinking=name.endswith("on"),
            result_keys=keys,
        )
        invalid_reasons = list(trace_audit["invalid_reasons"])
        if duplicates:
            invalid_reasons.append("duplicate_episode_keys")
        if not keys:
            invalid_reasons.append("empty_result_set")
        report_arms[name] = {
            "result_path": str(result_path),
            "trace_path": str(trace_path),
            "episode_keys": sorted(keys),
            "duplicate_episode_keys": sorted(duplicates),
            "metrics": _metrics(rows.values()),
            "trace_audit": trace_audit,
            "valid": not invalid_reasons,
            "invalid_reasons": sorted(set(invalid_reasons)),
        }
    exact_keys = key_sets[ABLATION_ARMS[0]]
    exact_key_match = all(key_sets[name] == exact_keys for name in ABLATION_ARMS[1:])
    if not exact_key_match:
        for arm in report_arms.values():
            arm["valid"] = False
            arm["invalid_reasons"] = sorted(
                set(arm["invalid_reasons"] + ["incomplete_exact_episode_keys"])
            )
    comparisons = {
        "phase3_dynamic_visual": _ablation_comparison(
            report_arms,
            "fixed_off",
            "dynamic_off",
            exact_key_match,
        ),
        "phase4_thinking": _ablation_comparison(
            report_arms,
            "dynamic_off",
            "dynamic_on",
            exact_key_match,
        ),
        "thinking_fixed_selection": _ablation_comparison(
            report_arms,
            "fixed_off",
            "fixed_on",
            exact_key_match,
        ),
        "dynamic_with_thinking": _ablation_comparison(
            report_arms,
            "fixed_on",
            "dynamic_on",
            exact_key_match,
        ),
    }
    return {
        "valid": exact_key_match and all(arm["valid"] for arm in report_arms.values()),
        "exact_key_match": exact_key_match,
        "exact_episode_keys": sorted(exact_keys) if exact_key_match else [],
        "arms": report_arms,
        "comparisons": comparisons,
    }


def _ablation_trace_audit(
    trace_path: Path,
    *,
    expected_dynamic: bool,
    expected_thinking: bool,
    result_keys: set[str],
) -> Dict[str, Any]:
    rows = _load_jsonl(trace_path)
    model_rows = []
    trace_keys = set()
    action_counts: Counter[str] = Counter()
    route_stage_counts: Counter[str] = Counter()
    selected_count_distribution: Counter[str] = Counter()
    selected_role_counts: Counter[str] = Counter()
    missing_role_counts: Counter[str] = Counter()
    final_action_changed_count = 0
    stop_block_count = 0
    requery_count = 0
    reasoning_tokens = 0
    latency_values: List[float] = []
    provider_failure_count = 0
    for row in rows:
        key = _episode_key(row)
        if key:
            trace_keys.add(key)
        action = str(row.get("final_action") or row.get("action_text") or "")
        if action:
            action_counts[action] += 1
        candidate = str(row.get("candidate_action") or "")
        if candidate and action and candidate != action:
            final_action_changed_count += 1
        if row.get("stop_gate_decision") == "blocked":
            stop_block_count += 1
        if row.get("qwen_direct_requery_triggered") is True:
            requery_count += 1
        stage = str(row.get("qwen_route_stage") or "")
        if stage:
            route_stage_counts[stage] += 1
        audit = row.get("context_audit") if isinstance(row.get("context_audit"), dict) else {}
        if row.get("qwen_failure") is True or audit.get("qwen_failure") is True:
            provider_failure_count += 1
        if audit.get("qwen_model_called") is not True:
            continue
        model_rows.append(audit)
        selected_count = audit.get("selected_image_count")
        if selected_count is not None:
            selected_count_distribution[str(selected_count)] += 1
        for role in audit.get("selected_image_roles") or audit.get("model_image_sources") or []:
            selected_role_counts[str(role)] += 1
        for role in audit.get("missing_image_roles") or []:
            missing_role_counts[str(role)] += 1
        reasoning_tokens += int(_float(audit.get("reasoning_tokens")))
        if audit.get("provider_latency_ms") is not None:
            latency_values.append(_float(audit.get("provider_latency_ms")))

    invalid_reasons: List[str] = []
    if not model_rows:
        invalid_reasons.append("no_qwen_model_calls")
    if not result_keys.issubset(trace_keys):
        invalid_reasons.append("incomplete_trace_episode_keys")
    required_values = {
        "dynamic_visual_context_enabled": expected_dynamic,
        "qwen_thinking_enabled": expected_thinking,
        "qwen_output_schema": "route_v2",
        "qwen_transport_mode": "sync",
        "qwen_temperature": 0,
        "model_image_count_policy": (
            "role_adaptive" if expected_dynamic else "legacy_cap"
        ),
        "openclaw_model_max_images_applied": not expected_dynamic,
        "openclaw_model_image_interval_steps": 1,
        "qwen_thinking_budget": 1024,
        "map_assist_mode": "floorplan_map_assisted",
        "map_frame_interval_steps": 5,
        "motion_feedback_enabled": True,
        "forward_stall_odometry_enabled": True,
        "map_collision_overlay_enabled": True,
    }
    for field, expected in required_values.items():
        if any(audit.get(field) != expected for audit in model_rows):
            invalid_reasons.append(f"unexpected_{field}")
    if any(
        audit.get("configured_model_id") != f"qwen/{PINNED_QWEN_MODEL}"
        or audit.get("configured_model_id_canonical") != PINNED_QWEN_MODEL
        for audit in model_rows
    ):
        invalid_reasons.append("mutable_or_wrong_configured_model")
    if any(
        audit.get("returned_model_id_canonical") != PINNED_QWEN_MODEL
        for audit in model_rows
    ):
        invalid_reasons.append("configured_returned_model_mismatch")
    if any(audit.get("qwen_output_json_valid") is not True for audit in model_rows):
        invalid_reasons.append("json_invalid_or_unproven")
    fallback_count = sum(
        audit.get("thinking_capability_fallback") is True for audit in model_rows
    )
    transport_count = sum(
        audit.get("thinking_transport_incompatible") is True for audit in model_rows
    )
    exercised_true_count = sum(
        audit.get("qwen_thinking_exercised") is True for audit in model_rows
    )
    if expected_thinking and fallback_count:
        invalid_reasons.append("thinking_capability_fallback")
    if expected_thinking and transport_count:
        invalid_reasons.append("thinking_transport_incompatible")
    if expected_thinking and exercised_true_count == 0:
        invalid_reasons.append("thinking_not_exercised")
    if expected_thinking and any(
        audit.get("thinking_model_supported") is not True for audit in model_rows
    ):
        invalid_reasons.append("thinking_support_unproven")
    if model_rows and len(latency_values) != len(model_rows):
        invalid_reasons.append("missing_provider_latency")
    return {
        "trace_rows": len(rows),
        "model_call_rows": len(model_rows),
        "trace_episode_keys": sorted(trace_keys),
        "valid": not invalid_reasons,
        "invalid_reasons": sorted(set(invalid_reasons)),
        "thinking_exercised_true_count": exercised_true_count,
        "thinking_fallback_count": fallback_count,
        "thinking_transport_incompatible_count": transport_count,
        "json_valid_count": sum(
            audit.get("qwen_output_json_valid") is True for audit in model_rows
        ),
        "json_valid_rate": (
            round(
                sum(audit.get("qwen_output_json_valid") is True for audit in model_rows)
                / len(model_rows),
                6,
            )
            if model_rows
            else None
        ),
        "thinking_exercised_rate": (
            round(exercised_true_count / len(model_rows), 6) if model_rows else None
        ),
        "reasoning_tokens": reasoning_tokens,
        "provider_latency_ms_mean": (
            round(sum(latency_values) / len(latency_values), 3) if latency_values else None
        ),
        "selected_image_count_distribution": dict(sorted(selected_count_distribution.items())),
        "selected_image_role_counts": dict(sorted(selected_role_counts.items())),
        "missing_image_role_counts": dict(sorted(missing_role_counts.items())),
        "missing_image_role_rate": (
            round(
                sum(missing_role_counts.values())
                / (
                    sum(missing_role_counts.values())
                    + sum(selected_role_counts.values())
                ),
                6,
            )
            if sum(missing_role_counts.values()) + sum(selected_role_counts.values())
            else None
        ),
        "route_stage_counts": dict(sorted(route_stage_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "stop_block_count": stop_block_count,
        "requery_count": requery_count,
        "controller_changed_action_count": final_action_changed_count,
        "provider_failure_count": provider_failure_count,
    }


def _ablation_comparison(
    arms: Dict[str, Dict[str, Any]],
    baseline_name: str,
    treatment_name: str,
    exact_key_match: bool,
) -> Dict[str, Any]:
    baseline = arms[baseline_name]
    treatment = arms[treatment_name]
    if not exact_key_match or not baseline["valid"] or not treatment["valid"]:
        outcome = "invalid_comparison"
    else:
        baseline_sr = baseline["metrics"]["success_rate"] or 0.0
        treatment_sr = treatment["metrics"]["success_rate"] or 0.0
        baseline_spl = baseline["metrics"]["spl_rate"] or 0.0
        treatment_spl = treatment["metrics"]["spl_rate"] or 0.0
        if treatment_sr > baseline_sr:
            outcome = "navigation_improvement"
        elif treatment_sr < baseline_sr:
            outcome = "negative"
        elif treatment_spl > baseline_spl:
            outcome = "efficiency_improvement_only"
        else:
            outcome = "no_improvement"
    return {
        "baseline": baseline_name,
        "treatment": treatment_name,
        "outcome": outcome,
        "success_rate_delta": _metric_delta(baseline, treatment, "success_rate"),
        "spl_rate_delta": _metric_delta(baseline, treatment, "spl_rate"),
    }


def _metric_delta(
    baseline: Dict[str, Any],
    treatment: Dict[str, Any],
    metric: str,
) -> Optional[float]:
    baseline_value = baseline["metrics"].get(metric)
    treatment_value = treatment["metrics"].get(metric)
    if baseline_value is None or treatment_value is None:
        return None
    return round(float(treatment_value) - float(baseline_value), 6)


def _load_episode_rows(path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    rows_by_key: Dict[str, Dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in _load_jsonl(path):
        key = _episode_key(row)
        if not key:
            continue
        counts[key] += 1
        rows_by_key.setdefault(key, row)
    duplicates = [key for key, count in counts.items() if count > 1]
    return rows_by_key, duplicates


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _episode_key(row: Dict[str, Any]) -> str:
    if "scene_id" not in row or "episode_id" not in row:
        return ""
    return f"{row.get('scene_id')}:{row.get('episode_id')}"


def _metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    row_list = list(rows)
    count = len(row_list)
    success_sum = sum(_float(row.get("success")) for row in row_list)
    spl_sum = sum(_float(row.get("spl")) for row in row_list)
    os_sum = sum(_float(row.get("oracle_success")) for row in row_list)
    return {
        "count": count,
        "success_rate": round(success_sum / count, 6) if count else None,
        "spl_rate": round(spl_sum / count, 6) if count else None,
        "oracle_success_rate": round(os_sum / count, 6) if count else None,
    }


def _overlap_buckets(
    qwen_rows: List[Dict[str, Any]],
    janus_rows: List[Dict[str, Any]],
) -> Dict[str, int]:
    buckets = {
        "both_success": 0,
        "qwen_only": 0,
        "janus_only": 0,
        "both_fail": 0,
    }
    for qwen_row, janus_row in zip(qwen_rows, janus_rows):
        qwen_success = _float(qwen_row.get("success")) > 0.0
        janus_success = _float(janus_row.get("success")) > 0.0
        if qwen_success and janus_success:
            buckets["both_success"] += 1
        elif qwen_success:
            buckets["qwen_only"] += 1
        elif janus_success:
            buckets["janus_only"] += 1
        else:
            buckets["both_fail"] += 1
    return buckets


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-result", type=Path, default=None)
    parser.add_argument("--janus-result", type=Path, default=None)
    parser.add_argument("--qwen-trace", type=Path, default=None)
    parser.add_argument("--ablation-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.ablation_manifest is not None:
        manifest = json.loads(args.ablation_manifest.read_text(encoding="utf-8"))
        raw_arms = manifest.get("arms") if isinstance(manifest, dict) else None
        if not isinstance(raw_arms, dict):
            parser.error("--ablation-manifest must contain an arms object")
        base_dir = args.ablation_manifest.parent
        resolved_arms = {}
        for name, spec in raw_arms.items():
            if not isinstance(spec, dict):
                parser.error(f"ablation arm {name} must be an object")
            resolved_arms[name] = {
                key: (base_dir / Path(spec[key])).resolve()
                for key in ("result", "trace")
            }
        report = compare_ablation_2x2(resolved_arms)
    else:
        if args.qwen_result is None or args.janus_result is None:
            parser.error("--qwen-result and --janus-result are required without --ablation-manifest")
        report = compare_result_files(
            args.qwen_result,
            args.janus_result,
            qwen_trace_path=args.qwen_trace,
        )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if report.get("trace_preflight") and not report["trace_preflight"]["valid"]:
        return 2
    if args.ablation_manifest is not None and not report.get("valid"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
