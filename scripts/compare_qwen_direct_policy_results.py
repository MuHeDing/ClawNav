#!/usr/bin/env python
"""Compare qwen_direct_policy results against a Janus baseline on shared keys."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--janus-result", type=Path, required=True)
    parser.add_argument("--qwen-trace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
