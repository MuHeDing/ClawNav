#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def check_runtime_smoke(
    run_dir: Path | str,
    *,
    mode: str = "controlled_seed",
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    rows = _load_trace_rows(run_path)
    tool_counts: Counter[str] = Counter()
    read_statuses: Counter[str] = Counter()
    trigger_sources: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()

    completed_readback_count = 0
    completed_with_current_plus_memory = 0
    completed_with_matched_memory_ids = 0
    completed_used_by_policy_count = 0
    recall_hit_rows = 0
    fallback_rows = 0
    runtime_error_rows = 0
    planner_fallback_rows = 0
    visual_readback_config: Dict[str, Any] = {}

    for row in rows:
        runtime = _runtime_block(row)
        for call in _tool_calls(row):
            tool_name = str(call.get("tool_name") or "")
            if tool_name:
                tool_counts[tool_name] += 1

        visual_readback = _visual_readback_block(row)
        if visual_readback:
            read_status = str(visual_readback.get("read_status") or "")
            trigger_source = str(visual_readback.get("trigger_source") or "")
            skip_reason = str(visual_readback.get("skip_reason") or "")
            if read_status:
                read_statuses[read_status] += 1
            if trigger_source:
                trigger_sources[trigger_source] += 1
            if skip_reason:
                skip_reasons[skip_reason] += 1
            if read_status == "completed":
                completed_readback_count += 1
                if len(visual_readback.get("actually_read_image_paths") or []) >= 2:
                    completed_with_current_plus_memory += 1
                if visual_readback.get("matched_memory_ids"):
                    completed_with_matched_memory_ids += 1
                if (
                    visual_readback.get("used_by_policy")
                    or visual_readback.get("final_policy_payload_has_memory")
                ):
                    completed_used_by_policy_count += 1

        config = row.get("visual_readback_config") or runtime.get("visual_readback_config")
        if isinstance(config, dict) and config:
            visual_readback_config = dict(config)

        if row.get("fallback") or runtime.get("fallback"):
            fallback_rows += 1
        if row.get("planner_fallback") or runtime.get("planner_fallback"):
            planner_fallback_rows += 1
        if _runtime_error(row):
            runtime_error_rows += 1
        if _recall_has_image_hit(row):
            recall_hit_rows += 1

    natural_image_hit_ready = (
        tool_counts.get("MemoryQuerySkill", 0) > 0
        and completed_with_current_plus_memory > 0
        and completed_with_matched_memory_ids > 0
    )
    summary = {
        "run_dir": str(run_path),
        "mode": mode,
        "trace_rows": len(rows),
        "tool_counts": dict(tool_counts),
        "read_statuses": dict(read_statuses),
        "trigger_sources": dict(trigger_sources),
        "skip_reasons": dict(skip_reasons),
        "completed_readback_count": completed_readback_count,
        "completed_with_current_plus_memory": completed_with_current_plus_memory,
        "completed_with_matched_memory_ids": completed_with_matched_memory_ids,
        "completed_used_by_policy_count": completed_used_by_policy_count,
        "recall_hit_rows": recall_hit_rows,
        "fallback_rows": fallback_rows,
        "runtime_error_rows": runtime_error_rows,
        "planner_fallback_rows": planner_fallback_rows,
        "natural_image_hit_ready": natural_image_hit_ready,
        "diagnostic_status": _diagnostic_status(
            natural_image_hit_ready=natural_image_hit_ready,
            skip_reasons=skip_reasons,
            tool_counts=tool_counts,
            completed_readback_count=completed_readback_count,
            completed_with_matched_memory_ids=completed_with_matched_memory_ids,
        ),
        "visual_readback_config": visual_readback_config,
        "issues": [],
        "passed": False,
    }
    issues = _issues(summary, mode=mode)
    summary["issues"] = issues
    summary["passed"] = not issues
    return summary


def _load_trace_rows(run_path: Path) -> List[Dict[str, Any]]:
    trace_path = _trace_path(run_path)
    rows = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _trace_path(run_path: Path) -> Path:
    candidate = run_path / "harness_traces" / "harness_trace_rank0.jsonl"
    if candidate.exists():
        return candidate
    trace_dir = run_path / "harness_traces"
    matches = sorted(trace_dir.glob("harness_trace_rank*.jsonl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"missing harness trace under {trace_dir}")


def _runtime_block(row: Dict[str, Any]) -> Dict[str, Any]:
    runtime = row.get("runtime")
    return runtime if isinstance(runtime, dict) else {}


def _tool_calls(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    runtime = _runtime_block(row)
    calls = row.get("tool_calls")
    if not isinstance(calls, list):
        calls = runtime.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _visual_readback_block(row: Dict[str, Any]) -> Dict[str, Any]:
    runtime = _runtime_block(row)
    visual_readback = row.get("visual_readback")
    if not isinstance(visual_readback, dict):
        visual_readback = runtime.get("visual_readback")
    return visual_readback if isinstance(visual_readback, dict) else {}


def _runtime_error(row: Dict[str, Any]) -> bool:
    runtime = _runtime_block(row)
    status = str(row.get("runtime_status") or runtime.get("runtime_status") or "")
    error_type = str(row.get("error_type") or runtime.get("error_type") or "")
    error = str(row.get("error") or runtime.get("error") or "")
    return status == "failed" or bool(error_type) or bool(error)


def _recall_has_image_hit(row: Dict[str, Any]) -> bool:
    recall_usage = row.get("recall_usage") or _runtime_block(row).get("recall_usage") or []
    if not isinstance(recall_usage, list):
        return False
    for event in recall_usage:
        if not isinstance(event, dict):
            continue
        if int(event.get("num_hits") or 0) <= 0:
            continue
        if event.get("hit_image_paths"):
            return True
    return False


def _diagnostic_status(
    *,
    natural_image_hit_ready: bool,
    skip_reasons: Counter[str],
    tool_counts: Counter[str],
    completed_readback_count: int,
    completed_with_matched_memory_ids: int,
) -> str:
    if natural_image_hit_ready:
        return "natural_image_hit"
    if skip_reasons.get("no_memory_hit", 0) > 0:
        return "no_memory_hit"
    if tool_counts.get("MemoryQuerySkill", 0) == 0:
        return "no_memory_query"
    if completed_readback_count > 0 and completed_with_matched_memory_ids == 0:
        return "readback_without_matched_memory"
    if skip_reasons.get("no_trigger", 0) > 0:
        return "no_trigger"
    return "insufficient_evidence"


def _issues(summary: Dict[str, Any], *, mode: str) -> List[str]:
    issues: List[str] = []
    if summary["trace_rows"] <= 0:
        issues.append("trace_rows must be non-empty")
    if summary["fallback_rows"] > 0:
        issues.append("fallback_rows must be zero")
    if summary["runtime_error_rows"] > 0:
        issues.append("runtime_error_rows must be zero")
    if summary["planner_fallback_rows"] > 0:
        issues.append("planner_fallback_rows must be zero")

    config = summary.get("visual_readback_config") or {}
    seed_enabled = bool(config.get("visual_readback_smoke_seed_memory"))
    if mode == "controlled_seed":
        if not seed_enabled:
            issues.append("controlled_seed mode requires visual_readback_smoke_seed_memory=true")
        if summary["trigger_sources"].get("controlled_smoke_seed", 0) <= 0:
            issues.append("controlled_seed mode requires trigger_source=controlled_smoke_seed")
        if summary["tool_counts"].get("MemoryQuerySkill", 0) <= 0:
            issues.append("MemoryQuerySkill must appear")
        if summary["tool_counts"].get("VisualMemoryReadSkill", 0) <= 0:
            issues.append("VisualMemoryReadSkill must appear")
        if summary["recall_hit_rows"] <= 0:
            issues.append("MemoryQuerySkill must return a non-empty image hit")
        if summary["completed_readback_count"] <= 0:
            issues.append("visual_readback.read_status=completed must appear")
        if summary["completed_with_current_plus_memory"] <= 0:
            issues.append("actually_read_image_paths must include current + memory")
        if summary["completed_with_matched_memory_ids"] <= 0:
            issues.append("matched_memory_ids must be non-empty")
        if summary["completed_used_by_policy_count"] > 0:
            issues.append("completed readback rows must keep used_by_policy=false")
    elif mode == "no_seed_diagnostic":
        if seed_enabled:
            issues.append("no_seed_diagnostic requires visual_readback_smoke_seed_memory=false")
        if summary["trigger_sources"].get("controlled_smoke_seed", 0) > 0:
            issues.append("no_seed_diagnostic must not use controlled_smoke_seed")
    else:
        issues.append(f"unknown mode: {mode}")
    return issues


def _format_text(summary: Dict[str, Any]) -> str:
    lines = [
        f"run_dir={summary['run_dir']}",
        f"mode={summary['mode']}",
        f"passed={summary['passed']}",
        f"diagnostic_status={summary['diagnostic_status']}",
        f"trace_rows={summary['trace_rows']}",
        f"tool_counts={summary['tool_counts']}",
        f"read_statuses={summary['read_statuses']}",
        f"trigger_sources={summary['trigger_sources']}",
        f"skip_reasons={summary['skip_reasons']}",
        f"completed_with_current_plus_memory={summary['completed_with_current_plus_memory']}",
        f"completed_with_matched_memory_ids={summary['completed_with_matched_memory_ids']}",
        f"natural_image_hit_ready={summary['natural_image_hit_ready']}",
    ]
    if summary["issues"]:
        lines.append("issues:")
        lines.extend(f"- {issue}" for issue in summary["issues"])
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument(
        "--mode",
        choices=("controlled_seed", "no_seed_diagnostic"),
        default="controlled_seed",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = check_runtime_smoke(Path(args.run_dir), mode=args.mode)
    text = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        if args.format == "json"
        else _format_text(summary)
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(text)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
