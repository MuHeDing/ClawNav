#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _increment(counts: Dict[str, int], key: Any) -> None:
    text = str(key or "")
    if not text:
        return
    counts[text] = counts.get(text, 0) + 1


def _is_useful_recall(event: Dict[str, Any]) -> bool:
    return bool(
        event.get("used_by_planner")
        or event.get("used_by_policy")
        or event.get("used_by_critic")
        or event.get("action_changed_after_recall")
        or event.get("stop_blocked_after_recall")
        or event.get("replan_created_after_recall")
    )


def _is_duplicate_skip(write: Dict[str, Any]) -> bool:
    write_gate = write.get("write_gate") or {}
    skip_reason = str(write.get("skip_reason") or "").lower()
    return bool(
        "duplicate" in skip_reason
        or write_gate.get("duplicate_of_memory_id")
        or "duplicate" in str(write_gate.get("curator_reason") or "").lower()
    )


def _result_step_stats(path: Path) -> Dict[str, Any]:
    result_path = path / "result.json"
    if not result_path.exists():
        return {
            "result_episodes": 0,
            "min_episode_steps": 0,
            "max_episode_steps": 0,
            "all_episodes_one_step": False,
            "invalid_run_reason": "",
        }

    steps: List[int] = []
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        value = record.get("steps")
        if isinstance(value, int):
            steps.append(value)

    all_one_step = bool(steps) and all(step == 1 for step in steps)
    return {
        "result_episodes": len(steps),
        "min_episode_steps": min(steps) if steps else 0,
        "max_episode_steps": max(steps) if steps else 0,
        "all_episodes_one_step": all_one_step,
        "invalid_run_reason": "all_episodes_ended_after_one_step" if all_one_step else "",
    }


def summarize_run(path: Path) -> Dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    result_step_stats = _result_step_stats(path)
    trace_path = path / "harness_traces" / "harness_trace_rank0.jsonl"
    trace_steps = 0
    memory_recall_steps = 0
    oracle_leakage_steps = 0
    visual_analysis_steps = 0
    visual_analysis_failures = 0
    visual_analysis_latencies: List[float] = []
    write_novelty_scores: List[float] = []
    memory_write_steps = 0
    memory_write_attempts = 0
    memory_write_written = 0
    memory_write_skipped = 0
    duplicate_skip_count = 0
    write_gate_decisions: Dict[str, int] = {}
    memory_scope_counts: Dict[str, int] = {}
    memory_namespace_counts: Dict[str, int] = {}
    memory_recall_events = 0
    recall_events_with_hits = 0
    useful_recall_events = 0
    planner_use_events = 0
    policy_use_events = 0
    critic_use_events = 0
    action_changed_after_recall_steps = 0
    action_changed_after_recall_events = 0
    planner_intent_changed_after_recall_events = 0
    episode_namespace_hits = 0
    scene_namespace_hits = 0
    task_namespace_hits = 0

    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trace_steps += 1
            record = json.loads(line)
            if record.get("planned_intent") == "recall_memory" or record.get("intent") == "recall_memory":
                memory_recall_steps += 1
            if record.get("oracle_metrics_used_for_decision"):
                oracle_leakage_steps += 1
            visual_analysis = record.get("visual_analysis") or {}
            if visual_analysis.get("ran"):
                visual_analysis_steps += 1
                if visual_analysis.get("error"):
                    visual_analysis_failures += 1
                latency_ms = visual_analysis.get("vlm_latency_ms")
                if latency_ms is None:
                    latency_ms = visual_analysis.get("latency_ms")
                if isinstance(latency_ms, (int, float)):
                    visual_analysis_latencies.append(float(latency_ms))
            memory_writes = record.get("memory_writes") or []
            if memory_writes:
                memory_write_steps += 1
            for write in memory_writes:
                if not isinstance(write, dict):
                    continue
                memory_write_attempts += 1
                if write.get("written"):
                    memory_write_written += 1
                if write.get("skipped"):
                    memory_write_skipped += 1
                if _is_duplicate_skip(write):
                    duplicate_skip_count += 1
                write_gate = write.get("write_gate") or {}
                novelty_score = write_gate.get("novelty_score")
                if isinstance(novelty_score, (int, float)):
                    write_novelty_scores.append(float(novelty_score))
                _increment(write_gate_decisions, write_gate.get("curator_decision"))
                _increment(memory_scope_counts, write.get("memory_scope"))
                _increment(memory_namespace_counts, write.get("memory_namespace"))
            for event in record.get("recall_usage") or []:
                if not isinstance(event, dict):
                    continue
                memory_recall_events += 1
                if int(event.get("num_hits") or 0) > 0:
                    recall_events_with_hits += 1
                if _is_useful_recall(event):
                    useful_recall_events += 1
                if event.get("used_by_planner"):
                    planner_use_events += 1
                if event.get("used_by_policy"):
                    policy_use_events += 1
                if event.get("used_by_critic"):
                    critic_use_events += 1
                if event.get("action_changed_after_recall"):
                    action_changed_after_recall_events += 1
                    action_changed_after_recall_steps += 1
                if (
                    event.get("planner_intent_before_recall")
                    and event.get("planner_intent_after_recall")
                    and event.get("planner_intent_before_recall")
                    != event.get("planner_intent_after_recall")
                ):
                    planner_intent_changed_after_recall_events += 1
                namespace = str(event.get("selected_namespace") or "")
                if namespace.startswith("episode:"):
                    episode_namespace_hits += 1
                elif namespace.startswith("scene:"):
                    scene_namespace_hits += 1
                elif namespace.startswith("task:"):
                    task_namespace_hits += 1

    return {
        "success": summary.get("sucs_all", 0.0),
        "spl": summary.get("spls_all", 0.0),
        "length": summary.get("length", 0),
        **result_step_stats,
        "trace_steps": trace_steps,
        "memory_recall_steps": memory_recall_steps,
        "oracle_leakage_steps": oracle_leakage_steps,
        "visual_analysis_steps": visual_analysis_steps,
        "visual_analysis_failures": visual_analysis_failures,
        "visual_analysis_latency_ms_avg": _mean(visual_analysis_latencies),
        "avg_vlm_latency_ms": _mean(visual_analysis_latencies),
        "memory_write_steps": memory_write_steps,
        "memory_write_attempts": memory_write_attempts,
        "memory_write_written": memory_write_written,
        "memory_write_skipped": memory_write_skipped,
        "duplicate_skip_count": duplicate_skip_count,
        "avg_write_novelty_score": _mean(write_novelty_scores),
        "write_acceptance_rate": _ratio(memory_write_written, memory_write_attempts),
        "memory_pollution_rate": _ratio(memory_write_skipped, memory_write_attempts),
        "write_gate_decisions": write_gate_decisions,
        "memory_scope_counts": memory_scope_counts,
        "memory_namespace_counts": memory_namespace_counts,
        "memory_recall_events": memory_recall_events,
        "recall_events_with_hits": recall_events_with_hits,
        "useful_recall_events": useful_recall_events,
        "useful_recall_rate": _ratio(useful_recall_events, memory_recall_events),
        "planner_use_events": planner_use_events,
        "policy_use_events": policy_use_events,
        "critic_use_events": critic_use_events,
        "action_changed_after_recall_steps": action_changed_after_recall_steps,
        "action_changed_after_recall_events": action_changed_after_recall_events,
        "planner_intent_changed_after_recall_events": planner_intent_changed_after_recall_events,
        "episode_namespace_hits": episode_namespace_hits,
        "scene_namespace_hits": scene_namespace_hits,
        "task_namespace_hits": task_namespace_hits,
    }


def format_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "run",
        "success",
        "spl",
        "visual_analysis_steps",
        "memory_write_attempts",
        "write_acceptance_rate",
        "memory_recall_events",
        "useful_recall_rate",
        "action_changed_after_recall_events",
        "oracle_leakage_steps",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--format", choices=("jsonl", "markdown"), default="jsonl")
    args = parser.parse_args()

    rows = [{"run": run_dir, **summarize_run(Path(run_dir))} for run_dir in args.run_dirs]
    if args.format == "markdown":
        print(format_markdown(rows))
        return
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
