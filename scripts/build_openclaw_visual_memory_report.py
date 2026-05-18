#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.summarize_openclaw_vln_ablation import _format_cell, format_markdown, summarize_run


def build_report(run_dirs: List[Path], require_visual: bool = False) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        row = _summarize_or_error(run_dir)
        runs.append(row)

    baseline = runs[0] if runs else {}
    baseline_success = float(baseline.get("success") or 0.0)
    baseline_spl = float(baseline.get("spl") or 0.0)
    for row in runs:
        row["success_delta_vs_baseline"] = float(row.get("success") or 0.0) - baseline_success
        row["spl_delta_vs_baseline"] = float(row.get("spl") or 0.0) - baseline_spl

    checks = _build_checks(runs, require_visual=require_visual)
    status = "pass" if all(check["passed"] for check in checks.values()) else "fail"
    return {
        "status": status,
        "require_visual": require_visual,
        "checks": checks,
        "runs": runs,
    }


def _summarize_or_error(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    trace_path = run_dir / "harness_traces" / "harness_trace_rank0.jsonl"
    row: Dict[str, Any] = {
        "run": str(run_dir),
        "run_name": run_dir.name,
        "summary_exists": summary_path.exists(),
        "trace_exists": trace_path.exists(),
    }
    if not summary_path.exists():
        row["error"] = "missing summary.json"
        return row
    try:
        row.update(summarize_run(run_dir))
    except Exception as exc:
        row["error"] = str(exc)
    return row


def _build_checks(runs: List[Dict[str, Any]], require_visual: bool) -> Dict[str, Dict[str, Any]]:
    checks = {
        "runs_present": _check(bool(runs), f"{len(runs)} run(s) supplied"),
        "summaries_present": _check(
            all(run.get("summary_exists") for run in runs),
            _failed_runs(runs, "summary_exists"),
        ),
        "traces_present": _check(
            all(run.get("trace_exists") for run in runs),
            _failed_runs(runs, "trace_exists"),
        ),
        "no_oracle_leakage": _check(
            sum(int(run.get("oracle_leakage_steps") or 0) for run in runs) == 0,
            "oracle_leakage_steps must be 0 for all runs",
        ),
    }
    if require_visual:
        checks.update(
            {
                "visual_analysis_present": _check(
                    any(int(run.get("visual_analysis_steps") or 0) > 0 for run in runs),
                    "at least one run must contain visual_analysis_steps > 0",
                ),
                "visual_memory_written": _check(
                    any(int(run.get("memory_write_written") or 0) > 0 for run in runs),
                    "at least one run must contain accepted visual memory writes",
                ),
                "useful_recall_present": _check(
                    any(int(run.get("useful_recall_events") or 0) > 0 for run in runs),
                    "at least one run must contain useful recall events",
                ),
            }
        )
    return checks


def _check(passed: bool, detail: str) -> Dict[str, Any]:
    return {"passed": bool(passed), "detail": detail}


def _failed_runs(runs: List[Dict[str, Any]], key: str) -> str:
    failed = [run.get("run_name") or run.get("run") for run in runs if not run.get(key)]
    if not failed:
        return "ok"
    return "missing for: " + ", ".join(str(item) for item in failed)


def format_report_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# OpenClaw Visual Memory Evidence Report",
        "",
        f"**Status:** {report.get('status', 'fail')}",
        "",
        "## Checks",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for name, check in report.get("checks", {}).items():
        status = "pass" if check.get("passed") else "fail"
        lines.append(f"| {name} | {status} | {check.get('detail', '')} |")

    runs = report.get("runs") or []
    if runs:
        lines.extend(
            [
                "",
                "## Metrics",
                "",
                _format_report_metrics(runs),
            ]
        )
    return "\n".join(lines)


def _format_report_metrics(runs: List[Dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        row = dict(run)
        row["run"] = run.get("run_name") or run.get("run")
        rows.append(row)
    table = format_markdown(rows)
    extra_headers = ["run", "success_delta_vs_baseline", "spl_delta_vs_baseline"]
    extra_lines = [
        "",
        "| " + " | ".join(extra_headers) + " |",
        "| " + " | ".join("---" for _ in extra_headers) + " |",
    ]
    for row in rows:
        extra_lines.append(
            "| " + " | ".join(_format_cell(row.get(header)) for header in extra_headers) + " |"
        )
    return table + "\n" + "\n".join(extra_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--require_visual", action="store_true")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = build_report([Path(path) for path in args.run_dirs], require_visual=args.require_visual)
    if args.format == "json":
        text = json.dumps(report, indent=2, sort_keys=True)
    else:
        text = format_report_markdown(report)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
