#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict


def summarize_run(path: Path) -> Dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    trace_path = path / "harness_traces" / "harness_trace_rank0.jsonl"
    trace_steps = 0
    memory_recall_steps = 0
    oracle_leakage_steps = 0

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

    return {
        "success": summary.get("sucs_all", 0.0),
        "spl": summary.get("spls_all", 0.0),
        "length": summary.get("length", 0),
        "trace_steps": trace_steps,
        "memory_recall_steps": memory_recall_steps,
        "oracle_leakage_steps": oracle_leakage_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        print(json.dumps({"run": run_dir, **summarize_run(Path(run_dir))}, sort_keys=True))


if __name__ == "__main__":
    main()
