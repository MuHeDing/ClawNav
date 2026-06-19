#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List, Optional

from harness.visual_readback.metrics import (
    build_phase_a_gate_report,
    format_phase_a_gate_markdown,
    format_visual_readback_markdown,
    summarize_visual_readback_run,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default="")
    parser.add_argument("--event_gated_phase0_manifest", default="")
    parser.add_argument("--expected_manifest_sha256", default="")
    parser.add_argument("--expected_phase0_thresholds_json", default="")
    parser.add_argument(
        "--phase_a_summary",
        action="append",
        default=[],
        help="Phase A arm summary as arm_name=/path/to/summary.json",
    )
    args = parser.parse_args(argv)

    if args.phase_a_summary:
        summary = build_phase_a_gate_report(_load_phase_a_summaries(args.phase_a_summary))
    else:
        if not args.run_dir:
            parser.error("run_dir is required unless --phase_a_summary is provided")
        expected_thresholds = None
        if args.expected_phase0_thresholds_json:
            expected_thresholds = json.loads(
                Path(args.expected_phase0_thresholds_json).read_text(encoding="utf-8")
            )
        summary = summarize_visual_readback_run(
            Path(args.run_dir),
            phase0_manifest_path=Path(args.event_gated_phase0_manifest)
            if args.event_gated_phase0_manifest
            else None,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_phase0_thresholds=expected_thresholds,
        )
    if args.format == "json":
        text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    elif args.phase_a_summary:
        text = format_phase_a_gate_markdown(summary)
    else:
        text = format_visual_readback_markdown(summary)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def _load_phase_a_summaries(specs: List[str]) -> dict:
    summaries = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--phase_a_summary must use arm_name=/path/to/summary.json")
        name, path = spec.split("=", 1)
        if not name:
            raise ValueError("--phase_a_summary arm_name cannot be empty")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"phase A summary must be a JSON object: {path}")
        summaries[name] = data
    return summaries


if __name__ == "__main__":
    raise SystemExit(main())
