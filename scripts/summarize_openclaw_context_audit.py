#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_audit_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        audit = row.get("context_audit")
        if isinstance(audit, dict):
            merged = dict(audit)
            if "step_id" not in merged and "step_id" in row:
                merged["step_id"] = row["step_id"]
            records.append(merged)
        else:
            records.append(row)
    records.sort(key=lambda record: int(record.get("step_id") or 0))
    return records


def summarize_audit_file(path: Path) -> Dict[str, Any]:
    records = load_audit_records(path)
    provider_values = _numeric_values(records, "provider_input_tokens")
    assembled_values = _numeric_values(records, "assembled_prompt_tokens")
    step10 = _record_at_or_after(records, 10) or (records[0] if records else {})
    last = records[-1] if records else {}
    step10_provider = _to_int(step10.get("provider_input_tokens"))
    last_provider = _to_int(last.get("provider_input_tokens"))
    step10_assembled = _to_int(step10.get("assembled_prompt_tokens"))
    last_assembled = _to_int(last.get("assembled_prompt_tokens"))
    provider_growth = (
        last_provider - step10_provider
        if step10_provider is not None and last_provider is not None
        else None
    )
    assembled_growth = (
        last_assembled - step10_assembled
        if step10_assembled is not None and last_assembled is not None
        else None
    )
    session_ids = [
        str(record.get("openclaw_session_id") or "")
        for record in records
        if record.get("openclaw_session_id")
    ]
    checks = _build_checks(records, provider_growth, step10_provider, assembled_growth, step10_assembled)
    status = "pass" if all(check["passed"] for check in checks.values()) else "fail"
    return {
        "status": status,
        "audit_path": str(path),
        "total_steps": len(records),
        "min_step_id": min((int(record.get("step_id") or 0) for record in records), default=0),
        "max_step_id": max((int(record.get("step_id") or 0) for record in records), default=0),
        "unique_session_ids": len(set(session_ids)),
        "step10_provider_input_tokens": step10_provider,
        "last_provider_input_tokens": last_provider,
        "provider_growth_from_step10_to_last": provider_growth,
        "assembled_growth_from_step10_to_last": assembled_growth,
        "max_provider_input_tokens": max(provider_values) if provider_values else None,
        "max_assembled_prompt_tokens": max(assembled_values) if assembled_values else None,
        "checks": checks,
    }


def _build_checks(
    records: List[Dict[str, Any]],
    provider_growth: Optional[int],
    step10_provider: Optional[int],
    assembled_growth: Optional[int],
    step10_assembled: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    session_ids = [
        str(record.get("openclaw_session_id") or "")
        for record in records
        if record.get("openclaw_session_id")
    ]
    stable_threshold = max(1000, int((step10_provider or 0) * 0.2))
    assembled_stable_threshold = max(100, int((step10_assembled or 0) * 0.2))
    base_checks = {
        "enough_steps": _check(
            len(records) >= 100,
            f"{len(records)} audit record(s); strict acceptance requires at least 100",
        ),
        "no_history_tokens": _check(
            all(int(record.get("history_tokens") or 0) == 0 for record in records),
            "history_tokens must be 0 for every step",
        ),
        "step0_marker_not_leaked_after_step0": _check(
            all(
                not record.get("prompt_contains_step0_marker")
                for record in records
                if int(record.get("step_id") or 0) > 0
            ),
            "step0 marker must not appear after step 0",
        ),
        "no_qwen_hard_limit_violation": _check(
            all(not record.get("qwen_hard_limit_exceeded") for record in records),
            "qwen_hard_limit_exceeded must be false for every step",
        ),
    }
    if records and all(record.get("openclaw_session_mode") == "stateless_model" for record in records):
        model_checks = {
            "provider_usage_optional": _check(
                all(
                    record.get("provider_input_tokens") is None
                    or record.get("provider_usage_source") == "reported"
                    for record in records
                ),
                "model.run may omit provider usage; provider_input_tokens is optional in stateless_model mode",
            ),
            "assembled_prompt_stable": _check(
                assembled_growth is not None and assembled_growth <= assembled_stable_threshold,
                f"growth={assembled_growth} threshold={assembled_stable_threshold}",
            ),
            "stateless_model": _check(
                True,
                "openclaw_session_mode is stateless_model for every step",
            ),
            "no_session_ids": _check(
                len(session_ids) == 0,
                f"non_empty_session_ids={len(session_ids)}",
            ),
        }
        return {**base_checks, **model_checks}
    agent_checks = {
        "provider_usage_present": _check(
            all(record.get("provider_input_tokens") is not None for record in records),
            "provider_input_tokens must be present for every step",
        ),
        "provider_tokens_stable": _check(
            provider_growth is not None and provider_growth <= stable_threshold,
            f"growth={provider_growth} threshold={stable_threshold}",
        ),
        "fresh_per_step": _check(
            all(record.get("openclaw_session_mode") == "fresh_per_step" for record in records),
            "openclaw_session_mode must be fresh_per_step for every step",
        ),
        "session_ids_unique": _check(
            len(session_ids) == len(records) and len(set(session_ids)) == len(records),
            f"unique={len(set(session_ids))} total={len(records)}",
        ),
    }
    return {**base_checks, **agent_checks}


def _record_at_or_after(
    records: List[Dict[str, Any]],
    step_id: int,
) -> Optional[Dict[str, Any]]:
    for record in records:
        if int(record.get("step_id") or 0) >= step_id:
            return record
    return None


def _numeric_values(records: Iterable[Dict[str, Any]], key: str) -> List[int]:
    values: List[int] = []
    for record in records:
        value = _to_int(record.get(key))
        if value is not None:
            values.append(value)
    return values


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check(passed: bool, detail: str) -> Dict[str, Any]:
    return {"passed": bool(passed), "detail": detail}


def format_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# OpenClaw Context Audit Report",
        "",
        f"**Status:** {summary.get('status', 'fail')}",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        "total_steps",
        "min_step_id",
        "max_step_id",
        "unique_session_ids",
        "step10_provider_input_tokens",
        "last_provider_input_tokens",
        "provider_growth_from_step10_to_last",
        "assembled_growth_from_step10_to_last",
        "max_provider_input_tokens",
        "max_assembled_prompt_tokens",
    ):
        lines.append(f"| {key} | {summary.get(key)} |")

    lines.extend(["", "## Checks", "", "| check | status | detail |", "| --- | --- | --- |"])
    for name, check in (summary.get("checks") or {}).items():
        status = "pass" if check.get("passed") else "fail"
        lines.append(f"| {name} | {status} | {check.get('detail', '')} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_path")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    summary = summarize_audit_file(Path(args.audit_path))
    if args.format == "json":
        text = json.dumps(summary, indent=2, sort_keys=True)
    else:
        text = format_markdown(summary)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
