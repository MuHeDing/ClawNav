import json
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


def summarize_visual_readback_run(run_dir: Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    loaded_rows = _load_trace_rows(_trace_paths(run_path))
    blocks = [_visual_readback_block(row) for row in loaded_rows]
    summary_json = _load_summary_json(run_path)
    endpoint_report = build_endpoint_report(blocks)
    return {
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
        "actually_read_image_count": sum(len(_list_value(block.get("actually_read_image_paths"))) for block in blocks),
        "matched_memory_count": sum(len(_list_value(block.get("matched_memory_ids"))) for block in blocks),
        "visual_grounded_readback_count": sum(1 for block in blocks if _is_grounded(block)),
        "adjudication_correct_readback_count": sum(1 for block in blocks if _is_adjudication_correct(block)),
        "controller_decision_changed_after_visual_read_count": sum(
            1 for block in blocks if bool(block.get("controller_decision_changed_after_visual_read"))
        ),
        "replan_request_logged_after_visual_read_count": sum(
            1 for block in blocks if bool(block.get("replan_request_logged_after_visual_read"))
        ),
        "grounded_but_wrong_label_count": sum(1 for block in blocks if _is_grounded_but_wrong(block)),
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


def _rate(count: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return count / denominator
