import json
from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from harness.visual_readback.manifest import (
    EventGatedPhase0Manifest,
    load_event_gated_phase0_manifest_jsonl,
)


PRIMARY_TRIGGER_SOURCES = {"online_controller", "static_pre_run"}
OFFLINE_TRIGGER_SOURCES = {"offline_fixture"}
DEFAULT_INTERVAL_STEP = 10


def build_phase0_audit_report(
    trace_paths: Iterable[Path],
    manifest_path: Optional[Path] = None,
    interval_step: int = DEFAULT_INTERVAL_STEP,
) -> Dict[str, Any]:
    manifest = (
        load_event_gated_phase0_manifest_jsonl(Path(manifest_path))
        if manifest_path
        else None
    )
    episodes = set()
    trigger_distribution: Counter[str] = Counter()
    candidate_case_set: List[Dict[str, Any]] = []
    offline_stress_set: List[Dict[str, Any]] = []
    oracle_rejected: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    parent_online_trigger_count = 0
    parent_static_pre_run_count = 0

    for trace_path in trace_paths:
        path = Path(trace_path)
        for row_index, row in enumerate(_read_jsonl(path)):
            scene_id = str(row.get("scene_id") or "")
            episode_id = str(row.get("episode_id") or "")
            if scene_id or episode_id:
                episodes.add((scene_id, episode_id))

            block = _visual_readback_block(row)
            blocks.append(block)
            trigger_source = str(block.get("trigger_source") or "")
            trigger_rule = str(block.get("trigger_rule") or "")
            oracle_fields = _oracle_fields(block)
            if trigger_rule:
                trigger_distribution[trigger_rule] += 1
            if trigger_source == "online_controller":
                parent_online_trigger_count += 1
            if trigger_source == "static_pre_run":
                parent_static_pre_run_count += 1

            case = {
                "trace_path": str(path),
                "row_index": row_index,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "trigger_source": trigger_source,
                "trigger_rule": trigger_rule,
                "oracle_selection_fields": oracle_fields,
            }
            if trigger_source in OFFLINE_TRIGGER_SOURCES:
                offline_stress_set.append(case)
                continue
            if oracle_fields:
                oracle_rejected.append(case)
                continue
            if trigger_source in PRIMARY_TRIGGER_SOURCES:
                candidate_case_set.append(case)

    status = _gate_status(candidate_case_set)
    report = {
        "phase0_status": status,
        "parent_episode_count": len(episodes),
        "parent_online_trigger_count": parent_online_trigger_count,
        "parent_static_pre_run_count": parent_static_pre_run_count,
        "parent_eligible_trigger_count": len(candidate_case_set),
        "primary_quantitative_set_count": len(candidate_case_set),
        "candidate_case_set_count": len(candidate_case_set),
        "offline_stress_set_count": len(offline_stress_set),
        "oracle_rejected_count": len(oracle_rejected),
        "trigger_type_distribution": dict(sorted(trigger_distribution.items())),
        "candidate_case_set": candidate_case_set,
        "offline_stress_set": offline_stress_set,
        "oracle_rejected_cases": oracle_rejected,
    }
    if manifest is not None:
        report.update(
            _manifest_audit_fields(
                manifest=manifest,
                blocks=blocks,
                interval_step=interval_step,
            )
        )
    return report


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _visual_readback_block(row: Dict[str, Any]) -> Dict[str, Any]:
    block = row.get("visual_readback") or row.get("visual_memory_read") or {}
    if isinstance(block, dict):
        merged = dict(row)
        merged.update(block)
        return merged
    return dict(row)


def _oracle_fields(block: Dict[str, Any]) -> List[str]:
    fields = block.get("oracle_selection_fields")
    if fields is None:
        fields = block.get("oracle_fields_used", [])
    if isinstance(fields, str):
        return [fields] if fields else []
    if isinstance(fields, list):
        return [str(field) for field in fields if str(field)]
    return [str(fields)] if fields else []


def _gate_status(candidate_case_set: List[Dict[str, Any]]) -> str:
    scene_count = len({case["scene_id"] for case in candidate_case_set if case["scene_id"]})
    trigger_counts = Counter(case["trigger_rule"] for case in candidate_case_set)
    trigger_type_count = len([rule for rule, count in trigger_counts.items() if rule and count])
    risky_stop_count = trigger_counts.get("risky_stop", 0)
    decision_point_count = trigger_counts.get("decision_point", 0)
    if (
        len(candidate_case_set) >= 30
        and scene_count >= 3
        and trigger_type_count >= 2
        and (risky_stop_count >= 10 or decision_point_count >= 10)
    ):
        return "candidate_set_ready"
    if risky_stop_count >= 10:
        return "downscope_stop_only"
    if decision_point_count >= 10:
        return "downscope_turn_only"
    return "no_go"


def _manifest_audit_fields(
    manifest: EventGatedPhase0Manifest,
    blocks: List[Dict[str, Any]],
    interval_step: int,
) -> Dict[str, Any]:
    thresholds = dict(manifest.rows[0].get("phase0_stop_go_thresholds") or {})
    route_metrics = _route_event_metrics(manifest, interval_step)
    attachment_metrics = _attachment_metrics(blocks, interval_step=interval_step)
    trigger_metrics = _trigger_metrics(manifest, blocks)
    evidence_metrics = _legacy_evidence_metrics(blocks)
    threshold_results = _threshold_results(
        thresholds,
        {
            **route_metrics,
            **attachment_metrics,
            **evidence_metrics,
        },
    )
    comparison_valid = all(item["passed"] for item in threshold_results.values())
    decision = "no_go_interval_cleanup" if comparison_valid else "go_event_gated_smoke"
    return {
        "manifest_schema_version": manifest.manifest_schema_version,
        "manifest_sha256": manifest.manifest_sha256,
        "phase0_stop_go_thresholds": thresholds,
        "phase0_threshold_results": threshold_results,
        "phase0_comparison_valid": True,
        "phase0_stop_go_decision": decision,
        **route_metrics,
        **attachment_metrics,
        **trigger_metrics,
        **evidence_metrics,
    }


def _route_event_metrics(
    manifest: EventGatedPhase0Manifest,
    interval_step: int,
) -> Dict[str, Any]:
    total = 0
    missed = 0
    missed_events: List[Dict[str, Any]] = []
    for row in manifest.rows:
        interval_steps = set(range(0, int(row.get("max_steps") or 0) + 1, interval_step))
        tolerance = int(row.get("accepted_step_tolerance") or 0)
        for label in row.get("route_event_labels") or []:
            total += 1
            step_id = int(label.get("step_id"))
            matched = any(abs(step_id - saved_step) <= tolerance for saved_step in interval_steps)
            if not matched:
                missed += 1
                missed_events.append(
                    {
                        "scene_id": row.get("scene_id"),
                        "episode_id": row.get("episode_id"),
                        "event_type": label.get("event_type"),
                        "step_id": step_id,
                    }
                )
    return {
        "route_event_label_count": total,
        "missed_route_event_count": missed,
        "route_event_miss_rate": _rate(missed, total),
        "missed_route_events": missed_events,
    }


def _attachment_metrics(blocks: List[Dict[str, Any]], interval_step: int) -> Dict[str, Any]:
    readback_blocks = [block for block in blocks if _has_readback_attempt(block)]
    duplicate_drops = 0
    retrieved_total = 0
    deduped_total = 0
    adjacent_triplet_count = 0
    adjacent_denominator = 0
    for block in readback_blocks:
        paths = [str(path) for path in _list_value(block.get("retrieved_image_paths")) if str(path)]
        unique_paths = list(dict.fromkeys(paths))
        retrieved_total += len(paths)
        deduped_total += len(unique_paths)
        duplicate_drops += len(paths) - len(unique_paths)
        steps = [_step_from_path(path) for path in unique_paths]
        step_values = [step for step in steps if step is not None]
        if len(step_values) >= 3:
            adjacent_denominator += 1
            if _contains_adjacent_triplet(step_values, expected_gap=interval_step):
                adjacent_triplet_count += 1
    return {
        "readback_attempt_count": len(readback_blocks),
        "retrieved_image_count": retrieved_total,
        "deduped_attached_image_count": deduped_total,
        "attached_image_count": deduped_total,
        "exact_duplicate_drop_count": duplicate_drops,
        "exact_duplicate_rate": _rate(duplicate_drops, retrieved_total),
        "adjacent_triplet_count": adjacent_triplet_count,
        "adjacent_triplet_denominator": adjacent_denominator,
        "adjacent_triplet_rate": _rate(adjacent_triplet_count, adjacent_denominator),
    }


def _trigger_metrics(
    manifest: EventGatedPhase0Manifest,
    blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    expected = set()
    for row in manifest.rows:
        expected.update(str(key) for key in row.get("shared_readback_trigger_keys") or [])
    observed = {_trigger_key(block) for block in blocks if _trigger_key(block)}
    missing = sorted(expected.difference(observed))
    extra = sorted(observed.difference(expected))
    return {
        "shared_trigger_key_count": len(expected),
        "observed_trigger_key_count": len(observed),
        "missing_shared_trigger_key_count": len(missing),
        "extra_trigger_key_count": len(extra),
        "unmatched_trigger_count": len(missing) + len(extra),
        "missing_shared_trigger_keys": missing,
        "extra_trigger_keys": extra,
    }


def _legacy_evidence_metrics(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [block for block in blocks if block.get("read_status") == "completed"]
    stable = 0
    ambiguous = 0
    for block in completed:
        if (
            _list_value(block.get("matched_memory_ids"))
            or _list_value(block.get("attached_memory_ids"))
            or _list_value(block.get("retrieved_image_paths"))
        ):
            stable += 1
        else:
            ambiguous += 1
    return {
        "legacy_evidence_row_count": len(completed),
        "legacy_stable_evidence_count": stable,
        "legacy_ambiguous_evidence_count": ambiguous,
        "ambiguous_evidence_rate": _rate(ambiguous, len(completed)),
    }


def _threshold_results(
    thresholds: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    mapping = {
        "route_event_miss_rate_max": "route_event_miss_rate",
        "exact_duplicate_rate_max": "exact_duplicate_rate",
        "adjacent_triplet_rate_max": "adjacent_triplet_rate",
        "ambiguous_evidence_rate_max": "ambiguous_evidence_rate",
    }
    results: Dict[str, Dict[str, Any]] = {}
    for threshold_key, metric_key in mapping.items():
        value = float(metrics.get(metric_key) or 0.0)
        threshold = float(thresholds.get(threshold_key) or 0.0)
        results[threshold_key] = {
            "metric": metric_key,
            "value": value,
            "threshold": threshold,
            "passed": value <= threshold,
        }
    return results


def _has_readback_attempt(block: Dict[str, Any]) -> bool:
    return bool(
        block.get("read_status")
        or block.get("trigger_rule")
        or block.get("retrieved_image_paths")
    )


def _list_value(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _step_from_path(path: str) -> Optional[int]:
    match = re.search(r"(?:step_|keyframe_)?(\d+)(?:\.[A-Za-z0-9]+)?$", path)
    if not match:
        return None
    return int(match.group(1))


def _contains_adjacent_triplet(steps: List[int], expected_gap: Optional[int] = None) -> bool:
    unique_steps = sorted(set(steps))
    for first, second, third in zip(unique_steps, unique_steps[1:], unique_steps[2:]):
        gap = second - first
        if gap == third - second and (expected_gap is None or gap == expected_gap):
            return True
    return False


def _trigger_key(block: Dict[str, Any]) -> str:
    explicit = str(block.get("trigger_key") or block.get("readback_trigger_key") or "")
    if explicit:
        return explicit
    scene_id = str(block.get("scene_id") or "")
    episode_id = str(block.get("episode_id") or "")
    step_id = block.get("step_id")
    trigger_rule = str(block.get("trigger_rule") or "")
    if scene_id and episode_id and step_id is not None and trigger_rule:
        return f"{scene_id}:{episode_id}:{step_id}:{trigger_rule}"
    return ""


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
