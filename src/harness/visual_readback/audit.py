import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PRIMARY_TRIGGER_SOURCES = {"online_controller", "static_pre_run"}
OFFLINE_TRIGGER_SOURCES = {"offline_fixture"}


def build_phase0_audit_report(trace_paths: Iterable[Path]) -> Dict[str, Any]:
    episodes = set()
    trigger_distribution: Counter[str] = Counter()
    candidate_case_set: List[Dict[str, Any]] = []
    offline_stress_set: List[Dict[str, Any]] = []
    oracle_rejected: List[Dict[str, Any]] = []
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
    return {
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
