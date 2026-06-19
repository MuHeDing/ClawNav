import json
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


MANIFEST_VERSION = "visual_readback_fixed_v1"
EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION = "event_gated_phase0_v1"
REQUIRED_TOP_LEVEL_FIELDS = {
    "manifest_version",
    "generation_run_id",
    "primary_set_name",
    "selection_basis",
    "oracle_selection_fields",
    "shuffle_scope",
    "shuffle_seed",
    "cases",
}
REQUIRED_CASE_FIELDS = {
    "case_id",
    "scene_id",
    "episode_id",
    "step_id",
    "trigger_rule",
    "trigger_source",
    "candidate_action_from_v0",
    "current_image_path",
    "query_text",
    "memory_hits",
    "frozen_policy_inputs",
    "required_verifier_labels",
    "required_images",
    "adjudication",
}
REQUIRED_EVENT_GATED_PHASE0_ROW_FIELDS = {
    "manifest_schema_version",
    "scene_id",
    "episode_id",
    "max_steps",
    "route_event_labels",
    "accepted_step_tolerance",
    "shared_readback_trigger_keys",
    "trigger_key_source",
    "phase0_stop_go_thresholds",
    "selection_rules",
    "selection_seed",
    "source_dataset_id",
    "source_run_id",
    "source_file_hashes",
}
REQUIRED_PHASE0_THRESHOLD_FIELDS = {
    "route_event_miss_rate_max",
    "exact_duplicate_rate_max",
    "adjacent_triplet_rate_max",
    "ambiguous_evidence_rate_max",
    "comparison_direction",
    "invalid_comparison_behavior",
}
REQUIRED_ROUTE_EVENT_FIELDS = {"event_type", "step_id"}


class VisualReadbackManifestError(ValueError):
    pass


@dataclass
class VisualReadbackManifest:
    data: Dict[str, Any]

    @property
    def manifest_version(self) -> str:
        return str(self.data.get("manifest_version") or "")

    @property
    def cases(self) -> List[Dict[str, Any]]:
        return self.data["cases"]

    @property
    def shuffle_seed(self) -> int:
        return int(self.data.get("shuffle_seed") or 0)

    @property
    def negative_control_pool(self) -> List[Dict[str, Any]]:
        return list(self.data.get("negative_control_pool") or [])


@dataclass
class EventGatedPhase0Manifest:
    path: Path
    rows: List[Dict[str, Any]]
    manifest_sha256: str

    @property
    def manifest_schema_version(self) -> str:
        if not self.rows:
            return ""
        return str(self.rows[0].get("manifest_schema_version") or "")

    @property
    def cases(self) -> List[Dict[str, Any]]:
        return self.rows


def load_visual_readback_manifest(path: Path) -> VisualReadbackManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VisualReadbackManifestError("manifest root must be an object")
    _validate_top_level(data)
    _validate_cases(data["cases"])
    return VisualReadbackManifest(data)


def load_event_gated_phase0_manifest_jsonl(path: Path) -> EventGatedPhase0Manifest:
    manifest_path = Path(path)
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise VisualReadbackManifestError(f"row {line_number} must be an object")
        rows.append(data)
    if not rows:
        raise VisualReadbackManifestError("event-gated Phase 0 manifest must contain at least one row")
    _validate_event_gated_phase0_rows(rows)
    manifest_sha256 = compute_event_gated_phase0_manifest_sha256(rows)
    embedded_hashes = {
        str(row.get("manifest_sha256") or "")
        for row in rows
        if str(row.get("manifest_sha256") or "")
    }
    if len(embedded_hashes) > 1:
        raise VisualReadbackManifestError("manifest_sha256 must be identical across rows")
    if embedded_hashes and embedded_hashes != {manifest_sha256}:
        raise VisualReadbackManifestError("manifest_sha256 does not match canonical rows")
    return EventGatedPhase0Manifest(
        path=manifest_path,
        rows=rows,
        manifest_sha256=manifest_sha256,
    )


def compute_event_gated_phase0_manifest_sha256(rows: Iterable[Dict[str, Any]]) -> str:
    canonical_lines = []
    for row in rows:
        canonical = dict(row)
        canonical.pop("manifest_sha256", None)
        canonical_lines.append(
            json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        )
    payload = "\n".join(canonical_lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_v5_replacement_hits(
    manifest: VisualReadbackManifest,
    case: Dict[str, Any],
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    candidates = [
        dict(item)
        for item in manifest.negative_control_pool
        if item.get("negative_control_label") in {"wrong", "unrelated"}
        and item.get("memory_id") not in _memory_ids(case.get("memory_hits") or [])
    ]
    if not candidates:
        case["negative_control_label"] = "negative_control_unjudgeable"
        case["v5_attached_memory_hits"] = []
        return []
    rng = random.Random(f"{manifest.shuffle_seed}:{case.get('case_id')}")
    rng.shuffle(candidates)
    selected = candidates[:top_k]
    case["negative_control_label"] = "wrong"
    case["v5_attached_memory_hits"] = selected
    return selected


def _validate_top_level(data: Dict[str, Any]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL_FIELDS.difference(data.keys()))
    if missing:
        raise VisualReadbackManifestError(
            "manifest missing required fields: " + ", ".join(missing)
        )
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise VisualReadbackManifestError(
            f"manifest_version must be {MANIFEST_VERSION}"
        )
    if not isinstance(data.get("cases"), list):
        raise VisualReadbackManifestError("cases must be a list")
    oracle_fields = data.get("oracle_selection_fields")
    if oracle_fields:
        raise VisualReadbackManifestError(
            "oracle_selection_fields must be empty for primary manifests"
        )


def _validate_cases(cases: List[Dict[str, Any]]) -> None:
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise VisualReadbackManifestError(f"case {index} must be an object")
        missing = sorted(REQUIRED_CASE_FIELDS.difference(case.keys()))
        if missing:
            raise VisualReadbackManifestError(
                f"case {case.get('case_id', index)} missing required fields: "
                + ", ".join(missing)
            )
        if not isinstance(case.get("memory_hits"), list):
            raise VisualReadbackManifestError("memory_hits must be a list")
        if not isinstance(case.get("frozen_policy_inputs"), dict):
            raise VisualReadbackManifestError("frozen_policy_inputs must be an object")
        if not isinstance(case.get("required_verifier_labels"), list):
            raise VisualReadbackManifestError(
                "required_verifier_labels must be a list"
            )
        if not isinstance(case.get("adjudication"), dict):
            raise VisualReadbackManifestError("adjudication must be an object")


def _memory_ids(hits: List[Dict[str, Any]]) -> set[str]:
    return {str(hit.get("memory_id") or "") for hit in hits if isinstance(hit, dict)}


def _validate_event_gated_phase0_rows(rows: List[Dict[str, Any]]) -> None:
    first_thresholds = None
    for index, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_EVENT_GATED_PHASE0_ROW_FIELDS.difference(row.keys()))
        if missing:
            raise VisualReadbackManifestError(
                f"row {index} missing required fields: " + ", ".join(missing)
            )
        if row.get("manifest_schema_version") != EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION:
            raise VisualReadbackManifestError(
                f"row {index} manifest_schema_version must be "
                f"{EVENT_GATED_PHASE0_MANIFEST_SCHEMA_VERSION}"
            )
        _validate_non_empty_string(row, "scene_id", index)
        _validate_non_empty_string(row, "episode_id", index)
        _validate_non_negative_int(row, "max_steps", index)
        _validate_non_negative_int(row, "accepted_step_tolerance", index)
        _validate_route_event_labels(row, index)
        _validate_shared_trigger_keys(row, index)
        _validate_source_hashes(row, index)
        thresholds = _validate_phase0_thresholds(row, index)
        if first_thresholds is None:
            first_thresholds = thresholds
        elif thresholds != first_thresholds:
            raise VisualReadbackManifestError(
                "phase0_stop_go_thresholds must be identical across rows"
            )


def _validate_non_empty_string(row: Dict[str, Any], key: str, index: int) -> None:
    if not str(row.get(key) or ""):
        raise VisualReadbackManifestError(f"row {index} {key} must be non-empty")


def _validate_non_negative_int(row: Dict[str, Any], key: str, index: int) -> None:
    value = row.get(key)
    if not isinstance(value, int) or value < 0:
        raise VisualReadbackManifestError(f"row {index} {key} must be a non-negative integer")


def _validate_route_event_labels(row: Dict[str, Any], index: int) -> None:
    labels = row.get("route_event_labels")
    if not isinstance(labels, list) or not labels:
        raise VisualReadbackManifestError(f"row {index} route_event_labels must be a non-empty list")
    for label_index, label in enumerate(labels):
        if not isinstance(label, dict):
            raise VisualReadbackManifestError(
                f"row {index} route_event_labels[{label_index}] must be an object"
            )
        missing = REQUIRED_ROUTE_EVENT_FIELDS.difference(label.keys())
        if missing:
            raise VisualReadbackManifestError(
                f"row {index} route_event_labels[{label_index}] missing required fields: "
                + ", ".join(sorted(missing))
            )
        if str(label.get("event_type") or "") not in {"STOP", "TURN_LEFT", "TURN_RIGHT"}:
            raise VisualReadbackManifestError(
                f"row {index} route_event_labels[{label_index}] has unsupported event_type"
            )
        if not isinstance(label.get("step_id"), int) or label["step_id"] < 0:
            raise VisualReadbackManifestError(
                f"row {index} route_event_labels[{label_index}] step_id must be a non-negative integer"
            )


def _validate_shared_trigger_keys(row: Dict[str, Any], index: int) -> None:
    keys = row.get("shared_readback_trigger_keys")
    if not isinstance(keys, list):
        raise VisualReadbackManifestError(
            f"row {index} shared_readback_trigger_keys must be a list"
        )
    for key in keys:
        if not isinstance(key, str) or not key:
            raise VisualReadbackManifestError(
                f"row {index} shared_readback_trigger_keys must contain non-empty strings"
            )


def _validate_source_hashes(row: Dict[str, Any], index: int) -> None:
    hashes = row.get("source_file_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise VisualReadbackManifestError(f"row {index} source_file_hashes must be a non-empty object")
    for key, value in hashes.items():
        if not str(key) or not str(value):
            raise VisualReadbackManifestError(
                f"row {index} source_file_hashes keys and values must be non-empty"
            )


def _validate_phase0_thresholds(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    thresholds = row.get("phase0_stop_go_thresholds")
    if not isinstance(thresholds, dict):
        raise VisualReadbackManifestError(
            f"row {index} phase0_stop_go_thresholds must be an object"
        )
    missing = sorted(REQUIRED_PHASE0_THRESHOLD_FIELDS.difference(thresholds.keys()))
    if missing:
        raise VisualReadbackManifestError(
            f"row {index} phase0_stop_go_thresholds missing required fields: "
            + ", ".join(missing)
        )
    for key in (
        "route_event_miss_rate_max",
        "exact_duplicate_rate_max",
        "adjacent_triplet_rate_max",
        "ambiguous_evidence_rate_max",
    ):
        value = thresholds.get(key)
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise VisualReadbackManifestError(
                f"row {index} phase0_stop_go_thresholds.{key} must be between 0 and 1"
            )
    if thresholds.get("comparison_direction") not in {"lower_is_better"}:
        raise VisualReadbackManifestError(
            f"row {index} phase0_stop_go_thresholds.comparison_direction is unsupported"
        )
    if thresholds.get("invalid_comparison_behavior") not in {"mark_invalid"}:
        raise VisualReadbackManifestError(
            f"row {index} phase0_stop_go_thresholds.invalid_comparison_behavior is unsupported"
        )
    return thresholds
