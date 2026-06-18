import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


MANIFEST_VERSION = "visual_readback_fixed_v1"
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


def load_visual_readback_manifest(path: Path) -> VisualReadbackManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VisualReadbackManifestError("manifest root must be an object")
    _validate_top_level(data)
    _validate_cases(data["cases"])
    return VisualReadbackManifest(data)


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
