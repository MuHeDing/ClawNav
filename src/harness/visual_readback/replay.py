from typing import Any, Dict, List


PROMPT_MODES = {"text_only_prompt", "path_only", "image_read_prompt"}


def assert_live_hits_match_manifest(
    case: Dict[str, Any],
    live_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    expected = _memory_ids(case.get("memory_hits") or [])
    actual = _memory_ids(live_hits)
    if actual == expected:
        return {
            "status": "ok",
            "expected_memory_ids": expected,
            "actual_memory_ids": actual,
        }
    return {
        "status": "replay_mismatch",
        "expected_memory_ids": expected,
        "actual_memory_ids": actual,
    }


def build_replay_payload(case: Dict[str, Any], mode: str) -> Dict[str, Any]:
    policy_inputs = dict(case.get("frozen_policy_inputs") or {})
    policy_inputs.pop("action_text", None)
    base = {
        "mode": mode,
        "case_id": case.get("case_id"),
        "scene_id": case.get("scene_id"),
        "episode_id": case.get("episode_id"),
        "step_id": case.get("step_id"),
        "trigger_rule": case.get("trigger_rule"),
        "candidate_action_from_v0": case.get("candidate_action_from_v0"),
        "current_image_path": case.get("current_image_path"),
        "query_text": case.get("query_text"),
        "policy_inputs": policy_inputs,
        "required_verifier_labels": list(case.get("required_verifier_labels") or []),
        "required_images": case.get("required_images"),
        "adjudication": dict(case.get("adjudication") or {}),
    }
    if mode in PROMPT_MODES:
        base["memory_hits"] = list(case.get("memory_hits") or [])
        base["recompute_policy"] = True
        return base
    if mode == "image_read_controller":
        base["memory_hits"] = list(case.get("memory_hits") or [])
        base["control_only"] = True
        return base
    if mode == "current_only_controller":
        base["memory_hits"] = []
        base["control_only"] = True
        base["current_only"] = True
        return base
    if mode == "shuffled_image_read_controller":
        base["memory_hits"] = list(case.get("v5_attached_memory_hits") or [])
        base["control_only"] = True
        base["negative_control_label"] = case.get("negative_control_label", "")
        return base
    if mode == "off":
        base["memory_hits"] = []
        return base
    raise ValueError(f"unsupported visual readback replay mode: {mode}")


def _memory_ids(hits: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        ids.append(str(hit.get("memory_id") or ""))
    return ids
