from omegaconf import OmegaConf


def is_summary_row(row):
    return "sucs_all" in row or "success_all" in row


def ensure_ndtw_measurement(measurements, measurement_factory=None):
    if "ndtw" not in measurements:
        if measurement_factory is None:
            measurements["ndtw"] = OmegaConf.create({"type": "NDTW"})
        else:
            measurements["ndtw"] = measurement_factory()
    return measurements


def append_existing_episode_metrics(row, metric_lists):
    if is_summary_row(row):
        return False

    if "ndtw" not in row:
        raise ValueError(
            "Existing result.json contains episode rows without 'ndtw'. "
            "Delete the stale result file and rerun evaluation."
        )

    metric_lists["sucs"].append(row["success"])
    metric_lists["spls"].append(row["spl"])
    metric_lists["oss"].append(row["os"])
    metric_lists["ones"].append(row["ne"])
    metric_lists["ndtws"].append(row["ndtw"])
    return True


def build_episode_result_record(
    scene_id,
    episode_id,
    episode_instruction,
    metrics,
    step_id,
    action_normalized,
):
    return {
        "scene_id": scene_id,
        "episode_id": episode_id,
        "success": metrics["success"],
        "spl": metrics["spl"],
        "os": metrics["oracle_success"],
        "ne": metrics["distance_to_goal"],
        "ndtw": metrics["ndtw"],
        "steps": step_id,
        "episode_instruction": episode_instruction,
        "action_normalized": action_normalized,
    }


def build_summary_result_record(
    sucs,
    spls,
    oss,
    ones,
    ndtws,
    total_action_norm_stats,
):
    total_episodes = len(sucs)
    return {
        "sucs_all": _mean(sucs),
        "spls_all": _mean(spls),
        "oss_all": _mean(oss),
        "ones_all": _mean(ones),
        "ndtws_all": _mean(ndtws),
        "length": total_episodes,
        "normalization_ratio": (
            total_action_norm_stats / total_episodes if total_episodes > 0 else 0
        ),
    }


def _mean(values):
    if len(values) == 0:
        raise ValueError("Cannot compute a mean from an empty sequence.")

    mean_value = sum(values) / len(values)
    if hasattr(mean_value, "item"):
        return mean_value.item()
    return float(mean_value)
