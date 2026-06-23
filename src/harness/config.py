from dataclasses import dataclass
from typing import Any, Union


VISUAL_READBACK_PHASE01_MODES = {
    "off",
    "text_only_prompt",
    "path_only",
    "image_read_prompt",
    "image_read_controller",
    "image_read_replan_prompt",
    "image_read_action_override",
    "current_only_controller",
    "shuffled_image_read_controller",
}
VISUAL_READBACK_CONTROLLER_ONLY_MODES = {
    "image_read_controller",
    "current_only_controller",
}
VISUAL_READBACK_IMAGE_BACKED_MEMORY_BACKENDS = {
    "spatial_http",
    "image_backed_local",
}
VISUAL_READBACK_STOP_FALLBACK_POLICIES = {
    "log_only",
    "previous_non_stop_else_move_forward",
}
VISUAL_READBACK_TRIGGER_POLICIES = {
    "sparse_action_override",
    "dense_action_override",
}
KEYFRAME_POLICY_MODES = {
    "interval",
    "event_gated_smoke",
}
KEYFRAME_POLICY_CHECKPOINT_LABELS = {
    "event_gated_smoke_gate",
    "event_gated_smoke_audit",
}


def parse_visual_readback_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_action_override_budget(value: Any) -> Union[int, str]:
    normalized = str(value).strip().lower()
    if normalized == "adaptive":
        return "adaptive"
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("visual_readback_max_action_overrides_per_episode must be positive")
    return parsed


def validate_visual_readback_config(
    config: "HarnessConfig",
    require_fixed_manifest: bool = False,
) -> None:
    mode = str(config.visual_readback_mode or "off")
    if mode == "direct_policy_images":
        raise ValueError("direct_policy_images is not supported in Phase 0/1")
    if mode not in VISUAL_READBACK_PHASE01_MODES:
        raise ValueError(f"unknown visual_readback_mode: {mode}")
    if config.visual_readback_top_k <= 0:
        raise ValueError("visual_readback_top_k must be positive")
    if config.visual_readback_timeout_ms <= 0:
        raise ValueError("visual_readback_timeout_ms must be positive")
    if config.visual_readback_low_confidence >= config.visual_readback_high_confidence:
        raise ValueError(
            "visual_readback_low_confidence must be lower than "
            "visual_readback_high_confidence"
        )
    if mode in VISUAL_READBACK_CONTROLLER_ONLY_MODES and not config.visual_readback_control_only:
        raise ValueError(f"{mode} requires visual_readback_control_only=true")
    if mode == "shuffled_image_read_controller" and not config.visual_readback_shuffle_seed:
        raise ValueError("shuffled_image_read_controller requires shuffle_seed")
    if (
        config.visual_readback_smoke_seed_memory
        and mode != "image_read_controller"
    ):
        raise ValueError(
            "visual_readback_smoke_seed_memory requires "
            "visual_readback_mode=image_read_controller"
        )
    if config.visual_readback_stop_fallback_policy not in VISUAL_READBACK_STOP_FALLBACK_POLICIES:
        raise ValueError(
            "unknown visual_readback_stop_fallback_policy: "
            f"{config.visual_readback_stop_fallback_policy}"
        )
    try:
        parse_action_override_budget(
            config.visual_readback_max_action_overrides_per_episode
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "visual_readback_max_action_overrides_per_episode must be "
            "positive or adaptive"
        ) from exc
    if config.visual_readback_trigger_policy not in VISUAL_READBACK_TRIGGER_POLICIES:
        raise ValueError(
            "unknown visual_readback_trigger_policy: "
            f"{config.visual_readback_trigger_policy}"
        )
    if mode != "off" and config.memory_backend not in VISUAL_READBACK_IMAGE_BACKED_MEMORY_BACKENDS:
        raise ValueError(
            "visual readback requires image-backed memory; "
            f"memory_backend={config.memory_backend!r}"
        )
    if require_fixed_manifest and not config.visual_readback_fixed_case_manifest_path:
        raise ValueError("fixed_case_manifest is required for Phase 1b replay")
    if config.keyframe_policy_mode in KEYFRAME_POLICY_CHECKPOINT_LABELS:
        raise ValueError(
            f"{config.keyframe_policy_mode} is a validation checkpoint label; "
            "use keyframe_policy_mode=event_gated_smoke"
        )
    if config.keyframe_policy_mode not in KEYFRAME_POLICY_MODES:
        raise ValueError(f"unknown keyframe_policy_mode: {config.keyframe_policy_mode}")
    if config.keyframe_min_gap_steps <= 0:
        raise ValueError("keyframe_min_gap_steps must be positive")
    if config.keyframe_episode_cap <= 0:
        raise ValueError("keyframe_episode_cap must be positive")
    if config.keyframe_coverage_gap_steps <= 0:
        raise ValueError("keyframe_coverage_gap_steps must be positive")


@dataclass
class HarnessConfig:
    harness_mode: str = "memory_recall"
    harness_runtime: str = "phase2"
    memory_backend: str = "fake"
    spatial_memory_url: str = "http://127.0.0.1:8022"
    memory_manifest_path: str = ""
    openclaw_workspace_path: str = ""
    openclaw_service_registry_path: str = ""
    openclaw_service_host: str = "127.0.0.1"
    openclaw_planner_backend: str = "rule"
    openclaw_gateway_url: str = ""
    openclaw_gateway_timeout_s: float = 5.0
    openclaw_executor_backend: str = "habitat"
    openclaw_robot_executor_url: str = ""
    openclaw_subagent_backend: str = "fake"
    openclaw_enable_subagent_planner: bool = False
    openclaw_enable_subagent_critic: bool = False
    openclaw_enable_subagent_memory_curator: bool = False
    openclaw_allow_planner_action_override: bool = False
    max_internal_calls_per_step: int = 3
    recall_interval_steps: int = 5
    max_replans_per_episode: int = 3
    max_memory_images: int = 2
    max_prompt_context_chars: int = 1200
    allow_oracle_metrics_for_decision: bool = False
    memory_source: str = "episode-local"
    expose_sim_pose_online: bool = False
    visual_readback_mode: str = "off"
    visual_readback_trigger_policy: str = "sparse_action_override"
    visual_readback_top_k: int = 3
    visual_readback_timeout_ms: int = 90000
    visual_readback_low_confidence: float = 0.6
    visual_readback_high_confidence: float = 0.8
    visual_readback_shuffle_scope: str = "same_scene_other_episode"
    visual_readback_shuffle_seed: str = ""
    visual_readback_control_only: bool = True
    visual_readback_fixed_case_manifest_path: str = ""
    visual_readback_stop_fallback_policy: str = "log_only"
    visual_readback_smoke_seed_memory: bool = False
    visual_readback_max_action_overrides_per_episode: Union[int, str] = 1
    keyframe_policy_mode: str = "interval"
    keyframe_min_gap_steps: int = 5
    keyframe_episode_cap: int = 64
    keyframe_coverage_gap_steps: int = 20
    keyframe_debug_save_all_eligible: bool = False
