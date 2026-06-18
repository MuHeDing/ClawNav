from dataclasses import dataclass
from typing import Any


VISUAL_READBACK_PHASE01_MODES = {
    "off",
    "text_only_prompt",
    "path_only",
    "image_read_prompt",
    "image_read_controller",
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


def parse_visual_readback_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


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
    if mode != "off" and config.memory_backend not in VISUAL_READBACK_IMAGE_BACKED_MEMORY_BACKENDS:
        raise ValueError(
            "visual readback requires image-backed memory; "
            f"memory_backend={config.memory_backend!r}"
        )
    if require_fixed_manifest and not config.visual_readback_fixed_case_manifest_path:
        raise ValueError("fixed_case_manifest is required for Phase 1b replay")


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
