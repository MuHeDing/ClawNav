from dataclasses import dataclass


@dataclass
class HarnessConfig:
    policy_backend: str = "janus_policy"
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
    keyframe_policy_mode: str = "interval"
    keyframe_min_gap_steps: int = 5
    keyframe_episode_cap: int = 64
    keyframe_coverage_gap_steps: int = 20
    keyframe_debug_save_all_eligible: bool = False
    map_assist_mode: str = "off"
    map_frame_interval_steps: int = 5
    motion_feedback_enabled: bool = False
    forward_stall_odometry_enabled: bool = False
    map_collision_overlay_enabled: bool = False
    dynamic_visual_context_enabled: bool = False
    staged_visual_memory_enabled: bool = False
    staged_memory_treatment: str = "on"
    staged_segmentation_timeout_s: float = 120.0
    staged_instruction_max_chars: int = 4096
    staged_segmentation_request_max_bytes: int = 64 * 1024
    staged_max_instruction_stages: int = 12
    stage_min_translation_m: float = 0.25
    stage_min_heading_change_deg: float = 15.0
    staged_visual_store_capacity: int = 64
    staged_registry_max_candidates: int = 2
    staged_semantic_query_max_records: int = 4
    staged_memory_event_cap: int = 64
    staged_recovery_retrigger_steps: int = 3
    staged_shadow_manifest_path: str = ""
    staged_shadow_max_events: int = 5

    def __post_init__(self) -> None:
        if self.staged_visual_memory_enabled and self.policy_backend != "qwen_direct":
            raise ValueError(
                "staged_visual_memory_enabled requires policy_backend=qwen_direct"
            )
        if (
            self.staged_visual_memory_enabled
            and not self.dynamic_visual_context_enabled
        ):
            raise ValueError(
                "staged_visual_memory_enabled requires dynamic visual context"
            )
        if self.staged_memory_treatment not in {"on", "off_ablation"}:
            raise ValueError("staged_memory_treatment must be one of: on, off_ablation")
        if (
            self.staged_memory_treatment == "off_ablation"
            and not self.staged_visual_memory_enabled
        ):
            raise ValueError(
                "staged_memory_treatment=off_ablation requires "
                "staged_visual_memory_enabled"
            )
        positive_integer_fields = (
            "staged_instruction_max_chars",
            "staged_segmentation_request_max_bytes",
            "staged_max_instruction_stages",
            "staged_visual_store_capacity",
            "staged_registry_max_candidates",
            "staged_semantic_query_max_records",
            "staged_memory_event_cap",
            "staged_recovery_retrigger_steps",
            "staged_shadow_max_events",
        )
        for field_name in positive_integer_fields:
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if float(self.staged_segmentation_timeout_s) <= 0:
            raise ValueError("staged_segmentation_timeout_s must be positive")
        if float(self.stage_min_translation_m) < 0:
            raise ValueError("stage_min_translation_m must be non-negative")
        if float(self.stage_min_heading_change_deg) < 0:
            raise ValueError("stage_min_heading_change_deg must be non-negative")
