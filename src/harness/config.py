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
