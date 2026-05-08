from dataclasses import dataclass


@dataclass
class HarnessConfig:
    harness_mode: str = "memory_recall"
    memory_backend: str = "fake"
    spatial_memory_url: str = "http://127.0.0.1:8022"
    max_internal_calls_per_step: int = 3
    recall_interval_steps: int = 5
    max_replans_per_episode: int = 3
    max_memory_images: int = 2
    max_prompt_context_chars: int = 1200
    allow_oracle_metrics_for_decision: bool = False
    memory_source: str = "episode-local"
    expose_sim_pose_online: bool = False

