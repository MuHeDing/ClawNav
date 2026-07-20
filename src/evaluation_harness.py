import argparse
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from harness.config import HarnessConfig
from harness.controller import HarnessController
from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.logging.harness_logger import HarnessLogger
from harness.memory.episode_visual_store import EpisodeVisualMemoryStore
from harness.memory.memory_manager import MemoryManager
from harness.memory.spatial_memory_client import (
    FakeSpatialMemoryClient,
    SpatialMemoryHttpClient,
)
from harness.memory.task_memory import TaskMemory
from harness.memory.working_memory import WorkingMemory
from harness.openclaw.map_context import FloorplanMapContextProvider
from harness.skill_registry import SkillRegistry
from harness.skills.memory_query import MemoryQuerySkill
from harness.skills.memory_write import MemoryWriteSkill
from harness.skills.navigation_policy import NavigationPolicySkill
from harness.skills.progress_critic import ProgressCriticSkill
from harness.skills.replanner import ReplannerSkill
from harness.skills.visual_memory_curator import VisualMemoryCuratorSkill
from harness.types import SkillResult


ACTIONS2IDX = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}
ACTION_SCALE_CONTEXT = {
    "MOVE_FORWARD": {"effect": "advance", "distance_m": 0.25},
    "TURN_LEFT": {"effect": "rotate_left", "angle_degrees": 15},
    "TURN_RIGHT": {"effect": "rotate_right", "angle_degrees": 15},
    "STOP": {"effect": "stop"},
}

JANUS_POLICY_BACKEND = "janus_policy"
QWEN_DIRECT_POLICY_BACKEND = "qwen_direct"


def action_scale_context() -> Dict[str, Dict[str, Any]]:
    return {action: dict(details) for action, details in ACTION_SCALE_CONTEXT.items()}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ClawNav OpenClaw-style harness evaluation"
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument(
        "--policy_backend",
        choices=(JANUS_POLICY_BACKEND, QWEN_DIRECT_POLICY_BACKEND),
        default=JANUS_POLICY_BACKEND,
    )
    parser.add_argument(
        "--habitat_config_path", type=str, default="config/vln_r2r.yaml"
    )
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--min_pixels", type=int, default=28 * 28)
    parser.add_argument("--kv_start_size", type=int, default=8)
    parser.add_argument("--kv_recent_size", type=int, default=24)
    parser.add_argument("--max_steps", type=positive_int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--save_video_ratio", type=float, default=0.05)
    parser.add_argument(
        "--harness_stream_video",
        action="store_true",
        default=False,
        help=(
            "Also write raw harness current-frame mp4s under videos/. "
            "By default --save_video writes only the evaluator vis_<epoch>/ mp4."
        ),
    )
    parser.add_argument("--save_step_artifacts", action="store_true", default=False)
    parser.add_argument(
        "--save_step_artifacts_with_video_only", action="store_true", default=False
    )
    parser.add_argument(
        "--disable_qualitative_json", action="store_true", default=False
    )
    parser.add_argument("--harness_mode", type=str, default="memory_recall")
    parser.add_argument("--harness_memory_backend", type=str, default="fake")
    parser.add_argument(
        "--spatial_memory_url", type=str, default="http://127.0.0.1:8022"
    )
    parser.add_argument("--memory_manifest_path", type=str, default="")
    parser.add_argument("--harness_memory_source", type=str, default="episode-local")
    parser.add_argument("--harness_max_internal_calls", type=int, default=3)
    parser.add_argument("--harness_recall_interval_steps", type=int, default=5)
    parser.add_argument("--harness_debug_max_episodes", type=int, default=None)
    parser.add_argument(
        "--harness_episode_keys",
        type=str,
        default="",
        help="Comma-separated scene_id:episode_id keys to evaluate, preserving the requested order.",
    )
    parser.add_argument("--harness_trace_rank", type=int, default=0)
    parser.add_argument("--expose_sim_pose_online", action="store_true", default=False)
    parser.add_argument(
        "--keyframe_policy_mode",
        type=str,
        default=os.environ.get("OPENCLAW_KEYFRAME_POLICY_MODE", "interval"),
    )
    parser.add_argument(
        "--keyframe_min_gap_steps",
        type=int,
        default=env_int("OPENCLAW_KEYFRAME_MIN_GAP_STEPS", 5),
    )
    parser.add_argument(
        "--keyframe_episode_cap",
        type=int,
        default=env_int("OPENCLAW_KEYFRAME_EPISODE_CAP", 64),
    )
    parser.add_argument(
        "--keyframe_coverage_gap_steps",
        type=int,
        default=env_int("OPENCLAW_KEYFRAME_COVERAGE_GAP_STEPS", 20),
    )
    parser.add_argument(
        "--keyframe_debug_save_all_eligible",
        action="store_true",
        default=env_bool("OPENCLAW_KEYFRAME_DEBUG_SAVE_ALL_ELIGIBLE", False),
    )
    parser.add_argument(
        "--map_assist_mode",
        choices=("off", "floorplan_map_assisted"),
        default=os.environ.get("OPENCLAW_MAP_ASSIST_MODE", "off"),
    )
    parser.add_argument(
        "--map_frame_interval_steps",
        type=positive_int,
        default=env_int("OPENCLAW_MAP_FRAME_INTERVAL_STEPS", 5),
    )
    parser.add_argument(
        "--motion_feedback_enabled",
        action="store_true",
        default=env_bool("OPENCLAW_MOTION_FEEDBACK_ENABLED", False),
    )
    parser.add_argument(
        "--forward_stall_odometry_enabled",
        action="store_true",
        default=env_bool("OPENCLAW_FORWARD_STALL_ODOMETRY_ENABLED", False),
    )
    parser.add_argument(
        "--map_collision_overlay_enabled",
        action="store_true",
        default=env_bool("OPENCLAW_MAP_COLLISION_OVERLAY_ENABLED", False),
    )
    parser.add_argument(
        "--dynamic_visual_context_enabled",
        action="store_true",
        default=env_bool("OPENCLAW_DYNAMIC_VISUAL_CONTEXT_ENABLED", False),
    )
    parser.add_argument(
        "--staged_visual_memory_enabled",
        action="store_true",
        default=env_bool("OPENCLAW_STAGED_VISUAL_MEMORY_ENABLED", False),
    )
    parser.add_argument(
        "--staged_memory_treatment",
        choices=("on", "off_ablation"),
        default=os.environ.get("OPENCLAW_STAGED_MEMORY_TREATMENT", "on"),
    )
    parser.add_argument(
        "--stage_min_translation_m",
        type=float,
        default=float(os.environ.get("OPENCLAW_STAGE_MIN_TRANSLATION_M", "0.25")),
    )
    parser.add_argument(
        "--stage_min_heading_change_deg",
        type=float,
        default=float(os.environ.get("OPENCLAW_STAGE_MIN_HEADING_CHANGE_DEG", "15.0")),
    )
    parser.add_argument("--harness_runtime", type=str, default="phase2")
    parser.add_argument("--openclaw_workspace_path", type=str, default="")
    parser.add_argument("--openclaw_service_registry_path", type=str, default="")
    parser.add_argument("--openclaw_service_host", type=str, default="127.0.0.1")
    parser.add_argument("--openclaw_planner_backend", type=str, default="rule")
    parser.add_argument("--openclaw_gateway_url", type=str, default="")
    parser.add_argument("--openclaw_gateway_timeout", type=float, default=5.0)
    parser.add_argument("--openclaw_executor_backend", type=str, default="habitat")
    parser.add_argument("--openclaw_robot_executor_url", type=str, default="")
    parser.add_argument("--openclaw_subagent_backend", type=str, default="fake")
    parser.add_argument(
        "--openclaw_enable_subagent_planner", action="store_true", default=False
    )
    parser.add_argument(
        "--openclaw_enable_subagent_critic", action="store_true", default=False
    )
    parser.add_argument(
        "--openclaw_enable_subagent_memory_curator",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--openclaw_allow_planner_action_override", action="store_true", default=False
    )
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    policy_backend = getattr(args, "policy_backend", JANUS_POLICY_BACKEND)
    if policy_backend == JANUS_POLICY_BACKEND and not getattr(args, "model_path", ""):
        raise SystemExit("--model_path is required when --policy_backend=janus_policy")
    return args


def _canonical_episode_scene_id(scene_id: Any) -> str:
    raw_scene_id = str(scene_id or "")
    normalized = raw_scene_id.rstrip("/").replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return Path(normalized).stem


def _parse_harness_episode_keys(raw_keys: str) -> list[str]:
    keys = [key.strip() for key in str(raw_keys or "").split(",") if key.strip()]
    invalid_keys = [key for key in keys if ":" not in key]
    if invalid_keys:
        raise ValueError(
            "--harness_episode_keys entries must use scene_id:episode_id format: "
            + ", ".join(invalid_keys)
        )
    return keys


def filter_harness_episodes_by_keys(episodes: list[Any], raw_keys: str) -> list[Any]:
    requested_keys = _parse_harness_episode_keys(raw_keys)
    if not requested_keys:
        return episodes

    episodes_by_key = {
        f"{_canonical_episode_scene_id(getattr(episode, 'scene_id', ''))}:{getattr(episode, 'episode_id', '')}": episode
        for episode in episodes
    }
    missing_keys = [key for key in requested_keys if key not in episodes_by_key]
    if missing_keys:
        raise ValueError(
            "--harness_episode_keys requested episodes that are not in the dataset: "
            + ", ".join(missing_keys)
        )
    return [episodes_by_key[key] for key in requested_keys]


def build_harness_config(args: argparse.Namespace) -> HarnessConfig:
    config = HarnessConfig(
        policy_backend=getattr(args, "policy_backend", JANUS_POLICY_BACKEND),
        harness_mode=args.harness_mode,
        memory_backend=args.harness_memory_backend,
        spatial_memory_url=args.spatial_memory_url,
        memory_manifest_path=args.memory_manifest_path,
        max_internal_calls_per_step=args.harness_max_internal_calls,
        recall_interval_steps=args.harness_recall_interval_steps,
        memory_source=args.harness_memory_source,
        expose_sim_pose_online=args.expose_sim_pose_online,
        keyframe_policy_mode=args.keyframe_policy_mode,
        keyframe_min_gap_steps=args.keyframe_min_gap_steps,
        keyframe_episode_cap=args.keyframe_episode_cap,
        keyframe_coverage_gap_steps=args.keyframe_coverage_gap_steps,
        keyframe_debug_save_all_eligible=args.keyframe_debug_save_all_eligible,
        map_assist_mode=args.map_assist_mode,
        map_frame_interval_steps=args.map_frame_interval_steps,
        motion_feedback_enabled=args.motion_feedback_enabled,
        forward_stall_odometry_enabled=args.forward_stall_odometry_enabled,
        map_collision_overlay_enabled=args.map_collision_overlay_enabled,
        dynamic_visual_context_enabled=getattr(
            args,
            "dynamic_visual_context_enabled",
            False,
        ),
        staged_visual_memory_enabled=getattr(
            args,
            "staged_visual_memory_enabled",
            False,
        ),
        staged_memory_treatment=getattr(args, "staged_memory_treatment", "on"),
        stage_min_translation_m=getattr(args, "stage_min_translation_m", 0.25),
        stage_min_heading_change_deg=getattr(
            args,
            "stage_min_heading_change_deg",
            15.0,
        ),
        harness_runtime=args.harness_runtime,
        openclaw_workspace_path=args.openclaw_workspace_path,
        openclaw_service_registry_path=args.openclaw_service_registry_path,
        openclaw_service_host=args.openclaw_service_host,
        openclaw_planner_backend=args.openclaw_planner_backend,
        openclaw_gateway_url=args.openclaw_gateway_url,
        openclaw_gateway_timeout_s=getattr(args, "openclaw_gateway_timeout", 5.0),
        openclaw_executor_backend=args.openclaw_executor_backend,
        openclaw_robot_executor_url=args.openclaw_robot_executor_url,
        openclaw_subagent_backend=args.openclaw_subagent_backend,
        openclaw_enable_subagent_planner=args.openclaw_enable_subagent_planner,
        openclaw_enable_subagent_critic=args.openclaw_enable_subagent_critic,
        openclaw_enable_subagent_memory_curator=args.openclaw_enable_subagent_memory_curator,
        openclaw_allow_planner_action_override=getattr(
            args,
            "openclaw_allow_planner_action_override",
            False,
        ),
    )
    if (
        config.openclaw_service_registry_path
        and config.memory_backend == "spatial_http"
    ):
        from harness.openclaw.service_registry import OpenClawServiceRegistry

        registry = OpenClawServiceRegistry.from_file(
            Path(config.openclaw_service_registry_path),
            service_host=config.openclaw_service_host,
        )
        spatial_url = registry.spatial_memory_url()
        if spatial_url:
            config.spatial_memory_url = spatial_url
    if config.memory_manifest_path:
        from harness.memory.manifest import load_memory_manifest

        load_memory_manifest(Path(config.memory_manifest_path))
    return config


def build_memory_client(config: HarnessConfig):
    if config.memory_backend == "spatial_http":
        return SpatialMemoryHttpClient(
            config.spatial_memory_url,
            memory_source=config.memory_source,
        )
    return FakeSpatialMemoryClient(memory_source=config.memory_source)


def build_harness_components(
    args: argparse.Namespace,
    model: Any = None,
) -> Dict[str, Any]:
    output_path = Path(args.output_path).resolve()
    config = build_harness_config(args)
    direct_policy = config.policy_backend == QWEN_DIRECT_POLICY_BACKEND
    if direct_policy and model is not None:
        raise ValueError("qwen_direct policy backend must not receive a Janus model")
    memory_client = build_memory_client(config)
    episode_visual_store = EpisodeVisualMemoryStore(
        capacity=config.staged_visual_store_capacity
    )
    memory_manager = MemoryManager(
        memory_client,
        config,
        episode_visual_store=episode_visual_store,
    )
    task_memory = TaskMemory()
    working_memory = WorkingMemory(max_recent_frames=args.num_history)
    registry = SkillRegistry()

    if model is not None:
        registry.register(
            NavigationPolicySkill(
                model=model,
                num_history=args.num_history,
                max_memory_images=config.max_memory_images,
                max_prompt_context_chars=config.max_prompt_context_chars,
            )
        )
    registry.register(MemoryQuerySkill(memory_manager))
    registry.register(MemoryWriteSkill(client=memory_client))
    registry.register(ProgressCriticSkill())
    registry.register(ReplannerSkill())
    if config.openclaw_enable_subagent_memory_curator:
        registry.register(VisualMemoryCuratorSkill())

    controller = HarnessController(registry, config)
    adapter = HabitatVLNAdapter(expose_pose_online=config.expose_sim_pose_online)
    openclaw_runtime = None
    if config.harness_runtime == "openclaw_bridge":
        from harness.openclaw.executor import HabitatOpenClawExecutor
        from harness.openclaw.planner import RuleOpenClawPlanner
        from harness.openclaw.runtime import OpenClawVLNRuntime

        if direct_policy and config.openclaw_enable_subagent_planner:
            raise ValueError(
                "qwen_direct policy backend does not support subagent planner fallback"
            )
        if direct_policy and config.openclaw_planner_backend != "gateway":
            raise ValueError(
                "qwen_direct policy backend requires openclaw_planner_backend=gateway"
            )

        if config.openclaw_enable_subagent_planner:
            from harness.openclaw.planner import SubagentOpenClawPlanner
            from harness.openclaw.subagents import FakeSubagentClient

            planner = SubagentOpenClawPlanner(
                FakeSubagentClient(
                    {
                        "intent": "act",
                        "tool_name": "NavigationPolicySkill",
                        "arguments": {},
                        "reason": "fake_subagent",
                    }
                )
            )
        elif config.openclaw_planner_backend == "gateway":
            if not config.openclaw_gateway_url:
                raise ValueError(
                    "openclaw_gateway_url is required for gateway planner backend"
                )
            from harness.openclaw.gateway import OpenClawGatewayClient

            planner = OpenClawGatewayClient(
                base_url=config.openclaw_gateway_url,
                timeout_s=config.openclaw_gateway_timeout_s,
                segmentation_timeout_s=config.staged_segmentation_timeout_s,
            )
        else:
            planner = RuleOpenClawPlanner(
                recall_interval_steps=config.recall_interval_steps,
            )
        if config.openclaw_executor_backend == "robot_http":
            if not config.openclaw_robot_executor_url:
                raise ValueError(
                    "openclaw_robot_executor_url is required for robot_http executor"
                )
            from harness.openclaw.robot_executor import RobotHttpExecutor

            executor = RobotHttpExecutor(config.openclaw_robot_executor_url)
        else:
            executor = HabitatOpenClawExecutor(adapter)
        openclaw_runtime = OpenClawVLNRuntime(
            tool_registry=registry,
            planner=planner,
            executor=executor,
            fallback_planner=RuleOpenClawPlanner(
                recall_interval_steps=config.recall_interval_steps,
            ),
            allow_planner_action_override=config.openclaw_allow_planner_action_override,
            policy_backend=config.policy_backend,
            keyframe_policy_mode=config.keyframe_policy_mode,
            keyframe_min_gap_steps=config.keyframe_min_gap_steps,
            keyframe_episode_cap=config.keyframe_episode_cap,
            keyframe_coverage_gap_steps=config.keyframe_coverage_gap_steps,
            keyframe_debug_save_all_eligible=config.keyframe_debug_save_all_eligible,
            dynamic_visual_context_enabled=config.dynamic_visual_context_enabled,
            staged_visual_memory_enabled=config.staged_visual_memory_enabled,
            episode_visual_store=episode_visual_store,
        )
    logger = HarnessLogger(
        output_path / "harness_traces",
        rank=args.harness_trace_rank,
    )

    return {
        "config": config,
        "memory_client": memory_client,
        "args": args,
        "memory_manager": memory_manager,
        "episode_visual_store": episode_visual_store,
        "task_memory": task_memory,
        "working_memory": working_memory,
        "skill_registry": registry,
        "controller": controller,
        "adapter": adapter,
        "openclaw_runtime": openclaw_runtime,
        "logger": logger,
        "output_path": output_path,
    }


class HarnessModelProxy:
    def __init__(self, base_model: Any, components: Dict[str, Any]) -> None:
        self.base_model = base_model
        self.model = base_model.model
        self.components = components
        args = components.get("args")
        self.save_video = (
            bool(getattr(args, "save_video", False))
            and bool(getattr(args, "harness_stream_video", False))
            if args is not None
            else False
        )
        self.save_video_ratio = (
            float(getattr(args, "save_video_ratio", 0.0)) if args is not None else 0.0
        )
        self.last_action_text = None
        self.episode_invalid = False
        self.episode_invalid_reason = ""
        self.current_scene_id = ""
        self.current_episode_id = ""
        self.recent_keyframe_paths = []
        self._pending_env_state = None
        self._pending_map_context = None
        self._map_local_recovery_active = False
        self._init_map_context_provider()
        self._episode_video_writer = None
        self._episode_save_video = False
        self._episode_video_disabled = False
        self._episode_video_frame_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def start_episode(self, scene_id: str, episode_id: str) -> None:
        self.finalize_episode()
        working_memory = self.components.get("working_memory")
        if working_memory is not None and hasattr(working_memory, "reset"):
            working_memory.reset()
        self.current_scene_id = str(scene_id or "")
        self.current_episode_id = str(episode_id or "")
        self.last_action_text = None
        self.episode_invalid = False
        self.episode_invalid_reason = ""
        self.recent_keyframe_paths = []
        self._pending_env_state = None
        self._pending_map_context = None
        self._map_local_recovery_active = False
        runtime = self.components.get("openclaw_runtime")
        if runtime is not None and hasattr(runtime, "reset_episode"):
            runtime.reset_episode(self.current_scene_id, self.current_episode_id)
        self._reset_map_context_provider()
        self._episode_save_video = self.save_video and (
            random.random() < self.save_video_ratio
        )
        self._episode_video_disabled = False
        self._episode_video_frame_count = 0

    def observe_environment_state(
        self,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        metrics: Dict[str, Any],
        step_id: int,
    ) -> None:
        adapter = self.components.get("adapter")
        if adapter is None or not hasattr(adapter, "build_state"):
            return
        try:
            state = adapter.build_state(
                env,
                episode,
                observations,
                metrics,
                step_id,
                last_action=self.last_action_text,
            )
        except Exception:
            self._pending_env_state = None
            self._pending_map_context = None
            return
        self._pending_env_state = state
        self._update_pending_map_context(
            env=env,
            state=state,
            step_id=step_id,
        )
        safe_diagnostics = self._proxy_safe_diagnostics(state.diagnostics)
        working_memory = self.components.get("working_memory")
        if working_memory is not None:
            if state.online_metrics and hasattr(
                working_memory, "append_online_metrics"
            ):
                working_memory.append_online_metrics(state.online_metrics)
            if safe_diagnostics and hasattr(working_memory, "append_diagnostics"):
                working_memory.append_diagnostics(safe_diagnostics)

    def call_model(self, images, task, step_id):
        current_image = images[-1] if images else None
        self._record_video_frame(current_image)
        state = self._build_proxy_state(task, step_id, current_image)
        runtime = self.components.get("openclaw_runtime")
        if runtime is not None:
            runtime_result = runtime.step(
                state,
                self._runtime_payload(images, step_id),
            )
            action_text = runtime_result.action_text if runtime_result.ok else "STOP"
            self._remember_map_recovery_from_runtime(runtime_result)
            self._remember_promoted_keyframe_from_runtime(runtime_result)
            self.last_action_text = action_text
            self._append_working_memory(images, action_text)
            self._log_runtime_step(state, runtime_result, action_text)
            return [action_text]

        result = self.components["controller"].run_step(
            state,
            {
                "recent_frames": list(images[:-1]),
                "policy_action": self.last_action_text,
            },
        )
        action_text = result.payload.get("action_text", "STOP") if result.ok else "STOP"
        self.last_action_text = action_text
        self._append_working_memory(images, action_text)
        self._log_step(state, result, action_text)
        return [action_text]

    def _runtime_payload(self, images, step_id: int) -> Dict[str, Any]:
        current_image = images[-1] if images else None
        payload = {
            "recent_frames": list(images[:-1]),
            "recent_actions": list(self.components["working_memory"].actions),
            "policy_action": self.last_action_text,
            "run_id": str(self.components.get("output_path") or ""),
            "keyframe_policy_mode": self.components["config"].keyframe_policy_mode,
            "dynamic_visual_context_enabled": self.components[
                "config"
            ].dynamic_visual_context_enabled,
        }
        keyframe_target_path = self._keyframe_target_path(current_image, step_id)
        payload["keyframe_target_path"] = keyframe_target_path
        if self.components["config"].keyframe_policy_mode == "event_gated_smoke":
            payload["current_image_path"] = self._save_current_image_if_needed(
                current_image,
                step_id,
            )
        elif self.components["working_memory"].should_promote_keyframe(step_id):
            image_path = self._save_keyframe_if_needed(current_image, step_id)
            if image_path:
                self._remember_keyframe_path(image_path)
            payload["keyframe_candidate"] = {
                "step_id": step_id,
                "reason": "interval",
                "has_current_image": bool(images),
                "image_path": image_path,
            }
            payload["current_image_path"] = image_path
        else:
            payload["current_image_path"] = self._save_current_image_if_needed(
                current_image,
                step_id,
            )
        payload["recent_keyframe_paths"] = list(self.recent_keyframe_paths)
        self._attach_structured_runtime_context(
            payload,
            step_id=step_id,
            has_current_image=current_image is not None,
        )
        self._attach_map_context_to_payload(payload)
        return payload

    def _init_map_context_provider(self) -> None:
        self._map_context_provider = None
        config = self.components.get("config")
        if config is None or getattr(config, "map_assist_mode", "off") == "off":
            return
        self._map_context_provider = FloorplanMapContextProvider(
            output_root=self.components["output_path"],
            mode=getattr(config, "map_assist_mode", "off"),
            frame_interval_steps=getattr(config, "map_frame_interval_steps", 5),
            collision_overlay_enabled=getattr(
                config,
                "map_collision_overlay_enabled",
                False,
            ),
        )

    def _reset_map_context_provider(self) -> None:
        provider = getattr(self, "_map_context_provider", None)
        if provider is not None:
            provider.reset_episode(self.current_scene_id, self.current_episode_id)

    def _update_pending_map_context(self, env: Any, state: Any, step_id: int) -> None:
        provider = getattr(self, "_map_context_provider", None)
        if provider is None:
            self._pending_map_context = None
            return
        try:
            self._pending_map_context = provider.build_context(
                env=env,
                state=state,
                step_id=step_id,
                scene_id=self.current_scene_id or getattr(state, "scene_id", ""),
                episode_id=self.current_episode_id or getattr(state, "episode_id", ""),
                local_focus=bool(self._map_local_recovery_active),
            )
        except Exception as exc:
            self._pending_map_context = {
                "mode": self.components["config"].map_assist_mode,
                "map_frame_interval_steps": self.components[
                    "config"
                ].map_frame_interval_steps,
                "map_step_id": step_id,
                "map_frame_due": False,
                "map_available": False,
                "map_generation_error": exc.__class__.__name__,
            }

    def _attach_map_context_to_payload(self, payload: Dict[str, Any]) -> None:
        config = self.components["config"]
        if config.map_assist_mode == "off":
            return
        payload["map_assist_mode"] = config.map_assist_mode
        payload["map_frame_interval_steps"] = config.map_frame_interval_steps
        payload["motion_feedback_enabled"] = config.motion_feedback_enabled
        payload[
            "forward_stall_odometry_enabled"
        ] = config.forward_stall_odometry_enabled
        payload["map_collision_overlay_enabled"] = config.map_collision_overlay_enabled
        map_context = getattr(self, "_pending_map_context", None)
        if isinstance(map_context, dict):
            payload["input_regime"] = map_context.get(
                "input_regime",
                "rgb_plus_privileged_floorplan_pose",
            )
            payload["map_context"] = dict(map_context)

    def _remember_map_recovery_from_runtime(self, runtime_result: Any) -> None:
        metadata = getattr(runtime_result, "runtime_metadata", {}) or {}
        self._map_local_recovery_active = bool(
            isinstance(metadata, dict)
            and metadata.get("turn_loop_recovery_active") is True
        )

    def _attach_structured_runtime_context(
        self,
        payload: Dict[str, Any],
        step_id: int,
        has_current_image: bool,
    ) -> None:
        working_memory = self.components["working_memory"]
        recent_actions = list(payload.get("recent_actions") or [])
        action_counts: Dict[str, int] = {}
        for action in recent_actions:
            action_text = str(action or "")
            if not action_text:
                continue
            action_counts[action_text] = action_counts.get(action_text, 0) + 1
        non_oracle_metrics = working_memory.decision_metrics()
        payload["policy_input"] = {
            "policy_backend": self.components["config"].policy_backend,
            "step_id": step_id,
            "last_action_text": self.last_action_text or "",
            "action_scale": action_scale_context(),
        }
        payload["control_context"] = {
            "recent_action_counts": action_counts,
            "recent_forward_count": action_counts.get("MOVE_FORWARD", 0),
            "recent_action_count": len(recent_actions),
            "last_action_text": self.last_action_text or "",
            "current_step_id": step_id,
            "non_oracle_metrics": non_oracle_metrics,
        }
        payload["evidence_context"] = {
            "has_current_image": bool(has_current_image),
            "current_image_path": str(payload.get("current_image_path") or ""),
            "recent_keyframe_count": len(payload.get("recent_keyframe_paths") or []),
        }

    def _save_keyframe_if_needed(self, image, step_id: int) -> str:
        return self._save_image_artifact(image, "keyframes", step_id)

    def _save_current_image_if_needed(self, image, step_id: int) -> str:
        return self._save_image_artifact(image, "openclaw_current_frames", step_id)

    def _save_image_artifact(self, image, root_dir_name: str, step_id: int) -> str:
        if image is None:
            return ""
        path = self._image_artifact_path(image, root_dir_name, step_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(image, "save"):
            image.save(path)
        else:
            path.write_text(str(image), encoding="utf-8")
        return str(path)

    def _keyframe_target_path(self, image, step_id: int) -> str:
        if image is None:
            return ""
        return str(self._image_artifact_path(image, "keyframes", step_id))

    def _image_artifact_path(self, image, root_dir_name: str, step_id: int) -> Path:
        image_dir = self._episode_artifact_dir(root_dir_name)
        suffix = "png" if hasattr(image, "save") else "txt"
        return image_dir / f"step_{step_id:06d}.{suffix}"

    def _episode_artifact_dir(self, root_dir_name: str) -> Path:
        root_dir = Path(self.components["output_path"]) / root_dir_name
        if not self.current_scene_id or not self.current_episode_id:
            return root_dir
        return (
            root_dir
            / self._safe_path_part(self.current_scene_id)
            / self._safe_path_part(self.current_episode_id)
        )

    def _safe_path_part(self, value: str) -> str:
        return "".join(
            char if char.isalnum() or char in "._-" else "_" for char in value
        )

    def _remember_keyframe_path(self, image_path: str) -> None:
        if image_path in self.recent_keyframe_paths:
            return
        self.recent_keyframe_paths.append(image_path)
        self.recent_keyframe_paths = self.recent_keyframe_paths[-8:]

    def _remember_promoted_keyframe_from_runtime(self, runtime_result) -> None:
        metadata = getattr(runtime_result, "runtime_metadata", {}) or {}
        if not isinstance(metadata, dict):
            return
        gate = metadata.get("keyframe_gate") or {}
        if not isinstance(gate, dict):
            return
        if gate.get("promotion_status") != "promoted":
            return
        image_path = str(
            gate.get("promoted_image_path")
            or gate.get("keyframe_image_path")
            or gate.get("image_path")
            or ""
        )
        if image_path:
            self._remember_keyframe_path(image_path)

    def finalize_episode(self):
        self._release_video_writer()

    def _output_path(self) -> Path:
        return self.components["output_path"]

    def _episode_video_path(self) -> Path:
        safe_scene = self._safe_path_part(self.current_scene_id or "scene")
        safe_episode = self._safe_path_part(self.current_episode_id or "episode")
        return self._output_path() / "videos" / safe_scene / f"{safe_episode}.mp4"

    def _episode_video_tmp_path(self) -> Path:
        final_path = self._episode_video_path()
        return final_path.with_name(f"{final_path.stem}.part{final_path.suffix}")

    def _record_video_frame(self, image) -> None:
        if not self._episode_save_video or self._episode_video_disabled:
            return
        frame_bgr = self._to_video_frame_bgr(image)
        if frame_bgr is None:
            return
        if self._episode_video_writer is None:
            self._init_video_writer(frame_bgr.shape[1], frame_bgr.shape[0])
        if self._episode_video_writer is None:
            return
        self._episode_video_writer.write(frame_bgr)
        self._episode_video_frame_count += 1

    def _to_video_frame_bgr(self, image):
        frame_rgb = self._to_rgb_frame(image)
        if frame_rgb is None:
            return None
        import cv2

        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def _to_rgb_frame(self, image):
        if image is None:
            return None
        if hasattr(image, "convert"):
            img = image.convert("RGB")
            return np.array(img)
        if isinstance(image, np.ndarray):
            frame = image
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            if frame.ndim == 3 and frame.shape[2] > 3:
                frame = frame[:, :, :3]
            if frame.ndim != 3:
                return None
            return frame.astype(np.uint8)
        return None

    def _init_video_writer(self, width: int, height: int) -> None:
        import cv2

        output_path = self._episode_video_tmp_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            6,
            (width, height),
        )
        if not writer.isOpened():
            self._episode_video_disabled = True
            writer.release()
            return
        self._episode_video_writer = writer

    def _release_video_writer(self) -> None:
        if self._episode_video_writer is not None:
            self._episode_video_writer.release()
            tmp_path = self._episode_video_tmp_path()
            final_path = self._episode_video_path()
            if self._episode_video_frame_count > 0 and tmp_path.exists():
                tmp_path.replace(final_path)
        self._episode_video_writer = None

    def consume_last_visual_prune_profile(self):
        return self.base_model.consume_last_visual_prune_profile()

    def _build_proxy_state(self, task: str, step_id: int, current_image: Any):
        from harness.types import VLNState

        pending_state = getattr(self, "_pending_env_state", None)
        if pending_state is not None and pending_state.step_id == step_id:
            return VLNState(
                scene_id=self.current_scene_id,
                episode_id=self.current_episode_id,
                instruction=task,
                step_id=step_id,
                current_image=current_image,
                online_metrics=dict(pending_state.online_metrics),
                diagnostics=self._proxy_safe_diagnostics(pending_state.diagnostics),
                pose=pending_state.pose,
                diagnostic_pose=pending_state.diagnostic_pose,
                last_action=self.last_action_text,
            )
        return VLNState(
            scene_id=self.current_scene_id,
            episode_id=self.current_episode_id,
            instruction=task,
            step_id=step_id,
            current_image=current_image,
            online_metrics={},
            diagnostics={},
            last_action=self.last_action_text,
        )

    def _proxy_safe_diagnostics(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(diagnostics, dict):
            return {}
        return {
            key: value for key, value in diagnostics.items() if key != "raw_metrics"
        }

    def _append_working_memory(self, images, action_text: str) -> None:
        working_memory = self.components["working_memory"]
        if images:
            working_memory.append_frame(images[-1])
        working_memory.append_action(action_text)

    def _log_step(self, state, result: SkillResult, action_text: str) -> None:
        trace = self.components["controller"].last_trace
        calls = trace.get("calls", [])
        intent = "recall_memory" if "MemoryQuerySkill" in calls else "act"
        self.components["logger"].log_step(
            state,
            intent=intent,
            skill=",".join(calls),
            reason=trace.get("fallback_reason", ""),
            memory_backend=self.components["config"].memory_backend,
            memory_source=self.components["config"].memory_source,
            action_text=action_text,
            fallback=bool(trace.get("fallback", False)),
            decision_inputs={},
            runtime=self._latest_runtime(trace),
        )

    def _log_runtime_step(self, state, runtime_result, action_text: str) -> None:
        metadata = dict(runtime_result.runtime_metadata)
        executor_command = runtime_result.executor_command or {}
        if "runtime_executor" in executor_command:
            metadata["runtime_executor"] = executor_command["runtime_executor"]
        metadata["runtime_status"] = "completed" if runtime_result.ok else "failed"
        if runtime_result.error:
            metadata["error_type"] = "openclaw_runtime_error"

        self.components["logger"].log_step(
            state,
            intent=metadata.get("planned_intent", "act"),
            skill=metadata.get("planned_tool", ""),
            reason=metadata.get("planner_reason", runtime_result.error),
            memory_backend=self.components["config"].memory_backend,
            memory_source=self.components["config"].memory_source,
            action_text=action_text,
            fallback=not runtime_result.ok,
            decision_inputs={},
            runtime=metadata,
        )

    def _latest_runtime(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        runtime_items = trace.get("skill_runtime") or []
        if not runtime_items:
            return {}
        return dict(runtime_items[-1])


class _InertQwenDirectBackbone:
    def __init__(self) -> None:
        self.past_key_values_vggt = None
        self.config = SimpleNamespace()


class QwenDirectPolicyProxy:
    def __init__(self, components: Dict[str, Any]) -> None:
        if components.get("openclaw_runtime") is None:
            raise ValueError("qwen_direct policy backend requires openclaw_runtime")
        args = components.get("args")
        self.save_video = (
            bool(getattr(args, "save_video", False))
            and bool(getattr(args, "harness_stream_video", False))
            if args is not None
            else False
        )
        self.save_video_ratio = (
            float(getattr(args, "save_video_ratio", 0.0)) if args is not None else 0.0
        )
        self.processor = None
        self.tokenizer = None
        self.model = _InertQwenDirectBackbone()
        self.components = components
        self.last_action_text = None
        self.episode_invalid = False
        self.episode_invalid_reason = ""
        self.current_scene_id = ""
        self.current_episode_id = ""
        self._pending_env_state = None
        self._pending_map_context = None
        self._map_local_recovery_active = False
        self._episode_video_writer = None
        self._episode_save_video = False
        self._episode_video_disabled = False
        self._episode_video_frame_count = 0
        self.recent_keyframe_paths = []
        self._init_map_context_provider()

    start_episode = HarnessModelProxy.start_episode
    finalize_episode = HarnessModelProxy.finalize_episode
    _runtime_payload = HarnessModelProxy._runtime_payload
    _save_keyframe_if_needed = HarnessModelProxy._save_keyframe_if_needed
    _save_current_image_if_needed = HarnessModelProxy._save_current_image_if_needed
    _save_image_artifact = HarnessModelProxy._save_image_artifact
    _keyframe_target_path = HarnessModelProxy._keyframe_target_path
    _image_artifact_path = HarnessModelProxy._image_artifact_path
    _episode_artifact_dir = HarnessModelProxy._episode_artifact_dir
    _output_path = HarnessModelProxy._output_path
    _episode_video_path = HarnessModelProxy._episode_video_path
    _episode_video_tmp_path = HarnessModelProxy._episode_video_tmp_path
    _record_video_frame = HarnessModelProxy._record_video_frame
    _to_video_frame_bgr = HarnessModelProxy._to_video_frame_bgr
    _to_rgb_frame = HarnessModelProxy._to_rgb_frame
    _init_video_writer = HarnessModelProxy._init_video_writer
    _release_video_writer = HarnessModelProxy._release_video_writer
    _safe_path_part = HarnessModelProxy._safe_path_part
    observe_environment_state = HarnessModelProxy.observe_environment_state
    _remember_keyframe_path = HarnessModelProxy._remember_keyframe_path
    _remember_promoted_keyframe_from_runtime = (
        HarnessModelProxy._remember_promoted_keyframe_from_runtime
    )
    _attach_structured_runtime_context = (
        HarnessModelProxy._attach_structured_runtime_context
    )
    _init_map_context_provider = HarnessModelProxy._init_map_context_provider
    _reset_map_context_provider = HarnessModelProxy._reset_map_context_provider
    _update_pending_map_context = HarnessModelProxy._update_pending_map_context
    _attach_map_context_to_payload = HarnessModelProxy._attach_map_context_to_payload
    _remember_map_recovery_from_runtime = (
        HarnessModelProxy._remember_map_recovery_from_runtime
    )
    _build_proxy_state = HarnessModelProxy._build_proxy_state
    _proxy_safe_diagnostics = HarnessModelProxy._proxy_safe_diagnostics
    _append_working_memory = HarnessModelProxy._append_working_memory
    _latest_runtime = HarnessModelProxy._latest_runtime

    def call_model(self, images, task, step_id):
        current_image = images[-1] if images else None
        self._record_video_frame(current_image)
        state = self._build_proxy_state(task, step_id, current_image)
        runtime_result = self.components["openclaw_runtime"].step(
            state,
            self._runtime_payload(images, step_id),
        )
        runtime_metadata = dict(runtime_result.runtime_metadata or {})
        if runtime_metadata.get("episode_invalid"):
            self.episode_invalid = True
            self.episode_invalid_reason = str(
                runtime_metadata.get("qwen_failure_reason")
                or "qwen_direct_policy_abort"
            )
        action_text = runtime_result.action_text if runtime_result.ok else "STOP"
        self._remember_map_recovery_from_runtime(runtime_result)
        self._remember_promoted_keyframe_from_runtime(runtime_result)
        self.last_action_text = action_text
        self._append_working_memory(images, action_text)
        self._log_runtime_step(state, runtime_result, action_text)
        return [action_text]

    def consume_last_visual_prune_profile(self):
        return None

    def _log_runtime_step(self, state, runtime_result, action_text: str) -> None:
        metadata = dict(runtime_result.runtime_metadata)
        metadata.setdefault("policy_backend", QWEN_DIRECT_POLICY_BACKEND)
        metadata.setdefault("direct_policy", True)
        metadata.setdefault("janus_loaded", False)
        metadata.setdefault("navigation_policy_skill_called", False)
        executor_command = runtime_result.executor_command or {}
        if "runtime_executor" in executor_command:
            metadata["runtime_executor"] = executor_command["runtime_executor"]
        metadata["runtime_status"] = "completed" if runtime_result.ok else "failed"
        if runtime_result.error:
            metadata["error_type"] = "openclaw_runtime_error"

        self.components["logger"].log_step(
            state,
            intent=metadata.get("planned_intent", "act"),
            skill=metadata.get("planned_tool", ""),
            reason=metadata.get("planner_reason", runtime_result.error),
            memory_backend=self.components["config"].memory_backend,
            memory_source=self.components["config"].memory_source,
            action_text=action_text,
            fallback=not runtime_result.ok,
            decision_inputs={},
            runtime=metadata,
        )


def evaluate_harness(model: Any, args: argparse.Namespace) -> None:
    import json
    import os

    import torch
    import torch.distributed as dist

    import evaluation as eval_mod
    from utils.dist import get_rank, get_world_size

    policy_backend = getattr(args, "policy_backend", JANUS_POLICY_BACKEND)
    components = build_harness_components(args, model=model)
    if policy_backend == QWEN_DIRECT_POLICY_BACKEND:
        proxy_model = QwenDirectPolicyProxy(components)
    else:
        proxy_model = HarnessModelProxy(model, components)

    class HarnessVLNEvaluator(eval_mod.VLNEvaluator):
        def config_env(self):
            env = super().config_env()
            episode_keys = getattr(self.args, "harness_episode_keys", "")
            env.episodes = filter_harness_episodes_by_keys(env.episodes, episode_keys)
            max_episodes = getattr(self.args, "harness_debug_max_episodes", None)
            if max_episodes is not None:
                env.episodes = env.episodes[: int(max_episodes)]
            return env

    world_size = get_world_size()
    evaluator = HarnessVLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=proxy_model,
        epoch=0,
        args=args,
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank())
    if hasattr(proxy_model, "finalize_episode"):
        proxy_model.finalize_episode()

    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [
        torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device)
        for i in range(world_size)
    ]
    spls_all = [
        torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device)
        for i in range(world_size)
    ]
    oss_all = [
        torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device)
        for i in range(world_size)
    ]
    ones_all = [
        torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device)
        for i in range(world_size)
    ]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)

    result_all = {
        "sucs_all": (sum(sucs_all) / len(sucs_all)).item() if len(sucs_all) else 0.0,
        "spls_all": (sum(spls_all) / len(spls_all)).item() if len(spls_all) else 0.0,
        "oss_all": (sum(oss_all) / len(oss_all)).item() if len(oss_all) else 0.0,
        "ones_all": (sum(ones_all) / len(ones_all)).item() if len(ones_all) else 0.0,
        "length": len(sucs_all),
    }
    print(result_all)
    if get_rank() == 0:
        os.makedirs(args.output_path, exist_ok=True)
        with open(
            os.path.join(args.output_path, "summary.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(result_all, file)


def main() -> None:
    parser = build_parser()
    args = validate_args(parser.parse_args())
    # Heavy imports and model loading are intentionally delayed until main().
    import evaluation as eval_mod  # pylint: disable=import-outside-toplevel
    from utils.dist import (
        init_distributed_mode,
    )  # pylint: disable=import-outside-toplevel

    init_distributed_mode(args)
    eval_mod.max_pixels = args.max_pixels
    eval_mod.min_pixels = args.min_pixels
    model = None
    if args.policy_backend == JANUS_POLICY_BACKEND:
        from evaluation import (
            JanusVLN_Inference,
        )  # pylint: disable=import-outside-toplevel

        model = JanusVLN_Inference(
            args.model_path,
            device=f"cuda:{args.local_rank}",
            kv_start_size=args.kv_start_size,
            kv_recent_size=args.kv_recent_size,
        )
    evaluate_harness(model, args)


if __name__ == "__main__":
    main()
