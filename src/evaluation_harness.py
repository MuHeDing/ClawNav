import argparse
from pathlib import Path
from typing import Any, Dict

from harness.config import HarnessConfig
from harness.controller import HarnessController
from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.logging.harness_logger import HarnessLogger
from harness.memory.memory_manager import MemoryManager
from harness.memory.spatial_memory_client import (
    FakeSpatialMemoryClient,
    SpatialMemoryHttpClient,
)
from harness.memory.task_memory import TaskMemory
from harness.memory.working_memory import WorkingMemory
from harness.skill_registry import SkillRegistry
from harness.skills.memory_query import MemoryQuerySkill
from harness.skills.memory_write import MemoryWriteSkill
from harness.skills.navigation_policy import NavigationPolicySkill
from harness.skills.progress_critic import ProgressCriticSkill
from harness.skills.replanner import ReplannerSkill
from harness.types import SkillResult


ACTIONS2IDX = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ClawNav OpenClaw-style harness evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--min_pixels", type=int, default=28 * 28)
    parser.add_argument("--kv_start_size", type=int, default=8)
    parser.add_argument("--kv_recent_size", type=int, default=24)
    parser.add_argument("--max_steps", type=int, default=400)
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
    parser.add_argument("--save_step_artifacts", action="store_true", default=False)
    parser.add_argument("--save_step_artifacts_with_video_only", action="store_true", default=False)
    parser.add_argument("--disable_qualitative_json", action="store_true", default=False)
    parser.add_argument("--harness_mode", type=str, default="memory_recall")
    parser.add_argument("--harness_memory_backend", type=str, default="fake")
    parser.add_argument("--spatial_memory_url", type=str, default="http://127.0.0.1:8022")
    parser.add_argument("--harness_memory_source", type=str, default="episode-local")
    parser.add_argument("--harness_max_internal_calls", type=int, default=3)
    parser.add_argument("--harness_recall_interval_steps", type=int, default=5)
    parser.add_argument("--harness_debug_max_episodes", type=int, default=None)
    parser.add_argument("--harness_trace_rank", type=int, default=0)
    parser.add_argument("--expose_sim_pose_online", action="store_true", default=False)
    return parser


def build_harness_config(args: argparse.Namespace) -> HarnessConfig:
    return HarnessConfig(
        harness_mode=args.harness_mode,
        memory_backend=args.harness_memory_backend,
        spatial_memory_url=args.spatial_memory_url,
        max_internal_calls_per_step=args.harness_max_internal_calls,
        recall_interval_steps=args.harness_recall_interval_steps,
        memory_source=args.harness_memory_source,
        expose_sim_pose_online=args.expose_sim_pose_online,
    )


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
    config = build_harness_config(args)
    memory_client = build_memory_client(config)
    memory_manager = MemoryManager(memory_client, config)
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

    controller = HarnessController(registry, config)
    adapter = HabitatVLNAdapter(expose_pose_online=config.expose_sim_pose_online)
    logger = HarnessLogger(
        Path(args.output_path) / "harness_traces",
        rank=args.harness_trace_rank,
    )

    return {
        "config": config,
        "memory_client": memory_client,
        "memory_manager": memory_manager,
        "task_memory": task_memory,
        "working_memory": working_memory,
        "skill_registry": registry,
        "controller": controller,
        "adapter": adapter,
        "logger": logger,
    }


class HarnessModelProxy:
    def __init__(self, base_model: Any, components: Dict[str, Any]) -> None:
        self.base_model = base_model
        self.model = base_model.model
        self.components = components
        self.last_action_text = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    def call_model(self, images, task, step_id):
        current_image = images[-1] if images else None
        state = self._build_proxy_state(task, step_id, current_image)
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

    def consume_last_visual_prune_profile(self):
        return self.base_model.consume_last_visual_prune_profile()

    def _build_proxy_state(self, task: str, step_id: int, current_image: Any):
        from harness.types import VLNState

        return VLNState(
            scene_id="",
            episode_id="",
            instruction=task,
            step_id=step_id,
            current_image=current_image,
            online_metrics={},
            diagnostics={},
            last_action=self.last_action_text,
        )

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
        )


def evaluate_harness(model: Any, args: argparse.Namespace) -> None:
    import json
    import os

    import torch
    import torch.distributed as dist

    import evaluation as eval_mod
    from utils.dist import get_rank, get_world_size

    components = build_harness_components(args, model=model)
    proxy_model = HarnessModelProxy(model, components)

    class HarnessVLNEvaluator(eval_mod.VLNEvaluator):
        def config_env(self):
            env = super().config_env()
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

    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
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
        with open(os.path.join(args.output_path, "summary.json"), "w", encoding="utf-8") as file:
            json.dump(result_all, file)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Heavy imports and model loading are intentionally delayed until main().
    import evaluation as eval_mod  # pylint: disable=import-outside-toplevel
    from evaluation import JanusVLN_Inference  # pylint: disable=import-outside-toplevel
    from utils.dist import init_distributed_mode  # pylint: disable=import-outside-toplevel

    init_distributed_mode(args)
    eval_mod.max_pixels = args.max_pixels
    eval_mod.min_pixels = args.min_pixels
    model = JanusVLN_Inference(
        args.model_path,
        device=f"cuda:{args.local_rank}",
        kv_start_size=args.kv_start_size,
        kv_recent_size=args.kv_recent_size,
    )
    evaluate_harness(model, args)


if __name__ == "__main__":
    main()
