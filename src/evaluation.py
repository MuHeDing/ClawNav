import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import cv2
import json
import gzip
import random
import argparse
import itertools
import time
import quaternion
import transformers
import numpy as np

from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat_extensions import sensors
from habitat_extensions import task
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from utils.dist import *
import base64
from datetime import datetime
from io import BytesIO
from evaluation_debug_utils import (
    append_record_to_json_array_file,
    build_episode_multi_goal_lookup,
    build_topdown_goal_display_settings,
    build_episode_qualitative_record,
    build_model_step_record,
    canonical_scene_id,
    multi_goal_overlay_pad_meters,
    normalize_vln_dataset_json_text,
    resolve_sanitized_vln_dataset_path,
    resolve_step_image_output_path,
    resolve_step_map_output_path,
)
from habitat_extensions import maps as habitat_extension_maps
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# 过滤通过 logging 模块发出的日志 # 过滤 imageio_ffmpeg 的 Python 日志
import logging
logging.getLogger('imageio').setLevel(logging.ERROR)
logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# Suppress imageio macro_block_size warnings from FFMPEG writer

import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
min_pixels: int = 28 * 28
max_pixels: int = 1605632


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ======================== DEBUG UTILITY ========================
def debug_print_var(var_name, var_value, indent=0):
    """
    Debug utility to print variable information.
    If tensor: print name, shape, dtype, device
    If not tensor: print name, type, length/content
    """
    prefix = "  " * indent
    separator = "=" * 60
    print(f"\n{prefix}{separator}")
    print(f"{prefix}[DEBUG] Variable: {var_name}")

    if isinstance(var_value, torch.Tensor):
        print(f"{prefix}  Type: torch.Tensor")
        print(f"{prefix}  Shape: {var_value.shape}")
        print(f"{prefix}  Dtype: {var_value.dtype}")
        print(f"{prefix}  Device: {var_value.device}")
    elif isinstance(var_value, (list, tuple)):
        print(f"{prefix}  Type: {type(var_value).__name__}")
        print(f"{prefix}  Length: {len(var_value)}")
        if len(var_value) > 0:
            first_elem = var_value[0]
            if isinstance(first_elem, torch.Tensor):
                print(f"{prefix}  First element type: torch.Tensor")
                print(f"{prefix}  First element shape: {first_elem.shape}")
                print(f"{prefix}  First element dtype: {first_elem.dtype}")
            elif isinstance(first_elem, Image.Image):
                print(f"{prefix}  First element type: PIL.Image")
                print(f"{prefix}  First element size: {first_elem.size}")
                print(f"{prefix}  First element mode: {first_elem.mode}")
            elif isinstance(first_elem, (list, dict)):
                print(f"{prefix}  First element type: {type(first_elem).__name__}")
                if isinstance(first_elem, list):
                    print(f"{prefix}  First element length: {len(first_elem)}")
                elif isinstance(first_elem, dict):
                    print(f"{prefix}  First element keys: {list(first_elem.keys())}")
            else:
                print(f"{prefix}  First element type: {type(first_elem).__name__}")
                print(f"{prefix}  First element: {first_elem}")
    elif isinstance(var_value, dict):
        print(f"{prefix}  Type: dict")
        print(f"{prefix}  Keys: {list(var_value.keys())}")
        for key in var_value.keys():
            if isinstance(var_value[key], torch.Tensor):
                print(f"{prefix}    '{key}': Tensor, shape={var_value[key].shape}")
            elif isinstance(var_value[key], (list, tuple)):
                print(f"{prefix}    '{key}': {type(var_value[key]).__name__}, len={len(var_value[key])}")
            else:
                print(f"{prefix}    '{key}': {type(var_value[key]).__name__}")
    elif isinstance(var_value, Image.Image):
        print(f"{prefix}  Type: PIL.Image")
        print(f"{prefix}  Size: {var_value.size}")
        print(f"{prefix}  Mode: {var_value.mode}")
    else:
        print(f"{prefix}  Type: {type(var_value).__name__}")
        if hasattr(var_value, '__len__') and not isinstance(var_value, str):
            print(f"{prefix}  Length: {len(var_value)}")
        print(f"{prefix}  Value: {var_value}")

    print(f"{prefix}{separator}\n")
# ======================== END DEBUG UTILITY ========================


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        raw_dataset_path = None

        if args is not None and getattr(args, "data_path", None) is not None:
            raw_dataset_path = args.data_path
        else:
            raw_dataset_path = self.config.habitat.dataset.data_path
        if isinstance(raw_dataset_path, str) and "{split}" in raw_dataset_path:
            raw_dataset_path = raw_dataset_path.format(split=self.split)
        self.episode_multi_goals = self._load_episode_multi_goals(raw_dataset_path)
        print(f"Loaded multi-goal overlays for {len(self.episode_multi_goals)} episodes")
        topdown_goal_display_settings = build_topdown_goal_display_settings(
            self.episode_multi_goals
        )

        rank = get_rank()
        with habitat.config.read_write(self.config):
            if args is not None and getattr(args, "data_path", None) is not None:
                self.config.habitat.dataset.data_path = args.data_path
            dataset_data_path = self._prepare_dataset_path(
                self.config.habitat.dataset.data_path,
                rank,
            )
            self.config.habitat.dataset.data_path = dataset_data_path
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=2048,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=False,
                        draw_view_points=True,
                        draw_goal_positions=topdown_goal_display_settings[
                            "draw_goal_positions"
                        ],
                        draw_goal_aabbs=topdown_goal_display_settings[
                            "draw_goal_aabbs"
                        ],
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print('Dataset data_path:', self.config.habitat.dataset.data_path)

        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.save_video_ratio = args.save_video_ratio

        self.image_processor = model.processor
        self.model = model
        self.tokenizer = model.tokenizer
        self.visual_prune_profile_path = None
        self.visual_prune_profile_summary = {
            "profiled_calls": 0,
            "forward_time_ms_total": 0.0,
            "generate_time_ms_total": 0.0,
            "peak_memory_allocated_mb_max": 0.0,
            "visual_before_total": 0,
            "visual_after_total": 0,
        }
        if not getattr(args, "disable_visual_prune_eval_profile", False):
            self.visual_prune_profile_path = os.path.join(
                self.output_path,
                f"visual_prune_eval_profile_rank{get_rank()}.jsonl",
            )
        
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "MOVE_FORWARD": [1],
            "TURN_LEFT": [2],
            "TURN_RIGHT": [3]
        })

        self.num_history = args.num_history
        print(f"Using num_history = {self.num_history}")
        
        self.action_norm_stats = 0
        
        # ✅ 用于标记当前episode是否已经统计过
        self.current_episode_normalized = False 
        
        if getattr(args, "disable_qualitative_json", False):
            self.qualitative_output_path = None
            print("Qualitative trajectory dump disabled")
        else:
            self.qualitative_output_path = os.path.join(
                self.output_path,
                f"qualitative_trajectories_rank{rank}.json",
            )
            print(f"Qualitative trajectory dump: {self.qualitative_output_path}")

    def _load_episode_multi_goals(self, dataset_path: Optional[str]) -> Dict[tuple, List[List[float]]]:
        if not dataset_path or not os.path.exists(dataset_path):
            return {}

        if dataset_path.endswith(".json.gz"):
            with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
                raw = f.read()
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw = f.read()
        return build_episode_multi_goal_lookup(raw)

    def _prepare_dataset_path(self, dataset_path: str, rank: int) -> str:
        if not dataset_path or not os.path.exists(dataset_path):
            return dataset_path

        requires_gzip_cache = dataset_path.endswith(".json") and not dataset_path.endswith(".json.gz")
        if dataset_path.endswith(".json.gz"):
            with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
                raw = f.read()
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw = f.read()

        cleaned = None
        fix_reason = None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            cleaned = normalize_vln_dataset_json_text(raw)
            json.loads(cleaned)
            fix_reason = str(exc)
        else:
            if (
                not isinstance(payload.get("instruction_vocab"), dict)
                or "word_list" not in payload["instruction_vocab"]
            ):
                cleaned = normalize_vln_dataset_json_text(raw)
                json.loads(cleaned)
                fix_reason = "missing instruction_vocab.word_list"
            elif requires_gzip_cache:
                cleaned = raw if raw.endswith("\n") else raw + "\n"
                fix_reason = "plain json input cached as json.gz for Habitat loader"
            else:
                return dataset_path

        if cleaned is None:
            return dataset_path

        sanitized_path = resolve_sanitized_vln_dataset_path(
            output_path=self.output_path,
            dataset_path=dataset_path,
            rank=rank,
        )
        with gzip.open(sanitized_path, "wt", encoding="utf-8") as f:
            f.write(cleaned)

        print(
            "Sanitized malformed dataset JSON:",
            dataset_path,
            f"(reason: {fix_reason}) -> {sanitized_path}",
        )
        return str(sanitized_path)

    def _current_agent_position(self, env: Env) -> Optional[List[float]]:
        state = env.sim.get_agent_state()
        position = getattr(state, "position", None)
        if position is None:
            return None
        return [float(v) for v in position]

    def _current_agent_rotation(self, env: Env) -> Optional[List[float]]:
        state = env.sim.get_agent_state()
        rotation = getattr(state, "rotation", None)
        if rotation is None:
            return None
        try:
            return quaternion.as_float_array(rotation).tolist()
        except Exception:
            if all(hasattr(rotation, attr) for attr in ("x", "y", "z", "w")):
                return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]
            return [float(v) for v in rotation]
        
    def _normalize_action(self, action: str) -> str:
        """Map model free-form text to one of the valid actions."""
        

        # ✅ 只在当前episode第一次归一化时计数
        if not self.current_episode_normalized:
            self.action_norm_stats += 1
            self.current_episode_normalized = True

        act = action.upper().replace("-", "_")
        # Strip punctuation/line breaks and keep keywords
        act = re.sub(r"[^A-Z_ ]+", " ", act).strip()

        if "FORWARD" in act:
            return "MOVE_FORWARD"
        if "LEFT" in act:
            return "TURN_LEFT"
        if "RIGHT" in act:
            return "TURN_RIGHT"
        if "STOP" in act:
            return "STOP"

        return act

    def _episode_multi_goal_positions(self, episode: Any) -> List[List[float]]:
        return self.episode_multi_goals.get(
            (canonical_scene_id(getattr(episode, "scene_id", "")), str(getattr(episode, "episode_id", ""))),
            [],
        )

    def _overlay_extra_goals_on_info(
        self,
        info: Dict[str, Any],
        env: Env,
        extra_goal_positions: List[List[float]],
    ) -> Dict[str, Any]:
        if not extra_goal_positions or info.get("top_down_map") is None:
            return info

        frame_info = dict(info)
        frame_info["top_down_map"] = dict(info["top_down_map"])
        frame_info["top_down_map"]["map"] = np.array(info["top_down_map"]["map"], copy=True)
        top_down_map = frame_info["top_down_map"]["map"]
        lower_bound, upper_bound = env.sim.pathfinder.get_bounds()
        bounds = {
            "lower": tuple(float(v) for v in lower_bound),
            "upper": tuple(float(v) for v in upper_bound),
        }
        meters_per_px = habitat_maps.calculate_meters_per_pixel(
            top_down_map.shape[0],
            sim=env.sim,
        )

        for goal_xyz in extra_goal_positions:
            g_x, g_y = habitat_extension_maps.static_to_grid(
                goal_xyz[2],
                goal_xyz[0],
                top_down_map.shape[0:2],
                bounds,
            )
            if 0 < g_x < top_down_map.shape[0] and 0 < g_y < top_down_map.shape[1]:
                habitat_extension_maps.drawpoint(
                    top_down_map,
                    (g_x, g_y),
                    habitat_extension_maps.MAP_TARGET_POINT_INDICATOR,
                    meters_per_px,
                    pad=multi_goal_overlay_pad_meters(),
                )

        return frame_info

    def _render_step_map(self, frame_info: Dict[str, Any], output_height: int) -> Optional[np.ndarray]:
        top_down_map_info = frame_info.get("top_down_map")
        if top_down_map_info is None:
            return None
        return habitat_maps.colorize_draw_agent_and_fit_to_height(
            top_down_map_info,
            output_height,
        )

    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def _log_visual_prune_profile(
        self,
        scene_id: str,
        episode_id: str,
        step_id: int,
        episode_instruction: str,
        profile: Optional[Dict[str, Any]],
    ) -> None:
        if profile is None or self.visual_prune_profile_path is None:
            return

        prune_info = profile.get("prune", {})
        sample_stats = prune_info.get("samples", [])
        visual_before = sum(sample.get("visual_before", 0) for sample in sample_stats)
        visual_after = sum(sample.get("visual_after", 0) for sample in sample_stats)
        peak_memory_mb = profile.get("peak_memory_allocated_mb")

        self.visual_prune_profile_summary["profiled_calls"] += 1
        self.visual_prune_profile_summary["forward_time_ms_total"] += float(profile.get("forward_time_ms", 0.0))
        self.visual_prune_profile_summary["generate_time_ms_total"] += float(profile.get("generate_time_ms", 0.0))
        self.visual_prune_profile_summary["visual_before_total"] += int(visual_before)
        self.visual_prune_profile_summary["visual_after_total"] += int(visual_after)
        if peak_memory_mb is not None:
            self.visual_prune_profile_summary["peak_memory_allocated_mb_max"] = max(
                self.visual_prune_profile_summary["peak_memory_allocated_mb_max"],
                float(peak_memory_mb),
            )

        record = {
            "scene_id": scene_id,
            "episode_id": str(episode_id),
            "step_id": int(step_id),
            "episode_instruction": episode_instruction,
            "profile": profile,
        }
        with open(self.visual_prune_profile_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            "visual_prune_profile",
            f"scene_episode={scene_id}_{episode_id}",
            f"step={step_id}",
            f"visual={visual_before}->{visual_after}",
            f"forward_ms={profile.get('forward_time_ms')}",
            f"generate_ms={profile.get('generate_time_ms')}",
            f"peak_mem_mb={peak_memory_mb}",
        )

    

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]

            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                
                self.current_episode_normalized = False

                env.current_episode = episode
                observations = env.reset()

                vis_frames = []
                step_id = 0

                should_save_video = self.save_video and (random.random() < self.save_video_ratio)
                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}'), exist_ok=True)

                episode_multi_goals = self._episode_multi_goal_positions(episode)
                save_step_artifacts = (
                    should_save_video
                    if getattr(self.args, "save_step_artifacts_with_video_only", False)
                    else True
                )
                if save_step_artifacts:
                    episode_step_image_dir = resolve_step_image_output_path(
                        output_path=self.output_path,
                        scene_id=scene_id,
                        episode_id=episode_id,
                        step_id=0,
                    ).parent
                    episode_step_image_dir.mkdir(parents=True, exist_ok=True)
                    episode_step_map_dir = resolve_step_map_output_path(
                        output_path=self.output_path,
                        scene_id=scene_id,
                        episode_id=episode_id,
                        step_id=0,
                    ).parent
                    episode_step_map_dir.mkdir(parents=True, exist_ok=True)

                rgb_list = []
                model_path = [] if self.qualitative_output_path else None

                self.model.model.past_key_values_vggt = None
                
                while not env.episode_over:
                    rgb = observations["rgb"]

                    image = Image.fromarray(rgb).convert('RGB')
                    if save_step_artifacts:
                        step_image_path = resolve_step_image_output_path(
                            output_path=self.output_path,
                            scene_id=scene_id,
                            episode_id=episode_id,
                            step_id=step_id,
                        )
                        image.save(step_image_path)
                    rgb_list.append(image)

                    info = env.get_metrics()
                    frame_info = self._overlay_extra_goals_on_info(info, env, episode_multi_goals)
                    step_map = self._render_step_map(frame_info, rgb.shape[0]) if save_step_artifacts else None
                    if step_map is not None:
                        step_map_path = resolve_step_map_output_path(
                            output_path=self.output_path,
                            scene_id=scene_id,
                            episode_id=episode_id,
                            step_id=step_id,
                        )
                        Image.fromarray(step_map).save(step_map_path)

                    history_len = len(rgb_list) - 1 
                    
                    if history_len <= self.num_history:
                        history_images = rgb_list[:history_len]
                        images = history_images + [rgb_list[-1]]
                    else:
                        indices = np.linspace(0, history_len, self.num_history + 1, dtype=int)
                        images = [rgb_list[i] for i in indices]

                    action = self.model.call_model(images, episode_instruction, step_id)[0]
                    self._log_visual_prune_profile(
                        scene_id=scene_id,
                        episode_id=episode_id,
                        step_id=step_id,
                        episode_instruction=episode_instruction,
                        profile=self.model.consume_last_visual_prune_profile(),
                    )
                    action_text = action
                    
                    if action not in self.actions2idx:
                        action = self._normalize_action(action)
                    action_text = action
                    
                    if step_id % 50 == 0:  # 每3步清理，更激进地释放显存
                        torch.cuda.empty_cache()
                    
                    if frame_info.get('top_down_map') is not None and should_save_video:
                        frame = observations_to_image({'rgb':observations['rgb']}, frame_info)
                        vis_frames.append(frame)
                    
                    if action in self.actions2idx:
                        action = self.actions2idx[action][0]
                    else:
                        action = 0


                    if step_id >= self.args.max_steps:
                        action = 0
                        action_text = "STOP"

                    observations = env.step(action)
                    step_metrics = env.get_metrics()
                    if model_path is not None:
                        model_path.append(
                            build_model_step_record(
                                step_id=step_id,
                                action_text=action_text,
                                action_id=action,
                                position=self._current_agent_position(env),
                                rotation=self._current_agent_rotation(env),
                                distance_to_goal=step_metrics.get("distance_to_goal"),
                                episode_over=env.episode_over,
                            )
                        )
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()
                if should_save_video:
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, f'vis_{self.epoch}'),
                        f'{scene_id}_{episode_id}',
                        fps=6,
                        quality=9,
                        output_params=["-loglevel", "error"],
                    )
                vis_frames.clear()
                
                self.model.model.past_key_values_vggt = None
                rgb_list.clear()
                
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                    "action_normalized": self.current_episode_normalized
                }
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

                if self.qualitative_output_path is not None:
                    qualitative_record = build_episode_qualitative_record(
                        episode=episode,
                        scene_id=scene_id,
                        episode_instruction=episode_instruction,
                        model_path=model_path,
                    )
                    append_record_to_json_array_file(
                        self.qualitative_output_path,
                        qualitative_record,
                    )

        if self.visual_prune_profile_path is not None:
            summary_record = {
                "summary": True,
                **self.visual_prune_profile_summary,
            }
            with open(self.visual_prune_profile_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary_record) + "\n")
            print("visual_prune_profile_summary", summary_record)

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)     




class JanusVLN_Inference:
    def __init__(self, pretrained, device="cuda",kv_start_size=4, kv_recent_size=24):
        config = AutoConfig.from_pretrained(pretrained)
        
        config.kv_start_size = kv_start_size
        config.kv_recent_size = kv_recent_size
        config.enable_visual_prune_eval_profile = True
        config.allow_cache_prefill_visual_prune = False
        
        
        self.model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            mode='evaluation'
        ).eval()
        
        
        print(f"JanusVLN_Inference Using max_pixels={max_pixels}, min_pixels={min_pixels} for memory optimization")
        print(f"JanusVLN_Inference VGGT KV Cache: start_size={kv_start_size}, recent_size={kv_recent_size}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        
        self.device = device
        self.last_visual_prune_profile = None

    def consume_last_visual_prune_profile(self):
        profile = self.last_visual_prune_profile
        self.last_visual_prune_profile = None
        return profile

    @torch.no_grad()
    def call_model(
        self,
        observations, 
        task,
        step_id,
        add_frame_index: bool=False,
        gen_kwargs: dict = {},
    ):
        
        messages = []
        message = [
                {"role": "system", 
                "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
                }
            ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        
        #print('task: ', task)  instructions
        for i in enumerate([task]):

            visual = observations
            # DEBUG: Print observations
            #debug_print_var("visual (observations)", visual)

            if isinstance(visual, Image.Image): 
                message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
            elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  
                image_content = []
                image_count = 0
                for v in visual:
                    if add_frame_index:
                        image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                    image_content.append({"type": "image", "image": v})
                    image_count += 1
                message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            # DEBUG: Print message
            #print("message", message)

            messages.append(message)

        # DEBUG: Print messages
        #print("messages", messages)


        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #print('text: ', text)
        images_vggt = []
        image_inputs = []
        for message in messages:
            vision_info = extract_vision_info(message)
            #debug_print_var("vision_info: ", vision_info)
            #debug_print_var("vision_info[0]: ", vision_info[0])
            #print(f'vision_info[0]: {vision_info[0]}')
            cur_images_vggt = []
            for i, ele in enumerate(vision_info):
                if "image" in ele:
                    image = ele["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")   
                else:
                    raise NotImplementedError("Unsupported vision info type")
    
                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image = load_and_preprocess_images([image])[0]

                if i == len(vision_info) - 1:
                    # 确保只有最新的当前帧进入 VGGT。
                    cur_images_vggt.append(image)
                    # DEBUG: Print cur_images_vggt after adding current frame
                    #debug_print_var("cur_images_vggt (after adding current frame)", cur_images_vggt)

                _, height, width = image.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image = image[:, :height, :width]
                image_inputs.append(image)
            
            images_vggt.append(torch.stack(cur_images_vggt)) 

        # DEBUG: Print final images_vggt and image_inputs
        #debug_print_var("images_vggt", images_vggt) #list 一直是 ([1, 3, 490, 644])
        #debug_print_var("image_inputs", image_inputs) # list 图像一直增加

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device

        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        inputs = inputs.to(device)
        if getattr(self.model.config, "use_llm_visual_prune", False) and getattr(
            self.model.config, "enable_visual_prune_eval_profile", False
        ):
            _ = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            self.last_visual_prune_profile = self.model.consume_visual_prune_profile()
        else:
            self.last_visual_prune_profile = None
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        
        
        pad_token_id = self.tokenizer.pad_token_id
        generate_start_time = time.perf_counter()
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )
        generate_time_ms = round((time.perf_counter() - generate_start_time) * 1000.0, 3)
        if self.last_visual_prune_profile is not None:
            self.last_visual_prune_profile["generate_time_ms"] = generate_time_ms

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return answers




   
def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument("--save_video_ratio", type=float, default=0.05, help="0~1")
    parser.add_argument(
        "--save_step_artifacts_with_video_only",
        action="store_true",
        default=False,
        help="Only save step_images and step_maps for episodes selected by --save_video",
    )
    parser.add_argument(
        "--disable_qualitative_json",
        action="store_true",
        default=False,
        help="Disable qualitative_trajectories_rank*.json output",
    )
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override dataset.data_path in the Habitat config")
    
    parser.add_argument("--max_pixels", type=int, default=401408,
                        help="Maximum pixels for image processing. Lower = less memory. Try: 12544->6272->3136->1568")
    parser.add_argument("--min_pixels", type=int, default=28*28,
                        help="Minimum pixels for image processing")
    
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--max_steps', default=450, type=int,
                        help='max_steps')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--kv_start_size", type=int, default=8,
                        help="VGGT KV cache start size (keep first N frames)")
    parser.add_argument("--kv_recent_size", type=int, default=24,
                        help="VGGT KV cache recent size (keep last N frames)")
    parser.add_argument("--disable_visual_prune_eval_profile", action="store_true", default=False)
    
    args = parser.parse_args()
    set_seed(args.seed)
    init_distributed_mode(args)
    local_rank = args.local_rank
    
        # 使用命令行参数覆盖全局 max_pixels/min_pixels
    global max_pixels, min_pixels
    max_pixels = args.max_pixels
    min_pixels = args.min_pixels
    print(f"Using max_pixels={max_pixels}, min_pixels={min_pixels} for memory optimization")
    print(f"Using max_steps={args.max_steps} for evaluation rollout limit")

    model = JanusVLN_Inference(
        args.model_path, 
        device=f"cuda:{local_rank}",
        kv_start_size=args.kv_start_size,
        kv_recent_size=args.kv_recent_size)
    model.model.config.enable_visual_prune_eval_profile = not args.disable_visual_prune_eval_profile

    evaluate(model, args)



def evaluate(model, args):
    
    world_size = get_world_size()

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
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
    
    action_norm_stats_tensor = torch.tensor(evaluator.action_norm_stats).to(sucs.device)
    dist.all_reduce(action_norm_stats_tensor, op=dist.ReduceOp.SUM)
    total_action_norm_stats = action_norm_stats_tensor.item()
    
    total_episodes = len(sucs_all)
    
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all),
                    'episodes_need_normalization': total_action_norm_stats,  # ✅ 需要归一化的episode数
                    'normalization_ratio': total_action_norm_stats / total_episodes if total_episodes > 0 else 0  # ✅ 归一化比例
                }
    
    print(result_all)
    
    
    if get_rank() == 0:
        with open(os.path.join(args.output_path, 'summary.json'), 'w') as f:
            json.dump(result_all, f)

if __name__ == "__main__":
    eval()
