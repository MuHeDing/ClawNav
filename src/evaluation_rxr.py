import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import cv2
import json
import random
import argparse
import itertools
import time
import quaternion
import transformers
import numpy as np
from dataclasses import dataclass

from typing import Any, Optional
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
    MeasurementConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from utils.dist import *
import base64
from datetime import datetime
from io import BytesIO
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from src.evaluation_rxr_metrics import (
    append_existing_episode_metrics,
    build_episode_result_record,
    build_summary_result_record,
    ensure_ndtw_measurement,
    is_summary_row,
)


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


@dataclass
class NDTWMeasurementConfig(MeasurementConfig):
    type: str = "NDTW"


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
        #os.makedirs(self.output_path, exist_ok=True)
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.save_video_ratio = args.save_video_ratio

        print('Dataset data_path:', self.config.habitat.dataset.data_path)

        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            # map_resolution=1024
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=4096,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
            ensure_ndtw_measurement(
                self.config.habitat.task.measurements,
                measurement_factory=NDTWMeasurementConfig,
            )

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
        
        #self.episode_counter = 0
        
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

    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def _log_visual_prune_profile(
        self,
        scene_id: str,
        episode_id: str,
        step_id: int,
        episode_instruction: str,
        profile: Optional[dict],
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

        
        sucs, spls, oss, ones, ndtws = [], [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    if is_summary_row(res):
                        continue
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        append_existing_episode_metrics(
                            res,
                            {
                                "sucs": sucs,
                                "spls": spls,
                                "oss": oss,
                                "ones": ones,
                                "ndtws": ndtws,
                            },
                        )
        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            # print(f"scene_id = {scene_id}")  scene_id = 2azQ1b91cZZ
            
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                #print("\nepisode start: ",episode_instruction) #Walk into the living room and keep walking straight past the living room. Then walk into the entrance under the balcony. Wait in the entrance to the other room.
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                
                # ✅ 每个episode开始时重置标志位
                self.current_episode_normalized = False

                env.current_episode = episode
                observations = env.reset()

                vis_frames = []
                step_id = 0

                # DEBUG: Print vis_frames initialization
                # debug_print_var("vis_frames (initialized)", vis_frames) 

                should_save_video = self.save_video and (random.random() < self.save_video_ratio)
                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}'), exist_ok=True)
                

                rgb_list = []
                time_ids = []
                action_seq = []

                # DEBUG: Print rgb_list initialization
                # debug_print_var("rgb_list (initialized)", rgb_list)

                self.model.model.past_key_values_vggt = None
                
                while not env.episode_over:
                    
                    time_ids.append(step_id)
                    rgb = observations["rgb"] #  每次执行环境步骤获得一帧
                    
                    #print('rgb: ', rgb.shape) # (480, 640, 3) 
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    rgb_list.append(image) 

                    # DEBUG: Print rgb_list after appending (only on first step)
                    #if step_id == 0:
                    #    debug_print_var("rgb_list (after first append)", rgb_list)

                    info = env.get_metrics()
                        
                    history_len = len(rgb_list) - 1 
                    
                    if history_len <= self.num_history:
                        history_images = rgb_list[:history_len]
                        images = history_images + [rgb_list[-1]]
                    else:
                        indices = np.linspace(0, history_len, self.num_history + 1, dtype=int)
                        images = [rgb_list[i] for i in indices]

                    # DEBUG: Print images after construction (only on first step)
                    #if step_id == 0:
                    #    debug_print_var("images (sampled history + current)", images)

                    # images # 通过采样固定在 9 张 self.num_history + 1
                    action = self.model.call_model(images, episode_instruction,step_id)[0]
                    self._log_visual_prune_profile(
                        scene_id=scene_id,
                        episode_id=episode_id,
                        step_id=step_id,
                        episode_instruction=episode_instruction,
                        profile=self.model.consume_last_visual_prune_profile(),
                    )
                    
                    # ✅ 如果action不在预定义列表中，进行归一化（只会在第一次时统计）
                    if action not in self.actions2idx:
                        action = self._normalize_action(action)
                    
                    if step_id % 50 == 0:  # 每3步清理，更激进地释放显存
                        torch.cuda.empty_cache()
                    
                    
                    if info['top_down_map'] is not None and should_save_video:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        vis_frames.append(frame)
                    
                    if action in self.actions2idx:
                        action = self.actions2idx[action][0]
                    else:
                        action = 0


                    if step_id >= self.args.max_steps:
                        action = 0

                    observations = env.step(action)
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()
                if should_save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9,
                        output_params=["-loglevel", "error"]
                    )
                vis_frames.clear()
                
                self.model.model.past_key_values_vggt = None
                
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                ndtws.append(metrics['ndtw'])
                
                # ✅ 打印时显示当前episode是否需要归一化
                
                #self.episode_counter += 1
                
                print(
                    f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, "
                    f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                    f"ne: {metrics['distance_to_goal']}, ndtw: {metrics['ndtw']}"
                )
                
                result = build_episode_result_record(
                    scene_id=scene_id,
                    episode_id=episode_id,
                    episode_instruction=episode_instruction,
                    metrics=metrics,
                    step_id=step_id,
                    action_normalized=self.current_episode_normalized,
                )
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")
                
        if self.visual_prune_profile_path is not None:
            summary_record = {
                "summary": True,
                **self.visual_prune_profile_summary,
            }
            with open(self.visual_prune_profile_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary_record) + "\n")
            print("visual_prune_profile_summary", summary_record)

        env.close()
        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(ones).to(self.device),
            torch.tensor(ndtws).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )




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
    parser.add_argument('--max_steps', default=400, type=int,
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
    sucs, spls, oss, ones, ndtws, ep_num = evaluator.eval_action(get_rank()) 
    
    
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    ndtws_all = [torch.zeros(ep_num_all[i], dtype=ndtws.dtype).to(ndtws.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.all_gather(ndtws_all, ndtws)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    ndtws_all = torch.cat(ndtws_all, dim=0)
    
    action_norm_stats_tensor = torch.tensor(evaluator.action_norm_stats).to(sucs.device)
    dist.all_reduce(action_norm_stats_tensor, op=dist.ReduceOp.SUM)
    total_action_norm_stats = action_norm_stats_tensor.item()
    
    result_all = build_summary_result_record(
        sucs=sucs_all,
        spls=spls_all,
        oss=oss_all,
        ones=ones_all,
        ndtws=ndtws_all,
        total_action_norm_stats=total_action_norm_stats,
    )
    
    print(result_all)
    
    
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()
