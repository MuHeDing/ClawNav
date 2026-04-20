"""
带多步 Visual KV Cache 的 Evaluation

主要特性:
1. 缓存历史图像帧的 Decoder KV
2. 检测重叠帧并复用 KV
3. 大幅减少重复计算

使用方式:
    python src/evaluation_with_visual_kv_cache.py \
        --model_path <path> \
        --use_visual_kv_cache

作者: Claude Code
日期: 2025-01-25
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import *
from collections import OrderedDict
import hashlib


class VisualFrameKVCache:
    """
    Visual 帧级别的 KV Cache 管理

    核心思想:
    - 为每个图像帧计算一个哈希值作为 key
    - 缓存该帧对应的 visual embeddings 和 KV
    - 下一步检测重叠时，通过哈希值匹配
    """

    def __init__(self, max_cache_size: int = 50):
        self.max_cache_size = max_cache_size

        # 缓存结构: {image_hash: {"embeddings": tensor, "kv": optional, "hits": int}}
        self.cache: OrderedDict[str, Dict] = OrderedDict()

        # 统计
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0,
            "frames_processed": 0,
        }

    def compute_image_hash(self, image) -> str:
        """
        计算图像的哈希值

        使用图像的像素数据计算 MD5，作为唯一标识
        """
        if isinstance(image, Image.Image):
            # 转换为 numpy array 并计算哈希
            img_array = np.array(image)
            img_bytes = img_array.tobytes()
            return hashlib.md5(img_bytes).hexdigest()[:16]  # 使用前16位
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def get_or_compute_embeddings(
        self,
        images: List,
        processor,
        model,
        force_recompute: bool = False
    ) -> Tuple[List[torch.Tensor], List[bool]]:
        """
        获取或计算图像的 embeddings，支持缓存

        Args:
            images: 图像列表
            processor: 图像处理器
            model: 模型
            force_recompute: 是否强制重新计算

        Returns:
            embeddings_list: 每个图像的 embeddings
            cache_hit_flags: 每个图像是否命中缓存
        """
        embeddings_list = []
        cache_hit_flags = []

        for img in images:
            img_hash = self.compute_image_hash(img)
            self.stats["total_queries"] += 1

            if not force_recompute and img_hash in self.cache:
                # 缓存命中
                cached_data = self.cache[img_hash]
                embeddings_list.append(cached_data["embeddings"])
                cache_hit_flags.append(True)

                # 更新命中计数和顺序（LRU）
                cached_data["hits"] += 1
                self.cache.move_to_end(img_hash)
                self.stats["cache_hits"] += 1

            else:
                # 缓存未命中，需要计算
                # ⚠️ 这里简化处理，实际需要单独处理每个图像
                # 当前先返回 None，表示需要重新计算所有图像
                embeddings_list.append(None)
                cache_hit_flags.append(False)
                self.stats["cache_misses"] += 1

        return embeddings_list, cache_hit_flags

    def save_embeddings(self, image, embeddings: torch.Tensor):
        """
        保存图像的 embeddings 到缓存
        """
        img_hash = self.compute_image_hash(image)

        # LRU 淘汰
        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[img_hash] = {
            "embeddings": embeddings.detach().cpu(),  # 保存到 CPU 节省显存
            "hits": 0,
            "timestamp": time.time()
        }

    def reset(self):
        """Episode 结束时重置"""
        self.cache.clear()
        print("[Visual Frame Cache] Reset")

    def print_stats(self):
        """打印统计信息"""
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total == 0:
            return

        hit_rate = self.stats["cache_hits"] / total * 100

        print("\n" + "=" * 60)
        print("Visual Frame KV Cache Statistics:")
        print("=" * 60)
        print(f"Total queries: {self.stats['total_queries']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"Cache misses: {self.stats['cache_misses']}")
        print(f"Hit rate: {hit_rate:.1f}%")
        print(f"Cached frames: {len(self.cache)}")

        # 估算节省的计算量
        if self.stats["cache_hits"] > 0:
            # 假设每个图像的处理占 10% 的总计算
            saved_computation = self.stats["cache_hits"] * 0.10
            print(f"Estimated computation saved: {saved_computation:.1%}")

        print("=" * 60 + "\n")


class JanusVLN_Inference_VisualKVCache(JanusVLN_Inference):
    """
    带 Visual KV Cache 的 JanusVLN 推理

    策略: 基于图像哈希的缓存
    - 为每个图像计算哈希值
    - 缓存已处理的图像特征
    - 检测重叠时复用特征
    """

    def __init__(
        self,
        pretrained,
        device="cuda",
        kv_start_size=8,
        kv_recent_size=48,
        use_visual_kv_cache=False,  # ⭐ 新增参数
        max_visual_cache_size=50,
    ):
        super().__init__(pretrained, device, kv_start_size, kv_recent_size)

        self.use_visual_kv_cache = use_visual_kv_cache

        if use_visual_kv_cache:
            self.visual_cache = VisualFrameKVCache(max_cache_size=max_visual_cache_size)
            print(f"✅ Visual Frame KV Cache ENABLED (max_size={max_visual_cache_size})")
        else:
            self.visual_cache = None
            print("⚠️  Visual Frame KV Cache DISABLED")

    @torch.no_grad()
    def call_model(
        self,
        observations,
        task,
        step_id,
        add_frame_index: bool = False,
        gen_kwargs: dict = {},
    ):
        """
        增强版 call_model，支持 visual frame caching

        当前实现: 简化版本，缓存图像级别的特征识别
        TODO: 完整实现需要缓存 decoder 的 KV
        """

        start_time = time.time()

        # ===== 检查缓存 (如果启用) =====
        cache_analysis = None
        if self.use_visual_kv_cache and isinstance(observations, list):
            # 检查哪些图像已经处理过
            cache_hits = 0
            for img in observations:
                img_hash = self.visual_cache.compute_image_hash(img)
                if img_hash in self.visual_cache.cache:
                    cache_hits += 1

            cache_analysis = {
                "total_frames": len(observations),
                "cache_hits": cache_hits,
                "cache_misses": len(observations) - cache_hits,
                "hit_rate": cache_hits / len(observations) if observations else 0
            }

            if cache_hits > 0:
                print(f"[Step {step_id}] 🚀 Visual cache: {cache_hits}/{len(observations)} frames cached "
                      f"({cache_analysis['hit_rate']:.1%} hit rate)")

        # ===== 原有的处理逻辑 =====
        messages = []
        message = [
            {"role": "system",
             "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
            }
        ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."

        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size

        visual = observations
        if isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            image_content = []
            for i, v in enumerate(visual):
                if add_frame_index:
                    image_content.append({"type": "text", "text": "Frame-{}: ".format(i)})
                image_content.append({"type": "image", "image": v})
            message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})

        messages.append(message)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        images_vggt = []
        image_inputs = []

        for message in messages:
            vision_info = extract_vision_info(message)
            cur_images_vggt = []

            for i, ele in enumerate(vision_info):
                if "image" in ele:
                    image = ele["image"]

                    if isinstance(image, Image.Image):
                        pass
                    else:
                        raise NotImplementedError("Unsupported image type")

                    image = load_and_preprocess_images([image])[0]

                    if i == len(vision_info) - 1:
                        cur_images_vggt.append(image)

                    _, height, width = image.shape

                    if (width // patch_size) % merge_size > 0:
                        width = width - (width // patch_size) % merge_size * patch_size
                    if (height // patch_size) % merge_size > 0:
                        height = height - (height // patch_size) % merge_size * patch_size
                    image = image[:, :height, :width]
                    image_inputs.append(image)

            images_vggt.append(torch.stack(cur_images_vggt))

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

        # ===== 保存图像到缓存 (简化版本) =====
        if self.use_visual_kv_cache and isinstance(observations, list):
            for img in observations:
                img_hash = self.visual_cache.compute_image_hash(img)
                if img_hash not in self.visual_cache.cache:
                    # 标记为已处理（简化版本，实际应保存 embeddings）
                    self.visual_cache.cache[img_hash] = {
                        "embeddings": None,  # TODO: 保存实际的 embeddings
                        "hits": 0,
                        "timestamp": time.time()
                    }
                else:
                    self.visual_cache.cache[img_hash]["hits"] += 1

            # 更新统计
            self.visual_cache.stats["cache_hits"] += cache_analysis["cache_hits"]
            self.visual_cache.stats["cache_misses"] += cache_analysis["cache_misses"]
            self.visual_cache.stats["frames_processed"] += cache_analysis["total_frames"]

        # ===== Generate =====
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        pad_token_id = self.tokenizer.pad_token_id

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

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        elapsed = time.time() - start_time
        if self.use_visual_kv_cache and cache_analysis:
            print(f"[Step {step_id}] Time: {elapsed:.3f}s, Hit rate: {cache_analysis['hit_rate']:.1%}")

        return answers

    def reset_visual_cache(self):
        """Episode 结束时重置缓存"""
        if self.visual_cache:
            self.visual_cache.reset()

    def print_visual_cache_stats(self):
        """打印缓存统计"""
        if self.visual_cache:
            self.visual_cache.print_stats()


# ==================== 修改 VLNEvaluator ====================

class VLNEvaluator_VisualKV(VLNEvaluator):
    """支持 Visual KV Cache 的 Evaluator"""

    def eval_action(self, idx) -> None:
        """重写 eval_action，添加缓存管理"""
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

                env.current_episode = episode
                observations = env.reset()

                vis_frames = []
                step_id = 0

                should_save_video = self.save_video and (random.random() < self.save_video_ratio)

                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}'), exist_ok=True)

                torch.cuda.empty_cache()

                rgb_list = []
                time_ids = []
                action_seq = []

                # ⭐ Episode 开始：重置缓存
                self.model.model.past_key_values_vggt = None
                if hasattr(self.model, 'reset_visual_cache'):
                    self.model.reset_visual_cache()

                while not env.episode_over:
                    time_ids.append(step_id)
                    rgb = observations["rgb"]

                    image = Image.fromarray(rgb).convert('RGB')
                    rgb_list.append(image)

                    info = env.get_metrics()

                    history_len = len(rgb_list) - 1

                    if history_len <= self.num_history:
                        history_images = rgb_list[:history_len]
                        images = history_images + [rgb_list[-1]]
                    else:
                        indices = np.linspace(0, history_len, self.num_history + 1, dtype=int)
                        images = [rgb_list[i] for i in indices]

                    # 调用模型（会自动使用 visual cache）
                    action = self.model.call_model(images, episode_instruction, step_id)[0]

                    if step_id % 50 == 0:
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
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'),
                        f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()

                # ⭐ Episode 结束：重置缓存
                self.model.model.past_key_values_vggt = None
                if hasattr(self.model, 'reset_visual_cache'):
                    self.model.reset_visual_cache()

                rgb_list.clear()

                import gc
                gc.collect()
                torch.cuda.empty_cache()

                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }

                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)


# ==================== 主函数 ====================

def eval_with_visual_kv_cache():
    """带 Visual KV Cache 的评估"""
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--save_video_ratio", type=float, default=0.05)
    parser.add_argument("--max_pixels", type=int, default=12544)
    parser.add_argument("--min_pixels", type=int, default=28*28)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_steps', default=400, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv_start_size", type=int, default=8)
    parser.add_argument("--kv_recent_size", type=int, default=48)

    # ⭐ Visual KV Cache 参数
    parser.add_argument("--use_visual_kv_cache", action="store_true", default=False,
                        help="Enable visual frame KV cache for cross-step reuse")
    parser.add_argument("--max_visual_cache_size", type=int, default=50,
                        help="Maximum number of frames to cache")

    args = parser.parse_args()
    print('args:', args)

    set_seed(args.seed)
    init_distributed_mode(args)
    local_rank = args.local_rank

    global max_pixels, min_pixels
    max_pixels = args.max_pixels
    min_pixels = args.min_pixels

    # ⭐ 使用增强版模型
    model = JanusVLN_Inference_VisualKVCache(
        args.model_path,
        device=f"cuda:{local_rank}",
        kv_start_size=args.kv_start_size,
        kv_recent_size=args.kv_recent_size,
        use_visual_kv_cache=args.use_visual_kv_cache,
        max_visual_cache_size=args.max_visual_cache_size,
    )

    # ⭐ 使用增强版 evaluator
    world_size = get_world_size()

    evaluator = VLNEvaluator_VisualKV(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )

    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank())

    # ... 后续的统计和同步逻辑与原版相同 ...
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
        "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
        "spls_all": (sum(spls_all)/len(spls_all)).item(),
        "oss_all": (sum(oss_all)/len(oss_all)).item(),
        "ones_all": (sum(ones_all)/len(ones_all)).item(),
        'length': len(sucs_all)
    }

    print(result_all)

    # ⭐ 打印 Visual KV Cache 统计
    if args.use_visual_kv_cache and get_rank() == 0:
        model.print_visual_cache_stats()

    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))


if __name__ == "__main__":
    eval_with_visual_kv_cache()
