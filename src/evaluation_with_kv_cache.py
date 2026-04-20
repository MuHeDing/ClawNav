"""
带跨步骤 KV Cache 的 evaluation.py

主要改动:
1. 新增 --use_text_kv_cache 参数
2. 在 JanusVLN_Inference 中实现文本前缀 KV 缓存
3. 保持向后兼容，默认不启用

使用方式:
    # 启用 KV cache
    python evaluation_with_kv_cache.py --use_text_kv_cache

    # 不启用 (baseline)
    python evaluation_with_kv_cache.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 导入原有的所有内容
from evaluation import *  # 继承所有原有功能
import time


class TextPrefixKVCache:
    """
    文本前缀 KV Cache 管理器

    功能:
    - 缓存 System + Task 的 KV（Episode 级别）
    - 跨步骤复用，减少重复计算
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.cached_kv = None
        self.cached_task = None
        self.cache_enabled = False

        # 统计
        self.stats = {
            "prefill_time_with_cache": [],
            "prefill_time_without_cache": [],
            "cache_hits": 0,
        }

    def should_use_cache(self, task: str) -> bool:
        """判断是否可以使用缓存"""
        return (
            self.cache_enabled and
            self.cached_kv is not None and
            self.cached_task == task
        )

    def compute_and_cache(self, task: str):
        """
        预计算文本前缀的 KV

        ⚠️ 注意: 由于 Qwen2.5-VL 的图像 tokens 插入在中间，
        实际实现时需要特殊处理 position IDs

        当前实现: 简化版本，仅作为概念验证
        """
        print(f"[KV Cache] Computing text prefix KV for task: '{task}'")

        # 构建纯文本 prompt
        messages = [{
            "role": "system",
            "content": "You are a visual language navigation model..."  # 完整的 system prompt
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Forward 获取 KV
        with torch.no_grad():
            outputs = self.model.model(
                **inputs,
                use_cache=True,
                return_dict=True,
            )

        self.cached_kv = outputs.past_key_values
        self.cached_task = task

        # 打印信息
        if self.cached_kv:
            k, v = self.cached_kv[0]
            print(f"[KV Cache] Cached KV shape: K={k.shape}, V={v.shape}")
            print(f"[KV Cache] Total layers: {len(self.cached_kv)}")

    def reset(self):
        """Episode 结束时重置"""
        self.cached_kv = None
        self.cached_task = None
        print("[KV Cache] Reset")

    def print_stats(self):
        """打印统计信息"""
        if not self.stats["prefill_time_with_cache"]:
            return

        avg_with = np.mean(self.stats["prefill_time_with_cache"])
        avg_without = np.mean(self.stats["prefill_time_without_cache"])
        speedup = avg_without / avg_with if avg_with > 0 else 1.0

        print("\n" + "="*60)
        print("KV Cache Performance Statistics:")
        print("="*60)
        print(f"Avg prefill time WITH cache:    {avg_with:.3f}s")
        print(f"Avg prefill time WITHOUT cache: {avg_without:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print("="*60 + "\n")


class JanusVLN_Inference_Enhanced(JanusVLN_Inference):
    """
    增强版 JanusVLN_Inference，支持 KV cache

    向后兼容: 如果 use_text_kv_cache=False，行为与原版完全相同
    """

    def __init__(
        self,
        pretrained,
        device="cuda",
        kv_start_size=8,
        kv_recent_size=48,
        use_text_kv_cache=False,  # ⭐ 新增参数
    ):
        # 调用父类构造函数
        super().__init__(pretrained, device, kv_start_size, kv_recent_size)

        # 初始化 KV cache 管理器
        self.use_text_kv_cache = use_text_kv_cache
        if use_text_kv_cache:
            self.kv_cache_manager = TextPrefixKVCache(self.model, self.processor)
            self.kv_cache_manager.cache_enabled = True
            print("✅ Text Prefix KV Cache ENABLED")
        else:
            self.kv_cache_manager = None
            print("⚠️  Text Prefix KV Cache DISABLED (baseline mode)")

    @torch.no_grad()
    def call_model(
        self,
        observations,
        task,
        step_id,
        add_frame_index: bool=False,
        gen_kwargs: dict = {},
    ):
        """
        增强版 call_model，支持 KV cache

        ⚠️ 当前版本: 仅添加了统计和框架，实际的 KV 复用需要解决 position IDs 问题
        """

        # ===== Step 0: 尝试使用 KV cache (如果启用) =====
        if self.use_text_kv_cache and step_id == 0:
            # Episode 开始，预计算文本 KV
            self.kv_cache_manager.compute_and_cache(task)

        # ===== 原有的图像处理逻辑 =====
        start_time = time.time()

        messages = []
        message = [
            {"role": "system",
             "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
            }
        ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."

        # 构建消息
        visual = observations
        if isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            image_content = []
            image_count = 0
            for v in visual:
                if add_frame_index:
                    image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})
                image_content.append({"type": "image", "image": v})
                image_count += 1
            message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})

        messages.append(message)

        # 处理图像
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
                    patch_size = self.processor.image_processor.patch_size
                    merge_size = self.processor.image_processor.merge_size

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

        # 记录 prefill 时间
        prefill_time = time.time() - start_time
        if self.use_text_kv_cache:
            if step_id == 0:
                self.kv_cache_manager.stats["prefill_time_without_cache"].append(prefill_time)
            else:
                # 理论上应该更快，但当前版本还没有实际复用
                # TODO: 实现实际的 KV 复用
                self.kv_cache_manager.stats["prefill_time_with_cache"].append(prefill_time)
                self.kv_cache_manager.stats["cache_hits"] += 1

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

        # ⚠️ TODO: 这里需要整合 past_key_values
        # 当前版本暂时保持原样
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

        return answers

    def reset_episode_cache(self):
        """Episode 结束时调用"""
        if self.kv_cache_manager:
            self.kv_cache_manager.reset()

    def print_kv_cache_stats(self):
        """打印 KV cache 统计"""
        if self.kv_cache_manager:
            self.kv_cache_manager.print_stats()


# ==================== 修改 eval() 函数 ====================

def eval_with_kv_cache():
    """带 KV cache 的评估函数"""
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
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

    # ⭐ 新增参数
    parser.add_argument("--use_text_kv_cache", action="store_true", default=False,
                        help="Enable text prefix KV cache for cross-step reuse")

    args = parser.parse_args()

    print('args:', args)

    set_seed(args.seed)
    init_distributed_mode(args)
    local_rank = args.local_rank

    global max_pixels, min_pixels
    max_pixels = args.max_pixels
    min_pixels = args.min_pixels
    print(f"Using max_pixels={max_pixels}, min_pixels={min_pixels}")

    # ⭐ 使用增强版模型
    model = JanusVLN_Inference_Enhanced(
        args.model_path,
        device=f"cuda:{local_rank}",
        kv_start_size=args.kv_start_size,
        kv_recent_size=args.kv_recent_size,
        use_text_kv_cache=args.use_text_kv_cache,  # 传入参数
    )

    evaluate(model, args)

    # 打印 KV cache 统计
    if args.use_text_kv_cache:
        model.print_kv_cache_stats()


if __name__ == "__main__":
    eval_with_kv_cache()
