# JanusVLN + Slow-Fast Active Memory Reuse Implementation Notes

Date: 2026-04-28

## Goal

在 `Fast_JanusVLN` 的 JanusVLN 原始推理框架上加入第一版 Slow-Fast Active Memory Reuse, SF-AMR，用来验证长程导航中复用 active history memory 是否可行。

当前阶段不接入 SMC / RSMS，也不实现完整 MemEdit-VLN。实现重点是：

- 保证 `enable_slow_fast=False` 时 baseline 行为不变。
- 先支持 debug-only 和 history-selection prototype。
- Fast Step 必须保留 current observation 和 instruction。
- Slow Step 周期性刷新 selected middle memory。
- 可选接入 SpargeAttn 的 adaptive sparse attention 到 Qwen LLM self-attention。

## Code Changes

### 1. Slow-Fast Active Memory Module

File:

- `src/qwen_vl/model/slow_fast_active_memory.py`

新增：

- `SlowFastAMRConfig`
- `SlowFastActiveMemoryReuse`
- `build_frame_janus_memory`
- `active_memory_to_frame_indices`
- `build_active_attention_mask`

核心 API：

```python
active_memory, debug_info = sf_amr.get_active_memory(
    janus_memory,
    nav_state,
    model_state,
)
```

`janus_memory` 支持字段：

- `instruction_indices`
- `current_observation_indices`
- `history_indices`
- `initial_indices`
- `middle_indices`
- `recent_indices`
- `history_attention_scores`
- `history_similarity_scores`

Slow Step 触发条件：

- init
- fixed interval
- optional `subgoal_changed`
- optional `room_change`
- optional `new_landmark_detected`
- optional `stuck`
- optional low action confidence
- optional high attention entropy

Middle memory selection strategy：

- `uniform`
- `recency`
- `attention_topk`
- `similarity_topk`

每步输出 debug 信息：

- `step_id`
- `step_type`
- `slow_trigger_reason`
- `num_active_kv`
- `num_total_kv`
- `active_kv_ratio`
- `num_selected_middle`
- `num_recent`
- `selected_memory_age`
- `slow_step_count`
- `fast_step_count`

### 2. Evaluation Loop Integration

File:

- `src/evaluation.py`

新增命令行参数：

```bash
--enable_slow_fast
--slow_fast_mode debug_only|history_selection
--slow_fast_refresh_interval 4
--slow_fast_selection_strategy uniform|recency|attention_topk|similarity_topk
--slow_fast_selected_middle_budget 128
--slow_fast_initial_window 4
--slow_fast_recent_window 24
--slow_fast_low_action_confidence_threshold
--slow_fast_high_attention_entropy_threshold
--slow_fast_disable_debug_log
```

接入位置：

原逻辑每步根据 `rgb_list` 构造历史图像：

```python
if history_len <= self.num_history:
    images = rgb_list[:history_len] + [rgb_list[-1]]
else:
    indices = np.linspace(0, history_len, self.num_history + 1, dtype=int)
    images = [rgb_list[i] for i in indices]
```

现在改为：

```python
images = self._select_images_for_step(
    rgb_list=rgb_list,
    step_id=step_id,
    info=info,
    scene_id=scene_id,
    episode_id=episode_id,
)
```

`slow_fast_mode` 行为：

- `debug_only`: 只计算并记录 active memory，不改变原 JanusVLN 输入。
- `history_selection`: 用 SF-AMR 的 active frame indices 替换原历史帧采样。

Debug log：

```text
slow_fast_amr_debug_rank*.jsonl
```

其中包含：

- active/effective/baseline frame indices
- active frame count
- baseline frame count
- slow/fast timeline
- active KV ratio

每个 episode 开始时会 reset：

- VGGT KV cache
- SF-AMR state
- adaptive sparse attention state

### 3. SpargeAttn Adaptive Sparse Attention Integration

Files:

- `src/qwen_vl/model/adaptive_sparse_attention.py`
- `src/qwen_vl/model/modeling_qwen2_5_vl.py`
- `src/evaluation.py`

新增 installer：

```python
install_adaptive_sparse_attention_qwen(model, ...)
```

新增命令行参数：

```bash
--use_llm_adaptive_sparse_attention
--spargeattn_path /ssd/dingmuhe/Embodied-task/SpargeAttn
--adaptive_sparse_min_seq_len 128
--adaptive_sparse_pvthreshd 1000000
--adaptive_sparse_target_blocks 79
--adaptive_sparse_target_drop_mass 0.68
```

实现方式：

- 参考 `/ssd/dingmuhe/Embodied-task/SpargeAttn/evaluate/cogvideo_example.py` 的 `args.attn_mode == "adaptive"`。
- 使用 `AdaptiveBlockMasker` 和 `AdaptiveSparseAttention`。
- 将 adaptive sparse attention 安装到 Qwen decoder layer 的 `self_attn` 上。
- Qwen eager / flash_attention_2 / sdpa attention path 都加了受保护 sparse branch。

Sparse branch 只在满足以下条件时启用：

- 显式开启 `--use_llm_adaptive_sparse_attention`
- batch size 为 1
- 推理态
- CUDA
- query sequence length >= `adaptive_sparse_min_seq_len`
- 无显式 `attention_mask`

否则自动回退原 dense attention。

## Current Behavior and Scope

当前 JanusVLN evaluation 代码每个导航步都会重新构造完整 prompt/image sequence 并调用 `generate(...)`，没有跨导航步复用 LLM `past_key_values`。

因此本版 SF-AMR 的实际验证层级是：

1. Debug-only active memory ratio。
2. Frame-level history selection 是否影响导航指标。
3. 可选 LLM prefill adaptive sparse attention 是否能降低 attention 代价。

当前还没有实现：

- 真正的 LLM KV cache 跨导航步复用。
- active KV gather。
- contiguous KV packing。
- 自定义 CUDA / Triton sparse KV kernel。
- SMC / RSMS / Q-Former memory edit。

## Example Commands

### Debug-only, 不改变模型输入

```bash
PYTHONPATH=src python src/evaluation.py \
  --enable_slow_fast \
  --slow_fast_mode debug_only \
  --slow_fast_refresh_interval 4 \
  --slow_fast_selection_strategy uniform \
  --slow_fast_selected_middle_budget 64
```

### 使用 SF-AMR 选择后的历史帧

```bash
PYTHONPATH=src python src/evaluation.py \
  --enable_slow_fast \
  --slow_fast_mode history_selection \
  --slow_fast_refresh_interval 8 \
  --slow_fast_selection_strategy recency \
  --slow_fast_selected_middle_budget 128
```

### 同时启用 LLM adaptive sparse attention

```bash
PYTHONPATH=src python src/evaluation.py \
  --enable_slow_fast \
  --slow_fast_mode history_selection \
  --slow_fast_refresh_interval 8 \
  --slow_fast_selection_strategy uniform \
  --slow_fast_selected_middle_budget 128 \
  --use_llm_adaptive_sparse_attention \
  --spargeattn_path /ssd/dingmuhe/Embodied-task/SpargeAttn
```

## Tests Added

Files:

- `tests/test_slow_fast_active_memory.py`
- `tests/test_adaptive_sparse_attention.py`

Covered behavior:

- First step must be Slow Step.
- Fixed interval refresh.
- Event triggers such as stuck / low action confidence / high attention entropy.
- Fast Step reuses previous selected middle memory.
- Fast Step updates current observation and recent memory.
- `uniform`, `recency`, `attention_topk` selection.
- Disabled mode returns full memory.
- Frame indices always keep current observation.
- Adaptive sparse attention safe-use gate.

Verification command:

```bash
PYTHONPATH=src pytest \
  tests/test_llm_visual_pruner.py \
  tests/test_slow_fast_active_memory.py \
  tests/test_adaptive_sparse_attention.py \
  -q
```

Observed result:

```text
12 passed
```

Syntax check:

```bash
python -m py_compile \
  src/qwen_vl/model/slow_fast_active_memory.py \
  src/qwen_vl/model/adaptive_sparse_attention.py \
  src/qwen_vl/model/modeling_qwen2_5_vl.py \
  src/evaluation.py
```

Observed result: passed.

## Recommended Experiments

Start with debug-only:

- `refresh_interval=4,8,16`
- `selection_strategy=uniform,recency`
- `selected_middle_budget=64,128,256`
- `initial_window=4`
- `recent_window=24`

Compare:

- JanusVLN baseline
- JanusVLN + SF-AMR debug-only
- JanusVLN + SF-AMR history-selection
- JanusVLN + SF-AMR + adaptive sparse attention

Metrics:

- SR
- SPL
- NE
- active KV/frame ratio
- generate latency
- long-horizon vs short-horizon difference

## Next Steps

1. Run debug-only on long-horizon validation and inspect `slow_fast_amr_debug_rank*.jsonl`.
2. If active ratio is meaningfully lower, run `history_selection`.
3. If performance holds, add timing around LLM attention/prefill.
4. If useful, implement true active KV gather.
5. Later integrate SMC / RSMS edited memory as higher-quality middle memory candidates.
