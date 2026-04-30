# JanusVLN LLM Adaptive Sparse Attention + Slow-Fast Active Memory Reuse Implementation Plan

> **For Claude/Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` before implementing this plan task-by-task.

**Goal:** 先把 `/ssd/dingmuhe/Embodied-task/SpargeAttn/evaluate/cogvideo_example.py` 中 `args.attn_mode == "adaptive"` 的方法接入 JanusVLN 的 Qwen LLM self-attention，形成可独立验证的 `LLM + Sparse Attention` baseline；确认 sparse LLM 可运行、可记录 sparsity/latency 后，再在此基础上实现 Slow-Fast Active Memory Reuse, SF-AMR。

**Architecture:** 第一阶段不接入 SMC / RSMS，不实现完整 MemEdit-VLN。实施顺序调整为：Phase A 先做 Qwen LLM adaptive sparse attention installer 和安全回退路径；Phase B 再把 JanusVLN 的历史输入抽象成 `janus_memory`，由独立 `SlowFastActiveMemoryReuse` 模块输出 `active_memory / active_kv_indices / debug_info`；Phase C 才考虑 active KV gather / kernel 优化。所有能力默认关闭，保证 baseline 完全不变。

**Tech Stack:** Python, PyTorch, Transformers Qwen2.5-VL, JanusVLN evaluation loop, optional SpargeAttn `AdaptiveBlockMasker` / `AdaptiveSparseAttention`.

---

## 0. 当前代码理解

### JanusVLN 推理主路径

当前主要入口在：

- `Fast_JanusVLN/src/evaluation.py`
- `Fast_JanusVLN/src/qwen_vl/model/modeling_qwen2_5_vl.py`

`evaluation.py` 中的核心流程是：

```text
env observation
-> rgb_list append current frame
-> 根据 num_history 从 rgb_list 中采样历史图像
-> JanusVLN_Inference.call_model(images, instruction, step_id)
-> processor 构造 text/images
-> Qwen2_5_VLForConditionalGenerationForJanusVLN.generate(...)
-> action
-> env.step(action)
```

需要注意：

1. 当前 navigation step 之间没有显式复用 LLM `past_key_values`。
2. 每一步都会重新构造 prompt/image sequence 并调用 `generate(...)`。
3. 已存在 VGGT aggregator 的 `past_key_values_vggt`，它是视觉/3D 分支缓存，不等价于 LLM KV cache。
4. 当前优先实现目标不是 SF-AMR，而是先把 SpargeAttn adaptive sparse attention 安全接入 Qwen LLM，得到 `JanusVLN + LLM Sparse` baseline。
5. 在 `LLM Sparse` 可运行后，再实现 SF-AMR。第一版 SF-AMR 最稳妥的接入点仍是 `evaluation.py` 的历史帧选择层；token/KV-level reuse 后续再推进。

### 当前历史组织方式

当前历史近似是 frame-level：

```python
history_len = len(rgb_list) - 1

if history_len <= num_history:
    images = rgb_list[:history_len] + [rgb_list[-1]]
else:
    indices = np.linspace(0, history_len, num_history + 1, dtype=int)
    images = [rgb_list[i] for i in indices]
```

也就是说，当前输入给 LLM 的历史不是显式 KV index，而是历史图像帧采样结果。真正 token-level / KV-level 的 active selection 需要后续在 Qwen forward 或 cache 管理处扩展。

### SpargeAttn adaptive 参考路径

参考文件：

- `/ssd/dingmuhe/Embodied-task/SpargeAttn/evaluate/cogvideo_example.py`
- `/ssd/dingmuhe/Embodied-task/SpargeAttn/evaluate/modify_model/modify_cogvideo.py`
- `/ssd/dingmuhe/Embodied-task/SpargeAttn/spas_sage_attn/adaptive_attention.py`

CogVideo 的 adaptive 模式核心做法：

```python
set_adaptive_attn_cogvideox(
    transformer,
    pvthreshd=1e6,
    stateless=False,
    mask_kwargs=dict(
        prefill_min_blocks=2,
        target_blocks=79,
        target_drop_mass=0.68,
        recent_window_len=2,
        tau_init=0.5,
        lam_init=0.0,
        mu_reg=0.01,
        beta=0.2,
        lr_tau=0.1,
        alpha_min=0.0,
        alpha_max=4.5,
        rho_lambda=0.5,
    ),
)
```

它会为每层 attention 构造共享 `AdaptiveBlockMasker`，并把 `AdaptiveSparseAttention` 装到 attention 内部。迁移到 Qwen LLM 时不能直接复用 CogVideo processor，需要为 Qwen decoder self-attention 单独写 installer。

---

## 1. 设计边界和实施顺序

### Phase A: 先做 LLM + Sparse Attention

第一优先级是实现：

```text
JanusVLN baseline
-> JanusVLN + Qwen LLM Adaptive Sparse Attention
```

目标：

- 复用 SpargeAttn `args.attn_mode == "adaptive"` 的思想。
- 将 `AdaptiveBlockMasker + AdaptiveSparseAttention` 接入 Qwen decoder self-attention。
- 先验证 sparse attention 在 JanusVLN LLM forward / generate 中能稳定运行。
- 记录 sparse ratio、generate latency、是否 fallback dense。
- 不改变 JanusVLN 的历史采样逻辑。

### Phase B: 再做 Slow-Fast Active Memory Reuse

在 Phase A 可运行后，再实现：

```text
JanusVLN + Qwen LLM Adaptive Sparse Attention
-> JanusVLN + Qwen LLM Adaptive Sparse Attention + SF-AMR
```

目标：

- 新增独立 `SlowFastActiveMemoryReuse` 模块。
- 只依赖 JanusVLN 原始历史，不依赖 SMC / RSMS。
- 支持 `janus_memory, nav_state, model_state` 输入。
- 支持 fixed interval slow/fast。
- 支持 optional event trigger。
- 支持 `uniform / recency / attention_topk / similarity_topk`。
- 每步输出完整 debug 信息。
- 第一版 SF-AMR 先做 debug-only 或 frame-level history-selection。

### 本阶段不做

- 不接入 Q-Former / SMC。
- 不接入 RSMS。
- 不实现 MemEdit-VLN 完整版本。
- 不一开始改 CUDA kernel。
- 不一开始做 true KV gather。
- 不改变默认 baseline 行为。

### Baseline 保持原则

所有新增能力必须由显式参数开启：

```bash
--enable_slow_fast
--use_llm_adaptive_sparse_attention
```

默认情况下：

- history frame selection 仍使用原 `num_history + np.linspace` 逻辑。
- Qwen attention 仍使用原 attention implementation。
- 不写 slow-fast debug log。
- 不创建 adaptive masker。

### 推荐实施顺序

```text
1. LLM adaptive sparse attention unit gate
2. Qwen adaptive sparse attention installer
3. Qwen attention forward sparse branch + dense fallback
4. Evaluation CLI and sparse logging
5. Run JanusVLN + LLM Sparse baseline
6. Standalone SF-AMR module
7. SF-AMR debug-only integration
8. SF-AMR history-selection / mask prototype
```

---

## 2. Phase A: Qwen LLM Adaptive Sparse Attention 接入计划

### Files

Create:

- `Fast_JanusVLN/src/qwen_vl/model/adaptive_sparse_attention.py`

Modify:

- `Fast_JanusVLN/src/qwen_vl/model/modeling_qwen2_5_vl.py`
- `Fast_JanusVLN/src/evaluation.py`

### Installer 设计

```python
def install_adaptive_sparse_attention_qwen(model, *, pvthreshd, stateless, mask_kwargs):
    from spas_sage_attn.adaptive_attention import AdaptiveSparseAttention
    from spas_sage_attn.mask_strategies import AdaptiveBlockMasker

    layers = model.model.layers
    masker = AdaptiveBlockMasker(
        num_layers=len(layers),
        num_heads=layers[0].self_attn.num_heads,
        head_dim=layers[0].self_attn.head_dim,
        q_block_size=...,
        kv_block_size=...,
        **mask_kwargs,
    )

    for layer_idx, layer in enumerate(layers):
        layer.self_attn.adaptive_sparse_attention = AdaptiveSparseAttention(masker, layer_idx, ...)
```

### Qwen attention forward 接入方式

在 Qwen attention 中，在 q/k/v + RoPE + repeat_kv 后尝试 sparse branch：

```python
adaptive_output = self._adaptive_sparse_attention_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
)
if adaptive_output is not None:
    return projected_output
```

### 安全回退条件

只在满足以下条件时启用 sparse：

- 显式安装了 `adaptive_sparse_attention`
- `attention_mask is None`
- `batch_size == 1`
- `not self.training`
- CUDA
- `q_len >= adaptive_sparse_min_seq_len`

否则回退原 dense/flash/sdpa attention。

这样做的原因：

1. SpargeAttn adaptive 当前不支持显式 attention mask。
2. 当前 JanusVLN evaluation 通常 batch=1，适合先验证。
3. 如果有 padding / mask，强行 sparse 容易破坏因果或 padding 语义。

### CLI 参数

```python
parser.add_argument("--use_llm_adaptive_sparse_attention", action="store_true", default=False)
parser.add_argument("--spargeattn_path", type=str, default="/ssd/dingmuhe/Embodied-task/SpargeAttn")
parser.add_argument("--adaptive_sparse_min_seq_len", type=int, default=128)
parser.add_argument("--adaptive_sparse_pvthreshd", type=float, default=1e6)
parser.add_argument("--adaptive_sparse_target_blocks", type=float, default=79)
parser.add_argument("--adaptive_sparse_target_drop_mass", type=float, default=0.68)
```

### 风险

- FlashAttention path 可能始终传入 attention mask，导致 sparse branch 不触发。
- Qwen multimodal RoPE 与 sparse kernel 的 causal 语义必须保持一致。
- Sparse branch fallback 必须稳定，不应让 evaluation 崩溃。
- 真正 latency 是否下降取决于 SpargeAttn kernel 是否适配 Qwen head_dim / seq_len / dtype。

---

## 3. Phase B: SF-AMR 数据结构设计

### SlowFastAMRConfig

建议字段：

```python
@dataclass
class SlowFastAMRConfig:
    enable_slow_fast: bool = False
    refresh_interval: int = 4
    selection_strategy: str = "uniform"
    selected_middle_budget: int = 128
    initial_window: int = 4
    recent_window: int = 24
    low_action_confidence_threshold: Optional[float] = None
    high_attention_entropy_threshold: Optional[float] = None
```

### janus_memory

第一版可以是 frame-level index，后续升级为 token/KV index：

```python
janus_memory = {
    "instruction_indices": [...],
    "current_observation_indices": [...],
    "history_indices": [...],
    "initial_indices": [...],
    "middle_indices": [...],
    "recent_indices": [...],
    "history_attention_scores": optional,
    "history_similarity_scores": optional,
}
```

在 `evaluation.py` 第一版中：

- `history_indices`: 历史帧 id，范围 `[0, history_len - 1]`
- `current_observation_indices`: 当前帧 id `[history_len]`
- `instruction_indices`: frame-level 阶段可以先为空或用 sentinel，token-level 阶段再精确填写。

### active_memory

```python
active_memory = {
    "instruction_indices": [...],
    "current_observation_indices": [...],
    "initial_indices": [...],
    "recent_indices": [...],
    "selected_middle_indices": [...],
    "active_kv_indices": [...],
}
```

拼接规则：

```text
active_kv_indices =
instruction
+ current_observation
+ initial
+ recent
+ selected_middle
```

实现时需要去重，但不能丢掉：

- instruction
- current observation
- initial
- recent

### debug_info

每步记录：

```python
debug_info = {
    "step_id": int,
    "step_type": "slow" | "fast" | "disabled",
    "slow_trigger_reason": str | None,
    "num_active_kv": int,
    "num_total_kv": int,
    "active_kv_ratio": float,
    "num_selected_middle": int,
    "num_recent": int,
    "selected_memory_age": int | None,
    "slow_step_count": int,
    "fast_step_count": int,
}
```

---

## 4. Phase B: SF-AMR 模块设计

### File

Create:

- `Fast_JanusVLN/src/qwen_vl/model/slow_fast_active_memory.py`

### Class

```python
class SlowFastActiveMemoryReuse:
    def __init__(self, config: SlowFastAMRConfig): ...
    def reset(self): ...
    def should_slow_step(self, nav_state, model_state): ...
    def slow_step_refresh(self, janus_memory, step_id): ...
    def fast_step_reuse(self, janus_memory): ...
    def select_middle_indices(self, janus_memory): ...
    def merge_indices(self, *parts): ...
    def get_active_memory(self, janus_memory, nav_state, model_state): ...
    def build_debug_info(...): ...
```

### should_slow_step

优先级建议：

```python
if first step or no previous selected middle:
    return True, "init"
if subgoal_changed:
    return True, "subgoal_changed"
if room_change:
    return True, "room_change"
if new_landmark_detected:
    return True, "new_landmark_detected"
if stuck:
    return True, "stuck"
if action_confidence < threshold:
    return True, "low_action_confidence"
if attention_entropy > threshold:
    return True, "high_attention_entropy"
if step_id - last_slow_step_id >= refresh_interval:
    return True, "fixed_interval"
return False, None
```

第一版实验建议只启用：

- init
- fixed interval
- stuck/collision if already available from Habitat metrics

### select_middle_indices

策略：

```python
if strategy == "uniform":
    均匀采样 middle_indices
elif strategy == "recency":
    取 middle_indices 最后 budget 个
elif strategy == "attention_topk":
    根据 history_attention_scores top-k
elif strategy == "similarity_topk":
    根据 history_similarity_scores top-k
```

注意：

- top-k 选择后建议按原时间顺序返回，避免打乱历史顺序。
- 如果 score 不存在，应该 fallback 到 `uniform`，或者直接报错。第一版建议 fallback 到 `uniform`，便于实验不中断。

---

## 5. Phase B: JanusVLN Evaluation 接入计划

### File

Modify:

- `Fast_JanusVLN/src/evaluation.py`

### Step 1: 添加 CLI 参数

```python
parser.add_argument("--enable_slow_fast", action="store_true", default=False)
parser.add_argument("--slow_fast_mode", choices=["debug_only", "history_selection"], default="debug_only")
parser.add_argument("--slow_fast_refresh_interval", type=int, default=4)
parser.add_argument("--slow_fast_selection_strategy", choices=["uniform", "recency", "attention_topk", "similarity_topk"], default="uniform")
parser.add_argument("--slow_fast_selected_middle_budget", type=int, default=128)
parser.add_argument("--slow_fast_initial_window", type=int, default=4)
parser.add_argument("--slow_fast_recent_window", type=int, default=24)
```

可选：

```python
parser.add_argument("--slow_fast_low_action_confidence_threshold", type=float, default=None)
parser.add_argument("--slow_fast_high_attention_entropy_threshold", type=float, default=None)
parser.add_argument("--slow_fast_disable_debug_log", action="store_true", default=False)
```

### Step 2: 初始化 SF-AMR

在 `VLNEvaluator.__init__` 中：

```python
if args.enable_slow_fast:
    self.slow_fast_amr = SlowFastActiveMemoryReuse(config)
else:
    self.slow_fast_amr = None
```

Debug log:

```text
output_path/slow_fast_amr_debug_rank{rank}.jsonl
```

### Step 3: 每个 episode reset

在 episode reset 后：

```python
if self.slow_fast_amr is not None:
    self.slow_fast_amr.reset()
```

### Step 4: 替换历史图像选择函数

新增 helper：

```python
def _select_images_for_step(self, rgb_list, step_id, info, scene_id, episode_id):
    baseline_indices = self._baseline_frame_indices(history_len)

    if self.slow_fast_amr is None:
        return [rgb_list[i] for i in baseline_indices]

    janus_memory = build_frame_janus_memory(...)
    active_memory, debug_info = self.slow_fast_amr.get_active_memory(...)

    if self.args.slow_fast_mode == "debug_only":
        effective_indices = baseline_indices
    else:
        effective_indices = active_frame_indices

    log_debug(...)
    return [rgb_list[i] for i in effective_indices]
```

### Step 5: 保证 current observation 不丢

必须有测试约束：

```python
assert history_len in effective_indices
```

或者 helper 内强制追加当前帧。

## 6. 分阶段实施计划

### Task 1: Adaptive Sparse Attention Unit Gate

**Files:**

- Create: `Fast_JanusVLN/tests/test_adaptive_sparse_attention.py`
- Create: `Fast_JanusVLN/src/qwen_vl/model/adaptive_sparse_attention.py`

**Test cases:**

- allow only batch=1, no mask, eval, long enough seq.
- reject training.
- reject batch>1.
- reject explicit attention mask.
- reject too-short sequence.
- reject non-CUDA when `require_cuda=True`.

**Expected command:**

```bash
cd Fast_JanusVLN
PYTHONPATH=src pytest tests/test_adaptive_sparse_attention.py -q
```

### Task 2: Qwen Adaptive Sparse Installer

**Files:**

- Create/Modify: `Fast_JanusVLN/src/qwen_vl/model/adaptive_sparse_attention.py`

**Implementation notes:**

- Add `install_adaptive_sparse_attention_qwen(model, ...)`.
- Import SpargeAttn from `/ssd/dingmuhe/Embodied-task/SpargeAttn`.
- Mirror CogVideo adaptive config:
  - `AdaptiveBlockMasker`
  - `AdaptiveSparseAttention`
  - `pvthreshd=1e6`
  - `stateless=False`
  - `target_blocks=79`
  - `target_drop_mass=0.68`
- Attach one `AdaptiveSparseAttention` module per Qwen decoder self-attention layer.
- Expose `reset_adaptive_sparse_attention_state(model)`.
- Expose `summarize_adaptive_sparsity(model)`.

**Acceptance:**

- Unit tests pass.
- Import failure gives clear message if SpargeAttn is not on `PYTHONPATH`.
- Installer does not mutate model unless explicitly called.

### Task 3: Qwen Attention Forward Sparse Branch

**Files:**

- Modify: `Fast_JanusVLN/src/qwen_vl/model/modeling_qwen2_5_vl.py`

**Behavior:**

- In Qwen self-attention, after q/k/v projection, RoPE, cache update, and repeat_kv, try adaptive sparse branch.
- If sparse branch is unsafe or fails, fallback to original eager / flash_attention_2 / sdpa path.
- Do not alter output shape or return contract.

**Sparse branch safety gate:**

```python
if adaptive_sparse_attention is installed
and attention_mask is None
and batch_size == 1
and not self.training
and q_len >= adaptive_sparse_min_seq_len
and query_states.device.type == "cuda":
    use AdaptiveSparseAttention
else:
    use original attention
```

**Acceptance:**

- Syntax check passes.
- Existing tests pass.
- With flag disabled, original attention path is used.
- With flag enabled but unsafe, dense fallback is used.

### Task 4: Evaluation CLI for LLM Sparse

**Files:**

- Modify: `Fast_JanusVLN/src/evaluation.py`

**Behavior:**

- Add explicit sparse flags:

```python
--use_llm_adaptive_sparse_attention
--spargeattn_path /ssd/dingmuhe/Embodied-task/SpargeAttn
--adaptive_sparse_min_seq_len 128
--adaptive_sparse_pvthreshd 1000000
--adaptive_sparse_target_blocks 79
--adaptive_sparse_target_drop_mass 0.68
```

- Install adaptive sparse attention only after model load and only when flag is set.
- Reset adaptive masker at episode start.
- Periodically log sparse summary.

**Acceptance:**

- JanusVLN can start without sparse flag.
- JanusVLN can start with sparse flag if SpargeAttn import works.
- Generate latency and sparse summary are visible in logs.

### Task 5: LLM Sparse Smoke Test Script

**Files:**

- Create: `Fast_JanusVLN/scripts/evaluation_llm_adaptive_sparse.sh`

**Behavior:**

- Run normal JanusVLN evaluation with:

```bash
--use_llm_adaptive_sparse_attention
--spargeattn_path /ssd/dingmuhe/Embodied-task/SpargeAttn
```

- Keep `--enable_slow_fast` off.

**Acceptance:**

- This script tests only `JanusVLN + LLM Sparse`.
- No slow-fast reuse is involved yet.

### Task 6: Standalone SF-AMR Unit Tests

**Files:**

- Create: `Fast_JanusVLN/tests/test_slow_fast_active_memory.py`

**Test cases:**

1. first step is Slow Step.
2. fixed interval triggers Slow Step.
3. Fast Step reuses previous selected middle.
4. Fast Step updates current observation and recent memory.
5. disabled mode returns full memory.
6. `uniform / recency / attention_topk / similarity_topk` behavior.
7. current observation is always retained.

**Expected command:**

```bash
cd Fast_JanusVLN
PYTHONPATH=src pytest tests/test_slow_fast_active_memory.py -q
```

### Task 7: Implement Standalone SF-AMR

**Files:**

- Create: `Fast_JanusVLN/src/qwen_vl/model/slow_fast_active_memory.py`

**Acceptance:**

- Unit tests pass.
- No dependency on Habitat / Qwen / SpargeAttn.
- Can be tested on CPU.

### Task 8: SF-AMR Evaluation Debug-only Integration

**Files:**

- Modify: `Fast_JanusVLN/src/evaluation.py`

**Behavior:**

- `--enable_slow_fast --slow_fast_mode debug_only` logs active memory but keeps original `images` input.
- Without `--enable_slow_fast`, code path is identical to baseline.
- Can be combined with `--use_llm_adaptive_sparse_attention`, but should also run independently.

**Acceptance:**

- Generated `slow_fast_amr_debug_rank*.jsonl`.
- `effective_frame_indices == baseline_frame_indices` in debug-only.
- No metric change expected except logging overhead.

### Task 9: SF-AMR History-selection Prototype

**Files:**

- Modify: `Fast_JanusVLN/src/evaluation.py`

**Behavior:**

- `--slow_fast_mode history_selection` uses active frame indices.
- Current frame is always included.
- Recent and initial are always included.

**Acceptance:**

- Can run short evaluation without crash.
- Debug log shows lower active frame ratio.
- SR/SPL/NE can be compared against:
  - JanusVLN baseline
  - JanusVLN + LLM Sparse

### Task 10: Combined Experiment Scripts

**Files:**

- Modify or create scripts under `Fast_JanusVLN/scripts/`

Suggested scripts:

- `evaluation_llm_adaptive_sparse.sh`
- `evaluation_slow_fast_debug.sh`
- `evaluation_slow_fast_history_selection.sh`
- `evaluation_slow_fast_plus_adaptive_sparse.sh`

Each slow-fast script should expose:

- `refresh_interval`
- `selection_strategy`
- `selected_middle_budget`
- `initial_window`
- `recent_window`

---

## 7. Experiment Matrix

### Stage 1: LLM Adaptive Sparse Attention Only

```text
baseline
baseline + Qwen LLM adaptive sparse attention
```

Goal:

- 确认 SpargeAttn adaptive 能接入 Qwen LLM self-attention。
- 确认 generate 不崩溃。
- 记录 sparse ratio。
- 记录 generate latency。
- 看 SR / SPL / NE 是否保持。

### Stage 2: SF-AMR Debug-only on Dense and Sparse LLM

```text
baseline + SF-AMR debug_only interval=4 uniform budget=64
LLM sparse + SF-AMR debug_only interval=4 uniform budget=64
LLM sparse + SF-AMR debug_only interval=8 uniform budget=128
LLM sparse + SF-AMR debug_only interval=16 uniform budget=256
```

Goal:

- 看 active frame / active KV ratio。
- 看 slow/fast timeline。
- 看 selected middle 是否稳定。
- 确认 debug-only 不改变模型输入。

### Stage 3: SF-AMR History-selection on Sparse LLM

```text
LLM sparse + SF-AMR history_selection interval=4 uniform budget=64
LLM sparse + SF-AMR history_selection interval=8 uniform budget=128
LLM sparse + SF-AMR history_selection interval=16 uniform budget=256
LLM sparse + SF-AMR history_selection interval=8 recency budget=128
```

Goal:

- 看 SR / SPL / NE 是否保持。
- 看 active history 变小后是否进一步降低 attention latency。
- 看长程任务是否更受益。

### Stage 4: Dense vs Sparse vs Sparse+SF-AMR 对比

```text
JanusVLN baseline
JanusVLN + LLM adaptive sparse attention
JanusVLN + SF-AMR history_selection
JanusVLN + LLM adaptive sparse attention + SF-AMR history_selection
```

Goal:

- 区分收益来自 sparse attention 还是 slow-fast reuse。
- 判断两者是否叠加有效。

---

## 8. Logging and Metrics

### LLM Sparse Log

建议记录：

```json
{
  "scene_id": "...",
  "episode_id": "...",
  "step_id": 12,
  "adaptive_sparse_enabled": true,
  "adaptive_sparse_used_layers": 24,
  "adaptive_sparse_fallback_layers": 4,
  "mean_sparsity": 0.55,
  "generate_time_ms": 123.4
}
```

### SF-AMR Debug JSONL

每步一行：

```json
{
  "scene_id": "...",
  "episode_id": "...",
  "step_id": 12,
  "step_type": "fast",
  "slow_trigger_reason": null,
  "active_kv_ratio": 0.42,
  "num_selected_middle": 128,
  "num_recent": 24,
  "selected_memory_age": 3,
  "active_frame_indices": [0, 4, 8, 30, 31, 32],
  "effective_frame_indices": [0, 4, 8, 30, 31, 32],
  "baseline_frame_indices": [0, 5, 10, 15, 20, 25, 32]
}
```

### Metrics to compare

- SR
- SPL
- NE
- OS
- steps
- generate time
- adaptive sparse mean sparsity
- adaptive sparse fallback count
- active KV/frame ratio
- GPU peak memory if available

---

## 9. Open Questions for Discussion

1. Qwen adaptive sparse attention 第一版接哪个 path？
   - `flash_attention_2` 是当前 evaluation 使用路径，但 varlen/mask 语义更复杂。
   - `eager/sdpa` 更容易 debug，但可能与当前运行配置不一致。
   - 建议先保留三路 hook，但只有满足安全条件时启用 sparse，否则 fallback。

2. Adaptive sparse attention 的日志粒度？
   - 每 step 打印 summary 简单但日志多。
   - 每 N step 或每 episode 汇总更适合长跑。

3. LLM sparse 是否必须先独立跑完再做 SF-AMR？
   - 建议是。否则一旦指标变化，难以区分 sparse attention 和 slow-fast reuse 的影响。

4. 第一版 active unit 用 frame-level 还是 token-level？
   - frame-level 最容易接入 `evaluation.py`。
   - token-level 更接近最终 KV sparse，但改动更大。

5. `selected_middle_budget=64/128/256` 对 frame-level 是否过大？
   - 如果 index 是 frame，budget 应该小很多。
   - 如果 index 是 token/KV block，budget 才适合 64/128/256。
   - 建议文档和代码里明确 `budget_unit=frame|token|block`。

6. `attention_topk` 的 attention score 从哪里取？
   - Qwen output attentions 会增加显存和计算。
   - 可以先不启用，只保留接口。
   - 后续从 sparse attention stats 或轻量 similarity scorer 获得。

7. 是否需要新增 `--slow_fast_budget_unit`？
   - 如果后续从 frame-level 升到 token/KV-level，建议提前留参数。

8. 是否把 SF-AMR debug log 纳入 qualitative trajectory JSON？
   - 独立 JSONL 更利于分析。
   - 合并到 qualitative JSON 更利于单 episode 可视化。

---

## 10. Recommended Discussion Decision

实现前建议先确定这几个决策：

1. 先独立实现并评估 `JanusVLN + LLM adaptive sparse attention`，暂不启用 SF-AMR。
2. Qwen sparse branch 第一版是否允许只在安全条件下触发，其他情况 fallback dense。
3. Sparse 日志按 step、按 episode，还是每 N step 汇总。
4. LLM sparse 稳定后，SF-AMR Stage 1 是否只做 frame-level debug-only。
5. SF-AMR Stage 2 的 prototype 是 history-selection，还是直接 attention mask。
6. `selected_middle_budget` 的单位是 frame、token 还是 KV block。

建议默认选择：

```text
Stage 1: Qwen LLM adaptive sparse attention only
Stage 2: LLM sparse + frame-level SF-AMR debug-only
Stage 3: LLM sparse + frame-level SF-AMR history-selection
Stage 4: token/KV-level gather after验证有效
```

这样可以先回答一个更基础的问题：SpargeAttn adaptive sparse attention 能否稳定接入 JanusVLN 的 Qwen LLM，并带来可测的 attention/generate 加速；之后再回答 slow-fast active memory reuse 是否能进一步减少历史访问成本。
