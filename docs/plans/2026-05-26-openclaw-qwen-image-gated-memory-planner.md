# OpenClaw Qwen Image-Gated Memory Planner

## 背景

当前 direct Qwen API planner 已经证明路径正确：

```text
model_provider=qwen_api
openclaw_session_mode=stateless_model
openclaw_session_id=""
fallback_steps=0
```

但 30 样本测试暴露出主要瓶颈：每个 navigation step 都向 Qwen API
发送图像+文本请求。已停止的测试目录：

```text
results/clawnav_openclaw_gateway_30_direct_qwen_api_20260526
```

在停止前记录到：

```text
trace_rows 247
episodes_started 7
image_count_dist {1: 7, 2: 84, 3: 156}
fallback_steps 0
```

第一个 episode `2azQ1b91cZZ:11` 用时约 10 分 23 秒，29 steps。
trace 中本地 `NavigationPolicySkill` latency 总和约 54.7 秒，因此主要耗时不是本地
Habitat/JanusVLN 执行，而是每步 Qwen 图像+文本 API 等待。

## 目标

第一阶段目标不是立刻取消 Qwen planner，而是减少图像 API 成本，同时保持方法可比性。

核心目标：

- 大多数 step 不再向 Qwen 发送图像。
- 每个 step 仍由 Qwen planner 输出导航 action。
- 关键帧或定期刷新 step 才向 Qwen 发送图像。
- 普通 step 使用最近一次 visual memory summary 作为文本上下文。
- trace 能明确区分 `visual_update` 和 `fast_text`。
- 30 样本评测中 `fallback_steps=0`，success/SPL 不明显塌。

非目标：

- 第一阶段不做 no-Qwen local-policy fast path。
- 第一阶段不接入 `spatial_http` 真实跨 episode visual memory。
- 第一阶段不实现复杂 collision/stuck/progress-failure trigger，除非已有可靠 runtime 信号。
- 第一阶段不改变 Qwen planner 的 action 决策权。

## 方法边界

本方案分三阶段。

### Stage 1: Image-Gated Qwen

这是第一版必须先实现的版本。

语义：

```text
visual_update step:
  Qwen 输入 = compact text + current image + at most one keyframe image
  Qwen 输出 = action + visual summary/subgoal
  runtime 缓存 episode-local visual summary

fast_text step:
  Qwen 输入 = compact text only + cached visual summary/subgoal
  Qwen 输出 = action
  不发送图像
```

重要点：

- `planner_authority=qwen` 始终成立。
- `qwen_api_called=true` 始终成立。
- 只有 `visual_update` 的 `model_image_count > 0`。
- `fast_text` 的 `model_image_count=0`。

这阶段只改变“是否传图”，不改变“谁决策”。

### Stage 2: Memory Text Compression

Stage 1 跑通后再做。目标是进一步减少 text-only Qwen 调用的 prompt 长度。

普通 `fast_text` step 只保留：

```text
instruction
scene_id / episode_id / step_id
last_action
policy_action
last_visual_summary
suggested_subgoal
visual_memory_age_steps
recent_step_summary
```

不再发送：

```text
current_image_path
full recent_keyframe_paths
memory_images
过长 memory_context_text
完整历史
```

### Stage 3: Memory-Guided Local Policy Fast Path

这是后续新方法，不能混入 Stage 1 结果。

语义：

```text
visual_update step:
  Qwen 看图并产出 visual summary/subgoal

memory_fast step:
  不调用 Qwen
  使用 JanusVLN local policy action
  使用最近 Qwen subgoal/guidance 作为约束和 trace 证据
```

Stage 3 必须单独命名，例如：

```text
memory_guided_policy_fast
```

它不再是纯 Qwen planner 加速，而是 hybrid policy 方法，必须单独和 Stage 1/Stage 2
以及 baseline 比较。

Implementation note, 2026-05-27: `memory_guided_policy_fast` is implemented in
the adapter. Cached `fast_text` steps return an `act` decision without
`action_text`, pass cached visual memory as `memory_context_text` /
`active_subgoal`, and let local `NavigationPolicySkill` choose the action.
`QwenApiModelClient` also supports configurable retry/backoff via
`OPENCLAW_QWEN_API_RETRIES` and `OPENCLAW_QWEN_API_RETRY_BACKOFF_S`; retry
metadata is written into `context_audit`.

## Stage 1 具体设计

### 配置

新增 adapter 配置：

```text
OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=10
OPENCLAW_MODEL_FAST_MODE=qwen_text_only
OPENCLAW_MODEL_MAX_IMAGES=2
OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=1
```

含义：

- `OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=10`：每 10 step 强制一次 visual update。
- `OPENCLAW_MODEL_FAST_MODE=qwen_text_only`：普通 step 仍调用 Qwen，但不传图。
- `OPENCLAW_MODEL_MAX_IMAGES=2`：visual update 最多传当前图 + 1 张 keyframe。
- `OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=1`：fast_text step 注入 cached visual memory summary。

保留现有：

```text
OPENCLAW_MODEL_PROVIDER=qwen_api
OPENCLAW_PLANNER_MODE=model
OPENCLAW_AGENT_MAX_INPUT_TOKENS=50000
```

### Step Mode 判定

新增 step mode：

```text
visual_update
fast_text
```

Stage 1 判定规则：

```text
visual_update if:
  step_id == 0
  OR step_id % OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS == 0
  OR runtime_context.keyframe_candidate.image_path exists
  OR cached visual memory is missing
  OR visual_memory_age_steps >= OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS
  OR previous visual_update failed

otherwise:
  fast_text
```

暂不启用：

```text
collision/stuck/progress-failure forced refresh
```

原因：当前 proposal review 指出这些信号尚未定义清楚数据路径。它们可以在后续阶段补，
但不能作为 Stage 1 的 blocking 逻辑。

### Image Selection

当前 `src/harness/openclaw/openclaw_cli_plan_gateway.py` 的
`_model_image_files()` 直接取：

```text
_visual_image_paths(runtime_context)[:openclaw_model_max_images]
```

而 `_visual_image_paths()` 是：

```text
current_image_path + recent_keyframe_paths
```

因此当前几乎每步都会带图。

Stage 1 改为：

```text
if planner_step_mode == fast_text:
  return {"paths": [], "missing_paths": []}

if planner_step_mode == visual_update:
  attach current_image_path first
  then attach the most recent keyframe path if distinct
  cap by OPENCLAW_MODEL_MAX_IMAGES
```

### Episode-Local Visual Memory

Stage 1 不接 `spatial_http`。只在 adapter 进程内维护 episode-local cached memory。

缓存 key：

```text
run_id
scene_id
episode_id
```

缓存内容：

```text
last_visual_step_id
last_visual_summary
last_suggested_subgoal
last_qwen_reason
last_action_text
valid_until_step
source_image_paths
last_update_error
```

Stage 1 的 memory payload 保持最小，只保存 fast step 立刻会消费的字段：

```text
image_path
visual_summary 或 caption
suggested_subgoal
qwen_reason
valid_until_step
```

不在第一阶段强制结构化：

```text
objects
landmarks
spatial_cues
```

这些字段只有在当前 consumer 明确使用时再加，避免过度设计。

### Fast Text Prompt

`fast_text` step 的 prompt 需要明确告诉 Qwen：

```text
No image is attached for this step.
Use the cached visual summary from the last visual_update as visual memory.
If the memory is insufficient, return an action and include reason indicating visual refresh is needed.
```

注入字段：

```text
last_visual_summary
last_suggested_subgoal
visual_memory_age_steps
valid_until_step
last_visual_step_id
```

不要在 `fast_text` 中把 image path 当作视觉证据传给 Qwen。image path 可以留在 trace，
但不应让 Qwen 误以为有图像附件。

## 实现单元

### U1: 增加配置和 step mode 判定

Files:

- `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- `scripts/start_openclaw_cli_plan_gateway.sh`
- `tests/test_openclaw_cli_plan_gateway.py`
- `tests/test_evaluation_scripts.py`

Approach:

- 为 adapter 增加 `openclaw_model_image_interval_steps`。
- 为 adapter 增加 `openclaw_model_fast_mode`，第一阶段只支持 `qwen_text_only`。
- 为 start script 增加对应 env var。
- 在 planner 内实现 `_planner_step_mode(prompt_payload)`。

Test scenarios:

- step 0 returns `visual_update`。
- step 1 with valid memory returns `fast_text`。
- step 10 returns `visual_update` when interval is 10。
- keyframe candidate returns `visual_update`。
- missing cached memory returns `visual_update`。

### U2: 改造 image attachment

Files:

- `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- `tests/test_openclaw_cli_plan_gateway.py`

Approach:

- `_model_image_files()` 接收或计算 `planner_step_mode`。
- `fast_text` 返回空 image list。
- `visual_update` 只传当前图和最多一张 keyframe。
- trace 中记录 `model_missing_image_paths`，避免静默丢图。

Test scenarios:

- `fast_text` step 的 `model_image_count=0`。
- `visual_update` step 的 `model_image_count<=2`。
- current image 优先于 keyframe。
- 缺失图片进入 `model_missing_image_paths`。

### U3: Episode-local visual memory cache

Files:

- `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- `tests/test_openclaw_cli_plan_gateway.py`

Approach:

- 在 `OpenClawCliPlanPlanner` 内维护 per-episode cache。
- visual_update 成功解析 Qwen 输出后更新 cache。
- fast_text prompt 注入 cache。
- Qwen 输出缺少 visual summary 时仍更新 action，但标记 `visual_memory_update_status=missing_summary`。

Test scenarios:

- visual_update 后 cache 包含 summary/subgoal/action/reason。
- episode 切换后 cache 不串 episode。
- fast_text prompt 包含 cached summary。
- previous visual_update error forces next step visual_update。

### U4: Trace 和 summary evidence

Files:

- `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- `src/harness/openclaw/runtime.py`
- `scripts/summarize_openclaw_vln_ablation.py`
- `tests/test_openclaw_cli_plan_gateway.py`

Trace fields:

```text
planner_step_mode
planner_authority
qwen_api_called
model_image_count
visual_memory_age_steps
visual_memory_valid_until_step
memory_context_used
fast_break_reason
visual_memory_update_status
```

Stage 1 expected values:

```text
planner_authority=qwen
qwen_api_called=true
planner_step_mode in {visual_update, fast_text}
```

Test scenarios:

- trace audit contains all new fields。
- fast_text has `model_image_count=0` and `planner_authority=qwen`。
- visual_update has `model_image_count>0` when image exists。
- summarizer reports step-mode counts and image-count distribution。

### U5: Runbook for Stage 1 evaluation

Files:

- `docs/runbooks/openclaw-qwen-image-gated-memory-planner-eval.md`

Approach:

- Add 1-episode smoke command.
- Add 30-sample command.
- Add trace check script.
- Add validity gates.

Required trace check:

```text
providers ['qwen_api']
session_modes ['stateless_model']
planner_authority ['qwen']
qwen_api_called [true]
planner_step_modes include visual_update and fast_text
fast_text image count always 0
fallback_steps 0
```

## Validation Plan

### Unit tests

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_cli_plan_gateway.py tests/test_evaluation_scripts.py -q
```

### Smoke

Run 1 episode with:

```text
OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=10
OPENCLAW_MODEL_FAST_MODE=qwen_text_only
OPENCLAW_MODEL_MAX_IMAGES=2
HARNESS_DEBUG_MAX_EPISODES=1
MAX_STEPS=400
```

Pass criteria:

```text
fallback_steps 0
visual_update steps have images
fast_text steps have model_image_count=0
planner_authority=qwen on all steps
success result is recorded
```

### 30-sample test

Run after smoke passes.

Compare against:

```text
results/clawnav_openclaw_gateway_30_direct_qwen_api_20260526
```

Metrics:

```text
episode_wall_time
steps
success
spl
model_image_count distribution
planner_step_mode distribution
provider_input_tokens
fallback_steps
```

Go/no-go:

- Go if runtime is materially lower and 30-sample success/SPL does not collapse.
- No-go if fast_text outputs become unstable, fallback appears, success drops sharply,
  or any planner reason starts with `openclaw_cli_model_fallback:` or
  `openclaw_cli_agent_fallback:`.

## Risks and Mitigations

### Risk: Qwen text-only still too slow

Mitigation:

- Stage 1 still valuable because it isolates image cost.
- Stage 2 compresses text prompt.
- Stage 3 skips Qwen on cached fast steps as a separate hybrid method.

### Risk: cached visual memory becomes stale

Mitigation:

- Use conservative interval 8 first.
- Force visual_update on keyframe candidate and missing memory.
- Record `visual_memory_age_steps` and `fast_break_reason`.
- Do not rely on undefined collision/stuck signals in Stage 1.

### Risk: method label becomes misleading

Mitigation:

- Stage 1/2 always record `planner_authority=qwen`.
- Stage 3 must use a different method label: `memory_guided_policy_fast`.
- Do not mix Stage 3 results with Stage 1 direct Qwen planner results.

### Risk: memory backend confusion

Mitigation:

- Stage 1 explicitly uses adapter-local episode memory.
- `spatial_http` real memory is deferred to Stage 3+ or a separate plan.
- Trace should record `visual_memory_backend=adapter_episode_local`.

## Deferred Work

- Add collision/stuck/progress-failure forced refresh after reliable runtime signals exist.
- Add real `spatial_http` visual memory recall.
- Run 30-sample validation for `memory_guided_policy_fast`.
- Add ablation matrix:
  - direct Qwen every step
  - image-gated Qwen text-only
  - image-gated Qwen compressed text
  - memory-guided local-policy fast

## Recommended Next Step

Implement Stage 1 only:

```text
Image-Gated Qwen with planner_authority=qwen
```

This is the smallest change that uses memory, reduces image API cost, and preserves method comparability.
