# Instruction-Stage Aligned Keyframe Retrieval

## 背景

当前 Phase 3 visual readback 已经能在部分 episode 上通过关键帧纠正动作，例如 `215` 和 `277`。但在 `117`、`267` 这类长路径上，readback 后半程仍反复读取最早的 keyframe，例如 `step 0/5/11` 或 `step 0/13/18`，导致早期路线阶段的视觉证据持续控制后续动作。

这个问题的本质不是 keyframe 数量太少，而是 keyframe 没有和 instruction 的执行阶段对齐。系统知道保存了哪些图，但不知道这些图对应“已经完成的哪一步指令”。

## 目标

把 keyframe 从普通历史图像改成“指令阶段的视觉证据”。

运行时维护当前 instruction 执行进度：

```text
stage 0: 从起点走到第一个目标区域
stage 1: 从当前区域转向/进入下一个区域
stage 2: 到达最终停点
```

后续 readback 只优先读取当前阶段、下一阶段相关的 keyframe。已经完成很久的旧阶段 keyframe 可以作为审计背景，但不能反复作为 action override 的主要证据。

## 方案名称

`Instruction-Stage Aligned Keyframe Retrieval v1`

## 设计边界

这版只改 controller-side visual readback 的证据选择和动作接管资格，不把 stage 文字作为新的 `memory_context_text` 塞回 JanusVLN policy。stage 状态是 runtime/controller 维护的结构化元数据，用于：

```text
1. 给 keyframe 打 stage 标签
2. 选择 readback 要读的 keyframe
3. 判断 action override 是否有当前/下一阶段视觉证据支持
4. 在 trace 中解释每次 readback 和 override
```

如果 stage 判断不确定，系统必须降级为审计/记录，不能因为一个不确定的 stage 推进而强制改动作。

## 运行流程

### 1. Episode 开始时拆分 instruction

把自然语言 instruction 拆成有序 stage。

示例：

```text
Walk from closet into bedroom. Turn left into bathroom. Wait on bathroom mat.
```

拆成：

```text
stage 0: Walk from closet into bedroom
stage 1: Turn left into bathroom
stage 2: Wait on bathroom mat
```

每个 stage 记录：

```text
stage_id
stage_text
expected_action_type: move | turn | stop
status: pending | active | completed
```

第一版使用简单规则拆分即可：按句号、then、once、walk、turn、stop 等路线动词或连接词切分，不引入复杂 NLP 模型。

### 2. Runtime 维护 current_stage_id

每个 episode 初始化：

```text
current_stage_id = 0
```

运行过程中，第一版只从 VisualMemoryReadSkill readback 接受 stage 进展候选判断：

```text
predicted_current_stage_id
completed_stage_ids
next_stage_id
stage_evidence
```

这里的输出只能作为候选判断，最终 stage 状态由 runtime 接受或拒绝。stage 进展不能只依赖历史 keyframe 的文字说明，必须包含当前视角证据。

planner visual_update 暂不推进 stage，避免 gateway 视觉缓存和 readback controller 各维护一套进度判断。后续如果要让 visual_update 也更新 stage，需要单独把 `openclaw_cli_plan_gateway.py` 的输出 schema、cache 字段和 trace 字段纳入方案。

stage 更新必须保守：

```text
只允许 i -> i+1
不能大幅跳跃
不能轻易回退
必须有 current view 视觉证据
必须记录 stage_update_source 和 stage_update_reason
```

如果证据不足，则保持原 stage。

第一版不做复杂回退。如果 agent 明显走回旧区域，允许在 trace 里标记：

```text
stage_alignment_status = possible_backtrack
```

但 `current_stage_id` 默认不回退，旧 stage keyframe 只能作为审计背景，不直接触发 action override。

### 3. 保存 keyframe 时绑定 stage

每张 keyframe 写入 memory 时增加 stage metadata：

```text
step_id
image_path
save_reason
candidate_event_type
stage_id
stage_text
stage_status_at_save
stage_confidence
stage_update_source
```

这样 keyframe 不只是历史图，而是某个 instruction stage 的视觉证据。

示例：

```text
step 0  -> stage 0: Walk from closet into bedroom
step 21 -> stage 1: Turn left into bathroom
step 60 -> stage 2: Wait on bathroom mat
```

### 4. Readback 检索 stage-aware top5

触发 VisualMemoryReadSkill 时，选 5 张 keyframe：

```text
top_k = 5

1. current stage 最近 keyframe：最多 2 张
2. current/next stage 最近 decision-point keyframe：最多 2 张
3. current stage entry anchor 或 initial anchor：最多 1 张
```

`current stage entry anchor` 定义为：`current_stage_id` 推进到该 stage 后保存的第一张 keyframe。只有 `current_stage_id == 0` 才允许使用 initial anchor；如果 `current_stage_id > 0` 且当前 stage 没有 keyframe，则 anchor bucket 留空，由 fill bucket 补齐，不能回退到起点图。

去重后如果不足 5 张，从最近到最旧补齐，但优先选择：

```text
stage_id >= current_stage_id - 1
```

也就是说，已经完成很久的旧 stage 不再优先参与 readback。

排序规则固定为：

```text
1. bucket 顺序：current_recent -> current_or_next_decision -> stage_entry_anchor -> fill
2. 同一 bucket 内按 step_id 从新到旧
3. 相同 image_path 去重
4. 最终最多 visual_readback_top_k 张；本方案默认把 visual_readback_top_k 设为 5
```

如果 keyframe 没有 stage metadata，第一版把它当作 `stage_id = unknown`，只允许进入 fill bucket，不能作为 action override 的主要证据。

### 5. Action override 只允许基于当前/下一 stage

readback 仍然可以读取旧 stage keyframe 做审计，但 action override 不能主要依赖旧 stage 证据。

允许 override 的基本条件：

```text
route_conflict
recommended_action 是 immediate_next_action
confidence 足够高
证据包含 current view
关键 keyframe 证据来自 current_stage 或 next_stage
```

禁止情况：

```text
主要证据来自 completed old stage
当前 stage 已推进，但 readback 仍依赖很早的起点 keyframe 改动作
selected keyframe 全部是 stage_id = unknown
stage_alignment_status = possible_backtrack 且当前视角证据不足
```

这不是新增固定 override budget。现有 `visual_readback_max_action_overrides_per_episode=adaptive` 可以保留；新增的是 stage eligibility gate：

```text
stage_override_allowed = has_current_view_evidence
  and has_current_or_next_stage_memory_evidence
  and not old_stage_dominant
```

其中：

```text
has_current_view_evidence:
  VisualMemoryReadSkill 的 evidence_sources/current_only_evidence_count 显示当前图参与了判断。

has_current_or_next_stage_memory_evidence:
  支持 recommended_action 的 matched memory 或 evidence source 至少包含一个
  stage_id in {current_stage_id, current_stage_id + 1} 的 keyframe。

old_stage_dominant:
  支持 recommended_action 的 keyframe 证据全部来自
  stage_id < current_stage_id - 1 或 stage_id = unknown。
```

如果 `stage_override_allowed=false`，readback 仍记录 `audit_action_hint`、`visual_evidence` 和 `recommended_action`，但 `_apply_visual_readback_action_override` 不能执行该动作。

## 预期效果

### 对 `117`

旧行为：

```text
step 370 仍读 step 0/5/11
反复 TURN_RIGHT -> TURN_LEFT
最终左右摆动到 401 步
```

新行为：

```text
step 0/5/11 属于早期 closet/bedroom stage
如果当前 stage 已推进，旧 stage keyframe 不能直接 override
readback 优先读当前 bathroom/bedroom 附近 keyframe
```

### 对 `267`

旧行为：

```text
后半程仍读 oven/kitchen 早期 keyframe
导致厨房阶段证据持续影响 bathroom doorway 阶段
```

新行为：

```text
完成 ovens -> hallway 后，后续优先读 hallway / bathroom doorway stage 的 keyframe
厨房 keyframe 只作为背景审计，不直接控制动作
```

### 对 `215` 和 `277`

这两个成功救回样例主要依赖早期少量纠偏。新策略仍允许早期 stage keyframe 参与 action override，因此应保留这类正向效果。

## Trace 字段

为了可审计，trace 中增加：

```text
instruction_stages
current_stage_id
current_stage_text
stage_update_source
stage_update_reason
stage_alignment_status
selected_keyframe_steps
selected_keyframe_stage_ids
selected_keyframe_stage_texts
selected_keyframe_save_reasons
selected_keyframe_bucket_names
stage_entry_anchor_step
stage_override_allowed
stage_override_block_reason
```

核心判断看：

```text
selected_keyframe_stage_ids 是否围绕 current_stage_id
action override 是否由 current/next stage 支持
旧 stage keyframe 是否只作为审计证据
```

## 代码落点

第一版应复用现有 Phase 3 路径，不新建另一套 memory 系统。

主要落点：

```text
src/evaluation_harness.py
- _runtime_payload: 不维护 stage 状态；继续传 current_image_path/keyframe_target_path/run_id，stage 初始化和更新留在 runtime

src/harness/openclaw/runtime.py
- episode reset: 初始化 instruction_stages/current_stage_id
- _event_gated_keyframe_write_payload: 写入 stage metadata
- _event_gated_smoke_candidate_pool: 从“取最早/现有顺序”改为 stage-aware top5
- _run_visual_readback: 把 instruction_stages/current_stage_id 传给 VisualMemoryReadSkill，并 trace selected keyframe stage/bucket 信息
- _apply_visual_readback_action_override: 增加 stage_override_allowed gate

src/harness/skills/visual_memory_read.py
- input schema 增加 instruction_stages/current_stage_id
- output schema 增加 predicted_current_stage_id/completed_stage_ids/next_stage_id/stage_evidence
- stage_progress 只是候选 evidence，不直接改 runtime stage

src/harness/config.py 或 scripts/run_memory_keyframe_phase23.sh
- 确保该实验的 visual_readback_top_k = 5；当前 HarnessConfig 默认值是 3，不能只在文档里写 top_k=5

tests/test_openclaw_runtime_bridge.py
- 增加 stage-aware candidate pool 和 stage override gate 单测

tests/test_harness_config.py 或现有配置测试
- 覆盖 visual_readback_top_k=5 的配置/launcher约束
```

## 最小实现范围

第一版只做五件事：

```text
1. instruction -> ordered stages
2. runtime 维护 current_stage_id，并记录保守推进证据
3. keyframe metadata 增加 stage_id/stage_text/stage_confidence
4. readback candidate pool 按 current_stage_id 选择 top5
5. action override 增加 stage_override_allowed gate，并在 trace 记录原因
```

暂不做：

```text
跨 episode memory
复杂全局地图
复杂语义检索
STOP 专用 override 策略
固定 override 次数预算
复杂 backtrack 恢复策略
```

## 失败模式和降级

需要明确处理四种非理想路径：

```text
instruction 拆不出多个 stage:
  使用单 stage，退化为最近 keyframe top5，但仍记录 stage_id=0

没有当前 stage keyframe:
  用最近 decision-point keyframe 和 fill bucket 补齐，stage_override_allowed 默认更保守

Qwen/readback 没有给出可靠 stage_progress:
  current_stage_id 不变，只更新 trace，不推进 stage

所有候选 keyframe 都来自 completed old stage:
  readback 可以执行，override 必须 blocked，block_reason=old_stage_dominant
```

## 验收样例

重点跑同一组 20 个 `2azQ1b91cZZ` episode。

必须检查：

```text
117: 不再后半程反复读 step 0/5/11；左右摆动减少
267: 不再后半程反复读 step 0/13/18
215: 仍保持成功
277: 仍保持成功
10: 暂不作为第一版强制救回目标
```

整体目标：

```text
Janus 成功集的退化减少
失败集至少保留 215/277 的救回效果
trace 能解释每次 readback 对应的 instruction stage
```

最低可验收指标：

```text
117: step >= 80 后，selected_keyframe_steps 不能长期固定为 [0,5,11]
267: step >= 80 后，selected_keyframe_steps 不能长期固定为 [0,13,18]
215/277: 成功状态不退化
20 episode: 每次 readback trace 必须包含 stage_override_allowed；被 block 时必须有 stage_override_block_reason
trace: 每次 completed readback 都能看到 selected_keyframe_bucket_names
trace: stage_override_allowed=false 时必须有 stage_override_block_reason；为 true 时 block_reason 可以为空
```

## 一句话总结

这版不是简单多读 keyframe，而是让 keyframe 和 instruction 的执行阶段对齐。系统走到某个地方并确认该阶段完成后，后续 readback 就不再反复依赖上一个地方的 keyframe 来控制动作。
