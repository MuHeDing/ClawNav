# Memory-Gated OpenClaw Harness for JanusVLN

## 背景

当前 `JanusVLN + OpenClaw` 方向的核心创新不是修改 JanusVLN 模型结构，而是在现有 navigation policy 外围构建 OpenClaw-style training-free Harness。原始设计见 `docs/plans/2026-05-08-claw-style-harness-for-fast-janusvln.md`。

该 Harness 将 JanusVLN 抽象为 `NavigationPolicySkill`，并通过 memory、critic、replanner、verification 和 logging 在 inference time 动态决定：

```text
何时感知
何时召回 memory
何时调用 JanusVLN
何时验证 STOP
何时 replan
何时写入关键帧
```

因此，本方案不是削弱 memory，而是将 memory 从简单 prompt augmentation 升级为 Harness 的分层控制信号。

## 当前问题

当前 `memory_guided_policy_fast` 的主要路径是：

```text
Qwen 低频看图
-> 生成 visual_summary / suggested_subgoal / reason
-> fast_text step 跳过 Qwen
-> 将 Cached visual memory / Suggested subgoal / Last Qwen reason 注入 JanusVLN prompt
-> JanusVLN 输出动作
```

这个路径的优势是快：Qwen 不需要每一步都看图，多数 step 仍由 JanusVLN 本地 policy 执行。

但当前方式也带来一个关键风险：JanusVLN 原本适应的是 R2R 风格 instruction，现在额外收到 `Current subgoal`、`Relevant memory`、`Last Qwen reason` 等自由文本，会造成 prompt distribution shift。

从已有 100-episode 对比看，OpenClaw memory-guided fast 并不是整体导航崩溃，而是出现较多 near-miss STOP 问题：`success=0, os=1` 的 episode 明显增多。这说明系统经常接近目标区域，但 STOP 时机或 STOP 判断被 memory/subgoal prompt 干扰。

## STOP 语义的定义

这里说的 STOP 语义，不是原始 R2R instruction 中的 `stop` / `wait`。原始任务指令里的停止描述必须保留，因为它定义了目标位置。

需要避免的是额外 memory prompt 中由 Qwen 生成的到达或终止暗示，例如：

```text
The agent has reached the target.
The destination is visible.
The agent is already at the doorway.
Stop near the sofa.
Successfully arrived.
Wait here.
This is the final location.
You should stop now.
```

这些内容不应作为 `policy_context` 直接注入 JanusVLN。它们可以保留为 `control_context`，由 Harness / Critic / STOP verifier 使用。

## 设计目标

本方案希望同时满足三个目标：

1. 保留并强化 memory 作为 ClawNav / OpenClaw-VLN 的核心创新。
2. 减少自由文本 memory 对 JanusVLN action distribution 的污染。
3. 用非 oracle 的方式降低 wrong STOP、near-miss fail 和 stuck / loop。

明确不做：

```text
不修改 JanusVLN 模型结构
不使用 distance_to_goal / success / SPL / oracle path 做在线决策
不让 OpenClaw action guidance 默认覆盖 JanusVLN action
不把 Qwen 原始 reasoning 全量塞进 JanusVLN prompt
```

## 核心思想

Memory 不再只有一种输出形式。它应被拆成三类上下文：

```text
policy_context
  给 JanusVLN 的极简、动作中性、分布友好的导航线索。

control_context
  给 Harness / Critic / Replanner / STOP verifier 的控制证据。

evidence_context
  保存 keyframe、视觉摘要、Qwen reasoning、trace 和检索记录，用于 recall 与离线分析。
```

换句话说：

```text
不是去掉 memory；
而是让 memory 进入正确的位置。
```

JanusVLN 继续负责低层局部导航，OpenClaw Harness 负责 memory 管理、状态判断、STOP 复核和重规划。

## 模块设计

### 1. Memory Distiller

输入：

```text
Qwen visual_summary
Qwen suggested_subgoal
Qwen reason
retrieved memory
current instruction
```

输出：

```text
policy_context
control_context
evidence_context
```

Memory Distiller 的职责不是总结得更长，而是将 Qwen 原始输出拆解成可控结构。

示例：

```text
Raw Qwen output:
  The agent is facing a hallway and appears close to the final doorway.
  It should continue forward and then stop near the entrance.

policy_context:
  Navigation cue: continue toward the doorway ahead.
  Visible landmarks: hallway, doorway.

control_context:
  possible_goal_region: true
  stop_confidence: medium
  requires_stop_verification: true

evidence_context:
  raw_visual_summary: ...
  raw_qwen_reason: ...
  source_image_paths: ...
```

### 2. PolicyMemoryAdapter

`PolicyMemoryAdapter` 负责将 memory 转成 JanusVLN 可消费的短提示。

允许进入 JanusVLN 的内容：

```text
Navigation cue: continue toward the arched doorway.
Visible landmarks: hallway, sofa, arched doorway.
```

不允许进入 JanusVLN 的内容：

```text
Last Qwen reason: ...
The agent has arrived.
You should stop.
This is the destination.
The task is complete.
```

基本规则：

```text
只保留 landmark / direction / local navigation cue
不保留 Qwen 原始 reasoning
不保留 STOP / arrived / reached / successfully 等到达暗示
不把 action_text 作为 Current subgoal 注入
cue 尽量控制在 1-2 行
与原始 instruction 无关的 memory 不注入
```

这一步不是删除 memory，而是把 memory 翻译成 JanusVLN 更稳定的输入形式。

### 3. Control Memory Critic

`control_context` 不进入 JanusVLN prompt，而是给 Harness 做决策。

它可以包含：

```text
possible_goal_region
stop_confidence
subgoal_confidence
memory_freshness
memory_instruction_overlap
visual_novelty
repeated_observation
low_displacement
action_oscillation
requires_stop_verification
```

这些信号用于：

```text
是否允许 memory cue 注入
是否触发 recall
是否认为 STOP risky
是否需要重新调用 Qwen visual_update
是否需要 replan
```

这里必须保持 non-oracle：不能使用 `distance_to_goal`、`success`、SPL、oracle shortest path 或未来观测。

### 4. STOP Verification Gate

当前最值得优先解决的问题是 wrong STOP 和 near-miss fail。

当 JanusVLN 输出 `STOP` 时，尤其是 memory prompt 下输出 `STOP` 时，进入 STOP verification：

```text
JanusVLN proposes STOP
-> check control_context
-> if low confidence or memory-induced STOP, trigger verification
-> Stage A3 audit_clean_prompt: record clean-prompt action and disagreement, but keep original execution
-> Stage A3b clean_prompt_block: only after audit threshold passes, block unsafe STOP / continue / replan
```

验证方式可以分阶段做：

```text
Stage 1:
  audit_clean_prompt verification
  比较 original instruction 下 JanusVLN 是否仍输出 STOP
  只记录 disagreement，不默认 block

Stage 1b:
  clean_prompt_block verification
  仅在 audit 显示 clean-prompt disagreement 不会大量误伤 true STOP 后启用

Stage 2:
  Qwen current-image verification
  使用当前图像和原始 instruction 判断是否具备目标证据

Stage 3:
  memory consistency verification
  检查 current observation 与 active subgoal / retrieved memory 是否一致
```

STOP verifier 不应读取 oracle distance。

### 5. Subgoal State Machine

当前 `active_subgoal` 是文本，容易变成 prompt 污染源。建议将 subgoal 升级为状态机，但这不是第一版 implementation spec 的默认任务。

第一版先做 grammar-based `safe_cue` 和 `active_subgoal` 注入率统计；只有当 Phase 1/2 指标显示字符串 cue 仍然太脆弱或太弱时，才进入 Subgoal State Machine。

示例结构：

```text
route_stage: seeking_landmark | passing_landmark | entering_region | verify_stop
target_landmark: arched doorway
direction_hint: continue forward
negative_condition: not yet at doorway
confidence: 0.0-1.0
fresh_until_step: 20
```

给 JanusVLN 的仍然只是短 cue：

```text
Navigation cue: continue toward the arched doorway.
```

其余状态由 Harness 使用，不直接注入 policy prompt。

## 推荐系统流程

```text
每个 step:

1. Habitat / evaluator 提供 current observation 和 history frames
2. OpenClaw gateway 判断 step mode
   - visual_update: 调 Qwen 看图
   - fast_text: 跳过 Qwen，使用缓存 memory

3. Memory Distiller 处理 Qwen / retrieved memory
   - policy_context
   - control_context
   - evidence_context

4. PolicyMemoryAdapter 生成 JanusVLN-safe cue

5. Harness 判断是否允许注入 cue
   - memory fresh
   - 与 instruction 有关
   - 不包含 STOP 语义
   - 不与当前控制状态冲突

6. NavigationPolicySkill 调 JanusVLN
   - original instruction
   - optional policy-safe cue

7. 如果 JanusVLN 输出 STOP:
   - STOP Verification Gate
   - A3 audit_clean_prompt 先记录 clean action / memory action / disagreement，不默认改执行动作
   - A3b clean_prompt_block 仅在 audit 阈值通过后启用
   - 如果 blocking verifier 判定 unsafe STOP，则 block STOP / clear cue / refresh Qwen / replan

8. 写入 trace 和 evidence_context
```

## 与原始设计的关系

本方案不推翻 `docs/plans/2026-05-08-claw-style-harness-for-fast-janusvln.md`，而是对其中的 memory 设计做细化。

原始设计已经提出：

```text
JanusVLN = NavigationPolicySkill
Harness = 动态调度与控制系统
Memory = 分层组织视觉经验
Critic = non-oracle progress / risky STOP 检测
Replanner = subgoal-level recovery
```

本方案补充的是：

```text
memory 不应只作为 prompt augmentation
policy_context 和 control_context 必须分开
STOP 相关语义应留给 control side
JanusVLN 只接收 policy-compatible memory cue
```

这使得 ClawNav 的创新点更清楚：不是 retrieval-augmented JanusVLN，而是 memory-aware OpenClaw Harness。

## 实验路线

建议按以下 ablation 顺序推进：

### A0: Current memory_guided_policy_fast

当前方案：

```text
Cached visual memory
Suggested subgoal
Last Qwen reason
-> memory_context_text
-> JanusVLN prompt
```

用途：作为当前 baseline。

### A1: Remove Last Qwen reason

只移除 `Last Qwen reason`，保留 visual summary 和 suggested subgoal。

目标：验证 Qwen reasoning 是否是 STOP 分布偏移主因。

### A2: Policy-safe cue only

启用 `PolicyMemoryAdapter`，只给 JanusVLN 短 landmark / direction cue。

目标：验证安全 memory cue 是否能保留收益、减少 wrong STOP。

### A3: Policy-safe cue + STOP verification

在 A2 基础上加入 audit-first STOP verifier。

```text
A3 = safe_cue + audit_clean_prompt
```

此阶段只记录 clean-prompt action、memory action、disagreement 和 hypothetical block 风险，不默认改变执行动作。

目标：降低 `success=0, os=1` 的 near-miss fail。

### A3b: Policy-safe cue + blocking STOP verification

```text
A3b = safe_cue + clean_prompt_block
```

只有当 A3 audit 证明 clean-prompt disagreement 不会大量误伤 true STOP 时，才运行 A3b。

目标：验证 blocking verifier 是否能进一步降低 near-miss fail，且不增加 timeout / loop 或损失 ClawNav-only success。

### A4: Full memory-gated harness

这是 deferred full-harness 阶段，不等同于 A3b。

启用完整三层 memory：

```text
policy_context
control_context
evidence_context
```

并使用 critic / replanner / verification 动态调度。

目标：验证完整 OpenClaw-style memory harness 是否提升 SR/SPL，同时降低 wrong STOP 和 stuck。

### A5: Clean-off policy-memory ablation

关闭 fast-step policy memory 注入：

```text
no active_subgoal
no memory_context_text
no memory-derived memory_images
```

用途：验证 memory cue 是否真的优于 clean-off，而不是只靠过滤或 STOP control 提升。

## 评估指标

除常规 VLN 指标外：

```text
SR
SPL
OS
NE
average steps
```

需要额外记录 Harness 指标：

```text
near_miss_fail_count          # success=0, os=1
wrong_stop_count              # final_action=STOP but success=0
far_wrong_stop_count          # success=0, os=0, final_action=STOP
timeout_or_loop_count         # steps >= max_steps
stop_verifier_trigger_count
stop_verifier_block_count
blocked_stop_later_success
memory_changed_action_rate
memory_changed_to_stop_rate
policy_context_injection_rate
control_context_recall_rate
stuck_recovery_rate
replan_frequency
fallback_count
qwen_call_count
latency_overhead
prompt_token_overhead
```

还应保留对照集合：

```text
Janus-only success but current ClawNav fail
ClawNav-only success but Janus fail
```

新方案必须尽量恢复 Janus-only success，同时不能大量损失 ClawNav-only success。

## 预期收益

本方案预期解决三类问题：

```text
1. Prompt distribution shift
   减少自由文本 memory 对 JanusVLN 的干扰。

2. Wrong STOP / near-miss fail
   用 control_context 和 STOP verification 降低错误停止。

3. Stuck / loop
   用 memory freshness、visual novelty、action pattern 和 replanner 做非 oracle recovery。
```

## 论文表述建议

可以将该方法表述为：

```text
Memory-Gated OpenClaw Harness
```

或更具体：

```text
Causal Memory Gating for Prompt-Sensitive VLN Policies
```

核心表述：

```text
We do not directly inject raw memory into JanusVLN.
Instead, visual memory is distilled into policy-compatible navigation cues and
control-side verification evidence. The former minimally guides the reactive
navigation policy, while the latter powers OpenClaw-style progress criticism,
STOP verification, memory recall, and replanning.
```

中文表述：

```text
我们不是将记忆作为自由文本直接附加给 JanusVLN，
而是将记忆分解为低层策略可消费的导航线索和高层控制可使用的验证证据。
这使得系统能够在不改变 JanusVLN 模型结构的前提下，
缓解 prompt 分布偏移，并利用 OpenClaw-style Harness 提升长程导航鲁棒性。
```

## 待思考问题

1. `PolicyMemoryAdapter` 的 cue 应该多短？
   - 1 行 cue 是否足够？
   - 是否允许 visible landmarks 列表？

2. A3 audit 到 A3b blocking 的阈值应该多严格？
   - clean prompt verification 成本低，但可能不够强。
   - 必须记录 hypothetical true-positive block rate。
   - Qwen current-image verification 更强，但增加调用成本。

3. `control_context` 的置信度如何计算？
   - rule-based first
   - Qwen self-confidence
   - memory / instruction overlap
   - visual novelty / repeated observation

4. 是否需要双路 Janus 推理？
   - clean prompt vs memory prompt 能更直接检测 memory intervention
   - 但会增加 JanusVLN 调用成本

5. 如何证明 memory 真的赋能而不是只做过滤？
   - 需要记录 memory changed action rate
   - useful recall rate
   - stop blocked later success
   - stuck recovery success
