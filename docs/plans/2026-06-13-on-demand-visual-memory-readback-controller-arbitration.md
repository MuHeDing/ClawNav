# On-demand Visual Memory Readback and Controller-side Arbitration

## 背景

当前 ClawNav + OpenClaw 的 visual-memory 链路已经能做到：

```text
Habitat 当前帧保存为 image_path
-> 关键帧 / 当前帧进入 /plan payload
-> Qwen / OpenClaw visual analyzer 生成文字 observation
-> VisualMemoryCuratorSkill / MemoryWriteSkill 写入 episode-local memory
-> MemoryQuerySkill 召回短文本 memory_context_text
-> NavigationPolicySkill 将 memory_context_text / active_subgoal 拼进 JanusVLN instruction
```

这一条路径的优势是实现成本低、评测速度可控，也适合作为 `A0 raw`、`A1 no_reason`、`A2 safe_cue`、`A5 off` 这一组 prompt-side ablation。

但它仍然存在一个关键问题：视觉记忆最后主要变成了文字提示。即使文字来自图像，它对 JanusVLN 的影响仍然是 prompt augmentation。对于长程 VLN，这不足以支撑“具身环境中的大语言模型构建可控视觉记忆机制”的核心主张。

本方案把下一阶段贡献点收束为：

```text
关键时刻读正确的历史图片
-> 证明 VLM 真的读取了 retrieved memory images
-> 生成可审计视觉证据
-> 用证据检查 JanusVLN candidate action
-> 分阶段证明读图对 controller decision / STOP gate / replan 产生影响
```

主方法不是继续把读图结果写回 JanusVLN prompt，而是把 readback evidence 放到 controller 侧做仲裁。

## 目标

本 spec 要验证三个问题：

1. 系统是否能在关键决策点 query 到正确的历史 memory image。
2. VLM 是否真实读取了这些 retrieved memory images，而不是只依赖 summary 或 image path 字符串。
3. 读图 evidence 是否按阶段改变 controller decision、STOP 判断、replan request 或后续可执行动作。

换句话说，本方法的贡献不是“给模型更多上下文”，而是一个可控、可审计、可验证因果影响的视觉记忆读回机制。

### Claim gate

不同阶段允许声明的结论必须分开：

```text
Phase 1:
  只能声明机制验证：
  - 是否召回正确 memory image；
  - 是否 attach 并读图；
  - grounding falsification 是否通过；
  - controller 是否基于 verifier label 产生 shadow STOP-block 或 replan_request 记录；
  - executed STOP fallback pilot 如果单独运行，只能声明 STOP-specific pilot result。

Phase 1 不能单独声明：
  - TURN_LEFT / TURN_RIGHT 被成功修正；
  - real replanner 改善最终路径；
  - subgoal 被有效更新；
  - V4 在完整 VLN 指标上优于 baseline。

Phase 2 / Phase 3:
  只有当 trace 中存在非零 executed_action_changed_after_visual_read_count、
  replan_executed_after_visual_read_count 或成功的 executed STOP-block case，
  才允许声明 turn/action / replan / subgoal 级因果影响。
```

## 非目标

第一阶段不做：

```text
不修改 JanusVLN 模型结构。
不把所有历史图片塞进 JanusVLN / Qwen 上下文。
不让 Qwen action_hint 直接覆盖 JanusVLN action。
不使用 distance_to_goal / success / SPL / oracle path / future frames 做在线决策。
不把 readback evidence 默认拼进 NavigationPolicySkill prompt。
不把该方法混入 A0/A1/A2/A5 prompt-side ablation 命名。
```

允许保留 `image_read_prompt` 作为 ablation，但它不是主方法。

## 和现有方案的关系

现有文档：

- `docs/visual-memory-keyframe-write-recall-summary.md`
- `docs/视觉信息注入文档.md`
- `docs/plans/2026-06-09-memory-gated-openclaw-harness.md`
- `docs/plans/2026-06-09-memory-gated-openclaw-harness-implementation-spec.md`

定义的是当前 visual memory 写入、召回和 prompt-side memory gating。

本方案复用这些基础能力：

```text
current_image_path / keyframe image_path 保存
episode-local memory_namespace
MemoryWriteSkill
MemoryQuerySkill
VisualMemoryCuratorSkill
NavigationPolicySkill as clean policy
runtime trace / evaluation summary
```

但新增一个明确边界：

```text
retrieved memory image 被 VLM 真实读图后，
其输出不默认进入 JanusVLN prompt，
而是进入 controller-side arbitration。
```

因此它是 current prompt-memory 方案之后的 controller-level visual readback 扩展。

## 当前代码约束

现有 `OpenClawVLNRuntime` 会在 planner 前自动 recall memory，并通过 `_merge_tool_navigation_context(...)` 将 `policy_context.memory_context_text` 和可选 `memory_images` 合入 `nav_payload`。`NavigationPolicySkill` 之后会把 `memory_context_text` 作为 `Relevant memory` 拼进 JanusVLN instruction。

此外，当前 runtime 还可能通过 `MemoryAwareContextEngine.prepare_plan_context(...)` 或等价 context-engine merge，在 `NavigationPolicySkill` payload 构造前写入 `active_subgoal` / `memory_context_text` / `memory_images`。因此 V4 的 clean-policy 判定不能只检查 `MemoryQuerySkill.policy_context` 是否被 merge，而必须检查所有 context source 之后的最终 `NavigationPolicySkill` payload。

因此，`V4_image_read_controller` 必须显式声明为 control-only recall：

```text
V4_image_read_controller:
  NavigationPolicySkill payload 必须保持 clean。
  不允许写入 active_subgoal。
  不允许写入 memory_context_text。
  不允许写入 memory_images。
  MemoryQuerySkill 的 policy_context 只能用于 V1/V3 prompt ablation。
  readback evidence 只能进入 runtime_metadata / controller gate。
  trace 必须记录 used_by_policy=false。
```

如果实现时复用 `MemoryQuerySkill`，必须在 V4 路径上只消费：

```text
memory_hits
control_context
image_path
retrieval metadata
```

不能调用现有自动 merge 路径把 recall 结果注入 `NavigationPolicySkill`。否则 V4 会退化成 prompt augmentation，无法和 `V1_text_only_prompt` / `V3_image_read_prompt` 做干净对照。

V4 的 trace 也不能简单用 `policy_context` 是否非空来推断 `used_by_policy`。因为当前 `MemoryQuerySkill` 可能天然返回 `policy_context.memory_context_text` / `memory_images`，但 V4 control-only 路径并不允许把它 merge 到 `NavigationPolicySkill` payload。实现时必须采用以下二选一合同：

```text
Option A:
  V4 tool_calls / runtime_metadata 中剥离 policy_context，
  只保留 memory_hits / control_context / image_path。

Option B:
  保留 raw policy_context 供审计，
  但 trace 必须分开记录：
    policy_context_available=true|false
    actual_policy_payload_merge=false
    used_by_policy=false
```

`used_by_policy` 必须由 `actual_policy_payload_merge` 推导，而不是由 raw recall 是否产生了 prompt context 推导。需要增加一个测试：`MemoryQuerySkill` 返回非空 `memory_context_text`，但 V4 的 `NavigationPolicySkill` payload 仍为空 memory 字段，trace 记录 `used_by_policy=false`。

V4 control-only 合同必须同时覆盖 context engine：

```text
image_read_controller:
  disable MemoryAwareContextEngine plan-context injection
  或在所有 context sources merge 完成后，
  从最终 NavigationPolicySkill payload 中剥离：
    active_subgoal
    memory_context_text
    memory_images

trace 必须记录：
  context_engine_context_available=true|false
  context_engine_policy_context_stripped=true|false
  final_policy_payload_has_memory=false
```

也就是说，`used_by_policy=false` 的来源必须是最终 payload 审计，而不是“某一个 recall path 没有 merge”的局部判断。

## 总体架构

主流程是双分支：

```text
Current observation + instruction
        |
        v
NavigationPolicySkill / JanusVLN clean policy
        |
        v
candidate_action
        |
        v
Readback trigger gate
  - candidate-independent signals:
      decision point / corner / junction / door
      route uncertainty
      repeated actions / loop
  - candidate-dependent signals:
      risky STOP
      turn-at-decision-point risk
        |
        v
------------------------------------------------------------
Branch B: Visual Memory Readback Stream

MemoryQuerySkill
        |
        v
Top-K memory hits
  - memory_id
  - summary / retrieval_text
  - image_path
  - retrieval_confidence
        |
        v
VisualMemoryReadSkill
        |
        v
Qwen / VLM reads:
  current image + retrieved memory images
        |
        v
structured visual readback evidence
  - actually_read_image_paths
  - matched_memory_ids
  - verifier_labels
  - visual_evidence
  - audit_relative_direction_hint
  - audit_action_hint
  - readback_confidence

------------------------------------------------------------
candidate_action + visual readback evidence
        |
        v
Controller-side arbitration / STOP verifier / Replanner
        |
        v
final_action
  - must be executable Habitat action:
    STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
  - controller meta-decisions are separate:
    block_stop / replan_request_logged / evidence_support
        |
        v
trace + counterfactual verification
```

核心区别：

```text
JanusVLN 仍然先给 candidate action。
Readback trigger 在 candidate_action 之后评估，避免 risky_stop / action_conflict 数据流成环。
VLM readback 只提供外部证据。
controller 根据证据检查、阻止、记录 replan request 或记录 support。
route-memory conflict 只能在 retrieved memory image 被读取后，由 `route_conflict` / `route_support` verifier label 计算为 post-retrieval classification，不能作为 Phase 1 的 pre-retrieval trigger。
`turn-at-decision-point risk` 只能表示 candidate_action 是 TURN_LEFT / TURN_RIGHT 且满足
decision_point / route_uncertainty 这类 online trigger；不能表示已经发生 action_conflict。
```

## Memory 记录格式

历史关键帧继续以 summary-indexed image path 形式存储。summary 用于检索，image_path 用于关键时刻真实读图。

```json
{
  "memory_id": "episode_x_step_000120",
  "scene_id": "...",
  "episode_id": "...",
  "step_id": 120,
  "image_path": "results/.../keyframes/step_000120.png",
  "summary": "hallway with two possible exits",
  "retrieval_text": "hallway two exits left opening painting",
  "navigation_relevance": "left corridor leads toward target room",
  "objects": ["doorway", "table"],
  "landmarks": ["painting", "arched doorway"],
  "spatial_cues": ["left opening", "right wall"],
  "action_context": {
    "last_action": "MOVE_FORWARD",
    "next_action": "TURN_LEFT"
  },
  "memory_scope": "episode",
  "memory_namespace": "episode:<scene_id>:<episode_id>",
  "source_image_role": "keyframe"
}
```

第一阶段仍以 episode-local memory 为默认边界，避免跨 episode 召回引入未来信息或无关视觉记忆。

## Key Decision Trigger

该方法不能每一步都读历史图。触发器负责把 VLM readback 限制在关键节点。

触发器分成两类。只有 online controller triggers 可以用于正式 V4 评测；offline diagnostics 只能用于 debug、case study 或预注册固定集合的构建，不能在 episode 运行中读取 `success`、`os`、`distance_to_goal`、oracle path 或未来帧。

### Online controller triggers

第一阶段只允许使用当前 observation、当前/历史 action、candidate_action、非 oracle runtime state 和当前 episode 内已写入 memory。

```text
decision_point:
  candidate_action 是 TURN_LEFT / TURN_RIGHT，
  且 current_visual_observation / latest visual summary 中出现
  intersection / junction / fork / doorway / left opening / right opening /
  two exits / corridor split 等分叉信号。

risky_stop:
  candidate_action 是 STOP，
  且当前 observation 没有明确 goal_visible / target_landmark_visible 证据。

route_uncertainty:
  当前 observation 和 instruction landmark 对齐不明确。

stuck_or_loop:
  连续动作重复、视觉重复、route progress 不变或 action oscillation。
  Phase 1 只记录，不触发 recovery。
```

`action_conflict` 不是 Phase 1 的 readback trigger。它只能在 `MemoryQuerySkill -> VisualMemoryReadSkill` 之后，由 controller 根据 `route_conflict` / `route_support` verifier label 计算为 post-retrieval classification。这样不会出现“需要 query 结果才能决定是否 query”的循环依赖。

### Offline diagnostics

```text
near_miss_recovery:
  离线诊断中出现过 success=0, os=1、wrong turn、early STOP 或 loop 风险的位置。
  只能用于生成 debug fixtures 或 offline_stress_set。
  不能进入 primary_quantitative_set。
  不能作为 online trigger。
```

在线触发 trace 必须记录：

```text
trigger_source = online_controller | static_pre_run | offline_fixture
trigger_rule
candidate_action
oracle_fields_used = []
```

第一阶段优先实现：

```text
risky_stop
decision_point
```

`stuck_or_loop` 第一阶段只记录触发和 disagreement，不做 recovery；真正的 stuck / loop recovery 放到 Phase 3。

## MemoryQuerySkill 合同

`MemoryQuerySkill` 只负责检索，不负责决策。

输入示例：

```json
{
  "text": "go down the hallway and turn at the painting",
  "step_id": 134,
  "visual_observation": "facing a hallway intersection",
  "reason": "decision_point",
  "n_results": 3,
  "allowed_scopes": ["episode"],
  "memory_namespace": "episode:<scene_id>:<episode_id>"
}
```

如果上游 controller 持有的是更语义化的字段，进入现有 `MemoryQuerySkill` 前必须做显式映射：

```text
instruction -> text
current_visual_observation -> visual_observation
trigger_rule -> reason
candidate_action -> planner_reason or trace-only metadata
```

`candidate_action` 可以进入 query construction 的 `planner_reason`，也可以只写入 trace；不能假设现有 `MemoryQuerySkill` 会直接读取 `candidate_action` 或 `trigger_rule`。

输出示例：

```json
{
  "query_text": "hallway intersection turn at painting",
  "memory_hits": [
    {
      "memory_id": "mem_001",
      "step_id": 58,
      "image_path": ".../keyframes/step_000058.png",
      "summary": "similar hallway before the corner",
      "retrieval_confidence": 0.82,
      "retrieval_reason": "matching landmark and intersection"
    }
  ]
}
```

该模块只回答：

```text
哪些历史图片可能相关？
它们在哪里？
为什么被召回？
```

它不读图、不输出最终动作、不直接修改 policy payload。

## VisualMemoryReadSkill 合同

`VisualMemoryReadSkill` 是新增核心模块。它接收当前图像与 Top-K retrieved memory images，调用 Qwen / VLM 真实读取图片，并输出结构化证据。

第一版实现应优先复用现有 OpenClaw / Qwen image capability。`VisualMemoryReadSkill` 只是 readback 语义 wrapper，除非现有 image capability 无法支持多图输入、JSON-only 输出或 per-image trace，才新增独立 adapter。

输入：

```json
{
  "instruction": "...",
  "candidate_action": "TURN_RIGHT",
  "trigger_rule": "decision_point",
  "current_image_path": ".../step_000134.png",
  "memory_hits": [
    {
      "memory_id": "mem_001",
      "image_path": ".../step_000058.png",
      "summary": "similar hallway before the corner"
    }
  ]
}
```

读图 prompt 应强调导航证据，而不是泛泛 caption：

```text
You are reading retrieved visual memory for embodied navigation.

Current image: image_0
Retrieved memory images: image_1..image_k

Task:
1. Compare the current view with retrieved memory images.
2. Identify whether any retrieved image contains visual evidence useful for the next turn.
3. Do not rely only on text summaries.
4. Return JSON only.
```

输出：

```json
{
  "actually_read_image_paths": [
    ".../current/step_000134.png",
    ".../keyframes/step_000058.png"
  ],
  "matched_memory_ids": ["mem_001"],
  "verifier_labels": ["route_conflict", "target_landmark_visible"],
  "visual_evidence": "The recalled image shows the painting before a left opening, matching the instruction landmark.",
  "audit_relative_direction_hint": "left",
  "audit_action_hint": "TURN_LEFT",
  "readback_confidence": 0.81,
  "retrieval_confidence": 0.82,
  "verifier_confidence": 0.81,
  "visual_grounding_status": "attached_only",
  "reason": "The current intersection resembles the recalled keyframe and the target landmark is on the left branch."
}
```

要求：

```text
actually_read_image_paths 必须来自实际 attach 给 VLM 的图片列表。
model_image_count 必须等于 current image + memory image 数量。
matched_memory_ids 必须只引用本次 readback 实际 attach 给 VLM 的 memory records；
V4 中这些 records 是 query hits，V5 中这些 records 是 shuffled replacements，
original retrieved IDs 只能单独记录为 original_retrieved_memory_ids。
retrieval_confidence、readback_confidence 和 verifier_confidence 必须分开记录。
readback_confidence < 0.6 时不得改变动作。
audit_action_hint / audit_relative_direction_hint 只能进入 trace，不得直接进入 controller gate。
```

第一阶段 controller 只允许读取以下枚举 verifier labels：

```text
goal_visible
goal_not_visible
target_landmark_visible
route_support
route_conflict
repeated_keyframe
insufficient_evidence
```

`visual_evidence` 和 `reason` 是审计文本，不是动作命令。任何控制流改变必须引用一个 verifier label，而不是引用自由文本 reason 或 `audit_action_hint`。

### Visual grounding falsification

`actually_read_image_paths` 只能证明图片被 attach，不能证明 VLM 使用了像素。每个被用于论文证据的 readback case 必须额外跑 falsification checks：

```text
summary_redacted:
  去掉 memory summary / retrieval_reason，只保留 image role 和 image id。

current_only:
  只给 current image，不给 retrieved memory images。

retrieved_only:
  只给 retrieved memory images，不给 current image。

blank_or_occluded_memory:
  用空白图或遮挡后的 memory image 替代正确 memory image。

shuffled_memory:
  用同样数量的错误 memory images 替代正确 memory image。
```

`visual_grounding_status` 必须按 label 类型判定，不能用同一个全局 pass rule
套所有 verifier labels：

```text
historical_memory_grounded labels:
  route_support / route_conflict / target_landmark_visible / repeated_keyframe
  correct memory image -> required verifier label / evidence 可复现；
  summary_redacted -> evidence 仍能从图像得出；
  current_only 或 blank/occluded/shuffled -> memory-dependent evidence 消失
  或变为 insufficient_evidence；
  matched_memory_ids 只来自本次 attached memory images。

current_view labels:
  goal_visible / goal_not_visible / insufficient_evidence
  current_only 可以仍然成立；
  这类 label 只能计入 current_view_stop_verification_count，
  不能单独计入 historical_memory_stop_causal_count。

mixed labels:
  如果 route_support / route_conflict 同时依赖 current image 和 memory image，
  trace 必须记录 required_images=current_and_memory。
  retrieved_only 不必失败；但 current_only、blank/occluded/shuffled 必须移除
  memory-dependent evidence。
```

否则只能标记为：

```text
attached_only
```

不能作为“模型真的读了 retrieved memory image”的主证据。

### Grounding eval protocol

`visual_grounding_status=grounded` 不能由单次自由文本判断。每个用于论文证据的 case 必须保存完整 grounding eval trace：

```json
{
  "grounding_eval_protocol": {
    "model": "qwen-or-openclaw-adapter-name",
    "model_version": "...",
    "prompt_version": "visual_memory_read_v1",
    "decoding": {
      "temperature": 0,
      "top_p": 1
    },
    "repeat_count": 1,
    "pass_rule": "label_level_all_required_arms"
  }
}
```

判定规则：

```text
1. 首选 deterministic decoding：temperature=0，repeat_count=1。

2. 如果后端无法保证 deterministic decoding：
   repeat_count=3，至少 2/3 次产生相同 required verifier labels 才算可复现。

3. evidence disappears 的定义不是自由文本变短，
   而是 required verifier labels 消失，
   或 verifier_labels 变成 insufficient_evidence。

4. 每个 falsification arm 都必须落 trace：
   original、summary_redacted、current_only、retrieved_only、
   blank_or_occluded_memory、shuffled_memory。

5. 只有 label-level pass/fail 通过时，
   才能把 original case 计入 visual_grounded_readback_count。
```

### Verifier-label correctness adjudication

Grounding falsification 只能证明 verifier label 依赖图像像素，不能证明 label 本身正确。因此 Phase 1b 的 fixed_replay_manifest 必须增加独立 verifier-label adjudication：

```text
1. 在 V1-V5 + V4c 对照运行前冻结每个 case 的 required verifier labels。
2. 标注来源可以是人工标注、预注册规则或两者结合，但不能来自 V-mode 运行结果。
3. 标注输入必须盲化：
   - 标注者不能看到 V-mode 名称、controller 输出或最终任务结果。
   - 标注者只能看到 current image、candidate memory image、
     instruction 片段和冻结 trigger_rule。
   - why_text_is_insufficient / why_image_is_needed 只能在 case set 冻结后标注。
4. 至少记录 annotator_count、agreement_rate 和 conflict_resolution_rule。
5. 每个 label 单独判定：
   route_support
   route_conflict
   goal_visible
   goal_not_visible
   target_landmark_visible
   repeated_keyframe
   insufficient_evidence
6. 报告 per-label accuracy / confusion table。
7. 只有同时满足 visual_grounding_status=grounded
   且 verifier label 与 adjudicated label 一致的 case，
   才能作为 controller intervention 的主要证据。
```

如果 label-level grounding 通过但 correctness adjudication 失败，trace 可以保留为 `grounded_but_wrong_label`。这类 case 可以说明 VLM 使用了图像，但不能计入 `adjudication_correct_readback_count`，也不能作为 action / STOP / replan causality 的正例。

## Controller-side Arbitration

controller 的输入是：

```text
JanusVLN candidate_action
+ VisualMemoryReadSkill structured evidence
+ trigger_rule
+ current runtime state
```

主规则：

```text
readback_confidence < 0.6:
  log only，不影响动作。

0.6 <= readback_confidence < 0.8:
  只作为 audit evidence。
  如果 evidence 与 candidate_action 冲突，记录 disagreement。

readback_confidence >= 0.8:
  只在受限场景中影响控制流。
```

允许影响控制流的受限场景：

```text
1. risky STOP:
   candidate_action == STOP，
   verifier_labels 包含 goal_not_visible 或 insufficient_evidence，
   controller 可以 block STOP。

2. ambiguous turn:
   JanusVLN 在路口输出 TURN_LEFT / TURN_RIGHT，
   verifier_labels 包含 route_conflict。
   Phase 1 只记录 replan_request_logged=true，不执行 replanner。
   Phase 2 以后才允许请求 clean-policy reassessment 或 replanner。

3. stuck / loop:
   verifier_labels 包含 repeated_keyframe。
   Phase 1 只记录；Phase 3 才允许 recovery。

4. agreement support:
   verifier_labels 包含 route_support 或 target_landmark_visible。
   controller 不改变动作，只记录 evidence_support。
```

Phase 1a 的主报告默认使用 shadow STOP-block，不改变 Habitat 执行动作：

```text
OPENCLAW_VISUAL_READBACK_STOP_FALLBACK_POLICY=log_only

if candidate_action == STOP
and readback_confidence >= OPENCLAW_VISUAL_READBACK_HIGH_CONFIDENCE
and verifier_labels contains goal_not_visible or insufficient_evidence:
    controller_decision = block_stop_shadow
    final_action = candidate_action
    executed_action_changed_after_visual_read = false
```

如果要运行 executed STOP fallback pilot，必须作为单独 denominator 报告，并且
final_action 仍必须是可执行 Habitat action，不能把 `REPLAN` / `BLOCK_STOP` /
`WAIT_FOR_REPLAN` 这类 meta decision 传给 executor。pilot 合同如下：

```text
OPENCLAW_VISUAL_READBACK_STOP_FALLBACK_POLICY=previous_non_stop_else_move_forward

if candidate_action == STOP
and readback_confidence >= OPENCLAW_VISUAL_READBACK_HIGH_CONFIDENCE
and verifier_labels contains goal_not_visible or insufficient_evidence:
    blocked_action = STOP
    if last_executed_non_stop_action in {MOVE_FORWARD, TURN_LEFT, TURN_RIGHT}
       and last_executed_non_stop_action_age <= 3:
        final_action = last_executed_non_stop_action
        fallback_source = previous_non_stop
    else:
        final_action = MOVE_FORWARD
        fallback_source = default_move_forward
else:
    final_action = candidate_action
```

`last_executed_non_stop_action` 必须由 `OpenClawVLNRuntime` 或等价 runtime controller 明确持有，不能隐式依赖 planner reason、working-memory append 顺序或外部脚本。状态合同如下：

```text
state_owner = OpenClawVLNRuntime
state_key = (scene_id, episode_id)
reset_when scene_id / episode_id changes
update_after final_action is executed
track:
  last_executed_non_stop_action
  last_executed_non_stop_action_age

nil path:
  last_executed_non_stop_action missing
  -> fallback_source=default_move_forward
```

STOP-block trace 必须额外记录：

```text
last_executed_non_stop_action
last_executed_non_stop_action_age
fallback_source
fallback_denominator=shadow_primary|executed_pilot
```

如果 executed pilot 无法做 valid-action / collision / progress 安全检查，
必须回退到 shadow-only：

```text
OPENCLAW_VISUAL_READBACK_STOP_FALLBACK_POLICY=log_only
```

此时必须记录 `controller_decision=block_stop_shadow`、`executed_action_changed_after_visual_read=false`，并且不能把该 run 用作“executed action causality”的证据。

STOP-block 还必须区分 current-view verifier 和 historical-memory readback effect。只有满足以下条件的 STOP-block case，才能作为 visual-memory causal evidence：

```text
visual_grounding_status=grounded
current_only arm 移除 decisive verifier label
retrieved memory image 的 adjudicated label 与 controller label 一致
```

如果 STOP-block 只依赖当前图像即可得到 `goal_not_visible` / `insufficient_evidence`，则只能计入：

```text
current_view_stop_verification_count
```

不能计入：

```text
historical_memory_stop_causal_count
```

STOP-block trace 必须包含：

```json
{
  "blocked_action": "STOP",
  "fallback_policy": "log_only",
  "fallback_source": "log_only",
  "fallback_denominator": "shadow_primary",
  "final_action": "STOP",
  "executed_action_changed_after_visual_read": false,
  "stop_block_reason": "goal_not_visible"
}
```

executed pilot 另行记录：

```json
{
  "blocked_action": "STOP",
  "fallback_policy": "previous_non_stop_else_move_forward",
  "fallback_source": "previous_non_stop|default_move_forward",
  "fallback_denominator": "executed_pilot",
  "final_action": "MOVE_FORWARD|TURN_LEFT|TURN_RIGHT",
  "executed_action_changed_after_visual_read": true,
  "stop_block_reason": "goal_not_visible"
}
```

第一阶段建议只启用：

```text
block risky STOP
log replan_request on high-confidence turn conflict
log agreement / disagreement
```

第一阶段不得直接执行 `Qwen action_hint`，也不得把 `audit_action_hint` 传给 replanner。否则该方法会退化为另一层 prompt-driven policy。

## Downstream 使用边界

主方法：

```text
VLM readback evidence -> controller arbitration
```

不默认走：

```text
VLM readback evidence -> text prompt -> NavigationPolicySkill
```

为了论文和实验完整性，可以保留 prompt 版作为 ablation：

```text
V1_text_only_prompt:
  只注入 summary / memory_context_text。

V2_path_only:
  只给 image_path 字符串，不 attach 图片。

V3_image_read_prompt:
  VLM 读历史图，但把读图文字再拼进 JanusVLN prompt。

V4_image_read_controller:
  VLM 读历史图，输出 structured evidence，
  evidence 进入 controller arbitration。
  这是主方法。

V4c_current_only_controller:
  VLM 只读当前图，走同一个 controller arbitration；
  用于区分 current-view STOP verification 和 historical-memory readback effect。
```

## Trace 合同

每次 readback 都必须写入 trace。trace 的目的不是只记录“调用了模型”，而是证明：

```text
query 了哪些 memory image；
实际 attach 给 VLM 的图片有哪些；
VLM 输出了哪些 visual evidence；
evidence 是否进入 NavigationPolicySkill；
visual grounding falsification 是否通过；
controller 是否因此改变下游决策；
如果改变，改变原因是什么；
如果不改变，是否是 confidence / gate 不满足。
```

示例：

```json
{
  "visual_memory_read": {
    "trigger_rule": "decision_point",
    "retrieved_image_paths": [".../keyframes/step_000058.png"],
    "actually_read_image_paths": [
      ".../current/step_000134.png",
      ".../keyframes/step_000058.png"
    ],
    "model_image_count": 2,
    "read_status": "completed",
    "policy_context_available": true,
    "actual_policy_payload_merge": false,
    "used_by_policy": false,
    "readback_state_used_by_policy": false,
    "matched_memory_ids": ["mem_001"],
    "verifier_labels": ["route_conflict", "target_landmark_visible"],
    "visual_evidence": "The recalled image shows the left opening near the target landmark.",
    "audit_relative_direction_hint": "left",
    "audit_action_hint": "TURN_LEFT",
    "readback_confidence": 0.81,
    "retrieval_confidence": 0.82,
    "verifier_confidence": 0.81,
    "visual_grounding_status": "grounded",
    "falsification": {
      "summary_redacted_passed": true,
      "current_only_removed_evidence": true,
      "retrieved_only_removed_evidence": true,
      "blank_memory_removed_evidence": true,
      "shuffled_memory_removed_evidence": true
    },
    "visual_readback_config": {
      "mode": "image_read_controller",
      "top_k": 3,
      "control_only": true,
      "fixed_case_manifest_path": "results/.../visual_readback_fixed_manifest.json",
      "stop_fallback_policy": "log_only"
    }
  },
  "downstream_effect": {
    "candidate_action": "TURN_RIGHT",
    "controller_decision": "log_replan_request",
    "replan_requested": true,
    "final_action": "TURN_RIGHT",
    "controller_decision_changed_after_visual_read": true,
    "executed_action_changed_after_visual_read": false,
    "controller_reason": "verified route_conflict; Phase 1 logs replan request without executing replanner"
  }
}
```

可选 shadow counterfactual：

```json
{
  "counterfactual": {
    "text_only_action": "TURN_RIGHT",
    "path_only_action": "TURN_RIGHT",
    "image_read_prompt_action": "TURN_RIGHT",
    "image_read_controller_decision": "log_replan_request",
    "image_read_final_action": "TURN_RIGHT",
    "controller_decision_changed": true,
    "executed_action_changed": false
  }
}
```

## 实验对照

建议把 readback 方法单独编号为 `V*`，避免和 `A0/A1/A2/A5` 的 prompt-side memory gating 混淆。

```text
V0_clean_off:
  不使用 visual memory readback。
  可对应 A5 clean-off 或 clean NavigationPolicySkill。

V1_text_only_prompt:
  query memory，只注入文字 summary / memory_context_text。

V2_path_only:
  query memory，给 image_path 字符串，但不 attach 图片。

V3_image_read_prompt:
  VLM 读取 retrieved image，再把读图结果以文字形式拼进 NavigationPolicySkill prompt。

V4_image_read_controller:
  VLM 读取 retrieved image，输出 structured visual evidence，
  evidence 进入 controller arbitration。
  这是主方法。

V4c_current_only_controller:
  VLM 只读取 current image，不 attach retrieved memory images，
  evidence 进入同一个 controller arbitration。
  这是历史记忆因果 claim 的 current-view baseline。

V5_shuffled_image_read_controller:
  attach 错误或随机 memory images，走同样 readback + controller，
  作为负控制。
  replacement images 必须预先标注为 wrong / unrelated；
  否则该 case 标为 negative_control_unjudgeable。

V6_direct_policy_images:
  将 retrieved images 直接输入 NavigationPolicySkill / JanusVLN，
  作为后续强 ablation，不作为第一阶段主方法。
```

理想验证关系：

```text
V4 > V1
V4 > V2
V4 > V3
V4 > V4c
V4 > V5
```

这些关系只能在 Candidate-set gate 和 Fixed-manifest gate 都通过后，基于
`fixed_replay_manifest` 的 fixed-trigger replay protocol 报告。Phase 1 只解释为
grounding / controller-decision 层面的机制优势。完整 VLN 指标提升、turn correction、
replan improvement 或 subgoal improvement 必须等 Phase 2 / Phase 3 的 executed
intervention trace 支撑。

每个 `V4 > V*` claim 必须在运行前绑定一个 endpoint，不能从多个 readback metrics 中事后挑选有利指标。Phase 1 默认 endpoint 表如下：

| Claim | Primary endpoint | Denominator | Pass rule | Failure handling | Allowed interpretation |
|---|---|---|---|---|---|
| `V4 > V1` | adjudicated_controller_support_rate | fixed_replay_manifest judgeable cases | V4 produces grounded, adjudication-correct controller evidence more often than text-only prompt produces the same adjudicated support | replay_state_mismatch / replay_mismatch / unjudgeable labels excluded and counted | controller evidence has value beyond text-only summary |
| `V4 > V2` | visual_grounded_readback_rate | fixed_replay_manifest judgeable cases with image_path | V4 grounded rate > path-only grounded rate | missing image_path cases excluded and counted | gain is not from image path strings |
| `V4 > V3` | paired_adjudicated_support_rate | fixed_replay_manifest judgeable cases | V4 and V3 are compared on the same adjudicated support/conflict target; V4 may claim superiority only on shared label correctness, while control-only cleanliness is reported separately | replay_state_mismatch, prompt-contaminated, or missing shared labels reported separately | controller-side arbitration preserves attribution while matching prompt-image evidence |
| `V4 > V4c` | historical_memory_incremental_effect_rate | fixed_replay_manifest cases where current_only is insufficient | V4 produces adjudication-correct memory-dependent evidence that V4c_current_only_controller cannot produce | current_only-sufficient cases excluded from this endpoint and counted separately | effect depends on historical memory rather than current-view verification |
| `V4 > V5` | historical_memory_dependent_effect_rate | fixed_replay_manifest cases with pre-labeled wrong V5 replacements | V4 effect survives and V5 wrong-image effect disappears or becomes insufficient_evidence | negative_control_unjudgeable excluded and counted | effect depends on the correct retrieved memory image |

每个 endpoint 还必须在运行前写入：

```text
minimum_effect_size
paired_test_or_confidence_interval
downgrade_to_descriptive_rule
```

如果样本量或 effect size 不足以支持 paired statistical test / confidence interval，
表中 endpoint 只能作为 descriptive mechanism result 报告，不能写成显著优于 baseline。
任何 SR / SPL / NE / nDTW 提升都属于 Phase 2+ 或 Phase 3+ executed-intervention claim。

解释：

```text
V4 > V1:
  说明只给文字 summary 不够，真实图像读回有收益。

V4 > V2:
  说明收益不是 image_path 字符串或检索 side effect 带来的。

V4 > V3:
  说明在相同 adjudicated support/conflict target 上，controller-side evidence
  arbitration 的可归因效果不弱于 image-read-prompt。

V4 > V4c:
  说明收益来自 historical memory，而不是只看 current image 的 STOP / route verifier。

V4 > V5:
  说明收益依赖正确 memory image，不是随机图片扰动。
```

## V-mode 启动合同

实现时需要把 `V*` ablation 做成显式模式，而不是靠临时脚本分叉。

建议新增统一开关：

```text
OPENCLAW_VISUAL_READBACK_MODE=
  off
  text_only_prompt
  path_only
  image_read_prompt
  image_read_controller
  current_only_controller
  shuffled_image_read_controller

OPENCLAW_VISUAL_READBACK_TOP_K=3
OPENCLAW_VISUAL_READBACK_TIMEOUT_MS=90000
OPENCLAW_VISUAL_READBACK_LOW_CONFIDENCE=0.6
OPENCLAW_VISUAL_READBACK_HIGH_CONFIDENCE=0.8
OPENCLAW_VISUAL_READBACK_SHUFFLE_SCOPE=same_scene_other_episode
OPENCLAW_VISUAL_READBACK_SHUFFLE_SEED=...
OPENCLAW_VISUAL_READBACK_CONTROL_ONLY=1
OPENCLAW_VISUAL_READBACK_FIXED_CASE_MANIFEST=...
OPENCLAW_VISUAL_READBACK_STOP_FALLBACK_POLICY=log_only
```

Phase 0 / Phase 1 的 mode support 到 `V5_shuffled_image_read_controller` 为止，
并包含 `V4c_current_only_controller` 作为 current-view baseline。
`direct_policy_images` 只在 Phase 4 新增，不能提前出现在第一阶段 runner 的默认可选值里。

实现顺序必须分层，避免在 Phase 0 之前提前构建完整矩阵：

```text
Phase 0 implementation only:
  audit existing A0/A1/A2/A5 traces
  verify write-then-query image-backed memory
  run multi-image JSON readback smoke
  produce candidate_case_set
  report go / downscope / no-go

Only after Common gate and Phase 1a minimal-loop gate pass:
  add Phase 1a minimal V4 controller loop

Only after Candidate-set gate pass:
  add fixed_replay_manifest generator from candidate_case_set
  verify frozen_policy_inputs capture for replay cases

Only after Fixed-manifest gate passes:
  add V1-V5 + V4c mode runner
  add fixed replay matrix execution
  add summary / trace parser for endpoint table

If Phase 0 chooses STOP-only or turn-only downscope:
  implement only the V-modes and metrics needed by that narrower study
  do not claim V4 > V1/V2/V3/V4c/V5 as the full matrix result
```

需要在 implementation plan 中落到以下文件或等价位置：

```text
src/harness/config.py
src/evaluation_harness.py
src/harness/openclaw/runtime.py
src/harness/skills/memory_query.py 或调用侧 payload adapter
src/harness/skills/visual_memory_read.py
scripts/run_memory_guided_fast_large_eval.py 或新的 V-mode runner
scripts/run_openclaw_vln_ablation_matrix.py
summary / trace parser
```

`image_read_controller` 模式必须强制：

```text
OPENCLAW_VISUAL_READBACK_CONTROL_ONLY=1
used_by_policy=false
actual_policy_payload_merge=false
```

配置落地规则：

```text
1. 新增 HarnessConfig 字段：
   visual_readback_mode
   visual_readback_top_k
   visual_readback_timeout_ms
   visual_readback_low_confidence
   visual_readback_high_confidence
   visual_readback_shuffle_scope
   visual_readback_shuffle_seed
   visual_readback_control_only
   visual_readback_fixed_case_manifest_path
   visual_readback_stop_fallback_policy

2. 新增 argparse flags，优先级固定为：
   CLI flag > environment variable > HarnessConfig default。

3. build_harness_config 必须 fail fast：
   - unknown visual_readback_mode -> error；
   - low_confidence >= high_confidence -> error；
   - image_read_controller 且 control_only=false -> error；
   - visual_readback_mode != off 且 memory backend 不能写入并召回非空 image_path -> error；
   - fixed-set Phase 1b 没有 manifest -> error；
   - shuffled mode 没有 shuffle_seed -> error。

4. 每个 trace / summary 必须记录最终生效的 visual_readback_config。
```

Phase 0 必须包含 image-backed memory smoke test：

```text
MemoryWriteSkill writes episode-local keyframe record with image_path
MemoryQuerySkill retrieves the same episode-local record
retrieved hit contains non-empty image_path
image_path points to an existing readable file
```

如果当前 backend 是 synthetic / fake memory，或者只能返回 text summary 而不能返回真实 `image_path`，则 V1-V5 + V4c fixed-set matrix 不能启动。此时只能输出 Phase 0 audit report，或先实现 image-backed episode-local memory client。

`image_read_prompt` 是 Phase 1 中唯一允许把 readback 产物进入 `NavigationPolicySkill` 的 V-mode。`direct_policy_images` 的 policy-image 输入只属于 Phase 4 强 ablation。

## 场景筛选协议

研究动机关注“文字 summary 不够、需要看图”的场景，但 primary set inclusion
不能直接使用这个判断。正式预筛选必须完全来自 online controller trigger 或运行前静态字段；
`why_text_is_insufficient` / `why_image_is_needed` 只能在 case set 冻结后作为盲标注描述字段。
否则 V4 的优势会变成按预期结果筛出来的。

实验集必须拆成两类，不能混用：

```text
primary_quantitative_set:
  只能使用 online controller trigger 或运行前静态信息筛选。
  用于 V4 > V1/V2/V3/V4c/V5 的主定量结论。
  oracle_selection_fields 必须为空列表。

offline_stress_set:
  可以来自 success=0、os=1、wrong turn、early STOP、loop 等离线诊断。
  只能用于 debug、qualitative case study 或单独命名的 stress-test 报告。
  不能进入 V4 > baseline 的主定量 denominator。
```

`primary_quantitative_set` 预筛选只能使用在线可观测或运行前静态信息：

```text
1. 当前 step 位于路口、拐角、门口、分叉或多方向可通行位置。
2. instruction 包含 turn、landmark、door、room、relative direction 等方向性约束。
3. candidate_action 是 STOP / TURN_LEFT / TURN_RIGHT，或 online trigger 触发 route_uncertainty。
4. 当前 episode 内存在 prior keyframe memory image。
5. trigger_source 是 online_controller 或 static_pre_run。
```

以下内容只能作为运行后标签，不能作为预筛选条件：

```text
why_text_is_insufficient
why_image_is_needed
text_only_failed
correct_image_contains_directional_evidence
shuffled_image_failed_to_support_same_evidence
image_read_changed_controller_decision
image_read_changed_final_outcome
```

每次实验必须在运行前冻结 case set，并在报告中列出 denominator。生成
`why_text_is_insufficient` / `why_image_is_needed` 的标注者不能看到 V-mode 输出、
controller outcome 或最终任务结果。单个 case 只能作为 qualitative trace example；
不能单独支撑 V4 优于 V1/V2/V3/V4c/V5 的论文级结论。

主报告必须同时给出 parent pool：

```text
parent_episode_count
parent_online_trigger_count
parent_static_pre_run_count
parent_eligible_trigger_count = parent_online_trigger_count + parent_static_pre_run_count
primary_quantitative_set_count
eligible_case_rate = primary_quantitative_set_count / parent_eligible_trigger_count
trigger_type_distribution
online_vs_static_pre_run_split
offline_stress_set_count
```

场景记录需要包含：

```text
scene_id
episode_id
step_id
trigger_rule
trigger_source
candidate_action
selection_basis
oracle_selection_fields
retrieved_memory_id
retrieved_memory_step
why_text_is_insufficient
why_image_is_needed
posthoc_text_only_failed
posthoc_shuffled_control_failed
```

### Fixed-trigger replay manifest

Phase 1b 的 V1-V5 + V4c 对照必须使用 fixed-trigger shadow protocol，而不是每个 V-mode 各自重新触发。流程如下：

```text
1. 先运行 V0 / clean policy，冻结：
   - candidate_action
   - trigger_rule
   - trigger_source
   - query_text
   - ordered top-K memory hits
   - frozen NavigationPolicySkill inputs

2. V1 / V2 / V3 / V4 / V4c / V5 都消费同一个 manifest。

3. 对 visual-grounding 和 controller-causal claim，
   只使用 fixed-trigger shadow protocol 的结果。

4. 如果后续要报告 per-mode online-trigger protocol，
   必须作为单独分析，不能和 fixed-trigger 结果混成一个 denominator。
```

Fixed-trigger replay 是 same-state counterfactual protocol：

```text
V0 candidate_action:
  用于冻结 trigger point、query_text、ordered top-K memory hits
  和 NavigationPolicySkill 的同状态输入。

V1 / V3 prompt modes:
  必须在 frozen state 上重新调用 NavigationPolicySkill，
  输入对应的 prompt memory payload。
  除 prompt memory payload 差异外，instruction、last_action、
  recent_frames/current_image、history config 必须与 V0 frozen state 一致。
  不能直接复用 V0 candidate_action 当作 V1/V3 action。

V4 controller mode:
  使用 V0 clean candidate_action 作为 controller 待检查动作。

Fixed replay can claim:
  same-state visual grounding
  same-state controller-decision causality
  prompt-vs-controller counterfactual at frozen trigger points

Fixed replay alone cannot claim:
  per-mode online trigger frequency
  full online SR / SPL superiority over prompt modes
```

manifest schema：

```json
{
  "manifest_version": "visual_readback_fixed_v1",
  "generation_run_id": "...",
  "primary_set_name": "online_static_trigger_v1",
  "selection_basis": "online_controller|static_pre_run",
  "oracle_selection_fields": [],
  "shuffle_scope": "same_scene_other_episode",
  "shuffle_seed": 12345,
  "cases": [
    {
      "case_id": "scene_episode_step",
      "scene_id": "...",
      "episode_id": "...",
      "step_id": 134,
      "trigger_rule": "decision_point",
      "trigger_source": "online_controller",
      "candidate_action_from_v0": "TURN_RIGHT",
      "frozen_policy_inputs": {
        "instruction": "go down the hallway and turn at the painting",
        "last_action": "MOVE_FORWARD",
        "recent_frame_paths": [
          ".../openclaw_current_frames/step_000132.png",
          ".../openclaw_current_frames/step_000133.png"
        ],
        "current_image_path": ".../current/step_000134.png",
        "clean_navigation_payload": {
          "recent_frames_from_paths": true,
          "active_subgoal": "",
          "memory_context_text": "",
          "memory_images": []
        },
        "navigation_policy_config": {
          "num_history": 8,
          "max_memory_images": 0,
          "max_prompt_context_chars": 1200
        }
      },
      "current_image_path": ".../current/step_000134.png",
      "query_text": "hallway intersection turn at painting",
      "memory_hits": [
        {
          "rank": 1,
          "memory_id": "mem_001",
          "step_id": 58,
          "image_path": ".../keyframes/step_000058.png",
          "summary": "similar hallway before the corner",
          "retrieval_confidence": 0.82
        }
      ],
      "v5_attached_memory_hits": [
        {
          "rank": 1,
          "memory_id": "shuffled_mem_007",
          "image_path": ".../keyframes/step_000017.png",
          "shuffle_reason": "same_scene_other_episode",
          "negative_control_label": "wrong"
        }
      ]
    }
  ]
}
```

`frozen_policy_inputs` 是 V1 / V3 prompt-mode replay 的同状态合同。它必须足以重建
`NavigationPolicySkill.run(state, payload)` 的非 memory-prompt 输入，包括：

```text
instruction
last_action
recent_frame_paths 或等价 recent_frames artifact references
current_image_path
clean_navigation_payload
navigation_policy_config
```

V1 / V3 只能改变 prompt memory payload，例如 `memory_context_text` / `memory_images`
的 ablation 形态；不能改变当前帧、history frames、instruction 原文、step_id、
policy config 或 last_action。否则该 case 必须标为：

```text
replay_state_mismatch
```

并从 prompt-vs-controller counterfactual denominator 中排除，同时在 summary 中计数。

V5 的“相同 retrieved memory candidates”含义是：使用同一个 original retrieval query 和 top-K slot，但在 attach 给 VLM 前用 deterministic shuffle 替换图片。trace 必须同时记录：

```text
original_retrieved_memory_ids
attached_memory_ids
matched_memory_ids_from_attached_only=true
shuffle_seed
shuffle_scope
```

V5 中的 `matched_memory_ids` 必须只引用实际 attach 给 VLM 的 `attached_memory_ids`，不能引用 original retrieved IDs。`original_retrieved_memory_ids` 只用于证明 V5 与 V4 使用同一个 retrieval slot 和 query，不代表 VLM 实际读过那些原始图片。

V5 replacement 必须满足 fixed label contract：

```text
negative_control_label=wrong|unrelated
```

如果 `same_scene_other_episode` 或 `same_episode_wrong_step` 找不到预标注为 wrong / unrelated 的 replacement image，该 case 标为：

```text
negative_control_unjudgeable
```

并从 `V4 > V5` denominator 中排除，同时在 summary 中计数。

如果 live `MemoryQuerySkill` 没有 bypass，而是重新执行 query，则必须 assert live ordered hits 与 manifest 一致；不一致的 case 标为 `replay_mismatch`，不能进入主对照统计。

## Metrics

除了常规 VLN 指标：

```text
SR
SPL
OS
NE
nDTW / SDTW if available
```

必须新增 readback-specific metrics：

```text
readback_trigger_count
readback_completed_count
readback_failure_count
retrieved_memory_image_count
actually_read_image_count
matched_memory_count
judgeable_queries
correct_memory_hit_rate
topk_correct_memory_hit_rate
partial_hit_rate
visual_grounded_readback_count
attached_only_readback_count
grounded_but_wrong_label_count
adjudication_correct_readback_count
falsification_pass_rate
verifier_label_accuracy_by_type
verifier_label_confusion_matrix
shuffled_memory_control_success_rate
negative_control_unjudgeable_count
controller_decision_changed_after_visual_read_count
executed_action_changed_after_visual_read_count
stop_blocked_after_visual_read_count
current_view_stop_verification_count
historical_memory_stop_causal_count
replan_request_logged_after_visual_read_count
replan_executed_after_visual_read_count
agreement_support_count
disagreement_logged_count
low_confidence_log_only_count
```

metrics ownership 必须按 phase 分层：

```text
Phase 0 audit:
  parent_episode_count
  parent_online_trigger_count
  parent_static_pre_run_count
  parent_eligible_trigger_count
  primary_quantitative_set_count
  candidate_case_set_count
  offline_stress_set_count
  trigger_type_distribution
  image_backed_memory_smoke_pass
  multi_image_json_parse_success_rate
  image_attach_trace_rate
  readback_latency_p50 / p95
  readback_timeout_rate

Phase 1a minimal loop:
  readback_trigger_count
  readback_completed_count
  readback_failure_count
  retrieved_memory_image_count
  actually_read_image_count
  matched_memory_count
  visual_grounded_readback_count
  attached_only_readback_count
  controller_decision_changed_after_visual_read_count
  stop_blocked_after_visual_read_count
  current_view_stop_verification_count
  historical_memory_stop_causal_count
  replan_request_logged_after_visual_read_count
  agreement_support_count
  disagreement_logged_count
  low_confidence_log_only_count

Phase 1b fixed matrix:
  judgeable_queries
  correct_memory_hit_rate
  topk_correct_memory_hit_rate
  partial_hit_rate
  grounded_but_wrong_label_count
  adjudication_correct_readback_count
  falsification_pass_rate
  verifier_label_accuracy_by_type
  verifier_label_confusion_matrix
  shuffled_memory_control_success_rate
  negative_control_unjudgeable_count

Phase 2+ reserved until executed intervention exists:
  executed_action_changed_after_visual_read_count
  replan_executed_after_visual_read_count
```

### Correct memory label

第一阶段采用固定标注合同，而不是事后自由解释 `correct_memory_hit_rate`。

每个 retrieved memory image 标注为：

```text
correct:
  memory image 是当前 episode 的 prior keyframe，
  且人工标注或冻结规则确认它包含与当前 trigger 相关的
  landmark / directional affordance / decision-point geometry。

partial:
  memory image 是当前 episode 的 prior keyframe，
  但只包含 landmark 或 geometry 其中一类证据。

wrong:
  memory image 与当前 trigger 无关，或来自错误 namespace / 错误 episode。

unjudgeable:
  图像不可读、遮挡严重、标注者无法判断。
```

第一阶段报告：

```text
correct_memory_hit_rate = correct_top1 / judgeable_queries
topk_correct_memory_hit_rate = queries_with_any_correct_hit / judgeable_queries
partial_hit_rate 单独报告，不并入 correct。
```

`retrieval_confidence`、`readback_confidence` 和 `verifier_confidence` 必须分开记录。0.6 / 0.8 阈值只用于 `readback_confidence`，并且需要在小规模标注集上做校准报告；不能用 retrieval score 直接驱动 controller。

关键论文证据不是只看最终 SR/SPL，而是：

```text
正确图片是否被召回；
图片是否真的被 VLM 读取；
读图后下游 action / STOP / replan 是否变化；
错误图片是否不能产生同样变化；
这些变化是否改善最终任务结果或降低 near-miss / wrong STOP。
```

## 失败与降级

readback 失败时必须保守降级：

```text
MemoryQuerySkill 无 hit:
  不触发 VisualMemoryReadSkill，执行 clean candidate_action，记录 no_memory_hit。

retrieved hit 没有 image_path:
  跳过该 hit，只保留可读图 hit；若为空则 log only。

VLM read timeout / parse failure:
  不改变动作，记录 read_status=failed。

confidence 低:
  不改变动作，记录 low_confidence_log_only。

evidence 与 candidate_action 冲突但不在受限场景:
  不改变动作，只记录 disagreement。
```

任何失败都不能回退到 oracle 信息。

## 实现插入点

V4 的 runtime hook 必须位于：

```text
NavigationPolicySkill returns action_text
        ↓
candidate_action = action_text
        ↓
readback trigger gate
        ↓
MemoryQuerySkill / VisualMemoryReadSkill
        ↓
controller-side gate
        ↓
executor.command_for_action(final_action)
```

不能把 `VisualMemoryReadSkill` 放在 `NavigationPolicySkill` 之前，因为 `risky_stop` 和 `action_conflict` 需要 candidate action。也不能让 readback evidence 先进入 `NavigationPolicySkill` 再输出 candidate action，否则 V4 不再是 clean-policy arbitration。

如果 `openclaw_allow_planner_action_override` 使 runtime 跳过 `NavigationPolicySkill`，V4 第一阶段必须：

```text
policy_skipped=true -> skip visual readback arbitration
```

或在 V-mode runner 中强制关闭 planner action override。不能在没有 JanusVLN clean candidate action 的情况下声称做了 controller-side verification。

## 阶段规划

### Phase 0: Baseline audit and integration decision

目标：先确认当前问题足够多，且复用路径明确。

```text
1. 从现有 A0/A1/A2/A5 trace 中统计：
   - risky STOP
   - ambiguous turn
   - loop / repeated observation
   - near-miss success=0, os=1 只进入 offline_stress_set 统计

2. 输出 `candidate_case_set`：
   - primary_quantitative_set 只用 online controller trigger 或 static_pre_run 条件。
   - offline_stress_set 单独保存，不进入主定量 claim。
   - 记录 denominator，不能按 V4 是否成功筛选。
   - 该产物只是候选集合，不等同于 Phase 1b 的 fixed replay manifest。

3. 确认 VisualMemoryReadSkill 实现路径：
   - 默认复用现有 OpenClaw/Qwen image capability。
   - 只有多图 JSON readback 不可行时才新增 adapter。

4. 验证 image-backed memory path：
   - MemoryWriteSkill 能写入带 image_path 的 episode-local record。
   - MemoryQuerySkill 能召回同一条带 image_path 的 record。
   - image_path 指向真实可读图像文件。
```

Phase 0 exit gate：

```text
Common gate for any Phase 1 work:

1. image-backed memory smoke 通过：
   - write-then-query returns non-empty image_path；
   - retrieved image_path points to an existing readable file。

2. multi-image JSON readback smoke 通过：
   - parse_success_rate >= 0.90 on >= 20 calls；
   - image_attach_trace_rate = 1.00；
   - model_image_count 与 actually_read_image_paths 一致。

3. latency budget 通过：
   - timeout_ms = 90000；
   - readback_timeout_rate <= 0.10；
   - 每个 episode 的 readback 调用数有上限并写入 summary。

Phase 1a minimal-loop gate:

4. 至少有一个 viable trigger family：
   - >=10 risky_stop judgeable cases；或
   - >=10 decision_point judgeable cases。
5. Phase 1a 可以只启动对应 narrow study：
   STOP-only 或 turn-only。
6. Phase 1a 不需要满足 >=30 cases / >=3 scenes / >=2 trigger types。

Candidate-set gate for later Phase 1b manifest generation:

7. primary_quantitative_set 至少有 30 个 judgeable trigger cases。
8. 覆盖至少 3 个 scene，且至少 2 类 trigger：
   risky_stop / decision_point / route_uncertainty / stuck_or_loop。
9. 至少有 10 个 risky_stop 或 10 个 decision_point cases。

该 gate 只说明候选集合足以进入 Phase 1b manifest generation 和 V4 > V* claim。
它不表示 `OPENCLAW_VISUAL_READBACK_FIXED_CASE_MANIFEST` 已经生成。
它也不是启动 Phase 1a minimal loop 的前置条件。

Fixed-manifest gate for Phase 1b and V4 > V1/V2/V3/V4c/V5 claims:

10. Phase 1a minimal loop 已跑通，且从 `candidate_case_set` 生成
   `fixed_replay_manifest`。
11. fixed replay manifest 中每个 case 都包含足以重建
   NavigationPolicySkill 同状态输入的 `frozen_policy_inputs`。
12. fixed manifest 中每个 V5 replacement 都有 wrong / unrelated label，
   或明确标为 negative_control_unjudgeable 并从 V4 > V5 denominator 排除。

Narrow-study gate:

13. 如果 Common gate 和 Phase 1a minimal-loop gate 通过，
    但 Candidate-set gate 只因 case distribution 不足失败：
   - >=10 risky_stop cases -> 可以进入 STOP-only Phase 1a study。
   - >=10 decision_point cases -> 可以进入 turn-only Phase 1a study。
   - 只实现该 narrower study 所需 V-modes / metrics。
   - 不能声明完整 V4 > V1/V2/V3/V4c/V5。

No-go:

14. 如果 Common gate 任一项失败，停止在 Phase 0 audit report，
   或先实现缺失 backend / adapter 后重新运行 Phase 0。
15. 如果 Common gate 通过但 risky_stop 和 decision_point 都不足 10 个，
    只输出 qualitative / offline_stress_set 结果，
    不启动 Phase 1a / Phase 1b。
```

### Phase 1: 最小闭环

目标：证明 V4 control-only readback 机制跑通并能 trace。

#### Phase 1a: V4 trace and conservative gate

```text
Clean NavigationPolicySkill candidate_action
-> readback trigger gate
-> MemoryQuerySkill top-K image hits
-> VisualMemoryReadSkill reads current + memory images
-> structured evidence
-> conservative STOP verifier / controller gate
-> trace
```

Phase 1a 只允许：

```text
risky STOP block
high-confidence turn conflict -> log replan_request only
agreement / disagreement logging
```

不执行 real replanner，不改变 TURN_LEFT / TURN_RIGHT。这样 Phase 1a 不会吃掉 Phase 2 scope。

#### Phase 1b: Fixed-set controls

在 Phase 1a 跑通后，再跑固定 case set 的对照：

```text
V1_text_only_prompt
V2_path_only
V3_image_read_prompt
V4_image_read_controller
V4c_current_only_controller
V5_shuffled_image_read_controller
```

Phase 1b 必须消费 `OPENCLAW_VISUAL_READBACK_FIXED_CASE_MANIFEST`。该 manifest
必须在 Phase 1a minimal loop 跑通后由 `candidate_case_set` 生成，并在相同 V0
clean trigger points、相同 frozen policy inputs、相同 original retrieval query 和相同
top-K slots 上运行。单 case 只作为 qualitative example；aggregate claim 必须报告
primary denominator、失败例、replay_state_mismatch 和 replay_mismatch 数量。

### Phase 2: Replanner 接入

目标：让 high-confidence visual evidence 影响 replanning，而不是只 block。

```text
visual evidence
-> route consistency check
-> update controller_private_replan_state / route_stage
-> replan request
```

Phase 2 / Phase 3 仍然继承 V4 control-only invariant：

```text
readback-derived route_stage / replan_state 只能影响 controller gate。
不得写入 NavigationPolicySkill payload。
不得写入 active_subgoal。
不得写入 memory_context_text。
不得写入 memory_images。
trace 必须记录 readback_state_used_by_policy=false。
```

如果后续实验希望把 readback-derived subgoal 传给 `NavigationPolicySkill`，必须改成单独 V-mode，例如 `image_read_replan_prompt`，不能继续叫 `V4_image_read_controller`。

### Phase 3: 完整 Controller Arbitration

目标：覆盖更广的关键决策。

```text
candidate action rerank
STOP block
turn conflict recovery
stuck / loop recovery
route memory consistency check
```

### Phase 4: Strong Ablations

目标：回答 reviewer 对“为什么不直接把图片给 policy”的问题。

```text
V6_direct_policy_images
oracle-free direct image input comparison
latency / context / accuracy trade-off analysis
```

## Acceptance Criteria

Phase 0 / Phase 1 完成标准分开计算：

```text
Phase 0 audit complete:

1. 输出 parent pool 和 candidate_case_set：
   parent_episode_count；
   parent_online_trigger_count；
   parent_static_pre_run_count；
   parent_eligible_trigger_count；
   primary_quantitative_set_count；
   candidate_case_set_count；
   online_vs_static_pre_run_split；
   trigger_type_distribution。

2. image-backed memory smoke 通过：
   MemoryWriteSkill 写入带 image_path 的 episode-local record；
   MemoryQuerySkill 召回同一条带 image_path 的 record；
   image_path 指向真实可读图像。

3. multi-image JSON readback smoke 通过：
   parse_success_rate >= 0.90 on >= 20 calls；
   image_attach_trace_rate = 1.00；
   model_image_count 与 actually_read_image_paths 一致。

4. latency / timeout budget 通过：
   timeout_ms = 90000；
   readback_timeout_rate <= 0.10；
   每个 episode 的 readback 调用数有上限并写入 summary。

Phase 1a minimal-loop acceptance:

5. V4_image_read_controller 保持 control-only：
   NavigationPolicySkill payload 中没有 active_subgoal、memory_context_text、memory_images。
   trace 记录 policy_context_available、context_engine_context_available、
   actual_policy_payload_merge=false、used_by_policy=false、
   readback_state_used_by_policy=false、final_policy_payload_has_memory=false。

6. Trace 中能看到每次 readback 的：
   trigger_source、trigger_rule、retrieved_image_paths、
   actually_read_image_paths、model_image_count、
   verifier_labels、readback_confidence、retrieval_confidence、verifier_confidence、
   visual_grounding_status、visual_readback_config。

7. VisualMemoryReadSkill 输出 JSON 结构稳定，parse failure 有保守 fallback。

8. Controller 不读取 audit_action_hint / free-form reason 作为控制输入。
   Phase 1a 主报告只允许 shadow STOP-block 或 log replan_request。
   如果显式运行 executed STOP fallback pilot，
   final_action 必须落到 MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP 之一，
   不能把 meta decision 传给 Habitat executor；
   pilot 必须单独报告 denominator、fallback_policy 和 safety check。

9. STOP fallback state owner 明确：
   OpenClawVLNRuntime 或等价 runtime controller 维护
   last_executed_non_stop_action / age；
   scene_id / episode_id 变化时 reset；
   final_action 执行后 update；
   nil path 落到 default_move_forward。

10. Phase 1a falsification smoke 可运行：
   至少在 narrow study 样例上跑 summary_redacted、current_only、
   blank_or_occluded_memory 或 shuffled_memory 中的必要 arms。
   full per-label falsification matrix 留到 Phase 1b fixed_replay_manifest。

11. Action-changing pilot 的 confidence calibration gate 通过：
    如果只做 shadow/log-only，可只报告 calibration draft；
    如果 executed STOP fallback pilot 改变动作，
    必须报告 readback_confidence reliability bins、
    adjudicated-label precision/recall 和 calibration_failed_disable_action_change 规则。

Narrow-study acceptance:

12. STOP-only Phase 1a study 可独立完成：
    Common gate 通过；
    >=10 risky_stop judgeable cases；
    只声明 STOP verifier / shadow block 机制结果；
    不声明完整 V4 > V1/V2/V3/V4c/V5。

13. Turn-only Phase 1a study 可独立完成：
    Common gate 通过；
    >=10 decision_point judgeable cases；
    只声明 route_conflict / replan_request_logged 机制结果；
    不声明 executed turn correction 或完整 V4 > V1/V2/V3/V4c/V5。

Phase 1b fixed-set matrix acceptance:

14. Candidate-set gate 和 fixed-manifest gate 均通过：
    candidate_case_set 中 primary_quantitative_set >= 30 judgeable cases；
    candidate_case_set 覆盖 >= 3 scenes；
    candidate_case_set 覆盖 >= 2 trigger types；
    candidate_case_set 中 >= 10 risky_stop 或 >= 10 decision_point cases；
    Phase 1a minimal loop 已跑通；
    fixed_replay_manifest 已从 candidate_case_set 生成；
    fixed_replay_manifest 每个 case 都包含 frozen_policy_inputs；
    V5 replacement 均有 wrong / unrelated label，
    或标为 negative_control_unjudgeable 并从 V4 > V5 denominator 排除。

15. Phase 1b 对照可运行：
   V1_text_only_prompt、V2_path_only、V3_image_read_prompt、
   V4_image_read_controller、V4c_current_only_controller、
   V5_shuffled_image_read_controller。
   所有模式消费同一个 fixed_replay_manifest。
   V1 / V3 在 frozen state 重新调用 NavigationPolicySkill；
   除 prompt memory payload 外，policy 输入必须与 V0 frozen state 一致。
   replay_state_mismatch 单独计数并从 prompt-vs-controller denominator 排除。
   V5 必须记录 original_retrieved_memory_ids、attached_memory_ids、
   matched_memory_ids_from_attached_only 和 negative_control_label。

16. Verifier-label correctness adjudication 可运行：
    fixed labels 在 V-mode 对照前冻结；
    标注者不能看到 V-mode、controller outcome 或 final task result；
    annotator_count、agreement_rate、conflict_resolution_rule 写入 summary；
    per-label accuracy / confusion table 写入 summary；
    grounded_but_wrong_label_count 单独报告。

17. `correct_memory_hit_rate` 按固定标注合同计算，
   partial / wrong / unjudgeable 分开报告。

18. 单个 qualitative case 可以展示机制，
   但 V4 > V1/V2/V3/V4c/V5 的 claim 必须来自
   fixed_replay_manifest 中 primary_quantitative_set 对应 case 的 aggregate report，
   不能混入 offline_stress_set。

19. 每个 V4 > V* claim 使用预注册 endpoint table，
    报告 denominator、pass rule、failure handling、minimum_effect_size、
    paired_test_or_confidence_interval 和 allowed interpretation。

20. 全流程不使用 oracle metric 做在线决策。

21. Phase 1 的论文/报告 claim 只允许写成 mechanism validation。
    executed STOP fallback pilot 如果运行，只能作为 STOP-specific pilot 单独报告；
    action / replan / subgoal causality 必须等 Phase 2 / Phase 3 有 executed intervention trace 后再声明。
```

## 论文表述

可作为方法段落的最终表述：

```text
We propose an on-demand visual memory readback and controller-side arbitration
framework for long-horizon VLN. Historical keyframes are stored as
summary-indexed image paths and retrieved only after a clean JanusVLN policy
produces a candidate action at a critical navigation decision. A VLM then reads
the current observation together with retrieved memory images and emits bounded
verifier labels plus auditable visual evidence. In the main controller variant,
the recalled evidence is kept out of the JanusVLN prompt and is used only by an
external controller to verify risky stopping, record shadow STOP-blocks by
default, execute a bounded STOP fallback only in a separately reported pilot,
and log route conflicts for later replanning. Text-only,
path-only, image-read-prompt, image-read-controller, shuffled-image, blank-image,
retrieved-only, current-only, and summary-redacted controls are first scoped by
a primary pre-registered candidate case set. After the minimal controller loop is
validated, the V-mode matrix runs on a fixed-trigger replay manifest generated
from that candidate set with frozen policy inputs. Phase 1 evaluates visual
grounding and controller-decision causality, while claims about executed turn
correction, replanning, subgoal updates, or full VLN metric improvements are
gated on later executed-intervention traces.
```

## 已采用默认决策

第二轮 review 后，本 spec 采用以下默认值，implementation plan 不应再把它们当作开放设计问题：

```text
1. 多图 readback 路径：
   默认复用现有 OpenClaw / Qwen image capability。
   Phase 0 必须用 >=20 calls smoke test 验证 JSON parse、image attach trace 和 latency。
   不通过时才新增独立 adapter。
   同时必须通过 image-backed memory write-then-query smoke；
   fake / synthetic text-only memory backend 不能启动 V1-V5 + V4c fixed-set matrix。

2. risky STOP fallback：
   Phase 1a 主报告默认 OPENCLAW_VISUAL_READBACK_STOP_FALLBACK_POLICY=log_only。
   需要 executed STOP fallback pilot 时显式设置 previous_non_stop_else_move_forward，
   并单独报告 denominator、安全检查和 fallback_policy-specific outcome。

3. 第一版 decision_point 规则：
   使用当前 observation / latest visual summary 中的
   intersection、junction、fork、doorway、left opening、right opening、
   two exits、corridor split，
   且 candidate_action 是 TURN_LEFT / TURN_RIGHT。
   规则先记录在 candidate_case_set；
   Phase 1b fixed_replay_manifest 继承该冻结规则。

4. shuffled negative control：
   默认 same_scene_other_episode。
   如果同 scene 不足，才 fallback 到 same_episode_wrong_step。
   global_random 只用于补充 stress analysis，不作为主表默认。
   每个 replacement image 必须预标注为 wrong / unrelated；
   找不到合格 replacement 时标为 negative_control_unjudgeable。

5. primary candidate/fixed denominator：
   candidate_case_set 默认门槛是 >=30 judgeable cases、>=3 scenes、
   >=2 trigger types。
   完整 V4 > V1/V2/V3/V4c/V5 claim 还必须通过 fixed-manifest gate。
   不满足 Candidate-set gate 时可以进入 STOP-only / turn-only narrower study，
   但不能声明完整 V4 > V1/V2/V3/V4c/V5。

6. latency / timeout：
   timeout_ms=90000。
   readback_timeout_rate <= 0.10 才能进入 Phase 1b fixed-set matrix。
```

## 仍需实测报告

这些不是开放设计问题，而是 Phase 0 / Phase 1 运行后必须填入 summary 的实测值：

```text
primary_quantitative_set_count
offline_stress_set_count
parent_episode_count
parent_online_trigger_count
parent_static_pre_run_count
parent_eligible_trigger_count
eligible_case_rate
online_vs_static_pre_run_split
trigger_type_distribution
scene_count
multi_image_json_parse_success_rate
image_attach_trace_rate
image_backed_memory_smoke_pass
readback_timeout_rate
readback_latency_p50 / p95
replay_mismatch_count
negative_control_unjudgeable_count
grounded_but_wrong_label_count
adjudication_correct_readback_count
verifier_label_accuracy_by_type
current_view_stop_verification_count
historical_memory_stop_causal_count
stop_fallback_executed_count
executed_action_changed_after_visual_read_count
```
