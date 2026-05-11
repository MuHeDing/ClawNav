# ClawNav / OpenClaw-VLN Design Notes

## 背景

当前目标不是继续改 ClawNav 代码库中既有 JanusVLN 导航模型的结构，也不是接入 LLM sparse attention 或 slow-fast active memory reuse，而是在现有导航 inference 外围构建一套 OpenClaw-style training-free Harness。

这套 Harness 面向 Habitat / VLN-CE 环境，核心问题是：在不重新训练模型的前提下，能否通过 inference-time 的任务拆解、视觉记忆组织与召回、skill 调度、progress critic 和 replan 机制，提高 long-horizon VLN 以及 open-world-oriented embodied navigation 设置下的鲁棒性。

指定 baseline 为：

```bash
ClawNav/scripts/evaluation_lowmem_no_llm_sparse.sh
```

保持以下配置不变：

```text
max_pixels = 401408
kv_start_size = 8
kv_recent_size = 24
num_history = 8
不启用 --use_llm_adaptive_sparse_attention
不加入 slow-fast active memory reuse
不改 Qwen attention
不改 VGGT KV cache 策略
```

当前文档将该方向暂命名为：

```text
ClawNav: An OpenClaw-Style Visual Memory Harness for Training-Free Long-Horizon VLN
```

也可以在论文或项目中表述为：

```text
OpenClaw-VLN: A Training-Free Visual Memory Harness for Long-Horizon Vision-Language Navigation
```

## 核心思想

ClawNav 代码库中的既有 JanusVLN navigation policy 不再被理解为完整系统的唯一主控，而是 Harness 可调用的一个 `NavigationPolicySkill`。

原始范式：

```text
observation + history frames + instruction
-> JanusVLN navigation policy
-> action
```

新的范式：

```text
observation + instruction + current state
-> Harness 判断当前需要什么
-> 调用 memory / critic / replanner / NavigationPolicySkill
-> action
```

因此，核心贡献不是模型补丁，而是 inference-time AI system 机制：

```text
Harness
+ Memory
+ Skill / Subagent
+ Critic
+ Executor
+ Logging
```

可以用 System 1 / System 2 来理解：

```text
JanusVLN navigation policy = System 1 / reactive navigation policy
Harness Controller = System 2 / slow planner and scheduler
MemoryManager = persistent grounded context
Critic = closed-loop progress evaluator
Executor = embodied action interface
```

需要明确：Phase 1 中的 System 2 不是完全由 LLM 自由规划，而是 rule-constrained controller。它通过结构化状态、预算约束和固定 skill interface 实现可控的 inference-time scheduling。Phase 2 / Phase 3 再将 planner、critic、memory curator 升级为 LLM/VLM subagent 或 OpenClaw runtime 中的 callable skills。

## Claw 思想如何迁移到 VLN

RoboClaw 和 ABot-Claw 不直接服务于 Habitat VLN benchmark。它们的价值在于提供系统范式：

- 主控 Harness 负责调度，而不是固定流程。
- Memory 是可查询、可验证、可导航的外部世界状态。
- Skill / Subagent 是可组合能力单元。
- Critic 负责检查进展、失败和是否需要恢复。
- Executor 负责执行动作。
- 全过程保留结构化日志。

在当前项目中的映射关系：

```text
真实机器人 camera       -> Habitat RGB observation
真实机器人 pose/state   -> Habitat agent state / metrics
真实机器人 action API   -> Habitat discrete actions / env.step(action)
真实机器人 success      -> Habitat success / SPL / distance_to_goal, evaluation-only
Claw memory service     -> SpatialMemory / Harness MemoryManager
Claw skill runtime      -> Python SkillRegistry, later OpenClaw Skill / Tool schema
```

因此，当前工程第一阶段不需要安装 OpenClaw，但这不能写成最终形态。更准确的路线是三阶段：

```text
Phase 1: OpenClaw-free Harness Prototype
  ClawNav repo + Python Harness + Fake/Local Memory + Rule Critic
  目标是快速验证 inference-time memory and control 是否有效。

Phase 2: OpenClaw-compatible Harness Interface
  SkillRegistry -> OpenClaw Skill / Tool schema
  MemoryManager -> OpenClaw / ABot-Claw memory service protocol
  HabitatAdapter -> Robot embodiment adapter
  Harness trace -> OpenClaw runtime logs
  目标是即使暂不部署 OpenClaw，也保证接口形态能对齐 OpenClaw。
  Implementation plan: docs/plans/2026-05-11-clawnav-phase2-openclaw-compatible-interface.md

Phase 3: OpenClaw + VLN Runtime
  OpenClaw Planner
    -> VLN Navigation Skill
    -> Spatial Memory Skill
    -> Progress Critic Skill
    -> Habitat / Real Robot Executor
  目标是形成真正的 OpenClaw+VLN embodied navigation system。
```

当前文档后续的具体代码路线主要对应 Phase 1，同时在接口命名和数据结构上为 Phase 2 / Phase 3 保留对齐空间。

## 与 ABot-Claw / RoboClaw 的关系

ABot-Claw 的核心不是单一机器人任务，而是 embodied runtime：统一具身接口、视觉中心的跨具身多模态记忆、critic-based closed-loop feedback。当前 VLN Harness 对应关系如下：

```text
ABot-Claw OpenClaw Layer        -> Harness Controller / Planner
ABot-Claw Robot Embodiment      -> Habitat / VLN-CE Env Adapter
ABot-Claw Shared Service Layer  -> MemoryManager / SpatialMemoryClient / Critic
Unified Embodiment Interface    -> Habitat discrete action interface
Visual Memory                   -> episodic visual memory / scene long-term memory
Object / Place Memory           -> object / place / semantic / keyframe memory
Critic Feedback                 -> ProgressCriticSkill
Replanning                      -> ReplannerSkill
Skill Runtime                   -> SkillRegistry, later OpenClaw-compatible skills
```

RoboClaw 的核心是 VLM meta-controller、structured memory、deployment-time skill orchestration 和 process supervision。当前 VLN Harness 对应关系如下：

```text
RoboClaw VLM meta-controller       -> Harness Controller, later LLM/VLM planner
Structured memory                  -> task-level + working + episodic + scene memory
Task-level memory                  -> active subgoal / subgoal status / recovery history
MCP tools                          -> Python SkillRegistry, later OpenClaw tools
Learned policy primitives          -> JanusVLN navigation policy as NavigationPolicySkill
Process supervision                -> ProgressCriticSkill
Retry / recovery / switch policy   -> ReplannerSkill / fallback
Deployment-time skill scheduling   -> recall / act / verify / replan dynamic intents
```

本方案借鉴 RoboClaw 的 deployment-time process supervision 和 skill orchestration，不采用其 manipulation-oriented EAP forward-reverse data collection 机制。EAP 属于真实机器人操作数据采集生命周期，当前 Habitat VLN benchmark 第一阶段不需要实现。

## 两层动态性

需要区分提前定义的系统骨架和运行时动态决策。

提前定义：

- Controller / Planner
- Memory Manager
- Skill Registry
- NavigationPolicySkill
- MemoryQuerySkill
- MemoryWriteSkill
- ProgressCriticSkill
- ReplannerSkill
- Habitat Env Adapter
- Task-Level Memory / Subgoal Memory
- 结构化输入输出协议
- 调用预算和稳定性规则

运行时动态决定：

- 当前任务拆成几个子问题
- 当前 active subgoal 是否完成
- 是否切换到下一个 subgoal
- 是否召回 memory
- 查询哪类 memory
- 是否调用 critic
- 是否 replan
- 是否阻止 STOP
- 是否写入关键帧
- 给 NavigationPolicySkill 哪些历史图像和记忆证据

也就是说，不是每来一个 VLN 任务都手工写 workflow，而是先定义通用 Harness，再让系统在这个骨架内动态选择调用路径。

## Memory 设计路线

Memory 是该方向中最重要的部分。它不是简单扩大 `num_history`，也不是把所有历史帧塞进上下文，而是分层组织视觉经验。

### Non-Oracle Constraint

在线 Harness Controller 和 ProgressCritic 不能使用 oracle evaluation 信息做决策。禁止进入在线 controller 的信号包括：

```text
distance_to_goal
success
SPL
oracle shortest path
oracle shortest path action
ground-truth goal distance
future observations
future trajectory frames
```

这些信号只能用于：

```text
offline logging
diagnostic analysis
ablation visualization
final success / failure evaluation
```

Phase 1 的 controller 只能使用 agent 可访问或系统自身产生的非 oracle 信号：

```text
recent action pattern
collision flag, if available from agent-accessible observation / metric
pose displacement / rotation change
visual novelty / repeated observation
policy output pattern and optional confidence
instruction-observation semantic alignment
memory hit consistency
elapsed steps / max_steps
active subgoal status
```

因此，`distance_to_goal` 可以写入日志用于离线诊断，但不能用于 recall、block STOP、replan 或任何在线动作决策。

### Memory Evaluation Protocol / No-Leakage Rule

Scene long-term memory 容易引入数据泄露，因此必须显式规定 memory 来源。

合法设置：

```text
Episode-local memory:
  只使用当前 episode 内当前 timestep 之前在线写入的 episodic memory。

Scene-prior memory:
  先在不使用目标指令、不使用 oracle path 的 exploration phase 建立 scene memory，
  evaluation phase 只能读取该 prior memory。

Train-scene memory:
  只从 train scenes 建立长期记忆，不使用 val_seen / val_unseen evaluation scene 的未来信息。
```

不合法设置：

```text
提前读取 val_seen / val_unseen evaluation episode 的完整轨迹
使用目标位置、oracle path、shortest path action 构建 memory
使用当前 evaluation episode 的未来观测帧构建 memory
把 success / distance_to_goal 等 evaluation metric 写入可召回 memory
```

实验报告必须标明 memory source：

```text
episode-local
scene-prior
train-scene-only
```

默认 Phase 1 先使用 `episode-local` 和 `fake memory` 做机制验证；真实 `scene-prior` / `train-scene-only` memory 作为后续 ablation。

### Task-Level Memory / Subgoal Memory

Task-Level Memory 是让系统更像 RoboClaw / ABot-Claw Harness 的关键。它记录任务级结构，而不是只记录局部视觉历史。

保存内容：

```text
global_instruction
current_subgoal
pending_subgoals
completed_subgoals
subgoal_success_criteria
failure_reason
recovery_attempts
last_replan_step
```

例如指令：

```text
Walk through the hallway, turn left at the sofa, go into the kitchen, and stop near the sink.
```

运行时可动态维护：

```text
Subgoal 1: find hallway
Subgoal 2: identify sofa landmark
Subgoal 3: turn left after sofa
Subgoal 4: enter kitchen
Subgoal 5: stop near sink
```

Harness 每一步不仅判断是否 recall，还要判断：

```text
当前 subgoal 是否完成？
当前 observation 是否支持 active subgoal？
是否需要召回该 subgoal 对应的 landmark？
是否应该切换到下一个 subgoal？
是否需要根据失败原因重写 subgoal？
```

### Working Memory

当前 episode 内短期状态：

- 最近帧
- 最近动作
- 当前位姿
- agent-accessible metrics / offline diagnostics
- 失败计数
- stuck 状态
- active subgoal
- 最近召回过的 memory

它存在内存里，episode reset 时清空。

### Episodic Visual Memory

当前轨迹中的关键帧记忆。不是每帧都存，而是在关键事件触发时保存：

- step 0
- 每 N 步
- 视觉变化大
- 连续转向
- 非 oracle progress / stuck 信号
- collision / stuck
- 接近目标或尝试 STOP

保存内容：

```text
image
pose
step_id
caption / note
instruction progress tag
```

### Scene Long-Term Memory

跨 episode / 跨任务的场景记忆，可复用 ABot-Claw SpatialMemory 的四类结构：

- object memory
- place memory
- keyframe memory
- semantic frame memory

召回结果应返回：

```text
memory_type
target_pose
confidence
evidence
image_path
note
timestamp
```

使用 Scene Long-Term Memory 时必须遵守上面的 no-leakage rule。换句话说，long-term memory 是 prior experience，不是从当前 evaluation episode 的未来帧、目标位置或 oracle path 中抽取出的隐藏答案。

### Task / Skill Memory

记录任务层面的经验：

- 哪类指令通常如何拆解
- 哪些召回策略有效
- 哪些动作模式导致失败
- 哪些恢复策略有效

第一阶段可以不实现，但应作为后续扩展方向。

### Memory 的三种作用

Memory 不应只是“给 NavigationPolicySkill 多塞上下文”。如果最终只是把 memory image/text 拼进模型输入，创新会弱化成 retrieval-augmented JanusVLN。这里的 Memory 需要同时服务三类功能。

第一类是 Policy Context：

```text
text summary
evidence images
retrieved landmark descriptions
```

这些内容进入 `NavigationPolicySkill`，帮助底层导航 policy 选动作。

第二类是 Control Context：

```text
target_pose
confidence
distance_to_recalled_place
visited / unvisited hint
revisit suggestion
stop verification signal
```

这些内容给 Harness Controller / Critic / Replanner 使用，决定是否 recall、是否 replan、是否阻止 STOP。Control Context 不能包含 oracle distance、success 或未来轨迹信息。

第三类是 Executor Context：

```text
navigable target hint
topological anchor
landmark direction
candidate waypoint / pose
```

这些内容不一定直接喂给模型，而是服务 embodied action interface。当前 Habitat 第一版仍用离散动作，后续 OpenClaw / real robot 阶段可将其映射到导航目标或局部规划目标。

### 召回内容与格式

embedding 只用于检索，不直接交给主控模型。给 Harness / NavigationPolicySkill 的应该是：

```text
明文摘要
少量证据图
pose / topological hint
confidence
source / timestamp
```

示例：

```text
Relevant remembered observations:
1. place: kitchen entrance, confidence=0.82, pose=(...), evidence=...
2. semantic_frame: hallway with table on left, confidence=0.74, pose=(...)
```

同时，给 Harness 控制逻辑的结构化结果应包含：

```text
Policy Context:
  memory_context_text
  memory_images
  landmark_descriptions

Control Context:
  target_pose
  confidence
  stop_verification_signal
  recall_reason
  replan_hint
```

## Skill / Subagent 设计

第一阶段不需要复杂 LLM subagent，先实现固定 Python skill，保证可控、可测、可跑完整 evaluation。

### 固定 Skill

```text
NavigationPolicySkill
  包装既有 JanusVLN inference policy，输入 selected frames + memory context，输出动作。

MemoryQuerySkill
  调用 MemoryManager / SpatialMemoryClient，输出 memory hits。

MemoryWriteSkill
  将关键帧或语义帧写入 episodic / long-term memory。

ProgressCriticSkill
  根据非 oracle 状态、动作历史、位姿变化、视觉重复和 memory consistency 判断是否卡住、偏航、过早 STOP。

ReplannerSkill
  根据失败原因和 memory hits 更新 active subgoal。
```

### 后续可升级为小 Agent 的模块

```text
PlannerSubagent
MemoryCuratorSubagent
ProgressCriticSubagent
FailureRecoverySubagent
```

但这些应在 Harness 稳定跑通之后再加入。

## 稳定性原则

动态调度必须受控，否则系统会乱调用。第一版 Harness 应实现硬约束：

```text
每个 navigation step 最多 2-3 次 internal skill call
memory recall 有间隔和预算
critic 只能判断，不能直接执行动作
replanner 只能更新 subgoal，不能越过 executor
skill 必须返回结构化 SkillResult
skill 失败时 fallback 到 baseline navigation policy path
memory 写入只发生在关键帧，不每步写
所有 decision 都要记录 reason
```

这样系统是动态的，但不是失控的。

### Skill Failure / Fallback Rule

任何 skill 出现以下情况时，controller 必须进入 fallback：

```text
skill 抛出异常
skill timeout
skill 返回 invalid output
skill 返回 ok=false 且没有可恢复 payload
internal skill call 超过预算
memory backend 不可用
```

Fallback 的默认行为是：本步退回到 baseline navigation policy action path，即只调用 `NavigationPolicySkill`，不使用失败 skill 的输出。如果 `NavigationPolicySkill` 自身失败，则返回安全动作 `STOP` 并记录 `fallback=true` 和 failure reason。Fallback 不能读取 oracle metric。

## 代码实现路线

### 目录结构

新增：

```text
ClawNav/src/harness/
  __init__.py
  types.py
  config.py
  controller.py
  skill_registry.py

  memory/
    __init__.py
    task_memory.py
    working_memory.py
    spatial_memory_client.py
    memory_manager.py

  skills/
    __init__.py
    base.py
    navigation_policy.py
    memory_query.py
    memory_write.py
    progress_critic.py
    replanner.py

  env_adapters/
    __init__.py
    habitat_vln_adapter.py

  logging/
    __init__.py
    harness_logger.py
```

新增评估入口：

```text
ClawNav/src/evaluation_harness.py
ClawNav/scripts/evaluation_lowmem_harness.sh
```

保留原 baseline：

```text
ClawNav/src/evaluation.py
ClawNav/scripts/evaluation_lowmem_no_llm_sparse.sh
```

### Phase 1: Harness Contract

文件：

```text
harness/types.py
harness/config.py
```

定义：

```text
VLNState
HarnessDecision
SkillResult
MemoryHit
TaskState
SubgoalState
HarnessConfig
```

目标：所有模块只通过结构化对象通信。

### Phase 2: Task-Level Memory

文件：

```text
harness/memory/task_memory.py
```

实现：

```text
reset
set_global_instruction
set_active_subgoal
mark_subgoal_complete
mark_subgoal_failed
record_recovery_attempt
should_advance_subgoal
decompose_instruction_rule_based
```

Phase 1 支持两种模式：

```text
Single-subgoal mode:
  active_subgoal = full instruction
  用于最小可跑验证，保证稳定。

Rule-based subgoal mode:
  根据 then / and then / turn / enter / stop / near / after 等语言线索轻量拆解 instruction。
  用于验证 task-level memory 是否有帮助。
```

例如：

```text
Walk down the hallway and turn left at the sofa, then enter the kitchen and stop near the sink.
```

可拆成：

```text
1. walk down the hallway
2. turn left at the sofa
3. enter the kitchen
4. stop near the sink
```

后续再升级为 LLM/VLM PlannerSubagent。

### Phase 3: Working Memory

文件：

```text
harness/memory/working_memory.py
```

实现：

```text
reset
append_observation
append_action
get_recent_frames
is_stuck
has_low_progress_non_oracle
should_promote_keyframe
```

### Phase 4: SpatialMemory Client

文件：

```text
harness/memory/spatial_memory_client.py
```

实现：

```text
FakeSpatialMemoryClient
SpatialMemoryHttpClient
```

第一版先用 fake backend 跑通；真实 backend 通过 HTTP 接 ABot-Claw SpatialMemory。

### Phase 5: Memory Manager

文件：

```text
harness/memory/memory_manager.py
```

职责：

```text
决定什么时候 recall
决定 query type
选择 memory images
构造 memory_context_text
构造 control_context
决定哪些关键帧要写入
```

职责边界：

```text
MemoryManager = 决策层
  决定 should_write / write_type / write_payload
  决定 should_recall / query_type / recall_budget
  构造 policy_context / control_context / executor_context

MemoryWriteSkill = 执行层
  真正保存 image / embedding / metadata / JSON record
  或通过 SpatialMemory HTTP API 写入 long-term memory
```

MemoryManager 不直接执行副作用写入；所有写入都通过 `MemoryWriteSkill`。

第一版规则：

```text
step_id == 0 -> semantic query
instruction 含地点词 -> place / semantic query
instruction 含物体词 -> object / semantic query
非 oracle stuck 信号 -> semantic / position-prior query
risky STOP 信号 -> verify 前 recall
```

MemoryManager 输出需要区分：

```text
policy_context
control_context
executor_context
```

### Phase 6: Skill Registry

文件：

```text
harness/skills/base.py
harness/skill_registry.py
```

统一接口：

```python
class Skill:
    name: str
    def run(self, state, payload):
        ...
```

### Phase 7: NavigationPolicySkill

文件：

```text
harness/skills/navigation_policy.py
```

包装现有 `JanusVLN_Inference.call_model()`。

输入：

```text
recent_frames
memory_images
instruction
active_subgoal
memory_context_text
```

真实输入边界：

```text
不改变模型结构。
active_subgoal 和 memory_context_text 通过 instruction augmentation 注入。
memory_images 只有在现有 history-frame interface 可承载时才作为额外视觉上下文注入。
如果模型接口或显存预算不支持额外图像，则 memory_images 只进入 Harness control logic，不传给底层导航模型。
```

第一版推荐先只做 text augmentation + existing recent frames，避免改变底层导航模型的视觉输入假设。后续再 ablation memory image injection。

### Phase 8: Critic / Replanner

文件：

```text
harness/skills/progress_critic.py
harness/skills/replanner.py
```

第一版使用非 oracle 规则：

```text
连续左右转或动作震荡 -> possible stuck
位姿变化过小 -> low displacement
视觉 embedding 连续高度相似 -> repeated observation
policy 输出 STOP 但当前 observation 与 active subgoal / retrieved memory 不一致 -> risky STOP
接近 max_steps -> should_replan
```

明确禁止：

```text
distance_to_goal 连续不下降 -> should_recall
STOP 但 distance_to_goal 大 -> block_stop
success / SPL / oracle path 参与在线 critic
```

`distance_to_goal` 只能写入日志，用于离线分析 critic 是否有效。

### Phase 9: Controller

文件：

```text
harness/controller.py
```

主循环：

```text
append observation
decide intent
run skill
integrate result
返回 action 或继续有限内部调用
```

第一版 intent：

```text
act
recall_memory
write_memory
verify_progress
replan
stop
```

Controller 每步需要同时读取：

```text
TaskMemory
WorkingMemory
MemoryManager outputs
ProgressCritic outputs
```

关键调度问题：

```text
active subgoal 是否完成？
是否需要切换 subgoal？
是否需要为了 active subgoal 召回 landmark？
是否需要把 memory result 作为控制信号而不是 policy context？
```

### Phase-1 Controller Policy

Phase 1 采用 rule-constrained controller。推荐初始策略：

```text
if risky_stop:
  verify_progress or recall_memory
elif stuck:
  recall_memory -> replan if recall does not resolve uncertainty
elif should_write_keyframe:
  write_memory -> act
elif recall_interval reached and active_subgoal needs landmark:
  recall_memory -> act
else:
  act
```

所有分支都受以下预算限制：

```text
max_internal_calls_per_step
recall_interval_steps
max_replans_per_episode
max_memory_images
max_prompt_context_chars
```

### Phase 10: Habitat Adapter

文件：

```text
harness/env_adapters/habitat_vln_adapter.py
```

将 Habitat 环境状态转成 `VLNState`：

```text
observations["rgb"]
episode.scene_id
episode.episode_id
episode.instruction.instruction_text
env.get_metrics()
env.sim.get_agent_state().position
env.sim.get_agent_state().rotation
```

### Phase 11: Evaluation Harness

文件：

```text
src/evaluation_harness.py
```

复用 `evaluation.py` 的环境、metric、result 逻辑，只替换动作生成：

原始：

```python
images = sample_history(rgb_list)
action = model.call_model(images, instruction, step_id)[0]
```

Harness：

```python
state = adapter.build_state(env, observations, episode, step_id, last_action)
action_text = harness.step(state)
action = actions2idx[action_text]
observations = env.step(action)
```

## 低显存 Harness 脚本

新增：

```text
scripts/evaluation_lowmem_harness.sh
```

从 `evaluation_lowmem_no_llm_sparse.sh` 复制，保留：

```bash
max_pixels=401408
kv_start_size=8
kv_recent_size=24
num_history=8
```

改为：

```bash
src/evaluation_harness.py
OUTPUT_PATH="results/janusvln_extra_lowmem_401408_start8_recent24_history8_harness"
```

不添加：

```bash
--use_llm_adaptive_sparse_attention
--enable_slow_fast
```

可新增 Harness 参数：

```bash
--harness_mode memory_recall
--harness_max_internal_calls 3
--harness_recall_interval 5
--harness_memory_backend fake
```

真实 memory 后续可用：

```bash
--harness_memory_backend spatial_http
--spatial_memory_url http://127.0.0.1:8022
```

## 实验路线

按阶段做 ablation。除 SR / SPL / NE / OS 等 VLN 指标外，还要记录 Harness 系统指标，否则无法证明贡献来自 Harness。

### S0: ClawNav Baseline

```bash
bash scripts/evaluation_lowmem_no_llm_sparse.sh
```

目标：获得原始 ClawNav lowmem no sparse 结果。

### S1: Harness Act-Only

Harness 只调用 `NavigationPolicySkill`，不 recall、不 critic、不 replan。

目标：确认 Harness 包装不破坏 baseline 行为。

### S2: Harness + Working / Task Memory

记录状态、动作、metrics、active subgoal、subgoal status，但不改变模型输入。

目标：验证 trace、task state 和 working memory 正确。

### S3: Harness + Episodic Visual Memory

只使用 episode 内关键帧召回。

目标：验证关键帧存储、召回、上下文压缩是否有效。

### S4: Harness + Scene Long-Term Memory

接 fake 或真实 SpatialMemory，使用跨 episode / scene 的 object、place、semantic frame、keyframe retrieval。

目标：验证长期场景记忆是否能帮助 landmark / place / object 定位。

### S5: Harness + Progress Critic

引入 stuck、non-oracle low progress、bad STOP 检测。

目标：验证 closed-loop feedback 能否减少过早 STOP、原地打转和长程偏航。

### S6: Harness + Replanner

引入 active subgoal update、failure reason、recovery attempt。

目标：验证 subgoal-level recovery 是否帮助系统走出 stuck。

### S7: Full ClawNav

组合 memory + critic + replan + dynamic scheduling。

目标：验证完整 OpenClaw-style Harness 是否提升 long-horizon VLN 和 open-world-oriented embodied navigation 的鲁棒性。

### Harness 系统指标

除导航指标外，必须记录：

```text
Recall Frequency
Useful Recall Rate
Replan Frequency
Bad STOP Reduction
Stuck Recovery Rate
Average Internal Calls per Step
Memory Hit Usage Rate
Fallback Rate
Latency Overhead
GPU Memory Overhead
CPU Memory Overhead
Memory Storage Size
Average Prompt Length
Average Number of Visual Frames
Subgoal Completion Rate
Subgoal Switch Accuracy
```

其中 `Useful Recall Rate` 比 `Recall Precision` 更适合 Phase 1，因为 recall 的 ground truth 较难定义。第一版可定义为：

```text
一次 recall 后 K 步内出现以下任一现象：
  action oscillation 减少
  risky STOP 被避免
  active subgoal 状态改善
  final success ablation 中有贡献
```

`distance_to_goal` 只能用于离线分析该指标，不能作为在线 controller 的决策输入。

## 日志设计

新增：

```text
results/.../harness_trace_rank0.jsonl
```

每步写入：

```json
{
  "scene_id": "...",
  "episode_id": "...",
  "step_id": 12,
  "intent": "recall_memory",
  "skill": "MemoryQuerySkill",
  "reason": "low_progress",
  "memory_backend": "spatial_http",
  "num_memory_hits": 3,
  "num_memory_images": 2,
  "active_subgoal": "...",
  "subgoal_status": "in_progress",
  "control_context": {
    "has_target_pose": true,
    "stop_verification_signal": "uncertain"
  },
  "action_text": "TURN_LEFT",
  "fallback": false,
  "diagnostics": {
    "distance_to_goal": 4.2,
    "success": false,
    "oracle_metrics_used_for_decision": false,
    "memory_source": "episode-local"
  }
}
```

这些日志用于分析：

- 系统什么时候 recall
- 为什么 recall
- recall 后动作是否改善
- critic 是否减少错误 STOP
- replan 是否帮助走出 stuck
- 是否发生 oracle metric leakage
- memory retrieval 的来源是 episode-local、scene-prior 还是 train-scene-only

## 最小可跑版本

第一版建议只做：

```text
types.py
task_memory.py
working_memory.py
fake/spatial memory client
memory_manager.py
navigation_policy.py
controller.py
habitat_vln_adapter.py
evaluation_harness.py
evaluation_lowmem_harness.sh
```

暂不做：

```text
LLM planner
真正 subagent
VLAC critic
token-level memory
sparse attention
slow-fast
真实机器人 OpenClaw runtime
```

这样可以最快得到第一条可比较结果，并且不破坏指定 baseline。

## 论文叙事

不要将工作表述为：

```text
Memory-augmented JanusVLN
```

更准确的表述是：

```text
ClawNav: An OpenClaw-Style Visual Memory Harness for Training-Free Long-Horizon VLN
```

或：

```text
OpenClaw-VLN: A Training-Free Visual Memory Harness for Long-Horizon Vision-Language Navigation
```

核心贡献可归纳为：

```text
1. We formulate long-horizon VLN as an inference-time harness problem rather than a model-architecture modification problem.
2. We design a Claw-style VLN Harness that organizes a navigation policy, visual memory, progress critic, replanner, and executor through structured skill interfaces.
3. We propose a hierarchical memory mechanism that separates task-level, working, episodic, and scene-level memory, and uses retrieved memory not only as policy input but also as control and executor context.
4. We provide a controlled no-oracle evaluation protocol and ablation suite on Habitat / VLN-CE to measure both navigation performance and harness-level behaviors.
```

## 当前结论

目前最合理路线不是把“不安装 OpenClaw”作为最终结论，而是分阶段推进：

```text
Phase 1:
  直接在 ClawNav 中实现 OpenClaw-free Harness prototype。
  使用 Habitat / VLN-CE 作为验证环境。
  ClawNav 低显存 no sparse 配置保持不变。
  目标是跑出可控 ablation，验证 inference-time memory and control 是否有效。

Phase 2:
  将 Harness 接口改造成 OpenClaw-compatible。
  SkillRegistry 对齐 OpenClaw Skill / Tool schema。
  MemoryManager 对齐 ABot-Claw SpatialMemory service protocol。
  HabitatAdapter 抽象为 embodiment adapter。
  Harness trace 对齐 OpenClaw runtime logs。

Phase 3:
  接入 OpenClaw runtime。
  形成 OpenClaw Planner + VLN Navigation Skill + Spatial Memory Skill + Progress Critic Skill + Habitat / Real Robot Executor。
```

最终定位：

```text
本项目不是继续修改 ClawNav 代码库中的 JanusVLN 模型结构，而是在其外围构建一个 OpenClaw-style training-free Harness。该 Harness 将既有 JanusVLN navigation policy 抽象为可调用的 NavigationPolicySkill，并通过 task-level memory、episodic visual memory、scene-level spatial memory、non-oracle progress critic 和 replanning mechanism，在 inference time 动态决定何时感知、何时召回、何时执行、何时验证和何时重规划，从而验证 Claw-style AI system 是否能够提升 long-horizon VLN 和 open-world-oriented embodied navigation 的鲁棒性。
```
