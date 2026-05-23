# OpenClaw Memory-Aware Context Engine

## 背景

当前 OpenClaw + ClawNav visual-memory 实验里已经验证过一个关键问题：如果 `/plan` adapter 复用同一个 OpenClaw agent session，每一步 planner 调用都会继续追加到同一段对话历史里。运行到第 N 步时，Qwen/OpenClaw planner 看到的就不只是当前 step，而可能是前 N-1 步完整 `/plan` 对话历史加当前 step。这会让 input token 随步数线性甚至更快增长，最终出现 token 爆炸。

这类历史累积不是 visual-memory 本身必须具备的能力。导航 planner 需要的是当前决策相关信息，而不是完整聊天流水账。

核心原则：

```text
Session transcript is storage, not prompt.
ContextBuilder is the only path into Qwen.
```

完整 transcript 可以长期保存到本地，供审计、复盘、检索和摘要使用；但每次 API 调用都必须经过 ContextBuilder 的预算化组装，不能直接把整个 session history 注入 Qwen。

## 目标

本方案要实现一个 OpenClaw Memory-Aware Context Engine，用于替代“复用完整 agent 对话历史”的上下文模式。

实现上分两层：

1. **ClawNav `/plan` bounded adapter**：第一优先级，只解决导航逐步决策的 token 稳定性、session history 隔离、视觉/记忆短文本注入和 trace 证据。
2. **长期用户交互 memory engine**：第二优先级，用于长期聊天、复盘、文档和跨 session 项目记忆。它不能阻塞 `/plan` 的 Phase 1 修复。

目标：

- 同一个用户 session 可以长期连续聊天。
- 完整历史落盘保存，但不直接进入模型。
- 每轮模型调用只注入当前最有用的上下文。
- 旧信息通过摘要、索引、检索、任务状态和显式 memory 被召回。
- `/plan` 每步输入规模稳定，不随 episode step 数线性增长。
- 每次 assemble 都有 token 审计，能定位 token 来源。

非目标：

- 不做简单 `/new` 代替长期 session。
- 不依赖 OpenClaw agent session 自动历史来保存状态。
- 不把全部 `memory/*.md`、全部规则文件、完整 tool outputs 或完整 transcript 常驻 prompt。
- 不让 OpenClaw planner 默认接管 JanusVLN 动作决策。
- 不在 Phase 1 实现平台级长期 memory、vector index 或 dreaming；这些属于后续层。

## 两类 Session 必须分开

这个设计里要明确区分两种 session。

### 用户交互 Session

用户交互 session 可以长期存在。它用于持续讨论、调试、写文档、改代码、复盘实验。

允许：

- transcript 长期落盘
- rolling summary
- memory search
- task state
- recent turns
- compact
- durable memory flush

但即使是用户交互 session，也不能无限注入全部历史。它仍然应该由 ContextBuilder 组装预算内上下文。

### ClawNav `/plan` Planner Session

导航 `/plan` session 是逐步决策调用，不应该复用完整 OpenClaw agent 对话历史。

不希望出现：

```text
step N prompt =
  step 0..N-1 的完整 OpenClaw 对话历史
  + 当前 step payload
```

希望出现：

```text
step N prompt =
  当前 instruction / scene / episode / step
  + last_action / policy_action
  + 当前图片路径和必要视觉描述
  + 压缩后的 task state
  + 显式检索出的 memory_context_text
  + 少量 recent step summary
```

也就是说，`/plan` 是 bounded decision call，不是长期聊天窗口。

### OpenClaw CLI agent session 隔离

当前 ClawNav adapter 的风险点不是只在 ClawNav 自己拼 prompt。只要继续调用：

```text
openclaw agent --session-id <same-session> --message <bounded-prompt>
```

OpenClaw CLI 仍可能在 agent session 内部把历史消息自动拼回 provider input。这样 ClawNav trace 里看到的 `assembled_prompt_tokens` 可能稳定在 6k，但 Qwen/OpenClaw provider 实际看到的 `input_tokens` 仍然随 step 增长。

因此 Phase 1 必须明确切断 hidden session history。允许三种实现路径，按优先级选择：

1. **无历史调用模式**：如果 OpenClaw CLI/API 支持禁用 session transcript 注入，`/plan` 必须显式开启该模式，并保留本地 transcript 只作审计。
2. **每步 fresh planner session**：如果无法禁用 CLI history，`/plan` 的 `--session-id` 必须至少包含 `run_id`、`scene_id`、`episode_id`、`step_id` 或一次性 nonce，使 OpenClaw 不会把前一步 agent 对话注入当前步。
3. **改走 stateless 底层 provider/gateway**：如果 OpenClaw agent 命令无法保证无历史，应绕过 agent session 层，由 ContextBuilder 直接调用不自动拼历史的接口。

仅改变 ClawNav 侧 prompt payload 大小不能算完成 Phase 1。必须用 provider usage 证明 hidden history 没有进入 Qwen。

## 总体架构

```text
User message or /plan request
  ↓
Transcript Store
完整历史落盘，不直接进模型
  ↓
MemoryWriter
提取偏好、决策、错误修复、任务状态、项目事实、视觉记忆
  ↓
Memory Store
  ├── MEMORY.md
  ├── memory/*.md
  ├── PROJECT.md / AGENTS.md
  ├── rules/*.md
  ├── running_summary.md
  ├── task_state.json
  ├── decision_log.jsonl
  ├── error_fixes.jsonl
  └── vector_index.sqlite
  ↓
MemoryRetriever
按任务、路径、scene、episode、query 检索
  ↓
ContextBuilder
按 profile 和 token budget 组装上下文
  ↓
LLM API
```

一句话：

```text
Transcript 是仓库；
MemoryWriter 是整理员；
MemoryRetriever 是检索员；
ContextBuilder 是入口闸门；
LLM 只看被筛选后的上下文。
```

## 记忆分层

| 层级 | 名称 | 内容 | 注入策略 |
| --- | --- | --- | --- |
| 1 | System / Agent Policy | agent 行为、工具规则、安全边界 | 固定注入，保持短 |
| 2 | User Memory | 用户长期偏好 | 短文本注入 |
| 3 | Project Memory | 项目架构、命令、约束 | 控制长度注入 |
| 4 | Path Rules | 目录或模块规则 | 按 touched files 加载 |
| 5 | Memory Index | `MEMORY.md` 索引 | 注入短索引，不展开全部细节 |
| 6 | Running Summary | 当前 session 压缩状态 | 常驻，但预算内 |
| 7 | Task State | 当前任务目标、阶段、约束、下一步 | 常驻，但结构化 |
| 8 | Retrieved Memories | 相关旧记忆 | 检索 top-k，预算内 |
| 9 | Recent Turns / Recent Steps | 最近少量原文或 step 摘要 | 用户聊天可保留；`/plan` 默认极少 |

关键点：`MEMORY.md` 应该是索引，不是所有细节的容器。详细内容放在 topic files 或 memory chunks 中，需要时检索。

Phase 1 不要求完整实现上表所有层级。Phase 1 的最小边界是：

- `plan` profile 的固定 system block。
- 当前 step state。
- 当前图片路径和必要视觉描述。
- episode-local `memory_context_text`。
- 少量 step summary 或空 summary。
- 每次调用的 token/历史审计。

`MEMORY.md`、topic files、path rules、vector index 和 dreaming 都是长期交互层能力，不应成为阻断 `/plan` token 修复的前置依赖。

## Context Profiles

必须为不同调用类型设置不同 profile。不要用同一个 25k 上下文预算同时服务用户聊天和 VLN `/plan`。

### Interactive Chat Profile

用于用户和 OpenClaw 长期聊天、代码调试、写文档。

建议预算：

```python
INTERACTIVE_CONTEXT_BUDGET = {
    "system": 3000,
    "user_memory": 800,
    "project_memory": 1500,
    "memory_index": 800,
    "path_rules": 2000,
    "task_state": 1000,
    "running_summary": 2500,
    "retrieved_memories": 4000,
    "recent_turns": 6000,
    "current_user_msg": 3000,
}

INTERACTIVE_MAX_TOTAL_TOKENS = 25000
```

### `/plan` Navigation Profile

用于 ClawNav 每一步导航 planner 调用。这个 profile 必须更小、更稳定。

建议预算：

```python
PLAN_CONTEXT_BUDGET = {
    "system": 1200,
    "task_state": 600,
    "current_step_state": 800,
    "visual_observations": 1600,
    "retrieved_memories": 1500,
    "recent_step_summary": 500,
    "current_instruction": 500,
}

PLAN_MAX_TOTAL_TOKENS = 6000
```

`/plan` profile 默认不注入普通聊天 recent turns。它需要的是结构化 state 和显式召回的 memory，而不是完整对话历史。

### Qwen 输入硬阈值

除了 profile 自身的目标预算，还要设置所有 Qwen 调用都必须遵守的硬阈值：

```python
QWEN_HARD_MAX_INPUT_TOKENS = 50000
```

这个阈值是最后防线，不是日常目标。日常目标仍然是：

```text
interactive chat: 15k-25k tokens
/plan navigation: 4k-6k tokens
hard ceiling: <= 50k tokens
```

ContextBuilder 的行为必须是：

```text
1. 先按当前 profile 的预算组装上下文。
2. 如果超过 profile 预算，按低优先级块裁剪。
3. 在调用 OpenClaw/Qwen 前估算 assembled prompt tokens。
4. 如果 assembled prompt 估算仍超过 hard ceiling，直接拒绝调用 Qwen。
5. 调用返回后读取 provider usage input_tokens。
6. 如果 provider input_tokens 超过 hard ceiling，后续请求进入 fail-fast，并记录 hidden-history 风险。
7. trace 中记录 token_limit_exceeded 和各块 token 贡献。
```

也就是说，`QWEN_HARD_MAX_INPUT_TOKENS=50000` 用来防止异常路径、旧 session 历史、过长视觉描述或错误检索结果把输入撑爆。正常 `/plan` 不应该接近 50k。

当前 adapter 里已有的 `OPENCLAW_AGENT_MAX_INPUT_TOKENS` / `agent_max_input_tokens` 是 agent usage guard，默认可能低于 50k。它和本文的阈值语义需要分开：

```text
PLAN_MAX_TOTAL_TOKENS:
  ContextBuilder 日常目标预算，例如 6000。

OPENCLAW_AGENT_MAX_INPUT_TOKENS:
  当前运行脚本的 provider usage fail-fast 阈值，例如 10000。

QWEN_HARD_MAX_INPUT_TOKENS:
  永远不能越过的最后防线，例如 50000。
```

Phase 1 可以继续把 `OPENCLAW_AGENT_MAX_INPUT_TOKENS=10000` 作为更严格的实验保护，但文档、trace 和测试里必须明确它不是 50k hard ceiling 的同义词。

裁剪优先级：

```text
必须保留：
  system
  current instruction
  current step state
  task_state

优先裁剪：
  retrieved_memories
  visual_observations
  recent_step_summary
  running_summary

必须禁止：
  full transcript
  full OpenClaw agent history
  unbounded previous /plan turns
```

## `/plan` ContextBuilder 组装内容

每次 `/plan` 调用建议只包含：

```text
[System]
OpenClaw planner 的固定职责、输出 schema、动作边界。

[Task State]
当前 run_id、scene_id、episode_id、step_id、目标约束。

[Current Step State]
instruction、last_action、policy_action、当前位置相关 metadata。

[Visual Observations]
当前 step 或关键帧的视觉描述，严格限长。

[Retrieved Memories]
按 episode/scene/task scope 检索出的短记忆。

[Recent Step Summary]
最近少量 step 的压缩摘要，不是原始对话。

[Current Request]
当前 planner 需要判断的问题。
```

不得包含：

```text
完整 OpenClaw agent history
前 N-1 步完整 /plan prompt
前 N-1 步完整 assistant response
完整 transcript.jsonl
全部 memory 文件
全部历史 tool output
```

## Memory Scope Gate

为了避免跨 episode 污染，检索必须有 scope gate。

默认召回顺序：

```text
episode memory
  ↓
scene memory
  ↓
task memory
  ↓
project memory
```

默认策略：

- `/plan` 优先只使用 `episode:<scene_id>:<episode_id>`。
- `scene` scope 需要显式启用。
- `task` scope 只用于稳定任务知识，不用于具体目标位置判断。
- 如果检索结果的 `memory_scope`、`memory_namespace`、`scene_id`、`episode_id`、`run_id` 不匹配，应默认过滤。
- 对 `/plan`，缺少 scope/namespace/episode 标识的 hit 默认丢弃，而不是放行。旧格式或脏数据只能进入审计日志，不能进入 planner prompt。
- 如果需要跨 episode 共享，必须在 trace 中明确记录 `cross_episode_recall=true`。

这个规则用于避免旧任务目标串入当前 episode，例如把 kitchen 目标带入 archway episode。

## MemoryWriter

MemoryWriter 在 turn 或 step 结束后运行，把原始历史沉淀为可控记忆。

第一版不要使用复杂 LLM extractor。先用规则和结构化字段实现：

写入内容：

- 用户长期偏好
- 项目事实
- 关键设计决策
- 错误根因和修复方式
- 当前任务状态
- visual memory 条目
- episode-level navigation evidence

不写入：

- 寒暄
- 重复解释
- 长工具输出
- 临时日志
- 无效中间过程
- 模型过程废话

建议结构：

```python
class MemoryWriter:
    def process_turn(self, session_id, new_turns, assistant_reply, tool_results):
        candidates = self.extract_candidates(new_turns, assistant_reply, tool_results)
        for item in candidates:
            if item.importance < 0.55:
                continue
            self.route(item)
        self.update_running_summary(session_id, new_turns)

    def process_plan_step(self, run_id, scene_id, episode_id, step_id, trace):
        visual_memory = self.extract_visual_memory(trace)
        if visual_memory:
            self.write_episode_memory(run_id, scene_id, episode_id, visual_memory)
        self.update_step_summary(run_id, scene_id, episode_id, step_id, trace)
```

`process_plan_step()` 必须比普通聊天更保守，只保存短导航证据，不保存完整 planner prose。

## MemoryRetriever

检索不要只用向量相似度。建议混合打分：

```python
final_score = (
    0.40 * semantic_similarity
    + 0.20 * keyword_match
    + 0.20 * importance
    + 0.10 * recency
    + 0.10 * active_task_match
)
```

对 `/plan` 还要额外乘以 scope gate：

```python
if memory.scope == "episode" and memory.episode_id == current_episode_id:
    scope_weight = 1.0
elif memory.scope == "scene" and allow_scene_recall:
    scope_weight = 0.7
elif memory.scope == "task" and allow_task_recall:
    scope_weight = 0.4
else:
    scope_weight = 0.0
```

最终：

```python
score = final_score * scope_weight
```

## Running Summary 模板

`running_summary.md` 应该是可继续执行的状态，而不是聊天流水账。

模板：

```markdown
# Running Summary

## 当前目标
...

## 已确认约束
- ...

## 关键设计决策
- ...

## 当前任务状态
- ...

## 已知风险
- ...

## 下一步
- ...
```

对于 `/plan`，建议使用更短的 step summary：

```markdown
# Episode Step Summary

Scene: ...
Episode: ...
Instruction: ...

## Stable Observations
- ...

## Useful Memories
- ...

## Recent Navigation State
- last_action: ...
- policy_action: ...
- planner_intent: ...

## Do Not Carry Forward
- stale planner action text
- failed STOP guidance
- unrelated scene/episode memory
```

## ContextBuilder 伪代码

```python
class MemoryAwareContextBuilder:
    def build(self, request):
        if request.kind == "plan":
            return self.build_plan_context(request)
        return self.build_interactive_context(request)

    def build_plan_context(self, request):
        blocks = [
            self.block("system", self.load_plan_system_prompt()),
            self.block("task_state", self.load_task_state(request.run_id)),
            self.block("current_step_state", self.format_step_state(request)),
            self.block("visual_observations", self.load_visual_observations(request)),
            self.block("retrieved_memories", self.retrieve_plan_memories(request)),
            self.block("recent_step_summary", self.load_recent_step_summary(request)),
            self.block("current_instruction", request.instruction),
        ]
        return self.fit_to_budget(blocks, PLAN_CONTEXT_BUDGET, PLAN_MAX_TOTAL_TOKENS)

    def fit_to_budget(self, blocks, block_budgets, profile_max_tokens):
        trimmed = self.trim_each_block(blocks, block_budgets)
        trimmed = self.shrink_to_total(trimmed, profile_max_tokens)

        hard_total = self.count_tokens(trimmed)
        if hard_total > QWEN_HARD_MAX_INPUT_TOKENS:
            trimmed = self.emergency_shrink(trimmed, QWEN_HARD_MAX_INPUT_TOKENS)

        final_total = self.count_tokens(trimmed)
        if final_total > QWEN_HARD_MAX_INPUT_TOKENS:
            raise QwenContextTooLarge(
                final_total=final_total,
                hard_limit=QWEN_HARD_MAX_INPUT_TOKENS,
            )

        return self.render(trimmed)

    def retrieve_plan_memories(self, request):
        memories = self.memory_retriever.search(
            query=request.query_text,
            scene_id=request.scene_id,
            episode_id=request.episode_id,
            run_id=request.run_id,
            allowed_scopes=["episode"],
            top_k=4,
            strict_scope=True,
        )
        return self.format_short_navigation_memories(memories)
```

最重要的约束：

```python
# 禁止直接取完整 OpenClaw session history
messages = context_builder.build(request)
usage = openclaw_agent_call(
    messages=messages,
    session_history=False,
    session_id=context_builder.stateless_session_id(request),
)
context_builder.audit_provider_usage(messages, usage)
```

如果实际 OpenClaw CLI 不支持 `session_history=False` 这类语义，就不能在代码里假装已经关闭历史；必须改用每步 fresh session 或 stateless gateway，并在 trace 中写明所选模式：

```json
{
  "openclaw_session_mode": "stateless|fresh_per_step|reused_with_no_history",
  "openclaw_session_id": "..."
}
```

## Token 审计

每次 assemble 必须写审计记录。

建议字段：

```json
{
  "run_id": "20260523-ctx-engine",
  "scene_id": "2azQ1b91cZZ",
  "episode_id": "11",
  "step_id": 42,
  "context_profile": "plan",
  "assembled_prompt_tokens": 5420,
  "provider_input_tokens": 5480,
  "hidden_history_tokens_estimate": 60,
  "max_total_tokens": 6000,
  "agent_max_input_tokens": 10000,
  "qwen_hard_max_input_tokens": 50000,
  "token_limit_exceeded": false,
  "system_tokens": 980,
  "task_state_tokens": 410,
  "current_step_state_tokens": 620,
  "visual_observation_tokens": 1440,
  "retrieved_memory_tokens": 1130,
  "recent_step_summary_tokens": 360,
  "history_tokens": 0,
  "openclaw_session_mode": "fresh_per_step",
  "openclaw_session_id": "clawnav-run1-scene-episode-step42",
  "retrieved_memory_ids": ["mem_001", "mem_004"],
  "cross_episode_recall": false
}
```

关键验收项：

```text
history_tokens 必须为 0 或固定小值。
assembled_prompt_tokens 不应随 step_id 线性增长。
provider_input_tokens 不应随 step_id 线性增长。
hidden_history_tokens_estimate 应接近 0 或固定小值。
assembled_prompt_tokens 必须小于等于 profile 预算或明确记录裁剪。
provider_input_tokens 必须小于等于 agent_max_input_tokens 和 qwen_hard_max_input_tokens。
正常 /plan assembled_prompt_tokens 应接近 4k-6k，而不是接近 50k。
```

## 验收标准

最小可用版验收：

```text
连续调用 /plan 100 次；
第 100 次 assembled_prompt_tokens 不应显著大于第 10 次；
第 100 次 provider_input_tokens 不应显著大于第 10 次；
step 0 的原始 prompt/response 不应出现在 step 99；
除非它被 running_summary 或 retrieved memory 显式选择；
trace 中 history_tokens=0；
trace 中 openclaw_session_mode 明确说明是 stateless、fresh_per_step 或 reused_with_no_history；
每次 `retrieved_memory_ids` 可追踪；
任意一次 Qwen input tokens <= 50000；
超过 50000 时必须拒绝调用，而不是继续请求 Qwen。
```

实验验收：

- 对同一 episode 跑 100 step stress test。
- 比较旧方案和新方案的 assembled prompt token 曲线。
- 比较旧方案和新方案的 provider input token 曲线。
- 新方案两条 token 曲线都应接近水平线。
- 如果启用 visual observations，token 增长只与当前视觉文本长度有关，不与历史步数有关。
- 如果启用 recall，token 增长只与 top-k 和预算有关，不与 memory 总量有关。
- 增加异常输入测试：构造超长 transcript、超长 visual observation、超量 retrieved memories，确认最终 Qwen 输入不超过 50k。

## 落地路线

### Phase 1：阻断线性历史

先实现：

- `/plan` 不复用完整 OpenClaw agent history。
- 明确 OpenClaw session 策略：优先无历史调用；否则每步 fresh session；否则改走 stateless gateway。
- `--session-id` 不能只按 scene/timestamp 复用；如果选择 fresh session，必须纳入 `run_id`、`scene_id`、`episode_id`、`step_id` 或 nonce。
- ContextBuilder 输出 bounded prompt。
- `PLAN_MAX_TOTAL_TOKENS` 硬上限。
- `QWEN_HARD_MAX_INPUT_TOKENS=50000` 全局硬阈值。
- 保留或配置 `OPENCLAW_AGENT_MAX_INPUT_TOKENS` 作为更严格的实验 fail-fast guard，并和 50k hard ceiling 分开记录。
- `history_tokens` 审计。
- `assembled_prompt_tokens` 和 provider `input_tokens` 同时写入 trace。
- `hidden_history_tokens_estimate = provider_input_tokens - assembled_prompt_tokens` 写入 trace。
- `openclaw_session_mode` / `openclaw_session_id` 写入 trace。
- `token_limit_exceeded` / `qwen_hard_max_input_tokens` 写入 trace。
- `run_id` / `scene_id` / `episode_id` / `step_id` 明确进入 payload。

这一步先不做复杂 RAG。

### Phase 2：加入显式摘要和 episode memory

实现：

- `running_summary.md`
- `task_state.json`
- episode step summary
- episode-scope memory retrieval
- visual memory 短文本格式化
- top-k recall

这一步开始替代“靠 agent session 记住前面发生了什么”。

### Phase 3：加入长期记忆和 hybrid search

实现：

- `MEMORY.md` 索引
- `memory/*.md` topic files
- keyword + vector hybrid search
- decision log
- error fix log
- path rules

这一步用于长期用户交互和跨 session 复盘，不应默认进入 `/plan`。如果 Phase 1 的 session 隔离和 provider token 曲线尚未通过，不应推进到 Phase 3。

### Phase 4：后台整理和 dreaming

实现：

- 每 5-10 轮整理一次 memory candidates
- compact 前 memory flush
- 去重、合并、过期清理
- contradiction / freshness check
- 人工可审阅的 DREAMS 或 review log

这一步提高长期记忆质量，不影响 Phase 1 的核心 token 控制。它不能作为 `/plan` token 爆炸问题的解决方案。

## 风险和约束

### 风险 0：ClawNav prompt 变短，但 OpenClaw session 仍拼历史

这是 Phase 1 的最高优先级风险。只要使用相同 `--session-id` 调用 `openclaw agent --message`，就要假设 OpenClaw 可能把 session transcript 自动注入 provider input。

缓解：

- 不以 `assembled_prompt_tokens` 单独作为成功证据。
- 必须记录 provider `input_tokens`。
- 必须记录 `openclaw_session_mode`。
- 如果 provider token 曲线随 step 增长，立即判定 Phase 1 未完成。
- 如果 CLI 没有无历史模式，默认改成 fresh-per-step planner session。

### 风险 1：把长期聊天模型误套到 `/plan`

用户聊天可以保留 recent turns；`/plan` 不应该默认保留 recent turns。

缓解：

- 使用独立 `plan` profile。
- `/plan` 默认 `recent_turns=0`。
- 只允许 recent step summary。

### 风险 2：检索召回污染当前 episode

跨 episode memory 可能把旧目标带进当前任务。

缓解：

- episode scope 默认开启。
- scene/task scope 默认关闭。
- `/plan` strict scope 默认开启。
- hit 缺少 `memory_scope` 或 `memory_namespace` 时默认过滤。
- `run_id` 不匹配的具体 episode evidence 默认过滤；只有稳定项目知识可跨 run。
- trace 记录 `cross_episode_recall`。

### 风险 3：摘要变成新的无限上下文

running summary 如果只追加不压缩，也会膨胀。

缓解：

- summary 自身有 token budget。
- 每次更新重写摘要，不做无限 append。
- 超出预算时保留目标、约束、决策、下一步，删除流水账。

### 风险 4：MemoryWriter 写入低价值内容

如果把模型废话和临时日志写进 memory，检索质量会下降。

缓解：

- importance 阈值。
- kind 白名单。
- topic files 定期整理。
- trace memory ids，便于回溯错误召回。

## 推荐最终表述

最终方案可以这样对外描述：

```text
OpenClaw Memory-Aware Context Engine
不是删除历史，也不是每轮 /new。
它把完整历史保存在本地，把长期知识沉淀到 memory，
但每次模型调用只通过 ContextBuilder 注入预算内的必要上下文。

对 ClawNav /plan 来说，
每一步都是 bounded decision call：
当前状态 + 视觉摘要 + 显式 memory recall + task state，
而不是前 N-1 步完整对话历史。
```

最重要的一句话：

```text
不要让 session history 直接成为 prompt；
让 ContextBuilder 从 memory system 里挑选当前最有用的信息再喂给模型。
```

## 参考

- OpenClaw Context: https://docs.openclaw.ai/concepts/context
- OpenClaw Context Engine: https://docs.openclaw.ai/concepts/context-engine
- OpenClaw Memory Overview: https://docs.openclaw.ai/concepts/memory
- OpenClaw Compaction: https://docs.openclaw.ai/concepts/compaction
- Claude Code Memory: https://code.claude.com/docs/en/memory
