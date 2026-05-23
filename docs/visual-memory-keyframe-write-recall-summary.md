# Visual Memory Keyframe / Write / Recall Summary

本文总结当前 ClawNav + OpenClaw visual-memory 方案中的三部分核心机制：

1. 关键帧如何保存
2. visual memory 如何写入
3. visual memory 如何召回并注入 policy

## 1. 关键帧保存

每个导航 step，ClawNav 会从 JanusVLN 当前输入图像序列中取最后一帧作为当前观测：

```text
images[-1] -> current frame
```

然后根据固定间隔规则判断是否保存为关键帧：

```python
step_id == 0 or step_id % 10 == 0
```

因此默认关键帧 step 是：

```text
step 0, 10, 20, 30, ...
```

关键帧保存路径类似：

```text
<OUTPUT_PATH>/keyframes/<scene_id>/<episode_id>/step_000010.png
```

非关键帧保存为普通当前帧：

```text
<OUTPUT_PATH>/openclaw_current_frames/<scene_id>/<episode_id>/step_000011.png
```

这些图片路径会进入 `/plan` payload，主要字段包括：

```text
current_image_path
recent_keyframe_paths
keyframe_candidate.image_path
```

当前关键帧策略是固定间隔策略，主要目标是降低 VLM describe 频率，避免每一步都看图导致评测过慢和 token/延迟成本过高。

## 2. Visual Memory Write

Visual memory write 的前提是 ClawNav `/plan` adapter 以 describe 模式启动：

```bash
OPENCLAW_VISUAL_MODE=describe
```

此时 adapter 会调用 OpenClaw 图像描述能力：

```text
openclaw capability image describe-many
```

输入是当前图片和最近关键帧路径，输出会被规范化为：

```text
visual_observations
caption
visual_observation
landmarks
objects
spatial_cues
navigation_relevance
```

当前有两种写入路径。

### 2.1 Planner 显式写入

OpenClaw planner 可以直接返回：

```json
{
  "intent": "write_memory",
  "tool_name": "MemoryWriteSkill"
}
```

runtime 会基于 `keyframe_candidate`、`current_image_path` 和匹配的 `visual_observations` 补全写入字段。

### 2.2 Runtime 自动写入

即使 planner 没有显式返回 `write_memory`，只要本步满足：

```text
visual_analysis.ran = true
```

并且有匹配的 visual observation，runtime 也会自动写入：

```text
visual_analysis.observations
-> VisualMemoryCuratorSkill
-> MemoryWriteSkill
-> recent_visual_memories
```

写入内容通常包括：

```text
scene_id
episode_id
step_id
image_path
caption / visual_observation
objects / landmarks / spatial_cues
write_gate
memory_scope = episode
memory_namespace = episode:<scene_id>:<episode_id>
write_type = episodic_keyframe
```

当前默认写入范围是 episode-local，也就是优先服务当前 episode，避免跨 episode 误用未来信息或无关视觉记忆。

## 3. Visual Memory Recall

Visual memory recall 由以下链路完成：

```text
MemoryQuerySkill
-> MemoryManager.recall()
-> spatial memory client
```

当前有两种召回触发方式。

### 3.1 Planner 显式召回

OpenClaw planner 可以返回：

```json
{
  "intent": "recall_memory",
  "tool_name": "MemoryQuerySkill"
}
```

runtime 根据当前 instruction、scene、episode 和 step 查询相关 memory。

### 3.2 Runtime 自动召回

如果 runtime 已经记录了最近写入的 visual memories，后续 step 可以自动做 pre-planner 或 after-write recall。

默认召回范围限制在当前 episode：

```text
allowed_scopes = ["episode"]
memory_namespace = episode:<scene_id>:<episode_id>
```

这样做是为了避免当前 episode 误召回其他 episode 或未来路径信息。

召回结果会被拆成三类上下文：

```text
policy_context   -> 给 JanusVLN NavigationPolicySkill
control_context  -> 给 critic / runtime 判断
executor_context -> 给 executor / anchor
```

真正进入 JanusVLN policy 的主要是压缩后的短文本：

```text
memory_context_text
```

当前默认不把 `memory_images` 直接传给 JanusVLN policy，避免外部 memory image 污染原始图像历史或引起输入语义混乱。

## 4. 总结

当前 visual-memory 方案可以概括为：

```text
连续视觉流
-> 固定间隔保存关键帧
-> 对关键帧做 VLM 文字描述
-> 将稳定地标、空间关系、可导航线索写入 episode memory
-> 后续 step 召回相关短文本 memory_context_text
-> 注入 JanusVLN policy 辅助动作决策
```

它的核心思想不是让 OpenClaw planner 直接长期持有所有图片，而是把连续图像流稀疏化、文字化、记忆化，再以短文本形式召回给导航策略使用。

