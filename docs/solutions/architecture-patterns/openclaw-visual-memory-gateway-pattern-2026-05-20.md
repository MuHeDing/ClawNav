---
title: OpenClaw visual memory gateway pattern
date: 2026-05-20
category: docs/solutions/architecture-patterns
module: ClawNav OpenClaw visual memory
problem_type: architecture_pattern
component: assistant
severity: medium
applies_when:
  - "OpenClaw needs current-frame visual content during VLN inference"
  - "Gateway payloads must stay JSON-safe and non-oracle"
  - "Visual observations should become reusable memory, not only prompt text"
tags: [openclaw, visual-memory, qwen-vl, gateway, vln, memory-recall]
---

# OpenClaw visual memory gateway pattern

## Context

OpenClaw visual memory is the inference-time layer that lets ClawNav expose navigation-relevant image evidence to an external OpenClaw planner without sending raw image bytes through `/plan` JSON. The design comes from `docs/plans/2026-05-18-openclaw-visual-memory-qwen35-plan.md`, `docs/plans/2026-05-19-openclaw-visual-memory-completion-plan.md`, and the external gateway plan.

The key boundary is:

```text
Habitat frame
  -> ClawNav saves current/keyframe image artifact
  -> /plan payload carries image paths
  -> OpenClaw gateway optionally calls Qwen VLM image describe
  -> structured visual observations enter planner prompt and memory tools
  -> MemoryWriteSkill stores visual evidence
  -> MemoryQuerySkill recalls scoped visual memories later
```

Do not conflate this with JanusVLN's internal image-token path. JanusVLN still consumes raw visual tensors in the Qwen/VGGT model. OpenClaw visual memory is a planner and memory path that converts selected image artifacts into structured text evidence.

## Guidance

Use image artifact paths as the gateway contract, and perform visual analysis at the OpenClaw adapter boundary.

ClawNav creates the image-path payload in `src/evaluation_harness.py`:

```text
current_image_path
keyframe_candidate.image_path
recent_keyframe_paths
```

The `/plan` adapter is started by `scripts/start_openclaw_cli_plan_gateway.sh`. Path-only mode is the baseline:

```bash
HOST=127.0.0.1 PORT=8011 ./scripts/start_openclaw_cli_plan_gateway.sh
```

Visual mode is enabled on the gateway, not inside the evaluation script:

```bash
OPENCLAW_VISUAL_MODE=describe \
OPENCLAW_VISUAL_MODEL=qwen/qwen3.5-flash \
OPENCLAW_VISUAL_MAX_IMAGES=2 \
OPENCLAW_VISUAL_TIMEOUT_MS=90000 \
HOST=127.0.0.1 PORT=8011 ./scripts/start_openclaw_cli_plan_gateway.sh
```

Then run the evaluation against that gateway:

```bash
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=180 \
HARNESS_DEBUG_MAX_EPISODES=30 \
OUTPUT_PATH=results/clawnav_openclaw_gateway_30_visual_maxsteps0_timeout180 \
./scripts/evaluation_openclaw_gateway.sh
```

Inside the adapter, `OpenClawVisualAnalyzer` calls:

```bash
openclaw capability image describe-many --file <path> --prompt <prompt> --json
```

It returns compact fields such as `caption`, `visual_observation`, `landmarks`, `objects`, `spatial_cues`, `navigation_relevance`, and `confidence`, plus analysis metadata like `vlm_latency_ms`, cache hits, and failures.

Those observations should flow into three places:

- planner prompt payload as `visual_observations`
- `MemoryWriteSkill` arguments for first-class visual memory records
- `MemoryQuerySkill` query text for recall conditioned on current visual evidence

Use the visual-memory completion rules from the plans:

- Push `allowed_scopes` and `memory_namespace` into semantic retrieval before backend ranking.
- Keep a defensive post-retrieval filter for leakage prevention.
- Rerank hits by query overlap, confidence, recency, and scope.
- Store visual fields separately from `retrieval_text`.
- Gate writes through `VisualMemoryCuratorSkill` so generic, duplicate, or low-confidence frames do not pollute memory.
- Log visual latency separately from OpenClaw agent/runtime latency.
- Log recall causality: planner intent before/after recall, action before/after recall, whether recall changed behavior, and whether a replan was created.

## Why This Matters

This pattern keeps the online planner contract small and non-oracle while still allowing OpenClaw to reason over actual current-frame visual content. Passing raw images through `/plan` would make the gateway heavier, harder to test, and easier to break across process boundaries. Passing only paths without VLM analysis leaves OpenClaw blind to visual content and limits it to path decoration.

The memory layer matters because one-step captioning is not enough for the system claim. Visual observations need to be written, scoped, recalled, and audited so later behavior can be attributed to visual memory instead of planner randomness.

## When to Apply

- Use `path` mode when validating the gateway contract, fallback behavior, and non-oracle JSON payloads.
- Use `describe` mode when evaluating OpenClaw visual observations or any visual-memory ablation.
- Use `episode-local` memory for online current-episode visual memories.
- Use `scene-prior` or `train-scene-only` only when the memory manifest is known not to contain future evaluation-episode evidence.
- Treat `MemoryWriteSkill` and `MemoryQuerySkill` metrics as unavailable unless the run actually emits `memory_writes` or `recall_usage` trace events.

## Examples

The current diagnostic run is:

```text
results/clawnav_openclaw_gateway_30_visual_maxsteps0_timeout180
```

Observed evidence from that run:

- 30 result rows and 30 harness trace rows.
- `visual_analysis_steps=30`, `visual_analysis_failures=0`.
- Average VLM latency was about `39164 ms`; min/max were about `24042 ms` and `52504 ms`.
- `planner_backend=gateway` and `planner_fallback=false` for all trace rows.
- `oracle_leakage_steps=0`.
- All planned intents were `act`; actions were `MOVE_FORWARD` 28 times, `TURN_LEFT` once, and `TURN_RIGHT` once.
- `memory_write_attempts=0` and `memory_recall_events=0`.
- Navigation summary was `success=0.0`, `spl=0.0`, `length=30`, and average `ne=10.1146`.

Interpret this run as a visual gateway smoke/diagnostic result: OpenClaw received visual observations and produced action overrides without fallback or oracle leakage, but it did not exercise visual memory write/recall. It should not be used as evidence that A3/A4 visual memory improves navigation.

For ablations, use the ladder from `scripts/run_openclaw_vln_ablation_matrix.py`:

- A0: JanusVLN low-memory baseline.
- A1: OpenClaw bridge/path-only.
- A2: OpenClaw plus Qwen visual observations in prompt.
- A3: A2 plus episodic visual memory write.
- A4: A3 plus visual memory recall.
- A5: A4 plus progress critic and replan.
- A6: scene prior memory with no leakage.
- A7: full visual memory system with critic, curator, scoped recall, and causal logs.

## Related

- `docs/plans/2026-05-18-openclaw-visual-memory-qwen35-plan.md`
- `docs/plans/2026-05-19-openclaw-visual-memory-completion-plan.md`
- `docs/plans/2026-05-14-external-openclaw-plan-gateway.md`
- `docs/runbooks/external-openclaw-gateway.md`
- `docs/protocols/openclaw-plan-gateway-api.md`
- `scripts/start_openclaw_cli_plan_gateway.sh`
- `scripts/evaluation_openclaw_gateway.sh`
- `scripts/run_openclaw_vln_ablation_matrix.py`
- `scripts/summarize_openclaw_vln_ablation.py`
