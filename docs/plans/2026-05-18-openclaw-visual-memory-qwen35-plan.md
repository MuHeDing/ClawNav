# OpenClaw Visual Memory Qwen3.5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let OpenClaw consume actual current-frame visual content, use Qwen3.5 vision-language analysis to produce structured visual memories, store and organize those memories, recall them during navigation, and evaluate whether the training-free inference-time system improves open-world embodied navigation.

**Architecture:** Keep JanusVLN as the low-level navigation policy and build the system around it. OpenClaw becomes the inference-time planner and visual-memory curator: ClawNav saves frame artifacts, OpenClaw/Qwen3.5 analyzes selected images, MemoryWriteSkill stores structured visual evidence, MemoryQuerySkill recalls it, and OpenClaw decides when to recall, act, verify, write, or replan. Raw image bytes are not pushed through JSON gateway requests; image files are passed to OpenClaw's image capability or an explicit VLM adapter, and the resulting visual observations are injected into the planner prompt and memory records.

**Tech Stack:** Python, pytest, OpenClaw CLI, Qwen3.5 vision-language model via OpenClaw image capability, existing ClawNav Harness/OpenClaw runtime, SpatialMemory/FakeSpatialMemory clients, Habitat VLN evaluation scripts.

---

## Current State

The current code does not yet pass image content into OpenClaw planner reasoning.

- `src/evaluation_harness.py` saves current frames and keyframes as artifacts and passes only paths through `current_image_path`, `keyframe_candidate.image_path`, and `recent_keyframe_paths`.
- `src/harness/openclaw/openclaw_cli_plan_gateway.py` injects those paths into the text prompt as `Visual context image paths`, but it does not call a VLM on the images.
- `src/harness/openclaw/runtime.py` can now enrich `write_memory` with `image_path`, `should_write=True`, and `write_type=episodic_keyframe`, but the stored memory is still path-backed unless another component produces captions/semantic observations.
- `src/harness/memory/memory_manager.py` can return `memory_context_text` and `memory_images`; `NavigationPolicySkill` can optionally pass recalled image paths to JanusVLN, but OpenClaw itself still sees text/path context only.

This matches the reference document's early OpenClaw-compatible Harness direction, but not the final goal of visual-memory-centered OpenClaw planning.

## Target Data Flow

```text
Habitat frame
  -> ClawNav saves current frame / keyframe artifact
  -> OpenClaw visual analyzer calls Qwen3.5 on selected image files
  -> visual observations become structured text + optional tags/entities
  -> OpenClaw agent planner receives instruction + state + visual observations
  -> planner emits intent:
       act | recall_memory | write_memory | verify_progress | replan
  -> MemoryWriteSkill stores visual memory record:
       image_path, caption, landmarks, objects, place cues, step_id, source, confidence
  -> MemoryQuerySkill recalls relevant visual memories
  -> policy_context/control_context/executor_context guide future planning and action
```

## Key Decisions

1. **Do not pass raw image bytes through `/plan` JSON.**
   Keep gateway payloads JSON-safe and compact. Use image files as local artifacts and let OpenClaw/Qwen3.5 read them through a visual analysis capability.

2. **Use OpenClaw image capability as the first integration surface.**
   Local CLI supports:
   `openclaw capability image describe --file <path> --prompt <text> --json`
   and `describe-many`. This is the lowest-risk path to make OpenClaw see image content without changing the OpenClaw agent CLI.

3. **Visual analysis is a skill-stage input, not just prompt decoration.**
   The output should be structured and written into memory, not only appended to the planner prompt. Otherwise the system collapses into one-step visual captioning.

4. **Memory has three outputs, following the reference document.**
   `policy_context` helps JanusVLN, `control_context` helps OpenClaw decide recall/replan/stop verification, and `executor_context` prepares future embodied control.

5. **Training-free claim requires ablations.**
   The evaluation must compare baseline JanusVLN, path-only OpenClaw, visual-analysis OpenClaw, visual memory write/recall, and full OpenClaw skill orchestration.

## Strengthened Visual Memory Mechanisms

### Retrieval Index

Visual memory must be indexed by a purpose-built retrieval document, not by raw captions alone. Each `MemoryWriteSkill` record should include a derived `retrieval_text` field composed from non-oracle, inference-time evidence:

```text
retrieval_text =
  instruction_context
  + active_subgoal
  + caption
  + visual_observation
  + objects
  + landmarks
  + place_category
  + spatial_cues
  + navigation_relevance
  + action_context
```

The storage schema should separate retrieval and display fields:

- `retrieval_text`: compact text used for semantic indexing.
- `caption`: Qwen3.5 visual caption.
- `visual_observation`: structured navigation-focused observation.
- `objects`: normalized object names.
- `landmarks`: stable place/object anchors useful for navigation.
- `place_category`: hallway, room, doorway, stairs, kitchen-like area, living-room-like area, etc.
- `spatial_cues`: relative layout cues such as "doorway ahead", "sofa on right", "corridor bends left".
- `navigation_relevance`: why this frame is worth recalling.
- `image_path`: local artifact path.
- `memory_scope`: `episode | scene | task`.
- `memory_namespace`: stable namespace used to prevent leakage and control retrieval.
- `scene_id`, `episode_id`, `step_id`.
- `source_image_role`: `current | keyframe | recalled`.
- `created_by`: `openclaw_visual_analyzer | openclaw_planner | memory_curator`.

Retrieval should support filters before semantic ranking:

- `memory_scope=episode`: only the current episode namespace.
- `memory_scope=scene`: same scene, allowed prior episodes only.
- `memory_scope=task`: task-level strategy memories, not frame evidence.
- `scene_id`: required for scene memory.
- `episode_id`: required for episode memory.
- `memory_source`: `episode-local`, `train-scene-only`, or another approved non-oracle source.

Query construction should combine current task and current visual evidence:

```text
query_text =
  instruction
  + active_subgoal
  + current visual observation
  + planner uncertainty reason
  + progress critic signal
```

Ranking should be two-stage:

1. Semantic retrieval over `retrieval_text`.
2. Lightweight rerank by scope, recency, landmark overlap, object overlap, and confidence.

This avoids retrieving a visually similar but task-irrelevant frame.

### Write Gating

The system must not write every frame. A frame becomes a memory candidate only if it passes a cheap candidate gate and then a visual-memory curator gate.

Candidate gate:

- `step_id == 0`
- or `step_id % keyframe_interval == 0`
- or progress critic reports stuck/oscillation/uncertainty
- or OpenClaw planner requests memory write
- or current visual observation has high novelty compared with recent observations
- or a stable landmark/object/place cue is detected

Curator gate:

- write if Qwen3.5 identifies a navigation-relevant landmark, object, place category, doorway, room transition, or topological cue
- write if the frame can support future recall, replan, or STOP verification
- skip if caption is generic, duplicate, low-confidence, or visually redundant with a recent stored memory
- skip if the only evidence is oracle-derived or future-looking

The write decision should produce an explicit `write_gate` object:

```json
{
  "candidate_reason": "interval|stuck|novelty|planner_request|landmark",
  "curator_decision": "write|skip",
  "curator_reason": "doorway landmark useful for return route",
  "novelty_score": 0.71,
  "duplicate_of_memory_id": null,
  "confidence": 0.84
}
```

`MemoryWriteSkill` should store `write_gate` with the record. This is necessary for later analysis of memory pollution and useful recall.

### Scene / Episode Memory Namespace

Memory must be partitioned so that evaluation does not leak future information.

Use these scopes:

- `episode`: memories written during the current episode. Namespace format: `episode:<scene_id>:<episode_id>`.
- `scene`: memories from allowed prior episodes or training-scene prior. Namespace format: `scene:<scene_id>:<memory_source>`.
- `task`: cross-scene strategy memories that do not contain frame-specific future evidence. Namespace format: `task:<dataset_or_run_id>`.

Rules:

- Episode memory can include frames from the current episode only up to the current step.
- Scene memory can include only approved prior experience, never future frames from the evaluation episode.
- Task memory can include strategy summaries, but not hidden target coordinates or oracle path information.
- Every query must declare allowed scopes. Default for online evaluation should be `episode` first, then approved `scene` if enabled.
- Every memory record must include `memory_source` and `memory_namespace`.

Recommended query order:

```text
1. Query episode namespace for recently seen landmarks.
2. If confidence is low and scene prior is enabled, query scene namespace.
3. If still low and replanning is needed, query task namespace for recovery strategy.
```

### Recall Usage Log

Every recall must be logged as a causal event, not only as a list of hits. The trace should make it possible to answer whether memory actually changed behavior.

Per recall event:

```json
{
  "event_type": "memory_recall",
  "step_id": 18,
  "query_text": "...",
  "allowed_scopes": ["episode", "scene"],
  "selected_namespace": "episode:s1:e7",
  "num_hits": 3,
  "hit_ids": ["mem-001", "mem-004"],
  "hit_image_paths": [".../step_000010.png"],
  "hit_captions": ["..."],
  "best_hit_id": "mem-001",
  "recall_confidence": 0.82,
  "used_by_planner": true,
  "used_by_policy": false,
  "used_by_critic": true,
  "planner_intent_before_recall": "act",
  "planner_intent_after_recall": "replan",
  "action_before_recall": "TURN_LEFT",
  "action_after_recall": "MOVE_FORWARD",
  "action_changed_after_recall": true,
  "stop_blocked_after_recall": false,
  "replan_created_after_recall": true
}
```

This log is required for the system claim. Without it, success improvements cannot be attributed to visual memory rather than random planner variation.

## Implementation Units

### Unit 1: OpenClaw Visual Analyzer Adapter

**Files:**
- Create: `src/harness/openclaw/visual_analyzer.py`
- Test: `tests/test_openclaw_visual_analyzer.py`

**Behavior:**
- Accept one or more image paths.
- Call `openclaw capability image describe` or `describe-many`.
- Parse JSON or text output into stable records:
  - `image_path`
  - `caption`
  - `landmarks`
  - `objects`
  - `spatial_cues`
  - `navigation_relevance`
  - `confidence` when available
- Cache by image path so the same frame is not analyzed repeatedly.
- Fail soft: if VLM analysis fails, return an error field and preserve existing path-only behavior.

**Tests:**
- Fake runner returns JSON description for one image.
- Fake runner returns JSON descriptions for multiple images.
- Repeated path uses cache and does not invoke runner twice.
- Nonzero return code produces a fallback observation without crashing planner.

### Unit 2: Inject Visual Content Into OpenClaw Planner Prompt

**Files:**
- Modify: `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- Test: `tests/test_openclaw_cli_plan_gateway.py`

**Behavior:**
- Add configuration:
  - `openclaw_visual_mode`: `path | describe`
  - `openclaw_visual_max_images`: default 2
  - `openclaw_visual_timeout_ms`: default 30000
  - `openclaw_visual_model`: optional model override, e.g. Qwen3.5 VLM
- In `describe` mode, analyze `current_image_path` plus most recent keyframe paths.
- Add `visual_observations` to prompt payload.
- Preserve existing path-only mode as default-compatible behavior.

**Tests:**
- Path mode prompt remains path-only.
- Describe mode includes visual observation text and image paths.
- Describe mode limits images by `openclaw_visual_max_images`.
- Visual analyzer failure keeps planner call alive with path-only fallback.

### Unit 3: Make Visual Memory Records First-Class

**Files:**
- Modify: `src/harness/skills/memory_write.py`
- Modify: `src/harness/memory/memory_manager.py`
- Modify: `src/harness/memory/protocol.py`
- Test: `tests/test_memory_skills.py`
- Test: `tests/test_memory_manager.py`
- Test: `tests/test_spatial_memory_protocol.py`

**Behavior:**
- Extend memory write payload/record with:
  - `caption`
  - `visual_observation`
  - `landmarks`
  - `objects`
  - `place_category`
  - `spatial_cues`
  - `navigation_relevance`
  - `source_image_role`: `current | keyframe | recalled`
  - `retrieval_text`
  - `memory_scope`: `episode | scene | task`
  - `memory_namespace`
  - `write_gate`
- MemoryWriteSkill stores these fields alongside `image_path`.
- MemoryManager recall builds richer:
  - `memory_context_text`
  - `memory_images`
  - `control_context.best_landmark`
  - `control_context.recall_confidence`
  - `executor_context.topological_anchor`
- MemoryManager constructs query text from instruction, active subgoal, current visual observation, planner uncertainty, and progress critic signal.
- MemoryManager filters by namespace before semantic ranking.

**Tests:**
- Visual fields survive MemoryWriteSkill record creation.
- Recall context includes caption/object/place evidence.
- Memory image paths remain bounded by `max_memory_images`.
- No oracle fields enter memory records or decision contexts.
- Retrieval text includes caption, landmarks, objects, spatial cues, and navigation relevance.
- Episode-scoped recall never returns a different episode namespace.
- Scene-scoped recall returns only approved prior-scene memory.
- Duplicate or low-value write candidates are skipped when write gating says `skip`.

### Unit 4: Visual Memory Curator Intent

**Files:**
- Modify: `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- Modify: `src/harness/openclaw/runtime.py`
- Possibly create: `src/harness/skills/visual_memory_curator.py`
- Test: `tests/test_openclaw_runtime_bridge.py`
- Test: `tests/test_openclaw_cli_plan_gateway.py`

**Behavior:**
- Teach OpenClaw planner prompt that `write_memory` should include visual observations when:
  - step is a keyframe candidate
  - scene changes materially
  - important landmark/object appears
  - progress critic detects uncertainty/stuck behavior
- Runtime merges VLM observations into `MemoryWriteSkill` arguments.
- If the planner chooses `write_memory` without explicit visual fields, runtime can attach available `visual_observations` from payload.
- The planner prompt must ask OpenClaw to return a `write_gate` rationale for every `write_memory` decision.
- The planner prompt must distinguish episode memory from scene memory and must not request current-episode future evidence.

**Tests:**
- `write_memory` decision stores current image caption.
- Planner explicit fields override analyzer-generated defaults.
- Keyframe candidate observation is preferred over generic current-frame observation.
- Planner write decisions include candidate reason and curator reason.

### Unit 5: Qwen3.5 VLM Configuration And Run Scripts

**Files:**
- Modify: `scripts/start_openclaw_cli_plan_gateway.sh`
- Modify: `scripts/evaluation_openclaw_gateway.sh`
- Modify: `docs/runbooks/external-openclaw-gateway.md`
- Test: `tests/test_evaluation_scripts.py`
- Test: `tests/test_phase4_docs.py`

**Behavior:**
- Add env vars:
  - `OPENCLAW_VISUAL_MODE=describe`
  - `OPENCLAW_VISUAL_MODEL=qwen/<exact-qwen3.5-vl-model-id>`
  - `OPENCLAW_VISUAL_MAX_IMAGES=2`
  - `OPENCLAW_VISUAL_TIMEOUT_MS=30000`
- Document how to verify:
  - `openclaw capability image describe --file <sample> --model <qwen-model> --json`
  - `openclaw agent --agent main --json --message '只回复 OK'`
- Keep path-only mode available for baseline and debugging.

**Tests:**
- Scripts pass visual mode args to gateway.
- Runbook includes image describe smoke test and Qwen model configuration.

### Unit 6: Trace And Diagnostics

**Files:**
- Modify: `src/harness/logger.py` or existing runtime logging path
- Modify: `src/evaluation_harness.py`
- Test: `tests/test_harness_logger.py`
- Test: `tests/test_evaluation_harness_openclaw_runtime.py`

**Behavior:**
- Per step trace records:
  - selected image paths
  - whether VLM analysis ran
  - visual observation summary
  - memory writes
  - memory recall hits
  - whether recalled visual memory changed planner intent or action
  - latency and failure reason
- recall event fields listed in the Recall Usage Log section
- memory namespace, memory scope, and memory source for every write and recall
- write gate decision and duplicate target when a candidate is skipped
- Store visual analyzer latency separately from OpenClaw agent latency.

**Tests:**
- Trace includes visual analysis metadata when enabled.
- Trace omits large raw captions if configured to keep logs compact.
- VLM failure is visible in trace without breaking navigation.
- Trace records `action_changed_after_recall`.
- Trace records `used_by_planner`, `used_by_policy`, and `used_by_critic`.
- Trace records namespace and scope for every memory hit.

### Unit 7: Evaluation And Ablations

**Files:**
- Modify: `scripts/run_openclaw_vln_ablation_matrix.py`
- Modify: `scripts/summarize_openclaw_vln_ablation.py`
- Possibly create: `scripts/evaluation_openclaw_visual_memory.sh`
- Test: `tests/test_openclaw_ablation_runner.py`
- Test: `tests/test_openclaw_ablation_summary.py`

**Ablation Ladder:**
- A0: JanusVLN baseline.
- A1: OpenClaw path-only planner, no VLM content.
- A2: OpenClaw + Qwen3.5 visual observations in prompt, no memory write.
- A3: A2 + episodic visual memory write.
- A4: A3 + visual memory recall into planner/control context.
- A5: A4 + progress critic + replan.
- A6: scene long-term memory across episodes, no leakage.
- A7: full system with write gating, namespace-filtered recall, and causal recall usage logging.

**Metrics:**
- Navigation: SR, SPL, NE, oracle success where already standard offline metrics.
- System: recall rate, useful recall rate, memory write count, visual analyzer latency, OpenClaw agent latency, token size, memory storage size.
- Causality diagnostics: action changed after visual recall, STOP blocked after recall, replan success window.
- Memory quality: write acceptance rate, duplicate skip rate, useful recall rate, namespace hit distribution, visual-memory pollution rate.

**Tests:**
- Matrix includes all visual-memory variants.
- Summary reports memory and visual-analysis metrics.
- No-leakage guard verifies oracle metrics are not present in online planner payloads.

## Risks

- **Latency:** Qwen3.5 image analysis every step may be too slow. Mitigation: analyze only current frame plus one keyframe, cache results, and run only on keyframe/uncertainty intervals.
- **Prompt bloat:** Raw descriptions can grow. Mitigation: structured compact schema and character limits.
- **Double vision confusion:** JanusVLN already sees frames; OpenClaw visual analysis should focus on planning/memory, not duplicate low-level action selection.
- **Memory pollution:** Writing every frame will degrade recall. Mitigation: curator criteria and bounded write budget.
- **No-leakage:** Scene long-term memory must be built only from allowed prior experience, never future frames or oracle path.

## Recommended First Slice

Implement Units 1-3 first:

1. Add `OpenClawVisualAnalyzer` with fake-runner tests.
2. Enable `openclaw_visual_mode=describe` in `OpenClawCliPlanPlanner`.
3. Store analyzer outputs in `MemoryWriteSkill` records.
4. Run a smoke test on one short VLN episode with OpenClaw path-only vs describe mode.

This first slice proves the missing point in the current code: OpenClaw receives visual content derived from the image, not only a file path.
