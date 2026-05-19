# OpenClaw Visual Memory Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the remaining OpenClaw visual-memory mechanisms: pre-retrieval filtering, reranking, write novelty/duplicate gating, causal recall diagnostics, and VLM latency attribution.

**Architecture:** Keep the existing planner/runtime/memory boundaries. Push scope/namespace filters down into memory query payloads before semantic search, add a small deterministic reranker inside `MemoryManager`, extend `VisualMemoryCuratorSkill` with recent-memory comparison, and make runtime diagnostics record before/after planner decisions through an explicit recall-then-replan path. Attribute VLM latency at the gateway visual analyzer layer and propagate it through `/plan` metadata into runtime traces.

**Tech Stack:** Python, pytest, existing ClawNav harness, OpenClaw CLI gateway, `MemoryManager`, `SpatialMemoryHttpClient`, `OpenClawVLNRuntime`, `OpenClawVisualAnalyzer`.

---

## Current Gaps

- `MemoryManager.recall()` calls `client.query_semantic()` before applying `allowed_scopes` and `memory_namespace`, so leakage prevention is post-filter only.
- `SpatialMemoryHttpClient.query_semantic()` cannot send scope/namespace filters to the backend.
- Recall ordering is whatever the backend returns; no recency, landmark/object overlap, scope priority, or confidence rerank exists.
- `VisualMemoryCuratorSkill` can skip generic/low-confidence frames, but does not compute novelty against recent memories or identify duplicate targets.
- Recall usage logs have the requested fields, but `planner_intent_before_recall` and `planner_intent_after_recall` are not a real before/after comparison.
- Visual analyzer latency is not measured at the VLM adapter boundary or propagated separately from OpenClaw agent/runtime latency.

## Task 1: Add Filter-Aware Memory Client Query API

**Files:**
- Modify: `src/harness/memory/spatial_memory_client.py`
- Modify: `src/harness/memory/memory_manager.py`
- Modify: `src/harness/skills/memory_query.py`
- Test: `tests/test_spatial_memory_client.py`
- Test: `tests/test_memory_manager.py`
- Test: `tests/test_memory_skills.py`

**Step 1: Write failing tests for pre-retrieval filters**

Add a fake client that records query kwargs:

```python
class RecordingFilterClient(BaseSpatialMemoryClient):
    def __init__(self):
        super().__init__()
        self.calls = []

    def query_semantic(
        self,
        text,
        n_results=5,
        allowed_scopes=None,
        memory_namespace="",
        memory_source="",
    ):
        self.calls.append(
            {
                "text": text,
                "n_results": n_results,
                "allowed_scopes": allowed_scopes,
                "memory_namespace": memory_namespace,
                "memory_source": memory_source,
            }
        )
        return []
```

Test:

```python
def test_memory_manager_passes_scope_filters_before_semantic_query():
    client = RecordingFilterClient()
    manager = MemoryManager(client)

    manager.recall(
        text="find doorway",
        step_id=4,
        allowed_scopes=["episode"],
        memory_namespace="episode:s1:e1",
    )

    assert client.calls[0]["allowed_scopes"] == ["episode"]
    assert client.calls[0]["memory_namespace"] == "episode:s1:e1"
```

Add HTTP payload test:

```python
def test_spatial_http_semantic_query_sends_scope_filters(requests_mock):
    requests_mock.post(
        "http://memory/query/semantic/text",
        json={"results": []},
    )
    client = SpatialMemoryHttpClient("http://memory", memory_source="episode-local")

    client.query_semantic(
        "doorway",
        n_results=3,
        allowed_scopes=["episode"],
        memory_namespace="episode:s1:e1",
    )

    payload = requests_mock.last_request.json()
    assert payload["allowed_scopes"] == ["episode"]
    assert payload["memory_namespace"] == "episode:s1:e1"
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_memory_manager.py::test_memory_manager_passes_scope_filters_before_semantic_query tests/test_spatial_memory_client.py::test_spatial_http_semantic_query_sends_scope_filters -q
```

Expected: FAIL because `query_semantic()` does not accept/pass these arguments.

**Step 3: Implement filter-aware query signatures**

Change `BaseSpatialMemoryClient.query_semantic()` to:

```python
def query_semantic(
    self,
    text: str,
    n_results: int = 5,
    allowed_scopes: Optional[List[str]] = None,
    memory_namespace: str = "",
    memory_source: str = "",
) -> List[MemoryHit]:
    raise NotImplementedError
```

Update `FakeSpatialMemoryClient` to accept the same parameters and optionally place scope/namespace in hit metadata only when useful for tests.

Update `SpatialMemoryHttpClient.query_semantic()`:

```python
def query_semantic(
    self,
    text: str,
    n_results: int = 5,
    allowed_scopes: Optional[List[str]] = None,
    memory_namespace: str = "",
    memory_source: str = "",
) -> List[MemoryHit]:
    return self._post_query(
        "/query/semantic/text",
        build_query_payload(
            query_type="semantic",
            text=text,
            n_results=n_results,
            memory_source=memory_source or self.memory_source,
            allowed_scopes=allowed_scopes,
            memory_namespace=memory_namespace,
        ),
    )
```

Update `MemoryManager.recall()` so the backend receives filters:

```python
hits = self.client.query_semantic(
    query_text,
    n_results=n_results,
    allowed_scopes=allowed_scopes,
    memory_namespace=memory_namespace,
)
```

Keep `_filter_hits()` after retrieval as a defensive guard.

**Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_memory_manager.py tests/test_spatial_memory_client.py tests/test_memory_skills.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/harness/memory/spatial_memory_client.py src/harness/memory/memory_manager.py src/harness/skills/memory_query.py tests/test_spatial_memory_client.py tests/test_memory_manager.py tests/test_memory_skills.py
git commit -m "feat: pass memory scope filters before retrieval"
```

## Task 2: Add Deterministic Memory Reranking

**Files:**
- Modify: `src/harness/memory/memory_manager.py`
- Test: `tests/test_memory_manager.py`

**Step 1: Write failing rerank tests**

Test landmark/object overlap beats raw backend order:

```python
def test_memory_manager_reranks_by_visual_overlap_and_confidence():
    client = StaticHitClient(
        [
            MemoryHit(
                memory_id="generic",
                memory_type="semantic",
                name="generic hallway",
                confidence=0.95,
                metadata={"landmarks": ["window"], "objects": ["chair"], "memory_scope": "episode"},
            ),
            MemoryHit(
                memory_id="door",
                memory_type="semantic",
                name="doorway memory",
                confidence=0.70,
                metadata={"landmarks": ["red door"], "objects": ["sofa"], "memory_scope": "episode"},
            ),
        ]
    )
    manager = MemoryManager(client)

    recall = manager.recall(
        text="go toward the red door near the sofa",
        step_id=3,
        visual_observation="red door ahead with sofa on right",
        allowed_scopes=["episode"],
    )

    assert recall.hits[0].memory_id == "door"
```

Test recency breaks ties:

```python
def test_memory_manager_reranks_recent_hit_when_scores_tie():
    older = MemoryHit(memory_id="old", confidence=0.8, metadata={"step_id": 2})
    newer = MemoryHit(memory_id="new", confidence=0.8, metadata={"step_id": 9})
    manager = MemoryManager(StaticHitClient([older, newer]))

    recall = manager.recall("doorway", step_id=10)

    assert recall.hits[0].memory_id == "new"
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_memory_manager.py::test_memory_manager_reranks_by_visual_overlap_and_confidence tests/test_memory_manager.py::test_memory_manager_reranks_recent_hit_when_scores_tie -q
```

Expected: FAIL because ordering is unchanged.

**Step 3: Implement reranking**

Add helper methods in `MemoryManager`:

```python
def _rerank_hits(self, hits: List[MemoryHit], query_text: str, step_id: int) -> List[MemoryHit]:
    query_terms = self._term_set(query_text)
    return sorted(
        hits,
        key=lambda hit: self._rerank_score(hit, query_terms, step_id),
        reverse=True,
    )

def _rerank_score(self, hit: MemoryHit, query_terms: Set[str], step_id: int) -> float:
    metadata = hit.metadata or {}
    overlap_terms = self._metadata_terms(metadata)
    overlap = len(query_terms.intersection(overlap_terms))
    confidence = float(hit.confidence or 0.0)
    hit_step = metadata.get("step_id")
    recency = 0.0
    if isinstance(hit_step, int) and step_id >= hit_step:
        recency = max(0.0, 1.0 - min(step_id - hit_step, 50) / 50.0)
    scope_bonus = {"episode": 0.3, "scene": 0.15, "task": 0.05}.get(
        str(metadata.get("memory_scope") or ""),
        0.0,
    )
    return confidence + (0.35 * overlap) + (0.2 * recency) + scope_bonus
```

Tokenization should be simple and local:

```python
def _term_set(self, text: str) -> Set[str]:
    return {term for term in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(term) > 2}
```

Call after defensive filtering:

```python
hits = self._rerank_hits(hits, query_text=query_text, step_id=step_id)
```

**Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_memory_manager.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/harness/memory/memory_manager.py tests/test_memory_manager.py
git commit -m "feat: rerank visual memory recall hits"
```

## Task 3: Add Novelty And Duplicate Write Gating

**Files:**
- Modify: `src/harness/skills/visual_memory_curator.py`
- Modify: `src/harness/openclaw/runtime.py`
- Test: `tests/test_visual_memory_curator.py`
- Test: `tests/test_openclaw_runtime_bridge.py`

**Step 1: Write failing curator tests**

Create `tests/test_visual_memory_curator.py` if missing.

```python
def test_visual_memory_curator_skips_duplicate_recent_memory():
    skill = VisualMemoryCuratorSkill()
    payload = {
        "caption": "A red door next to a sofa.",
        "objects": ["sofa"],
        "landmarks": ["red door"],
        "spatial_cues": ["sofa on right"],
        "confidence": 0.9,
        "recent_visual_memories": [
            {
                "memory_id": "mem-1",
                "caption": "A red door next to a sofa.",
                "objects": ["sofa"],
                "landmarks": ["red door"],
                "spatial_cues": ["sofa on right"],
            }
        ],
    }

    result = skill.run(state=None, payload=payload)

    gate = result.payload["write_gate"]
    assert result.payload["should_write"] is False
    assert gate["curator_decision"] == "skip"
    assert gate["duplicate_of_memory_id"] == "mem-1"
    assert gate["novelty_score"] < 0.25
```

```python
def test_visual_memory_curator_writes_novel_landmark():
    skill = VisualMemoryCuratorSkill()
    payload = {
        "caption": "A stairway descends at the end of the corridor.",
        "landmarks": ["stairway"],
        "spatial_cues": ["corridor ends at stairs"],
        "confidence": 0.86,
        "recent_visual_memories": [
            {"memory_id": "mem-1", "landmarks": ["red door"], "objects": ["sofa"]}
        ],
    }

    result = skill.run(state=None, payload=payload)

    assert result.payload["should_write"] is True
    assert result.payload["write_gate"]["novelty_score"] > 0.5
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_visual_memory_curator.py -q
```

Expected: FAIL because novelty/duplicate fields do not exist.

**Step 3: Implement novelty helpers**

In `VisualMemoryCuratorSkill`, add:

```python
def _signature_terms(self, payload: Dict[str, Any]) -> Set[str]:
    terms = set()
    for key in ("caption", "visual_observation", "place_category", "navigation_relevance"):
        terms.update(self._terms(str(payload.get(key) or "")))
    for key in ("landmarks", "objects", "spatial_cues"):
        values = payload.get(key)
        if isinstance(values, list):
            for value in values:
                terms.update(self._terms(str(value)))
    return terms

def _novelty(self, payload: Dict[str, Any]) -> Tuple[float, str]:
    current = self._signature_terms(payload)
    if not current:
        return 0.0, ""
    best_similarity = 0.0
    duplicate_id = ""
    for memory in payload.get("recent_visual_memories") or []:
        if not isinstance(memory, dict):
            continue
        previous = self._signature_terms(memory)
        if not previous:
            continue
        similarity = len(current & previous) / max(1, len(current | previous))
        if similarity > best_similarity:
            best_similarity = similarity
            duplicate_id = str(memory.get("memory_id") or memory.get("id") or "")
    return 1.0 - best_similarity, duplicate_id if best_similarity >= 0.75 else ""
```

Update `run()`:

```python
novelty_score, duplicate_id = self._novelty(payload)
should_write = self._has_navigation_value(payload, reason, confidence)
if duplicate_id:
    should_write = False
```

Include in `write_gate`:

```python
"novelty_score": round(novelty_score, 4),
"duplicate_of_memory_id": duplicate_id or None,
```

**Step 4: Pass recent visual memories from runtime**

Add a bounded recent-memory list to `OpenClawVLNRuntime`:

```python
self.recent_visual_memories: List[Dict[str, Any]] = []
self.max_recent_visual_memories = 10
```

Before curator call in write path, attach:

```python
arguments.setdefault("recent_visual_memories", list(self.recent_visual_memories))
```

After a successful `MemoryWriteSkill` write, append a compact signature:

```python
self._remember_visual_memory(record)
```

Use fields:

```python
memory_id, image_path, caption, visual_observation, objects, landmarks, spatial_cues
```

Do not persist this helper list; it is an online gating aid only.

**Step 5: Run focused tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_visual_memory_curator.py tests/test_openclaw_runtime_bridge.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/harness/skills/visual_memory_curator.py src/harness/openclaw/runtime.py tests/test_visual_memory_curator.py tests/test_openclaw_runtime_bridge.py
git commit -m "feat: gate visual memory writes by novelty"
```

## Task 4: Implement Real Causal Recall Diagnostics

**Files:**
- Modify: `src/harness/openclaw/runtime.py`
- Modify: `src/harness/types.py` if `RuntimeResult` needs extra metadata fields
- Test: `tests/test_openclaw_runtime_bridge.py`

**Step 1: Write failing causal diagnostic test**

Add a planner fake that returns `recall_memory` first and `replan` after recall context is available.

```python
class RecallThenReplanPlanner:
    def __init__(self):
        self.payloads = []

    def plan(self, state, payload):
        self.payloads.append(payload)
        if payload.get("memory_context_text"):
            return PlannerDecision(
                intent="replan",
                tool_name="ReplannerSkill",
                arguments={"action_text": "MOVE_FORWARD", "active_subgoal": "follow recalled doorway"},
                reason="memory_changed_plan",
            )
        return PlannerDecision(
            intent="recall_memory",
            tool_name="MemoryQuerySkill",
            arguments={"text": "doorway", "allowed_scopes": ["episode"]},
            reason="need_memory",
        )
```

Test expected trace:

```python
def test_runtime_records_planner_intent_change_after_recall():
    result = runtime.step(state, {"policy_action": "TURN_LEFT"})

    event = result.runtime_metadata["recall_usage"][0]
    assert event["planner_intent_before_recall"] == "recall_memory"
    assert event["planner_intent_after_recall"] == "replan"
    assert event["replan_created_after_recall"] is True
    assert event["action_before_recall"] == "TURN_LEFT"
    assert event["action_after_recall"] == "MOVE_FORWARD"
    assert event["action_changed_after_recall"] is True
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_runtime_bridge.py::test_runtime_records_planner_intent_change_after_recall -q
```

Expected: FAIL because runtime does not perform or record an after-recall planning pass.

**Step 3: Add bounded after-recall replanning pass**

In `OpenClawVLNRuntime.step()`, after a successful `MemoryQuerySkill` call:

1. Merge `policy_context` into a copy of runtime payload.
2. Call planner once more with `memory_context_text`, `memory_images`, `control_context`, and a guard flag:

```python
if decision.intent == "recall_memory" and not payload.get("_after_recall_replan"):
    after_payload = dict(payload)
    after_payload["_after_recall_replan"] = True
    self._merge_tool_navigation_context(after_payload, memory_query_call)
    after_decision = self.planner.plan(state, after_payload)
```

3. Execute the after decision only when intent is `act` or `replan`; otherwise keep the original recall result and fall back to current behavior.
4. Store an internal `causal_recall` metadata object:

```python
{
    "before_decision": decision,
    "after_decision": after_decision,
    "before_action": runtime_context.get("policy_action"),
    "after_action": final_action_text,
}
```

Keep this bounded to one extra planner call per step to avoid loops.

**Step 4: Update `_recall_usage()`**

Pass causal metadata into `_recall_usage()` and fill:

```python
"planner_intent_before_recall": before_decision.intent,
"planner_intent_after_recall": after_decision.intent if after_decision else before_decision.intent,
"action_after_recall": final_action_text,
"replan_created_after_recall": after_decision.intent == "replan",
```

`used_by_planner` should mean the after-planning pass received memory context, not merely that the first intent was recall.

**Step 5: Run focused runtime tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_runtime_bridge.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/harness/openclaw/runtime.py src/harness/types.py tests/test_openclaw_runtime_bridge.py
git commit -m "feat: record causal recall replanning effects"
```

## Task 5: Attribute VLM Analyzer Latency Separately

**Files:**
- Modify: `src/harness/openclaw/visual_analyzer.py`
- Modify: `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- Modify: `src/harness/openclaw/gateway.py`
- Modify: `src/harness/openclaw/runtime.py`
- Test: `tests/test_openclaw_visual_analyzer.py`
- Test: `tests/test_openclaw_cli_plan_gateway.py`
- Test: `tests/test_openclaw_runtime_bridge.py`

**Step 1: Write failing analyzer metadata test**

```python
def test_visual_analyzer_records_latency_and_cache_status():
    runner = FakeRunner(json_stdout=[{"image_path": "/tmp/a.png", "caption": "door"}])
    analyzer = OpenClawVisualAnalyzer(run_openclaw=runner)

    observations = analyzer.analyze(["/tmp/a.png"])

    assert observations[0]["analysis_metadata"]["vlm_latency_ms"] >= 0
    assert observations[0]["analysis_metadata"]["cache_hit"] is False

    cached = analyzer.analyze(["/tmp/a.png"])
    assert cached[0]["analysis_metadata"]["cache_hit"] is True
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_visual_analyzer.py::test_visual_analyzer_records_latency_and_cache_status -q
```

Expected: FAIL because metadata is absent.

**Step 3: Implement analyzer timing**

In `OpenClawVisualAnalyzer.analyze()`:

```python
cached_paths = set(path for path in requested if path in self._cache)
```

In `_describe_and_cache()`:

```python
start = time.perf_counter()
result = self.run_openclaw(args, timeout_s)
latency_ms = (time.perf_counter() - start) * 1000.0
```

Attach metadata to each observation:

```python
observation["analysis_metadata"] = {
    "vlm_latency_ms": round(latency_ms, 3),
    "visual_model": self.model,
    "visual_mode": "describe",
    "cache_hit": False,
    "error": bool(observation.get("error")),
}
```

When returning cached observations, copy and set:

```python
observation["analysis_metadata"]["cache_hit"] = True
observation["analysis_metadata"]["vlm_latency_ms"] = 0.0
```

**Step 4: Propagate visual metadata through gateway response**

In `OpenClawCliPlanPlanner._agent_plan()` and `_heuristic_plan()` responses, include:

```python
"runtime_metadata": {
    "visual_analysis": self._visual_analysis_metadata(prompt_payload)
}
```

The metadata summary should contain:

```python
{
    "ran": True,
    "num_images": 2,
    "image_paths": [...],
    "vlm_latency_ms": total_uncached_latency,
    "cache_hits": 1,
    "failures": 0,
    "model": "...",
}
```

If the gateway response schema currently drops unknown keys, update `OpenClawGatewayClient` to preserve `runtime_metadata` from `/plan`.

**Step 5: Runtime trace uses gateway visual latency**

In `OpenClawVLNRuntime`, merge planner `runtime_metadata.visual_analysis` into final `runtime_metadata["visual_analysis"]`. Keep MemoryWriteSkill latency separate:

```python
metadata["visual_analysis"] = planner_metadata.get("visual_analysis", {})
metadata["openclaw_agent_latency_ms"] = planner_metadata.get("agent_latency_ms")
```

Do not overwrite gateway VLM latency with MemoryWriteSkill latency.

**Step 6: Run focused tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_visual_analyzer.py tests/test_openclaw_cli_plan_gateway.py tests/test_openclaw_runtime_bridge.py -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add src/harness/openclaw/visual_analyzer.py src/harness/openclaw/openclaw_cli_plan_gateway.py src/harness/openclaw/gateway.py src/harness/openclaw/runtime.py tests/test_openclaw_visual_analyzer.py tests/test_openclaw_cli_plan_gateway.py tests/test_openclaw_runtime_bridge.py
git commit -m "feat: attribute visual analyzer latency"
```

## Task 6: Add End-To-End Trace Coverage

**Files:**
- Modify: `tests/test_evaluation_harness_openclaw_runtime.py`
- Modify: `tests/test_harness_logger.py`
- Possibly modify: `src/evaluation_harness.py`
- Possibly modify: `src/harness/logging/harness_logger.py`

**Step 1: Write failing trace contract test**

Add a test that produces one runtime step with visual observations, memory write skip/write, recall usage, and planner metadata.

Assert trace includes:

```python
record["visual_analysis"]["ran"] is True
record["visual_analysis"]["vlm_latency_ms"] >= 0
record["memory_writes"][0]["write_gate"]["novelty_score"] >= 0
record["recall_usage"][0]["planner_intent_before_recall"] != ""
record["recall_usage"][0]["planner_intent_after_recall"] != ""
record["recall_usage"][0]["action_changed_after_recall"] in {True, False}
record["recall_usage"][0]["selected_namespace"].startswith("episode:")
```

Also assert no large raw image bytes and no oracle fields enter trace:

```python
assert "image_bytes" not in json.dumps(record)
assert record["oracle_guard_passed"] is True
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_evaluation_harness_openclaw_runtime.py tests/test_harness_logger.py -q
```

Expected: FAIL until previous task metadata is wired into the logger path.

**Step 3: Implement trace passthrough if needed**

If `HarnessLogger.log_step()` already writes `runtime` fields unchanged, only update runtime metadata generation. Otherwise, add explicit compact fields:

```python
for key in ("visual_analysis", "memory_writes", "recall_usage"):
    if key in runtime:
        record[key] = runtime[key]
```

Keep caption truncation if configured:

```python
record["visual_analysis"]["caption"] = caption[: max_caption_chars]
```

**Step 4: Run focused trace tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_evaluation_harness_openclaw_runtime.py tests/test_harness_logger.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/evaluation_harness.py src/harness/logging/harness_logger.py tests/test_evaluation_harness_openclaw_runtime.py tests/test_harness_logger.py
git commit -m "test: cover visual memory trace contract"
```

## Task 7: Update Ablation Summary Metrics

**Files:**
- Modify: `scripts/summarize_openclaw_vln_ablation.py`
- Test: `tests/test_openclaw_ablation_summary.py`

**Step 1: Write failing summary test**

Add a sample trace with:

- `visual_analysis.vlm_latency_ms`
- `memory_writes[].write_gate.novelty_score`
- `memory_writes[].write_gate.duplicate_of_memory_id`
- `recall_usage[].planner_intent_before_recall`
- `recall_usage[].planner_intent_after_recall`
- `recall_usage[].action_changed_after_recall`

Assert summary reports:

```python
assert summary["avg_vlm_latency_ms"] == 120.0
assert summary["avg_write_novelty_score"] == 0.71
assert summary["duplicate_skip_count"] == 1
assert summary["planner_intent_changed_after_recall_events"] == 1
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_ablation_summary.py -q
```

Expected: FAIL for new fields.

**Step 3: Implement summary fields**

Add accumulation for:

```python
vlm_latencies = []
write_novelty_scores = []
planner_intent_changed_after_recall_events = 0
```

Update per-record loop:

```python
visual = record.get("visual_analysis") or {}
if visual.get("vlm_latency_ms") is not None:
    vlm_latencies.append(float(visual["vlm_latency_ms"]))
```

For recall:

```python
if event.get("planner_intent_before_recall") != event.get("planner_intent_after_recall"):
    planner_intent_changed_after_recall_events += 1
```

**Step 4: Run focused summary tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_ablation_summary.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/summarize_openclaw_vln_ablation.py tests/test_openclaw_ablation_summary.py
git commit -m "feat: summarize visual memory causality metrics"
```

## Task 8: Final Verification

**Files:**
- No code changes expected.

**Step 1: Run focused visual-memory plan test suite**

Run:

```bash
PYTHONPATH=.:src pytest \
  tests/test_openclaw_visual_analyzer.py \
  tests/test_openclaw_cli_plan_gateway.py \
  tests/test_memory_skills.py \
  tests/test_memory_manager.py \
  tests/test_spatial_memory_client.py \
  tests/test_spatial_memory_protocol.py \
  tests/test_visual_memory_curator.py \
  tests/test_openclaw_runtime_bridge.py \
  tests/test_harness_logger.py \
  tests/test_evaluation_harness_openclaw_runtime.py \
  tests/test_evaluation_scripts.py \
  tests/test_phase4_docs.py \
  tests/test_openclaw_ablation_runner.py \
  tests/test_openclaw_ablation_summary.py \
  -q
```

Expected: PASS.

**Step 2: Run all owned tests**

Run:

```bash
PYTHONPATH=.:src pytest tests -q
```

Expected: PASS or only known sandbox socket failure in `tests/test_openclaw_gateway_server.py`.

If socket failure appears in sandbox, rerun just that file outside sandbox:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_gateway_server.py -q
```

Expected: PASS.

**Step 3: Run static compile check**

Run:

```bash
PYTHONPATH=src python -m py_compile \
  src/harness/memory/memory_manager.py \
  src/harness/memory/spatial_memory_client.py \
  src/harness/skills/visual_memory_curator.py \
  src/harness/openclaw/visual_analyzer.py \
  src/harness/openclaw/openclaw_cli_plan_gateway.py \
  src/harness/openclaw/runtime.py
```

Expected: exit 0.

**Step 4: Optional real gateway smoke**

Only run when OpenClaw and Qwen credentials are configured:

```bash
OPENCLAW_VISUAL_MODE=describe \
OPENCLAW_VISUAL_MODEL=qwen/qwen3.5-vl \
HOST=127.0.0.1 PORT=8011 ./scripts/start_openclaw_cli_plan_gateway.sh
```

Then:

```bash
PYTHONPATH=.:src python scripts/check_openclaw_visual_plan_gateway.py \
  --gateway_url http://127.0.0.1:8011 \
  --model qwen/qwen3.5-vl \
  --image_path /path/to/sample.png \
  --timeout 90
```

Expected: visual observations are non-empty and trace metadata includes VLM latency.

**Step 5: Final commit**

```bash
git status --short
git add docs/plans/2026-05-19-openclaw-visual-memory-completion-plan.md
git commit -m "docs: plan visual memory completion work"
```

## Acceptance Criteria

- Scope and namespace filters are included in semantic query payloads before backend retrieval.
- Defensive post-filtering remains in place.
- Recall hits are reranked by confidence, scope priority, recency, landmark overlap, and object/spatial cue overlap.
- Write gating emits `novelty_score` and `duplicate_of_memory_id`.
- Duplicate recent visual memories are skipped before storage.
- Recall usage logs compare a real before-recall decision with an after-recall planner pass.
- `action_changed_after_recall`, `replan_created_after_recall`, and planner intent changes are derived from actual before/after values.
- VLM analyzer latency is measured at `OpenClawVisualAnalyzer` and propagated separately from OpenClaw agent/runtime/tool latency.
- Ablation summary reports VLM latency, novelty score, duplicate skips, action changes, and planner-intent changes.
- Focused plan test suite passes.

