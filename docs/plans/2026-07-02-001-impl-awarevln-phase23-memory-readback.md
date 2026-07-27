---
artifact_contract: execution-plan/v1
created: 2026-07-02
source_plan: docs/plans/2026-07-01-001-feat-awarevln-phase23-memory-readback-plan.md
target_repo: AwareVLN
source_reference_repo: ClawNav
execution: code
---

# AwareVLN Phase 2/3 Memory Readback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port ClawNav Phase 2/3 keyframe memory, Qwen visual readback, and strict next-primitive action override into AwareVLN while preserving AwareVLN as the action-producing backbone.

**Architecture:** Keep the implementation AwareVLN-native under `evaluation/vlnce_baselines/aware_memory/`. The trainer remains the owner of model generation, macro-action parsing, and `envs.step(...)`; the new controller runs only at the primitive pre-step boundary and only changes the immediate action when the structured visual gate passes. Evidence is written as current-frame/keyframe artifacts, chunk-scoped JSONL traces, chunk-scoped summaries, and a merged run summary.

**Tech Stack:** Python 3.10, pytest, Habitat/AwareVLN eval loop, PIL image artifacts, JSON/JSONL trace files, bash launchers, mock Qwen backend for tests, optional real Qwen-compatible image backend for runtime.

---

## Source Contract

This implementation follows the approved design at `docs/plans/2026-07-01-001-feat-awarevln-phase23-memory-readback-plan.md`.

The implementation must preserve these contracts:

- AwareVLN remains the candidate-action producer; no JanusVLN/OpenClaw planner is imported.
- `AWAREVLN_MEMORY_MODE=off` preserves current eval behavior.
- Exact fixed-subset filtering uses `scene_id:episode_id` before dataset chunking.
- `reasoning_only` model outputs are traced as non-step rows with `final_action=null` and do not promote keyframes, call Qwen, or override actions.
- `model_candidate`, `queued_macro`, and `forced_stop` primitive decisions pass through the enabled memory boundary before `envs.step(...)`.
- A completed Qwen readback is not proof by itself; traces and summaries must separately prove correct memory image selection and executed-action changes.
- Chunked eval writes chunk-scoped trace/summary files and merges them without multiple workers writing one JSONL.

Run all commands below from the AwareVLN repo root.

```bash
cd /ssd/dingmuhe/Embodied-task/Navigation_Claw/AwareVLN
```

Use this real R2R smoke key for fixed-subset tests unless a curated manifest replaces it:

```bash
export AWAREVLN_EPISODE_KEYS="zsNo4HB9uLZ:1"
```

## Review Resolution Contract

The following doc-review fixes are now part of the execution contract:

- Test files must be created before any task references them in a verification command. `test_aware_memory_controller.py` starts in Task 6; Task 5 uses `test_aware_memory_keyframe_gate.py` and `test_aware_memory_memory_store.py` only.
- Interface sketches in this plan are planning targets, not copy-paste implementations. Committed tests and source files must not leave `...`, `pass`, `NotImplementedError`, or empty placeholder assertions behind.
- Real Qwen backend traces and logs must not include API key values, request headers, raw base64 image payloads, or full provider request bodies. Runtime evidence should keep local image paths, deterministic attached image ids, normalized response fields, and failure types only.
- The runtime oracle manifest may be empty during early unit-test work, but Task 10 cannot pass until at least one curated fixed-case scenario proves an expected memory id or keyframe step window.
- Smoke runs must set deterministic `RUN_NAME` values, and artifact audit must use the matching explicit `RUN_RESULTS_DIR` instead of relying on shell state from the launcher process.

---

## File Responsibility Map

Create:

- `evaluation/vlnce_baselines/aware_memory/__init__.py`: package boundary and public exports.
- `evaluation/vlnce_baselines/aware_memory/config.py`: environment parsing, modes, Qwen backend contract, defaults, validation.
- `evaluation/vlnce_baselines/aware_memory/actions.py`: canonical action names, Habitat action IDs, action text normalization, macro expansion helpers.
- `evaluation/vlnce_baselines/aware_memory/episode_filter.py`: exact `scene_id:episode_id` parsing, normalization, ordering, missing-key validation.
- `evaluation/vlnce_baselines/aware_memory/artifacts.py`: current-frame and keyframe path construction, image saving, safe path components.
- `evaluation/vlnce_baselines/aware_memory/keyframe_gate.py`: event-gated keyframe eligibility copied by behavior from ClawNav, not by package import.
- `evaluation/vlnce_baselines/aware_memory/memory_store.py`: per-episode ledger of `retrieval_text`, `visual_summary`, `summary_source`, and `image_path`.
- `evaluation/vlnce_baselines/aware_memory/trace_logger.py`: chunk-scoped JSONL writer and summary counters.
- `evaluation/vlnce_baselines/aware_memory/qwen_client.py`: mock and real backend boundary, timeout/retry/config failure handling.
- `evaluation/vlnce_baselines/aware_memory/visual_readback.py`: request builder, attached image id validation, normalized structured response.
- `evaluation/vlnce_baselines/aware_memory/memory_oracle.py`: fixed-case manifest schema and hit/miss evaluation.
- `evaluation/vlnce_baselines/aware_memory/controller.py`: primitive decision controller, keyframe/readback/override orchestration.
- `evaluation/vlnce_baselines/aware_memory/manifests/r2r_phase23_fixed_cases.json`: runtime oracle manifest.
- `evaluation/scripts/eval/r2r_memory_phase23.sh`: isolated memory launcher.
- `evaluation/scripts/summarize_aware_memory_phase23.py`: chunk trace/summary merger.
- `evaluation/tests/fixtures/aware_memory_oracle_manifest.json`: deterministic oracle fixture.
- `evaluation/tests/test_aware_memory_*.py`: unit and launcher tests listed per task below, including a dedicated memory-store test.

Modify:

- `evaluation/habitat_extensions/task.py`: apply exact-key filtering before `get_chunk(...)` for R2R; fail fast or explicitly implement RxR parity.
- `evaluation/vlnce_baselines/awarevln_trainer.py`: load config, create controller only when enabled, route primitive pre-step decisions through controller, trace `reasoning_only`, clear stale queue on accepted override.

Do not modify:

- `evaluation/scripts/eval/r2r.sh`: keep it as the clean baseline launcher.
- Training code and model weights.

---

## Task 0: Preflight Baseline and Work Isolation

**Files:**
- Read: `evaluation/vlnce_baselines/awarevln_trainer.py`
- Read: `evaluation/habitat_extensions/task.py`
- Read: `evaluation/scripts/eval/r2r.sh`
- Read: `evaluation/tests/`

- [ ] **Step 0.1: Confirm the working tree and baseline tests**

```bash
git status --short
PYTHONPATH=evaluation pytest evaluation/tests -q
bash -n evaluation/scripts/eval/r2r.sh
python -m py_compile evaluation/vlnce_baselines/awarevln_trainer.py evaluation/habitat_extensions/task.py
```

Expected:

- Existing tests pass before feature work starts.
- `r2r.sh` parses cleanly.
- The trainer and dataset loader compile.

- [ ] **Step 0.2: Record current eval seam**

Confirm these live seams before editing:

- `awarevln_trainer.py` has the forced stop path before queue handling.
- `awarevln_trainer.py` executes queued macro primitives through `envs.step([queue_actions[0]])`.
- `awarevln_trainer.py` continues without `envs.step(...)` for reasoning outputs.
- `task.py` currently chunks R2R/RxR episodes before `EPISODES_ALLOWED`.

Acceptance:

- No implementation starts until the observed seams still match the source plan. If a seam drifted, update this implementation plan before coding.

---

## Task 1: Config and Action Normalization

**Files:**
- Create: `evaluation/vlnce_baselines/aware_memory/__init__.py`
- Create: `evaluation/vlnce_baselines/aware_memory/config.py`
- Create: `evaluation/vlnce_baselines/aware_memory/actions.py`
- Create: `evaluation/tests/test_aware_memory_config.py`
- Create: `evaluation/tests/test_aware_memory_actions.py`

- [ ] **Step 1.1: Write config tests first**

Cover these cases in `evaluation/tests/test_aware_memory_config.py`:

```python
def test_default_config_is_off_and_does_not_require_qwen_credentials(monkeypatch): ...
def test_invalid_memory_mode_fails_fast(monkeypatch): ...
def test_keyframe_only_does_not_require_qwen_credentials(monkeypatch): ...
def test_real_readback_backend_requires_named_api_key_env(monkeypatch): ...
def test_mock_backend_is_valid_without_network_credentials(monkeypatch): ...
def test_override_budget_accepts_adaptive_or_positive_integer(monkeypatch): ...
def test_oracle_manifest_default_path_is_repo_relative(monkeypatch): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_config.py -q
```

Expected now: fails because `aware_memory.config` does not exist.

- [ ] **Step 1.2: Implement `config.py`**

Public interface:

```python
MEMORY_MODES = {"off", "keyframe_only", "image_read_log_only", "image_read_action_override"}
READBACK_BACKENDS = {"mock", "dashscope", "openai"}

@dataclass(frozen=True)
class AwareMemoryConfig:
    mode: str
    episode_keys: tuple[str, ...]
    trigger_policy: str
    top_k: int
    max_action_overrides_per_episode: str | int
    readback_backend: str
    readback_model: str
    readback_api_key_env: str
    readback_base_url: str
    readback_timeout_ms: int
    readback_retry_count: int
    oracle_manifest_path: str
    keyframe_min_gap_steps: int
    keyframe_episode_cap: int
    coverage_gap_steps: int

def load_memory_config(environ: Mapping[str, str] | None = None) -> AwareMemoryConfig: ...
def memory_enabled(config: AwareMemoryConfig) -> bool: ...
def readback_enabled(config: AwareMemoryConfig) -> bool: ...
```

Required defaults:

- `AWAREVLN_MEMORY_MODE=off`
- `AWAREVLN_READBACK_BACKEND=mock`
- `AWAREVLN_READBACK_TIMEOUT_MS=30000`
- `AWAREVLN_READBACK_RETRY_COUNT=1`
- `AWAREVLN_READBACK_TOP_K=3`
- `AWAREVLN_MEMORY_ORACLE_MANIFEST=evaluation/vlnce_baselines/aware_memory/manifests/r2r_phase23_fixed_cases.json`

- [ ] **Step 1.3: Write action tests first**

Cover these cases in `evaluation/tests/test_aware_memory_actions.py`:

```python
def test_normalize_action_accepts_canonical_actions(): ...
def test_normalize_action_maps_habitat_ids_to_canonical_names(): ...
def test_normalize_action_rejects_unknown_labels(): ...
def test_parse_model_action_text_uses_forward_fallback_like_current_awarevln(): ...
def test_expand_forward_macro_to_primitive_queue(): ...
def test_expand_turn_macro_to_primitive_queue(): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_actions.py -q
```

Expected now: fails until `actions.py` exists.

- [ ] **Step 1.4: Implement `actions.py`**

Public interface:

```python
ACTION_ID_STOP = 0
ACTION_ID_FORWARD = 1
ACTION_ID_LEFT = 2
ACTION_ID_RIGHT = 3

CANONICAL_TO_ACTION_ID = {
    "STOP": ACTION_ID_STOP,
    "MOVE_FORWARD": ACTION_ID_FORWARD,
    "TURN_LEFT": ACTION_ID_LEFT,
    "TURN_RIGHT": ACTION_ID_RIGHT,
}

def normalize_action(value: object) -> str: ...
def action_id_for_name(action_name: str) -> int | None: ...
def name_for_action_id(action_id: int) -> str: ...
def parse_awarevln_action_text(raw_output_content: str) -> str: ...
def expand_macro_to_next_and_queue(action_name: str, raw_output_content: str) -> tuple[int, list[int]]: ...
```

Behavior must mirror current AwareVLN:

- unrecognized model text maps to `MOVE_FORWARD`;
- `move forward N cm` rounds to 25/50/75 cm and queues `(N / 25) - 1` additional forward primitives;
- `turn left N degree` and `turn right N degree` round to 15/30/45 degrees and queue `(N / 15) - 1` additional turn primitives.

- [ ] **Step 1.5: Verify Task 1**

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_config.py \
  evaluation/tests/test_aware_memory_actions.py -q
python -m py_compile \
  evaluation/vlnce_baselines/aware_memory/config.py \
  evaluation/vlnce_baselines/aware_memory/actions.py
```

Acceptance:

- Config can be loaded without Habitat/model imports.
- `off` and `keyframe_only` never require Qwen credentials.
- Action parsing preserves current AwareVLN fallback and macro semantics.

---

## Task 2: Off-Mode Baseline Characterization and Trainer Seam

**Files:**
- Modify: `evaluation/vlnce_baselines/awarevln_trainer.py`
- Create: `evaluation/tests/test_aware_memory_baseline_off.py`

- [ ] **Step 2.1: Write off-mode characterization tests**

Cover these cases:

```python
def test_off_mode_does_not_create_controller_or_trace_writer(monkeypatch): ...
def test_off_mode_preserves_direct_stop_forward_left_right(monkeypatch): ...
def test_off_mode_preserves_forward_macro_queue(monkeypatch): ...
def test_off_mode_preserves_turn_macro_queue(monkeypatch): ...
def test_off_mode_preserves_reasoning_continue_without_step(monkeypatch): ...
def test_off_mode_preserves_forced_stop_bypass(monkeypatch): ...
```

Use small fakes for `envs.step`, `queue_actions`, model output strings, and parsed action text. Do not instantiate the model or Habitat simulator.

Run:

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_baseline_off.py -q
```

Expected now: fails until the trainer exposes a testable seam or delegates parsing to `actions.py`.

- [ ] **Step 2.2: Add a minimal trainer seam without changing behavior**

Modify `awarevln_trainer.py` so parsing and queue expansion use `aware_memory.actions`, while the `off` branch executes the same `envs.step(...)` calls as before.

Implementation rule:

- No controller object is created when `AWAREVLN_MEMORY_MODE=off`.
- Existing direct action, macro queue, forced stop, video, and result-writing behavior remain on the original path.
- The first trainer edit should be a behavior-preserving refactor only.

- [ ] **Step 2.3: Verify Task 2**

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_actions.py \
  evaluation/tests/test_aware_memory_baseline_off.py -q
python -m py_compile evaluation/vlnce_baselines/awarevln_trainer.py
```

Acceptance:

- Off-mode tests prove no memory components are instantiated.
- The existing AwareVLN eval loop still compiles after using shared action helpers.

---

## Task 3: Exact Episode-Key Filtering Before Chunking

**Files:**
- Create: `evaluation/vlnce_baselines/aware_memory/episode_filter.py`
- Modify: `evaluation/habitat_extensions/task.py`
- Create: `evaluation/tests/test_aware_memory_episode_filter.py`

- [ ] **Step 3.1: Write filter tests first**

Cover these cases:

```python
def test_empty_key_list_keeps_episode_order(): ...
def test_valid_keys_filter_and_preserve_requested_order(): ...
def test_missing_keys_raise_with_missing_names(): ...
def test_scene_path_normalization_extracts_mp3d_scene_id(): ...
def test_duplicate_episode_ids_in_different_scenes_are_distinct(): ...
def test_r2r_loader_applies_exact_keys_before_get_chunk(monkeypatch): ...
def test_multi_chunk_partitions_filtered_subset(monkeypatch): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_episode_filter.py -q
```

Expected now: fails until the helper and loader hook exist.

- [ ] **Step 3.2: Implement `episode_filter.py`**

Public interface:

```python
def parse_episode_keys(raw: str) -> tuple[str, ...]: ...
def canonical_scene_id(scene_id: str) -> str: ...
def canonical_episode_key(episode: Mapping[str, object] | object) -> str: ...
def filter_episodes_by_exact_keys(episodes: list[object], keys: Sequence[str]) -> list[object]: ...
```

Rules:

- Key format is exactly `scene_id:episode_id`.
- Matching uses normalized scene id plus string episode id.
- Output order follows the requested key order.
- Missing keys fail fast with every missing key in the error message.

- [ ] **Step 3.3: Wire R2R loader before `get_chunk(...)`**

Modify `evaluation/habitat_extensions/task.py` so R2R filtering happens in `from_json(...)` after JSON deserialization and before `get_chunk(deserialized["episodes"], ...)`.

Required behavior:

- The helper reads `AWAREVLN_EPISODE_KEYS` through `load_memory_config()` or a narrow env parser.
- If no exact keys are set, loader behavior is unchanged.
- Filtering happens before scene path expansion and before chunk partitioning.
- Existing `EPISODES_ALLOWED` behavior remains available for non-memory baseline use.

- [ ] **Step 3.4: Decide RxR behavior explicitly**

First pass can fail fast for RxR exact keys. Implement one of these two behaviors:

- support RxR with the same pre-chunk helper; or
- raise a clear unsupported-mode error when `AWAREVLN_EPISODE_KEYS` is set during RxR loading.

The implementation must not silently fall back to post-chunk `EPISODES_ALLOWED`.

- [ ] **Step 3.5: Verify Task 3**

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_episode_filter.py -q
python -m py_compile evaluation/habitat_extensions/task.py
```

Acceptance:

- The test proves filtering happens before `get_chunk(...)`.
- Duplicate `episode_id` values across scenes cannot collide.
- Exact fixed subset order is preserved.

---

## Task 4: Trace Logger Foundation

**Files:**
- Create: `evaluation/vlnce_baselines/aware_memory/trace_logger.py`
- Create: `evaluation/tests/test_aware_memory_trace_logger.py`

- [ ] **Step 4.1: Write trace logger tests first**

Cover these cases:

```python
def test_trace_path_is_chunk_scoped(): ...
def test_write_primitive_decision_row_jsonl(tmp_path): ...
def test_write_reasoning_only_non_step_row(tmp_path): ...
def test_summary_counts_readbacks_separately_from_action_changes(tmp_path): ...
def test_summary_counts_reasoning_rows_separately_from_primitive_rows(tmp_path): ...
def test_summary_counts_oracle_hits_misses_and_no_scenario(tmp_path): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_trace_logger.py -q
```

Expected now: fails until `trace_logger.py` exists.

- [ ] **Step 4.2: Implement chunk-scoped trace writer**

Public interface:

```python
def trace_suffix(split: str, total_chunks: int, chunk_idx: int) -> str: ...
def trace_path(results_dir: str, split: str, total_chunks: int, chunk_idx: int) -> str: ...
def summary_path(results_dir: str, split: str, total_chunks: int, chunk_idx: int) -> str: ...

class AwareMemoryTraceLogger:
    def write_row(self, row: Mapping[str, object]) -> None: ...
    def write_reasoning_only(self, *, scene_id: str, episode_id: str, step_id: int, raw_output: str, extracted_reasoning: str, reason_stuck_num: int, last_reason_step: int) -> None: ...
    def write_summary(self) -> dict[str, object]: ...
```

Required row fields:

- `decision_source`
- `scene_id`
- `episode_id`
- `step_id`
- `candidate_action`
- `final_action`
- `queue_before`
- `queue_after`
- `keyframe_gate`
- `readback`
- `override`
- `failure_reason`

For `decision_source=reasoning_only`, set `final_action` to `null` and include `raw_output`, `extracted_reasoning`, `reason_stuck_num`, and `last_reason_step`.

- [ ] **Step 4.3: Verify Task 4**

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_trace_logger.py -q
python -m py_compile evaluation/vlnce_baselines/aware_memory/trace_logger.py
```

Acceptance:

- Trace filenames include split, total chunks, and chunk index.
- Reasoning rows are non-step rows and are counted separately.

---

## Task 5: Current Frames, Keyframe Gate, and Episode-Local Memory Store

**Files:**
- Create: `evaluation/vlnce_baselines/aware_memory/artifacts.py`
- Create: `evaluation/vlnce_baselines/aware_memory/keyframe_gate.py`
- Create: `evaluation/vlnce_baselines/aware_memory/memory_store.py`
- Create: `evaluation/tests/test_aware_memory_keyframe_gate.py`
- Create: `evaluation/tests/test_aware_memory_memory_store.py`

- [ ] **Step 5.1: Write artifact and gate tests first**

Cover these cases:

```python
def test_current_frame_path_contains_scene_episode_and_zero_padded_step(tmp_path): ...
def test_artifact_writer_creates_parent_directories(tmp_path): ...
def test_missing_rgb_returns_empty_path_and_skip_reason(tmp_path): ...
def test_keyframe_gate_saves_step_zero_initial_context(): ...
def test_keyframe_gate_saves_turns_and_stop_as_decision_points(): ...
def test_keyframe_gate_blocks_duplicate_turn_segment(): ...
def test_keyframe_gate_enforces_cooldown_and_episode_cap(): ...
def test_keyframe_gate_saves_coverage_gap_after_forward_segment(): ...
def test_keyframe_gate_resets_on_new_episode(): ...
```

Cover these memory-store cases in `evaluation/tests/test_aware_memory_memory_store.py`:

```python
def test_memory_record_requires_metadata_source_when_visual_summary_is_none(): ...
def test_memory_record_preserves_image_path_scene_episode_step_and_save_reason(): ...
def test_episode_memory_store_resets_on_new_episode(): ...
def test_top_k_returns_records_with_retrieval_text_summary_source_and_image_path(): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_keyframe_gate.py \
  evaluation/tests/test_aware_memory_memory_store.py -q
```

Expected now: fails until artifact and gate modules exist.

- [ ] **Step 5.2: Implement artifact writer**

Public interface:

```python
def safe_component(value: object) -> str: ...
def current_frame_path(results_dir: str, scene_id: str, episode_id: str, step_id: int) -> str: ...
def keyframe_path(results_dir: str, scene_id: str, episode_id: str, step_id: int, save_reason: str) -> str: ...
def save_rgb_image(rgb: object, output_path: str) -> dict[str, object]: ...
```

Rules:

- Save paths are under the run results directory.
- Current frames and keyframes are separated.
- Missing image input returns a structured skip result; it does not crash enabled eval.

- [ ] **Step 5.3: Implement keyframe gate**

Behavior mirrors ClawNav `EventGatedKeyframeGate`:

- step `0` saves `initial_context`;
- `STOP`, `TURN_LEFT`, and `TURN_RIGHT` save `candidate_decision_point`;
- repeated same-turn segment saves once;
- cooldown and per-episode cap block saves with explicit reasons;
- forward-only coverage gap saves `coverage_gap`;
- new scene/episode resets state.

- [ ] **Step 5.4: Implement memory store**

Public interface:

```python
@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    retrieval_text: str
    visual_summary: str | None
    summary_source: str
    image_path: str
    scene_id: str
    episode_id: str
    step_id: int
    save_reason: str

class EpisodeMemoryStore:
    def reset(self, scene_id: str, episode_id: str) -> None: ...
    def add(self, record: MemoryRecord) -> MemoryRecord: ...
    def top_k(self, query_text: str, k: int) -> list[MemoryRecord]: ...
```

Rules:

- `visual_summary=None` requires `summary_source="metadata_only"`.
- Image-derived summary sources must be explicit, for example `aware_reasoning_observation` or `qwen_caption`.
- `top_k` can start with deterministic metadata/text scoring; Qwen readback must still receive image paths.

- [ ] **Step 5.5: Verify Task 5**

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_keyframe_gate.py \
  evaluation/tests/test_aware_memory_memory_store.py -q
python -m py_compile \
  evaluation/vlnce_baselines/aware_memory/artifacts.py \
  evaluation/vlnce_baselines/aware_memory/keyframe_gate.py \
  evaluation/vlnce_baselines/aware_memory/memory_store.py
```

Acceptance:

- Keyframe-only behavior can save current frames and promoted keyframes without Qwen.
- Ledger records always carry text, source label, and image path together.

---

## Task 6: Controller Boundary in the Trainer

**Files:**
- Create: `evaluation/vlnce_baselines/aware_memory/controller.py`
- Modify: `evaluation/vlnce_baselines/awarevln_trainer.py`
- Create: `evaluation/tests/test_aware_memory_controller.py`
- Extend: `evaluation/tests/test_aware_memory_baseline_off.py`

- [ ] **Step 6.1: Write controller boundary tests first**

Cover these cases:

```python
def test_keyframe_only_model_candidate_saves_and_keeps_action(tmp_path): ...
def test_keyframe_only_queued_macro_saves_and_keeps_action(tmp_path): ...
def test_keyframe_only_forced_stop_saves_and_keeps_action(tmp_path): ...
def test_reasoning_only_trace_does_not_promote_keyframe_or_readback(tmp_path): ...
def test_controller_returns_queue_after_without_mutating_input_queue(tmp_path): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_controller.py -q
```

Expected now: fails until `controller.py` exists.

- [ ] **Step 6.2: Implement controller decision types**

Public interface:

```python
@dataclass(frozen=True)
class PrimitiveDecisionContext:
    decision_source: str
    scene_id: str
    episode_id: str
    step_id: int
    instruction: str
    current_rgb: object
    candidate_action: str
    candidate_action_id: int
    queue_before: tuple[int, ...]
    raw_model_output: str
    reasoning_context: str

@dataclass(frozen=True)
class PrimitiveDecisionResult:
    final_action: str
    final_action_id: int
    queue_after: tuple[int, ...]
    trace_row: dict[str, object]
    executed_action_changed_after_visual_read: bool

class AwareMemoryController:
    def before_step(self, context: PrimitiveDecisionContext) -> PrimitiveDecisionResult: ...
    def record_reasoning_only(
        self,
        *,
        scene_id: str,
        episode_id: str,
        step_id: int,
        raw_output: str,
        extracted_reasoning: str,
        reason_stuck_num: int,
        last_reason_step: int,
    ) -> None: ...
    def reset_episode(self, scene_id: str, episode_id: str) -> None: ...
```

In `keyframe_only`, `before_step` saves/traces but always returns the candidate action and original queue.

- [ ] **Step 6.3: Wire trainer enabled-mode boundary**

Modify `awarevln_trainer.py`:

- create controller only when `memory_enabled(config)` is true;
- route `model_candidate` before the immediate `envs.step(...)`;
- route `queued_macro` before stepping queued primitive action;
- route `forced_stop` before stepping STOP in enabled modes;
- call `record_reasoning_only(...)` before the existing `continue` for reasoning outputs in enabled modes;
- keep off mode on the baseline path.

Important trainer invariant:

- When an enabled controller changes `final_action_id`, assign `queue_actions` from `PrimitiveDecisionResult.queue_after`.
- When the controller does not change the action, preserve current queue behavior.

- [ ] **Step 6.4: Verify Task 6**

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_baseline_off.py \
  evaluation/tests/test_aware_memory_controller.py -q
python -m py_compile evaluation/vlnce_baselines/awarevln_trainer.py
```

Acceptance:

- Enabled `keyframe_only` covers all primitive decision sources.
- Off mode still bypasses controller construction.
- Reasoning-only rows are traceable without stepping the environment.

---

## Task 7: Visual Readback, Qwen Client, and Oracle Manifest

**Files:**
- Create: `evaluation/vlnce_baselines/aware_memory/qwen_client.py`
- Create: `evaluation/vlnce_baselines/aware_memory/visual_readback.py`
- Create: `evaluation/vlnce_baselines/aware_memory/memory_oracle.py`
- Create: `evaluation/vlnce_baselines/aware_memory/manifests/r2r_phase23_fixed_cases.json`
- Create: `evaluation/tests/fixtures/aware_memory_oracle_manifest.json`
- Create: `evaluation/tests/test_aware_memory_qwen_client.py`
- Create: `evaluation/tests/test_aware_memory_visual_readback.py`
- Create: `evaluation/tests/test_aware_memory_oracle.py`
- Extend: `evaluation/tests/test_aware_memory_controller.py`

- [ ] **Step 7.1: Write Qwen and readback tests first**

Cover these cases:

```python
def test_mock_backend_returns_fixture_response(tmp_path): ...
def test_real_backend_missing_credentials_fails_closed(monkeypatch): ...
def test_missing_current_image_returns_failed_readback(tmp_path): ...
def test_missing_memory_image_returns_failed_readback(tmp_path): ...
def test_parse_failure_returns_conservative_failed_payload(tmp_path): ...
def test_attached_image_ids_are_current_and_memory_ids(tmp_path): ...
def test_reported_memory_ids_must_be_attached(tmp_path): ...
def test_log_only_mode_never_changes_action_even_when_override_requested(tmp_path): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_qwen_client.py \
  evaluation/tests/test_aware_memory_visual_readback.py -q
```

Expected now: fails until readback modules exist.

- [ ] **Step 7.2: Implement `qwen_client.py`**

Public interface:

```python
class QwenReadbackClient:
    def read(self, request: Mapping[str, object]) -> dict[str, object]: ...

class MockQwenReadbackClient(QwenReadbackClient): ...
class ConfiguredQwenReadbackClient(QwenReadbackClient): ...

def build_qwen_client(config: AwareMemoryConfig) -> QwenReadbackClient | None: ...
```

Rules:

- `off` and `keyframe_only` return `None`.
- `mock` never requires credentials or network.
- real backends require `AWAREVLN_READBACK_API_KEY_ENV` and that env var must be present.
- local image paths are converted to provider-supported image payloads before dispatch.
- timeout, retry, parse failure, and network failure return conservative failed readback payloads.
- traces and logs must never include API key values, request headers, raw base64 image payloads, or full provider request bodies; record only local image paths, deterministic attached image ids, normalized response fields, and failure types.

- [ ] **Step 7.3: Implement `visual_readback.py`**

Public interface:

```python
def build_readback_request(
    *,
    current_image_path: str,
    memory_records: Sequence[MemoryRecord],
    candidate_action: str,
    trigger_rule: str,
    instruction: str,
    decision_source: str,
    step_id: int,
) -> dict[str, object]: ...
def validate_readback_images(request: Mapping[str, object]) -> str: ...
def normalize_visual_readback_response(response: object, request: Mapping[str, object]) -> dict[str, object]: ...
```

Required normalized fields include:

- `read_status`
- `attached_image_ids`
- `attached_memory_ids`
- `retrieved_image_paths`
- `actually_read_image_paths`
- `matched_memory_ids`
- `evidence_sources`
- `candidate_action_valid`
- `should_override`
- `recommended_action`
- `decision_scope`
- `action_confidence`
- `current_view_candidate_invalid`
- `current_view_recommended_action_supported`
- `readback_confidence`
- `retrieval_confidence`
- `verifier_confidence`

Reject or mark unsupported any Qwen-reported image id or memory id that was not attached.

- [ ] **Step 7.4: Write oracle tests first**

Cover these cases:

```python
def test_manifest_requires_schema_scenario_scene_episode_and_decision_source(): ...
def test_manifest_requires_query_step_or_window(): ...
def test_manifest_requires_expected_ids_or_expected_step_window(): ...
def test_oracle_hit_by_expected_memory_id(): ...
def test_oracle_hit_by_expected_step_window(): ...
def test_oracle_requires_both_checks_when_both_are_declared(): ...
def test_no_matching_scenario_is_reported_separately(): ...
```

- [ ] **Step 7.5: Implement `memory_oracle.py` and manifests**

Public interface:

```python
def load_oracle_manifest(path: str) -> list[dict[str, object]]: ...
def match_oracle_scenario(manifest: Sequence[Mapping[str, object]], *, scene_id: str, episode_id: str, step_id: int, decision_source: str) -> dict[str, object] | None: ...
def evaluate_oracle_hit(scenario: Mapping[str, object] | None, readback: Mapping[str, object], selected_records: Sequence[MemoryRecord]) -> dict[str, object]: ...
```

Fixture manifest must contain at least:

- one `expected_memory_ids` scenario;
- one `expected_keyframe_step_window` scenario;
- one scenario requiring both id and step-window checks.

The runtime manifest must be a versioned JSON array. It may remain empty during early unit-test implementation, but Task 10 cannot be marked complete until at least one curated runtime scenario exists for a fixed case where the expected memory id or keyframe step window is known.

- [ ] **Step 7.6: Wire log-only readback into controller**

In `image_read_log_only`:

- evaluate sparse trigger rules for risky STOP, turn decision point, and coverage-gap forward;
- retrieve Top-K memory records;
- attach current image plus keyframe images;
- call Qwen client;
- normalize response;
- evaluate oracle;
- write trace;
- never change `final_action_id`.

- [ ] **Step 7.7: Verify Task 7**

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_qwen_client.py \
  evaluation/tests/test_aware_memory_visual_readback.py \
  evaluation/tests/test_aware_memory_oracle.py \
  evaluation/tests/test_aware_memory_controller.py -q
python -m py_compile \
  evaluation/vlnce_baselines/aware_memory/qwen_client.py \
  evaluation/vlnce_baselines/aware_memory/visual_readback.py \
  evaluation/vlnce_baselines/aware_memory/memory_oracle.py
```

Acceptance:

- Unit tests prove Qwen receives image payloads derived from actual paths, not only summary text.
- Oracle hit/miss is independent of Qwen self-report.
- Log-only mode records readback evidence but never changes action.

---

## Task 8: Strict Phase 3 Override Gate and Queue Clearing

**Files:**
- Modify: `evaluation/vlnce_baselines/aware_memory/controller.py`
- Modify: `evaluation/vlnce_baselines/aware_memory/actions.py`
- Modify: `evaluation/vlnce_baselines/awarevln_trainer.py`
- Extend: `evaluation/tests/test_aware_memory_controller.py`
- Extend: `evaluation/tests/test_aware_memory_actions.py`

- [ ] **Step 8.1: Write override gate tests first**

Cover these cases:

```python
def test_override_accepts_high_confidence_invalid_candidate_with_current_view_support(tmp_path): ...
def test_low_verifier_confidence_blocks_override(tmp_path): ...
def test_low_action_confidence_blocks_override(tmp_path): ...
def test_missing_decision_scope_blocks_override(tmp_path): ...
def test_route_diagnosis_only_blocks_override(tmp_path): ...
def test_invalid_recommended_action_blocks_override(tmp_path): ...
def test_candidate_valid_blocks_override(tmp_path): ...
def test_missing_current_view_support_blocks_override(tmp_path): ...
def test_same_action_recommendation_records_no_executed_action_change(tmp_path): ...
def test_changed_action_clears_stale_macro_queue(tmp_path): ...
def test_fixed_override_budget_blocks_after_limit(tmp_path): ...
def test_adaptive_debounce_blocks_unstable_direction_changes(tmp_path): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest evaluation/tests/test_aware_memory_controller.py -q
```

Expected now: fails until override gate is implemented.

- [ ] **Step 8.2: Implement strict gate**

Accept override only when all conditions are true:

```text
mode == image_read_action_override
decision_source is supported by trigger policy
read_status == completed
verifier_confidence >= high_confidence
candidate_action_valid is false
should_override is true
decision_scope == immediate_next_action
action_confidence >= high_confidence
recommended_action is STOP, MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT
current_view_candidate_invalid is true
current_view_recommended_action_supported is true
budget/debounce allows this override
```

When accepted:

- set `final_action` and `final_action_id` from `recommended_action`;
- set `executed_action_changed_after_visual_read` based on final action vs original candidate action;
- clear stale `queue_actions` when final action changes;
- write `action_hint_override_accepted=true`.

When rejected:

- preserve candidate action and queue;
- write the first concrete failure reason, for example `low_verifier_confidence`, `candidate_action_valid`, `missing_current_view_support`, or `invalid_recommended_action`.

- [ ] **Step 8.3: Verify Task 8**

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_controller.py \
  evaluation/tests/test_aware_memory_actions.py -q
python -m py_compile \
  evaluation/vlnce_baselines/aware_memory/controller.py \
  evaluation/vlnce_baselines/awarevln_trainer.py
```

Acceptance:

- Override is impossible from free-text `audit_action_hint` alone.
- Accepted override changes only the next primitive action.
- Stale queued macro primitives do not execute after a changed action.

---

## Task 9: Launcher, Chunk-Safe Summary, and Result Artifacts

**Files:**
- Create: `evaluation/scripts/eval/r2r_memory_phase23.sh`
- Create: `evaluation/scripts/summarize_aware_memory_phase23.py`
- Extend: `evaluation/tests/test_r2r_launcher.py`
- Extend: `evaluation/tests/test_aware_memory_trace_logger.py`

- [ ] **Step 9.1: Write launcher and summary tests first**

Cover these cases:

```python
def test_memory_launcher_does_not_modify_baseline_r2r_script(): ...
def test_memory_launcher_defaults_to_single_chunk_when_episode_keys_are_set(): ...
def test_memory_launcher_prints_backend_model_key_env_and_oracle_path_without_secret_value(): ...
def test_memory_summary_merges_chunk_scoped_files(tmp_path): ...
def test_memory_summary_reports_missing_chunk_trace(tmp_path): ...
def test_memory_summary_reports_duplicate_chunk_trace(tmp_path): ...
```

Run:

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_r2r_launcher.py \
  evaluation/tests/test_aware_memory_trace_logger.py -q
```

Expected now: fails until launcher and summarizer exist.

- [ ] **Step 9.2: Implement `r2r_memory_phase23.sh`**

Launcher requirements:

- delegate to `python run.py` with the same baseline config as `evaluation/scripts/eval/r2r.sh`;
- preserve `MODEL_PATH`, `GPU_LIST`, `SAVE_VIDEO`, `SAVE_VIDEO_RATIO`, result naming, and chunk options;
- default `AWAREVLN_MEMORY_MODE=image_read_log_only` unless explicitly set;
- default `AWAREVLN_READBACK_BACKEND=mock` for smoke tests unless explicitly set;
- when `AWAREVLN_EPISODE_KEYS` is non-empty and `TOTAL_CHUNKS` is unset, set `TOTAL_CHUNKS=1`;
- print effective memory mode, episode keys, chunk config, backend, model, API key env name, timeout, retry count, trace suffix, summary suffix, and oracle manifest path;
- never print the secret value from the API key env var.

- [ ] **Step 9.3: Implement `summarize_aware_memory_phase23.py`**

Public behavior:

```bash
python evaluation/scripts/summarize_aware_memory_phase23.py \
  --results-dir results/awarevln_memory_phase23_example \
  --split val_unseen \
  --total-chunks 1
```

Summary must report:

- primitive row count;
- reasoning-only row count;
- readback completed/skipped/failed counts;
- action override accepted/rejected counts;
- `executed_action_changed_after_visual_read_count`;
- oracle eligible/hit/miss/no-scenario counts;
- missing chunk trace files;
- duplicate chunk trace files.

- [ ] **Step 9.4: Verify Task 9**

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_r2r_launcher.py \
  evaluation/tests/test_aware_memory_trace_logger.py -q
bash -n evaluation/scripts/eval/r2r_memory_phase23.sh
python -m py_compile evaluation/scripts/summarize_aware_memory_phase23.py
```

Acceptance:

- Original `evaluation/scripts/eval/r2r.sh` remains a clean baseline.
- Enabled memory runs write chunk-scoped traces/summaries and a merged summary.

---

## Task 10: End-to-End Smoke Runs and Evidence Audit

**Files:**
- Runtime output under the configured `RESULTS_DIR`
- No additional source files unless the smoke exposes a concrete bug

- [ ] **Step 10.1: Run unit suite for the feature**

```bash
PYTHONPATH=evaluation pytest \
  evaluation/tests/test_aware_memory_config.py \
  evaluation/tests/test_aware_memory_actions.py \
  evaluation/tests/test_aware_memory_baseline_off.py \
  evaluation/tests/test_aware_memory_episode_filter.py \
  evaluation/tests/test_aware_memory_keyframe_gate.py \
  evaluation/tests/test_aware_memory_memory_store.py \
  evaluation/tests/test_aware_memory_qwen_client.py \
  evaluation/tests/test_aware_memory_visual_readback.py \
  evaluation/tests/test_aware_memory_oracle.py \
  evaluation/tests/test_aware_memory_controller.py \
  evaluation/tests/test_aware_memory_trace_logger.py \
  evaluation/tests/test_r2r_launcher.py -q
```

Expected: all feature tests pass.

- [ ] **Step 10.2: Run syntax checks**

```bash
bash -n evaluation/scripts/eval/r2r_memory_phase23.sh
python -m py_compile \
  evaluation/vlnce_baselines/aware_memory/config.py \
  evaluation/vlnce_baselines/aware_memory/actions.py \
  evaluation/vlnce_baselines/aware_memory/episode_filter.py \
  evaluation/vlnce_baselines/aware_memory/artifacts.py \
  evaluation/vlnce_baselines/aware_memory/keyframe_gate.py \
  evaluation/vlnce_baselines/aware_memory/memory_store.py \
  evaluation/vlnce_baselines/aware_memory/trace_logger.py \
  evaluation/vlnce_baselines/aware_memory/qwen_client.py \
  evaluation/vlnce_baselines/aware_memory/visual_readback.py \
  evaluation/vlnce_baselines/aware_memory/memory_oracle.py \
  evaluation/vlnce_baselines/aware_memory/controller.py \
  evaluation/vlnce_baselines/awarevln_trainer.py \
  evaluation/habitat_extensions/task.py \
  evaluation/scripts/summarize_aware_memory_phase23.py
```

Expected: all files compile.

- [ ] **Step 10.3: Run off-mode fixed-key smoke**

```bash
MODEL_PATH=../weights/awarevln \
RESULTS_DIR=../results \
RUN_NAME=awarevln_memory_phase23_smoke_off \
TOTAL_CHUNKS=1 \
GPU_LIST=2 \
SAVE_VIDEO=False \
AWAREVLN_MEMORY_MODE=off \
AWAREVLN_EPISODE_KEYS="zsNo4HB9uLZ:1" \
bash evaluation/scripts/eval/r2r_memory_phase23.sh
```

Expected:

- run completes or reaches the same model/runtime constraints as the baseline;
- no Qwen/readback credentials are required;
- memory trace is absent or limited to explicit baseline characterization output.

- [ ] **Step 10.4: Run keyframe-only fixed-key smoke**

```bash
MODEL_PATH=../weights/awarevln \
RESULTS_DIR=../results \
RUN_NAME=awarevln_memory_phase23_smoke_keyframe_only \
TOTAL_CHUNKS=1 \
GPU_LIST=2 \
SAVE_VIDEO=False \
AWAREVLN_MEMORY_MODE=keyframe_only \
AWAREVLN_EPISODE_KEYS="zsNo4HB9uLZ:1" \
bash evaluation/scripts/eval/r2r_memory_phase23.sh
```

Expected artifacts:

- current-frame image paths;
- promoted keyframe image paths;
- ledger records with `retrieval_text`, `visual_summary`, `summary_source`, and `image_path`;
- no Qwen calls.

- [ ] **Step 10.5: Run mock log-only readback smoke**

```bash
MODEL_PATH=../weights/awarevln \
RESULTS_DIR=../results \
RUN_NAME=awarevln_memory_phase23_smoke_log_only \
TOTAL_CHUNKS=1 \
GPU_LIST=2 \
SAVE_VIDEO=False \
AWAREVLN_MEMORY_MODE=image_read_log_only \
AWAREVLN_READBACK_BACKEND=mock \
AWAREVLN_EPISODE_KEYS="zsNo4HB9uLZ:1" \
bash evaluation/scripts/eval/r2r_memory_phase23.sh
```

Expected trace proof:

- `read_status` is counted separately from final action changes;
- readback rows include `attached_image_ids`, `attached_memory_ids`, `retrieved_image_paths`, `actually_read_image_paths`, `matched_memory_ids`, and `evidence_sources`;
- final action equals candidate action for log-only rows.

- [ ] **Step 10.6: Run mock override smoke with a deterministic fixture**

Use a mock response that requests one high-confidence immediate action change and one rejected override.

```bash
MODEL_PATH=../weights/awarevln \
RESULTS_DIR=../results \
RUN_NAME=awarevln_memory_phase23_smoke_override \
TOTAL_CHUNKS=1 \
GPU_LIST=2 \
SAVE_VIDEO=False \
AWAREVLN_MEMORY_MODE=image_read_action_override \
AWAREVLN_READBACK_BACKEND=mock \
AWAREVLN_EPISODE_KEYS="zsNo4HB9uLZ:1" \
bash evaluation/scripts/eval/r2r_memory_phase23.sh
```

Expected trace proof:

- accepted rows set `action_hint_override_accepted=true`;
- changed rows set `executed_action_changed_after_visual_read=true`;
- rejected rows include concrete failure reasons;
- stale queued macro actions are cleared when the final action changes;
- summaries count executed action changes separately from completed readbacks.

- [ ] **Step 10.7: Audit run artifacts**

For the latest run directory, verify:

```bash
RUN_RESULTS_DIR=../results/awarevln_memory_phase23_smoke_override
python evaluation/scripts/summarize_aware_memory_phase23.py \
  --results-dir "$RUN_RESULTS_DIR" \
  --split val_unseen \
  --total-chunks 1
```

The final audit must answer yes/no for:

- Did the model read actual current and memory image payloads?
- Which memory image ids and paths were attached?
- Which memory ids did Qwen report using?
- Did the oracle scenario hit the expected memory id or expected keyframe step window?
- Did readback change the executed action?
- Did any queued macro primitive bypass the controller?
- Did any reasoning-only output disappear without a non-step trace row?

Acceptance:

- The migrated implementation is ready only when those questions are answerable from trace and summary artifacts without reading console logs.

---

## Final Definition of Done

- `AWAREVLN_MEMORY_MODE=off` is behavior-preserving and covered by tests.
- Exact `scene_id:episode_id` filtering runs before dataset chunking.
- `keyframe_only` writes current frames, keyframes, and metadata/source-labeled memory records.
- `image_read_log_only` reads current image plus Top-K memory image files and never changes action.
- `image_read_action_override` can change only the next primitive action and clears stale queued primitives when it changes the action.
- `reasoning_only` is traced as a non-step row with `final_action=null`.
- Chunked enabled-memory runs write chunk-scoped traces and summaries.
- Summary output distinguishes readback activity, correct-memory oracle hits, and executed-action changes.
- The original `evaluation/scripts/eval/r2r.sh` remains a baseline launcher.
- No training behavior is changed.
