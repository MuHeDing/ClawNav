# Memory-Gated OpenClaw Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a memory-gated OpenClaw Harness that keeps memory as the central innovation while preventing raw Qwen memory/subgoal text from shifting JanusVLN's action distribution, especially STOP behavior.

**Architecture:** Add a policy-safe memory adapter between the OpenClaw/Qwen visual-memory cache and `NavigationPolicySkill`, split memory into policy/control/evidence contexts, and add a STOP verification gate in the OpenClaw runtime. The implementation is staged so each ablation can be evaluated independently against the current `memory_guided_policy_fast` run.

**Tech Stack:** Python, existing ClawNav harness package, OpenClaw gateway adapter, pytest, JSONL harness traces, existing R2R val_unseen evaluation scripts.

---

## Origin Documents

- Design source: `docs/plans/2026-06-09-memory-gated-openclaw-harness.md`
- Original ClawNav/OpenClaw Harness concept: `docs/plans/2026-05-08-claw-style-harness-for-fast-janusvln.md`
- Current launcher under discussion: `scripts/run_memory_guided_fast_100_val_unseen_screen.sh`
- Current runner: `scripts/run_memory_guided_fast_large_eval.py`
- Current gateway planner: `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- Current runtime bridge: `src/harness/openclaw/runtime.py`
- Current JanusVLN wrapper skill: `src/harness/skills/navigation_policy.py`

## Problem Frame

The current `memory_guided_policy_fast` flow works as:

```text
Qwen visual_update
-> cache last_visual_summary / last_suggested_subgoal / last_qwen_reason
-> fast_text step skips Qwen
-> _memory_guided_policy_arguments() appends raw cached memory to NavigationPolicySkill
-> NavigationPolicySkill augments JanusVLN instruction with active_subgoal and memory_context_text
```

This keeps evaluation fast, but it sends free-form Qwen text into a JanusVLN prompt distribution that was primarily built around R2R instructions. Existing result analysis shows the failure signal is not a total navigation collapse: near-miss failures (`success=0, os=1`) increase, which suggests raw memory/subgoal text is shifting STOP behavior.

The implementation must therefore preserve memory as the system-level contribution while changing how memory enters JanusVLN.

## Non-Goals

- Do not remove memory from the method.
- Do not modify JanusVLN model internals.
- Do not use `distance_to_goal`, `success`, SPL, oracle shortest path, future observations, or future trajectory frames for online decisions.
- Do not enable planner action override by default.
- Do not pass Qwen raw reasoning directly to `NavigationPolicySkill`, except in explicit `raw` compatibility mode for apples-to-apples comparison.
- Do not introduce an external dependency for the first implementation.

## Implementation Principles

1. **Memory is split by consumer.**
   - `policy_context`: short, policy-safe cue for JanusVLN.
   - `control_context`: non-oracle evidence for Harness/Critic/Verifier.
   - `evidence_context`: raw Qwen text, source images, and trace evidence for audit and later recall.

2. **JanusVLN gets only policy-compatible cues.**
   - Allowed: landmarks, local direction, short navigation cue.
   - Disallowed: arrival claims, STOP instructions, Qwen raw reasoning, action text.

3. **STOP is high risk.**
   - STOP proposed under injected memory should be verified before execution.
   - First implementation should use a clean-prompt JanusVLN verification path because it is non-oracle and does not require a new VLM call.

4. **Every behavior is ablatable.**
   - `raw`: current behavior.
   - `no_reason`: remove `Last Qwen reason`.
   - `safe_cue`: policy-safe memory cue only.
   - `safe_cue + audit/block STOP verification`: ablation family composed from `MEMORY_POLICY_MODE=safe_cue` and `STOP_VERIFICATION_MODE=audit_clean_prompt|clean_prompt_block`.
   - `off`: no fast memory context.

5. **Trace everything needed for claims.**
   - Memory injected?
   - Raw memory filtered?
   - STOP verifier triggered?
   - STOP blocked?
   - Clean action vs memory action disagreement?

## Proposed File Structure

### New Files

- `src/harness/memory/policy_memory_adapter.py`
  - Converts raw OpenClaw/Qwen memory into policy/control/evidence contexts.
  - Owns STOP-semantics filtering and cue construction.

- `tests/test_policy_memory_adapter.py`
  - Unit tests for filtering, distillation, mode behavior, and no-oracle guarantees.

- `docs/plans/2026-06-09-memory-gated-openclaw-discriminative-keys.txt`
  - One `scene_id:episode_id` key per line for the 27-episode discriminative ablation set.

### Modified Files

- `src/harness/openclaw/openclaw_cli_plan_gateway.py`
  - Replace raw `_memory_guided_policy_arguments()` construction with adapter-backed distillation.
  - Add gateway CLI flags for memory policy mode and STOP semantic filtering.
  - Attach memory-gate audit metadata to `runtime_metadata`.

- `src/harness/openclaw/runtime.py`
  - Preserve `control_context` and `memory_gate` in runtime metadata, not `nav_payload`.
  - Add runtime-level policy-memory gate for clean `off` ablations.
  - Add clean-prompt STOP verification gate after `NavigationPolicySkill` returns STOP.
  - Add trace metadata for triggered/blocked STOP verification.

- `src/harness/skills/navigation_policy.py`
  - Keep existing public behavior, but optionally support safer prompt labels if required.
  - Do not add control-side fields to the input schema.

- `scripts/start_openclaw_cli_plan_gateway.sh`
  - Add env pass-through for memory-gating flags.

- `scripts/run_memory_guided_fast_large_eval.py`
  - Add `LargeEvalConfig` fields and CLI args for memory policy mode / STOP verifier mode.
  - Stop hardcoding `OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=1`; make it configurable.
  - Extend summary extraction for Phase 1 gate, STOP-verifier, and optional memory-effect metrics.

- `scripts/run_memory_guided_fast_100_val_unseen_screen.sh`
  - Add env defaults for the new ablation knobs.

- `src/evaluation_harness.py`
  - Pass STOP verification mode into `OpenClawVLNRuntime`.

- `scripts/evaluation_openclaw_gateway.sh`
  - Pass through `OPENCLAW_STOP_VERIFICATION_MODE`.

- `scripts/summarize_openclaw_vln_ablation.py`
  - Include new memory gate and STOP verifier metrics when summarizing runs.

- `tests/test_openclaw_cli_plan_gateway.py`
  - Update current `memory_guided_policy_fast` expectations and add mode-specific tests.

- `tests/test_openclaw_runtime_bridge.py`
  - Add clean-prompt STOP verification tests.

- `tests/test_memory_guided_fast_large_eval.py`
  - Lock runner env defaults and gate summary metrics.

- `tests/test_evaluation_harness_openclaw_runtime.py`
  - Verify evaluation harness runtime construction receives STOP verification mode.

- `tests/test_evaluation_scripts.py`
  - Verify evaluation script environment pass-through for STOP verification.

- `tests/test_openclaw_ablation_summary.py`
  - Verify ablation summary rendering tolerates and reports memory-gate metrics.

### Deferred / Future Work Files

- `src/harness/memory/subgoal_state.py`
  - Optional state skeleton for structured subgoal progress, implemented only if Phase 1/2 metrics show string cues are still insufficient.

- `tests/test_subgoal_state.py`
  - Unit tests for future subgoal-state transitions and serialization.

### Documentation Updates

- `docs/plans/2026-06-09-memory-gated-openclaw-harness.md`
  - Optional follow-up after implementation to mark which stages shipped.

- `docs/runbooks/openclaw-memory-guided-fast-large-eval.md`
  - Add ablation commands after code exists.

## Data Contracts

### Policy Context

Policy context is the only part that may enter `NavigationPolicySkill`.

```python
{
    "memory_context_text": "Navigation cue: continue toward the arched doorway.\nVisible landmarks: hallway, arched doorway.",
    "active_subgoal": "continue toward the arched doorway",
}
```

Requirements:

- `memory_context_text` is at most 240 characters by default.
- `active_subgoal` is optional and must be a cue, not a command.
- No arrival / completion / STOP semantics.
- No raw Qwen reasoning prefix such as `Last Qwen reason:`.
- No `action_text` values.
- In `safe_cue` mode, `active_subgoal` may be emitted only when it is generated by the safe-cue grammar below; do not pass raw `last_suggested_subgoal` through unchanged.

### Safe-Cue Extraction Contract

The first implementation must make `safe_cue` deterministic and conservative. It should prefer losing a cue over shifting JanusVLN into a new prompt distribution.

Allowed input fields for policy cue generation:

```text
last_visual_summary
last_suggested_subgoal
runtime_context.memory_context_text
```

Disallowed input fields for policy cue generation:

```text
last_qwen_reason
action_text
control_context
evidence_context
oracle metrics
```

Allowed cue shapes:

```text
Navigation cue: continue toward <landmark>.
Navigation cue: keep the <landmark> on the left/right.
Navigation cue: proceed through/past/along <landmark>.
Visible landmarks: <landmark_1>, <landmark_2>, <landmark_3>.
Direction hint: left/right/straight/forward/upstairs/downstairs.
```

Extraction rules:

- Keep at most one navigation cue, three landmarks, and one direction hint.
- Drop any candidate sentence containing STOP, arrival, completion, or imperative action semantics.
- Drop any candidate sentence containing raw-reasoning labels such as `Last Qwen reason`, `therefore`, `I think`, or `we should`.
- Treat unparseable free-form text as `evidence_context` only; do not force it into `policy_context`.
- Do not generate an `active_subgoal` unless the cue contains at least one explicit landmark or direction hint after filtering.

Disallowed policy-cue terms for the first implementation:

```text
stop
stopping
wait
arrived
arrival
reached
successfully
destination
task complete
goal reached
final location
turn left
turn right
go back
move forward
```

`wait` and action phrases may still appear in the original R2R instruction. The adapter must filter only injected memory text, never the original instruction.

### Clean-Off Contract

In this spec, `off` means a clean policy-memory ablation on fast local-policy steps:

```text
No active_subgoal enters NavigationPolicySkill.
No memory_context_text enters NavigationPolicySkill.
No memory_images enter NavigationPolicySkill unless they are part of the normal recent-frame input.
Qwen visual_update cache and trace/evidence storage may still run.
```

The clean-off check must inspect the actual `NavigationPolicySkill` payload, not only gateway return values.

### Control Context

Control context is for Harness logic and trace metadata only.

```python
{
    "possible_goal_region": True,
    "stop_confidence": "medium",
    "requires_stop_verification": True,
    "memory_freshness_steps": 3,
    "filtered_stop_terms": ["reached", "stop"],
    "policy_cue_used": True,
}
```

Requirements:

- Must never be merged into `nav_payload`.
- Must never contain oracle metrics.
- May be copied into `runtime_metadata["memory_gate"]["control_context"]`.

### Evidence Context

Evidence context is for audit, trace, and later analysis.

```python
{
    "raw_visual_summary": "...",
    "raw_suggested_subgoal": "...",
    "raw_qwen_reason": "...",
    "source_image_paths": ["results/.../step_000020.png"],
}
```

Requirements:

- May contain raw Qwen reasoning.
- Must not be passed to `NavigationPolicySkill`.
- Must be bounded in trace output.

### Memory Gate Audit

Gateway should attach this to `runtime_metadata`:

```python
{
    "memory_gate": {
        "mode": "safe_cue",
        "policy_context_used": True,
        "raw_reason_included": False,
        "stop_semantics_filtered": True,
        "filtered_terms": ["reached", "stop"],
        "control_context": {...},
        "evidence_context": {...}
    }
}
```

Runtime should copy this into top-level trace metadata so summary scripts do not need to inspect nested gateway responses.

Concrete runtime requirement:

- Update `OpenClawVLNRuntime._metadata()` to copy `memory_gate` from `decision.runtime_metadata`, alongside existing planner metadata such as `context_audit`.
- Keep `memory_gate.control_context` and `memory_gate.evidence_context` out of `NavigationPolicySkill` payloads.
- Summary scripts should read the top-level trace `memory_gate` field first, and tolerate older traces where it is absent.

## Configuration Contract

Add these environment variables and CLI args.

### Gateway Flags

Environment:

```text
OPENCLAW_MODEL_MEMORY_POLICY_MODE=raw|no_reason|safe_cue|off
OPENCLAW_MODEL_FILTER_STOP_SEMANTICS=0|1
```

CLI:

```text
--openclaw_model_memory_policy_mode raw|no_reason|safe_cue|off
--openclaw_model_filter_stop_semantics 0|1
```

Defaults:

```text
OPENCLAW_MODEL_MEMORY_POLICY_MODE=raw
OPENCLAW_MODEL_FILTER_STOP_SEMANTICS=0
```

Compatibility:

- `raw` preserves current behavior.
- `raw` must bypass STOP-semantic filtering even if `OPENCLAW_MODEL_FILTER_STOP_SEMANTICS=1` is accidentally set. It is the byte-shape compatibility baseline and must still include the current raw labels such as `Cached visual memory`, `Suggested subgoal`, and `Last Qwen reason`.
- `off` must produce no `active_subgoal` and no `memory_context_text` on fast steps.
- `OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=0` must behave like `off` for fast-step policy injection.
- Existing launchers should keep raw-compatible behavior until the small-set ablation explicitly promotes `safe_cue`.

### Runner Flags

Environment:

```text
MEMORY_POLICY_MODE=raw|no_reason|safe_cue|off
FILTER_STOP_SEMANTICS=0|1
STOP_VERIFICATION_MODE=off|audit_clean_prompt|clean_prompt_block
FAST_USE_MEMORY_CONTEXT=0|1
OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED=<unset>|0|1
EPISODE_KEYS_PATH=/path/to/scene_episode_keys.txt
```

CLI:

```text
--memory-policy-mode raw|no_reason|safe_cue|off
--filter-stop-semantics 0|1
--stop-verification-mode off|audit_clean_prompt|clean_prompt_block
--fast-use-memory-context 0|1
--openclaw-policy-memory-context-enabled 0|1
--episode-keys-path /path/to/scene_episode_keys.txt
```

Compatibility defaults for existing large eval launchers:

```text
MEMORY_POLICY_MODE=raw
FILTER_STOP_SEMANTICS=0
STOP_VERIFICATION_MODE=off
FAST_USE_MEMORY_CONTEXT=1
OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED=<unset, derived from FAST_USE_MEMORY_CONTEXT>
```

Explicit memory-gated experiment profile:

```text
MEMORY_POLICY_MODE=safe_cue
FILTER_STOP_SEMANTICS=1
STOP_VERIFICATION_MODE=audit_clean_prompt
FAST_USE_MEMORY_CONTEXT=1
OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED=1
```

Runner behavior:

- `FAST_USE_MEMORY_CONTEXT=0` disables gateway fast-step memory arguments.
- `OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED=0` disables runtime-side policy memory merging before `NavigationPolicySkill`.
- If `OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED` is unset, derive it from `FAST_USE_MEMORY_CONTEXT` so the clean-off ablation disables both gateway and runtime policy-memory channels.
- If `OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED` is explicitly set, store that explicit override in `large_eval_summary.json` and do not silently replace it with the derived value.

Use `STOP_VERIFICATION_MODE=clean_prompt_block` only after audit metrics show that clean-prompt disagreement has low false-positive risk for true STOP decisions.

## Task Breakdown

### Task 1: Add PolicyMemoryAdapter

**Files:**

- Create: `src/harness/memory/policy_memory_adapter.py`
- Create: `tests/test_policy_memory_adapter.py`

**Implementation requirements:**

- Define immutable data containers for:
  - `PolicyMemoryContext`
  - `ControlMemoryContext`
  - `EvidenceMemoryContext`
  - `MemoryGateResult`
- Implement:
  - `contains_stop_semantics(text: str) -> bool`
  - `filter_stop_semantics(text: str) -> tuple[str, list[str]]`
  - `distill_memory(raw_memory: dict, runtime_context: dict, mode: str, should_filter_stop_semantics: bool) -> MemoryGateResult`
  - `to_navigation_arguments(result: MemoryGateResult) -> dict`
- Keep lexical logic deterministic and dependency-free.
- Treat original task instruction separately from injected memory. Do not filter the original instruction.

**STOP semantic terms for first implementation:**

Use the `Disallowed policy-cue terms for the first implementation` list in the Safe-Cue Extraction Contract as the single authoritative list. Task 1 must not maintain a second shorter list. The adapter may add phrase aliases, but it must never enforce fewer terms than the contract list.

**Mode behavior:**

```text
raw:
  Match current behavior as closely as possible.
  Bypass stop-semantic filtering regardless of should_filter_stop_semantics.
  Preserve current memory_context_text labels, including Last Qwen reason.

no_reason:
  Use cached visual memory and suggested subgoal, but exclude Last Qwen reason.

safe_cue:
  Produce only short landmark/direction cue that satisfies the Safe-Cue Extraction Contract.
  Omit policy_context when no candidate cue satisfies the grammar.
  Put possible stop evidence into control_context.

off:
  Produce empty policy_context and no navigation arguments.
```

**Tests:**

- `test_stop_semantics_are_filtered_from_policy_context`
- `test_original_instruction_stop_words_are_not_filtered`
- `test_no_reason_mode_excludes_last_qwen_reason`
- `test_safe_cue_mode_uses_landmark_and_direction_only`
- `test_safe_cue_drops_unparseable_text_to_evidence_only`
- `test_safe_cue_filters_action_and_arrival_synonyms`
- `test_safe_cue_active_subgoal_requires_landmark_or_direction`
- `test_off_mode_returns_empty_navigation_arguments`
- `test_control_context_retains_stop_evidence_without_oracle_metrics`
- `test_policy_context_is_bounded`

**Verification command:**

```bash
PYTHONPATH=src pytest tests/test_policy_memory_adapter.py -q
```

### Task 2: Integrate Adapter Into OpenClaw Gateway Fast Path

**Files:**

- Modify: `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- Modify: `tests/test_openclaw_cli_plan_gateway.py`

**Implementation requirements:**

- Add planner constructor args:
  - `openclaw_model_memory_policy_mode: str = "raw"`
  - `openclaw_model_filter_stop_semantics: bool = False`
- Add CLI args and wire them into `OpenClawCliPlanPlanner`.
- Update `_memory_guided_policy_fast_decision()`:
  - If `openclaw_model_fast_use_memory_context` is false, keep the local-policy decision but set `arguments={}` and record `memory_gate.mode="off"`.
  - Otherwise call `PolicyMemoryAdapter` once and keep the returned `MemoryGateResult`.
  - Derive `NavigationPolicySkill` arguments from the same `MemoryGateResult`.
  - Attach the same result's `memory_gate` audit in `runtime_metadata`.
  - Keep `model_call_skipped=True`, `qwen_api_called=False`.
- Update `_memory_guided_policy_arguments()` to either accept a precomputed `MemoryGateResult` or become a thin helper around `to_navigation_arguments(result)`.
- Do not call the adapter twice for a single fast step.
- Keep the current visual-update Qwen cache intact.

**Tests:**

- Update existing `test_cli_plan_gateway_model_mode_memory_guided_policy_fast_skips_qwen_on_fast_steps`.
- Add:
  - `test_memory_guided_fast_respects_fast_use_memory_context_off`
  - `test_safe_cue_mode_does_not_include_last_qwen_reason`
  - `test_safe_cue_mode_filters_stop_semantics`
  - `test_no_reason_mode_keeps_summary_but_removes_reason`
  - `test_raw_mode_preserves_current_memory_context_shape`
  - `test_raw_mode_ignores_stop_semantic_filter_flag`
  - `test_memory_gate_audit_records_filtered_terms`
  - `test_memory_gate_audit_and_navigation_args_share_one_adapter_result`

**Verification command:**

```bash
PYTHONPATH=src pytest tests/test_openclaw_cli_plan_gateway.py -q
```

### Task 3: Pass New Config Through Scripts, Runner, and Runtime Clean-Off Gate

**Files:**

- Create: `docs/plans/2026-06-09-memory-gated-openclaw-discriminative-keys.txt`
- Modify: `src/harness/openclaw/runtime.py`
- Modify: `src/evaluation_harness.py`
- Modify: `scripts/start_openclaw_cli_plan_gateway.sh`
- Modify: `scripts/run_memory_guided_fast_large_eval.py`
- Modify: `scripts/run_memory_guided_fast_100_val_unseen_screen.sh`
- Modify: `scripts/evaluation_openclaw_gateway.sh`
- Modify: `tests/test_openclaw_runtime_bridge.py`
- Modify: `tests/test_evaluation_harness_openclaw_runtime.py`
- Modify: `tests/test_memory_guided_fast_large_eval.py`
- Modify: `tests/test_evaluation_scripts.py`

**Implementation requirements:**

- Add env defaults to `scripts/start_openclaw_cli_plan_gateway.sh`.
- Create `docs/plans/2026-06-09-memory-gated-openclaw-discriminative-keys.txt` with the exact 27 keys listed in the Evaluation Plan.
- Extend `LargeEvalConfig` with:
  - `memory_policy_mode: str = "raw"`
  - `filter_stop_semantics: int = 0`
  - `stop_verification_mode: str = "off"`
  - `fast_use_memory_context: int = 1`
  - `policy_memory_context_enabled: Optional[int] = None`
  - `episode_keys_path: Optional[str] = None`
- Update `build_adapter_env()`:
  - Use config fields instead of hardcoded `OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=1`.
  - Pass memory policy flags.
  - Pass `EPISODE_KEYS_PATH` when present for traceability, but do not rely on the evaluator to read it directly.
- Add `load_episode_keys_from_path(path: str) -> list[str]` in `scripts/run_memory_guided_fast_large_eval.py`:
  - Ignore blank lines and lines starting with `#`.
  - Validate every non-comment line uses `scene_id:episode_id`.
  - Preserve file order.
  - Raise a clear error on duplicate keys.
  - Return exactly the 27 keys for `docs/plans/2026-06-09-memory-gated-openclaw-discriminative-keys.txt`.
- Update `build_eval_env()`:
  - If `episode_keys_path` is set, read it with `load_episode_keys_from_path()`.
  - Pass the comma-joined keys through existing `HARNESS_EPISODE_KEYS`.
  - Set `HARNESS_DEBUG_MAX_EPISODES` / expected episodes to the loaded key count.
  - Store both `episode_keys_path` and `effective_episode_key_count` in `large_eval_summary.json`.
- Add Phase 1 gate metrics to `collect_run_summary()` so the Phase 1 small-set gate is computable before STOP verification exists:
  - `os_rate`
  - `average_steps`
  - `final_stop_count`
  - `near_miss_fail_count`
  - `wrong_stop_count`
  - `far_wrong_stop_count`
  - `timeout_or_loop_count`
  - `memory_gate_policy_context_used_count`
  - `policy_context_injection_rate`
  - `active_subgoal_injection_rate`
  - `memory_gate_raw_reason_included_count`
  - `memory_gate_stop_semantics_filtered_count`
  - `fast_policy_memory_context_enabled`
  - `policy_memory_context_enabled`
  - `janus_only_total_count`
  - `janus_only_recovered_count`
  - `claw_only_total_count`
  - `claw_only_preserved_count`
- Define the Phase 1 metric sources and formulas explicitly:
  - `final_stop_count`: count episodes whose final executed action is `STOP`. Prefer the latest per-episode action row from the harness trace; use a result-level final-action field only if one is added by the implementation.
  - `near_miss_fail_count`: count `success == 0 and os == 1`.
  - `wrong_stop_count`: count `final_action == STOP and success == 0`.
  - `far_wrong_stop_count`: count `final_action == STOP and success == 0 and os == 0`.
  - `timeout_or_loop_count`: count episodes reaching the configured maximum step limit, using result `steps` when available and otherwise the max trace step per episode.
  - `janus_only_recovered_count`: among the fixed 17 Janus-only-success discriminative keys, count keys that succeed in the candidate mode.
  - `claw_only_preserved_count`: among the fixed 10 prior Claw-only-success discriminative keys, count keys that still succeed in the candidate mode.
  - `policy_context_injection_rate`: episodes or fast-policy calls with policy context present divided by the same denominator used for `active_subgoal_injection_rate`; report the denominator name in summary metadata.
- Treat Task 6a STOP-verifier metrics and Task 6b optional memory-effect metrics as Phase 2/3 additions, not prerequisites for the Phase 1 ablation gate.
- Add runtime-level policy memory gate:
  - `OpenClawVLNRuntime(..., policy_memory_context_enabled: bool = True)`.
  - When false, suppress `active_subgoal`, `memory_context_text`, and memory-derived `memory_images` before calling `NavigationPolicySkill`.
  - The suppression must apply to direct planner arguments, `MemoryQuerySkill` payloads, and nested `policy_context` payloads.
  - The trace should record `policy_memory_context_enabled=false`.
- Wire the runtime gate through evaluation:
  - Add `--openclaw_policy_memory_context_enabled 0|1` to `src/evaluation_harness.py`.
  - Store it on `HarnessConfig`.
  - Pass it from `build_harness_components()` into `OpenClawVLNRuntime(policy_memory_context_enabled=...)`.
  - Add `OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED` env pass-through in `scripts/evaluation_openclaw_gateway.sh`.
  - In `scripts/run_memory_guided_fast_large_eval.py`, derive `OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED` from `fast_use_memory_context` only when `policy_memory_context_enabled is None`; otherwise pass the explicit override through unchanged.
- Update `parse_args()` and `config_from_args()`.
- Update `scripts/run_memory_guided_fast_100_val_unseen_screen.sh` to expose:
  - `MEMORY_POLICY_MODE`
  - `FILTER_STOP_SEMANTICS`
  - `STOP_VERIFICATION_MODE`
  - `FAST_USE_MEMORY_CONTEXT`
  - `OPENCLAW_POLICY_MEMORY_CONTEXT_ENABLED`
  - `EPISODE_KEYS_PATH`
- Preserve backward-compatible overrides via environment variables.

**Tests:**

- `test_command_env_defaults_lock_memory_guided_fast_contract` should assert new defaults.
- Add:
  - `test_runner_can_disable_fast_memory_context`
  - `test_runner_passes_memory_policy_mode_to_adapter`
  - `test_runner_derives_policy_memory_context_enabled_when_unset`
  - `test_runner_preserves_explicit_policy_memory_context_enabled_override`
  - `test_runner_reads_episode_keys_path_and_sets_harness_episode_keys`
  - `test_runner_rejects_duplicate_episode_keys_path_entries`
  - `test_build_harness_components_passes_policy_memory_context_enabled`
  - `test_eval_script_passes_policy_memory_context_enabled_env`
  - `test_collect_run_summary_counts_phase1_gate_metrics`
  - `test_screen_launcher_exports_memory_gate_envs`
  - `test_runtime_clean_off_suppresses_navigation_policy_memory_payload`

**Verification commands:**

```bash
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py -q
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py -q
PYTHONPATH=src pytest tests/test_memory_guided_fast_large_eval.py -q
PYTHONPATH=src pytest tests/test_evaluation_scripts.py -q
```

### Task 4: Add Runtime STOP Verification Gate

**Files:**

- Modify: `src/harness/openclaw/runtime.py`
- Modify: `tests/test_openclaw_runtime_bridge.py`

**Implementation requirements:**

- Add runtime constructor arg:
  - `stop_verification_mode: str = "off"` initially in runtime, wired from evaluation later.
- Supported modes:
  - `off`
  - `audit_clean_prompt`
  - `clean_prompt_block`
- After `NavigationPolicySkill` returns action:
  - If action is not `STOP`, do nothing.
  - If action is `STOP` and `stop_verification_mode == "off"`, do nothing.
  - If action is `STOP` and memory gate/control context requires verification, run clean-prompt verification.
- Clean-prompt verification:
  - Call `NavigationPolicySkill` again with the same `recent_frames` and current image, but without `active_subgoal`, `memory_context_text`, `memory_images`, nested `policy_context`, `control_context`, `evidence_context`, `memory_gate`, or planner action guidance fields.
  - Do not call the planner again and do not recurse into another `OpenClawVLNRuntime.step()`.
  - The verifier call must be audit-only input to the STOP verifier metadata. It must not create a second context-engine record, memory write, recall, working-memory update, or episode action-history update.
  - In `audit_clean_prompt`, record clean action and disagreement, but execute the original memory-prompted action.
  - In `clean_prompt_block`, if clean action is non-STOP, return the clean action and mark STOP blocked.
  - In `clean_prompt_block`, if clean action is STOP, execute STOP.
  - Only the final selected action, original STOP in audit mode or clean action in blocking mode, should be recorded as the executed runtime action.
- Metadata:
  - `stop_verifier.triggered`
  - `stop_verifier.mode`
  - `stop_verifier.memory_action`
  - `stop_verifier.clean_action`
  - `stop_verifier.disagreed`
  - `stop_verifier.blocked`
  - `stop_verifier.reason`
- Keep all verifier inputs non-oracle.
- Do not enable `clean_prompt_block` by default. It is an experiment mode that should be promoted only after `audit_clean_prompt` shows acceptable clean/memory STOP agreement.

**Decision rule for first implementation:**

Trigger clean verification when any of these are true:

```text
action_text == STOP
and memory_gate.policy_context_used == true
```

or:

```text
action_text == STOP
and memory_gate.control_context.requires_stop_verification == true
```

**Tests:**

- `test_stop_verifier_blocks_memory_induced_stop_when_clean_policy_moves`
- `test_stop_verifier_audit_mode_records_disagreement_without_blocking`
- `test_stop_verifier_allows_stop_when_clean_policy_also_stops`
- `test_stop_verifier_does_not_run_without_memory_context`
- `test_stop_verifier_metadata_is_recorded`
- `test_stop_verifier_does_not_use_oracle_metrics`

**Verification command:**

```bash
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py -q
```

### Task 5: Wire STOP Verification Through Evaluation Harness

**Files:**

- Modify: `src/evaluation_harness.py`
- Modify: `scripts/evaluation_openclaw_gateway.sh`
- Modify: `tests/test_evaluation_harness_openclaw_runtime.py`
- Modify: `tests/test_evaluation_scripts.py`

**Implementation requirements:**

- Add CLI arg:
  - `--openclaw_stop_verification_mode`
- Add env pass-through:
  - `OPENCLAW_STOP_VERIFICATION_MODE`
- Construct `OpenClawVLNRuntime(..., stop_verification_mode=args.openclaw_stop_verification_mode)`.
- Keep default `off` at low-level runtime if invoked directly.
- Use runner default `off` for compatibility launchers.
- Use `audit_clean_prompt` for the first memory-gated verifier experiment.
- Require explicit `STOP_VERIFICATION_MODE=clean_prompt_block` for blocking experiments.

**Tests:**

- `test_openclaw_runtime_receives_stop_verification_mode`
- `test_gateway_eval_script_passes_stop_verification_env`

**Verification command:**

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py tests/test_evaluation_scripts.py -q
```

### Task 6: Extend Trace Summary and Ablation Metrics

**Files:**

- Modify: `scripts/run_memory_guided_fast_large_eval.py`
- Modify: `scripts/summarize_openclaw_vln_ablation.py`
- Modify: `tests/test_memory_guided_fast_large_eval.py`
- Modify: `tests/test_openclaw_ablation_summary.py`

**Implementation requirements:**

Task 6 has two implementation bands:

- **Task 6a, required for Phase 2:** STOP-verifier summary metrics needed by the A3/A3b gates.
- **Task 6b, optional for Phase 3:** memory-effect metrics such as `memory_changed_action_rate` and ablation report rendering.

`collect_run_summary()` should count:

```text
result_count
success_rate
spl_rate
os_rate
average_steps
final_stop_count
near_miss_fail_count
wrong_stop_count
far_wrong_stop_count
timeout_or_loop_count
memory_gate_policy_context_used_count
policy_context_injection_rate
active_subgoal_injection_rate
memory_gate_stop_semantics_filtered_count
memory_gate_raw_reason_included_count
memory_changed_action_rate
memory_changed_action_win_count
memory_changed_action_loss_count
stop_verifier_trigger_count
stop_verifier_block_count
stop_verifier_disagreement_count
stop_verifier_clean_stop_count
stop_verifier_clean_nonstop_count
memory_induced_stop_count
blocked_stop_later_success
blocked_stop_later_failure
clean_nonstop_disagreement_executed_stop_success_count
clean_nonstop_disagreement_wrong_stop_count
hypothetical_true_positive_block_count
hypothetical_true_positive_block_rate
hypothetical_false_positive_block_count
hypothetical_false_positive_block_rate
janus_only_total_count
janus_only_recovered_count
claw_only_total_count
claw_only_preserved_count
```

STOP-verifier metric formulas:

```text
clean_nonstop_disagreement = verifier triggered, memory_action == STOP, clean_action != STOP

clean_nonstop_disagreement_wrong_stop_count =
  audit-mode clean_nonstop_disagreement rows whose executed memory STOP ends with episode success=0

clean_nonstop_disagreement_executed_stop_success_count =
  audit-mode clean_nonstop_disagreement rows whose executed memory STOP ends with episode success=1

hypothetical_true_positive_block_count =
  clean_nonstop_disagreement_wrong_stop_count

hypothetical_false_positive_block_count =
  clean_nonstop_disagreement_executed_stop_success_count

hypothetical_true_positive_block_rate =
  hypothetical_true_positive_block_count /
  max(1, hypothetical_true_positive_block_count + hypothetical_false_positive_block_count)

hypothetical_false_positive_block_rate =
  hypothetical_false_positive_block_count /
  max(1, hypothetical_true_positive_block_count + hypothetical_false_positive_block_count)

blocked_stop_later_success =
  blocking-mode episodes where a memory-prompted STOP was blocked and the final episode result has success=1

blocked_stop_later_failure =
  blocking-mode episodes where a memory-prompted STOP was blocked and the final episode result has success=0
```

`memory_changed_action_rate` must be based on one of these explicit methods:

```text
preferred: shadow clean-policy call sampled on fast steps, recorded as audit-only metadata
optional: paired run comparison on shared scene_id/episode_id and step_id-like trace rows only when the pre-action state is confirmed equivalent
```

Do not report `memory_changed_action_rate` if neither method is implemented; report `null` with `memory_changed_action_method="not_available"` instead. If a paired run diverges before the compared step, do not use that step for action-rate accounting; use the paired run only for episode-level outcome deltas.

Phase gates must be computable from generated summaries alone. Do not require ad hoc notebook analysis for:

```text
near_miss_fail_count
wrong_stop_count
timeout_or_loop_count
janus_only_recovered_count
claw_only_preserved_count
clean_nonstop_disagreement_executed_stop_success_count
clean_nonstop_disagreement_wrong_stop_count
hypothetical_true_positive_block_rate
hypothetical_false_positive_block_rate
```

The Janus-only / Claw-only subgroup counters should be computed by matching result rows on `(scene_id, episode_id)` against the fixed 27-key discriminative set labels in the Evaluation Plan.

`render_markdown_report()` should include a "Memory Gate / STOP Verification" section.

`scripts/summarize_openclaw_vln_ablation.py` should surface these metrics when present and tolerate old traces where fields are absent.

**Tests:**

- `test_collect_run_summary_counts_memory_gate_metrics`
- `test_collect_run_summary_counts_stop_verifier_metrics`
- `test_collect_run_summary_reports_memory_changed_action_method`
- `test_collect_run_summary_counts_gate_metrics`
- `test_collect_run_summary_counts_discriminative_subgroups`
- `test_collect_run_summary_counts_audit_hypothetical_stop_blocks`
- `test_ablation_summary_tolerates_missing_memory_gate_fields`

**Verification commands:**

```bash
PYTHONPATH=src pytest tests/test_memory_guided_fast_large_eval.py -q
PYTHONPATH=src pytest tests/test_openclaw_ablation_summary.py -q
```

### Task 7: Add Documentation and Runbook Updates

**Files:**

- Modify: `docs/plans/2026-06-09-memory-gated-openclaw-harness.md`
- Modify: `docs/runbooks/openclaw-memory-guided-fast-large-eval.md`

**Implementation requirements:**

- Document the new ablation knobs:

```text
MEMORY_POLICY_MODE
FILTER_STOP_SEMANTICS
STOP_VERIFICATION_MODE
FAST_USE_MEMORY_CONTEXT
```

- Add example commands for:
  - `A0 raw`
  - `A1 no_reason`
  - `A2 safe_cue`
  - `A3 safe_cue + audit_clean_prompt STOP verification`
  - `A3b safe_cue + clean_prompt_block STOP verification`
  - `A4 full_harness` as deferred/future work
  - `A5 off`
- Update `docs/plans/2026-06-09-memory-gated-openclaw-harness.md` so it matches this spec:
  - A3 is audit-first STOP verification.
  - A3b is explicit blocking STOP verification.
  - A4 remains full memory-gated harness with critic/replanner/control-memory scheduling and is deferred.
  - A5 is clean-off policy-memory ablation.
  - STOP verification should not block by default before audit metrics pass.
  - Subgoal State Machine is a future module gated by Phase 1/2 evidence, not a first implementation task.
- Add expected trace fields and summary fields.

**Verification:**

```bash
PYTHONPATH=src pytest tests/test_evaluation_scripts.py -q
```

## Suggested Implementation Sequence

### Phase 1: Make ablation controls real

Implement Tasks 1, 2, and 3.

Outcome:

```text
raw / no_reason / safe_cue / off modes exist
OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=0 is a real clean-off switch
gateway tests prove Last Qwen reason can be removed
large_eval_summary.json contains the Phase 1 gate metrics needed for 27-key ablation
```

Stop after this phase and run a small ablation on the discriminative episode set before building STOP verification.

### Phase 2: Add STOP verification

Implement Tasks 4, 5, and Task 6a.

Outcome:

```text
memory-induced STOP can be blocked by clean-prompt verification
trace records when STOP was blocked or allowed
large_eval_summary.json reports the STOP-verifier metrics required by the A3/A3b gates
```

### Phase 3: Add advanced metrics and runbook support

Implement Task 6b and Task 7.

Outcome:

```text
large_eval_summary.json reports optional memory-effect and extended ablation counts
ablation summaries can compare behavior across modes
```

## Test Matrix

Run focused tests after each phase:

```bash
PYTHONPATH=src pytest tests/test_policy_memory_adapter.py -q
PYTHONPATH=src pytest tests/test_openclaw_cli_plan_gateway.py -q
PYTHONPATH=src pytest tests/test_memory_guided_fast_large_eval.py -q
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py -q
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py -q
PYTHONPATH=src pytest tests/test_evaluation_scripts.py -q
PYTHONPATH=src pytest tests/test_openclaw_ablation_summary.py -q
```

Run a broader harness regression before launching evaluation:

```bash
PYTHONPATH=src pytest \
  tests/test_openclaw_cli_plan_gateway.py \
  tests/test_openclaw_runtime_bridge.py \
  tests/test_memory_guided_fast_large_eval.py \
  tests/test_evaluation_scripts.py \
  -q
```

## Evaluation Plan

### Fixed baseline artifacts

Use these artifacts as the current raw compatibility baseline:

```text
results/clawnav_openclaw_qwen_memory_guided_fast_100_val_unseen_20260607_100949_phase1_completion/large_eval_summary.json
results/clawnav_openclaw_qwen_memory_guided_fast_100_val_unseen_20260607_100949_phase1_completion/large_eval_report.md
results/clawnav_openclaw_qwen_memory_guided_fast_100_val_unseen_20260607_100949_phase1_completion/summary.json
results/clawnav_openclaw_qwen_memory_guided_fast_100_val_unseen_20260607_100949_phase1_completion/result.json
results/clawnav_openclaw_qwen_memory_guided_fast_100_val_unseen_20260607_100949_phase1_completion/harness_traces/
```

Baseline facts from these files:

```text
result_count=100
trace_rows=9740
success_rate=0.56
spl_rate=0.498229
os_rate=0.68
ne=5.401499
fallback_count=0
planner_fallback_count=0
cli_fallback_count=0
runtime_error_count=0
mode_counts.fast_text=8710
mode_counts.visual_update=1030
```

All new comparisons must be paired on shared `(scene_id, episode_id)` keys and must filter non-episode summary/footer records from JSONL-like result files.

### Small discriminative set

Use the 27-episode discriminative set before running the full 100. Task 3 must create this file and the runner must accept it through `EPISODE_KEYS_PATH` / `--episode-keys-path`:

```text
docs/plans/2026-06-09-memory-gated-openclaw-discriminative-keys.txt
```

File contents:

```text
2azQ1b91cZZ:10
8194nk5LbLH:220
EU6Fwq7SyZv:212
EU6Fwq7SyZv:390
QUCTc6BB5sX:97
QUCTc6BB5sX:105
QUCTc6BB5sX:153
TbHJrupSAjP:59
TbHJrupSAjP:135
X7HyMhZNoso:46
Z6MFQCViBuw:392
oLBMNvg9in8:533
pLe4wQe7qrG:412
x8F5xyUWy9e:7
x8F5xyUWy9e:33
x8F5xyUWy9e:310
x8F5xyUWy9e:522
8194nk5LbLH:1591
QUCTc6BB5sX:83
X7HyMhZNoso:375
oLBMNvg9in8:466
pLe4wQe7qrG:414
x8F5xyUWy9e:188
x8F5xyUWy9e:640
zsNo4HB9uLZ:1
zsNo4HB9uLZ:41
zsNo4HB9uLZ:87
```

The first 17 are Janus-only successes in the previous comparison; the last 10 are ClawNav-only successes. A good change should recover some of the first group without destroying the second group.

### Ablation modes

For the 27-key set, rerun all modes after implementation instead of mixing old and new code artifacts:

```text
A0 raw
A1 no_reason
A2 safe_cue
A3 safe_cue + audit_clean_prompt STOP verification
A3b safe_cue + clean_prompt_block STOP verification
A4 full memory-gated harness with critic/replanner/control-memory scheduling (deferred)
A5 off
```

Interpretation:

- If `no_reason` improves near-miss STOP, `Last Qwen reason` was a major contaminant.
- If `safe_cue` improves without losing Claw-only wins, policy-compatible memory works.
- If `audit_clean_prompt` frequently disagrees with memory-prompted STOP but those stops later succeed, clean-prompt verification is biased and must not block by default.
- If `clean_prompt_block` improves SR but increases 401-step timeouts, the verifier is too conservative.
- Do not report `A4` until critic/replanner/control-side memory scheduling is implemented; use `A3b` for the blocking-verifier-only mode.
- If `off` beats all memory modes, memory injection is still harmful and control-side-only memory should be prioritized.

### Phase 1 small-set pass/fail gates

Run after Tasks 1-3 only. This gate must not require STOP-verifier fields.

Modes:

```text
A0 raw
A1 no_reason
A2 safe_cue
A5 off
```

The Phase 1 small-set gate is considered passed only if all of these hold:

```text
all modes produce result_count=27
fallback_count=0
planner_fallback_count=0
cli_fallback_count=0
runtime_error_count=0
A2.safe_cue.success_rate >= A5.off.success_rate
A2.safe_cue.near_miss_fail_count <= A0.raw.near_miss_fail_count
A2.safe_cue preserves at least 8 of the 10 prior Claw-only successes
A2.safe_cue recovers at least 1 of the 17 prior Janus-only successes OR improves SPL over A5.off
policy_context_injection_rate > 0 for A2
active_subgoal_injection_rate is reported separately from policy_context_injection_rate
```

### Phase 2 STOP-verifier pass/fail gates

Run after Tasks 4, 5, and 6a only. This gate evaluates A3 and decides whether A3b may be run.

Modes:

```text
A3 safe_cue + audit_clean_prompt STOP verification
A3b safe_cue + clean_prompt_block STOP verification
```

The A3 audit gate is considered passed only if all of these hold:

```text
A3.audit_clean_prompt records stop_verifier_disagreement_count without changing executed actions
clean_nonstop_disagreement_executed_stop_success_count is reported
clean_nonstop_disagreement_wrong_stop_count is reported
hypothetical_true_positive_block_count is reported
hypothetical_false_positive_block_count is reported
hypothetical_true_positive_block_rate is reported
hypothetical_false_positive_block_rate is reported
hypothetical_false_positive_block_rate <= 0.20
timeout_or_loop_count does not increase over A2.safe_cue
blocked_stop_later_success is reported as 0 in audit mode
blocked_stop_later_failure is reported as 0 in audit mode
```

Run `A3b clean_prompt_block` only after the A3 audit gate passes. The A3b blocking gate is considered passed only if all of these hold:

```text
A3b.success_rate >= A2.safe_cue.success_rate
A3b.near_miss_fail_count <= A2.safe_cue.near_miss_fail_count
A3b.timeout_or_loop_count <= A2.safe_cue.timeout_or_loop_count
if A3b.stop_verifier_block_count > 0:
  A3b.blocked_stop_later_success > 0
  A3b.blocked_stop_later_failure / A3b.stop_verifier_block_count <= 0.20
if A3b.stop_verifier_block_count == 0:
  label A3b as no-op verifier behavior, not as evidence that blocking helped
```

### Full 100 val_unseen

Only run the full 100 after the relevant discriminative gate passes. For a policy-safe memory claim, the Phase 1 gate is enough. For a STOP-blocking verifier claim, the Phase 2 gate must pass first.

```text
A2 or A3b success_rate >= 0.56
A2 or A3b success_rate >= A5.off.success_rate
A2 or A3b near_miss_fail_count < A0.raw.near_miss_fail_count
A2 or A3b wrong_stop_count <= A0.raw.wrong_stop_count
A2 or A3b preserves at least 8 of the 10 prior Claw-only successes
policy_context_injection_rate > 0 for A2/A3/A3b
active_subgoal_injection_rate is reported separately from policy_context_injection_rate
fallback_count, planner_fallback_count, cli_fallback_count must remain zero
```

Claim-bearing memory results require memory-effect evidence. At least one of these must be true before the run is described as a memory contribution rather than only a filtering/control result:

```text
memory_changed_action_rate is reported with method=shadow_clean_policy or method=paired_trace_aligned
or
the candidate mode beats A5.off by at least one success on the 27-key set and does not lose more than 2 prior Claw-only successes
or
the candidate mode beats A5.off on full-100 success_rate and also improves SPL without increasing near_miss_fail_count
```

If none of these is true, label the run as `behavioral outcome only; memory action effect unverified`.

## Acceptance Criteria

### Phase 1 ready for discriminative ablation

- Unit tests pass for adapter, gateway, runtime clean-off gate, runner, and scripts.
- `MEMORY_POLICY_MODE=off` produces no `memory_context_text` or `active_subgoal` on fast policy steps.
- `FAST_USE_MEMORY_CONTEXT=0` produces no `memory_context_text`, `active_subgoal`, or memory-derived `memory_images` in the actual `NavigationPolicySkill` payload.
- `MEMORY_POLICY_MODE=no_reason` never includes `Last Qwen reason`.
- `MEMORY_POLICY_MODE=safe_cue` filters arrival / STOP semantics from `policy_context`.
- `MEMORY_POLICY_MODE=safe_cue` follows the Safe-Cue Extraction Contract and drops unparseable text to `evidence_context`.
- Raw Qwen reasoning is still available in evidence metadata, not lost.
- `control_context` is present in trace metadata and absent from `NavigationPolicySkill` payload.
- `memory_gate` is copied into top-level trace metadata.
- `policy_context_injection_rate` is reported for Phase 1 modes.
- `EPISODE_KEYS_PATH` can run the 27-key discriminative set without ad hoc dataset editing.
- Existing raw mode remains available for apples-to-apples comparison.
- No online decision path reads oracle metrics.

### Phase 2 ready for STOP-verifier ablation

- `STOP_VERIFICATION_MODE=audit_clean_prompt` records clean action, memory action, disagreement, and trigger reason without changing the executed action.
- `STOP_VERIFICATION_MODE=clean_prompt_block` is available only as an explicit override.
- STOP verifier metadata appears when a memory-prompted STOP is checked.
- `large_eval_summary.json` reports all Task 6a STOP-verifier metrics required by the A3/A3b gates.
- Verifier calls omit `active_subgoal`, `memory_context_text`, `memory_images`, nested `policy_context`, `control_context`, `evidence_context`, `memory_gate`, and planner action guidance fields.
- Verifier calls do not create a second context-engine record, memory write/recall, working-memory update, or episode action-history update.
- No verifier path reads oracle metrics.

### Full evaluation ready

- The 27-key discriminative set passes the relevant Phase 1 or Phase 2 gates for the claim being evaluated.
- `large_eval_summary.json` records `memory_policy_mode`, `filter_stop_semantics`, `stop_verification_mode`, `fast_use_memory_context`, `policy_memory_context_enabled`, `policy_memory_context_enabled_source`, `episode_keys_path`, and `effective_episode_key_count`.
- Claim-bearing memory results satisfy the memory-effect evidence requirement in the Full 100 gate.
- If `memory_changed_action_rate` is unavailable, the report must explicitly avoid claiming verified memory-action causality.
- Existing raw mode remains available for apples-to-apples comparison.

## Risks and Mitigations

### Risk: safe cue removes useful information

Mitigation:

- Keep `raw` and `no_reason` modes for comparison.
- Use `policy_context_injection_rate`, paired outcome deltas against `off`, and `memory_changed_action_rate` when a valid shadow or aligned-paired method exists to verify memory still affects decisions.

### Risk: STOP verifier blocks true positives

Mitigation:

- Track `blocked_stop_later_success` and timeout count.
- Start with `audit_clean_prompt` on memory-influenced STOP so the clean-prompt call is falsified before it blocks actions.
- Promote to `clean_prompt_block` only when `hypothetical_false_positive_block_rate <= 0.20` and timeout count does not increase over `A2.safe_cue`.
- Do not run verifier for every STOP until evidence supports it.

### Risk: extra JanusVLN calls slow evaluation

Mitigation:

- Trigger verifier only on STOP.
- Count `stop_verifier_trigger_count`.
- Compare latency overhead in summary.

### Risk: control context leaks into JanusVLN prompt

Mitigation:

- Keep `NAVIGATION_CONTEXT_KEYS` unchanged except for intentionally allowed policy fields.
- Add tests asserting control-only fields are absent from `navigation.calls[0]`.

### Risk: mode defaults make old runs incomparable

Mitigation:

- Preserve `raw` mode.
- Keep compatibility defaults as `MEMORY_POLICY_MODE=raw` and `STOP_VERIFICATION_MODE=off`.
- Record `memory_policy_mode`, `filter_stop_semantics`, `stop_verification_mode`, `fast_use_memory_context`, and `episode_keys_path` in `large_eval_summary.json`.

## Deferred Work

### Subgoal State Machine

Build `src/harness/memory/subgoal_state.py` only after Phase 1/2 metrics show that grammar-based `safe_cue` remains too brittle or too weak.

Deferred files:

- Create: `src/harness/memory/subgoal_state.py`
- Create: `tests/test_subgoal_state.py`
- Modify: `src/harness/memory/policy_memory_adapter.py`

Deferred states:

```text
seeking_landmark
passing_landmark
entering_region
verify_stop
```

Deferred acceptance gate:

```text
safe_cue improves STOP safety but loses more than 2 prior Claw-only successes
or
safe_cue has policy_context_injection_rate < 0.20 because the grammar drops too many cues
or
near_miss_fail_count remains above A5.off after safe-cue filtering
```

If the gate is met, implement a small immutable structure with `route_stage`, `target_landmark`, `direction_hint`, `negative_condition`, `confidence`, and `fresh_until_step`, plus `to_policy_cue()` and `to_control_context()`.

This future work corresponds to the design document's full-harness `A4` direction. Do not label `A3b clean_prompt_block` results as `A4`.

## Open Questions

1. How strict should the STOP semantic filter be for words like `wait`, which appear in original instructions but can also appear in harmful injected memory?
2. Should evidence metadata store raw Qwen reason in full, or a bounded excerpt plus pointer to source trace?
3. If `A3b clean_prompt_block` improves SR but hurts Claw-only success preservation, should the next verifier mode be `qwen_replan` or control-side-only memory?

## Implementation Handoff

Recommended first PR scope:

```text
Tasks 1-3 only:
  PolicyMemoryAdapter
  gateway integration
  runtime clean-off policy-memory gate
  runner/script ablation knobs
  27-key discriminative set file and EPISODE_KEYS_PATH support
  Phase 1 gate summary metrics
```

Reason:

```text
This makes the memory prompt shift measurable before adding STOP verification.
It also fixes the clean-off ablation path so future comparisons are trustworthy.
```

Recommended second PR scope:

```text
Tasks 4, 5, and 6a:
  STOP verifier
  evaluation harness wiring
  STOP-verifier summary metrics
```

Recommended third PR scope:

```text
Task 7:
  Documentation and runbook updates
```

Deferred PR scope, only if the Deferred Work gate is met:

```text
Subgoal state machine:
  src/harness/memory/subgoal_state.py
  tests/test_subgoal_state.py
  adapter integration for structured cue generation
```
