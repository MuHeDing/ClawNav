# OpenClaw Visual Readback Runbook

This runbook covers the Phase 0 through Phase 1b slice of the on-demand visual
memory readback controller work. The source-of-truth spec is
`docs/plans/2026-06-13-on-demand-visual-memory-readback-controller-arbitration.md`.

## Fast Local Checks

These checks do not require GPU, Habitat, or a live Qwen/OpenClaw gateway:

```bash
PYTHONPATH=.:src pytest \
  tests/test_visual_readback_config.py \
  tests/test_visual_readback_phase0.py \
  tests/test_visual_memory_read_skill.py \
  tests/test_visual_readback_manifest.py \
  tests/test_visual_readback_matrix_runner.py \
  tests/test_visual_readback_summary.py \
  tests/test_visual_readback_runbook.py \
  tests/test_visual_readback_runtime_smoke_checker.py \
  tests/test_openclaw_runtime_bridge.py \
  tests/test_evaluation_harness_openclaw_runtime.py \
  tests/test_evaluation_harness_imports.py \
  tests/test_memory_manager.py \
  tests/test_memory_skills.py \
  tests/test_harness_types.py \
  -q
```

Run a dry plan before any real evaluation:

```bash
PYTHONPATH=.:src python scripts/run_openclaw_visual_readback_matrix.py --phase phase0 --dry_run
```

The Phase 1b matrix must be dry-run checked with a passed Phase 0 gate artifact
and a fixed manifest:

```bash
PYTHONPATH=.:src python scripts/run_openclaw_visual_readback_matrix.py \
  --phase phase1b \
  --phase0_gate_artifact results/openclaw_visual_readback/phase0_gate.json \
  --manifest_path results/openclaw_visual_readback/fixed_replay_manifest.json \
  --dry_run
```

Summarize a completed run:

```bash
PYTHONPATH=.:src python scripts/summarize_openclaw_visual_readback.py \
  results/openclaw_visual_readback/Phase1a_V4 \
  --format markdown \
  --output results/openclaw_visual_readback/Phase1a_V4/visual_readback_report.md
```

## Gate Interpretation

Phase 0 may return:

- `no_go`: common gates failed. Stop at the audit report or fix the backend.
- `downscope_stop_only`: enough risky STOP cases for a narrow Phase 1a study, but
  not a full fixed replay matrix.
- `downscope_turn_only`: enough decision-point cases for a narrow Phase 1a study,
  but not a full fixed replay matrix.
- `candidate_set_ready`: enough primary candidate cases to generate a fixed
  replay manifest. This is not the same as a passed fixed manifest gate.
- `fixed_manifest_ready`: fixed replay manifest exists and can launch Phase 1b.

Do not launch the `V1/V2/V3/V4/V4c/V5` matrix unless the gate artifact says
`fixed_manifest_ready` and `fixed_manifest_gate_passed=true`.

## Integration Smoke

Real gateway smoke is an explicit integration gate, not a unit test. The smoke
record must include:

- `parse_success_rate`
- `image_attach_rate`
- `latency_ms`
- `timeout_rate`

Use the direct readback smoke before any Phase 1 matrix run:

```bash
PYTHONPATH=.:src python scripts/smoke_visual_readback_skill.py \
  --output_root results/qwen35flash_visual_readback_skill_smoke_manual \
  --model qwen/qwen3.5-flash \
  --include_runtime
```

The fixed artifacts are:

- `summary.json`: direct `VisualMemoryReadSkill` readback status, image attach
  rate, parse rate, timeout rate, latency, and matched memory IDs.
- `skill_payload.json`: normalized skill payload with
  `actually_read_image_paths`, `matched_memory_ids`, and grounding labels.
- `runtime_summary.json`: `OpenClawVLNRuntime -> MemoryQuerySkill ->
  VisualMemoryReadSkill -> controller` trace summary.
- `runtime_metadata.json`: full runtime metadata for inspecting tool call order.

For Phase 1, do not use `openclaw capability image describe-many` as the
evidence that `qwen/qwen3.5-flash` can perform readback in this path. The
readback path under test is `VisualMemoryReadSkill` using
`QwenDirectVisualReadbackAdapter + QwenApiModelClient`, because that is the
adapter that attaches `current + memory` images and emits the normalized
controller trace fields.

If image-backed memory cannot round-trip a real readable `image_path`, the
Phase 1b matrix is blocked even if text-only memory recall works.

### Controlled Runtime Smoke

Use this only to prove the real eval/runtime chain can produce a non-empty
`MemoryQuerySkill` image hit before `VisualMemoryReadSkill`. It is a smoke-only
fixture and must not be used as a navigation metric claim or a Phase 1b
denominator.

The smoke switch is:

- `OPENCLAW_VISUAL_READBACK_SMOKE_SEED_MEMORY=true`
- requires `OPENCLAW_VISUAL_READBACK_MODE=image_read_controller`
- requires an image-backed memory backend such as `HARNESS_MEMORY_BACKEND=image_backed_local`

The runtime sequence is:

1. `NavigationPolicySkill` produces the clean candidate action.
2. Runtime writes a controlled episode-local image memory with `MemoryWriteSkill`.
3. Runtime immediately queries that namespace with `MemoryQuerySkill`.
4. `VisualMemoryReadSkill` reads the current image plus retrieved memory images.

Example with the same model/gateway pattern as
`scripts/run_memory_guided_fast_100_val_unseen_screen.sh`:

```bash
MODE=controlled_seed \
OUTPUT_PATH=results/openclaw_visual_readback_phase1a_real_20260615/Phase1a_V4_smoke_seed_ep1 \
GPU=5 \
PORT=8013 \
MASTER_PORT=20615 \
EPISODE_KEY=zsNo4HB9uLZ:1 \
MAX_STEPS=80 \
bash scripts/run_visual_readback_runtime_smoke.sh
```

The wrapper starts `scripts/start_openclaw_cli_plan_gateway.sh`, runs
`scripts/evaluation_openclaw_gateway.sh`, writes
`visual_readback_summary.json`, and then runs:

```bash
PYTHONPATH=.:src python scripts/check_visual_readback_runtime_smoke.py \
  results/openclaw_visual_readback_phase1a_real_20260615/Phase1a_V4_smoke_seed_ep1 \
  --mode controlled_seed \
  --output results/openclaw_visual_readback_phase1a_real_20260615/Phase1a_V4_smoke_seed_ep1/runtime_smoke_check.json
```

Latest checked artifact:

- run dir:
  `results/openclaw_visual_readback_phase1a_real_20260615/Phase1a_V4_smoke_seed_ep1`
- trace rows: 29
- tool calls: `NavigationPolicySkill=29`, `MemoryWriteSkill=18`,
  `MemoryQuerySkill=46`, `VisualMemoryReadSkill=29`
- readback: 18 `completed`, 11 `skipped`, 17 completed rows with non-empty
  `matched_memory_ids`
- health: `fallback_rows=0`, `runtime_error_rows=0`, `planner_fallback_rows=0`
- summary:
  `results/openclaw_visual_readback_phase1a_real_20260615/Phase1a_V4_smoke_seed_ep1/visual_readback_summary.json`
- check:
  `results/openclaw_visual_readback_phase1a_real_20260615/Phase1a_V4_smoke_seed_ep1/runtime_smoke_check.json`

### No-Seed Runtime Diagnostic

After controlled smoke passes, run the same real eval/runtime path with
`OPENCLAW_VISUAL_READBACK_SMOKE_SEED_MEMORY=false` to see whether organic
harness memory produces an image hit:

```bash
MODE=no_seed_diagnostic \
OUTPUT_PATH=results/openclaw_visual_readback_phase1a_real_20260616/Phase1a_V4_gateway_update_bridge_ep1_escalated \
GPU=5 \
PORT=8015 \
MASTER_PORT=20618 \
EPISODE_KEY=zsNo4HB9uLZ:1 \
MAX_STEPS=20 \
bash scripts/run_visual_readback_runtime_smoke.sh
```

Latest checked no-seed artifact:

- run dir:
  `results/openclaw_visual_readback_phase1a_real_20260616/Phase1a_V4_gateway_update_bridge_ep1_escalated`
- `runtime_smoke_check.json`: `passed=true`,
  `diagnostic_status=natural_image_hit`
- trace rows: 21
- tool calls: `MemoryWriteSkill=3`, `MemoryQuerySkill=23`,
  `NavigationPolicySkill=21`, `VisualMemoryReadSkill=21`
- readback: 16 `completed`, 5 `skipped`; `skip_reasons={"no_trigger": 5}`
- natural image hit: `natural_image_hit_ready=true`
- bridge evidence: step 0 has
  `MemoryWriteSkill -> MemoryQuerySkill -> NavigationPolicySkill -> VisualMemoryReadSkill`,
  `visual_memory_update_status=updated`, non-empty `model_image_paths`,
  `matched_memory_ids=["image-backed-local-0"]`

This means the real gateway/eval path now naturally produces harness-side
`MemoryQuerySkill` image hits from gateway visual updates. The next step is a
larger mechanism-validation run, not another controlled hit fixture.

## Claim Boundary

Phase 1 is not an SR/SPL improvement claim. Phase 1 can report only mechanism
validation endpoints:

- `adjudicated_controller_support_rate`
- `visual_grounded_readback_rate`
- `paired_adjudicated_support_rate`
- `historical_memory_incremental_effect_rate`
- `historical_memory_dependent_effect_rate`

SR, SPL, NE, nDTW, and success rate may appear only as secondary diagnostics.
Executed replanner or broad controller-recovery claims belong to later phases.
