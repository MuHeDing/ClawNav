# OpenClaw Context Engine Stress Test

This runbook verifies the memory-aware `/plan` context path from
`docs/plans/2026-05-23-openclaw-memory-aware-context-engine.md`.

The goal is to prove that OpenClaw planner context stays bounded across many
navigation steps. A valid strict run must show stable ClawNav assembled prompt
size, stable provider input tokens, no carried OpenClaw session history, and no
step-0 prompt leakage into later steps.

## Smoke Test

Use fake provider usage first. This proves the local stress and report pipeline
works without calling OpenClaw or Qwen.

```bash
PYTHONPATH=.:src python scripts/stress_openclaw_context_engine.py \
  --mode fake \
  --steps 100 \
  --output_dir results/openclaw_context_stress_fake
```

Expected outputs:

```text
results/openclaw_context_stress_fake/context_audit.jsonl
results/openclaw_context_stress_fake/summary.json
results/openclaw_context_stress_fake/context_audit_report.md
```

The fake smoke should pass. It is not provider evidence because
`provider_input_tokens` is generated locally.

## Real OpenClaw Run

Run the same stress path through `openclaw agent` when the OpenClaw profile and
Qwen provider are available.

```bash
PYTHONPATH=.:src python scripts/stress_openclaw_context_engine.py \
  --mode agent \
  --steps 100 \
  --agent_timeout 180 \
  --agent_max_input_tokens 50000 \
  --output_dir results/openclaw_context_stress_agent
```

If using a reduced OpenClaw profile:

```bash
PYTHONPATH=.:src python scripts/stress_openclaw_context_engine.py \
  --mode agent \
  --openclaw_profile clawnav-plan \
  --steps 100 \
  --agent_timeout 180 \
  --agent_max_input_tokens 50000 \
  --output_dir results/openclaw_context_stress_agent_clawnav_plan
```

## Stateless Model Run

Use this when testing the reduced Qwen prompt path. By default this stress path
uses `OPENCLAW_MODEL_PROVIDER=openclaw_cli` / `openclaw capability model run` so
it can exercise OpenClaw-compatible model output without using an agent session.
For the faster 30-episode evaluation, use the runbook's direct
`OPENCLAW_MODEL_PROVIDER=qwen_api` adapter mode.

```bash
PYTHONPATH=.:src python scripts/stress_openclaw_context_engine.py \
  --mode model \
  --steps 100 \
  --agent_timeout 180 \
  --openclaw_model qwen/qwen3.5-flash \
  --openclaw_model_provider openclaw_cli \
  --openclaw_model_max_images 3 \
  --output_dir results/openclaw_context_stress_model
```

Current OpenClaw `model.run --json` output does not expose provider usage on
this machine. A model-mode pass therefore proves bounded ClawNav prompt
assembly, stateless OpenClaw invocation, no prompt leakage, and no hard-limit
violation. It does not prove real `provider_input_tokens` unless OpenClaw starts
returning usage metadata.

When image paths in the payload exist on disk, model mode attaches them with
repeated `--file <path>` arguments. Missing image paths remain text metadata only
and are recorded in `model_missing_image_paths`. Stress-mode fake paths under
`/tmp/openclaw_context_stress` usually do not exist, so stress reports may show
`model_image_count=0` unless the harness supplies real frame files.

## Summarize Existing Audit

The summarizer accepts either `context_audit.jsonl` rows or trace rows that
contain a nested `context_audit` object.

```bash
PYTHONPATH=.:src python scripts/summarize_openclaw_context_audit.py \
  results/openclaw_context_stress_agent/context_audit.jsonl \
  --output results/openclaw_context_stress_agent/context_audit_report.md
```

JSON output:

```bash
PYTHONPATH=.:src python scripts/summarize_openclaw_context_audit.py \
  results/openclaw_context_stress_agent/context_audit.jsonl \
  --format json \
  --output results/openclaw_context_stress_agent/summary.json
```

## Pass Criteria

Strict agent-mode acceptance requires:

- at least 100 audit records
- `provider_input_tokens` present on every step
- `provider_input_tokens` stable from step 10 to the final step
- `history_tokens=0` on every step
- `openclaw_session_mode=fresh_per_step` on every step
- unique `openclaw_session_id` for every step
- no `STEP0_SECRET_MARKER_DO_NOT_LEAK` marker after step 0
- no `qwen_hard_limit_exceeded`

`provider_input_tokens` must come from real OpenClaw/Qwen usage metadata for a
strict provider claim. Fake mode is only a pipeline smoke.

Model-mode acceptance requires:

- at least 100 audit records
- `openclaw_session_mode=stateless_model` on every step
- empty `openclaw_session_id` on every step
- stable `assembled_prompt_tokens` from step 10 to the final step
- `model_image_count > 0` on steps where real image files exist and image input
  is expected
- `history_tokens=0` on every step
- no `STEP0_SECRET_MARKER_DO_NOT_LEAK` marker after step 0
- no `qwen_hard_limit_exceeded`

Model-mode `provider_input_tokens` is optional until OpenClaw exposes usage from
`capability model run`.

## Interpreting Failures

- `provider_usage_present` fails: OpenClaw did not return usage metadata or the
  session usage file could not be read. This is incomplete evidence, not a pass.
- `provider_usage_optional` passes in model mode: `provider_input_tokens` may be
  missing because `model.run` does not expose usage.
- `provider_tokens_stable` fails: hidden session history or another growing
  provider-side context may still be present.
- `assembled_prompt_stable` fails: the ClawNav prompt itself is growing in model
  mode.
- `fresh_per_step` fails: the adapter is not using step-scoped OpenClaw sessions.
- `stateless_model` fails: the model stress run did not use the stateless model
  planner path.
- `session_ids_unique` fails: later `/plan` calls may reuse earlier OpenClaw
  transcript state.
- `no_session_ids` fails: model mode unexpectedly produced session ids.
- `step0_marker_not_leaked_after_step0` fails: the assembled prompt is carrying
  previous raw prompt content.
- `no_qwen_hard_limit_violation` fails: the hard guard did not keep the request
  below the 50k input ceiling.

## Related Files

- `scripts/stress_openclaw_context_engine.py`
- `scripts/summarize_openclaw_context_audit.py`
- `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- `tests/test_openclaw_context_stress_script.py`
- `tests/test_openclaw_context_audit_summary.py`
