# OpenClaw Memory-Guided Fast Large Eval

This runbook is for the hybrid fast method:

```text
OPENCLAW_MODEL_FAST_MODE=memory_guided_policy_fast
OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=10
```

Method contract:

```text
visual_update rows: Qwen API call with image(s)
fast_text rows: local NavigationPolicySkill using cached visual memory, no Qwen call
```

Use this runbook for large-scale testing only after the 1-episode smoke gate
passes.

## Runner

Main script:

```bash
scripts/run_memory_guided_fast_large_eval.py
```

It does four things:

```text
1. Start the ClawNav OpenClaw /plan adapter, unless --reuse-adapter is set.
2. Run a 1-episode smoke.
3. Audit trace rows for hard benchmark validity gates.
4. Run the large eval and stop early if fallback contamination appears.
```

The runner writes both files into each output directory:

```text
large_eval_summary.json
large_eval_report.md
```

## Dry Run

Check the exact commands and environment without starting anything:

```bash
PYTHONPATH=.:src python scripts/run_memory_guided_fast_large_eval.py \
  --gpu 1 \
  --port 8011 \
  --episodes 30 \
  --image-interval-steps 10 \
  --model-max-images 2 \
  --dry-run
```

## 1 + 30 Run

Recommended first large-run command:

```bash
PYTHONPATH=.:src python scripts/run_memory_guided_fast_large_eval.py \
  --gpu 1 \
  --port 8011 \
  --episodes 30 \
  --max-steps 400 \
  --image-interval-steps 10 \
  --model-max-images 2 \
  --qwen-retries 1 \
  --qwen-retry-backoff-s 2 \
  --agent-timeout-s 90 \
  --gateway-timeout-s 300 \
  --master-port-smoke 20501 \
  --master-port-eval 20502
```

If DashScope read timeouts continue on visual updates, rerun with fewer images
per visual update:

```bash
PYTHONPATH=.:src python scripts/run_memory_guided_fast_large_eval.py \
  --gpu 1 \
  --port 8011 \
  --episodes 30 \
  --image-interval-steps 10 \
  --model-max-images 1
```

Record `--model-max-images` in the result name or report when comparing runs.
`--model-max-images 1` and `--model-max-images 2` are different benchmark
settings.

## Reuse Adapter

If the adapter is already running and you intentionally want to use it:

```bash
PYTHONPATH=.:src python scripts/run_memory_guided_fast_large_eval.py \
  --reuse-adapter \
  --gpu 2 \
  --port 8011 \
  --episodes 30
```

Without `--reuse-adapter`, the runner refuses to use an already responding port.
This avoids accidentally benchmarking with a stale adapter configuration.

## Audit Existing Results

Audit a completed or partial run without launching Qwen/Habitat:

```bash
PYTHONPATH=.:src python scripts/run_memory_guided_fast_large_eval.py \
  --audit-only results/clawnav_openclaw_qwen_memory_guided_fast_30_timeoutguard_20260528_gpu1 \
  --expected-episodes 30 \
  --model-max-images 2
```

This writes or refreshes:

```text
results/<run>/large_eval_summary.json
results/<run>/large_eval_report.md
```

## Hard Validity Gates

A valid benchmark must satisfy all of these:

```text
result_count == expected episodes
trace_rows > 0
fallback_count == 0
planner_fallback_count == 0
cli_fallback_count == 0
```

For `visual_update` rows:

```text
planner_authority == qwen
qwen_api_called == true
1 <= model_image_count <= OPENCLAW_MODEL_MAX_IMAGES
```

For `fast_text` rows:

```text
planner_authority == local_policy
qwen_api_called == false
model_call_skipped == true
model_image_count == 0
```

Any `openclaw_cli_model_fallback:` or `openclaw_cli_agent_fallback:` planner
reason makes the run diagnostic-only, even if early trace rows look correct.

## Outputs To Check

Useful report fields:

```text
Status: VALID_BENCHMARK or INVALID_DIAGNOSTIC_ONLY
Results: result_count / expected episodes
Mode counts: visual_update vs fast_text
Fallbacks: runtime, planner, cli
Qwen calls: total, visual_update, fast_text
Token Summary: fast assembled tokens, visual assembled tokens, visual provider tokens
Gate Failures: failing rows with scene / episode / step context
```

For the current implementation, expected Qwen usage is:

```text
fast_text: 0 Qwen tokens, because Qwen is not called
visual_update: Qwen input tokens are recorded in context_audit.provider_input_tokens
```
