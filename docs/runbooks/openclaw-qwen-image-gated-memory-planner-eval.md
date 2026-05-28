# OpenClaw Qwen Image-Gated Memory Planner Eval

This runbook tests Stage 1 of the Qwen speedup plan:

```text
planner_authority=qwen
qwen_api_called=true
visual_update: text + image
fast_text: text only + adapter-local cached visual memory
```

The goal is to reduce Qwen image+text calls while keeping every navigation step under Qwen planner authority.

For the separate hybrid method, use:

```text
OPENCLAW_MODEL_FAST_MODE=memory_guided_policy_fast
```

That method intentionally skips Qwen on cached `fast_text` steps and records
`planner_authority=local_policy` for those rows. Do not mix those results with
the qwen-only Stage 1 result table.

Keep `OPENCLAW_QWEN_API_RETRIES=1` enabled for both methods. It only retries
the remaining Qwen calls (`visual_update` for the hybrid method, every step for
qwen-only Stage 1) and records retry metadata in `context_audit`.

The eval wrapper now enforces the adapter timeout budget reported by
`/health.timeout_budget` by default. If `OPENCLAW_GATEWAY_TIMEOUT` is below the
adapter's recommended value, preflight fails before any benchmark starts. This
prevents harness-side HTTP read timeouts from silently becoming
`planner_fallback=1`.

## Adapter

Start the OpenClaw gateway first if it is not already running:

```bash
openclaw gateway status
openclaw gateway run
```

Start the ClawNav `/plan` adapter:

```bash
OPENCLAW_PLANNER_MODE=model \
OPENCLAW_MODEL_PROVIDER=qwen_api \
OPENCLAW_MODEL=qwen/qwen3.5-flash \
OPENCLAW_MODEL_MAX_IMAGES=2 \
OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=10 \
OPENCLAW_MODEL_FAST_MODE=qwen_text_only \
OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=1 \
OPENCLAW_QWEN_API_RETRIES=1 \
OPENCLAW_QWEN_API_RETRY_BACKOFF_S=2 \
OPENCLAW_AGENT_TIMEOUT=180 \
OPENCLAW_AGENT_MAX_INPUT_TOKENS=50000 \
PORT=8011 \
bash scripts/start_openclaw_cli_plan_gateway.sh
```

Preflight:

```bash
curl -s http://127.0.0.1:8011/health
```

Expected:

```text
ok=true
planner_mode=model
```

## One-Episode Smoke

Run one sample first:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=20411 \
OUTPUT_PATH=results/clawnav_openclaw_qwen_image_gated_smoke_20260526 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=480 \
HARNESS_DEBUG_MAX_EPISODES=1 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Trace check:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("results/clawnav_openclaw_qwen_image_gated_smoke_20260526/harness_traces/harness_trace_rank0.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
audits = [row.get("context_audit") or {} for row in rows]
fast = [audit for audit in audits if audit.get("planner_step_mode") == "fast_text"]
visual = [audit for audit in audits if audit.get("planner_step_mode") == "visual_update"]

print("trace_rows", len(rows))
print("planner_authority", sorted({audit.get("planner_authority") for audit in audits}))
print("qwen_api_called", sorted({audit.get("qwen_api_called") for audit in audits}))
print("planner_step_modes", sorted({audit.get("planner_step_mode") for audit in audits}))
print("fast_text_image_counts", sorted({audit.get("model_image_count") for audit in fast}))
print("visual_update_image_counts", sorted({audit.get("model_image_count") for audit in visual}))
print("fallback_steps", sum(1 for row in rows if row.get("fallback")))
print("planner_fallback_steps", sum(1 for row in rows if row.get("planner_fallback")))
print(
    "cli_fallback_reason_steps",
    sum(
        1
        for row in rows
        if str(row.get("planner_reason") or row.get("reason") or "").startswith(
            ("openclaw_cli_model_fallback:", "openclaw_cli_agent_fallback:")
        )
    ),
)
print("planner_errors", [row.get("planner_error") for row in rows if row.get("planner_error")])

assert rows
assert {audit.get("planner_authority") for audit in audits} == {"qwen"}
assert {audit.get("qwen_api_called") for audit in audits} == {True}
assert fast, "expected at least one fast_text step"
assert visual, "expected at least one visual_update step"
assert all(audit.get("model_image_count") == 0 for audit in fast)
assert all((audit.get("model_image_count") or 0) <= 2 for audit in visual)
assert sum(1 for row in rows if row.get("fallback")) == 0
assert sum(1 for row in rows if row.get("planner_fallback")) == 0
assert not any(
    str(row.get("planner_reason") or row.get("reason") or "").startswith(
        ("openclaw_cli_model_fallback:", "openclaw_cli_agent_fallback:")
    )
    for row in rows
)
PY
```

## 30-Sample Test

Run only after the smoke passes:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=20412 \
OUTPUT_PATH=results/clawnav_openclaw_qwen_image_gated_30_20260526 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=480 \
HARNESS_DEBUG_MAX_EPISODES=30 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Use the same trace check with:

```text
results/clawnav_openclaw_qwen_image_gated_30_20260526/harness_traces/harness_trace_rank0.jsonl
```

## Validity Gates

The run is valid only if:

```text
model_provider=qwen_api
openclaw_session_mode=stateless_model
planner_authority=qwen
qwen_api_called=true
fallback_steps=0
planner_fallback_steps=0
no `openclaw_cli_model_fallback:` or `openclaw_cli_agent_fallback:` planner reason
fast_text model_image_count always 0
visual_update model_image_count <= 2
```

For `memory_guided_policy_fast`, replace the qwen-only authority gate with:

```text
visual_update rows: planner_authority=qwen and qwen_api_called=true
fast_text rows: planner_authority=local_policy, qwen_api_called=false, model_call_skipped=true
fallback_steps=0
planner_fallback_steps=0
no `openclaw_cli_model_fallback:` or `openclaw_cli_agent_fallback:` planner reason
```

Compare against:

```text
results/clawnav_openclaw_gateway_30_direct_qwen_api_20260526
```

Primary comparison fields:

```text
episode wall time
success
spl
steps
planner_step_mode distribution
model_image_count distribution
provider_input_tokens
```
