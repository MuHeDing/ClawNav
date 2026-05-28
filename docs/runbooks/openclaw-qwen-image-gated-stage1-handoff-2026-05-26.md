# OpenClaw Qwen Image-Gated Stage 1 Handoff

Date: 2026-05-26

This handoff is for continuing on 2026-05-27.

## Current Status

Stage 1 is implemented, and the trace gate now treats both
`openclaw_cli_agent_fallback:` and `openclaw_cli_model_fallback:` as hard
planner fallback. A separate hybrid fast path,
`memory_guided_policy_fast`, is also implemented and passed a 1-episode smoke.

Implemented scope:

- `visual_update` / `fast_text` step modes.
- `visual_update` sends current image plus newest keyframe, capped by `OPENCLAW_MODEL_MAX_IMAGES`.
- `fast_text` sends no image and uses adapter-local cached visual memory.
- Every non-fallback planner decision still uses Qwen API.
- Trace audit records `planner_step_mode`, `planner_authority`, `qwen_api_called`, `model_image_count`, `visual_memory_age_steps`, `fast_break_reason`, and `visual_memory_update_status`.
- `memory_guided_policy_fast` skips Qwen on cached `fast_text` steps and sends
  cached visual memory to the local `NavigationPolicySkill`.

Not implemented:

- Stage 2 full text compression.
- 30-sample validation for the hybrid `memory_guided_policy_fast` method.

## Relevant Docs

Plan:

```text
docs/plans/2026-05-26-openclaw-qwen-image-gated-memory-planner.md
```

Stage 1 eval runbook:

```text
docs/runbooks/openclaw-qwen-image-gated-memory-planner-eval.md
```

This handoff:

```text
docs/runbooks/openclaw-qwen-image-gated-stage1-handoff-2026-05-26.md
```

## Relevant Code

Main implementation:

```text
src/harness/openclaw/openclaw_cli_plan_gateway.py
```

Start script:

```text
scripts/start_openclaw_cli_plan_gateway.sh
```

Tests:

```text
tests/test_openclaw_cli_plan_gateway.py
tests/test_evaluation_scripts.py
```

## Verification Already Run

Unit/static checks passed before smoke:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_cli_plan_gateway.py tests/test_evaluation_scripts.py::test_openclaw_cli_plan_gateway_start_script_uses_adapter_module -q
```

Result:

```text
37 passed in 0.11s
```

Also passed:

```bash
PYTHONPATH=src python -m py_compile src/harness/openclaw/openclaw_cli_plan_gateway.py
bash -n scripts/start_openclaw_cli_plan_gateway.sh
git diff --check -- src/harness/openclaw/openclaw_cli_plan_gateway.py scripts/start_openclaw_cli_plan_gateway.sh tests/test_openclaw_cli_plan_gateway.py tests/test_evaluation_scripts.py docs/plans/2026-05-26-openclaw-qwen-image-gated-memory-planner.md docs/runbooks/openclaw-qwen-image-gated-memory-planner-eval.md
```

Note: full `tests/test_evaluation_scripts.py` still has an unrelated missing-file failure for `scripts/evaluation_lowmem_sparse.sh`.

Additional gate-fix verification on 2026-05-27:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_runtime_bridge.py tests/test_openclaw_cli_plan_gateway.py tests/test_evaluation_scripts.py::test_openclaw_cli_plan_gateway_start_script_uses_adapter_module -q
```

Result:

```text
71 passed in 0.19s
```

Also passed:

```bash
PYTHONPATH=src python -m py_compile src/harness/openclaw/runtime.py src/harness/openclaw/openclaw_cli_plan_gateway.py
bash -n scripts/start_openclaw_cli_plan_gateway.sh
git diff --check -- src/harness/openclaw/runtime.py src/harness/openclaw/openclaw_cli_plan_gateway.py tests/test_openclaw_runtime_bridge.py tests/test_openclaw_cli_plan_gateway.py docs/runbooks/openclaw-qwen-image-gated-memory-planner-eval.md docs/runbooks/openclaw-qwen-image-gated-stage1-handoff-2026-05-26.md docs/plans/2026-05-26-openclaw-qwen-image-gated-memory-planner.md
```

Additional hybrid fast-path verification on 2026-05-27:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_cli_plan_gateway.py tests/test_openclaw_runtime_bridge.py tests/test_evaluation_scripts.py::test_openclaw_cli_plan_gateway_start_script_uses_adapter_module -q
```

Result:

```text
74 passed in 0.23s
```

This includes:

- `memory_guided_policy_fast` skips Qwen on cached fast steps.
- `QwenApiModelClient` retries DashScope read timeout when configured.
- successful retry metadata is recorded as `qwen_api_request_attempts` and
  `qwen_api_retry_count` in `context_audit`.

## Smoke Attempt

Command used:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=20411 \
OUTPUT_PATH=results/clawnav_openclaw_qwen_image_gated_smoke_20260526 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=180 \
HARNESS_DEBUG_MAX_EPISODES=1 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Result directory:

```text
results/clawnav_openclaw_qwen_image_gated_smoke_20260526
```

Trace:

```text
results/clawnav_openclaw_qwen_image_gated_smoke_20260526/harness_traces/harness_trace_rank0.jsonl
```

Observed summary:

```text
rows 14
step_range (0, 13)
step_modes {'visual_update': 3, 'fast_text': 10, None: 1}
image_counts {1: 1, 0: 10, 2: 2, None: 1}
fast_break {'initial_step': 1, 'cached_visual_memory': 10, 'image_interval': 1, 'keyframe_candidate': 1, None: 1}
providers ['qwen_api']
session_modes ['stateless_model']
planner_authority ['qwen']
qwen_api_called [True]
fallback_top_level 0
planner_fallback 1
planner_errors ["HTTPConnectionPool(host='127.0.0.1', port=8011): Read timed out. (read timeout=180.0)"]
result_exists False
summary_exists False
```

Interpretation:

- Image gating works: most steps are `fast_text` with `model_image_count=0`.
- Step 8 correctly triggered `visual_update` by interval.
- Step 10 correctly triggered `visual_update` by keyframe candidate.
- The run is not valid because step 11 fell back to the rule planner after a 180s `/plan` timeout.
- There is no `result.json` or `summary.json` because the smoke was stopped after the invalid planner fallback was found.

## Gate-Fix Smoke Attempt On 2026-05-27

Command shape:

```bash
OPENCLAW_PLANNER_MODE=model \
OPENCLAW_MODEL_PROVIDER=qwen_api \
OPENCLAW_MODEL=qwen/qwen3.5-flash \
OPENCLAW_MODEL_MAX_IMAGES=2 \
OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=10 \
OPENCLAW_MODEL_FAST_MODE=qwen_text_only \
OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=1 \
OPENCLAW_AGENT_TIMEOUT=90 \
OPENCLAW_AGENT_MAX_INPUT_TOKENS=50000 \
OPENCLAW_GATEWAY_TIMEOUT=30 \
PORT=8011 \
bash scripts/start_openclaw_cli_plan_gateway.sh
```

```bash
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
MASTER_PORT=20411 \
OUTPUT_PATH=results/clawnav_openclaw_qwen_image_gated_smoke_20260527_201202_gatefix_gpu2 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=120 \
HARNESS_DEBUG_MAX_EPISODES=1 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Result directory:

```text
results/clawnav_openclaw_qwen_image_gated_smoke_20260527_201202_gatefix_gpu2
```

Observed summary:

```text
rows 5
audited_rows 5
missing_audit_rows 0
step_range (0, 4)
step_modes {'visual_update': 1, 'fast_text': 4}
image_counts {1: 1, 0: 4}
fast_break {'initial_step': 1, 'cached_visual_memory': 4}
providers ['qwen_api']
session_modes ['stateless_model']
planner_authority ['qwen']
qwen_api_called [True]
fallback_top_level 1
planner_fallback 1
cli_fallback_reason_steps 1
planner_errors ["HTTPSConnectionPool(host='dashscope.aliyuncs.com', port=443): Read timed out. (read timeout=90.0)"]
result_json {"success": 0.0, "spl": 0.0, "steps": 5}
```

Interpretation:

- The gate fix works: the hidden `openclaw_cli_model_fallback:` is now counted
  as `planner_model_fallback=true` and `planner_fallback=true`.
- The fallback row preserved `context_audit`; it was `fast_text`,
  `qwen_api_called=true`, and `model_image_count=0`.
- The run is still invalid, so do not run the 30-sample evaluation from this
  state.

## Memory-Guided Fast Smoke On 2026-05-27

This is not the same method as Stage 1 qwen-only. It is the hybrid fast path:

```text
visual_update: Qwen API, images allowed
fast_text: no Qwen API call, local NavigationPolicySkill with cached visual memory
method label: memory_guided_policy_fast
```

Adapter command shape:

```bash
OPENCLAW_PLANNER_MODE=model \
OPENCLAW_MODEL_PROVIDER=qwen_api \
OPENCLAW_MODEL=qwen/qwen3.5-flash \
OPENCLAW_MODEL_MAX_IMAGES=2 \
OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=10 \
OPENCLAW_MODEL_FAST_MODE=memory_guided_policy_fast \
OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=1 \
OPENCLAW_QWEN_API_RETRIES=1 \
OPENCLAW_QWEN_API_RETRY_BACKOFF_S=2 \
OPENCLAW_AGENT_TIMEOUT=90 \
OPENCLAW_AGENT_MAX_INPUT_TOKENS=50000 \
OPENCLAW_GATEWAY_TIMEOUT=30 \
PORT=8011 \
bash scripts/start_openclaw_cli_plan_gateway.sh
```

Smoke command:

```bash
NO_PROXY=127.0.0.1,localhost \
no_proxy=127.0.0.1,localhost \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 \
MASTER_PORT=20411 \
OUTPUT_PATH=results/clawnav_openclaw_qwen_memory_guided_fast_smoke_20260527_gpu2 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=120 \
HARNESS_DEBUG_MAX_EPISODES=1 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Result directory:

```text
results/clawnav_openclaw_qwen_memory_guided_fast_smoke_20260527_gpu2
```

Observed summary:

```text
rows 23
audited_rows 23
missing_audit_rows 0
step_range (0, 22)
step_modes {'visual_update': 3, 'fast_text': 20}
planner_authority {'qwen': 3, 'local_policy': 20}
qwen_api_called {True: 3, False: 20}
model_call_skipped 20
image_counts {1: 1, 0: 20, 2: 2}
fast_policy_modes {'memory_guided_policy_fast': 20}
fallback_top_level 0
planner_fallback 0
cli_fallback_reason_steps 0
planner_errors []
result {'success': 1.0, 'spl': 1.0, 'os': 1.0, 'ne': 1.6131069660186768, 'steps': 23}
```

Interpretation:

- The DashScope timeout was avoided by removing Qwen calls from cached fast
  steps, not by retrying failed Qwen requests.
- This run passed the hybrid trace gate and the episode succeeded.
- This should be compared as `memory_guided_policy_fast`, not as Stage 1
  `planner_authority=qwen` on every step.

## Retry + 30-Sample Attempt On 2026-05-27

After the successful hybrid smoke, a 30-sample attempt was started on GPU 2:

```text
OUTPUT_PATH=results/clawnav_openclaw_qwen_memory_guided_fast_30_20260527_gpu2
OPENCLAW_MODEL_FAST_MODE=memory_guided_policy_fast
```

It did not produce a result directory. The attempt failed during model loading
because GPU 2 had become occupied by another process:

```text
torch.OutOfMemoryError: CUDA out of memory
Process 230018 has 31.43 GiB memory in use
```

Before retry/backoff was added, that 30-sample attempt also showed that an
initial `visual_update` Qwen call can still hit DashScope timeout. That is why
`QwenApiModelClient` now supports:

```text
OPENCLAW_QWEN_API_RETRIES=1
OPENCLAW_QWEN_API_RETRY_BACKOFF_S=2
```

A retry-enabled 1-episode smoke was then run on GPU 6:

```text
results/clawnav_openclaw_qwen_memory_guided_fast_retry_smoke_20260527_gpu6
```

That run was planner-valid until step 3 but failed due shared-GPU CUDA OOM in
`NavigationPolicySkill`, not due Qwen or planner fallback:

```text
step_modes {'visual_update': 1, 'fast_text': 3}
planner_authority {'qwen': 1, 'local_policy': 3}
qwen_api_called {True: 1, False: 3}
fallback_top_level 1
planner_fallback 0
runtime_errors ['openclaw_runtime_error']
error: NavigationPolicySkill failed: CUDA out of memory
```

So the current valid end-to-end smoke remains:

```text
results/clawnav_openclaw_qwen_memory_guided_fast_smoke_20260527_gpu2
success=1.0, spl=1.0, steps=23, fallback=0, planner_fallback=0
```

## Important Runbook Fix

The local OpenClaw CLI does not support:

```bash
openclaw gateway start --stdio
```

Confirmed with:

```bash
openclaw gateway --help
```

Use one of:

```bash
openclaw gateway status
openclaw gateway run
openclaw gateway start
```

The Stage 1 eval runbook has been corrected to use `openclaw gateway status` and `openclaw gateway run`.

## Current Process State

After the smoke attempt:

```text
port 8011: not listening
port 20411: not listening
```

The Stage 1 adapter and smoke process were stopped.

## 30-Sample Status

The qwen-only Stage 1 30-sample test should still not be run from the timed-out
`qwen_text_only` smoke, because that method produced `planner_fallback=1`.

The hybrid `memory_guided_policy_fast` method is eligible for a 30-sample test
once a GPU with enough free memory is available. Keep it separate from qwen-only
Stage 1 results.

## Recommended Next Step For 2026-05-27

Run a 30-sample test for the hybrid `memory_guided_policy_fast` method on a GPU
with enough free memory. Keep `OPENCLAW_QWEN_API_RETRIES=1` enabled.

### Step 1: Use The Adapter Timeout Budget

Do not use the earlier `OPENCLAW_GATEWAY_TIMEOUT=120` debug setting for valid
smoke or 30-sample runs. With `OPENCLAW_QWEN_API_RETRIES=1`, a Qwen visual
update can outlive a 120s harness HTTP read timeout even when the adapter later
returns structured audit metadata. That produces `planner_fallback=1` and
invalidates the run.

The adapter now reports `/health.timeout_budget.recommended_gateway_timeout_s`,
and `scripts/evaluation_openclaw_gateway.sh` enforces it by default. For the
`OPENCLAW_AGENT_TIMEOUT=90` hybrid setting, use at least 300s on the harness
side unless `/health` recommends a larger value.

Suggested adapter:

```bash
OPENCLAW_PLANNER_MODE=model \
OPENCLAW_MODEL_PROVIDER=qwen_api \
OPENCLAW_MODEL=qwen/qwen3.5-flash \
OPENCLAW_MODEL_MAX_IMAGES=2 \
OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=10 \
OPENCLAW_MODEL_FAST_MODE=qwen_text_only \
OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=1 \
OPENCLAW_AGENT_TIMEOUT=90 \
OPENCLAW_AGENT_MAX_INPUT_TOKENS=50000 \
PORT=8011 \
bash scripts/start_openclaw_cli_plan_gateway.sh
```

Suggested smoke:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=20411 \
OUTPUT_PATH=results/clawnav_openclaw_qwen_image_gated_smoke_timeout_debug_20260527 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=300 \
HARNESS_DEBUG_MAX_EPISODES=1 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

### Step 2: If Returning To Qwen-Only Stage 1

Rerun the qwen-only 1-episode smoke with `OPENCLAW_QWEN_API_RETRIES=1` before
any qwen-only 30-sample evaluation. The timed-out qwen-only smoke remains
invalid.

## Trace Check Snippet

Use this after any smoke:

```bash
python - <<'PY'
import json
from pathlib import Path
from collections import Counter

path = Path("results/clawnav_openclaw_qwen_image_gated_smoke_20260526/harness_traces/harness_trace_rank0.jsonl")
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
audits = [row.get("context_audit") or {} for row in rows]
fast = [audit for audit in audits if audit.get("planner_step_mode") == "fast_text"]
visual = [audit for audit in audits if audit.get("planner_step_mode") == "visual_update"]

print("rows", len(rows))
print("step_modes", dict(Counter(audit.get("planner_step_mode") for audit in audits)))
print("image_counts", dict(Counter(audit.get("model_image_count") for audit in audits)))
print("providers", sorted({audit.get("model_provider") for audit in audits if audit.get("model_provider")}))
print("session_modes", sorted({audit.get("openclaw_session_mode") for audit in audits if audit.get("openclaw_session_mode")}))
print("planner_authority", sorted({audit.get("planner_authority") for audit in audits if audit.get("planner_authority")}))
print("qwen_api_called", sorted({audit.get("qwen_api_called") for audit in audits if audit.get("qwen_api_called") is not None}))
print("fallback_top_level", sum(1 for row in rows if row.get("fallback")))
print("planner_fallback", sum(1 for row in rows if row.get("planner_fallback")))
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
assert {audit.get("planner_authority") for audit in audits if audit.get("planner_authority")} == {"qwen"}
assert {audit.get("qwen_api_called") for audit in audits if audit.get("qwen_api_called") is not None} == {True}
assert fast
assert visual
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
