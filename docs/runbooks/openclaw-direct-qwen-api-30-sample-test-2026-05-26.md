# OpenClaw Direct Qwen API 30-Sample Test - 2026-05-26

This runbook is the recommended 30-sample test path after the direct-Qwen smoke
verified that the ClawNav `/plan` adapter is using stateless Qwen API calls.

Current verified smoke evidence from:

```text
results/clawnav_openclaw_gateway_1_direct_qwen_api_smoke_20260526/harness_traces/harness_trace_rank0.jsonl
```

was:

```text
trace_rows 16
providers ['qwen_api']
session_modes ['stateless_model']
session_ids ["''"]
max_images 3
fallback_steps 0
```

That is enough to start the 30-sample test. The 1-episode smoke was stopped
after confirming the path, so it is path-validation evidence, not a completed
accuracy result.

## Method

This test uses:

- JanusVLN/Habitat for environment rollout.
- ClawNav `openclaw_bridge` runtime.
- ClawNav `/plan` adapter at `http://127.0.0.1:8011`.
- `OPENCLAW_PLANNER_MODE=model`.
- `OPENCLAW_MODEL_PROVIDER=qwen_api`.
- Direct image attachments to Qwen, up to `OPENCLAW_MODEL_MAX_IMAGES=3`.
- Stateless per-step model calls with no OpenClaw agent session history.

Expected trace identity:

```text
model_provider=qwen_api
openclaw_session_mode=stateless_model
openclaw_session_id=""
planner_fallback=false
model_image_count>0 on visual steps
```

This is not the older `OPENCLAW_PLANNER_MODE=agent` describe path and not the
old per-step `openclaw capability model run --json` subprocess path.

## Before Running

Use a normal shell on the machine, not a restricted sandbox, because the adapter
needs access to OpenClaw auth/config files and the evaluation needs CUDA.

Check GPU pressure first:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

Pick a GPU with enough free memory for JanusVLN/Habitat. Qwen itself is API
backed here; GPU memory pressure comes from the local navigation model and
simulator, not from Qwen.

Check ports:

```bash
ss -ltnp | rg ':8011|:8022'
```

Port `8011` should be free before starting the adapter. `8022` is only relevant
if you intentionally use a real spatial memory service; this 30-sample command
uses the default fake/episode-local memory path from
`scripts/evaluation_openclaw_gateway.sh`.

## Terminal 1: Start the Adapter

```bash
cd /ssd/dingmuhe/Embodied-task/Navigation_Claw/ClawNav

openclaw gateway restart
openclaw gateway call health --json --timeout 90000

HOST=127.0.0.1 \
PORT=8011 \
OPENCLAW_PLANNER_MODE=model \
OPENCLAW_MODEL=qwen/qwen3.5-flash \
OPENCLAW_MODEL_PROVIDER=qwen_api \
OPENCLAW_MODEL_MAX_IMAGES=3 \
OPENCLAW_VISUAL_MODE=path \
OPENCLAW_AGENT_TIMEOUT=180 \
OPENCLAW_AGENT_MAX_INPUT_TOKENS=50000 \
./scripts/start_openclaw_cli_plan_gateway.sh
```

Keep this terminal running. It should print:

```text
OpenClaw CLI /plan adapter listening on http://127.0.0.1:8011
```

## Terminal 2: Optional Preflight

Run this before the 30-sample test:

```bash
cd /ssd/dingmuhe/Embodied-task/Navigation_Claw/ClawNav

PYTHONPATH=.:src /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python \
  scripts/check_openclaw_plan_gateway.py \
  --gateway_url http://127.0.0.1:8011 \
  --timeout 180 \
  --require_service openclaw_cli_plan_gateway
```

Healthy output should include a JSON response with:

```text
"model_provider": "qwen_api"
"openclaw_session_mode": "stateless_model"
"openclaw_session_id": ""
```

The preflight has no real navigation image, so `model_image_count=0` there is
normal. Image attachment is checked from the evaluation trace.

## Terminal 2: Run 30 Samples

Use a fresh output path for the 30-sample run:

```bash
cd /ssd/dingmuhe/Embodied-task/Navigation_Claw/ClawNav

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=20403 \
OUTPUT_PATH=results/clawnav_openclaw_gateway_30_direct_qwen_api_20260526 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=180 \
HARNESS_DEBUG_MAX_EPISODES=30 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Adjust only these values when needed:

- `CUDA_VISIBLE_DEVICES`: choose a free GPU from `nvidia-smi`.
- `MASTER_PORT`: change if another torchrun is already using `20403`.
- `OUTPUT_PATH`: change if you intentionally want a new run directory.

Do not remove `MAX_STEPS=400`. A stale `MAX_STEPS=0` makes episodes stop at
step 0 and creates invalid all-one-step results.

`HARNESS_DEBUG_MAX_EPISODES=30` is the sample count. The script uses its fixed
episode-key list and truncates to the first 30 selected episodes.

## Trace Check While Running

After several steps have written trace rows, use:

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("results/clawnav_openclaw_gateway_30_direct_qwen_api_20260526")
trace = run / "harness_traces" / "harness_trace_rank0.jsonl"
rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
audits = [row.get("context_audit") or {} for row in rows]

print("trace_rows", len(rows))
print("providers", sorted(set(a.get("model_provider") for a in audits)))
print("session_modes", sorted(set(a.get("openclaw_session_mode") for a in audits)))
print("session_ids", sorted(set(repr(a.get("openclaw_session_id")) for a in audits)))
print("max_images", max((a.get("model_image_count") or 0) for a in audits) if audits else 0)
print("max_provider_input_tokens", max((a.get("provider_input_tokens") or 0) for a in audits) if audits else 0)
print("fallback_steps", sum(1 for row in rows if row.get("planner_fallback")))
print("last_step", rows[-1].get("step_id") if rows else None)
PY
```

Expected:

```text
providers ['qwen_api']
session_modes ['stateless_model']
session_ids ["''"]
max_images 1-3
fallback_steps 0
```

If `fallback_steps` is nonzero, inspect the failed rows before continuing. The
usual causes are Qwen API auth/network errors, malformed model output, or a
gateway timeout.

## After the Run

Summarize:

```bash
python scripts/summarize_openclaw_vln_ablation.py \
  results/clawnav_openclaw_gateway_30_direct_qwen_api_20260526
```

Then check the trace again with the same script above.

The run is valid for this method if:

- `invalid_run_reason` is empty or absent.
- It did not end with all episodes at one step.
- `providers ['qwen_api']`.
- `session_modes ['stateless_model']`.
- `session_ids ["''"]`.
- `fallback_steps 0`.
- `max_images` is greater than 0.
- `oracle_leakage_steps=0` in the summary.

The run is a direct-Qwen planner/gateway test. It should not be reported as a
real OpenClaw visual-memory write/recall experiment unless trace rows also show
real memory writes and recall usage.

## Runtime Expectation

Direct Qwen image calls still happen once per navigation step. The previous
direct-image path could take roughly tens of seconds per navigation step, so a
full 30-sample run can be hours-scale depending on Qwen API latency and episode
length.

If the run is too slow but the trace identity is correct, reduce only for a
diagnostic run:

```bash
HARNESS_DEBUG_MAX_EPISODES=3
```

Do not use a reduced sample count as the final 30-sample result.

## Rollback Modes

Older OpenClaw CLI model subprocess path:

```bash
OPENCLAW_MODEL_PROVIDER=openclaw_cli ./scripts/start_openclaw_cli_plan_gateway.sh
```

Older agent describe path:

```bash
OPENCLAW_PLANNER_MODE=agent \
OPENCLAW_VISUAL_MODE=describe \
OPENCLAW_VISUAL_MODEL=qwen/qwen3.5-flash \
./scripts/start_openclaw_cli_plan_gateway.sh
```
