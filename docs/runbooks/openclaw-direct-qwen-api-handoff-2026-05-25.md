# OpenClaw Direct Qwen API Handoff - 2026-05-25

This handoff records the current state for tomorrow's ClawNav/OpenClaw test.
The key change is that `planner_mode=model` now avoids starting an OpenClaw CLI
subprocess on every navigation step.

## Current Status

- The previous slow path was:
  - ClawNav `/plan` adapter receives one navigation step.
  - Adapter starts `openclaw capability model run --json`.
  - OpenClaw CLI calls cloud Qwen with 1-3 images plus compact text.
  - This repeated every step and caused about 25-30 seconds per step in the
    interrupted 30-episode run.
- The new intended path is:
  - ClawNav `/plan` adapter stays running as one Python process.
  - For each step, it calls cloud Qwen API directly through `QwenApiModelClient`.
  - No OpenClaw agent session is created.
  - No per-step OpenClaw CLI model subprocess is created.
  - Images and text are still stateless per step and do not accumulate.

## Main Files Changed

- `src/harness/openclaw/openclaw_cli_plan_gateway.py`
  - Added `QwenApiModelClient`.
  - Added `openclaw_model_provider`.
  - Default provider is `qwen_api`.
  - Compatibility provider is `openclaw_cli`.
  - Direct client reads API key from environment or OpenClaw auth store:
    `~/.openclaw/agents/main/agent/auth-profiles.json`.
- `scripts/start_openclaw_cli_plan_gateway.sh`
  - Added `OPENCLAW_MODEL`, `OPENCLAW_MODEL_PROVIDER`,
    `OPENCLAW_MODEL_MAX_IMAGES`.
  - Default `OPENCLAW_MODEL_PROVIDER=qwen_api`.
- `docs/runbooks/openclaw-visual-gateway-30-episode-eval.md`
  - Updated the recommended 30-episode workflow to direct `qwen_api`.
- `tests/test_openclaw_cli_plan_gateway.py`
  - Added coverage proving direct Qwen API mode does not call OpenClaw CLI.
  - Added coverage for reading OpenClaw auth store.

Related untracked files from the context-stress work:

- `docs/runbooks/openclaw-context-engine-stress-test.md`
- `scripts/stress_openclaw_context_engine.py`
- `scripts/summarize_openclaw_context_audit.py`
- `tests/test_openclaw_context_audit_summary.py`
- `tests/test_openclaw_context_stress_script.py`

## Verified Today

Unit and syntax checks:

```bash
PYTHONPATH=.:src pytest \
  tests/test_openclaw_cli_plan_gateway.py \
  tests/test_openclaw_context_stress_script.py \
  tests/test_openclaw_context_audit_summary.py \
  tests/test_evaluation_scripts.py::test_openclaw_cli_plan_gateway_start_script_uses_adapter_module \
  -q
```

Result:

```text
40 passed in 0.20s
```

Additional checks:

```bash
bash -n scripts/start_openclaw_cli_plan_gateway.sh scripts/evaluation_openclaw_gateway.sh
PYTHONPATH=src python -m py_compile \
  src/harness/openclaw/openclaw_cli_plan_gateway.py \
  scripts/stress_openclaw_context_engine.py
```

Both passed.

Live Qwen API smoke without image:

```text
intent= act
action= MOVE_FORWARD
planner_error= None
model_provider= qwen_api
session_mode= stateless_model
session_id= ''
provider_input_tokens= 155
assembled_prompt_tokens= 143
model_image_count= 0
```

Live Qwen API smoke with one real image:

```text
intent= act
action= MOVE_FORWARD
planner_error= None
model_provider= qwen_api
session_mode= stateless_model
provider_input_tokens= 523
assembled_prompt_tokens= 182
model_image_count= 1
```

## Important Semantics

- `OPENCLAW_PLANNER_MODE=model` means compact stateless model planner mode.
- `OPENCLAW_MODEL_PROVIDER=qwen_api` means direct cloud Qwen API from Python.
- `OPENCLAW_MODEL_PROVIDER=openclaw_cli` means the older compatibility path:
  `openclaw capability model run --json`.
- Direct `qwen_api` still makes one Qwen API request per step, but it no longer
  starts one OpenClaw CLI subprocess per step.
- Direct `qwen_api` does not create an OpenClaw agent session/window.
- Trace should show:
  - `context_profile=plan_model`
  - `openclaw_session_mode=stateless_model`
  - `openclaw_session_id=""`
  - `model_provider=qwen_api`
  - `model_image_count > 0` when images exist

## Tomorrow Test Plan

Start the adapter:

```bash
cd /ssd/dingmuhe/Embodied-task/Navigation_Claw/ClawNav

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

Run a 1-episode smoke first:

```bash
CUDA_VISIBLE_DEVICES=4 \
MASTER_PORT=20402 \
OUTPUT_PATH=results/clawnav_openclaw_gateway_1_direct_qwen_api_smoke_20260526 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=180 \
HARNESS_DEBUG_MAX_EPISODES=1 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

If the smoke is healthy, run 30 episodes:

```bash
CUDA_VISIBLE_DEVICES=4 \
MASTER_PORT=20403 \
OUTPUT_PATH=results/clawnav_openclaw_gateway_30_direct_qwen_api_20260526 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=180 \
HARNESS_DEBUG_MAX_EPISODES=30 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Summarize after the run:

```bash
python scripts/summarize_openclaw_vln_ablation.py \
  results/clawnav_openclaw_gateway_30_direct_qwen_api_20260526
```

## Quick Trace Checks

Use this after smoke or full run:

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("results/clawnav_openclaw_gateway_1_direct_qwen_api_smoke_20260526")
trace = run / "harness_traces" / "harness_trace_rank0.jsonl"
rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
audits = [row.get("context_audit") or {} for row in rows]
print("trace_rows", len(rows))
print("providers", sorted(set(a.get("model_provider") for a in audits)))
print("session_modes", sorted(set(a.get("openclaw_session_mode") for a in audits)))
print("session_ids", sorted(set(repr(a.get("openclaw_session_id")) for a in audits)))
print("max_images", max((a.get("model_image_count") or 0) for a in audits))
print("max_provider_input_tokens", max((a.get("provider_input_tokens") or 0) for a in audits))
print("fallback_steps", sum(1 for row in rows if row.get("planner_fallback")))
PY
```

Expected:

- `providers ['qwen_api']`
- `session_modes ['stateless_model']`
- `session_ids ["''"]`
- `max_images` should be 1-3 after real image files exist
- `fallback_steps 0`

## Known Risks

- Qwen image API latency may still dominate runtime even without OpenClaw CLI
  subprocess overhead. The one-image smoke took only a few seconds, but full
  navigation can vary by network/API latency.
- `provider_input_tokens` includes Qwen-side image accounting when the provider
  reports usage. It will be larger than `assembled_prompt_tokens`.
- GPU memory pressure should come from JanusVLN/Habitat, not Qwen API. Check
  `nvidia-smi` before choosing `CUDA_VISIBLE_DEVICES`.
- If `planner_error` appears in fallback decisions, inspect the exact error
  before rerunning. Common causes are API auth, network, or malformed model
  output.

## Rollback

To force the older OpenClaw CLI compatibility model path:

```bash
OPENCLAW_MODEL_PROVIDER=openclaw_cli ./scripts/start_openclaw_cli_plan_gateway.sh
```

To force the older agent describe path:

```bash
OPENCLAW_PLANNER_MODE=agent \
OPENCLAW_VISUAL_MODE=describe \
OPENCLAW_VISUAL_MODEL=qwen/qwen3.5-flash \
./scripts/start_openclaw_cli_plan_gateway.sh
```
