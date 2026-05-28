# OpenClaw Visual Gateway 30-Episode Evaluation

This runbook stores the exact two-terminal workflow for running a 30-episode
OpenClaw visual gateway diagnostic/evaluation from ClawNav.

Current preferred mode is `planner_mode=model` with
`OPENCLAW_MODEL_PROVIDER=qwen_api`: ClawNav sends compact planner text plus real
image files directly from the Python `/plan` adapter to the cloud Qwen API. This
keeps each step stateless without starting an OpenClaw CLI subprocess for every
step. It is different from the older `planner_mode=agent` +
`OPENCLAW_VISUAL_MODE=describe` path, which first converted images into text and
then sent only text to the planner.

The previous bad run under:

```text
results/clawnav_openclaw_gateway_30_visual_maxsteps0_timeout180
```

should not be treated as a valid navigation-accuracy result because every
episode ended after one step. The likely cause was `MAX_STEPS=0`, which makes
`src/evaluation.py` force `STOP` on step 0. The commands below set
`MAX_STEPS=400` explicitly so the shell cannot inherit a stale `MAX_STEPS=0`.

## Terminal 1: Start OpenClaw and the ClawNav Adapter

First verify the OpenClaw internal gateway:

```bash
openclaw gateway restart
openclaw gateway call health --json --timeout 90000
```

Then start the ClawNav `/plan` adapter in stateless model mode with direct image
attachment enabled:

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

Keep this terminal running. It serves `POST /plan` at:

```text
http://127.0.0.1:8011
```

## Terminal 2: Run 30 Episodes

Run the evaluation in a second terminal:

```bash
cd /ssd/dingmuhe/Embodied-task/Navigation_Claw/ClawNav

CUDA_VISIBLE_DEVICES=4 \
MASTER_PORT=20401 \
OUTPUT_PATH=results/clawnav_openclaw_gateway_30_visual_fixed \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=180 \
HARNESS_DEBUG_MAX_EPISODES=30 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

For a fast direct-image smoke before committing to the full 30 episodes, run one
episode first:

```bash
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=20402 \
OUTPUT_PATH=results/clawnav_openclaw_gateway_1_model_images_smoke \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=180 \
HARNESS_DEBUG_MAX_EPISODES=1 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

`MAX_STEPS=400` is intentionally explicit. Do not remove it unless you have
checked that the current shell does not export `MAX_STEPS=0`.
`OPENCLAW_MODEL_MAX_IMAGES=3` sends at most the current frame plus recent
keyframes to Qwen for each model call. Images do not accumulate across steps;
only the current payload's selected image files are attached. If an image path
does not exist on disk it is not attached and appears in
`model_missing_image_paths`.

`OPENCLAW_AGENT_MAX_INPUT_TOKENS=50000` remains as the hard guard used by the
adapter audit path. In direct `qwen_api` model mode, `provider_input_tokens`
comes from the Qwen API response when available. If you explicitly set
`OPENCLAW_MODEL_PROVIDER=openclaw_cli`, the adapter falls back to
`openclaw capability model run --json`, which is useful for compatibility but
slower because it starts a CLI subprocess per step.
The gateway launcher now rejects `MAX_STEPS=0` and other non-positive values
before starting the model, because that configuration forces `STOP` at step 0
and produces a misleading all-zero accuracy run.

## Expected Evidence

After the run, inspect:

```bash
python scripts/summarize_openclaw_vln_ablation.py \
  results/clawnav_openclaw_gateway_30_visual_fixed
```

The visual gateway path is active if the summary/trace shows:

- `invalid_run_reason` empty
- `context_profile=plan_model`
- `openclaw_session_mode=stateless_model`
- `model_image_count` greater than 0 on steps with real image files
- `planner_backend=gateway`
- `planner_fallback=false`
- `oracle_leakage_steps=0`

`visual_analysis_steps` is expected to stay at 0 in direct-image model mode
because images are attached to Qwen directly instead of being pre-described into
text by `OpenClawVisualAnalyzer`.

The run is not exercising visual memory write/recall unless trace records also
contain:

- `memory_writes`
- `recall_usage`
- nonzero `memory_write_attempts`
- nonzero `memory_recall_events`

If all episodes still have `steps=1`, check whether `MAX_STEPS` was inherited as
`0` or whether OpenClaw is returning `STOP` at step 0. The summary script reports
this failure mode as `invalid_run_reason=all_episodes_ended_after_one_step`.

## Legacy Agent Describe Mode

Use this only when you intentionally want the older text-only visual describe
path:

```bash
HOST=127.0.0.1 \
PORT=8011 \
OPENCLAW_PLANNER_MODE=agent \
OPENCLAW_VISUAL_MODE=describe \
OPENCLAW_VISUAL_MODEL=qwen/qwen3.5-flash \
OPENCLAW_VISUAL_MAX_IMAGES=2 \
OPENCLAW_VISUAL_TIMEOUT_MS=90000 \
OPENCLAW_AGENT_TIMEOUT=90 \
OPENCLAW_AGENT_MAX_INPUT_TOKENS=10000 \
./scripts/start_openclaw_cli_plan_gateway.sh
```

## 2026-05-25 Direct-Image Smoke Evidence

Command started:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=20402 \
OUTPUT_PATH=results/clawnav_openclaw_gateway_30_model_images_gpu0_20260525 \
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=180 \
HARNESS_DEBUG_MAX_EPISODES=30 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Observed before manual stop:

- completed episode rows: 1
- completed episode: `2azQ1b91cZZ:11`
- success/SPL/OS: `1.0 / 1.0 / 1.0`
- episode steps: 29
- trace records: 31
- `openclaw_session_mode=stateless_model` on all trace records
- image attachments: `model_image_count=1` on 2 records, `2` on 12 records,
  `3` on 17 records
- max `assembled_prompt_tokens`: 559
- planner fallbacks: 0
- runtime error steps: 0

The run was stopped manually because direct-image Qwen calls were taking about
25-30 seconds per navigation step. A full 30-episode run is therefore expected
to be hours-scale unless the image-call cadence or maximum step count is
reduced.

An earlier attempt on `CUDA_VISIBLE_DEVICES=4` failed with CUDA OOM because that
GPU already had another large Python process resident. OpenClaw's configured
Qwen provider is API-backed (`openai-completions` against DashScope), so do not
attribute GPU memory pressure to the Qwen API call without checking the actual
process list first. Prefer checking GPU memory and process ownership before
choosing the JanusVLN device.
