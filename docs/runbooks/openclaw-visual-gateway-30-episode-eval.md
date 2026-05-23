# OpenClaw Visual Gateway 30-Episode Evaluation

This runbook stores the exact two-terminal workflow for running a 30-episode
OpenClaw visual gateway diagnostic/evaluation from ClawNav.

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

Then start the ClawNav `/plan` adapter with visual describe mode enabled:

```bash
cd /ssd/dingmuhe/Embodied-task/Navigation_Claw/ClawNav

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
OPENCLAW_GATEWAY_TIMEOUT=90 \
HARNESS_DEBUG_MAX_EPISODES=30 \
MAX_STEPS=400 \
REQUIRE_GATEWAY=1 \
./scripts/evaluation_openclaw_gateway.sh
```

`MAX_STEPS=400` is intentionally explicit. Do not remove it unless you have
checked that the current shell does not export `MAX_STEPS=0`.
`OPENCLAW_AGENT_MAX_INPUT_TOKENS=10000` is also intentional. If OpenClaw reports
an agent input token count at or above this value, the adapter returns `STOP`
with `reason=openclaw_agent_input_token_limit` and stops issuing more OpenClaw
agent calls for that adapter process.
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
- `visual_analysis_steps` greater than 0
- `visual_analysis_failures` equal to 0 or acceptably low
- `planner_backend=gateway`
- `planner_fallback=false`
- `oracle_leakage_steps=0`

The run is not exercising visual memory write/recall unless trace records also
contain:

- `memory_writes`
- `recall_usage`
- nonzero `memory_write_attempts`
- nonzero `memory_recall_events`

If all episodes still have `steps=1`, check whether `MAX_STEPS` was inherited as
`0` or whether OpenClaw is returning `STOP` at step 0. The summary script reports
this failure mode as `invalid_run_reason=all_episodes_ended_after_one_step`.
