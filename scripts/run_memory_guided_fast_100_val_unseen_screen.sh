#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
DATA_PATH=${DATA_PATH:-data/datasets/r2r/val_unseen/100_val_unseen.json.gz}
EPISODES=${EPISODES:-100}
GPU=${GPU:-4}
PORT=${PORT:-8011}
MAX_STEPS=${MAX_STEPS:-400}
MODEL_MAX_IMAGES=${MODEL_MAX_IMAGES:-2}
IMAGE_INTERVAL_STEPS=${IMAGE_INTERVAL_STEPS:-20}
QWEN_RETRIES=${QWEN_RETRIES:-1}
QWEN_RETRY_BACKOFF_S=${QWEN_RETRY_BACKOFF_S:-2}
SMOKE_EPISODE_KEY=${SMOKE_EPISODE_KEY:-zsNo4HB9uLZ:1}
AGENT_TIMEOUT_S=${AGENT_TIMEOUT_S:-90}
AGENT_MAX_INPUT_TOKENS=${AGENT_MAX_INPUT_TOKENS:-50000}
ADAPTER_TIMEOUT_S=${ADAPTER_TIMEOUT_S:-30}
GATEWAY_TIMEOUT_S=${GATEWAY_TIMEOUT_S:-300}
MASTER_PORT_SMOKE=${MASTER_PORT_SMOKE:-20601}
MASTER_PORT_EVAL=${MASTER_PORT_EVAL:-20602}
POLL_SECONDS=${POLL_SECONDS:-20}

# Temporary resume target. Reuse the interrupted run directory so evaluation.py
# skips the 14 completed rows in result.json and continues from the 15th episode.
#RESUME_OUTPUT_PATH=${RESUME_OUTPUT_PATH:-results/clawnav_openclaw_qwen_memory_guided_fast_100_val_unseen_20260606_125003}
#OUTPUT_PATH=${OUTPUT_PATH:-${RESUME_OUTPUT_PATH}}

# Original timestamped output naming. Restore this line when starting fresh runs.
OUTPUT_PATH=${OUTPUT_PATH:-results/clawnav_openclaw_qwen_memory_guided_fast_100_val_unseen_${RUN_TIMESTAMP}}
SMOKE_OUTPUT=${SMOKE_OUTPUT:-results/clawnav_openclaw_qwen_memory_guided_fast_100_val_unseen_smoke_${RUN_TIMESTAMP}}
RUN_LOG=${RUN_LOG:-${OUTPUT_PATH}/runner_logs/run_${RUN_TIMESTAMP}.log}

# Stop as soon as runtime/model fallback appears so failed Qwen calls do not
# continue burning token budget across later episodes.
ABORT_ON_INVALID=${ABORT_ON_INVALID:-1}
SKIP_SMOKE=${SKIP_SMOKE:-1}
REUSE_ADAPTER=${REUSE_ADAPTER:-0}

export RUN_TIMESTAMP DATA_PATH EPISODES GPU PORT MAX_STEPS
export MODEL_MAX_IMAGES IMAGE_INTERVAL_STEPS QWEN_RETRIES QWEN_RETRY_BACKOFF_S
export SMOKE_EPISODE_KEY
export AGENT_TIMEOUT_S AGENT_MAX_INPUT_TOKENS ADAPTER_TIMEOUT_S GATEWAY_TIMEOUT_S
export MASTER_PORT_SMOKE MASTER_PORT_EVAL POLL_SECONDS RESUME_OUTPUT_PATH OUTPUT_PATH SMOKE_OUTPUT
export RUN_LOG ABORT_ON_INVALID SKIP_SMOKE REUSE_ADAPTER

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${RUN_LOG}")"

runner_args=(
  --gpu "${GPU}"
  --port "${PORT}"
  --episodes "${EPISODES}"
  --max-steps "${MAX_STEPS}"
  --data-path "${DATA_PATH}"
  --smoke-output "${SMOKE_OUTPUT}"
  --eval-output "${OUTPUT_PATH}"
  --image-interval-steps "${IMAGE_INTERVAL_STEPS}"
  --model-max-images "${MODEL_MAX_IMAGES}"
  --qwen-retries "${QWEN_RETRIES}"
  --qwen-retry-backoff-s "${QWEN_RETRY_BACKOFF_S}"
  --smoke-episode-key "${SMOKE_EPISODE_KEY}"
  --agent-timeout-s "${AGENT_TIMEOUT_S}"
  --agent-max-input-tokens "${AGENT_MAX_INPUT_TOKENS}"
  --adapter-timeout-s "${ADAPTER_TIMEOUT_S}"
  --gateway-timeout-s "${GATEWAY_TIMEOUT_S}"
  --master-port-smoke "${MASTER_PORT_SMOKE}"
  --master-port-eval "${MASTER_PORT_EVAL}"
  --poll-seconds "${POLL_SECONDS}"
)

if [[ "${ABORT_ON_INVALID}" == "0" ]]; then
  runner_args+=(--no-abort-on-invalid)
fi
if [[ "${SKIP_SMOKE}" == "1" ]]; then
  runner_args+=(--skip-smoke)
fi
if [[ "${REUSE_ADAPTER}" == "1" ]]; then
  runner_args+=(--reuse-adapter)
fi

{
  echo "run_timestamp=${RUN_TIMESTAMP}"
  echo "data_path=${DATA_PATH}"
  echo "episodes=${EPISODES}"
  echo "gpu=${GPU}"
  echo "port=${PORT}"
  echo "output_path=${OUTPUT_PATH}"
  echo "smoke_output=${SMOKE_OUTPUT}"
  echo "run_log=${RUN_LOG}"
  echo "smoke_episode_key=${SMOKE_EPISODE_KEY}"
  echo "abort_on_invalid=${ABORT_ON_INVALID}"
  echo "skip_smoke=${SKIP_SMOKE}"
  echo "reuse_adapter=${REUSE_ADAPTER}"
  PYTHONPATH=.:src python scripts/run_memory_guided_fast_large_eval.py "${runner_args[@]}"
} 2>&1 | tee "${RUN_LOG}"
