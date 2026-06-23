#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

MODE=${MODE:-controlled_seed}
if [[ "${MODE}" != "controlled_seed" && "${MODE}" != "no_seed_diagnostic" ]]; then
  echo "MODE must be controlled_seed or no_seed_diagnostic; got '${MODE}'" >&2
  exit 2
fi

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
MODEL_PATH=${MODEL_PATH:-/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra}
DATA_PATH=${DATA_PATH:-data/datasets/r2r/val_unseen/100_val_unseen.json.gz}
EPISODE_KEY=${EPISODE_KEY:-zsNo4HB9uLZ:1}
EPISODE_KEYS=${EPISODE_KEYS:-${EPISODE_KEY}}
GPU=${GPU:-5}
PORT=${PORT:-8013}
HOST=${HOST:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-20615}
MAX_STEPS=${MAX_STEPS:-80}
HARNESS_DEBUG_MAX_EPISODES=${HARNESS_DEBUG_MAX_EPISODES:-1}
GATEWAY_TIMEOUT_S=${GATEWAY_TIMEOUT_S:-300}
REUSE_GATEWAY=${REUSE_GATEWAY:-0}
OUTPUT_PATH=${OUTPUT_PATH:-results/openclaw_visual_readback_runtime_${MODE}_${RUN_TIMESTAMP}}
RUN_LOG=${RUN_LOG:-${OUTPUT_PATH}/runner_logs/runtime_smoke_${RUN_TIMESTAMP}.log}
GATEWAY_LOG=${GATEWAY_LOG:-${OUTPUT_PATH}/runner_logs/gateway_${RUN_TIMESTAMP}.log}
OPENCLAW_VISUAL_READBACK_MODE=${OPENCLAW_VISUAL_READBACK_MODE:-image_read_controller}
OPENCLAW_VISUAL_READBACK_TRIGGER_POLICY=${OPENCLAW_VISUAL_READBACK_TRIGGER_POLICY:-sparse_action_override}

if [[ "${MODE}" == "controlled_seed" ]]; then
  SMOKE_SEED=true
else
  SMOKE_SEED=false
fi

mkdir -p "$(dirname "${RUN_LOG}")"

GATEWAY_PID=""
cleanup() {
  if [[ -n "${GATEWAY_PID}" ]]; then
    kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
    wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${REUSE_GATEWAY}" != "1" ]]; then
  HOST="${HOST}" \
  PORT="${PORT}" \
  OPENCLAW_QWEN_API_RETRIES=${OPENCLAW_QWEN_API_RETRIES:-1} \
  OPENCLAW_QWEN_API_RETRY_BACKOFF_S=${OPENCLAW_QWEN_API_RETRY_BACKOFF_S:-2} \
  OPENCLAW_PLANNER_MODE=${OPENCLAW_PLANNER_MODE:-model} \
  OPENCLAW_MODEL_PROVIDER=${OPENCLAW_MODEL_PROVIDER:-qwen_api} \
  OPENCLAW_MODEL=${OPENCLAW_MODEL:-qwen/qwen3.5-flash} \
  OPENCLAW_MODEL_MAX_IMAGES=${OPENCLAW_MODEL_MAX_IMAGES:-2} \
  OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=${OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS:-20} \
  OPENCLAW_MODEL_FAST_MODE=${OPENCLAW_MODEL_FAST_MODE:-memory_guided_policy_fast} \
  OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT=${OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT:-1} \
  OPENCLAW_AGENT_TIMEOUT=${OPENCLAW_AGENT_TIMEOUT:-90} \
  OPENCLAW_AGENT_MAX_INPUT_TOKENS=${OPENCLAW_AGENT_MAX_INPUT_TOKENS:-50000} \
  OPENCLAW_GATEWAY_TIMEOUT=${OPENCLAW_GATEWAY_TIMEOUT:-30} \
  NO_PROXY=${NO_PROXY:-127.0.0.1,localhost,::1} \
  no_proxy=${no_proxy:-127.0.0.1,localhost,::1} \
  TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false} \
  bash scripts/start_openclaw_cli_plan_gateway.sh >"${GATEWAY_LOG}" 2>&1 &
  GATEWAY_PID=$!
  sleep 2
fi

{
  echo "mode=${MODE}"
  echo "output_path=${OUTPUT_PATH}"
  echo "episode_key=${EPISODE_KEY}"
  echo "episode_keys=${EPISODE_KEYS}"
  echo "max_episodes=${HARNESS_DEBUG_MAX_EPISODES}"
  echo "gpu=${GPU}"
  echo "gateway_url=http://${HOST}:${PORT}"
  echo "smoke_seed=${SMOKE_SEED}"
  echo "visual_readback_mode=${OPENCLAW_VISUAL_READBACK_MODE}"
  echo "visual_readback_trigger_policy=${OPENCLAW_VISUAL_READBACK_TRIGGER_POLICY}"

  OUTPUT_PATH="${OUTPUT_PATH}" \
  MODEL_PATH="${MODEL_PATH}" \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  MASTER_PORT="${MASTER_PORT}" \
  OPENCLAW_GATEWAY_URL="http://${HOST}:${PORT}" \
  OPENCLAW_GATEWAY_TIMEOUT="${GATEWAY_TIMEOUT_S}" \
  OPENCLAW_ENFORCE_TIMEOUT_BUDGET=1 \
  CHECK_GATEWAY=1 \
  REQUIRE_GATEWAY=1 \
  REQUIRE_OPENCLAW_CLI_ADAPTER=1 \
  HARNESS_DEBUG_MAX_EPISODES="${HARNESS_DEBUG_MAX_EPISODES}" \
  HARNESS_EPISODE_KEYS="${EPISODE_KEYS}" \
  HARNESS_USE_DEFAULT_EPISODE_KEYS=0 \
  DATA_PATH="${DATA_PATH}" \
  MAX_STEPS="${MAX_STEPS}" \
  HARNESS_MEMORY_BACKEND=image_backed_local \
  HARNESS_MEMORY_SOURCE=episode-local \
  OPENCLAW_VISUAL_READBACK_MODE="${OPENCLAW_VISUAL_READBACK_MODE}" \
  OPENCLAW_VISUAL_READBACK_TRIGGER_POLICY="${OPENCLAW_VISUAL_READBACK_TRIGGER_POLICY}" \
  OPENCLAW_VISUAL_READBACK_SMOKE_SEED_MEMORY="${SMOKE_SEED}" \
  OPENCLAW_VISUAL_READBACK_CONTROL_ONLY=true \
  OPENCLAW_VISUAL_READBACK_STOP_FALLBACK_POLICY=log_only \
  OPENCLAW_VISUAL_READBACK_MAX_ACTION_OVERRIDES_PER_EPISODE="${OPENCLAW_VISUAL_READBACK_MAX_ACTION_OVERRIDES_PER_EPISODE:-1}" \
  NO_PROXY=${NO_PROXY:-127.0.0.1,localhost,::1} \
  no_proxy=${no_proxy:-127.0.0.1,localhost,::1} \
  TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false} \
  bash scripts/evaluation_openclaw_gateway.sh

  PYTHONPATH=.:src python scripts/summarize_openclaw_visual_readback.py \
    "${OUTPUT_PATH}" \
    --format json \
    --output "${OUTPUT_PATH}/visual_readback_summary.json"

  PYTHONPATH=.:src python scripts/check_visual_readback_runtime_smoke.py \
    "${OUTPUT_PATH}" \
    --mode "${MODE}" \
    --format text \
    --output "${OUTPUT_PATH}/runtime_smoke_check.json"
} 2>&1 | tee "${RUN_LOG}"
