#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra}
OUTPUT_PATH=${OUTPUT_PATH:-results/clawnav_openclaw_gateway}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
MASTER_PORT=${MASTER_PORT:-20401}
TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

OPENCLAW_GATEWAY_URL=${OPENCLAW_GATEWAY_URL:-http://127.0.0.1:8011}
OPENCLAW_SERVICE_REGISTRY=${OPENCLAW_SERVICE_REGISTRY:-/ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/SERVICE.md}
OPENCLAW_SERVICE_HOST=${OPENCLAW_SERVICE_HOST:-127.0.0.1}
NO_PROXY=${NO_PROXY:-127.0.0.1,localhost,::1}
no_proxy=${no_proxy:-${NO_PROXY}}

HARNESS_MEMORY_BACKEND=${HARNESS_MEMORY_BACKEND:-fake}
HARNESS_MEMORY_SOURCE=${HARNESS_MEMORY_SOURCE:-episode-local}
SPATIAL_MEMORY_URL=${SPATIAL_MEMORY_URL:-http://127.0.0.1:8022}
MEMORY_MANIFEST_PATH=${MEMORY_MANIFEST_PATH:-}

OPENCLAW_EXECUTOR_BACKEND=${OPENCLAW_EXECUTOR_BACKEND:-habitat}
OPENCLAW_ROBOT_EXECUTOR_URL=${OPENCLAW_ROBOT_EXECUTOR_URL:-}

EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}
DATA_PATH=${DATA_PATH:-}
HARNESS_DEBUG_MAX_EPISODES=${HARNESS_DEBUG_MAX_EPISODES:-}

extra_args=()
if [[ -n "${DATA_PATH}" ]]; then
  extra_args+=(--data_path "${DATA_PATH}")
fi
if [[ -n "${HARNESS_DEBUG_MAX_EPISODES}" ]]; then
  extra_args+=(--harness_debug_max_episodes "${HARNESS_DEBUG_MAX_EPISODES}")
fi
if [[ -n "${MEMORY_MANIFEST_PATH}" ]]; then
  extra_args+=(--memory_manifest_path "${MEMORY_MANIFEST_PATH}")
fi
if [[ "${HARNESS_MEMORY_BACKEND}" == "spatial_http" ]]; then
  extra_args+=(--spatial_memory_url "${SPATIAL_MEMORY_URL}")
fi
if [[ "${OPENCLAW_EXECUTOR_BACKEND}" == "robot_http" ]]; then
  if [[ -z "${OPENCLAW_ROBOT_EXECUTOR_URL}" ]]; then
    echo "OPENCLAW_ROBOT_EXECUTOR_URL is required when OPENCLAW_EXECUTOR_BACKEND=robot_http" >&2
    exit 2
  fi
  extra_args+=(--openclaw_robot_executor_url "${OPENCLAW_ROBOT_EXECUTOR_URL}")
fi

echo "OpenClaw gateway: ${OPENCLAW_GATEWAY_URL}"
echo "Executor backend: ${OPENCLAW_EXECUTOR_BACKEND}"
echo "Memory backend/source: ${HARNESS_MEMORY_BACKEND}/${HARNESS_MEMORY_SOURCE}"
echo "Output path: ${OUTPUT_PATH}"

export NO_PROXY no_proxy TOKENIZERS_PARALLELISM

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  src/evaluation_harness.py \
  --model_path "${MODEL_PATH}" \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split "${EVAL_SPLIT}" \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --output_path "${OUTPUT_PATH}" \
  --harness_runtime openclaw_bridge \
  --harness_mode memory_recall \
  --harness_memory_backend "${HARNESS_MEMORY_BACKEND}" \
  --harness_memory_source "${HARNESS_MEMORY_SOURCE}" \
  --openclaw_planner_backend gateway \
  --openclaw_gateway_url "${OPENCLAW_GATEWAY_URL}" \
  --openclaw_executor_backend "${OPENCLAW_EXECUTOR_BACKEND}" \
  --openclaw_service_registry_path "${OPENCLAW_SERVICE_REGISTRY}" \
  --openclaw_service_host "${OPENCLAW_SERVICE_HOST}" \
  "${extra_args[@]}"
