#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra}
DATA_PATH=${DATA_PATH:-/ssd/dingmuhe/Embodied-task/JanusVLN/data/datasets/r2r/val_unseen/val_unseen.json.gz}
OUTPUT_PATH=${OUTPUT_PATH:-results/clawnav_openclaw_gateway_val_unseen_full}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
MASTER_PORT=${MASTER_PORT:-20401}
TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a _visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE=${#_visible_devices[@]}
fi

OPENCLAW_GATEWAY_URL=${OPENCLAW_GATEWAY_URL:-http://127.0.0.1:8011}
OPENCLAW_GATEWAY_TIMEOUT=${OPENCLAW_GATEWAY_TIMEOUT:-90}
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

extra_args=()
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
echo "Dataset: ${DATA_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Executor backend: ${OPENCLAW_EXECUTOR_BACKEND}"
echo "Memory backend/source: ${HARNESS_MEMORY_BACKEND}/${HARNESS_MEMORY_SOURCE}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES}"
echo "Torch processes: ${NPROC_PER_NODE}"

export NO_PROXY no_proxy TOKENIZERS_PARALLELISM

PYTHONPATH=.:src /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python \
  scripts/check_openclaw_plan_gateway.py \
  --gateway_url "${OPENCLAW_GATEWAY_URL}" \
  --timeout "${OPENCLAW_GATEWAY_TIMEOUT}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  src/evaluation_harness.py \
  --model_path "${MODEL_PATH}" \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split "${EVAL_SPLIT}" \
  --data_path "${DATA_PATH}" \
  --num_history 8 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --output_path "${OUTPUT_PATH}" \
  --harness_runtime openclaw_bridge \
  --harness_mode memory_recall \
  --harness_memory_backend "${HARNESS_MEMORY_BACKEND}" \
  --harness_memory_source "${HARNESS_MEMORY_SOURCE}" \
  --openclaw_planner_backend gateway \
  --openclaw_gateway_url "${OPENCLAW_GATEWAY_URL}" \
  --openclaw_gateway_timeout "${OPENCLAW_GATEWAY_TIMEOUT}" \
  --openclaw_executor_backend "${OPENCLAW_EXECUTOR_BACKEND}" \
  --openclaw_service_registry_path "${OPENCLAW_SERVICE_REGISTRY}" \
  --openclaw_service_host "${OPENCLAW_SERVICE_HOST}" \
  "${extra_args[@]}"
