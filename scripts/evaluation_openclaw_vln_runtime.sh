#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra}
OUTPUT_PATH=${OUTPUT_PATH:-results/clawnav_openclaw_vln_runtime}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
MASTER_PORT=${MASTER_PORT:-20371}
OPENCLAW_SERVICE_REGISTRY=${OPENCLAW_SERVICE_REGISTRY:-/ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/SERVICE.md}
OPENCLAW_SERVICE_HOST=${OPENCLAW_SERVICE_HOST:-127.0.0.1}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=${MASTER_PORT} \
  src/evaluation_harness.py \
  --model_path "${MODEL_PATH}" \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --output_path "${OUTPUT_PATH}" \
  --harness_runtime openclaw_bridge \
  --harness_mode memory_recall \
  --harness_memory_backend fake \
  --harness_memory_source episode-local \
  --openclaw_service_registry_path "${OPENCLAW_SERVICE_REGISTRY}" \
  --openclaw_service_host "${OPENCLAW_SERVICE_HOST}"
