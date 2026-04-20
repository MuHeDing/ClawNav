#!/bin/bash
# 单卡 qualitative 可视化评估脚本

set -euo pipefail

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=$((RANDOM % 101 + 20000))
max_pixels=401408
kv_start_size=2
kv_recent_size=12  # 
num_history=8  # 6

CHECKPOINT="JanusVLN_Model/misstl/JanusVLN_Extra"
CONFIG="config/vln_r2r_longhorizon.yaml"
DATA_PATH=${1:-/ssd/dingmuhe/Embodied-task/JanusVLN/generate_path/final_experiments/val_unseen_90/unify_val_unseen_90/b_qualitive3.json.gz}
OUTPUT_PATH="results/janusvln_extra_lowmem_qualitive_401408_start${kv_start_size}_recent${kv_recent_size}_history${num_history}_qualitive_b5"
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
data_path_arg=()

echo "CHECKPOINT: ${CHECKPOINT}"
echo "CONFIG: ${CONFIG}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"

if [[ "${DATA_PATH}" != "None" ]]; then
    echo "DATA_PATH: ${DATA_PATH}"
    data_path_arg=(--data_path "${DATA_PATH}")
else
    echo "DATA_PATH: using config default"
fi

mkdir -p "${OUTPUT_PATH}"

if [ ! -f "${OUTPUT_PATH}/evaluation_sparse_qualitive.log" ]; then
    log_file="${OUTPUT_PATH}/evaluation_sparse_qualitive.log"
else
    counter=1
    while [ -f "${OUTPUT_PATH}/evaluation_sparse_qualitive_continue${counter}.log" ]; do
        counter=$((counter + 1))
    done
    log_file="${OUTPUT_PATH}/evaluation_sparse_qualitive_continue${counter}.log"
fi
echo "LOG_FILE: ${log_file}"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    src/evaluation_sparse_qualitive.py \
    --model_path "${CHECKPOINT}" \
    --habitat_config_path "${CONFIG}" \
    --num_history "${num_history}" \
    --max_pixels "${max_pixels}" \
    --kv_start_size "${kv_start_size}" \
    --kv_recent_size "${kv_recent_size}" \
    "${data_path_arg[@]}" \
    --output_path "${OUTPUT_PATH}" \
    --save_video \
    --save_video_ratio 1 \
    2>&1 | tee "${log_file}"
