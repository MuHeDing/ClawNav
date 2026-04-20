#!/bin/bash
# Low-memory evaluation script for RxR
# Based on scripts/evaluation_lowmem2.sh

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=$((RANDOM % 101 + 20000))
max_pixels=1605632
kv_start_size=8
kv_recent_size=24
num_history=8

CHECKPOINT="JanusVLN_Model/misstl/JanusVLN_Extra"
echo "CHECKPOINT: ${CHECKPOINT}"
OUTPUT_PATH="results/janusvln_extra_lowmem_rxr_${max_pixels}_start${kv_start_size}_recent${kv_recent_size}_history${num_history}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

mkdir -p "${OUTPUT_PATH}"

if [ ! -f "${OUTPUT_PATH}/evaluation_lowmem_rxr.log" ]; then
    log_file="${OUTPUT_PATH}/evaluation_lowmem_rxr.log"
else
    counter=1
    while [ -f "${OUTPUT_PATH}/evaluation_lowmem_rxr_continue${counter}.log" ]; do
        counter=$((counter + 1))
    done
    log_file="${OUTPUT_PATH}/evaluation_lowmem_rxr_continue${counter}.log"
fi
echo "LOG_FILE: ${log_file}"

CONFIG="config/vln_rxr.yaml"
echo "CONFIG: ${CONFIG}"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    src/evaluation_rxr.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --num_history ${num_history} \
    --max_pixels ${max_pixels} \
    --kv_start_size ${kv_start_size} \
    --kv_recent_size ${kv_recent_size} \
    --output_path $OUTPUT_PATH \
    --save_video \
    --save_video_ratio 0.1 \
    2>&1 | tee $log_file
