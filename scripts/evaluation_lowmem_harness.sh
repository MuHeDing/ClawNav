#!/bin/bash
# ClawNav Harness low-memory evaluation script: no LLM adaptive sparse attention, no slow-fast reuse.

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=$((RANDOM % 101 + 20000))
TORCHRUN="/ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun"
max_pixels=401408
kv_start_size=8
kv_recent_size=24
num_history=8

CHECKPOINT="/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra"
OUTPUT_PATH="results/janusvln_extra_lowmem_401408_start8_recent24_history8_clawnav"

echo "CHECKPOINT: ${CHECKPOINT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "VGGT KV cache: kv_start_size=${kv_start_size}, kv_recent_size=${kv_recent_size}"
echo "ClawNav Harness: enabled"
echo "LLM adaptive sparse: disabled"
echo "Slow-fast active memory reuse: disabled"

mkdir -p "${OUTPUT_PATH}"

if [ ! -f "${OUTPUT_PATH}/evaluation_harness.log" ]; then
    log_file="${OUTPUT_PATH}/evaluation_harness.log"
else
    counter=1
    while [ -f "${OUTPUT_PATH}/evaluation_harness_continue${counter}.log" ]; do
        counter=$((counter + 1))
    done
    log_file="${OUTPUT_PATH}/evaluation_harness_continue${counter}.log"
fi
echo "Log file: ${log_file}"

CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"
echo "TORCHRUN: ${TORCHRUN}"

CUDA_VISIBLE_DEVICES=4 "${TORCHRUN}" --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    src/evaluation_harness.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --num_history ${num_history} \
    --max_pixels ${max_pixels} \
    --kv_start_size ${kv_start_size} \
    --kv_recent_size ${kv_recent_size} \
    --output_path $OUTPUT_PATH \
    --harness_mode memory_recall \
    --harness_max_internal_calls 3 \
    --harness_recall_interval_steps 5 \
    --harness_memory_backend fake \
    --harness_memory_source episode-local \
    --harness_debug_max_episodes 1 \
    2>&1 | tee $log_file
