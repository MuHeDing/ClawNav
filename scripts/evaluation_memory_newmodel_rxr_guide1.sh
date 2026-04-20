#!/bin/bash
# Memory-bank evaluation using JanusVLN_QformerOnly checkpoint-4000 for both model and Q-Former

set -euo pipefail

#export HF_ENDPOINT="https://hf-mirror.com"

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=$((RANDOM % 101 + 20000))
max_pixels=1605632
kv_start_size=8
kv_recent_size=24
num_history=8
memory_bank_length=50
memory_num_query_tokens=32
memory_num_frames=0
DATA_PATH=${1:-/ssd/dingmuhe/Embodied-task/JanusVLN/data/datasets/rxr/val_unseen/val_unseen_guide1.json.gz}
data_path_arg=()


# Use the same checkpoint for the JanusVLN model and the Q-Former weights
CHECKPOINT="/ssd/dingmuhe/Embodied-task/JanusVLN_Memory/results/JanusVLN_QformerOnly_lr1e-4_batch_size1/checkpoint-8000"
qformer_ckpt="${CHECKPOINT}"

echo "CHECKPOINT: ${CHECKPOINT}"
echo "QFORMER_CKPT: ${qformer_ckpt}"
OUTPUT_PATH="results/rxrguide1_janusvln_qformeronly_memory_mbl${memory_bank_length}_ckpt8000_lr1e_4_batchsize1"
#OUTPUT_PATH="/ssd/dingmuhe/Embodied-task/JanusVLN_Memory/results/janusvln_qformeronly_memory_1605632_start8_recent24_history8_mbl50_ckpt8000"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
if [[ "${DATA_PATH}" != "None" ]]; then
    echo "DATA_PATH: ${DATA_PATH}"
    data_path_arg=(--data_path "${DATA_PATH}")
else
    echo "DATA_PATH: using default in config/vln_rxr.yaml"
fi




mkdir -p "${OUTPUT_PATH}"

if [ ! -f "${OUTPUT_PATH}/evaluation_memory_newmodel.log" ]; then
    log_file="${OUTPUT_PATH}/evaluation_memory_newmodel.log"
else
    counter=1
    while [ -f "${OUTPUT_PATH}/evaluation_memory_newmodel_continue${counter}.log" ]; do
        counter=$((counter + 1))
    done
    log_file="${OUTPUT_PATH}/evaluation_memory_newmodel_continue${counter}.log"
fi
echo "✓ 日志文件: ${log_file}"

CONFIG="config/vln_rxr.yaml"
echo "CONFIG: ${CONFIG}"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    src/evaluation.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --num_history ${num_history} \
    --max_pixels ${max_pixels} \
    --kv_start_size ${kv_start_size} \
    --kv_recent_size ${kv_recent_size} \
    --memory_bank_length ${memory_bank_length} \
    --memory_num_query_tokens ${memory_num_query_tokens} \
    --memory_num_frames ${memory_num_frames} \
    --qformer_ckpt_path ${qformer_ckpt} \
    "${data_path_arg[@]}" \
    --output_path $OUTPUT_PATH \
    --save_video \
    --save_video_ratio 0.1 \
    2>&1 | tee $log_file
