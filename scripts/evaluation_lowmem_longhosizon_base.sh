#!/bin/bash
# 低显存优化版评估脚本
# 适用于显存受限的场景

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # PyTorch 显存分配优化


MASTER_PORT=$((RANDOM % 101 + 20000))
max_pixels=401408
kv_start_size=4
kv_recent_size=24
num_history=6

CHECKPOINT="JanusVLN_Model/misstl/JanusVLN_Base"
echo "CHECKPOINT: ${CHECKPOINT}"
#OUTPUT_PATH="results/400_janusvln_start${kv_start_size}_recent${kv_recent_size}_history${num_history}_savevideo"
OUTPUT_PATH="results/janusvln_base_lowmem_401408_start4_recent24_history6_goal2_7_90_${num_history}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

# # 检查 OUTPUT_PATH 目录是否存在，存在则删除后重建，不存在则直接创建
# if [ -d "${OUTPUT_PATH}" ]; then
#     echo "⚠️  OUTPUT_PATH 已存在，将删除并重新创建: ${OUTPUT_PATH}"
#     rm -rf "${OUTPUT_PATH}"
# fi
# mkdir -p "${OUTPUT_PATH}"
# echo "✓ OUTPUT_PATH 创建成功: ${OUTPUT_PATH}"

mkdir -p "${OUTPUT_PATH}"

# 自动递增日志文件名
if [ ! -f "${OUTPUT_PATH}/evaluation_lowmem.log" ]; then
    log_file="${OUTPUT_PATH}/evaluation_lowmem.log"
else
    # 查找下一个可用的 continue{N}.log 文件名
    counter=1
    while [ -f "${OUTPUT_PATH}/evaluation_lowmem_continue${counter}.log" ]; do
        counter=$((counter + 1))
    done
    log_file="${OUTPUT_PATH}/evaluation_lowmem_continue${counter}.log"
fi
echo "✓ 日志文件: ${log_file}"

CONFIG="config/vln_r2r_longhorizon.yaml"
echo "CONFIG: ${CONFIG}"

# ============================================================
# 显存优化级别选择 (取消注释一个使用)
# ============================================================

# --- Level 1: 轻度优化 (推荐首先尝试) ---
# 2个GPU，num_history=1，max_pixels=6272
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    src/evaluation.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --num_history ${num_history} \
    --max_pixels ${max_pixels} \
    --kv_start_size ${kv_start_size} \
    --kv_recent_size ${kv_recent_size} \
    --output_path $OUTPUT_PATH \
    2>&1 | tee $log_file

# --save_video \
#     --save_video_ratio 0.1 \