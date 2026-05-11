#!/bin/bash
# 低显存优化版评估脚本
# 适用于显存受限的场景

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # PyTorch 显存分配优化


MASTER_PORT=$((RANDOM % 101 + 20000))
TORCHRUN="/ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun"
max_pixels=401408
kv_start_size=8
kv_recent_size=24
num_history=8
adaptive_sparse_min_seq_len=128
adaptive_sparse_pvthreshd=1000000
adaptive_sparse_target_blocks=50
adaptive_sparse_target_drop_mass=0.5
adaptive_sparse_log_interval=50
adaptive_sparse_scope=visual_middle
adaptive_sparse_llm_kv_block_size=64
adaptive_sparse_llm_start_blocks=7
adaptive_sparse_llm_recent_blocks=14
adaptive_sparse_llm_start_layer=14

CHECKPOINT="/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra"
echo "CHECKPOINT: ${CHECKPOINT}"
#OUTPUT_PATH="results/400_janusvln_start${kv_start_size}_recent${kv_recent_size}_history${num_history}_savevideo"
OUTPUT_PATH="results/janusvln_extra_st8_re24_h8_llm_adap_sparse_tb_${adaptive_sparse_target_blocks}_drop_${adaptive_sparse_target_drop_mass}_st_b_${adaptive_sparse_llm_start_blocks}_re_b_${adaptive_sparse_llm_recent_blocks}_ly${adaptive_sparse_llm_start_layer}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "SPARGEATTN: using vendored Fast_JanusVLN/src/spas_sage_attn"
echo "VGGT KV cache: kv_start_size=${kv_start_size}, kv_recent_size=${kv_recent_size}"
echo "LLM adaptive sparse: min_seq_len=${adaptive_sparse_min_seq_len}, pvthreshd=${adaptive_sparse_pvthreshd}, target_blocks=${adaptive_sparse_target_blocks}, target_drop_mass=${adaptive_sparse_target_drop_mass}, scope=${adaptive_sparse_scope}, llm_kv_block_size=${adaptive_sparse_llm_kv_block_size}, llm_start_blocks=${adaptive_sparse_llm_start_blocks}, llm_recent_blocks=${adaptive_sparse_llm_recent_blocks}, llm_start_layer=${adaptive_sparse_llm_start_layer}"

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

CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"
echo "TORCHRUN: ${TORCHRUN}"

# ============================================================
# 显存优化级别选择 (取消注释一个使用)
# ============================================================

# --- Level 1: 轻度优化 (推荐首先尝试) ---
# 2个GPU，num_history=1，max_pixels=6272
CUDA_VISIBLE_DEVICES=7 "${TORCHRUN}" --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    src/evaluation.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --num_history ${num_history} \
    --max_pixels ${max_pixels} \
    --kv_start_size ${kv_start_size} \
    --kv_recent_size ${kv_recent_size} \
    --disable_visual_prune_eval_profile \
    --use_llm_adaptive_sparse_attention \
    --adaptive_sparse_min_seq_len ${adaptive_sparse_min_seq_len} \
    --adaptive_sparse_pvthreshd ${adaptive_sparse_pvthreshd} \
    --adaptive_sparse_target_blocks ${adaptive_sparse_target_blocks} \
    --adaptive_sparse_target_drop_mass ${adaptive_sparse_target_drop_mass} \
    --adaptive_sparse_log_interval ${adaptive_sparse_log_interval} \
    --adaptive_sparse_scope ${adaptive_sparse_scope} \
    --adaptive_sparse_llm_kv_block_size ${adaptive_sparse_llm_kv_block_size} \
    --adaptive_sparse_llm_start_blocks ${adaptive_sparse_llm_start_blocks} \
    --adaptive_sparse_llm_recent_blocks ${adaptive_sparse_llm_recent_blocks} \
    --adaptive_sparse_llm_start_layer ${adaptive_sparse_llm_start_layer} \
    --output_path $OUTPUT_PATH \
    2>&1 | tee $log_file

# --save_video \
#     --save_video_ratio 0.1 \
