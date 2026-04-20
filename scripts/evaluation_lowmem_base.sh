#!/bin/bash
# 低显存优化版评估脚本
# 适用于显存受限的场景

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # PyTorch 显存分配优化


MASTER_PORT=$((RANDOM % 101 + 20000))

max_pixels=401408

CHECKPOINT="JanusVLN_Model/misstl/JanusVLN_Base"
echo "CHECKPOINT: ${CHECKPOINT}"
OUTPUT_PATH="results/janusvln_base_lowmem_${max_pixels}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

# 检查 OUTPUT_PATH 目录是否存在，存在则删除后重建，不存在则直接创建
if [ -d "${OUTPUT_PATH}" ]; then
    echo "⚠️  OUTPUT_PATH 已存在，将删除并重新创建: ${OUTPUT_PATH}"
    rm -rf "${OUTPUT_PATH}"
fi
mkdir -p "${OUTPUT_PATH}"
echo "✓ OUTPUT_PATH 创建成功: ${OUTPUT_PATH}"

log_file="${OUTPUT_PATH}/evaluation_lowmem.log"
CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"

# ============================================================
# 显存优化级别选择 (取消注释一个使用)
# ============================================================

# --- Level 1: 轻度优化 (推荐首先尝试) ---
# 2个GPU，num_history=1，max_pixels=6272
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    src/evaluation.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --num_history 8 \
    --max_pixels $max_pixels \
    --output_path $OUTPUT_PATH \
    2>&1 | tee $log_file

# --- Level 2: 中度优化 (如果 Level 1 仍 OOM) ---
# 1个GPU，num_history=0 (仅当前帧)，max_pixels=3136
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
#     --master_port=$MASTER_PORT \
#     src/evaluation.py \
#     --model_path $CHECKPOINT \
#     --habitat_config_path $CONFIG \
#     --num_history 0 \
#     --max_pixels 3136 \
#     --output_path $OUTPUT_PATH \
#     2>&1 | tee $log_file

# --- Level 3: 激进优化 (最小显存占用) ---
# 单GPU，无历史，max_pixels=1568，降低 max_steps
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
#     --master_port=$MASTER_PORT \
#     src/evaluation.py \
#     --model_path $CHECKPOINT \
#     --habitat_config_path $CONFIG \
#     --num_history 0 \
#     --max_pixels 1568 \
#     --max_steps 200 \
#     --output_path $OUTPUT_PATH \
#     2>&1 | tee $log_file

# ============================================================
# 优化说明:
# 1. max_pixels 现在可通过 --max_pixels 参数控制
#    Level 1: 6272 | Level 2: 3136 | Level 3: 1568
# 2. 代码已添加显存优化:
#    - @torch.no_grad() 装饰器防止梯度计算
#    - 每5步自动清理显存缓存
#    - Episode结束后重置VGGT缓存并清理显存
# 3. 如果仍 OOM，按注释逐级启用更激进的优化
# 4. 实时监控显存: watch -n 1 nvidia-smi
# 5. 性能影响: max_pixels越小速度越快但精度可能下降
# ============================================================
