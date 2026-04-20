export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
#log_file="/ssd/dingmuhe/Embodied-task/JanusVLN/results/janusvln_extra/evaluation.log"
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="../JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra"
echo "CHECKPOINT: ${CHECKPOINT}"
OUTPUT_PATH="results/janusvln_extra"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

mkdir -p "${OUTPUT_PATH}"
log_file="${OUTPUT_PATH}/evaluation2.log"
CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    src/evaluation.py \
    --model_path $CHECKPOINT \
    --habitat_config_path $CONFIG \
    --save_video \
    --num_history 8 \
    --output_path $OUTPUT_PATH \
    2>&1 | tee $log_file

