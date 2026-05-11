CUDA_VISIBLE_DEVICES=4 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20340 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_check_fresh \
  --harness_mode memory_recall \
  --harness_memory_backend fake \
  --harness_memory_source episode-local
