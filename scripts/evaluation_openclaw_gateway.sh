#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_PATH=${OUTPUT_PATH:-results/clawnav_openclaw_gateway3_${RUN_TIMESTAMP}}
POLICY_BACKEND=${POLICY_BACKEND:-janus_policy}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
MASTER_PORT=${MASTER_PORT:-20401}
TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -r -a _visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE=${#_visible_devices[@]}
fi

OPENCLAW_GATEWAY_URL=${OPENCLAW_GATEWAY_URL:-http://127.0.0.1:8011}
OPENCLAW_GATEWAY_TIMEOUT=${OPENCLAW_GATEWAY_TIMEOUT:-300}
OPENCLAW_ENFORCE_TIMEOUT_BUDGET=${OPENCLAW_ENFORCE_TIMEOUT_BUDGET:-1}
OPENCLAW_SERVICE_REGISTRY=${OPENCLAW_SERVICE_REGISTRY-/ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/SERVICE.md}
OPENCLAW_SERVICE_HOST=${OPENCLAW_SERVICE_HOST:-127.0.0.1}
NO_PROXY=${NO_PROXY:-127.0.0.1,localhost,::1}
no_proxy=${no_proxy:-${NO_PROXY}}

should_enable_flag() {
  local v="${1:-}"
  case "${v,,}" in
    1|true|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

HARNESS_MEMORY_BACKEND=${HARNESS_MEMORY_BACKEND:-fake}
HARNESS_MEMORY_SOURCE=${HARNESS_MEMORY_SOURCE:-episode-local}
SPATIAL_MEMORY_URL=${SPATIAL_MEMORY_URL:-http://127.0.0.1:8022}
MEMORY_MANIFEST_PATH=${MEMORY_MANIFEST_PATH:-}

OPENCLAW_EXECUTOR_BACKEND=${OPENCLAW_EXECUTOR_BACKEND:-habitat}
OPENCLAW_ROBOT_EXECUTOR_URL=${OPENCLAW_ROBOT_EXECUTOR_URL:-}
OPENCLAW_ENABLE_SUBAGENT_CRITIC=${OPENCLAW_ENABLE_SUBAGENT_CRITIC:-0}
OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR=${OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR:-0}
SAVE_VIDEO=${SAVE_VIDEO:-0}
SAVE_VIDEO_RATIO=${SAVE_VIDEO_RATIO:-0.05}
SAVE_STEP_ARTIFACTS=${SAVE_STEP_ARTIFACTS:-0}
SAVE_STEP_ARTIFACTS_WITH_VIDEO_ONLY=${SAVE_STEP_ARTIFACTS_WITH_VIDEO_ONLY:-0}
OPENCLAW_MAP_ASSIST_MODE=${OPENCLAW_MAP_ASSIST_MODE:-off}
OPENCLAW_MAP_FRAME_INTERVAL_STEPS=${OPENCLAW_MAP_FRAME_INTERVAL_STEPS:-5}
OPENCLAW_MOTION_FEEDBACK_ENABLED=${OPENCLAW_MOTION_FEEDBACK_ENABLED:-0}
OPENCLAW_FORWARD_STALL_ODOMETRY_ENABLED=${OPENCLAW_FORWARD_STALL_ODOMETRY_ENABLED:-0}
OPENCLAW_MAP_COLLISION_OVERLAY_ENABLED=${OPENCLAW_MAP_COLLISION_OVERLAY_ENABLED:-0}
OPENCLAW_DYNAMIC_VISUAL_CONTEXT_ENABLED=${OPENCLAW_DYNAMIC_VISUAL_CONTEXT_ENABLED:-0}
OPENCLAW_STAGED_VISUAL_MEMORY_ENABLED=${OPENCLAW_STAGED_VISUAL_MEMORY_ENABLED:-0}
OPENCLAW_STAGED_MEMORY_TREATMENT=${OPENCLAW_STAGED_MEMORY_TREATMENT:-on}
OPENCLAW_STAGE_MIN_TRANSLATION_M=${OPENCLAW_STAGE_MIN_TRANSLATION_M:-0.25}
OPENCLAW_STAGE_MIN_HEADING_CHANGE_DEG=${OPENCLAW_STAGE_MIN_HEADING_CHANGE_DEG:-15}
OPENCLAW_STAGED_MEMORY_EVENT_CAP=${OPENCLAW_STAGED_MEMORY_EVENT_CAP:-64}
OPENCLAW_STAGED_RECOVERY_RETRIGGER_STEPS=${OPENCLAW_STAGED_RECOVERY_RETRIGGER_STEPS:-3}
OPENCLAW_STAGED_SHADOW_MANIFEST_PATH=${OPENCLAW_STAGED_SHADOW_MANIFEST_PATH:-}
OPENCLAW_STAGED_SHADOW_MAX_EVENTS=${OPENCLAW_STAGED_SHADOW_MAX_EVENTS:-5}
OPENCLAW_QWEN_OUTPUT_SCHEMA=${OPENCLAW_QWEN_OUTPUT_SCHEMA:-legacy}

if should_enable_flag "${OPENCLAW_STAGED_VISUAL_MEMORY_ENABLED}"; then
  if [[ "${POLICY_BACKEND}" != "qwen_direct" || "${OPENCLAW_QWEN_OUTPUT_SCHEMA}" != "route_v3_staged" ]]; then
    echo "Staged visual memory requires qwen_direct with route_v3_staged" >&2
    exit 2
  fi
  if ! should_enable_flag "${OPENCLAW_DYNAMIC_VISUAL_CONTEXT_ENABLED}"; then
    echo "Staged visual memory requires dynamic visual context" >&2
    exit 2
  fi
fi

HARNESS_SELECTED_EPISODES=(
  "2azQ1b91cZZ:11"
  "2azQ1b91cZZ:10"
  "2azQ1b91cZZ:12"
  "2azQ1b91cZZ:16"
  "2azQ1b91cZZ:18"
  "2azQ1b91cZZ:17"
  "2azQ1b91cZZ:43"
  "2azQ1b91cZZ:44"
  "2azQ1b91cZZ:70"
  "2azQ1b91cZZ:71"
  "2azQ1b91cZZ:72"
  "2azQ1b91cZZ:78"
  "2azQ1b91cZZ:79"
  "2azQ1b91cZZ:80"
  "2azQ1b91cZZ:81"
  "zsNo4HB9uLZ:2"
  "zsNo4HB9uLZ:15"
  "zsNo4HB9uLZ:26"
  "zsNo4HB9uLZ:37"
  "zsNo4HB9uLZ:54"
  "zsNo4HB9uLZ:73"
  "zsNo4HB9uLZ:75"
  "zsNo4HB9uLZ:126"
  "zsNo4HB9uLZ:137"
  "zsNo4HB9uLZ:145"
  "zsNo4HB9uLZ:147"
  "zsNo4HB9uLZ:163"
  "zsNo4HB9uLZ:165"
  "zsNo4HB9uLZ:247"
  "zsNo4HB9uLZ:251"
)
HARNESS_USE_DEFAULT_EPISODE_KEYS=${HARNESS_USE_DEFAULT_EPISODE_KEYS:-1}
if [[ -z "${HARNESS_EPISODE_KEYS:-}" && "${HARNESS_USE_DEFAULT_EPISODE_KEYS}" == "1" ]]; then
  HARNESS_EPISODE_KEYS=$(IFS=,; echo "${HARNESS_SELECTED_EPISODES[*]}")
fi

EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}
DATA_PATH=${DATA_PATH:-}
HARNESS_DEBUG_MAX_EPISODES=${HARNESS_DEBUG_MAX_EPISODES:-30}
MAX_STEPS=${MAX_STEPS:-400}
OPENCLAW_CONTROLLER_DISTANCE_EARLY_STOP_M=${OPENCLAW_CONTROLLER_DISTANCE_EARLY_STOP_M:-0}
if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be a positive integer; got '${MAX_STEPS}'" >&2
  exit 2
fi
CHECK_GATEWAY=${CHECK_GATEWAY:-1}
REQUIRE_GATEWAY=${REQUIRE_GATEWAY:-0}
REQUIRE_OPENCLAW_CLI_ADAPTER=${REQUIRE_OPENCLAW_CLI_ADAPTER:-1}

extra_args=()
extra_args+=(--controller_distance_early_stop_m "${OPENCLAW_CONTROLLER_DISTANCE_EARLY_STOP_M}")
if [[ -n "${DATA_PATH}" ]]; then
  extra_args+=(--data_path "${DATA_PATH}")
fi
if [[ -n "${HARNESS_DEBUG_MAX_EPISODES}" ]]; then
  extra_args+=(--harness_debug_max_episodes "${HARNESS_DEBUG_MAX_EPISODES}")
fi
if [[ -n "${HARNESS_EPISODE_KEYS:-}" ]]; then
  extra_args+=(--harness_episode_keys "${HARNESS_EPISODE_KEYS}")
fi
if [[ -n "${MEMORY_MANIFEST_PATH}" ]]; then
  extra_args+=(--memory_manifest_path "${MEMORY_MANIFEST_PATH}")
fi
if [[ -n "${OPENCLAW_SERVICE_REGISTRY}" ]]; then
  extra_args+=(--openclaw_service_registry_path "${OPENCLAW_SERVICE_REGISTRY}")
fi
if [[ "${HARNESS_MEMORY_BACKEND}" == "spatial_http" ]]; then
  extra_args+=(--spatial_memory_url "${SPATIAL_MEMORY_URL}")
fi
if [[ "${OPENCLAW_EXECUTOR_BACKEND}" == "robot_http" ]]; then
  if [[ -z "${OPENCLAW_ROBOT_EXECUTOR_URL}" ]]; then
    echo "OPENCLAW_ROBOT_EXECUTOR_URL is required when OPENCLAW_EXECUTOR_BACKEND=robot_http" >&2
    exit 2
  fi
  extra_args+=(--openclaw_robot_executor_url "${OPENCLAW_ROBOT_EXECUTOR_URL}")
fi
if [[ "${OPENCLAW_ENABLE_SUBAGENT_CRITIC}" == "1" ]]; then
  extra_args+=(--openclaw_enable_subagent_critic)
fi
if [[ "${OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR}" == "1" ]]; then
  extra_args+=(--openclaw_enable_subagent_memory_curator)
fi
if should_enable_flag "${SAVE_VIDEO}"; then
  extra_args+=(--save_video)
fi
if [[ -n "${SAVE_VIDEO_RATIO}" ]]; then
  extra_args+=(--save_video_ratio "${SAVE_VIDEO_RATIO}")
fi
if should_enable_flag "${SAVE_STEP_ARTIFACTS}"; then
  extra_args+=(--save_step_artifacts)
fi
if should_enable_flag "${SAVE_STEP_ARTIFACTS_WITH_VIDEO_ONLY}"; then
  extra_args+=(--save_step_artifacts_with_video_only)
fi
extra_args+=(--map_assist_mode "${OPENCLAW_MAP_ASSIST_MODE}")
extra_args+=(--map_frame_interval_steps "${OPENCLAW_MAP_FRAME_INTERVAL_STEPS}")
if should_enable_flag "${OPENCLAW_MOTION_FEEDBACK_ENABLED}"; then
  extra_args+=(--motion_feedback_enabled)
fi
if should_enable_flag "${OPENCLAW_FORWARD_STALL_ODOMETRY_ENABLED}"; then
  extra_args+=(--forward_stall_odometry_enabled)
fi
if should_enable_flag "${OPENCLAW_MAP_COLLISION_OVERLAY_ENABLED}"; then
  extra_args+=(--map_collision_overlay_enabled)
fi
if should_enable_flag "${OPENCLAW_DYNAMIC_VISUAL_CONTEXT_ENABLED}"; then
  extra_args+=(--dynamic_visual_context_enabled)
fi
if should_enable_flag "${OPENCLAW_STAGED_VISUAL_MEMORY_ENABLED}"; then
  extra_args+=(--staged_visual_memory_enabled)
  extra_args+=(--staged_memory_treatment "${OPENCLAW_STAGED_MEMORY_TREATMENT}")
  extra_args+=(--stage_min_translation_m "${OPENCLAW_STAGE_MIN_TRANSLATION_M}")
  extra_args+=(--stage_min_heading_change_deg "${OPENCLAW_STAGE_MIN_HEADING_CHANGE_DEG}")
  extra_args+=(--staged_memory_event_cap "${OPENCLAW_STAGED_MEMORY_EVENT_CAP}")
  extra_args+=(--staged_recovery_retrigger_steps "${OPENCLAW_STAGED_RECOVERY_RETRIGGER_STEPS}")
  extra_args+=(--staged_shadow_max_events "${OPENCLAW_STAGED_SHADOW_MAX_EVENTS}")
  if [[ -n "${OPENCLAW_STAGED_SHADOW_MANIFEST_PATH}" ]]; then
    extra_args+=(--staged_shadow_manifest_path "${OPENCLAW_STAGED_SHADOW_MANIFEST_PATH}")
  fi
fi

echo "OpenClaw gateway: ${OPENCLAW_GATEWAY_URL}"
echo "Executor backend: ${OPENCLAW_EXECUTOR_BACKEND}"
echo "Memory backend/source: ${HARNESS_MEMORY_BACKEND}/${HARNESS_MEMORY_SOURCE}"
echo "OpenClaw critic/curator: ${OPENCLAW_ENABLE_SUBAGENT_CRITIC}/${OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR}"
echo "Map assist: ${OPENCLAW_MAP_ASSIST_MODE} interval=${OPENCLAW_MAP_FRAME_INTERVAL_STEPS}"
echo "Motion feedback/forward odometry/map collision overlay: ${OPENCLAW_MOTION_FEEDBACK_ENABLED}/${OPENCLAW_FORWARD_STALL_ODOMETRY_ENABLED}/${OPENCLAW_MAP_COLLISION_OVERLAY_ENABLED}"
echo "Require OpenClaw CLI adapter: ${REQUIRE_OPENCLAW_CLI_ADAPTER}"
echo "Max episodes: ${HARNESS_DEBUG_MAX_EPISODES:-all}"
echo "Use default episode keys: ${HARNESS_USE_DEFAULT_EPISODE_KEYS}"
echo "Episode keys: ${HARNESS_EPISODE_KEYS:-all}"
echo "Max steps per episode: ${MAX_STEPS}"
echo "Controller distance early stop (m, 0=off): ${OPENCLAW_CONTROLLER_DISTANCE_EARLY_STOP_M}"
echo "Output path: ${OUTPUT_PATH}"
echo "Policy backend: ${POLICY_BACKEND}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES}"
echo "Torch processes: ${NPROC_PER_NODE}"

export NO_PROXY no_proxy TOKENIZERS_PARALLELISM

if [[ "${CHECK_GATEWAY}" == "1" ]]; then
  gateway_check_args=(
    --gateway_url "${OPENCLAW_GATEWAY_URL}"
    --timeout "${OPENCLAW_GATEWAY_TIMEOUT}"
  )
  if [[ "${REQUIRE_OPENCLAW_CLI_ADAPTER}" == "1" ]]; then
    gateway_check_args+=(--require_service openclaw_cli_plan_gateway)
  fi
  if [[ "${OPENCLAW_ENFORCE_TIMEOUT_BUDGET}" == "1" ]]; then
    gateway_check_args+=(--enforce_timeout_budget)
  fi
  if should_enable_flag "${OPENCLAW_STAGED_VISUAL_MEMORY_ENABLED}"; then
    gateway_check_args+=(--required_stage_schema instruction_stages_v1)
    gateway_check_args+=(--required_action_schema route_v3_staged)
  fi
  if ! PYTHONPATH=.:src /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python \
    scripts/check_openclaw_plan_gateway.py \
    "${gateway_check_args[@]}"; then
    if [[ "${REQUIRE_GATEWAY}" == "1" ]]; then
      echo "OpenClaw gateway preflight failed and REQUIRE_GATEWAY=1" >&2
      exit 2
    fi
    echo "OpenClaw gateway preflight failed; continuing so runtime fallback can handle planner errors." >&2
  fi
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  src/evaluation_harness.py \
  --model_path "${MODEL_PATH}" \
  --policy_backend "${POLICY_BACKEND}" \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split "${EVAL_SPLIT}" \
  --max_steps "${MAX_STEPS}" \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --output_path "${OUTPUT_PATH}" \
  --harness_runtime openclaw_bridge \
  --harness_mode memory_recall \
  --harness_memory_backend "${HARNESS_MEMORY_BACKEND}" \
  --harness_memory_source "${HARNESS_MEMORY_SOURCE}" \
  --openclaw_planner_backend gateway \
  --openclaw_gateway_url "${OPENCLAW_GATEWAY_URL}" \
  --openclaw_gateway_timeout "${OPENCLAW_GATEWAY_TIMEOUT}" \
  --openclaw_executor_backend "${OPENCLAW_EXECUTOR_BACKEND}" \
  --openclaw_service_host "${OPENCLAW_SERVICE_HOST}" \
  "${extra_args[@]}"
