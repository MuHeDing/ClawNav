#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH=${OUTPUT_PATH:-results/clawnav_openclaw_visual_memory}
OPENCLAW_GATEWAY_URL=${OPENCLAW_GATEWAY_URL:-http://127.0.0.1:8011}
OPENCLAW_GATEWAY_TIMEOUT=${OPENCLAW_GATEWAY_TIMEOUT:-90}
OPENCLAW_VISUAL_MODE=${OPENCLAW_VISUAL_MODE:-describe}
OPENCLAW_VISUAL_MODEL=${OPENCLAW_VISUAL_MODEL:-qwen/qwen3.5-vl}
OPENCLAW_VISUAL_MAX_IMAGES=${OPENCLAW_VISUAL_MAX_IMAGES:-2}
OPENCLAW_VISUAL_TIMEOUT_MS=${OPENCLAW_VISUAL_TIMEOUT_MS:-30000}
VISUAL_PROBE_IMAGE=${VISUAL_PROBE_IMAGE:-}
CHECK_VISUAL_GATEWAY=${CHECK_VISUAL_GATEWAY:-1}
REQUIRE_VISUAL_GATEWAY=${REQUIRE_VISUAL_GATEWAY:-1}
HARNESS_MEMORY_BACKEND=${HARNESS_MEMORY_BACKEND:-fake}
HARNESS_MEMORY_SOURCE=${HARNESS_MEMORY_SOURCE:-episode-local}
OPENCLAW_ENABLE_SUBAGENT_CRITIC=${OPENCLAW_ENABLE_SUBAGENT_CRITIC:-1}
OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR=${OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR:-1}

cd "$(dirname "$0")/.."

echo "Visual mode/model: ${OPENCLAW_VISUAL_MODE}/${OPENCLAW_VISUAL_MODEL}"
echo "Visual max images: ${OPENCLAW_VISUAL_MAX_IMAGES}"
echo "OpenClaw gateway: ${OPENCLAW_GATEWAY_URL}"
echo "Output path: ${OUTPUT_PATH}"

if [[ "${OPENCLAW_VISUAL_MODE}" != "describe" ]]; then
  echo "evaluation_openclaw_visual_memory.sh expects OPENCLAW_VISUAL_MODE=describe" >&2
  exit 2
fi

if [[ "${CHECK_VISUAL_GATEWAY}" == "1" ]]; then
  visual_check_args=(
    --gateway_url "${OPENCLAW_GATEWAY_URL}"
    --timeout "${OPENCLAW_GATEWAY_TIMEOUT}"
    --model "${OPENCLAW_VISUAL_MODEL}"
  )
  if [[ -n "${VISUAL_PROBE_IMAGE}" ]]; then
    visual_check_args+=(--image_path "${VISUAL_PROBE_IMAGE}")
  fi
  if ! PYTHONPATH=.:src /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python \
    scripts/check_openclaw_visual_plan_gateway.py \
    "${visual_check_args[@]}"; then
    if [[ "${REQUIRE_VISUAL_GATEWAY}" == "1" ]]; then
      echo "OpenClaw visual gateway preflight failed and REQUIRE_VISUAL_GATEWAY=1" >&2
      exit 2
    fi
    echo "OpenClaw visual gateway preflight failed; continuing for diagnostic evaluation." >&2
  fi
fi

export OUTPUT_PATH
export OPENCLAW_GATEWAY_URL OPENCLAW_GATEWAY_TIMEOUT
export OPENCLAW_VISUAL_MODE OPENCLAW_VISUAL_MODEL OPENCLAW_VISUAL_MAX_IMAGES OPENCLAW_VISUAL_TIMEOUT_MS
export HARNESS_MEMORY_BACKEND HARNESS_MEMORY_SOURCE
export OPENCLAW_ENABLE_SUBAGENT_CRITIC OPENCLAW_ENABLE_SUBAGENT_MEMORY_CURATOR
export CHECK_GATEWAY=${CHECK_GATEWAY:-0}
export REQUIRE_GATEWAY=${REQUIRE_GATEWAY:-0}

./scripts/evaluation_openclaw_gateway.sh
