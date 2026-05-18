#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8011}
RECALL_INTERVAL_STEPS=${RECALL_INTERVAL_STEPS:-5}
OPENCLAW_GATEWAY_WS_URL=${OPENCLAW_GATEWAY_WS_URL:-}
OPENCLAW_GATEWAY_TIMEOUT=${OPENCLAW_GATEWAY_TIMEOUT:-5}
OPENCLAW_PLANNER_MODE=${OPENCLAW_PLANNER_MODE:-agent}
OPENCLAW_AGENT_ID=${OPENCLAW_AGENT_ID:-main}
OPENCLAW_AGENT_TIMEOUT=${OPENCLAW_AGENT_TIMEOUT:-60}
OPENCLAW_PROFILE=${OPENCLAW_PROFILE:-}
OPENCLAW_VISUAL_MODE=${OPENCLAW_VISUAL_MODE:-path}
OPENCLAW_VISUAL_MAX_IMAGES=${OPENCLAW_VISUAL_MAX_IMAGES:-2}
OPENCLAW_VISUAL_TIMEOUT_MS=${OPENCLAW_VISUAL_TIMEOUT_MS:-30000}
OPENCLAW_VISUAL_MODEL=${OPENCLAW_VISUAL_MODEL:-}

cd "$(dirname "$0")/.."

args=(
  --host "${HOST}"
  --port "${PORT}"
  --recall_interval_steps "${RECALL_INTERVAL_STEPS}"
  --timeout "${OPENCLAW_GATEWAY_TIMEOUT}"
  --planner_mode "${OPENCLAW_PLANNER_MODE}"
  --agent_id "${OPENCLAW_AGENT_ID}"
  --agent_timeout "${OPENCLAW_AGENT_TIMEOUT}"
  --openclaw_visual_mode "${OPENCLAW_VISUAL_MODE}"
  --openclaw_visual_max_images "${OPENCLAW_VISUAL_MAX_IMAGES}"
  --openclaw_visual_timeout_ms "${OPENCLAW_VISUAL_TIMEOUT_MS}"
)
if [[ -n "${OPENCLAW_GATEWAY_WS_URL}" ]]; then
  args+=(--openclaw_gateway_url "${OPENCLAW_GATEWAY_WS_URL}")
fi
if [[ -n "${OPENCLAW_PROFILE}" ]]; then
  args+=(--openclaw_profile "${OPENCLAW_PROFILE}")
fi
if [[ -n "${OPENCLAW_VISUAL_MODEL}" ]]; then
  args+=(--openclaw_visual_model "${OPENCLAW_VISUAL_MODEL}")
fi

PYTHONPATH=src /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python \
  -m harness.openclaw.openclaw_cli_plan_gateway \
  "${args[@]}"
