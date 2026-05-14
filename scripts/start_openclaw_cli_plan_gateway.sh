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

cd "$(dirname "$0")/.."

args=(
  --host "${HOST}"
  --port "${PORT}"
  --recall_interval_steps "${RECALL_INTERVAL_STEPS}"
  --timeout "${OPENCLAW_GATEWAY_TIMEOUT}"
  --planner_mode "${OPENCLAW_PLANNER_MODE}"
  --agent_id "${OPENCLAW_AGENT_ID}"
  --agent_timeout "${OPENCLAW_AGENT_TIMEOUT}"
)
if [[ -n "${OPENCLAW_GATEWAY_WS_URL}" ]]; then
  args+=(--openclaw_gateway_url "${OPENCLAW_GATEWAY_WS_URL}")
fi

PYTHONPATH=src /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python \
  -m harness.openclaw.openclaw_cli_plan_gateway \
  "${args[@]}"
