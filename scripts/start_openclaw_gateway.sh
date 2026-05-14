#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8011}
RECALL_INTERVAL_STEPS=${RECALL_INTERVAL_STEPS:-5}

cd "$(dirname "$0")/.."

PYTHONPATH=src /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python \
  -m harness.openclaw.gateway_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --recall_interval_steps "${RECALL_INTERVAL_STEPS}"
