#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
QWEN_ABLATION_PROFILE=${QWEN_ABLATION_PROFILE:-dynamic_on}
EPISODE_KEYS=${EPISODE_KEYS:-"2azQ1b91cZZ:10,2azQ1b91cZZ:11,2azQ1b91cZZ:12,2azQ1b91cZZ:16,2azQ1b91cZZ:70,2azQ1b91cZZ:1393"}
HARNESS_DEBUG_MAX_EPISODES=${HARNESS_DEBUG_MAX_EPISODES:-6}
OPENCLAW_GATEWAY_PORT=${OPENCLAW_GATEWAY_PORT:-18013}
OPENCLAW_KILL_EXISTING_GATEWAY=${OPENCLAW_KILL_EXISTING_GATEWAY:-1}
OUTPUT_PATH=${OUTPUT_PATH:-"results/qwen_latest_${QWEN_ABLATION_PROFILE}_6ep_${RUN_TIMESTAMP}"}

export QWEN_ABLATION_PROFILE
export EPISODE_KEYS
export HARNESS_DEBUG_MAX_EPISODES
export OPENCLAW_GATEWAY_PORT
export OPENCLAW_KILL_EXISTING_GATEWAY
export OUTPUT_PATH

# The local adapter must not inherit a URL that points back to itself.
unset OPENCLAW_GATEWAY_URL OPENCLAW_GATEWAY_WS_URL

printf 'Qwen profile: %s\n' "${QWEN_ABLATION_PROFILE}"
printf 'Episode keys: %s\n' "${EPISODE_KEYS}"
printf 'Gateway port: %s\n' "${OPENCLAW_GATEWAY_PORT}"
printf 'Output path: %s\n' "${OUTPUT_PATH}"

exec bash scripts/run_qwen.sh
