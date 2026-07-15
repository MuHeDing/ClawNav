#!/usr/bin/env bash
set -euo pipefail

GATEWAY_HOST=127.0.0.1
GATEWAY_PORT=${OPENCLAW_GATEWAY_PORT:-8013}
OPENCLAW_GATEWAY_URL=${OPENCLAW_GATEWAY_URL:-http://${GATEWAY_HOST}:${GATEWAY_PORT}}

check_port_in_use() {
  local host="$1"
  local port="$2"
  if python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

try:
    with socket.create_connection((host, port), timeout=0.5):
        raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
  then
    return 0
  fi
  return 1
}

if check_port_in_use "${GATEWAY_HOST}" "${GATEWAY_PORT}"; then
  echo "OpenClaw gateway port ${GATEWAY_PORT} is already in use." >&2
if [ "${OPENCLAW_KILL_EXISTING_GATEWAY:-0}" = "1" ]; then
    if command -v lsof >/dev/null 2>&1; then
      while IFS= read -r pid; do
        [ -z "${pid}" ] && continue
        echo "Killing existing process ${pid} bound to ${GATEWAY_HOST}:${GATEWAY_PORT}"
        kill "${pid}" >/dev/null 2>&1 || true
      done < <(lsof -tiTCP:${GATEWAY_PORT} -sTCP:LISTEN || true)
      sleep 1
      if check_port_in_use "${GATEWAY_HOST}" "${GATEWAY_PORT}"; then
        echo "Failed to free port ${GATEWAY_PORT}. Please stop the process manually." >&2
        exit 2
      fi
    else
      echo "lsof not available. Set OPENCLAW_KILL_EXISTING_GATEWAY=1 and install lsof, or free port ${GATEWAY_PORT} manually." >&2
      exit 2
    fi
  else
    echo "Set OPENCLAW_KILL_EXISTING_GATEWAY=1 to auto-kill the process, or set OPENCLAW_GATEWAY_PORT to another free port." >&2
    exit 2
  fi
fi

QWEN_DISABLE_PROXY=${QWEN_DISABLE_PROXY:-1}
if [[ "${QWEN_DISABLE_PROXY}" == "1" ]]; then
  export HTTPS_PROXY=""
  export HTTP_PROXY=""
  export ALL_PROXY=""
  export https_proxy=""
  export http_proxy=""
  export all_proxy=""
  export NO_PROXY="127.0.0.1,localhost,::1"
  export no_proxy="127.0.0.1,localhost,::1"
  echo "Qwen requests are running with local proxy disabled."
fi

OPENCLAW_QWEN_API_RETRIES=${OPENCLAW_QWEN_API_RETRIES:-2}
OPENCLAW_QWEN_API_RETRY_BACKOFF_S=${OPENCLAW_QWEN_API_RETRY_BACKOFF_S:-1}
OPENCLAW_MODEL_FAST_MODE=${OPENCLAW_MODEL_FAST_MODE:-qwen_text_only}
OPENCLAW_EVAL_GATEWAY_TIMEOUT=${OPENCLAW_EVAL_GATEWAY_TIMEOUT:-660}
OPENCLAW_ADAPTER_HEALTH_TIMEOUT=${OPENCLAW_ADAPTER_HEALTH_TIMEOUT:-15}
OPENCLAW_ADAPTER_STARTUP_TIMEOUT=${OPENCLAW_ADAPTER_STARTUP_TIMEOUT:-180}
OPENCLAW_GATEWAY_TOKEN=${OPENCLAW_GATEWAY_TOKEN:-}
OPENCLAW_GATEWAY_WORKDIR=${OPENCLAW_HOME:-/tmp/clawnav_openclaw_home}
OPENCLAW_CLI_HOME=${OPENCLAW_CLI_HOME:-/tmp}
OPENCLAW_AUTH_PROFILES_PATH=${OPENCLAW_AUTH_PROFILES_PATH:-/ssd/dingmuhe/.openclaw/agents/main/agent/auth-profiles.json}
if [[ -z "${OPENCLAW_GATEWAY_TOKEN}" ]] && [[ -f /ssd/dingmuhe/.openclaw/openclaw.json ]]; then
  OPENCLAW_GATEWAY_TOKEN=$( \
    /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python -c \
    "import json;print(json.load(open('/ssd/dingmuhe/.openclaw/openclaw.json'))['gateway']['auth']['token'])" \
  )
fi

mkdir -p "${OPENCLAW_GATEWAY_WORKDIR}"

# Start CLI plan gateway in background.
env -u OPENCLAW_GATEWAY_PORT \
HOST="${GATEWAY_HOST}" \
PORT="${GATEWAY_PORT}" \
HOME="${OPENCLAW_CLI_HOME}" \
OPENCLAW_HOME="${OPENCLAW_GATEWAY_WORKDIR}" \
OPENCLAW_MODEL_PROVIDER=qwen_api \
OPENCLAW_MODEL_MAX_IMAGES="${OPENCLAW_MODEL_MAX_IMAGES:-8}" \
OPENCLAW_MODEL_FAST_MODE="${OPENCLAW_MODEL_FAST_MODE}" \
OPENCLAW_PLANNER_MODE=model \
POLICY_BACKEND=qwen_direct \
OPENCLAW_GATEWAY_TIMEOUT="${OPENCLAW_ADAPTER_HEALTH_TIMEOUT}" \
OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN}" \
OPENCLAW_AUTH_PROFILES_PATH="${OPENCLAW_AUTH_PROFILES_PATH}" \
OPENCLAW_QWEN_API_RETRIES="${OPENCLAW_QWEN_API_RETRIES}" \
OPENCLAW_QWEN_API_RETRY_BACKOFF_S="${OPENCLAW_QWEN_API_RETRY_BACKOFF_S}" \
OPENCLAW_QWEN_BASE_URL="${OPENCLAW_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" \
./scripts/start_openclaw_cli_plan_gateway.sh &
GATEWAY_PID=$!

cleanup() {
  if ps -p "${GATEWAY_PID}" >/dev/null 2>&1; then
    kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Wait for gateway /health before launching evaluation. The local /health endpoint
# also checks the OpenClaw CLI gateway, so a 2s curl timeout is too aggressive.
/ssd/dingmuhe/anaconda3/envs/janusvln/bin/python - "${OPENCLAW_GATEWAY_URL}" "${OPENCLAW_ADAPTER_STARTUP_TIMEOUT}" "${OPENCLAW_ADAPTER_HEALTH_TIMEOUT}" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

gateway_url = sys.argv[1].rstrip("/")
startup_timeout_s = float(sys.argv[2])
health_timeout_s = float(sys.argv[3])
deadline = time.time() + startup_timeout_s
last_error = ""

while time.time() < deadline:
    try:
        with urllib.request.urlopen(
            f"{gateway_url}/health",
            timeout=health_timeout_s,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("ok") and data.get("service") == "openclaw_cli_plan_gateway":
            raise SystemExit(0)
        last_error = f"unhealthy response: {data}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        last_error = f"{exc.__class__.__name__}: {exc}"
    print(f"waiting for OpenClaw gateway health: {last_error}", file=sys.stderr, flush=True)
    time.sleep(2)

print(
    f"OpenClaw gateway failed to become ready within {startup_timeout_s:g}s: {last_error}",
    file=sys.stderr,
)
raise SystemExit(2)
PY

#EPISODE_KEYS="2azQ1b91cZZ:70,2azQ1b91cZZ:1393,2azQ1b91cZZ:1159,2azQ1b91cZZ:160,2azQ1b91cZZ:778,2azQ1b91cZZ:1756,2azQ1b91cZZ:76,2azQ1b91cZZ:71,2azQ1b91cZZ:72,2azQ1b91cZZ:45"
EPISODE_KEYS="2azQ1b91cZZ:10,2azQ1b91cZZ:11,2azQ1b91cZZ:12,2azQ1b91cZZ:16"
EVAL_VARS=(
  OPENCLAW_HOME=/tmp/clawnav_openclaw_home
  POLICY_BACKEND=qwen_direct
  OPENCLAW_GATEWAY_URL="${OPENCLAW_GATEWAY_URL}"
  OPENCLAW_GATEWAY_TIMEOUT="${OPENCLAW_EVAL_GATEWAY_TIMEOUT}"
  OPENCLAW_AUTH_PROFILES_PATH="${OPENCLAW_AUTH_PROFILES_PATH}"
  REQUIRE_GATEWAY=1
  CHECK_GATEWAY=1
  HARNESS_USE_DEFAULT_EPISODE_KEYS=0
  HARNESS_EPISODE_KEYS="${EPISODE_KEYS}"
  HARNESS_DEBUG_MAX_EPISODES="${HARNESS_DEBUG_MAX_EPISODES:-10}"
  SAVE_VIDEO=1
  SAVE_VIDEO_RATIO=1.0
  SAVE_STEP_ARTIFACTS_WITH_VIDEO_ONLY=1
  OPENCLAW_KEYFRAME_POLICY_MODE="${OPENCLAW_KEYFRAME_POLICY_MODE:-event_gated_smoke}"
  OPENCLAW_KEYFRAME_MIN_GAP_STEPS="${OPENCLAW_KEYFRAME_MIN_GAP_STEPS:-5}"
  OPENCLAW_KEYFRAME_EPISODE_CAP="${OPENCLAW_KEYFRAME_EPISODE_CAP:-64}"
  OPENCLAW_KEYFRAME_COVERAGE_GAP_STEPS="${OPENCLAW_KEYFRAME_COVERAGE_GAP_STEPS:-20}"
  OPENCLAW_MAP_ASSIST_MODE="${OPENCLAW_MAP_ASSIST_MODE:-off}"
  OPENCLAW_MAP_FRAME_INTERVAL_STEPS="${OPENCLAW_MAP_FRAME_INTERVAL_STEPS:-5}"
  OPENCLAW_MOTION_FEEDBACK_ENABLED="${OPENCLAW_MOTION_FEEDBACK_ENABLED:-0}"
  OPENCLAW_FORWARD_STALL_ODOMETRY_ENABLED="${OPENCLAW_FORWARD_STALL_ODOMETRY_ENABLED:-0}"
  OPENCLAW_MAP_COLLISION_OVERLAY_ENABLED="${OPENCLAW_MAP_COLLISION_OVERLAY_ENABLED:-0}"
  MAX_STEPS=200
  CUDA_VISIBLE_DEVICES=0
  MASTER_PORT=20420
  NPROC_PER_NODE=1
  TOKENIZERS_PARALLELISM=false
  OPENCLAW_QWEN_BASE_URL="${OPENCLAW_QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
)
if [[ -n "${OUTPUT_PATH:-}" ]]; then
  EVAL_VARS+=("OUTPUT_PATH=${OUTPUT_PATH}")
fi

env "${EVAL_VARS[@]}" bash scripts/evaluation_openclaw_gateway.sh
