#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:?Set DATA_PATH to the selected evaluation dataset}
EPISODE_KEYS=${EPISODE_KEYS:-2azQ1b91cZZ:10,2azQ1b91cZZ:11,2azQ1b91cZZ:12,2azQ1b91cZZ:16,2azQ1b91cZZ:70,2azQ1b91cZZ:1393}
OPENCLAW_STAGE_PLAN_MANIFEST_PATH=${OPENCLAW_STAGE_PLAN_MANIFEST_PATH:?Set OPENCLAW_STAGE_PLAN_MANIFEST_PATH}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
STAGED_MEMORY_OFF_OUTPUT_PATH=${STAGED_MEMORY_OFF_OUTPUT_PATH:-results/qwen_staged_memory_off_6ep_${RUN_TIMESTAMP}}
STAGED_MEMORY_ON_OUTPUT_PATH=${STAGED_MEMORY_ON_OUTPUT_PATH:-results/qwen_staged_memory_on_6ep_${RUN_TIMESTAMP}}
STAGED_MEMORY_COMPARISON_MANIFEST=${STAGED_MEMORY_COMPARISON_MANIFEST:-results/staged_memory_ablation_manifest_${RUN_TIMESTAMP}.json}
STAGED_MEMORY_COMPARISON_OUTPUT=${STAGED_MEMORY_COMPARISON_OUTPUT:-results/staged_memory_ablation_comparison_${RUN_TIMESTAMP}.json}
HARNESS_DEBUG_MAX_EPISODES=${HARNESS_DEBUG_MAX_EPISODES:-6}
MAX_STEPS=${MAX_STEPS:-200}

if [[ ! -f "${OPENCLAW_STAGE_PLAN_MANIFEST_PATH}" ]]; then
  QWEN_STAGE_PLAN_CAPTURE=1 \
  QWEN_ABLATION_PROFILE=staged_memory_on \
  OPENCLAW_STAGED_VISUAL_MEMORY_ENABLED=1 \
  OPENCLAW_QWEN_OUTPUT_SCHEMA=route_v3_staged \
  DATA_PATH="${DATA_PATH}" \
  EPISODE_KEYS="${EPISODE_KEYS}" \
  OPENCLAW_STAGE_PLAN_MANIFEST_PATH="${OPENCLAW_STAGE_PLAN_MANIFEST_PATH}" \
  bash scripts/run_qwen.sh
fi

QWEN_ABLATION_PROFILE=staged_memory_off \
DATA_PATH="${DATA_PATH}" \
OPENCLAW_STAGE_PLAN_MANIFEST_PATH="${OPENCLAW_STAGE_PLAN_MANIFEST_PATH}" \
EPISODE_KEYS="${EPISODE_KEYS}" \
HARNESS_DEBUG_MAX_EPISODES="${HARNESS_DEBUG_MAX_EPISODES}" \
OUTPUT_PATH="${STAGED_MEMORY_OFF_OUTPUT_PATH}" \
MAX_STEPS="${MAX_STEPS}" \
bash scripts/run_qwen.sh

QWEN_ABLATION_PROFILE=staged_memory_on \
DATA_PATH="${DATA_PATH}" \
OPENCLAW_STAGE_PLAN_MANIFEST_PATH="${OPENCLAW_STAGE_PLAN_MANIFEST_PATH}" \
EPISODE_KEYS="${EPISODE_KEYS}" \
HARNESS_DEBUG_MAX_EPISODES="${HARNESS_DEBUG_MAX_EPISODES}" \
OUTPUT_PATH="${STAGED_MEMORY_ON_OUTPUT_PATH}" \
MAX_STEPS="${MAX_STEPS}" \
bash scripts/run_qwen.sh

mkdir -p "$(dirname "${STAGED_MEMORY_COMPARISON_MANIFEST}")"
python - "${STAGED_MEMORY_COMPARISON_MANIFEST}" \
  "${OPENCLAW_STAGE_PLAN_MANIFEST_PATH}" \
  "${EPISODE_KEYS}" \
  "${STAGED_MEMORY_OFF_OUTPUT_PATH}" \
  "${STAGED_MEMORY_ON_OUTPUT_PATH}" <<'PY'
import json
from pathlib import Path
import sys

output, stage_manifest, keys, off_path, on_path = sys.argv[1:]
payload = {
    "comparison_type": "staged_memory",
    "stage_plan_manifest": str(Path(stage_manifest).resolve()),
    "expected_episode_keys": keys.split(","),
    "arms": {
        "staged_memory_off": {
            "result": str((Path(off_path) / "result.json").resolve()),
            "trace": str((Path(off_path) / "harness_traces/harness_trace_rank0.jsonl").resolve()),
            "treatment": "off_ablation",
        },
        "staged_memory_on": {
            "result": str((Path(on_path) / "result.json").resolve()),
            "trace": str((Path(on_path) / "harness_traces/harness_trace_rank0.jsonl").resolve()),
            "treatment": "on",
        },
    },
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

PYTHONPATH=src:. python scripts/compare_qwen_direct_policy_results.py \
  --ablation-manifest "${STAGED_MEMORY_COMPARISON_MANIFEST}" \
  --output "${STAGED_MEMORY_COMPARISON_OUTPUT}"
