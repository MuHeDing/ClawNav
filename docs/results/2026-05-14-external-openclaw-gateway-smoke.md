# 2026-05-14 External OpenClaw Gateway Smoke

## Command

```bash
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
CUDA_VISIBLE_DEVICES=4 \
MASTER_PORT=20401 \
HARNESS_DEBUG_MAX_EPISODES=1 \
./scripts/evaluation_openclaw_gateway.sh
```

## Gateway Check

`scripts/check_openclaw_plan_gateway.py` returned `ok=true` before the smoke run.

## Summary

- `length`: 1
- `sucs_all`: 0.0
- `spls_all`: 0.0
- `oss_all`: 0.0
- `ones_all`: 10.676755905151367

## Trace Counts

- `steps`: 29
- `planner_backend`: `gateway` for 29 steps
- `planner_fallback`: `false` for 29 steps
- `planned_intent`: `act` for 23 steps, `recall_memory` for 6 steps
- `planner_error`: none

## Memory Behavior

The smoke run exercised gateway memory recall. The connected gateway did not return
`write_memory`, so no keyframe memory write was observed in this run.
