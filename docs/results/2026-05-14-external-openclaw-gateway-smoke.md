# 2026-05-14 External OpenClaw Gateway Smoke

## Command

```bash
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
CUDA_VISIBLE_DEVICES=4 \
MASTER_PORT=20401 \
./scripts/evaluation_openclaw_gateway.sh
```

## Gateway Check

`scripts/check_openclaw_plan_gateway.py` returned `ok=true` before the smoke run.

## Summary

- `length`: 5
- `episodes`: 1, 2, 3, 13, 14
- `sucs_all`: 0.4000000059604645
- `spls_all`: 0.4
- `oss_all`: 0.4000000059604645
- `ones_all`: 5.4220356941223145

## Trace Counts

- `steps`: 714
- `planner_backend`: `gateway` for 714 steps
- `planner_fallback`: `false` for 714 steps
- `planned_intent`: `act` for 569 steps, `recall_memory` for 145 steps
- `planner_error`: none

## Memory Behavior

The smoke run exercised gateway memory recall. The connected gateway did not return
`write_memory`, so no keyframe memory write was observed in this run.
