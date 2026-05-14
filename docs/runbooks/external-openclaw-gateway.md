# External OpenClaw Gateway Runbook

## 1. Start real OpenClaw

External OpenClaw must expose:

- `GET /health`
- `POST /plan`

## 2. Check compliance

Run:

```bash
PYTHONPATH=.:src python scripts/check_openclaw_plan_gateway.py \
  --gateway_url http://127.0.0.1:8011
```

## 3. Run smoke

Run:

```bash
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
HARNESS_DEBUG_MAX_EPISODES=1 \
./scripts/evaluation_openclaw_gateway.sh
```

## 4. Verify trace

Expected:

- `planner_backend=gateway`
- `planner_fallback=false`

If `planner_fallback=true`, inspect `planner_error`.
