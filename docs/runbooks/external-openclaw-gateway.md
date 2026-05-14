# External OpenClaw Gateway Runbook

## 1. Start real OpenClaw

External OpenClaw must expose:

- `GET /health`
- `POST /plan`

If you are using the OpenClaw CLI WebSocket gateway, start the HTTP `/plan`
adapter in one terminal:

```bash
openclaw gateway restart
HOST=127.0.0.1 PORT=8011 ./scripts/start_openclaw_cli_plan_gateway.sh
```

The adapter checks `openclaw gateway call health --json` before serving each
planner decision. If the OpenClaw CLI gateway becomes unavailable, `/plan`
returns an error and ClawNav falls back to its local rule planner.

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
HARNESS_DEBUG_MAX_EPISODES=5 \
./scripts/evaluation_openclaw_gateway.sh
```

For strict experiments that should fail before model loading when `/plan` is not
healthy, set:

```bash
REQUIRE_GATEWAY=1
```

## 4. Verify trace

Expected:

- `planner_backend=gateway`
- `planner_fallback=false`

If `planner_fallback=true`, inspect `planner_error`.
