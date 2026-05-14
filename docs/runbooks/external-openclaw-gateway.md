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

By default the adapter uses `OPENCLAW_PLANNER_MODE=agent`, so each `/plan`
request calls:

```bash
openclaw agent --agent main --json --message <planner-prompt>
```

The OpenClaw agent must have a working model configured, for example:

```bash
openclaw models auth --agent main login \
  --provider qwen \
  --method standard-api-key-cn \
  --set-default
openclaw models --agent main set qwen/qwen3.5-plus
openclaw gateway restart
openclaw agent --agent main --json --message '只回复 OK' --timeout 60
```

If the OpenClaw agent call fails or returns invalid JSON, the adapter falls back
inside `/plan` to the local rule planner and annotates the returned reason with
`openclaw_cli_agent_fallback:`. For the older health-only behavior, set:

```bash
OPENCLAW_PLANNER_MODE=heuristic ./scripts/start_openclaw_cli_plan_gateway.sh
```

## 2. Check compliance

Run:

```bash
PYTHONPATH=.:src python scripts/check_openclaw_plan_gateway.py \
  --gateway_url http://127.0.0.1:8011 \
  --timeout 90
```

## 3. Run smoke

Run:

```bash
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=90 \
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
