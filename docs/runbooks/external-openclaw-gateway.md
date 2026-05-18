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

To enable OpenClaw visual analysis with Qwen, first verify the image capability
on a saved frame:

```bash
openclaw capability image describe \
  --file /path/to/sample.png \
  --model "${OPENCLAW_VISUAL_MODEL:-qwen/qwen3.5-vl}" \
  --json
```

Then start the adapter with visual describe mode:

```bash
OPENCLAW_VISUAL_MODE=describe \
OPENCLAW_VISUAL_MODEL=qwen/qwen3.5-vl \
OPENCLAW_VISUAL_MAX_IMAGES=2 \
OPENCLAW_VISUAL_TIMEOUT_MS=30000 \
HOST=127.0.0.1 PORT=8011 ./scripts/start_openclaw_cli_plan_gateway.sh
```

In this mode, ClawNav still sends JSON-safe image paths through `/plan`, while
the adapter calls OpenClaw's image capability to produce compact visual
observations for the planner and visual memory writer.

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

For visual-memory runs, verify both OpenClaw's Qwen image capability and a
`/plan` request carrying current/keyframe image paths:

```bash
PYTHONPATH=.:src python scripts/check_openclaw_visual_plan_gateway.py \
  --gateway_url http://127.0.0.1:8011 \
  --model "${OPENCLAW_VISUAL_MODEL:-qwen/qwen3.5-vl}" \
  --image_path /path/to/sample.png \
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

For the visual-memory smoke, start the adapter with
`OPENCLAW_VISUAL_MODE=describe` as shown above, then run:

```bash
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
OPENCLAW_GATEWAY_TIMEOUT=90 \
OPENCLAW_VISUAL_MODEL=qwen/qwen3.5-vl \
HARNESS_DEBUG_MAX_EPISODES=5 \
./scripts/evaluation_openclaw_visual_memory.sh
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
