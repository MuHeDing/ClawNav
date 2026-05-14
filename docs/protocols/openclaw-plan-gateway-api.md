# OpenClaw `/plan` Gateway API

## Endpoints

- `GET /health`
- `POST /plan`

## Request

The planner receives only online, non-oracle, JSON-safe state.

## Response

Allowed intents:

- `act`
- `recall_memory`
- `write_memory`
- `verify_progress`
- `replan`

Each response must include `intent`, `tool_name`, `arguments`, and `reason`.

## No-Oracle Rule

Online planner decisions must not use `distance_to_goal`, `success`, `SPL`,
oracle paths, oracle actions, or future observations.
