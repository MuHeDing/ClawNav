# OpenClaw Qwen Context Reduction Handoff

Date: 2026-05-23

## Current Status

The step-history token growth problem is fixed on the current ClawNav path.

Verified real OpenClaw agent traces:

```text
default profile, after Phase 1-4:
step=2 assembled=479 provider=16330 hidden=15851 exceeded=False
step=3 assembled=479 provider=16330 hidden=15851 exceeded=False
```

Interpretation:

- `provider_input_tokens` no longer grows with step id.
- `openclaw_session_mode=fresh_per_step` is active.
- The remaining hidden cost is fixed OpenClaw/Qwen-side overhead, not prior `/plan` history accumulation.
- Per-step Qwen input is still high at about 16.3k tokens, so the next goal is reducing fixed OpenClaw agent overhead.

## Implemented Code

Phase 1-4 implementation is already in the workspace.

Main files:

- `src/harness/openclaw/openclaw_cli_plan_gateway.py`
- `src/harness/openclaw/runtime.py`
- `src/harness/memory/context_engine.py`
- `tests/test_openclaw_cli_plan_gateway.py`
- `tests/test_openclaw_runtime_bridge.py`
- `tests/test_context_engine.py`

Key behaviors:

- `/plan` OpenClaw agent calls now use step-scoped session ids.
- Runtime metadata records `context_audit`.
- Context engine records:
  - `task_state.json`
  - `running_summary.md`
  - `decision_log.jsonl`
  - `error_fixes.jsonl`
  - `memory_index.jsonl`
  - `review_log.md`
  - `DREAMS.md`
- Runtime injects bounded `task_state`, `recent_step_summary`, and retrieved episode memory into later planner calls.
- `MemoryWriteSkill` outputs are mirrored into the context engine for later retrieval.

Fresh verification already passed:

```bash
PYTHONPATH=src pytest \
  tests/test_context_engine.py \
  tests/test_openclaw_runtime_bridge.py \
  tests/test_openclaw_cli_plan_gateway.py \
  tests/test_memory_manager.py -q
```

Result:

```text
67 passed
```

Compile check:

```bash
PYTHONPATH=src python -m py_compile \
  src/harness/memory/context_engine.py \
  src/harness/openclaw/runtime.py \
  src/harness/openclaw/openclaw_cli_plan_gateway.py
```

## Current Token Problem

Current measured `/plan` token split:

```text
ClawNav assembled prompt: ~479 tokens
OpenClaw/Qwen fixed overhead: ~15851 tokens
Provider input total: ~16330 tokens
```

The fixed overhead likely comes from the `openclaw agent` runtime layer:

- system/developer prompt
- agent runtime protocol
- model-visible skills/tool descriptions
- gateway/agent wrapper context
- provider-side scaffolding

The ClawNav payload is already small. Reducing `memory_context_text` or keyframe fields will not materially reduce the remaining 15.8k overhead.

## Minimal Profile Attempt

A dedicated profile was created:

```text
profile: clawnav-plan
config: ~/.openclaw-clawnav-plan/openclaw.json
```

Configured values:

```text
agents.defaults.skills = []
agents.defaults.model.primary = qwen/qwen3.5-flash
models.providers.qwen copied from default profile
auth/device identity copied from default profile
gateway copied from default profile with port 18790
```

Confirmed:

```bash
openclaw --profile clawnav-plan config validate
openclaw --profile clawnav-plan skills list --eligible --json
```

Results:

```text
Config valid
"skills": []
```

The profile can run the agent and returns non-fallback planner decisions after starting a gateway on port `18790`.

Temporary gateway command used:

```bash
openclaw --profile clawnav-plan gateway run --port 18790 --force --compact
```

The temporary gateway was killed after testing:

```bash
fuser -k 18790/tcp
openclaw --profile clawnav-plan gateway probe
```

Final probe result:

```text
Reachable: no
Connect: failed - connect ECONNREFUSED 127.0.0.1:18790
```

## Blocker

The minimal profile currently does not provide reliable provider usage numbers.

With `clawnav-plan`, session files record:

```json
{"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0}
```

This is not a real token reduction result. It means this profile/gateway path is not writing provider usage correctly, so the token drop cannot yet be measured.

Do not claim `clawnav-plan` reduced token count until usage is fixed or measured through another path.

## Tomorrow's Recommended Plan

### Step 1: Make usage observable for `clawnav-plan`

Start the profile gateway:

```bash
openclaw --profile clawnav-plan gateway run --port 18790 --force --compact
```

Health check:

```bash
openclaw --profile clawnav-plan health --json
```

Run the same two-step trace with:

```python
OpenClawCliPlanPlanner(
    planner_mode="agent",
    openclaw_profile="clawnav-plan",
    openclaw_session_dir="/ssd/dingmuhe/.openclaw-clawnav-plan/agents/main/sessions",
    agent_session_id="clawnav-plan-min-profile",
    agent_session_timestamp="<new-tag>",
    agent_timeout_s=120,
    agent_max_input_tokens=50000,
)
```

Expected good output should include nonzero usage:

```text
step=2 assembled=... provider=<nonzero> hidden=...
step=3 assembled=... provider=<same-or-close> hidden=...
```

If usage remains zero, inspect the profile gateway logs:

```bash
tail -n 200 /tmp/openclaw/openclaw-2026-05-23.log
```

Also inspect the generated session file without printing secrets:

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("/ssd/dingmuhe/.openclaw-clawnav-plan/agents/main/sessions")
for path in sorted(base.glob("*<new-tag>*.jsonl")):
    print("FILE", path.name)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        msg = obj.get("message") if isinstance(obj, dict) else None
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            print(msg.get("usage"))
PY
```

### Step 2: If usage becomes nonzero, compare against baseline

Baseline:

```text
default profile provider_input_tokens ~= 16330
```

Success target:

```text
clawnav-plan provider_input_tokens < 10000
```

Strong success:

```text
clawnav-plan provider_input_tokens < 5000
```

If `clawnav-plan` usage is nonzero but still around 16k, then skills were not the main source of overhead.

### Step 3: If minimal profile is not enough, switch to stateless model call

Candidate command:

```bash
openclaw capability model run \
  --model qwen/qwen3.5-flash \
  --prompt '<bounded ClawNav JSON prompt>' \
  --json
```

This path may bypass the `openclaw agent` runtime wrapper and should be tested as the next reduction path.

Goal:

```text
provider input ~= ClawNav assembled prompt + small provider overhead
expected range: 1000-4000 tokens
```

If this works, implement a new planner mode in `OpenClawCliPlanPlanner`, for example:

```text
planner_mode = "model_run"
```

Keep current `agent` mode as fallback until the new mode has enough tests and trace evidence.

## Important Warnings

- Do not modify the default OpenClaw profile unless explicitly needed.
- Do not print auth/profile secret contents.
- `clawnav-plan` profile files live outside the repo, under `~/.openclaw-clawnav-plan`.
- The repo worktree already has many unrelated dirty files. Avoid reverting them.
- A provider usage value of `0` is invalid evidence for token reduction.
- Continue to require step 2 and step 3 traces; one-step smoke is not enough.

## Useful Commands

Check current profile config:

```bash
openclaw --profile clawnav-plan config get agents.defaults.skills
openclaw --profile clawnav-plan config get agents.defaults.model.primary
openclaw --profile clawnav-plan config validate
```

Check skills:

```bash
openclaw --profile clawnav-plan skills list --eligible --json
```

Start temporary profile gateway:

```bash
openclaw --profile clawnav-plan gateway run --port 18790 --force --compact
```

Stop temporary gateway:

```bash
fuser -k 18790/tcp
openclaw --profile clawnav-plan gateway probe
```

Run focused tests:

```bash
PYTHONPATH=src pytest \
  tests/test_context_engine.py \
  tests/test_openclaw_runtime_bridge.py \
  tests/test_openclaw_cli_plan_gateway.py \
  tests/test_memory_manager.py -q
```

Compile touched modules:

```bash
PYTHONPATH=src python -m py_compile \
  src/harness/memory/context_engine.py \
  src/harness/openclaw/runtime.py \
  src/harness/openclaw/openclaw_cli_plan_gateway.py
```
