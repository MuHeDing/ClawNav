# External OpenClaw `/plan` Gateway Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the repo-local rule gateway with a real external OpenClaw planner gateway that controls VLN skill scheduling through `POST /plan`, including planner, critic, memory-curator, and keyframe recall decisions.

**Architecture:** ClawNav remains the VLN runtime and skill executor. External OpenClaw owns the high-level planning policy behind `POST /plan` and returns structured intents that ClawNav executes through repo-local tools. All online requests must be non-oracle and JSON-safe, while ClawNav preserves local fallback for gateway failures.

**Tech Stack:** Python, argparse, dataclasses, `requests`, `http.server` test doubles, pytest, JSON/JSONL traces, existing `OpenClawGatewayClient`, `OpenClawVLNRuntime`, `SkillRegistry`, `MemoryQuerySkill`, `MemoryWriteSkill`, `ProgressCriticSkill`, `ReplannerSkill`, `NavigationPolicySkill`, optional external OpenClaw HTTP service.

---

## Current State

The current gateway chain is runnable:

```text
evaluation_harness.py
-> OpenClawGatewayClient
-> POST /plan
-> OpenClawVLNRuntime
-> SkillRegistry tools
-> Habitat executor
```

The latest smoke run proves the HTTP path works:

```text
planner_backend=gateway: 29/29
planner_fallback=false: 29/29
planner_errors: 0
```

But the running gateway is `LocalOpenClawGatewayPlanner`, a repo-local rule-compatible service. It only chooses `recall_memory` at step 0 and every recall interval, then returns `act`. It does not run a real OpenClaw planner, critic, memory curator, or keyframe recall policy.

## Target Contract

External OpenClaw must expose:

```http
GET /health
POST /plan
```

`POST /plan` request:

```json
{
  "state": {
    "scene_id": "string",
    "episode_id": "string",
    "instruction": "string",
    "step_id": 0,
    "last_action": "TURN_LEFT"
  },
  "runtime_context": {
    "policy_action": "TURN_LEFT",
    "recent_actions": ["TURN_LEFT", "MOVE_FORWARD"],
    "memory_last_hits": [],
    "working_memory": {},
    "active_subgoal": "string"
  }
}
```

Allowed response intents:

```json
{
  "intent": "recall_memory",
  "tool_name": "MemoryQuerySkill",
  "arguments": {
    "text": "hallway sofa landmark",
    "step_id": 5,
    "reason": "openclaw_landmark_recall"
  },
  "reason": "openclaw_landmark_recall"
}
```

```json
{
  "intent": "write_memory",
  "tool_name": "MemoryWriteSkill",
  "arguments": {
    "write_type": "episodic_keyframe",
    "step_id": 10,
    "note": "turning at sofa landmark",
    "reason": "openclaw_keyframe_curator"
  },
  "reason": "openclaw_keyframe_curator"
}
```

```json
{
  "intent": "verify_progress",
  "tool_name": "ProgressCriticSkill",
  "arguments": {
    "reason": "openclaw_periodic_critic"
  },
  "reason": "openclaw_periodic_critic"
}
```

```json
{
  "intent": "replan",
  "tool_name": "ReplannerSkill",
  "arguments": {
    "failure_reason": "possible_stuck",
    "reason": "openclaw_recovery"
  },
  "reason": "openclaw_recovery"
}
```

```json
{
  "intent": "act",
  "tool_name": "NavigationPolicySkill",
  "arguments": {},
  "reason": "openclaw_default_act"
}
```

Forbidden request/response fields for online decisions:

```text
distance_to_goal
success
SPL
spl
oracle_path
oracle_shortest_path
oracle_shortest_path_action
future_observations
future_trajectory_frames
```

---

## Task 1: Freeze and Test the External `/plan` API Contract

**Files:**
- Create: `docs/protocols/openclaw-plan-gateway-api.md`
- Test: `tests/test_openclaw_gateway_contract.py`

**Step 1: Write the failing contract tests**

Create `tests/test_openclaw_gateway_contract.py`:

```python
from harness.openclaw.gateway import (
    OpenClawGatewayError,
    gateway_response_to_decision,
    json_safe_dict,
    strip_oracle_fields,
)


def test_plan_request_context_is_json_safe_and_non_oracle():
    context = {
        "policy_action": "TURN_LEFT",
        "success": True,
        "distance_to_goal": 1.0,
        "recent_frames": [object()],
        "recent_actions": ["TURN_LEFT"],
        "nested": {"ok": True, "SPL": 0.5, "bad": object()},
    }

    payload = json_safe_dict(strip_oracle_fields(context))

    assert payload == {
        "policy_action": "TURN_LEFT",
        "recent_actions": ["TURN_LEFT"],
        "nested": {"ok": True},
    }


def test_gateway_contract_accepts_all_runtime_intents():
    for intent, tool in [
        ("act", "NavigationPolicySkill"),
        ("recall_memory", "MemoryQuerySkill"),
        ("write_memory", "MemoryWriteSkill"),
        ("verify_progress", "ProgressCriticSkill"),
        ("replan", "ReplannerSkill"),
    ]:
        decision = gateway_response_to_decision(
            {
                "intent": intent,
                "tool_name": tool,
                "arguments": {},
                "reason": "contract_test",
            }
        )
        assert decision.intent == intent
        assert decision.tool_name == tool


def test_gateway_contract_rejects_missing_tool_name():
    try:
        gateway_response_to_decision({"intent": "act"})
    except OpenClawGatewayError as exc:
        assert "tool_name" in str(exc)
    else:
        raise AssertionError("expected OpenClawGatewayError")
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_gateway_contract.py -q
```

Expected: fail if `gateway_response_to_decision` rejects new intents or if docs do not exist after the doc test is added.

**Step 3: Write protocol documentation**

Create `docs/protocols/openclaw-plan-gateway-api.md` with:

```markdown
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
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_gateway_contract.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add docs/protocols/openclaw-plan-gateway-api.md tests/test_openclaw_gateway_contract.py
git commit -m "Document OpenClaw plan gateway contract"
```

---

## Task 2: Extend Runtime to Execute OpenClaw Planner Intents

**Files:**
- Modify: `src/harness/openclaw/runtime.py`
- Test: `tests/test_openclaw_runtime_bridge.py`

**Step 1: Write failing tests for new intents**

Append to `tests/test_openclaw_runtime_bridge.py`:

```python
class StaticPlanner:
    def __init__(self, decision):
        self.decision = decision

    def plan(self, state, runtime_context):
        return self.decision


class EchoWriteSkill(Skill):
    name = "MemoryWriteSkill"
    description = "Writes fake memory."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("memory_write", {"stored": True, **payload})


class EchoCriticSkill(Skill):
    name = "ProgressCriticSkill"
    description = "Returns fake critic result."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("critic", {"possible_stuck": False})


class EchoReplannerSkill(Skill):
    name = "ReplannerSkill"
    description = "Returns fake subgoal."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("replan", {"active_subgoal": "recover hallway"})


def make_full_runtime(decision):
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(EchoMemorySkill())
    registry.register(EchoWriteSkill())
    registry.register(EchoCriticSkill())
    registry.register(EchoReplannerSkill())
    return OpenClawVLNRuntime(
        tool_registry=registry,
        planner=StaticPlanner(decision),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )


def test_runtime_executes_write_memory_intent_before_action():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"step_id": 3, "note": "landmark"},
        reason="curator",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(make_state(step_id=3), payload={})

    assert result.ok is True
    assert result.runtime_metadata["planned_intent"] == "write_memory"
    assert result.runtime_metadata["tool_calls"][0]["tool_name"] == "MemoryWriteSkill"
    assert result.runtime_metadata["tool_calls"][-1]["tool_name"] == "NavigationPolicySkill"


def test_runtime_executes_critic_and_replan_intents_before_action():
    for intent, tool in [
        ("verify_progress", "ProgressCriticSkill"),
        ("replan", "ReplannerSkill"),
    ]:
        decision = OpenClawPlanDecision(
            intent=intent,
            tool_name=tool,
            arguments={"reason": "planner"},
            reason="planner",
            planner_backend="gateway",
        )
        runtime = make_full_runtime(decision)

        result = runtime.step(make_state(step_id=4), payload={})

        assert result.ok is True
        assert result.runtime_metadata["tool_calls"][0]["tool_name"] == tool
        assert result.runtime_metadata["tool_calls"][-1]["tool_name"] == "NavigationPolicySkill"
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py -q
```

Expected: fail because runtime only special-cases `recall_memory`.

**Step 3: Implement generic pre-action intent execution**

Modify `src/harness/openclaw/runtime.py`:

```python
PRE_ACTION_INTENTS = {
    "recall_memory",
    "write_memory",
    "verify_progress",
    "replan",
}
```

Replace:

```python
if decision.intent == "recall_memory":
    recall = self.tool_adapter.call_tool(...)
    tool_calls.append(recall)
```

with:

```python
if decision.intent in PRE_ACTION_INTENTS:
    tool_result = self.tool_adapter.call_tool(
        decision.tool_name,
        decision.arguments,
        state=state,
    )
    tool_calls.append(tool_result)
```

Keep the final `NavigationPolicySkill` call unchanged.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw/runtime.py tests/test_openclaw_runtime_bridge.py
git commit -m "Execute OpenClaw planner intents in runtime"
```

---

## Task 3: Add External OpenClaw Gateway Compliance Smoke Test

**Files:**
- Create: `scripts/check_openclaw_plan_gateway.py`
- Test: `tests/test_openclaw_gateway_check_script.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_gateway_check_script.py`:

```python
from scripts.check_openclaw_plan_gateway import build_probe_payload, validate_plan_response


def test_build_probe_payload_contains_no_oracle_fields():
    payload = build_probe_payload("go to kitchen")

    assert payload["state"]["instruction"] == "go to kitchen"
    assert "success" not in payload["runtime_context"]
    assert "distance_to_goal" not in payload["runtime_context"]


def test_validate_plan_response_accepts_valid_response():
    validate_plan_response(
        {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {},
            "reason": "ok",
        }
    )


def test_validate_plan_response_rejects_bad_response():
    try:
        validate_plan_response({"intent": "act"})
    except ValueError as exc:
        assert "tool_name" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_gateway_check_script.py -q
```

Expected: fail because the script does not exist.

**Step 3: Implement check script**

Create `scripts/check_openclaw_plan_gateway.py`:

```python
#!/usr/bin/env python
import argparse
import json
from typing import Any, Dict

import requests


ALLOWED_INTENTS = {"act", "recall_memory", "write_memory", "verify_progress", "replan"}
REQUIRED_KEYS = {"intent", "tool_name", "arguments", "reason"}


def build_probe_payload(instruction: str) -> Dict[str, Any]:
    return {
        "state": {
            "scene_id": "gateway_check",
            "episode_id": "gateway_check",
            "instruction": instruction,
            "step_id": 0,
            "last_action": None,
        },
        "runtime_context": {
            "policy_action": None,
            "recent_actions": [],
        },
    }


def validate_plan_response(data: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"missing required keys: {', '.join(sorted(missing))}")
    if data["intent"] not in ALLOWED_INTENTS:
        raise ValueError(f"unsupported intent: {data['intent']}")
    if not isinstance(data["arguments"], dict):
        raise ValueError("arguments must be an object")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway_url", default="http://127.0.0.1:8011")
    parser.add_argument("--instruction", default="go to kitchen")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    session = requests.Session()
    session.trust_env = False
    health = session.get(f"{args.gateway_url.rstrip('/')}/health", timeout=args.timeout)
    health.raise_for_status()
    response = session.post(
        f"{args.gateway_url.rstrip('/')}/plan",
        json=build_probe_payload(args.instruction),
        timeout=args.timeout,
    )
    response.raise_for_status()
    data = response.json()
    validate_plan_response(data)
    print(json.dumps({"ok": True, "response": data}, sort_keys=True))


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_gateway_check_script.py -q
```

Expected: pass.

**Step 5: Manually verify a running gateway**

Run:

```bash
PYTHONPATH=.:src python scripts/check_openclaw_plan_gateway.py \
  --gateway_url http://127.0.0.1:8011
```

Expected: prints `{"ok": true, ...}`.

**Step 6: Commit**

```bash
git add scripts/check_openclaw_plan_gateway.py tests/test_openclaw_gateway_check_script.py
git commit -m "Add OpenClaw gateway compliance check"
```

---

## Task 4: Add Keyframe Write Payload Support Without Passing Images to OpenClaw

**Files:**
- Modify: `src/evaluation_harness.py`
- Modify: `src/harness/openclaw/runtime.py`
- Test: `tests/test_evaluation_harness_openclaw_runtime.py`

**Step 1: Write failing test**

Append to `tests/test_evaluation_harness_openclaw_runtime.py`:

```python
def test_proxy_payload_exposes_keyframe_candidate_without_image_object(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(make_args(tmp_path), model=base_model)
    proxy = HarnessModelProxy(base_model, components)

    payload = proxy._runtime_payload(["frame0"], step_id=0)

    assert payload["keyframe_candidate"]["step_id"] == 0
    assert "image" not in payload["keyframe_candidate"]
    assert payload["keyframe_candidate"]["reason"] == "interval"
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py::test_proxy_payload_exposes_keyframe_candidate_without_image_object -q
```

Expected: fail because `_runtime_payload` does not exist.

**Step 3: Implement payload helper**

Modify `src/evaluation_harness.py`:

```python
def _runtime_payload(self, images, step_id: int) -> Dict[str, Any]:
    payload = {
        "recent_frames": list(images[:-1]),
        "policy_action": self.last_action_text,
    }
    if self.components["working_memory"].should_promote_keyframe(step_id):
        payload["keyframe_candidate"] = {
            "step_id": step_id,
            "reason": "interval",
            "has_current_image": bool(images),
        }
    return payload
```

Use it inside `call_model()`:

```python
runtime_result = runtime.step(state, self._runtime_payload(images, step_id))
```

Do not include raw PIL images in the gateway request. The external OpenClaw planner should decide `write_memory`; ClawNav executes `MemoryWriteSkill` using local state and saved keyframe path in a later task.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/evaluation_harness.py tests/test_evaluation_harness_openclaw_runtime.py
git commit -m "Expose keyframe candidates to OpenClaw planner"
```

---

## Task 5: Persist Episode Keyframes Locally for MemoryWriteSkill

**Files:**
- Modify: `src/evaluation_harness.py`
- Modify: `src/harness/openclaw/runtime.py`
- Test: `tests/test_evaluation_harness_openclaw_runtime.py`

**Step 1: Write failing test**

Append:

```python
def test_proxy_saves_keyframe_artifact_for_write_memory(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(make_args(tmp_path), model=base_model)
    proxy = HarnessModelProxy(base_model, components)

    path = proxy._save_keyframe_if_needed("frame0", step_id=0)

    assert path.endswith("keyframes/step_000000.txt")
    assert (tmp_path / "keyframes" / "step_000000.txt").exists()
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py::test_proxy_saves_keyframe_artifact_for_write_memory -q
```

Expected: fail because helper does not exist.

**Step 3: Implement minimal artifact saving**

Modify `src/evaluation_harness.py`:

```python
def _save_keyframe_if_needed(self, image, step_id: int) -> str:
    if image is None:
        return ""
    keyframe_dir = Path(self.components["config"].output_path) / "keyframes"
```

If `HarnessConfig` does not contain `output_path`, use `Path(args.output_path)` stored in components:

```python
components["output_path"] = Path(args.output_path)
```

Support real image objects:

```python
if hasattr(image, "save"):
    path = keyframe_dir / f"step_{step_id:06d}.png"
    image.save(path)
else:
    path = keyframe_dir / f"step_{step_id:06d}.txt"
    path.write_text(str(image), encoding="utf-8")
return str(path)
```

Add the saved path to `keyframe_candidate`:

```python
"image_path": self._save_keyframe_if_needed(images[-1] if images else None, step_id)
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/evaluation_harness.py tests/test_evaluation_harness_openclaw_runtime.py
git commit -m "Persist keyframe candidates for OpenClaw memory writes"
```

---

## Task 6: Wire External OpenClaw `write_memory` to Keyframe Memory

**Files:**
- Modify: `src/harness/openclaw/runtime.py`
- Test: `tests/test_openclaw_runtime_bridge.py`

**Step 1: Write failing test**

Append:

```python
def test_runtime_enriches_write_memory_with_keyframe_candidate():
    decision = OpenClawPlanDecision(
        intent="write_memory",
        tool_name="MemoryWriteSkill",
        arguments={"note": "landmark"},
        reason="curator",
        planner_backend="gateway",
    )
    runtime = make_full_runtime(decision)

    result = runtime.step(
        make_state(step_id=7),
        payload={
            "keyframe_candidate": {
                "step_id": 7,
                "image_path": "/tmp/keyframe.png",
                "reason": "interval",
            }
        },
    )

    first_call = result.runtime_metadata["tool_calls"][0]
    assert first_call["tool_name"] == "MemoryWriteSkill"
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py::test_runtime_enriches_write_memory_with_keyframe_candidate -q
```

Expected: fail until runtime merges candidate payload.

**Step 3: Merge write payload**

Modify pre-action intent execution in `src/harness/openclaw/runtime.py`:

```python
arguments = dict(decision.arguments)
if decision.intent == "write_memory":
    candidate = payload.get("keyframe_candidate") or {}
    arguments = {**candidate, **arguments}
    arguments.setdefault("write_type", "episodic_keyframe")
```

Use `arguments` for the tool call.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw/runtime.py tests/test_openclaw_runtime_bridge.py
git commit -m "Wire OpenClaw write memory intent to keyframes"
```

---

## Task 7: Add Real OpenClaw Gateway Runbook and Script Guardrails

**Files:**
- Modify: `scripts/evaluation_openclaw_gateway.sh`
- Create: `docs/runbooks/external-openclaw-gateway.md`
- Test: `tests/test_phase4_docs.py`

**Step 1: Write failing doc test**

Append to `tests/test_phase4_docs.py`:

```python
from pathlib import Path


def test_external_openclaw_gateway_runbook_exists():
    text = Path("docs/runbooks/external-openclaw-gateway.md").read_text(encoding="utf-8")

    assert "POST /plan" in text
    assert "planner_backend=gateway" in text
    assert "planner_fallback=false" in text
    assert "check_openclaw_plan_gateway.py" in text
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase4_docs.py::test_external_openclaw_gateway_runbook_exists -q
```

Expected: fail because runbook does not exist.

**Step 3: Write runbook**

Create `docs/runbooks/external-openclaw-gateway.md`:

```markdown
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
```

**Step 4: Add script preflight option**

Modify `scripts/evaluation_openclaw_gateway.sh`:

```bash
CHECK_GATEWAY=${CHECK_GATEWAY:-1}
if [[ "${CHECK_GATEWAY}" == "1" ]]; then
  PYTHONPATH=.:src /ssd/dingmuhe/anaconda3/envs/janusvln/bin/python \
    scripts/check_openclaw_plan_gateway.py \
    --gateway_url "${OPENCLAW_GATEWAY_URL}"
fi
```

Keep `CHECK_GATEWAY=0` available for local fallback experiments.

**Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase4_docs.py -q
bash -n scripts/evaluation_openclaw_gateway.sh
```

Expected: pass.

**Step 6: Commit**

```bash
git add docs/runbooks/external-openclaw-gateway.md scripts/evaluation_openclaw_gateway.sh tests/test_phase4_docs.py
git commit -m "Add external OpenClaw gateway runbook"
```

---

## Task 8: Verify End-to-End External Gateway Behavior

**Files:**
- No code changes unless verification fails.
- Outputs: `results/clawnav_openclaw_gateway/summary.json`
- Outputs: `results/clawnav_openclaw_gateway/harness_traces/harness_trace_rank0.jsonl`

**Step 1: Start real OpenClaw gateway**

Start external OpenClaw, not `scripts/start_openclaw_gateway.sh`.

It must listen on:

```text
http://127.0.0.1:8011
```

or set:

```bash
OPENCLAW_GATEWAY_URL=http://host:port
```

**Step 2: Check gateway contract**

Run:

```bash
PYTHONPATH=.:src python scripts/check_openclaw_plan_gateway.py \
  --gateway_url "${OPENCLAW_GATEWAY_URL:-http://127.0.0.1:8011}"
```

Expected: `{"ok": true, ...}`.

**Step 3: Run one-episode smoke**

Run:

```bash
OPENCLAW_GATEWAY_URL=http://127.0.0.1:8011 \
CUDA_VISIBLE_DEVICES=4 \
MASTER_PORT=20401 \
HARNESS_DEBUG_MAX_EPISODES=1 \
./scripts/evaluation_openclaw_gateway.sh
```

Expected: completes and writes `summary.json`.

**Step 4: Verify trace uses real gateway**

Run:

```bash
python - <<'PY'
import json, collections
p = "results/clawnav_openclaw_gateway/harness_traces/harness_trace_rank0.jsonl"
rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
print("steps", len(rows))
print("planner_backend", collections.Counter(r.get("planner_backend") for r in rows))
print("planner_fallback", collections.Counter(r.get("planner_fallback") for r in rows))
print("intents", collections.Counter(r.get("planned_intent") for r in rows))
print("errors", [r.get("planner_error") for r in rows if r.get("planner_error")][:3])
PY
```

Expected:

```text
planner_backend Counter({'gateway': N})
planner_fallback Counter({False: N})
errors []
```

**Step 5: Verify keyframe-related behavior**

If external OpenClaw returns `write_memory`, verify:

```bash
rg '"planned_intent": "write_memory"|MemoryWriteSkill|keyframe' \
  results/clawnav_openclaw_gateway/harness_traces/harness_trace_rank0.jsonl
```

Expected: at least one `write_memory` step in runs where OpenClaw memory curator decides to store a keyframe.

**Step 6: Commit verification notes**

If this produces a stable run, add a short note:

```bash
mkdir -p docs/results
```

Create `docs/results/YYYY-MM-DD-external-openclaw-gateway-smoke.md` with metrics, trace counts, and whether keyframe write/recall occurred.

Commit:

```bash
git add docs/results/YYYY-MM-DD-external-openclaw-gateway-smoke.md
git commit -m "Record external OpenClaw gateway smoke result"
```

---

## How This Solves Keyframe Recall

This plan turns keyframe recall from a partial interface into a closed loop:

```text
ClawNav observes current frame
-> ClawNav exposes JSON-safe keyframe_candidate metadata to OpenClaw
-> external OpenClaw memory curator decides write_memory
-> ClawNav executes MemoryWriteSkill with local image_path
-> external OpenClaw planner later decides recall_memory
-> ClawNav executes MemoryQuerySkill
-> MemoryManager returns memory_context_text + memory_images
-> NavigationPolicySkill receives memory context
-> trace records planned_intent, tool calls, memory hits, and fallback status
```

Important boundary:

```text
OpenClaw planner decides whether to write or recall.
ClawNav owns raw image files and actual skill execution.
```

This avoids sending PIL images through `/plan`, keeps OpenClaw requests JSON-safe, and prevents oracle leakage.

## Execution Order

Recommended order:

1. Task 1: freeze contract.
2. Task 2: make runtime execute all planner intents.
3. Task 3: add compliance checker.
4. Task 4: expose keyframe candidates.
5. Task 5: persist keyframe files.
6. Task 6: wire write_memory to keyframes.
7. Task 7: runbook and script preflight.
8. Task 8: verify with real external OpenClaw.

Do not claim research gains after Task 8 unless multi-episode ablations show metric improvement over baseline.
