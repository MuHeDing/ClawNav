# ClawNav Phase 4 Research System Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the Phase 3 repo-local OpenClaw bridge into a research-complete OpenClaw-VLN system with optional external OpenClaw gateway support, robot executor abstraction, no-leakage long-term memory experiments, LLM/VLM subagent backends, and full ablation evidence.

**Architecture:** Keep the current Phase 2/3 baseline import-safe and make every new external service optional behind repo-local protocols. Add gateway clients, executor adapters, memory experiment manifests, subagent backends, and ablation tooling as opt-in layers selected by CLI flags. Preserve no-oracle online decision constraints and keep the existing `phase2` and `openclaw_bridge` paths working throughout.

**Tech Stack:** Python, dataclasses, argparse, pytest, requests, JSON/JSONL manifests, shell scripts, existing `harness.openclaw` bridge, existing `SkillRegistry`, existing `SpatialMemoryHttpClient`, Habitat discrete executor, optional external OpenClaw/robot HTTP gateways.

---

## Current State

Phase 3 is complete as a local runtime prototype:

- `--harness_runtime openclaw_bridge` runs planner -> tool adapter -> skill registry -> Habitat executor.
- External `openclaw`, `abot_claw`, and `roboclaw` packages are not required.
- `SERVICE.md` can supply SpatialMemory defaults when explicitly requested.
- Phase 3 tests and one-episode smoke verification pass.

Remaining research-system gaps:

- No real external OpenClaw package or gateway backend.
- No real robot executor abstraction beyond Habitat discrete commands.
- Scene-prior and train-scene-only long-term memory are not complete experiment systems.
- Planner, critic, and memory curator are rule/local code only, not LLM/VLM subagents.
- No full ablation suite proving navigation metric or harness-behavior gains.

## Global Constraints

- Keep `src/evaluation_harness.py` import-safe without external OpenClaw, robot, or LLM service packages.
- Do not change JanusVLN model internals.
- Do not enable LLM adaptive sparse attention.
- Do not add slow-fast active memory reuse.
- Do not make external OpenClaw, robot gateway, or LLM/VLM services required dependencies.
- Do not use oracle metrics for online planner/controller/critic/executor decisions:
  - `distance_to_goal`
  - `success`
  - `SPL`
  - oracle shortest path
  - oracle shortest path action
  - future observations
- Oracle metrics may remain diagnostics-only.
- Existing Phase 1, Phase 2, and Phase 3 tests must keep passing after every task.
- Every external integration must have a fake/local test double.
- Full evaluation scripts must preserve low-memory settings:
  - `--max_pixels 401408`
  - `--kv_start_size 8`
  - `--kv_recent_size 24`
  - `--num_history 8`

## Phase 4 Completion Criteria

- External OpenClaw gateway mode exists behind `openclaw_planner_backend="gateway"` and is optional.
- Gateway client has timeout, error, and fallback behavior covered by tests.
- A robot executor protocol exists and Habitat remains the default executor.
- Optional HTTP robot executor can translate action commands without importing robot packages.
- Scene-prior and train-scene-only memory experiments use explicit manifests with no-leakage validation.
- LLM/VLM planner, critic, and memory-curator subagents exist behind optional backends with fake test doubles.
- Ablation runner can launch baseline, Phase 2, Phase 3, memory-source, subagent, and executor variants.
- Metrics aggregator produces navigation metrics and harness-level metrics from output directories.
- Documentation distinguishes Phase 3 runtime smoke from Phase 4 research evidence.
- Full test sweep, py_compile, and at least one small ablation smoke pass.

---

## Task 0: Baseline Boundary and Phase 3 Regression

**Files:**
- Read: `docs/plans/2026-05-12-clawnav-phase3-openclaw-vln-runtime.md`
- Read: `docs/protocols/openclaw-vln-runtime.md`
- Test: no new files

**Step 1: Confirm Phase 3 tests still pass**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_openclaw_service_registry.py \
  tests/test_openclaw_planner.py \
  tests/test_openclaw_executor.py \
  tests/test_openclaw_runtime_bridge.py \
  tests/test_evaluation_harness_openclaw_runtime.py \
  tests/test_phase3_docs.py \
  -q
```

Expected: pass.

**Step 2: Confirm optional dependency boundary**

Run:

```bash
/ssd/dingmuhe/anaconda3/envs/janusvln/bin/python -c "import importlib.util; print({m: importlib.util.find_spec(m) is not None for m in ['openclaw','abot_claw','roboclaw']})"
```

Expected today:

```text
{'openclaw': False, 'abot_claw': False, 'roboclaw': False}
```

If any package appears later, do not import it at module import time.

**Step 3: Commit**

No commit required.

---

## Task 1: External OpenClaw Gateway Client Protocol

**Files:**
- Create: `src/harness/openclaw/gateway.py`
- Test: `tests/test_openclaw_gateway.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_gateway.py`:

```python
from harness.openclaw.gateway import (
    FakeOpenClawGatewayClient,
    OpenClawGatewayClient,
    OpenClawGatewayError,
)
from harness.types import VLNState


def make_state():
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=0,
        current_image=None,
    )


def test_fake_gateway_returns_plan_decision_without_oracle_fields():
    client = FakeOpenClawGatewayClient(
        response={
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {"distance_to_goal": 1.0, "text": "go"},
            "reason": "fake",
        }
    )

    decision = client.plan(make_state(), runtime_context={"success": True})

    assert decision.intent == "act"
    assert decision.tool_name == "NavigationPolicySkill"
    assert decision.arguments == {"text": "go"}
    assert decision.planner_backend == "gateway"


def test_gateway_client_builds_request_payload_without_oracle_fields():
    captured = {}

    def post_json(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {
            "intent": "recall_memory",
            "tool_name": "MemoryQuerySkill",
            "arguments": {"text": "go"},
            "reason": "gateway",
        }

    client = OpenClawGatewayClient(
        base_url="http://gateway",
        post_json=post_json,
        timeout_s=1.5,
    )

    decision = client.plan(make_state(), {"success": True, "policy_action": "STOP"})

    assert captured["url"] == "http://gateway/plan"
    assert captured["timeout"] == 1.5
    assert "success" not in captured["payload"]["runtime_context"]
    assert captured["payload"]["runtime_context"]["policy_action"] == "STOP"
    assert decision.reason == "gateway"


def test_gateway_client_raises_typed_error_on_bad_response():
    client = FakeOpenClawGatewayClient(response={"intent": "act"})

    try:
        client.plan(make_state(), {})
    except OpenClawGatewayError as exc:
        assert "tool_name" in str(exc)
    else:
        raise AssertionError("expected OpenClawGatewayError")
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_gateway.py -q
```

Expected: fail because `gateway.py` does not exist.

**Step 3: Implement gateway client**

Create `src/harness/openclaw/gateway.py`:

```python
from dataclasses import dataclass
from typing import Any, Callable, Dict

import requests

from harness.logging.harness_logger import ORACLE_KEYS
from harness.openclaw.planner import OpenClawPlanDecision
from harness.types import VLNState


class OpenClawGatewayError(RuntimeError):
    pass


def strip_oracle_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if key not in ORACLE_KEYS}


@dataclass
class OpenClawGatewayClient:
    base_url: str
    timeout_s: float = 5.0
    post_json: Callable[[str, Dict[str, Any], float], Dict[str, Any]] | None = None

    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> OpenClawPlanDecision:
        payload = {
            "state": {
                "scene_id": state.scene_id,
                "episode_id": state.episode_id,
                "instruction": state.instruction,
                "step_id": state.step_id,
                "last_action": state.last_action,
            },
            "runtime_context": strip_oracle_fields(runtime_context),
        }
        response = self._post(f"{self.base_url.rstrip('/')}/plan", payload)
        return gateway_response_to_decision(response)

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.post_json is not None:
            return self.post_json(url, payload, self.timeout_s)
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise OpenClawGatewayError(str(exc)) from exc
        if not isinstance(data, dict):
            raise OpenClawGatewayError("gateway response must be a JSON object")
        return data


class FakeOpenClawGatewayClient:
    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response

    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> OpenClawPlanDecision:
        return gateway_response_to_decision(self.response)


def gateway_response_to_decision(response: Dict[str, Any]) -> OpenClawPlanDecision:
    missing = [key for key in ("intent", "tool_name") if key not in response]
    if missing:
        raise OpenClawGatewayError(f"gateway response missing {', '.join(missing)}")
    arguments = response.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise OpenClawGatewayError("gateway arguments must be an object")
    return OpenClawPlanDecision(
        intent=str(response["intent"]),
        tool_name=str(response["tool_name"]),
        arguments=strip_oracle_fields(arguments),
        reason=str(response.get("reason", "")),
        planner_backend="gateway",
    )
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_gateway.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw/gateway.py tests/test_openclaw_gateway.py
git commit -m "Add optional OpenClaw gateway client"
```

---

## Task 2: Gateway Planner Backend Wiring

**Files:**
- Modify: `src/evaluation_harness.py`
- Modify: `src/harness/openclaw/runtime.py`
- Test: `tests/test_evaluation_harness_openclaw_runtime.py`
- Test: `tests/test_openclaw_runtime_bridge.py`

**Step 1: Write failing tests**

Add to `tests/test_evaluation_harness_openclaw_runtime.py`:

```python
def test_gateway_backend_requires_gateway_url(tmp_path):
    try:
        build_harness_components(
            make_args(
                tmp_path,
                harness_runtime="openclaw_bridge",
                openclaw_planner_backend="gateway",
                openclaw_gateway_url="",
            ),
            model=FakeBaseModel(),
        )
    except ValueError as exc:
        assert "openclaw_gateway_url" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

Add to `tests/test_openclaw_runtime_bridge.py`:

```python
from harness.openclaw.gateway import FakeOpenClawGatewayClient


def test_runtime_can_use_gateway_planner_client():
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=FakeOpenClawGatewayClient(
            {
                "intent": "act",
                "tool_name": "NavigationPolicySkill",
                "arguments": {},
                "reason": "gateway_test",
            }
        ),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )

    result = runtime.step(make_state(step_id=1), payload={})

    assert result.ok is True
    assert result.runtime_metadata["planner_backend"] == "gateway"
    assert result.runtime_metadata["planner_reason"] == "gateway_test"
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_evaluation_harness_openclaw_runtime.py \
  tests/test_openclaw_runtime_bridge.py \
  -q
```

Expected: fail until planner protocol and gateway wiring exist.

**Step 3: Loosen runtime planner type**

In `src/harness/openclaw/runtime.py`, change the planner annotation from `RuleOpenClawPlanner` to a protocol:

```python
from typing import Protocol


class OpenClawPlannerProtocol(Protocol):
    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> Any:
        ...
```

Then update `OpenClawVLNRuntime.__init__`:

```python
planner: OpenClawPlannerProtocol
```

**Step 4: Build gateway backend only when requested**

In `src/evaluation_harness.py`, inside the `openclaw_bridge` runtime construction:

```python
if config.openclaw_planner_backend == "gateway":
    if not config.openclaw_gateway_url:
        raise ValueError("openclaw_gateway_url is required for gateway planner backend")
    from harness.openclaw.gateway import OpenClawGatewayClient

    planner = OpenClawGatewayClient(base_url=config.openclaw_gateway_url)
else:
    planner = RuleOpenClawPlanner(recall_interval_steps=config.recall_interval_steps)
```

Pass `planner=planner` to `OpenClawVLNRuntime`.

**Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_openclaw_gateway.py \
  tests/test_evaluation_harness_openclaw_runtime.py \
  tests/test_openclaw_runtime_bridge.py \
  -q
```

Expected: pass.

**Step 6: Commit**

```bash
git add src/evaluation_harness.py src/harness/openclaw/runtime.py tests/test_evaluation_harness_openclaw_runtime.py tests/test_openclaw_runtime_bridge.py
git commit -m "Wire optional OpenClaw gateway planner backend"
```

---

## Task 3: Robot Executor Protocol and HTTP Executor

**Files:**
- Create: `src/harness/openclaw/robot_executor.py`
- Modify: `src/harness/config.py`
- Modify: `src/evaluation_harness.py`
- Test: `tests/test_openclaw_robot_executor.py`
- Test: `tests/test_evaluation_harness_openclaw_runtime.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_robot_executor.py`:

```python
from harness.openclaw.robot_executor import (
    FakeRobotExecutor,
    RobotExecutorCommand,
    RobotHttpExecutor,
)


def test_fake_robot_executor_records_command():
    executor = FakeRobotExecutor()

    command = executor.command_for_action("TURN_LEFT")

    assert command["executor"] == "robot_http"
    assert command["action_text"] == "TURN_LEFT"
    assert command["runtime_executor"] == "openclaw_robot"
    assert executor.commands == ["TURN_LEFT"]


def test_robot_http_executor_posts_action_without_oracle_fields():
    captured = {}

    def post_json(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"command_id": "cmd-1", "accepted": True}

    executor = RobotHttpExecutor(
        base_url="http://robot",
        post_json=post_json,
        timeout_s=2.0,
    )

    command = executor.command_for_action("MOVE_FORWARD")

    assert captured["url"] == "http://robot/action"
    assert captured["payload"] == {"action_text": "MOVE_FORWARD"}
    assert command["command_id"] == "cmd-1"
    assert command["runtime_executor"] == "openclaw_robot"


def test_robot_executor_command_is_dict_compatible():
    command = RobotExecutorCommand(
        executor="robot_http",
        action_text="STOP",
        runtime_executor="openclaw_robot",
        command_id="cmd-2",
    )

    assert command.to_dict()["action_text"] == "STOP"
```

Add to `tests/test_evaluation_harness_openclaw_runtime.py`:

```python
def test_robot_executor_requires_url_when_selected(tmp_path):
    try:
        build_harness_components(
            make_args(
                tmp_path,
                harness_runtime="openclaw_bridge",
                openclaw_executor_backend="robot_http",
                openclaw_robot_executor_url="",
            ),
            model=FakeBaseModel(),
        )
    except ValueError as exc:
        assert "openclaw_robot_executor_url" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_robot_executor.py tests/test_evaluation_harness_openclaw_runtime.py -q
```

Expected: fail because robot executor files and config fields do not exist.

**Step 3: Add config fields and parser args**

In `src/harness/config.py`:

```python
openclaw_executor_backend: str = "habitat"
openclaw_robot_executor_url: str = ""
```

In `src/evaluation_harness.py` parser:

```python
parser.add_argument("--openclaw_executor_backend", type=str, default="habitat")
parser.add_argument("--openclaw_robot_executor_url", type=str, default="")
```

Populate them in `build_harness_config()`.

**Step 4: Implement robot executor**

Create `src/harness/openclaw/robot_executor.py`:

```python
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import requests


@dataclass
class RobotExecutorCommand:
    executor: str
    action_text: str
    runtime_executor: str
    command_id: str = ""
    accepted: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executor": self.executor,
            "action_text": self.action_text,
            "runtime_executor": self.runtime_executor,
            "command_id": self.command_id,
            "accepted": self.accepted,
        }


class FakeRobotExecutor:
    def __init__(self) -> None:
        self.commands: List[str] = []

    def command_for_action(self, action_text: str) -> Dict[str, Any]:
        self.commands.append(action_text)
        return RobotExecutorCommand(
            executor="robot_http",
            action_text=action_text,
            runtime_executor="openclaw_robot",
        ).to_dict()


class RobotHttpExecutor:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        post_json: Callable[[str, Dict[str, Any], float], Dict[str, Any]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.post_json = post_json

    def command_for_action(self, action_text: str) -> Dict[str, Any]:
        response = self._post(f"{self.base_url}/action", {"action_text": action_text})
        return RobotExecutorCommand(
            executor="robot_http",
            action_text=action_text,
            runtime_executor="openclaw_robot",
            command_id=str(response.get("command_id", "")),
            accepted=bool(response.get("accepted", True)),
        ).to_dict()

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.post_json is not None:
            return self.post_json(url, payload, self.timeout_s)
        response = requests.post(url, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return {}
        return data
```

**Step 5: Wire executor backend**

In `build_harness_components()`:

```python
if config.openclaw_executor_backend == "robot_http":
    if not config.openclaw_robot_executor_url:
        raise ValueError("openclaw_robot_executor_url is required for robot_http executor")
    from harness.openclaw.robot_executor import RobotHttpExecutor

    executor = RobotHttpExecutor(config.openclaw_robot_executor_url)
else:
    executor = HabitatOpenClawExecutor(adapter)
```

Pass `executor=executor` to `OpenClawVLNRuntime`.

**Step 6: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_robot_executor.py tests/test_evaluation_harness_openclaw_runtime.py -q
```

Expected: pass.

**Step 7: Commit**

```bash
git add src/harness/openclaw/robot_executor.py src/harness/config.py src/evaluation_harness.py tests/test_openclaw_robot_executor.py tests/test_evaluation_harness_openclaw_runtime.py
git commit -m "Add optional OpenClaw robot executor backend"
```

---

## Task 4: Memory Source Manifest and No-Leakage Validator

**Files:**
- Create: `src/harness/memory/manifest.py`
- Test: `tests/test_memory_manifest.py`

**Step 1: Write failing tests**

Create `tests/test_memory_manifest.py`:

```python
import json

from harness.memory.manifest import (
    MemoryManifest,
    MemoryManifestError,
    load_memory_manifest,
)


def test_episode_local_manifest_is_valid_without_files(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"memory_source": "episode-local", "entries": []}),
        encoding="utf-8",
    )

    manifest = load_memory_manifest(path)

    assert manifest.memory_source == "episode-local"
    assert manifest.entries == []


def test_scene_prior_requires_exploration_split_and_no_target_instruction(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "memory_source": "scene-prior",
                "entries": [
                    {
                        "scene_id": "s1",
                        "split": "exploration",
                        "uses_target_instruction": False,
                        "uses_oracle_path": False,
                        "uses_future_observations": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_memory_manifest(path)

    assert manifest.memory_source == "scene-prior"


def test_manifest_rejects_oracle_or_future_leakage(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "memory_source": "scene-prior",
                "entries": [
                    {
                        "scene_id": "s1",
                        "split": "val_unseen",
                        "uses_target_instruction": True,
                        "uses_oracle_path": True,
                        "uses_future_observations": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_memory_manifest(path)
    except MemoryManifestError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("expected MemoryManifestError")
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_memory_manifest.py -q
```

Expected: fail because `manifest.py` does not exist.

**Step 3: Implement manifest loader**

Create `src/harness/memory/manifest.py`:

```python
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


class MemoryManifestError(ValueError):
    pass


@dataclass
class MemoryManifest:
    memory_source: str
    entries: List[Dict[str, Any]] = field(default_factory=list)


def load_memory_manifest(path: Path) -> MemoryManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = MemoryManifest(
        memory_source=str(data.get("memory_source", "")),
        entries=list(data.get("entries", [])),
    )
    validate_memory_manifest(manifest)
    return manifest


def validate_memory_manifest(manifest: MemoryManifest) -> None:
    if manifest.memory_source not in {"episode-local", "scene-prior", "train-scene-only"}:
        raise MemoryManifestError(f"unsupported memory_source {manifest.memory_source}")
    for entry in manifest.entries:
        if entry.get("uses_target_instruction"):
            raise MemoryManifestError("leakage: target instruction used to build memory")
        if entry.get("uses_oracle_path"):
            raise MemoryManifestError("leakage: oracle path used to build memory")
        if entry.get("uses_future_observations"):
            raise MemoryManifestError("leakage: future observations used to build memory")
        if manifest.memory_source == "scene-prior" and entry.get("split") != "exploration":
            raise MemoryManifestError("leakage: scene-prior memory must come from exploration split")
        if manifest.memory_source == "train-scene-only" and entry.get("split") != "train":
            raise MemoryManifestError("leakage: train-scene-only memory must come from train split")
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_memory_manifest.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/memory/manifest.py tests/test_memory_manifest.py
git commit -m "Add no-leakage memory manifest validator"
```

---

## Task 5: Scene-Prior and Train-Scene Memory Experiment Scripts

**Files:**
- Create: `scripts/build_scene_prior_memory_manifest.py`
- Create: `scripts/evaluation_openclaw_scene_prior.sh`
- Create: `scripts/evaluation_openclaw_train_scene_memory.sh`
- Test: `tests/test_memory_experiment_scripts.py`

**Step 1: Write failing tests**

Create `tests/test_memory_experiment_scripts.py`:

```python
from pathlib import Path


def test_scene_prior_script_uses_manifest_and_spatial_http():
    text = Path("scripts/evaluation_openclaw_scene_prior.sh").read_text(encoding="utf-8")

    assert "--harness_runtime openclaw_bridge" in text
    assert "--harness_memory_backend spatial_http" in text
    assert "--harness_memory_source scene-prior" in text
    assert "--memory_manifest_path" in text
    assert "--max_pixels 401408" in text
    assert "--use_llm_adaptive_sparse_attention" not in text


def test_train_scene_script_uses_train_scene_memory_source():
    text = Path("scripts/evaluation_openclaw_train_scene_memory.sh").read_text(encoding="utf-8")

    assert "--harness_memory_source train-scene-only" in text
    assert "--memory_manifest_path" in text
    assert "--enable_slow_fast" not in text


def test_manifest_builder_mentions_no_leakage_fields():
    text = Path("scripts/build_scene_prior_memory_manifest.py").read_text(encoding="utf-8")

    assert "uses_target_instruction" in text
    assert "uses_oracle_path" in text
    assert "uses_future_observations" in text
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_memory_experiment_scripts.py -q
```

Expected: fail because scripts do not exist.

**Step 3: Add parser arg for manifest path**

Modify `src/harness/config.py`:

```python
memory_manifest_path: str = ""
```

Modify `src/evaluation_harness.py` parser:

```python
parser.add_argument("--memory_manifest_path", type=str, default="")
```

Populate it in `build_harness_config()`.

If `memory_manifest_path` is set, load and validate it with `load_memory_manifest(Path(...))`.

**Step 4: Create manifest builder skeleton**

Create `scripts/build_scene_prior_memory_manifest.py`:

```python
#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def build_manifest(memory_source: str, split: str) -> dict:
    return {
        "memory_source": memory_source,
        "entries": [
            {
                "split": split,
                "uses_target_instruction": False,
                "uses_oracle_path": False,
                "uses_future_observations": False,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_source", choices=["scene-prior", "train-scene-only"], required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    split = "exploration" if args.memory_source == "scene-prior" else "train"
    Path(args.output_path).write_text(
        json.dumps(build_manifest(args.memory_source, split), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

**Step 5: Create evaluation scripts**

Create `scripts/evaluation_openclaw_scene_prior.sh` and `scripts/evaluation_openclaw_train_scene_memory.sh` using the same low-memory settings as `scripts/evaluation_openclaw_vln_runtime.sh`, plus:

```bash
--harness_memory_backend spatial_http \
--harness_memory_source scene-prior \
--memory_manifest_path "${MEMORY_MANIFEST_PATH}" \
```

and:

```bash
--harness_memory_source train-scene-only \
```

respectively.

**Step 6: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_memory_manifest.py tests/test_memory_experiment_scripts.py tests/test_evaluation_harness_imports.py -q
```

Expected: pass.

**Step 7: Commit**

```bash
git add src/harness/config.py src/evaluation_harness.py scripts/build_scene_prior_memory_manifest.py scripts/evaluation_openclaw_scene_prior.sh scripts/evaluation_openclaw_train_scene_memory.sh tests/test_memory_experiment_scripts.py
git commit -m "Add long-term memory experiment manifests and scripts"
```

---

## Task 6: LLM/VLM Subagent Client Protocol

**Files:**
- Create: `src/harness/openclaw/subagents.py`
- Test: `tests/test_openclaw_subagents.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_subagents.py`:

```python
from harness.openclaw.subagents import (
    FakeSubagentClient,
    SubagentRequest,
    sanitize_subagent_context,
)


def test_sanitize_subagent_context_removes_oracle_fields():
    sanitized = sanitize_subagent_context(
        {"distance_to_goal": 1.0, "success": True, "instruction": "go"}
    )

    assert sanitized == {"instruction": "go"}


def test_fake_subagent_client_returns_structured_response():
    client = FakeSubagentClient({"intent": "act", "reason": "ok"})
    request = SubagentRequest(
        role="planner",
        instruction="go",
        context={"success": True, "step_id": 1},
    )

    response = client.call(request)

    assert response["intent"] == "act"
    assert client.requests[0].context == {"step_id": 1}
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_subagents.py -q
```

Expected: fail because `subagents.py` does not exist.

**Step 3: Implement subagent protocol**

Create `src/harness/openclaw/subagents.py`:

```python
from dataclasses import dataclass
from typing import Any, Dict, List

from harness.logging.harness_logger import ORACLE_KEYS


@dataclass
class SubagentRequest:
    role: str
    instruction: str
    context: Dict[str, Any]


def sanitize_subagent_context(context: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in context.items() if key not in ORACLE_KEYS}


class FakeSubagentClient:
    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        self.requests: List[SubagentRequest] = []

    def call(self, request: SubagentRequest) -> Dict[str, Any]:
        sanitized = SubagentRequest(
            role=request.role,
            instruction=request.instruction,
            context=sanitize_subagent_context(request.context),
        )
        self.requests.append(sanitized)
        return dict(self.response)
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_subagents.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw/subagents.py tests/test_openclaw_subagents.py
git commit -m "Add optional LLM VLM subagent protocol"
```

---

## Task 7: Subagent Planner, Critic, and Memory Curator Backends

**Files:**
- Modify: `src/harness/openclaw/planner.py`
- Create: `src/harness/openclaw/critic.py`
- Create: `src/harness/openclaw/memory_curator.py`
- Modify: `src/harness/config.py`
- Modify: `src/evaluation_harness.py`
- Test: `tests/test_openclaw_subagent_backends.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_subagent_backends.py`:

```python
from harness.openclaw.critic import SubagentProgressCritic
from harness.openclaw.memory_curator import SubagentMemoryCurator
from harness.openclaw.planner import SubagentOpenClawPlanner
from harness.openclaw.subagents import FakeSubagentClient
from harness.types import VLNState


def make_state():
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=3,
        current_image=None,
    )


def test_subagent_planner_returns_plan_decision():
    planner = SubagentOpenClawPlanner(
        FakeSubagentClient(
            {
                "intent": "act",
                "tool_name": "NavigationPolicySkill",
                "arguments": {},
                "reason": "subagent",
            }
        )
    )

    decision = planner.plan(make_state(), {"success": True})

    assert decision.planner_backend == "subagent"
    assert decision.reason == "subagent"


def test_subagent_critic_returns_non_oracle_feedback():
    critic = SubagentProgressCritic(
        FakeSubagentClient({"status": "stuck", "reason": "repeated_turns"})
    )

    result = critic.evaluate(make_state(), {"distance_to_goal": 1.0, "actions": ["TURN_LEFT"]})

    assert result["status"] == "stuck"
    assert "distance_to_goal" not in critic.client.requests[0].context


def test_subagent_memory_curator_returns_write_decision():
    curator = SubagentMemoryCurator(
        FakeSubagentClient({"should_write": True, "reason": "landmark"})
    )

    result = curator.should_write(make_state(), {"success": True, "visual_novelty": 0.7})

    assert result["should_write"] is True
    assert curator.client.requests[0].context == {"visual_novelty": 0.7}
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_subagent_backends.py -q
```

Expected: fail because modules do not exist.

**Step 3: Implement subagent planner**

Add to `src/harness/openclaw/planner.py`:

```python
from harness.openclaw.subagents import SubagentRequest


class SubagentOpenClawPlanner:
    def __init__(self, client) -> None:
        self.client = client

    def plan(self, state: VLNState, runtime_context: Dict[str, Any]) -> OpenClawPlanDecision:
        response = self.client.call(
            SubagentRequest(
                role="planner",
                instruction=state.instruction,
                context=runtime_context,
            )
        )
        return OpenClawPlanDecision(
            intent=str(response.get("intent", "act")),
            tool_name=str(response.get("tool_name", "NavigationPolicySkill")),
            arguments=dict(response.get("arguments", {})),
            reason=str(response.get("reason", "subagent")),
            planner_backend="subagent",
        )
```

**Step 4: Implement critic and memory curator**

Create `src/harness/openclaw/critic.py`:

```python
from typing import Any, Dict

from harness.openclaw.subagents import SubagentRequest
from harness.types import VLNState


class SubagentProgressCritic:
    def __init__(self, client) -> None:
        self.client = client

    def evaluate(self, state: VLNState, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.call(
            SubagentRequest(
                role="critic",
                instruction=state.instruction,
                context=context,
            )
        )
```

Create `src/harness/openclaw/memory_curator.py`:

```python
from typing import Any, Dict

from harness.openclaw.subagents import SubagentRequest
from harness.types import VLNState


class SubagentMemoryCurator:
    def __init__(self, client) -> None:
        self.client = client

    def should_write(self, state: VLNState, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.call(
            SubagentRequest(
                role="memory_curator",
                instruction=state.instruction,
                context=context,
            )
        )
```

**Step 5: Add config and parser selectors**

Add config fields:

```python
openclaw_subagent_backend: str = "fake"
openclaw_enable_subagent_planner: bool = False
openclaw_enable_subagent_critic: bool = False
openclaw_enable_subagent_memory_curator: bool = False
```

Add parser args with `action="store_true"` for the booleans.

Wire only the planner selector in this task:

```python
if config.openclaw_enable_subagent_planner:
    from harness.openclaw.planner import SubagentOpenClawPlanner
    from harness.openclaw.subagents import FakeSubagentClient

    planner = SubagentOpenClawPlanner(
        FakeSubagentClient(
            {
                "intent": "act",
                "tool_name": "NavigationPolicySkill",
                "arguments": {},
                "reason": "fake_subagent",
            }
        )
    )
```

Keep critic and curator objects available but not online by default.

**Step 6: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_subagents.py tests/test_openclaw_subagent_backends.py tests/test_evaluation_harness_imports.py -q
```

Expected: pass.

**Step 7: Commit**

```bash
git add src/harness/openclaw/planner.py src/harness/openclaw/critic.py src/harness/openclaw/memory_curator.py src/harness/config.py src/evaluation_harness.py tests/test_openclaw_subagent_backends.py
git commit -m "Add optional OpenClaw subagent backends"
```

---

## Task 8: Ablation Matrix Runner

**Files:**
- Create: `scripts/run_openclaw_vln_ablation_matrix.py`
- Test: `tests/test_openclaw_ablation_runner.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_ablation_runner.py`:

```python
from scripts.run_openclaw_vln_ablation_matrix import build_matrix


def test_ablation_matrix_contains_required_variants():
    names = [item["name"] for item in build_matrix()]

    assert "baseline_lowmem" in names
    assert "phase2_memory_recall" in names
    assert "phase3_openclaw_bridge" in names
    assert "scene_prior_memory" in names
    assert "train_scene_memory" in names
    assert "subagent_planner" in names


def test_ablation_matrix_preserves_lowmem_settings():
    for item in build_matrix():
        args = " ".join(item["args"])
        assert "--max_pixels 401408" in args
        assert "--kv_start_size 8" in args
        assert "--kv_recent_size 24" in args
        assert "--num_history 8" in args
        assert "--use_llm_adaptive_sparse_attention" not in args
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_ablation_runner.py -q
```

Expected: fail because runner does not exist.

**Step 3: Implement matrix builder**

Create `scripts/run_openclaw_vln_ablation_matrix.py`:

```python
#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path


LOWMEM_ARGS = [
    "--num_history 8",
    "--max_pixels 401408",
    "--kv_start_size 8",
    "--kv_recent_size 24",
]


def build_matrix():
    common = [
        "src/evaluation_harness.py",
        "--model_path ${MODEL_PATH}",
        "--habitat_config_path config/vln_r2r.yaml",
        *LOWMEM_ARGS,
    ]
    return [
        {"name": "baseline_lowmem", "args": common + ["--harness_runtime phase2", "--harness_mode act_only"]},
        {"name": "phase2_memory_recall", "args": common + ["--harness_runtime phase2", "--harness_mode memory_recall"]},
        {"name": "phase3_openclaw_bridge", "args": common + ["--harness_runtime openclaw_bridge"]},
        {"name": "scene_prior_memory", "args": common + ["--harness_runtime openclaw_bridge", "--harness_memory_source scene-prior"]},
        {"name": "train_scene_memory", "args": common + ["--harness_runtime openclaw_bridge", "--harness_memory_source train-scene-only"]},
        {"name": "subagent_planner", "args": common + ["--harness_runtime openclaw_bridge", "--openclaw_enable_subagent_planner"]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_root", default="results/openclaw_vln_ablation")
    args = parser.parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for item in build_matrix():
        command = ["bash", "-lc", " ".join(item["args"])]
        print(item["name"], command)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_ablation_runner.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/run_openclaw_vln_ablation_matrix.py tests/test_openclaw_ablation_runner.py
git commit -m "Add OpenClaw VLN ablation matrix runner"
```

---

## Task 9: Ablation Metrics Aggregator

**Files:**
- Create: `scripts/summarize_openclaw_vln_ablation.py`
- Test: `tests/test_openclaw_ablation_summary.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_ablation_summary.py`:

```python
import json

from scripts.summarize_openclaw_vln_ablation import summarize_run


def test_summarize_run_reads_navigation_and_harness_metrics(tmp_path):
    run = tmp_path / "run"
    traces = run / "harness_traces"
    traces.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"sucs_all": 0.5, "spls_all": 0.25, "length": 2}),
        encoding="utf-8",
    )
    (traces / "harness_trace_rank0.jsonl").write_text(
        json.dumps(
            {
                "runtime_mode": "openclaw_bridge",
                "planned_intent": "recall_memory",
                "tool_calls": [{"tool_name": "MemoryQuerySkill"}],
                "oracle_metrics_used_for_decision": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(run)

    assert summary["success"] == 0.5
    assert summary["spl"] == 0.25
    assert summary["trace_steps"] == 1
    assert summary["memory_recall_steps"] == 1
    assert summary["oracle_leakage_steps"] == 0
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_ablation_summary.py -q
```

Expected: fail because summarizer does not exist.

**Step 3: Implement summarizer**

Create `scripts/summarize_openclaw_vln_ablation.py`:

```python
#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict


def summarize_run(path: Path) -> Dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    trace_path = path / "harness_traces" / "harness_trace_rank0.jsonl"
    trace_steps = 0
    memory_recall_steps = 0
    oracle_leakage_steps = 0
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trace_steps += 1
            record = json.loads(line)
            if record.get("planned_intent") == "recall_memory" or record.get("intent") == "recall_memory":
                memory_recall_steps += 1
            if record.get("oracle_metrics_used_for_decision"):
                oracle_leakage_steps += 1
    return {
        "success": summary.get("sucs_all", 0.0),
        "spl": summary.get("spls_all", 0.0),
        "length": summary.get("length", 0),
        "trace_steps": trace_steps,
        "memory_recall_steps": memory_recall_steps,
        "oracle_leakage_steps": oracle_leakage_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        print(json.dumps({"run": run_dir, **summarize_run(Path(run_dir))}, sort_keys=True))


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=.:src pytest tests/test_openclaw_ablation_summary.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/summarize_openclaw_vln_ablation.py tests/test_openclaw_ablation_summary.py
git commit -m "Add OpenClaw VLN ablation summarizer"
```

---

## Task 10: Phase 4 Documentation

**Files:**
- Create: `docs/protocols/openclaw-vln-research-system.md`
- Test: `tests/test_phase4_docs.py`

**Step 1: Write failing tests**

Create `tests/test_phase4_docs.py`:

```python
from pathlib import Path


def test_phase4_research_system_doc_exists():
    text = Path("docs/protocols/openclaw-vln-research-system.md").read_text(encoding="utf-8")

    assert "External OpenClaw Gateway" in text
    assert "Robot Executor" in text
    assert "Scene-Prior Memory" in text
    assert "Train-Scene-Only Memory" in text
    assert "LLM/VLM Subagents" in text
    assert "Ablation Matrix" in text
    assert "No Oracle Decision Inputs" in text
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase4_docs.py -q
```

Expected: fail because docs do not exist.

**Step 3: Write docs**

Create `docs/protocols/openclaw-vln-research-system.md` with sections:

```text
# OpenClaw VLN Research System

## External OpenClaw Gateway
## Robot Executor
## Scene-Prior Memory
## Train-Scene-Only Memory
## LLM/VLM Subagents
## Ablation Matrix
## No Oracle Decision Inputs
## Evidence Required Before Research Claims
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase4_docs.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add docs/protocols/openclaw-vln-research-system.md tests/test_phase4_docs.py
git commit -m "Document OpenClaw VLN research system"
```

---

## Task 11: Phase 4 Full Test Sweep

**Files:**
- No code files unless fixes are required.

**Step 1: Run Phase 4 tests**

Run:

```bash
PYTHONPATH=.:src pytest \
  tests/test_openclaw_gateway.py \
  tests/test_openclaw_robot_executor.py \
  tests/test_memory_manifest.py \
  tests/test_memory_experiment_scripts.py \
  tests/test_openclaw_subagents.py \
  tests/test_openclaw_subagent_backends.py \
  tests/test_openclaw_ablation_runner.py \
  tests/test_openclaw_ablation_summary.py \
  tests/test_phase4_docs.py \
  -q
```

Expected: pass.

**Step 2: Run Phase 3 regression tests**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_openclaw_service_registry.py \
  tests/test_openclaw_planner.py \
  tests/test_openclaw_executor.py \
  tests/test_openclaw_runtime_bridge.py \
  tests/test_evaluation_harness_openclaw_runtime.py \
  tests/test_phase3_docs.py \
  -q
```

Expected: pass.

**Step 3: Run existing harness tests**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_harness_types.py \
  tests/test_task_memory.py \
  tests/test_working_memory.py \
  tests/test_spatial_memory_client.py \
  tests/test_memory_manager.py \
  tests/test_skill_registry.py \
  tests/test_navigation_policy_skill.py \
  tests/test_memory_skills.py \
  tests/test_progress_critic_replanner.py \
  tests/test_harness_controller.py \
  tests/test_harness_logger.py \
  tests/test_habitat_vln_adapter.py \
  tests/test_evaluation_harness_imports.py \
  tests/test_evaluation_scripts.py \
  -q
```

Expected: pass.

**Step 4: Compile changed modules**

Run:

```bash
python -m py_compile \
  src/evaluation_harness.py \
  src/harness/config.py \
  src/harness/openclaw/gateway.py \
  src/harness/openclaw/robot_executor.py \
  src/harness/openclaw/subagents.py \
  src/harness/openclaw/critic.py \
  src/harness/openclaw/memory_curator.py \
  src/harness/openclaw/planner.py \
  src/harness/openclaw/runtime.py \
  src/harness/memory/manifest.py \
  scripts/build_scene_prior_memory_manifest.py \
  scripts/run_openclaw_vln_ablation_matrix.py \
  scripts/summarize_openclaw_vln_ablation.py
```

Expected: exit 0.

**Step 5: Commit fixes**

If changes were needed:

```bash
git add <fixed-files>
git commit -m "Stabilize OpenClaw VLN research system tests"
```

---

## Task 12: Phase 4 Small Ablation Smoke

**Files:**
- No code files unless bugs are found.

**Step 1: Dry-run the ablation matrix**

Run:

```bash
PYTHONPATH=.:src python scripts/run_openclaw_vln_ablation_matrix.py --dry_run
```

Expected:

- Prints all required variants.
- Does not execute evaluation.

**Step 2: Run one baseline and one Phase 4 variant**

Run a one-episode baseline:

```bash
CUDA_VISIBLE_DEVICES=6 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20401 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_phase4_smoke_baseline \
  --harness_runtime phase2 \
  --harness_mode act_only \
  --harness_memory_backend fake \
  --harness_memory_source episode-local
```

Run one Phase 4 variant:

```bash
CUDA_VISIBLE_DEVICES=6 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20402 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_phase4_smoke_subagent_planner \
  --harness_runtime openclaw_bridge \
  --harness_mode memory_recall \
  --harness_memory_backend fake \
  --harness_memory_source episode-local \
  --openclaw_enable_subagent_planner
```

Expected:

- Both commands exit 0.
- Both output directories contain `summary.json`.
- Phase 4 variant trace contains `planner_backend=subagent`.
- `oracle_metrics_used_for_decision=false`.

**Step 3: Summarize outputs**

Run:

```bash
PYTHONPATH=.:src python scripts/summarize_openclaw_vln_ablation.py \
  results/clawnav_phase4_smoke_baseline \
  results/clawnav_phase4_smoke_subagent_planner
```

Expected:

- Prints one JSON line per run.
- Includes navigation metrics and harness metrics.
- `oracle_leakage_steps` is 0 for both runs.

**Step 4: Commit only code fixes**

Do not commit `results/` outputs.

---

## Recommended Implementation Order

1. Task 0: Baseline boundary and Phase 3 regression
2. Task 1: External OpenClaw gateway client protocol
3. Task 2: Gateway planner backend wiring
4. Task 3: Robot executor protocol and HTTP executor
5. Task 4: Memory source manifest and no-leakage validator
6. Task 5: Scene-prior and train-scene memory experiment scripts
7. Task 6: LLM/VLM subagent client protocol
8. Task 7: Subagent planner, critic, and memory curator backends
9. Task 8: Ablation matrix runner
10. Task 9: Ablation metrics aggregator
11. Task 10: Phase 4 documentation
12. Task 11: Full test sweep
13. Task 12: Small ablation smoke

## Notes for Execution

- Work from `ClawNav`, not the outer `Navigation_Claw` repository.
- Prefer one commit per task unless the user explicitly asks to batch commits.
- Keep `--harness_runtime phase2` as a stable baseline path.
- Keep `--harness_runtime openclaw_bridge` as the OpenClaw-style runtime path.
- Treat gateway, robot executor, memory manifests, and subagents as optional runtime integrations.
- Never import optional gateway packages at module import time.
- Keep every online decision non-oracle.
- Results can support research claims only after Task 12 and a larger ablation suite, not from one-episode smoke alone.
