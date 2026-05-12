# ClawNav Phase 3 OpenClaw VLN Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect the Phase 2 OpenClaw-compatible ClawNav harness to an optional OpenClaw/ABot-Claw-style runtime layer while preserving the current no-OpenClaw baseline path.

**Architecture:** Keep JanusVLN and the Phase 1/2 harness import-safe. Add a repo-local OpenClaw runtime bridge that can load ABot-Claw workspace/service metadata, expose VLN skills through the Phase 2 tool adapter, run a planner/executor loop, and optionally call SpatialMemory over HTTP. Treat external OpenClaw packages and services as optional runtime integrations, not required Python dependencies.

**Tech Stack:** Python, dataclasses, pytest, argparse, requests, existing `harness.openclaw.tool_adapter`, existing `SkillRegistry`, ABot-Claw `openclaw_layer` workspace files, Habitat discrete executor.

---

## Current Readiness

Phase 3 can start now.

Confirmed local state:

- Phase 2 tests, harness regressions, py_compile, and smoke tests pass.
- `ABot-Claw_Muhe/openclaw_layer` exists locally.
- The `janusvln` Python environment does not currently provide importable `openclaw`, `abot_claw`, or `roboclaw` packages.

Implication:

- Phase 3 must not start by importing an unavailable `openclaw` package.
- First implementation should be a ClawNav-local runtime bridge that can consume OpenClaw/ABot-Claw workspace files and call existing ClawNav tools.
- If a real OpenClaw gateway/package becomes available later, it should plug into this bridge behind an optional adapter.

## Global Constraints

- Keep `src/evaluation_harness.py` import-safe without OpenClaw installed.
- Do not change JanusVLN model internals.
- Do not enable LLM adaptive sparse attention.
- Do not add slow-fast active memory reuse.
- Do not make `openclaw`, ABot-Claw, or RoboClaw packages required dependencies.
- Do not use oracle metrics for online planner/controller/critic/executor decisions:
  - `distance_to_goal`
  - `success`
  - `SPL`
  - oracle shortest path
  - oracle shortest path action
  - future observations
- Oracle metrics may remain diagnostics-only.
- Existing Phase 1 and Phase 2 tests must keep passing after every task.
- New runtime code must use repo-local dataclasses/protocols first; avoid new dependencies unless tests prove they are necessary.

## Phase 3 Completion Criteria

- A repo-local OpenClaw VLN runtime bridge can be imported without external OpenClaw packages.
- The bridge can list VLN tools through the existing Phase 2 `OpenClawToolAdapter`.
- The bridge can run one runtime step with a planner result, skill call, and Habitat executor command.
- The runtime can load ABot-Claw service metadata from `openclaw_layer/SERVICE.md` or explicit CLI args.
- SpatialMemory defaults can align with ABot-Claw service registry without breaking existing `spatial_http` behavior.
- `evaluation_harness.py` can run with `--harness_runtime openclaw_bridge` and still supports `--harness_runtime phase2`.
- Harness traces include OpenClaw runtime metadata while preserving Phase 2 trace fields.
- Documentation explains the difference between Phase 2 compatibility and Phase 3 runtime execution.
- Phase 2 tests, existing harness tests, Phase 3 tests, lightweight regression, py_compile, and a one-episode smoke pass.

---

## Task 0: Phase 3 Baseline and Dependency Boundary

**Files:**
- Read: `docs/plans/2026-05-08-claw-style-harness-for-fast-janusvln.md`
- Read: `docs/plans/2026-05-11-clawnav-phase2-openclaw-compatible-interface.md`
- Read: `/ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/README.md`
- Read: `/ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/SERVICE.md`
- Test: no new files

**Step 1: Confirm current tests still pass**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_skill_manifest_schema.py \
  tests/test_openclaw_tool_adapter.py \
  tests/test_spatial_memory_protocol.py \
  tests/test_embodiment_adapter_contract.py \
  tests/test_phase2_docs.py \
  -q
```

Expected: pass.

**Step 2: Confirm OpenClaw package boundary**

Run:

```bash
/ssd/dingmuhe/anaconda3/envs/janusvln/bin/python -c "import importlib.util; print({m: importlib.util.find_spec(m) is not None for m in ['openclaw','abot_claw','roboclaw']})"
```

Expected today:

```text
{'openclaw': False, 'abot_claw': False, 'roboclaw': False}
```

If any package is present later, do not use it directly in import-time code. Hide it behind optional runtime adapter code.

**Step 3: Confirm ABot-Claw workspace files**

Run:

```bash
test -f /ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/SERVICE.md
test -f /ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/MISSION.md
```

Expected: exit 0.

**Step 4: Commit**

No commit required.

---

## Task 1: Runtime Config and Service Registry Parser

**Files:**
- Modify: `src/harness/config.py`
- Create: `src/harness/openclaw/service_registry.py`
- Test: `tests/test_openclaw_service_registry.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_service_registry.py`:

```python
from pathlib import Path

from harness.config import HarnessConfig
from harness.openclaw.service_registry import (
    OpenClawServiceRegistry,
    parse_service_table,
)


def test_harness_config_has_runtime_defaults():
    config = HarnessConfig()

    assert config.harness_runtime == "phase2"
    assert config.openclaw_workspace_path == ""
    assert config.openclaw_planner_backend == "rule"
    assert config.openclaw_service_registry_path == ""


def test_parse_service_table_extracts_spatial_memory():
    text = """
| Service | Purpose | IP / Host | Port | Base URL | Main Endpoint |
|---|---|---|---|---|---|
| SpatialMemory | Robot memory write / query / retrieval | `<SERVICE_HOST>` | `8012` | `http://<SERVICE_HOST>:8012` | `/health`, `/query/*`, `/memory/*` |
"""

    services = parse_service_table(text, service_host="127.0.0.1")

    assert services["SpatialMemory"].name == "SpatialMemory"
    assert services["SpatialMemory"].port == 8012
    assert services["SpatialMemory"].base_url == "http://127.0.0.1:8012"


def test_service_registry_loads_markdown_file(tmp_path):
    doc = tmp_path / "SERVICE.md"
    doc.write_text(
        """
| Service | Purpose | IP / Host | Port | Base URL | Main Endpoint |
|---|---|---|---|---|---|
| SpatialMemory | Memory | `<SERVICE_HOST>` | `8012` | `http://<SERVICE_HOST>:8012` | `/health` |
""",
        encoding="utf-8",
    )

    registry = OpenClawServiceRegistry.from_file(doc, service_host="localhost")

    assert registry.get("SpatialMemory").base_url == "http://localhost:8012"
    assert registry.spatial_memory_url() == "http://localhost:8012"
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_service_registry.py -q
```

Expected: fail because `service_registry.py` and config fields do not exist.

**Step 3: Implement minimal config fields**

In `src/harness/config.py`, add fields to `HarnessConfig`:

```python
harness_runtime: str = "phase2"
openclaw_workspace_path: str = ""
openclaw_service_registry_path: str = ""
openclaw_service_host: str = "127.0.0.1"
openclaw_planner_backend: str = "rule"
openclaw_gateway_url: str = ""
```

Do not change existing defaults.

**Step 4: Implement service registry parser**

Create `src/harness/openclaw/service_registry.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class OpenClawService:
    name: str
    purpose: str
    host: str
    port: int
    base_url: str
    endpoint: str


class OpenClawServiceRegistry:
    def __init__(self, services: Dict[str, OpenClawService]) -> None:
        self.services = services

    @classmethod
    def from_file(
        cls,
        path: Path,
        service_host: str = "127.0.0.1",
    ) -> "OpenClawServiceRegistry":
        return cls(parse_service_table(path.read_text(encoding="utf-8"), service_host))

    def get(self, name: str) -> Optional[OpenClawService]:
        return self.services.get(name)

    def spatial_memory_url(self) -> str:
        service = self.get("SpatialMemory")
        return service.base_url if service is not None else ""


def parse_service_table(text: str, service_host: str = "127.0.0.1") -> Dict[str, OpenClawService]:
    services: Dict[str, OpenClawService] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line or "Service" in line:
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        name, purpose, host, port_text, base_url, endpoint = cells[:6]
        host = host.replace("<SERVICE_HOST>", service_host)
        base_url = base_url.replace("<SERVICE_HOST>", service_host)
        try:
            port = int(port_text)
        except ValueError:
            continue
        services[name] = OpenClawService(
            name=name,
            purpose=purpose,
            host=host,
            port=port,
            base_url=base_url,
            endpoint=endpoint,
        )
    return services
```

**Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_service_registry.py -q
```

Expected: pass.

**Step 6: Commit**

```bash
git add src/harness/config.py src/harness/openclaw/service_registry.py tests/test_openclaw_service_registry.py
git commit -m "Add OpenClaw service registry config"
```

---

## Task 2: OpenClaw Planner Contract

**Files:**
- Create: `src/harness/openclaw/planner.py`
- Test: `tests/test_openclaw_planner.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_planner.py`:

```python
from harness.openclaw.planner import (
    OpenClawPlanDecision,
    RuleOpenClawPlanner,
)
from harness.types import VLNState


def make_state(step_id=0, instruction="go to the kitchen"):
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction=instruction,
        step_id=step_id,
        current_image=None,
    )


def test_rule_planner_starts_with_memory_recall():
    planner = RuleOpenClawPlanner(recall_interval_steps=5)

    decision = planner.plan(make_state(step_id=0), runtime_context={})

    assert isinstance(decision, OpenClawPlanDecision)
    assert decision.intent == "recall_memory"
    assert decision.tool_name == "MemoryQuerySkill"
    assert decision.arguments["text"] == "go to the kitchen"
    assert decision.reason == "initial_recall"


def test_rule_planner_acts_between_recall_intervals():
    planner = RuleOpenClawPlanner(recall_interval_steps=5)

    decision = planner.plan(make_state(step_id=2), runtime_context={})

    assert decision.intent == "act"
    assert decision.tool_name == "NavigationPolicySkill"


def test_rule_planner_never_forwards_oracle_context():
    planner = RuleOpenClawPlanner(recall_interval_steps=5)

    decision = planner.plan(
        make_state(step_id=0),
        runtime_context={"distance_to_goal": 1.0, "success": True},
    )

    assert "distance_to_goal" not in decision.arguments
    assert "success" not in decision.arguments
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_planner.py -q
```

Expected: fail because planner module does not exist.

**Step 3: Implement planner**

Create `src/harness/openclaw/planner.py`:

```python
from dataclasses import dataclass, field
from typing import Any, Dict

from harness.types import VLNState


@dataclass
class OpenClawPlanDecision:
    intent: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    planner_backend: str = "rule"


class RuleOpenClawPlanner:
    def __init__(self, recall_interval_steps: int = 5) -> None:
        self.recall_interval_steps = max(1, recall_interval_steps)

    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> OpenClawPlanDecision:
        if state.step_id == 0:
            return OpenClawPlanDecision(
                intent="recall_memory",
                tool_name="MemoryQuerySkill",
                arguments={
                    "text": state.instruction,
                    "step_id": state.step_id,
                    "reason": "initial_recall",
                },
                reason="initial_recall",
            )
        if state.step_id % self.recall_interval_steps == 0:
            return OpenClawPlanDecision(
                intent="recall_memory",
                tool_name="MemoryQuerySkill",
                arguments={
                    "text": state.instruction,
                    "step_id": state.step_id,
                    "reason": "interval_recall",
                },
                reason="interval_recall",
            )
        return OpenClawPlanDecision(
            intent="act",
            tool_name="NavigationPolicySkill",
            arguments={},
            reason="default_act",
        )
```

Keep this planner deliberately simple. It is the local OpenClaw planner stand-in until a real OpenClaw gateway is available.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_planner.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw/planner.py tests/test_openclaw_planner.py
git commit -m "Add OpenClaw VLN planner contract"
```

---

## Task 3: OpenClaw Habitat Executor Bridge

**Files:**
- Create: `src/harness/openclaw/executor.py`
- Test: `tests/test_openclaw_executor.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_executor.py`:

```python
from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.openclaw.executor import HabitatOpenClawExecutor


def test_executor_converts_action_text_to_habitat_command():
    executor = HabitatOpenClawExecutor(HabitatVLNAdapter())

    command = executor.command_for_action("TURN_LEFT")

    assert command["executor"] == "habitat_discrete"
    assert command["action_text"] == "TURN_LEFT"
    assert command["action_index"] == 2
    assert command["runtime_executor"] == "openclaw_habitat"


def test_executor_rejects_unsupported_action():
    executor = HabitatOpenClawExecutor(HabitatVLNAdapter())

    try:
        executor.command_for_action("JUMP")
    except ValueError as exc:
        assert "Unsupported Habitat action" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_executor.py -q
```

Expected: fail because executor module does not exist.

**Step 3: Implement executor**

Create `src/harness/openclaw/executor.py`:

```python
from typing import Any, Dict

from harness.env_adapters.base import BaseEmbodimentAdapter


class HabitatOpenClawExecutor:
    def __init__(self, adapter: BaseEmbodimentAdapter) -> None:
        self.adapter = adapter

    def command_for_action(self, action_text: str) -> Dict[str, Any]:
        command = self.adapter.action_to_executor_command(action_text)
        command["runtime_executor"] = "openclaw_habitat"
        return command
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_executor.py tests/test_embodiment_adapter_contract.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw/executor.py tests/test_openclaw_executor.py
git commit -m "Add OpenClaw Habitat executor bridge"
```

---

## Task 4: OpenClaw VLN Runtime Bridge

**Files:**
- Create: `src/harness/openclaw/runtime.py`
- Test: `tests/test_openclaw_runtime_bridge.py`

**Step 1: Write failing tests**

Create `tests/test_openclaw_runtime_bridge.py`:

```python
from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.planner import RuleOpenClawPlanner
from harness.openclaw.runtime import OpenClawVLNRuntime
from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.types import SkillResult, VLNState


class EchoNavigationSkill(Skill):
    name = "NavigationPolicySkill"
    description = "Returns a fixed action."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("action", {"action_text": "TURN_LEFT"})


class EchoMemorySkill(Skill):
    name = "MemoryQuerySkill"
    description = "Returns fake memory."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("memory", {"policy_context": {"memory_context_text": "kitchen"}})


def make_state(step_id=1):
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=step_id,
        current_image=None,
    )


def make_runtime():
    registry = SkillRegistry()
    registry.register(EchoNavigationSkill())
    registry.register(EchoMemorySkill())
    return OpenClawVLNRuntime(
        tool_registry=registry,
        planner=RuleOpenClawPlanner(recall_interval_steps=5),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
    )


def test_runtime_lists_tools():
    runtime = make_runtime()

    tools = runtime.list_tools()

    assert [tool["name"] for tool in tools] == ["MemoryQuerySkill", "NavigationPolicySkill"]


def test_runtime_step_calls_planned_tool_and_executor():
    runtime = make_runtime()

    result = runtime.step(make_state(step_id=1), payload={})

    assert result.ok is True
    assert result.action_text == "TURN_LEFT"
    assert result.executor_command["action_index"] == 2
    assert result.runtime_metadata["planner_backend"] == "rule"
    assert result.runtime_metadata["runtime_mode"] == "openclaw_bridge"


def test_runtime_initial_step_can_recall_then_act():
    runtime = make_runtime()

    result = runtime.step(make_state(step_id=0), payload={})

    assert result.ok is True
    assert result.runtime_metadata["planned_intent"] == "recall_memory"
    assert "MemoryQuerySkill" in result.runtime_metadata["tool_calls"][0]["tool_name"]
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py -q
```

Expected: fail because runtime module does not exist.

**Step 3: Implement runtime bridge**

Create `src/harness/openclaw/runtime.py`:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List

from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.planner import RuleOpenClawPlanner
from harness.openclaw.tool_adapter import OpenClawToolAdapter
from harness.skill_registry import SkillRegistry
from harness.types import VLNState


@dataclass
class OpenClawRuntimeStepResult:
    ok: bool
    action_text: str
    executor_command: Dict[str, Any] = field(default_factory=dict)
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class OpenClawVLNRuntime:
    def __init__(
        self,
        tool_registry: SkillRegistry,
        planner: RuleOpenClawPlanner,
        executor: HabitatOpenClawExecutor,
    ) -> None:
        self.tool_adapter = OpenClawToolAdapter(tool_registry)
        self.planner = planner
        self.executor = executor

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.tool_adapter.list_tools()

    def step(
        self,
        state: VLNState,
        payload: Dict[str, Any],
    ) -> OpenClawRuntimeStepResult:
        decision = self.planner.plan(state, runtime_context=payload)
        tool_calls: List[Dict[str, Any]] = []

        if decision.intent == "recall_memory":
            recall = self.tool_adapter.call_tool(
                decision.tool_name,
                decision.arguments,
                state=state,
            )
            tool_calls.append(recall)

        nav_payload = dict(payload)
        nav_result = self.tool_adapter.call_tool("NavigationPolicySkill", nav_payload, state=state)
        tool_calls.append(nav_result)

        if not nav_result.get("ok"):
            return OpenClawRuntimeStepResult(
                ok=False,
                action_text="STOP",
                runtime_metadata=self._metadata(decision, tool_calls),
                error=nav_result.get("error") or "navigation_failed",
            )

        action_text = str(nav_result.get("payload", {}).get("action_text") or "STOP")
        return OpenClawRuntimeStepResult(
            ok=True,
            action_text=action_text,
            executor_command=self.executor.command_for_action(action_text),
            runtime_metadata=self._metadata(decision, tool_calls),
        )

    def _metadata(self, decision, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "runtime_mode": "openclaw_bridge",
            "planner_backend": decision.planner_backend,
            "planned_intent": decision.intent,
            "planned_tool": decision.tool_name,
            "planner_reason": decision.reason,
            "tool_calls": tool_calls,
        }
```

This runtime intentionally recalls then acts on step 0. Later tasks can feed recall context into navigation.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_runtime_bridge.py tests/test_openclaw_tool_adapter.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw/runtime.py tests/test_openclaw_runtime_bridge.py
git commit -m "Add OpenClaw VLN runtime bridge"
```

---

## Task 5: Wire Runtime Bridge Into Evaluation Harness

**Files:**
- Modify: `src/evaluation_harness.py`
- Modify: `tests/test_evaluation_harness_imports.py`
- Test: `tests/test_evaluation_harness_imports.py`

**Step 1: Write failing tests**

Add to `tests/test_evaluation_harness_imports.py`:

```python
def test_parser_includes_openclaw_runtime_args():
    module = importlib.import_module("evaluation_harness")
    args = module.build_parser().parse_args(
        [
            "--model_path",
            "model",
            "--output_path",
            "out",
            "--harness_runtime",
            "openclaw_bridge",
            "--openclaw_workspace_path",
            "/tmp/openclaw",
            "--openclaw_service_host",
            "127.0.0.1",
            "--openclaw_planner_backend",
            "rule",
        ]
    )

    assert args.harness_runtime == "openclaw_bridge"
    assert args.openclaw_workspace_path == "/tmp/openclaw"
    assert args.openclaw_service_host == "127.0.0.1"
    assert args.openclaw_planner_backend == "rule"


def test_openclaw_runtime_import_has_no_external_dependency():
    import harness.openclaw.runtime  # noqa: F401
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_imports.py -q
```

Expected: fail because parser args are missing.

**Step 3: Add parser args**

In `build_parser()` add:

```python
parser.add_argument("--harness_runtime", type=str, default="phase2")
parser.add_argument("--openclaw_workspace_path", type=str, default="")
parser.add_argument("--openclaw_service_registry_path", type=str, default="")
parser.add_argument("--openclaw_service_host", type=str, default="127.0.0.1")
parser.add_argument("--openclaw_planner_backend", type=str, default="rule")
parser.add_argument("--openclaw_gateway_url", type=str, default="")
```

In `build_harness_config()`, populate the matching `HarnessConfig` fields.

**Step 4: Build runtime only when requested**

In `build_harness_components()`, after `adapter` is created:

```python
runtime = None
if config.harness_runtime == "openclaw_bridge":
    from harness.openclaw.executor import HabitatOpenClawExecutor
    from harness.openclaw.planner import RuleOpenClawPlanner
    from harness.openclaw.runtime import OpenClawVLNRuntime

    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=RuleOpenClawPlanner(recall_interval_steps=config.recall_interval_steps),
        executor=HabitatOpenClawExecutor(adapter),
    )
```

Return `"openclaw_runtime": runtime` in components.

Do not instantiate this runtime at import time.

**Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_imports.py tests/test_openclaw_runtime_bridge.py -q
```

Expected: pass.

**Step 6: Commit**

```bash
git add src/evaluation_harness.py tests/test_evaluation_harness_imports.py
git commit -m "Wire optional OpenClaw runtime into harness evaluation"
```

---

## Task 6: Use Runtime Bridge in HarnessModelProxy

**Files:**
- Modify: `src/evaluation_harness.py`
- Test: `tests/test_evaluation_harness_openclaw_runtime.py`

**Step 1: Write failing tests**

Create `tests/test_evaluation_harness_openclaw_runtime.py`:

```python
from types import SimpleNamespace

from evaluation_harness import build_harness_components


class FakeModel:
    model = object()


def make_args(**overrides):
    data = {
        "harness_mode": "memory_recall",
        "harness_memory_backend": "fake",
        "spatial_memory_url": "http://127.0.0.1:8022",
        "harness_memory_source": "episode-local",
        "harness_max_internal_calls": 3,
        "harness_recall_interval_steps": 5,
        "harness_trace_rank": 0,
        "output_path": "/tmp/clawnav-test",
        "num_history": 8,
        "expose_sim_pose_online": False,
        "harness_runtime": "openclaw_bridge",
        "openclaw_workspace_path": "",
        "openclaw_service_registry_path": "",
        "openclaw_service_host": "127.0.0.1",
        "openclaw_planner_backend": "rule",
        "openclaw_gateway_url": "",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_build_components_creates_openclaw_runtime_when_requested():
    components = build_harness_components(make_args(), model=FakeModel())

    assert components["openclaw_runtime"] is not None
    assert components["config"].harness_runtime == "openclaw_bridge"


def test_build_components_leaves_runtime_off_by_default():
    components = build_harness_components(
        make_args(harness_runtime="phase2"),
        model=FakeModel(),
    )

    assert components["openclaw_runtime"] is None
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py -q
```

Expected: fail until components expose `openclaw_runtime`.

**Step 3: Route proxy calls through runtime**

In `HarnessModelProxy.call_model()`, before the existing controller path:

```python
runtime = self.components.get("openclaw_runtime")
if runtime is not None:
    runtime_result = runtime.step(
        state,
        {
            "recent_frames": list(images[:-1]),
            "policy_action": self.last_action_text,
        },
    )
    action_text = runtime_result.action_text if runtime_result.ok else "STOP"
    self.last_action_text = action_text
    self._append_working_memory(images, action_text)
    self._log_runtime_step(state, runtime_result, action_text)
    return [action_text]
```

Add `_log_runtime_step()` that passes runtime metadata to `HarnessLogger.log_step()` without removing Phase 2 fields.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py tests/test_evaluation_harness_imports.py tests/test_harness_logger.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/evaluation_harness.py tests/test_evaluation_harness_openclaw_runtime.py
git commit -m "Run harness proxy through OpenClaw runtime bridge"
```

---

## Task 7: OpenClaw Runtime Trace Metadata

**Files:**
- Modify: `src/harness/logging/harness_logger.py`
- Test: `tests/test_harness_logger.py`

**Step 1: Write failing test**

Add to `tests/test_harness_logger.py`:

```python
def test_logger_writes_openclaw_runtime_metadata(tmp_path):
    logger = HarnessLogger(tmp_path, rank=0)
    state = VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=1,
        current_image=None,
    )

    record = logger.log_step(
        state,
        intent="act",
        skill="NavigationPolicySkill",
        runtime={
            "runtime_mode": "openclaw_bridge",
            "planner_backend": "rule",
            "planned_intent": "act",
            "planned_tool": "NavigationPolicySkill",
            "runtime_executor": "openclaw_habitat",
        },
    )

    assert record["trace_schema_version"] == "phase2.harness_trace.v2"
    assert record["runtime_mode"] == "openclaw_bridge"
    assert record["planner_backend"] == "rule"
    assert record["planned_tool"] == "NavigationPolicySkill"
    assert record["runtime_executor"] == "openclaw_habitat"
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_harness_logger.py -q
```

Expected: fail if logger does not preserve new runtime metadata.

**Step 3: Extend runtime defaults**

In `HarnessLogger.log_step()`, add defaults:

```python
"runtime_mode": "",
"planner_backend": "",
"planned_intent": "",
"planned_tool": "",
"planner_reason": "",
"runtime_executor": "",
"tool_calls": [],
```

Do not change `trace_schema_version`; keep `phase2.harness_trace.v2` unless a separate trace migration is justified later.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_harness_logger.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/logging/harness_logger.py tests/test_harness_logger.py
git commit -m "Add OpenClaw runtime metadata to harness traces"
```

---

## Task 8: ABot-Claw Service Registry Integration

**Files:**
- Modify: `src/evaluation_harness.py`
- Modify: `src/harness/config.py`
- Test: `tests/test_evaluation_harness_openclaw_runtime.py`

**Step 1: Write failing test**

Add to `tests/test_evaluation_harness_openclaw_runtime.py`:

```python
def test_service_registry_can_supply_spatial_memory_url(tmp_path):
    service_doc = tmp_path / "SERVICE.md"
    service_doc.write_text(
        """
| Service | Purpose | IP / Host | Port | Base URL | Main Endpoint |
|---|---|---|---|---|---|
| SpatialMemory | Memory | `<SERVICE_HOST>` | `8012` | `http://<SERVICE_HOST>:8012` | `/health` |
""",
        encoding="utf-8",
    )

    components = build_harness_components(
        make_args(
            harness_memory_backend="spatial_http",
            openclaw_service_registry_path=str(service_doc),
            openclaw_service_host="localhost",
        ),
        model=FakeModel(),
    )

    assert components["config"].spatial_memory_url == "http://localhost:8012"
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py -q
```

Expected: fail until service registry path is used.

**Step 3: Implement service override**

In `build_harness_config(args)`, after constructing config:

```python
if config.openclaw_service_registry_path and config.memory_backend == "spatial_http":
    from harness.openclaw.service_registry import OpenClawServiceRegistry

    registry = OpenClawServiceRegistry.from_file(
        Path(config.openclaw_service_registry_path),
        service_host=config.openclaw_service_host,
    )
    spatial_url = registry.spatial_memory_url()
    if spatial_url:
        config.spatial_memory_url = spatial_url
```

Do not change the existing default `spatial_memory_url` unless this explicit registry path is provided.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_openclaw_runtime.py tests/test_openclaw_service_registry.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/evaluation_harness.py src/harness/config.py tests/test_evaluation_harness_openclaw_runtime.py
git commit -m "Resolve SpatialMemory URL from OpenClaw service registry"
```

---

## Task 9: Phase 3 Documentation and Script

**Files:**
- Create: `docs/protocols/openclaw-vln-runtime.md`
- Create: `scripts/evaluation_openclaw_vln_runtime.sh`
- Modify: `tests/test_phase2_docs.py` or create `tests/test_phase3_docs.py`
- Modify: `tests/test_evaluation_scripts.py`

**Step 1: Write docs test**

Create `tests/test_phase3_docs.py`:

```python
from pathlib import Path


def test_openclaw_vln_runtime_doc_exists():
    text = Path("docs/protocols/openclaw-vln-runtime.md").read_text(encoding="utf-8")

    assert "OpenClaw VLN Runtime" in text
    assert "openclaw_bridge" in text
    assert "No Required OpenClaw Python Dependency" in text
    assert "No Oracle Decision Inputs" in text
    assert "SpatialMemory" in text


def test_openclaw_runtime_script_exists_and_preserves_lowmem_settings():
    text = Path("scripts/evaluation_openclaw_vln_runtime.sh").read_text(encoding="utf-8")

    assert "src/evaluation_harness.py" in text
    assert "--harness_runtime openclaw_bridge" in text
    assert "--max_pixels 401408" in text
    assert "--kv_start_size 8" in text
    assert "--kv_recent_size 24" in text
    assert "--num_history 8" in text
    assert "--use_llm_adaptive_sparse_attention" not in text
    assert "--enable_slow_fast" not in text
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase3_docs.py -q
```

Expected: fail because docs/script do not exist.

**Step 3: Write runtime doc**

Create `docs/protocols/openclaw-vln-runtime.md` with sections:

```text
# OpenClaw VLN Runtime

## Purpose
Explain that Phase 3 runs ClawNav through an OpenClaw-style planner/tool/executor runtime.

## No Required OpenClaw Python Dependency
Explain that external OpenClaw is optional and currently represented by a local runtime bridge.

## Runtime Modes
phase2
openclaw_bridge

## Tool Flow
Planner -> OpenClawToolAdapter -> SkillRegistry -> HabitatOpenClawExecutor

## SpatialMemory
Explain `SERVICE.md` parsing and `spatial_http` behavior.

## No Oracle Decision Inputs
List forbidden online fields.
```

**Step 4: Create script**

Create `scripts/evaluation_openclaw_vln_runtime.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra}
OUTPUT_PATH=${OUTPUT_PATH:-results/clawnav_openclaw_vln_runtime}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
MASTER_PORT=${MASTER_PORT:-20371}
OPENCLAW_SERVICE_REGISTRY=${OPENCLAW_SERVICE_REGISTRY:-/ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/SERVICE.md}
OPENCLAW_SERVICE_HOST=${OPENCLAW_SERVICE_HOST:-127.0.0.1}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
/ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=${MASTER_PORT} \
  src/evaluation_harness.py \
  --model_path "${MODEL_PATH}" \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --output_path "${OUTPUT_PATH}" \
  --harness_runtime openclaw_bridge \
  --harness_mode memory_recall \
  --harness_memory_backend fake \
  --harness_memory_source episode-local \
  --openclaw_service_registry_path "${OPENCLAW_SERVICE_REGISTRY}" \
  --openclaw_service_host "${OPENCLAW_SERVICE_HOST}"
```

Run:

```bash
chmod +x scripts/evaluation_openclaw_vln_runtime.sh
```

**Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase3_docs.py tests/test_evaluation_scripts.py -q
```

Expected: pass.

**Step 6: Commit**

```bash
git add docs/protocols/openclaw-vln-runtime.md scripts/evaluation_openclaw_vln_runtime.sh tests/test_phase3_docs.py tests/test_evaluation_scripts.py
git commit -m "Document OpenClaw VLN runtime mode"
```

---

## Task 10: Phase 3 Test Sweep

**Files:**
- No code files unless fixes are required.

**Step 1: Run Phase 3 tests**

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

**Step 2: Run Phase 2 tests**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_skill_manifest_schema.py \
  tests/test_openclaw_tool_adapter.py \
  tests/test_spatial_memory_protocol.py \
  tests/test_embodiment_adapter_contract.py \
  tests/test_phase2_docs.py \
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

**Step 4: Run lightweight regression**

Run:

```bash
PYTHONPATH=.:src pytest \
  tests/test_llm_visual_pruner.py \
  tests/test_adaptive_sparse_attention.py \
  tests/test_evaluation_scripts.py \
  -q
```

Expected: pass or skip only optional sparse-kernel cases.

**Step 5: Compile changed modules**

Run:

```bash
python -m py_compile \
  src/harness/config.py \
  src/harness/openclaw/service_registry.py \
  src/harness/openclaw/planner.py \
  src/harness/openclaw/executor.py \
  src/harness/openclaw/runtime.py \
  src/harness/logging/harness_logger.py \
  src/evaluation_harness.py
```

Expected: exit 0.

**Step 6: Commit fixes**

If changes were needed:

```bash
git add <fixed-files>
git commit -m "Stabilize Phase 3 OpenClaw runtime tests"
```

---

## Task 11: Phase 3 Smoke Verification

**Files:**
- No code files unless bugs are found.

**Step 1: Run OpenClaw bridge act-only smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=4 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20381 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_phase3_smoke_openclaw_act_only \
  --harness_runtime openclaw_bridge \
  --harness_mode act_only \
  --harness_memory_backend fake \
  --harness_memory_source episode-local
```

Expected:

- evaluation exits 0
- `summary.json` exists
- `harness_traces/harness_trace_rank0.jsonl` exists
- trace contains `runtime_mode=openclaw_bridge`
- trace contains `runtime_executor=openclaw_habitat`
- trace contains `oracle_metrics_used_for_decision=false`
- no sparse attention install log
- no slow-fast path

**Step 2: Run OpenClaw bridge memory-recall smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=4 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20382 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_phase3_smoke_openclaw_memory_recall \
  --harness_runtime openclaw_bridge \
  --harness_mode memory_recall \
  --harness_memory_backend fake \
  --harness_memory_source episode-local \
  --openclaw_service_registry_path /ssd/dingmuhe/Embodied-task/Navigation_Claw/ABot-Claw_Muhe/openclaw_layer/SERVICE.md \
  --openclaw_service_host 127.0.0.1
```

Expected:

- evaluation exits 0
- trace includes `planned_intent=recall_memory` at step 0
- trace includes at least one OpenClaw tool call
- trace includes runtime latency metadata
- `oracle_metrics_used_for_decision=false`
- no crash

**Step 3: Inspect outputs**

Run:

```bash
cat results/clawnav_phase3_smoke_openclaw_act_only/summary.json
head -1 results/clawnav_phase3_smoke_openclaw_act_only/harness_traces/harness_trace_rank0.jsonl
cat results/clawnav_phase3_smoke_openclaw_memory_recall/summary.json
head -1 results/clawnav_phase3_smoke_openclaw_memory_recall/harness_traces/harness_trace_rank0.jsonl
```

Expected: all required fields are present.

**Step 4: Commit only code fixes**

Do not commit `results/` outputs.

---

## Recommended Implementation Order

1. Task 0: Baseline and dependency boundary
2. Task 1: Runtime config and service registry
3. Task 2: Planner contract
4. Task 3: Executor bridge
5. Task 4: Runtime bridge
6. Task 5: Evaluation parser/config wiring
7. Task 6: Proxy runtime execution path
8. Task 7: Runtime trace metadata
9. Task 8: ABot-Claw service registry integration
10. Task 9: Docs and script
11. Task 10: Full test sweep
12. Task 11: Smoke verification

## Notes for Execution

- Work from `ClawNav`, not the outer `Navigation_Claw` repository.
- Prefer one commit per task.
- Keep `--harness_runtime phase2` as the default path.
- Treat `--harness_runtime openclaw_bridge` as the Phase 3 opt-in path.
- Do not make external OpenClaw importable packages mandatory.
- If a real OpenClaw gateway becomes available, add it behind `openclaw_planner_backend="gateway"` in a later follow-up plan.
- Keep all online decisions non-oracle. If a field is useful only for diagnostics, write it to trace but never forward it to planner/tool arguments.
