# ClawNav Phase 2 OpenClaw-Compatible Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the Phase 1 Python Harness into an OpenClaw-compatible interface layer without making the runtime depend on OpenClaw.

**Architecture:** Keep the existing Phase 1 harness path intact and add compatibility surfaces around it: skill manifests/tool schemas, an OpenClaw-style tool adapter, a documented SpatialMemory protocol, an embodiment adapter contract, and runtime-compatible trace records. Phase 2 is an interface-standardization phase, not a model-architecture phase.

**Tech Stack:** Python, dataclasses, pytest, requests, JSON Schema-style dict contracts, existing ClawNav harness modules.

---

## Global Constraints

- Keep `src/evaluation_harness.py` import-safe and runnable without OpenClaw installed.
- Do not change JanusVLN model internals.
- Do not enable LLM adaptive sparse attention.
- Do not add slow-fast active memory reuse.
- Do not replace the current Python `SkillRegistry`; add compatibility APIs around it.
- Do not make `openclaw` or ABot-Claw packages required dependencies.
- Online controller, critic, memory manager, and tool adapter must not use oracle metrics for decisions:
  - `distance_to_goal`
  - `success`
  - `SPL`
  - oracle shortest path
  - oracle shortest path action
  - future observations
- Oracle metrics may remain diagnostics-only.
- Existing Phase 1 tests must keep passing after every task.
- All new schemas must use repo-local plain Python dicts/dataclasses first; avoid introducing a schema dependency unless tests prove it is needed.

## Phase 2 Completion Criteria

- Every harness skill exposes a stable manifest with input/output schema metadata.
- `SkillRegistry` can list manifests without changing existing `run()` behavior.
- A new OpenClaw-compatible adapter can list tools and call tools through the existing registry.
- SpatialMemory HTTP protocol is documented and tested against fake and HTTP clients.
- Habitat adapter implements a generic embodiment adapter contract.
- Harness trace records include runtime-compatible call metadata while preserving Phase 1 fields.
- `PYTHONPATH=src pytest` passes for new Phase 2 tests and existing harness tests.
- Existing low-memory harness script remains no-sparse and no-slow-fast.

---

## Task 0: Baseline Characterization

**Files:**
- Read: `src/harness/skills/base.py`
- Read: `src/harness/skill_registry.py`
- Read: `src/harness/memory/spatial_memory_client.py`
- Read: `src/harness/env_adapters/habitat_vln_adapter.py`
- Read: `src/harness/logging/harness_logger.py`
- Test: no new files

**Step 1: Record current baseline**

Run:

```bash
PYTHONPATH=src pytest \
  tests/test_harness_types.py \
  tests/test_skill_registry.py \
  tests/test_memory_manager.py \
  tests/test_spatial_memory_client.py \
  tests/test_habitat_vln_adapter.py \
  tests/test_harness_logger.py \
  tests/test_evaluation_harness_imports.py \
  -q
```

Expected: pass.

**Step 2: Run lightweight regression with root path**

Run:

```bash
PYTHONPATH=.:src pytest \
  tests/test_llm_visual_pruner.py \
  tests/test_adaptive_sparse_attention.py \
  tests/test_evaluation_scripts.py \
  -q
```

Expected: pass or skip only for optional sparse kernels. If this fails because `evaluation_debug_utils` cannot import, confirm `PYTHONPATH=.:src` is being used.

**Step 3: Inspect git scope**

Run from the real repo:

```bash
git status --short
```

Expected: status is from `ClawNav`, not the outer `Navigation_Claw` repository.

**Step 4: Commit**

No commit required unless a baseline note file is added. Prefer no code changes in Task 0.

---

## Task 1: Skill Manifest Contract

**Files:**
- Modify: `src/harness/skills/base.py`
- Modify: `src/harness/types.py`
- Test: `tests/test_skill_manifest_schema.py`

**Step 1: Write the failing tests**

Create `tests/test_skill_manifest_schema.py`:

```python
from harness.skills.base import Skill, SkillManifest
from harness.types import SkillResult


class EchoSkill(Skill):
    name = "EchoSkill"
    description = "Echoes a text payload."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    output_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    timeout_ms = 1000
    side_effects = False
    oracle_safe = True
    callable_from_runtime = True

    def run(self, state, payload):
        return SkillResult.ok_result("echo", {"text": payload["text"]})


def test_skill_manifest_is_structured():
    manifest = EchoSkill().manifest()

    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "EchoSkill"
    assert manifest.description == "Echoes a text payload."
    assert manifest.input_schema["required"] == ["text"]
    assert manifest.output_schema["required"] == ["text"]
    assert manifest.timeout_ms == 1000
    assert manifest.side_effects is False
    assert manifest.oracle_safe is True
    assert manifest.callable_from_runtime is True


def test_skill_manifest_exports_plain_dict():
    data = EchoSkill().manifest().to_dict()

    assert data["name"] == "EchoSkill"
    assert data["schema_version"] == "phase2.skill_manifest.v1"
    assert data["input_schema"]["type"] == "object"
    assert data["output_schema"]["type"] == "object"
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_skill_manifest_schema.py -q
```

Expected: fail because `SkillManifest` and `Skill.manifest()` do not exist.

**Step 3: Implement minimal manifest contract**

In `src/harness/skills/base.py`, add:

```python
from dataclasses import dataclass, field
from typing import Any, Dict

from harness.types import SkillResult


@dataclass
class SkillManifest:
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 5000
    side_effects: bool = False
    oracle_safe: bool = True
    callable_from_runtime: bool = True
    schema_version: str = "phase2.skill_manifest.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "timeout_ms": self.timeout_ms,
            "side_effects": self.side_effects,
            "oracle_safe": self.oracle_safe,
            "callable_from_runtime": self.callable_from_runtime,
        }
```

Update `Skill`:

```python
class Skill:
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    timeout_ms: int = 5000
    side_effects: bool = False
    oracle_safe: bool = True
    callable_from_runtime: bool = True

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name=self.name or self.__class__.__name__,
            description=self.description,
            input_schema=dict(self.input_schema),
            output_schema=dict(self.output_schema),
            timeout_ms=self.timeout_ms,
            side_effects=self.side_effects,
            oracle_safe=self.oracle_safe,
            callable_from_runtime=self.callable_from_runtime,
        )
```

**Step 4: Run test**

Run:

```bash
PYTHONPATH=src pytest tests/test_skill_manifest_schema.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/skills/base.py tests/test_skill_manifest_schema.py
git commit -m "Add harness skill manifest contract"
```

---

## Task 2: Registry Manifest Listing

**Files:**
- Modify: `src/harness/skill_registry.py`
- Modify: `tests/test_skill_registry.py`
- Test: `tests/test_skill_registry.py`

**Step 1: Write the failing tests**

Add tests to `tests/test_skill_registry.py`:

```python
def test_registry_lists_skill_manifests():
    registry = SkillRegistry()
    registry.register(EchoSkill())

    manifests = registry.list_manifests()

    assert len(manifests) == 1
    assert manifests[0].name == "EchoSkill"


def test_registry_exports_manifest_dicts_sorted_by_name():
    registry = SkillRegistry()
    registry.register(EchoSkill(name="ZSkill"))
    registry.register(EchoSkill(name="ASkill"))

    data = registry.export_tool_schemas()

    assert [item["name"] for item in data] == ["ASkill", "ZSkill"]
    assert data[0]["schema_version"] == "phase2.skill_manifest.v1"
```

If the existing test helper `EchoSkill` does not accept a custom name, add a small local test skill class that does.

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_skill_registry.py -q
```

Expected: fail because `list_manifests()` and `export_tool_schemas()` do not exist.

**Step 3: Implement registry exports**

In `src/harness/skill_registry.py`, add:

```python
from harness.skills.base import Skill, SkillManifest
```

Add methods:

```python
def list_manifests(self) -> List[SkillManifest]:
    return [self._skills[name].manifest() for name in self.names()]


def export_tool_schemas(self) -> List[Dict[str, Any]]:
    return [manifest.to_dict() for manifest in self.list_manifests()]
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_skill_registry.py tests/test_skill_manifest_schema.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/skill_registry.py tests/test_skill_registry.py
git commit -m "Expose skill manifests from registry"
```

---

## Task 3: Add Manifests to Built-In Skills

**Files:**
- Modify: `src/harness/skills/navigation_policy.py`
- Modify: `src/harness/skills/memory_query.py`
- Modify: `src/harness/skills/memory_write.py`
- Modify: `src/harness/skills/progress_critic.py`
- Modify: `src/harness/skills/replanner.py`
- Test: `tests/test_skill_manifest_schema.py`

**Step 1: Write the failing tests**

Add to `tests/test_skill_manifest_schema.py`:

```python
from harness.memory.memory_manager import MemoryManager
from harness.memory.spatial_memory_client import FakeSpatialMemoryClient
from harness.skills.memory_query import MemoryQuerySkill
from harness.skills.memory_write import MemoryWriteSkill
from harness.skills.navigation_policy import NavigationPolicySkill
from harness.skills.progress_critic import ProgressCriticSkill
from harness.skills.replanner import ReplannerSkill


def test_builtin_skills_have_runtime_manifests():
    memory_manager = MemoryManager(FakeSpatialMemoryClient())
    skills = [
        NavigationPolicySkill(model=None),
        MemoryQuerySkill(memory_manager),
        MemoryWriteSkill(FakeSpatialMemoryClient()),
        ProgressCriticSkill(),
        ReplannerSkill(),
    ]

    for skill in skills:
        manifest = skill.manifest()
        assert manifest.name
        assert manifest.description
        assert manifest.input_schema["type"] == "object"
        assert manifest.output_schema["type"] == "object"
        assert manifest.oracle_safe is True


def test_memory_write_manifest_declares_side_effects():
    manifest = MemoryWriteSkill(FakeSpatialMemoryClient()).manifest()

    assert manifest.side_effects is True
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_skill_manifest_schema.py -q
```

Expected: fail because built-in skills lack descriptions and schemas.

**Step 3: Add minimal schemas**

For each built-in skill, add class attributes only. Do not change `run()` behavior.

Example for `MemoryQuerySkill`:

```python
description = "Query spatial memory for instruction, subgoal, or failure-recovery context."
input_schema = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "reason": {"type": "string"},
        "n_results": {"type": "integer"},
    },
    "required": ["text"],
}
output_schema = {
    "type": "object",
    "properties": {
        "hits": {"type": "array"},
        "policy_context": {"type": "object"},
        "control_context": {"type": "object"},
        "executor_context": {"type": "object"},
    },
}
oracle_safe = True
```

For `MemoryWriteSkill`, set:

```python
side_effects = True
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_skill_manifest_schema.py tests/test_memory_skills.py tests/test_navigation_policy_skill.py tests/test_progress_critic_replanner.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/skills tests/test_skill_manifest_schema.py
git commit -m "Add manifests for built-in harness skills"
```

---

## Task 4: OpenClaw Tool Adapter Skeleton

**Files:**
- Create: `src/harness/openclaw/__init__.py`
- Create: `src/harness/openclaw/tool_adapter.py`
- Test: `tests/test_openclaw_tool_adapter.py`

**Step 1: Write the failing tests**

Create `tests/test_openclaw_tool_adapter.py`:

```python
from harness.openclaw.tool_adapter import OpenClawToolAdapter
from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.types import SkillResult, VLNState


class EchoSkill(Skill):
    name = "EchoSkill"
    description = "Echoes text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("echo", {"text": payload["text"]})


def make_state():
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=0,
        current_image=None,
    )


def test_tool_adapter_lists_tools_from_registry():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = OpenClawToolAdapter(registry)

    tools = adapter.list_tools()

    assert tools[0]["name"] == "EchoSkill"
    assert tools[0]["input_schema"]["required"] == ["text"]


def test_tool_adapter_calls_registered_skill():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = OpenClawToolAdapter(registry)

    result = adapter.call_tool("EchoSkill", {"text": "hello"}, state=make_state())

    assert result["ok"] is True
    assert result["result_type"] == "echo"
    assert result["payload"]["text"] == "hello"


def test_tool_adapter_returns_structured_error_for_missing_tool():
    adapter = OpenClawToolAdapter(SkillRegistry())

    result = adapter.call_tool("MissingSkill", {}, state=make_state())

    assert result["ok"] is False
    assert "not registered" in result["error"]
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_tool_adapter.py -q
```

Expected: fail because `harness.openclaw.tool_adapter` does not exist.

**Step 3: Implement adapter**

In `src/harness/openclaw/tool_adapter.py`:

```python
from typing import Any, Dict, List, Optional

from harness.skill_registry import SkillRegistry
from harness.types import SkillResult, VLNState


class OpenClawToolAdapter:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.registry.export_tool_schemas()

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        for tool in self.list_tools():
            if tool["name"] == name:
                return tool
        return None

    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        state: VLNState,
    ) -> Dict[str, Any]:
        result = self.registry.run(name, state, arguments)
        return self._result_to_dict(result)

    def _result_to_dict(self, result: SkillResult) -> Dict[str, Any]:
        return {
            "ok": result.ok,
            "result_type": result.result_type,
            "payload": result.payload,
            "confidence": result.confidence,
            "error": result.error,
        }
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_tool_adapter.py tests/test_skill_registry.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw tests/test_openclaw_tool_adapter.py
git commit -m "Add OpenClaw-compatible tool adapter"
```

---

## Task 5: OpenClaw Tool Call Metadata and Oracle Guard

**Files:**
- Modify: `src/harness/openclaw/tool_adapter.py`
- Test: `tests/test_openclaw_tool_adapter.py`

**Step 1: Write the failing tests**

Add to `tests/test_openclaw_tool_adapter.py`:

```python
def test_tool_adapter_rejects_oracle_decision_inputs():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = OpenClawToolAdapter(registry)

    result = adapter.call_tool(
        "EchoSkill",
        {"text": "hello", "distance_to_goal": 1.0},
        state=make_state(),
    )

    assert result["ok"] is False
    assert result["error_type"] == "oracle_input_rejected"


def test_tool_adapter_adds_call_metadata():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = OpenClawToolAdapter(registry)

    result = adapter.call_tool("EchoSkill", {"text": "hello"}, state=make_state())

    assert result["tool_name"] == "EchoSkill"
    assert result["runtime_status"] == "completed"
    assert isinstance(result["latency_ms"], float)
    assert result["tool_schema_version"] == "phase2.skill_manifest.v1"
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_tool_adapter.py -q
```

Expected: fail because call metadata and oracle guard are absent.

**Step 3: Implement metadata and guard**

Add local oracle key set in `tool_adapter.py`:

```python
ORACLE_DECISION_KEYS = {
    "distance_to_goal",
    "success",
    "SPL",
    "spl",
    "oracle_path",
    "oracle_shortest_path",
    "oracle_shortest_path_action",
    "oracle_action",
}
```

Before calling the registry, reject if any key appears in `arguments`.

Measure latency with `time.perf_counter()`.

Include:

```text
tool_name
tool_schema_version
runtime_status
latency_ms
error_type
```

Use `runtime_status="completed"` for ok results and `"failed"` for error results.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_openclaw_tool_adapter.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/openclaw/tool_adapter.py tests/test_openclaw_tool_adapter.py
git commit -m "Add runtime metadata to OpenClaw tool calls"
```

---

## Task 6: SpatialMemory Protocol Schemas

**Files:**
- Create: `src/harness/memory/protocol.py`
- Modify: `src/harness/memory/spatial_memory_client.py`
- Test: `tests/test_spatial_memory_protocol.py`

**Step 1: Write the failing tests**

Create `tests/test_spatial_memory_protocol.py`:

```python
from harness.memory.protocol import (
    SPATIAL_MEMORY_PROTOCOL_VERSION,
    build_query_payload,
    normalize_memory_result,
)


def test_build_query_payload_includes_protocol_fields():
    payload = build_query_payload(
        query_type="semantic",
        text="kitchen",
        n_results=3,
        memory_source="episode-local",
    )

    assert payload["protocol_version"] == SPATIAL_MEMORY_PROTOCOL_VERSION
    assert payload["query_type"] == "semantic"
    assert payload["text"] == "kitchen"
    assert payload["n_results"] == 3
    assert payload["memory_source"] == "episode-local"


def test_normalize_memory_result_maps_evidence_fields():
    result = normalize_memory_result(
        {
            "id": "m1",
            "memory_type": "place",
            "name": "kitchen",
            "confidence": 0.9,
            "target_pose": {"x": 1.0},
            "evidence": {"text": "kitchen entrance", "image_path": "/tmp/a.jpg"},
        },
        memory_source="scene-prior",
    )

    assert result["memory_id"] == "m1"
    assert result["evidence_text"] == "kitchen entrance"
    assert result["image_path"] == "/tmp/a.jpg"
    assert result["memory_source"] == "scene-prior"


def test_normalize_memory_result_rejects_oracle_metadata():
    result = normalize_memory_result(
        {
            "id": "m1",
            "metadata": {"distance_to_goal": 1.0},
        },
        memory_source="episode-local",
    )

    assert "distance_to_goal" not in result["metadata"]
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_spatial_memory_protocol.py -q
```

Expected: fail because `harness.memory.protocol` does not exist.

**Step 3: Implement protocol helpers**

In `src/harness/memory/protocol.py`, define:

```python
SPATIAL_MEMORY_PROTOCOL_VERSION = "phase2.spatial_memory.v1"
```

Implement:

```text
build_query_payload(query_type, text=None, pose=None, n_results=5, memory_source="episode-local")
normalize_memory_result(item, memory_source)
strip_oracle_fields(metadata)
```

Oracle fields must be removed from metadata.

**Step 4: Wire HTTP client to protocol helper**

In `spatial_memory_client.py`, use `build_query_payload()` for query requests and `normalize_memory_result()` inside `_memory_hit_from_result()`.

Do not change public method names.

**Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_spatial_memory_protocol.py tests/test_spatial_memory_client.py -q
```

Expected: pass.

**Step 6: Commit**

```bash
git add src/harness/memory/protocol.py src/harness/memory/spatial_memory_client.py tests/test_spatial_memory_protocol.py tests/test_spatial_memory_client.py
git commit -m "Define SpatialMemory protocol helpers"
```

---

## Task 7: SpatialMemory Protocol Documentation

**Files:**
- Create: `docs/protocols/spatial-memory-protocol.md`
- Test: `tests/test_spatial_memory_protocol.py`

**Step 1: Write the failing test**

Add to `tests/test_spatial_memory_protocol.py`:

```python
from pathlib import Path


def test_spatial_memory_protocol_doc_exists_and_lists_endpoints():
    doc = Path("docs/protocols/spatial-memory-protocol.md")
    text = doc.read_text(encoding="utf-8")

    for endpoint in [
        "GET /health",
        "POST /query/object",
        "POST /query/place",
        "POST /query/position",
        "POST /query/semantic/text",
        "POST /query/unified",
        "POST /memory/semantic/ingest",
    ]:
        assert endpoint in text

    assert "memory_source" in text
    assert "No Oracle Fields" in text
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_spatial_memory_protocol.py -q
```

Expected: fail because the doc does not exist.

**Step 3: Write protocol doc**

Create `docs/protocols/spatial-memory-protocol.md` with:

```text
# SpatialMemory Protocol

Protocol version: phase2.spatial_memory.v1

## Endpoints
GET /health
POST /query/object
POST /query/place
POST /query/position
POST /query/semantic/text
POST /query/unified
POST /memory/semantic/ingest

## Required Result Fields
memory_id, memory_type, name, confidence, target_pose, evidence_text,
image_path, memory_source, metadata

## No Oracle Fields
The service must not return distance_to_goal, success, SPL, oracle path,
oracle shortest path action, or future observations in query results used by
online controller logic.
```

Add request/response examples after the required sections.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_spatial_memory_protocol.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add docs/protocols/spatial-memory-protocol.md tests/test_spatial_memory_protocol.py
git commit -m "Document SpatialMemory protocol"
```

---

## Task 8: Embodiment Adapter Base Contract

**Files:**
- Create: `src/harness/env_adapters/base.py`
- Modify: `src/harness/env_adapters/habitat_vln_adapter.py`
- Test: `tests/test_embodiment_adapter_contract.py`
- Test: `tests/test_habitat_vln_adapter.py`

**Step 1: Write the failing tests**

Create `tests/test_embodiment_adapter_contract.py`:

```python
from harness.env_adapters.base import BaseEmbodimentAdapter
from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter


def test_habitat_adapter_implements_base_contract():
    adapter = HabitatVLNAdapter()

    assert isinstance(adapter, BaseEmbodimentAdapter)
    assert adapter.action_space() == ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]


def test_action_to_executor_command_is_structured():
    adapter = HabitatVLNAdapter()

    command = adapter.action_to_executor_command("TURN_LEFT")

    assert command["action_text"] == "TURN_LEFT"
    assert command["action_index"] == 2
    assert command["executor"] == "habitat_discrete"
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_embodiment_adapter_contract.py -q
```

Expected: fail because `BaseEmbodimentAdapter` does not exist.

**Step 3: Implement base adapter**

In `src/harness/env_adapters/base.py`:

```python
from typing import Any, Dict, List, Optional

from harness.types import VLNState


class BaseEmbodimentAdapter:
    def build_state(
        self,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        metrics: Optional[Dict[str, Any]],
        step_id: int,
        last_action: Optional[str] = None,
    ) -> VLNState:
        raise NotImplementedError

    def action_space(self) -> List[str]:
        raise NotImplementedError

    def action_to_executor_command(self, action_text: str) -> Dict[str, Any]:
        raise NotImplementedError
```

Make `HabitatVLNAdapter(BaseEmbodimentAdapter)` and add:

```python
ACTIONS2IDX = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}
```

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_embodiment_adapter_contract.py tests/test_habitat_vln_adapter.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/env_adapters/base.py src/harness/env_adapters/habitat_vln_adapter.py tests/test_embodiment_adapter_contract.py tests/test_habitat_vln_adapter.py
git commit -m "Add embodiment adapter contract"
```

---

## Task 9: Runtime Trace Schema v2

**Files:**
- Modify: `src/harness/logging/harness_logger.py`
- Test: `tests/test_harness_logger.py`

**Step 1: Write the failing tests**

Add to `tests/test_harness_logger.py`:

```python
def test_logger_writes_runtime_trace_metadata(tmp_path):
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
        intent="recall_memory",
        skill="MemoryQuerySkill",
        reason="initial_recall",
        runtime={
            "skill_call_id": "call-1",
            "parent_call_id": "root",
            "tool_schema_version": "phase2.skill_manifest.v1",
            "latency_ms": 2.5,
            "runtime_status": "completed",
        },
    )

    assert record["trace_schema_version"] == "phase2.harness_trace.v2"
    assert record["skill_call_id"] == "call-1"
    assert record["parent_call_id"] == "root"
    assert record["tool_schema_version"] == "phase2.skill_manifest.v1"
    assert record["latency_ms"] == 2.5
    assert record["runtime_status"] == "completed"
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_harness_logger.py -q
```

Expected: fail because `runtime` metadata is not accepted.

**Step 3: Implement runtime trace fields**

In `HarnessLogger.log_step()`, add optional parameter:

```python
runtime: Optional[Dict[str, Any]] = None
```

Add default fields:

```python
"trace_schema_version": "phase2.harness_trace.v2",
"skill_call_id": "",
"parent_call_id": "",
"tool_schema_version": "",
"latency_ms": None,
"input_summary": {},
"output_summary": {},
"error_type": "",
"runtime_status": "completed" if not fallback else "fallback",
```

Then merge `runtime` over these defaults.

**Step 4: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_harness_logger.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/harness/logging/harness_logger.py tests/test_harness_logger.py
git commit -m "Add runtime metadata to harness trace"
```

---

## Task 10: Connect Tool Adapter Metadata to Logger

**Files:**
- Modify: `src/evaluation_harness.py`
- Modify: `src/harness/controller.py`
- Test: `tests/test_harness_controller.py`
- Test: `tests/test_evaluation_harness_imports.py`

**Step 1: Inspect current trace path**

Read:

```text
src/harness/controller.py
src/evaluation_harness.py
```

Identify where skill calls are recorded in `controller.last_trace` and where `HarnessModelProxy._log_step()` writes records.

**Step 2: Write failing test**

Add to `tests/test_harness_controller.py`:

```python
def test_controller_trace_records_skill_runtime_metadata():
    # Use the existing controller test setup.
    # After one run_step(), assert last_trace has a calls list and runtime metadata list.
    assert "skill_runtime" in controller.last_trace
    assert controller.last_trace["skill_runtime"][0]["skill"] == "NavigationPolicySkill"
    assert "latency_ms" in controller.last_trace["skill_runtime"][0]
```

Adapt the setup to the existing fixtures in the file.

**Step 3: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_harness_controller.py -q
```

Expected: fail because controller trace does not include runtime metadata.

**Step 4: Implement minimal controller runtime trace**

In `controller.py`, record for each `registry.run()`:

```text
skill
runtime_status
latency_ms
error_type
result_type
```

Do not change scheduling decisions.

In `evaluation_harness.py`, pass the latest runtime item into `logger.log_step(runtime=...)` when available.

**Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_harness_controller.py tests/test_harness_logger.py tests/test_evaluation_harness_imports.py -q
```

Expected: pass.

**Step 6: Commit**

```bash
git add src/harness/controller.py src/evaluation_harness.py tests/test_harness_controller.py
git commit -m "Propagate skill runtime metadata into harness traces"
```

---

## Task 11: Phase 2 Script and Import Guard

**Files:**
- Modify: `src/evaluation_harness.py`
- Modify: `tests/test_evaluation_harness_imports.py`
- Modify: `tests/test_evaluation_scripts.py`

**Step 1: Write tests for no new heavy imports**

Extend `tests/test_evaluation_harness_imports.py`:

```python
def test_openclaw_adapter_import_has_no_runtime_dependency():
    import harness.openclaw.tool_adapter  # noqa: F401
```

Confirm the test does not import Habitat, load model weights, initialize distributed mode, or require OpenClaw.

**Step 2: Run test**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_harness_imports.py -q
```

Expected: pass after prior tasks.

**Step 3: Add script test only if needed**

If Phase 2 adds a new script, it should be optional and must not replace `scripts/evaluation_lowmem_harness.sh`.

Preferred: no new script in Phase 2. Reuse existing script and expose interfaces through Python modules.

**Step 4: Run script tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_evaluation_scripts.py tests/test_evaluation_harness_imports.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/evaluation_harness.py tests/test_evaluation_harness_imports.py tests/test_evaluation_scripts.py
git commit -m "Guard Phase 2 compatibility imports"
```

---

## Task 12: Phase 2 Documentation

**Files:**
- Create: `docs/protocols/openclaw-tool-adapter.md`
- Modify: `docs/plans/2026-05-08-claw-style-harness-for-fast-janusvln.md`
- Test: optional docs tests in `tests/test_phase2_docs.py`

**Step 1: Write docs test**

Create `tests/test_phase2_docs.py`:

```python
from pathlib import Path


def test_openclaw_tool_adapter_doc_exists():
    doc = Path("docs/protocols/openclaw-tool-adapter.md")
    text = doc.read_text(encoding="utf-8")

    assert "OpenClaw Tool Adapter" in text
    assert "list_tools" in text
    assert "call_tool" in text
    assert "No OpenClaw Runtime Dependency" in text
    assert "No Oracle Decision Inputs" in text
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase2_docs.py -q
```

Expected: fail because doc does not exist.

**Step 3: Write adapter doc**

Create `docs/protocols/openclaw-tool-adapter.md` with:

```text
# OpenClaw Tool Adapter

## Purpose
Expose Phase 1 harness skills through an OpenClaw-compatible tool interface
without requiring OpenClaw runtime at import or evaluation time.

## APIs
list_tools()
get_tool_schema(name)
call_tool(name, arguments, state)

## No OpenClaw Runtime Dependency
The adapter depends only on harness SkillRegistry and SkillResult contracts.

## No Oracle Decision Inputs
Tool calls reject online arguments containing distance_to_goal, success, SPL,
oracle path, oracle shortest path action, or future observations.
```

Add one example tool schema and one example call result.

**Step 4: Update design doc**

In `docs/plans/2026-05-08-claw-style-harness-for-fast-janusvln.md`, add a short note under Phase 2 stating that implementation is tracked by this plan:

```text
Implementation plan: docs/plans/2026-05-11-clawnav-phase2-openclaw-compatible-interface.md
```

**Step 5: Run test**

Run:

```bash
PYTHONPATH=src pytest tests/test_phase2_docs.py -q
```

Expected: pass.

**Step 6: Commit**

```bash
git add docs/protocols/openclaw-tool-adapter.md docs/plans/2026-05-08-claw-style-harness-for-fast-janusvln.md tests/test_phase2_docs.py
git commit -m "Document Phase 2 OpenClaw compatibility protocols"
```

---

## Task 13: Full Phase 2 Test Sweep

**Files:**
- No code files unless fixes are required.

**Step 1: Run new Phase 2 tests**

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

**Step 2: Run existing harness tests**

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

**Step 3: Run lightweight regression**

Run:

```bash
PYTHONPATH=.:src pytest \
  tests/test_llm_visual_pruner.py \
  tests/test_adaptive_sparse_attention.py \
  tests/test_evaluation_scripts.py \
  -q
```

Expected: pass or skip only optional sparse-kernel cases.

**Step 4: Compile changed modules**

Run:

```bash
python -m py_compile \
  src/harness/skills/base.py \
  src/harness/skill_registry.py \
  src/harness/openclaw/tool_adapter.py \
  src/harness/memory/protocol.py \
  src/harness/memory/spatial_memory_client.py \
  src/harness/env_adapters/base.py \
  src/harness/env_adapters/habitat_vln_adapter.py \
  src/harness/logging/harness_logger.py \
  src/harness/controller.py \
  src/evaluation_harness.py
```

Expected: exit 0.

**Step 5: Commit fixes**

If changes were needed:

```bash
git add <fixed-files>
git commit -m "Stabilize Phase 2 compatibility test sweep"
```

---

## Task 14: Optional Smoke Verification

**Files:**
- No code files unless bugs are found.

**Step 1: Run act-only smoke only if GPU/model/data are available**

Run:

```bash
CUDA_VISIBLE_DEVICES=4 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20341 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_phase2_smoke_act_only \
  --harness_mode act_only \
  --harness_memory_backend fake \
  --harness_memory_source episode-local
```

Expected:

- evaluation starts
- no sparse attention install log
- no slow-fast path
- `summary.json` exists
- `harness_traces/harness_trace_rank0.jsonl` exists
- trace contains `trace_schema_version=phase2.harness_trace.v2`

**Step 2: Run memory-recall smoke only if GPU/model/data are available**

Run:

```bash
CUDA_VISIBLE_DEVICES=4 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20342 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_phase2_smoke_memory_recall \
  --harness_mode memory_recall \
  --harness_memory_backend fake \
  --harness_memory_source episode-local
```

Expected:

- `harness_trace_rank0.jsonl` includes `recall_memory`
- `oracle_metrics_used_for_decision=false`
- runtime metadata fields are present
- no crash

**Step 3: Commit only code fixes**

Do not commit generated `results/` outputs unless the repository already tracks smoke artifacts intentionally.

---

## Recommended Implementation Order

1. Task 0: Baseline Characterization
2. Task 1: Skill Manifest Contract
3. Task 2: Registry Manifest Listing
4. Task 3: Built-In Skill Manifests
5. Task 4: OpenClaw Tool Adapter Skeleton
6. Task 5: Tool Call Metadata and Oracle Guard
7. Task 6: SpatialMemory Protocol Schemas
8. Task 7: SpatialMemory Protocol Documentation
9. Task 8: Embodiment Adapter Base Contract
10. Task 9: Runtime Trace Schema v2
11. Task 10: Connect Runtime Metadata to Logger
12. Task 11: Import Guard
13. Task 12: Phase 2 Documentation
14. Task 13: Full Test Sweep
15. Task 14: Optional Smoke Verification

## Notes for Execution

- Work from `ClawNav`, not the outer `Navigation_Claw` repository.
- Prefer one commit per task.
- If a task exposes a mismatch between this plan and current Phase 1 code, update the test to protect the intended contract before changing implementation.
- Keep Phase 2 additive. Any change that breaks Phase 1 behavior should be treated as a regression unless explicitly justified in the task notes.
