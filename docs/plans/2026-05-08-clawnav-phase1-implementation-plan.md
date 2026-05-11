# ClawNav Phase 1 Implementation Plan

> **Implementation instruction:** Execute this plan task-by-task. For each task, first add failing tests, then implement the minimal code to pass, then run the specified tests before moving on.

**Goal:** Implement the Phase 1 OpenClaw-free ClawNav Harness prototype around the ClawNav low-memory no-sparse evaluation path, without changing the existing JanusVLN model internals.

**Architecture:** Add a `src/harness/` package that wraps the existing JanusVLN inference policy as `NavigationPolicySkill` and coordinates task memory, working memory, memory retrieval, progress critic, replanner, and logging through structured skill interfaces. Phase 1 uses a rule-constrained controller and fake / HTTP SpatialMemory clients, while preserving the original `evaluation_lowmem_no_llm_sparse.sh` baseline. The online controller must be non-oracle: it cannot use `distance_to_goal`, `success`, `SPL`, oracle path, or future observations for decisions.

**Tech Stack:** Python, dataclasses, pytest, requests, PIL, existing ClawNav evaluation stack, Habitat / VLN-CE only through the new evaluation entrypoint.

---

## Global Constraints

- Keep `ClawNav/scripts/evaluation_lowmem_no_llm_sparse.sh` unchanged.
- Add a new `evaluation_harness.py`; do not replace `evaluation.py`.
- Do not enable `--use_llm_adaptive_sparse_attention`.
- Do not add slow-fast active memory reuse.
- Do not modify Qwen attention, VGGT KV cache, model config, or model architecture.
- Online controller and critic must not use oracle metrics:
  - `distance_to_goal`
  - `success`
  - `SPL`
  - oracle shortest path
  - oracle shortest path action
  - future observations
- Oracle metrics may be logged only under diagnostics.
- Separate online state and diagnostics at the type level. Controller, MemoryManager, ProgressCritic, and Replanner may only read online state / online metrics.
- Pose and collision signals are online only if exposed by agent-accessible sensors or odometry. Simulator-only pose is diagnostics-only.
- First implementation uses `episode-local` and `fake` memory by default.
- Fake memory must not encode oracle answers; it is only for API, formatting, recall-control flow, and failure handling.
- Scene long-term memory must carry a `memory_source` field: `episode-local`, `scene-prior`, or `train-scene-only`.
- NavigationPolicySkill owns frame selection: `recent_frames` excludes `current_image`; the skill appends `state.current_image` before calling `model.call_model`.
- `evaluation_harness.py` must have no import-time side effects: no Habitat import, distributed initialization, or model loading at module import time if avoidable.

## New Files Overview

Create:

```text
ClawNav/src/harness/
  __init__.py
  types.py
  config.py
  controller.py
  skill_registry.py

  memory/
    __init__.py
    task_memory.py
    working_memory.py
    spatial_memory_client.py
    memory_manager.py

  skills/
    __init__.py
    base.py
    navigation_policy.py
    memory_query.py
    memory_write.py
    progress_critic.py
    replanner.py

  env_adapters/
    __init__.py
    habitat_vln_adapter.py

  logging/
    __init__.py
    harness_logger.py

ClawNav/src/evaluation_harness.py
ClawNav/scripts/evaluation_lowmem_harness.sh
```

Create tests:

```text
ClawNav/tests/test_harness_types.py
ClawNav/tests/test_task_memory.py
ClawNav/tests/test_working_memory.py
ClawNav/tests/test_spatial_memory_client.py
ClawNav/tests/test_memory_manager.py
ClawNav/tests/test_skill_registry.py
ClawNav/tests/test_navigation_policy_skill.py
ClawNav/tests/test_memory_skills.py
ClawNav/tests/test_progress_critic_replanner.py
ClawNav/tests/test_harness_controller.py
ClawNav/tests/test_harness_logger.py
ClawNav/tests/test_habitat_vln_adapter.py
ClawNav/tests/test_evaluation_harness_imports.py
```

Modify:

```text
ClawNav/tests/test_evaluation_scripts.py
```

---

## Task 0: Inspect Existing Evaluation Interfaces

**Files:**
- Read: `ClawNav/src/evaluation.py`
- Read: `ClawNav/scripts/evaluation_lowmem_no_llm_sparse.sh`
- Optional read: `ClawNav/tests/test_evaluation_scripts.py`

**Step 1: Inspect the current model/evaluation contract**

Confirm and write brief notes in your working summary before implementation:

```text
JanusVLN_Inference.call_model signature
actions2idx mapping
history frame sampling logic
rank/world_size handling
required lowmem CLI args
where result.json / summary.json are written
where max_pixels / min_pixels are set
```

Current expected facts:

```text
call_model(images, task, step_id)
actions: STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
history sampling: num_history with current frame included
baseline script: max_pixels=401408, kv_start_size=8, kv_recent_size=24, num_history=8
```

**Step 2: Do not edit code in this task**

This task is discovery only. If any expected fact is wrong, update the later implementation accordingly before coding.

---

## Task 1: Harness Contracts

**Files:**
- Create: `ClawNav/src/harness/__init__.py`
- Create: `ClawNav/src/harness/types.py`
- Create: `ClawNav/src/harness/config.py`
- Test: `ClawNav/tests/test_harness_types.py`

**Step 1: Write failing tests**

Create `tests/test_harness_types.py`:

```python
from harness.config import HarnessConfig
from harness.types import HarnessDecision, MemoryHit, SkillResult, VLNState


def test_skill_result_helpers():
    ok = SkillResult.ok_result("action", {"action_text": "MOVE_FORWARD"}, confidence=0.7)
    assert ok.ok is True
    assert ok.result_type == "action"
    assert ok.payload["action_text"] == "MOVE_FORWARD"
    assert ok.confidence == 0.7

    err = SkillResult.error_result("boom")
    assert err.ok is False
    assert err.error == "boom"


def test_harness_decision_is_structured():
    decision = HarnessDecision(
        intent="recall_memory",
        skill_name="MemoryQuerySkill",
        reason="initial_recall",
        payload={"text": "kitchen"},
    )
    assert decision.intent == "recall_memory"
    assert decision.payload["text"] == "kitchen"


def test_memory_hit_keeps_policy_and_control_fields():
    hit = MemoryHit(
        memory_id="m1",
        memory_type="place",
        name="kitchen",
        confidence=0.8,
        target_pose={"x": 1.0, "y": 2.0},
        evidence_text="kitchen entrance",
        image_path="/tmp/kitchen.jpg",
        memory_source="episode-local",
    )
    assert hit.target_pose["x"] == 1.0
    assert hit.memory_source == "episode-local"


def test_vln_state_separates_online_metrics_and_diagnostics():
    state = VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=0,
        current_image="frame",
        online_metrics={"collision": False},
        diagnostics={"distance_to_goal": 1.0, "success": False},
    )
    assert "distance_to_goal" not in state.online_metrics
    assert state.diagnostics["distance_to_goal"] == 1.0


def test_harness_config_defaults_are_bounded_and_non_oracle():
    cfg = HarnessConfig()
    assert cfg.max_internal_calls_per_step == 3
    assert cfg.recall_interval_steps >= 1
    assert cfg.memory_backend == "fake"
    assert cfg.allow_oracle_metrics_for_decision is False
```

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_harness_types.py -q
```

Expected: fail because `harness` does not exist.

**Step 3: Implement minimal contracts**

In `harness/types.py`, define:

```text
VLNState
HarnessDecision
SkillResult
MemoryHit
SubgoalState
TaskState
MemoryRecallResult
```

Required intents:

```text
act
recall_memory
write_memory
verify_progress
replan
stop
```

In `harness/config.py`, define `HarnessConfig` with:

```text
harness_mode = "memory_recall"
memory_backend = "fake"
spatial_memory_url = "http://127.0.0.1:8022"
max_internal_calls_per_step = 3
recall_interval_steps = 5
max_replans_per_episode = 3
max_memory_images = 2
max_prompt_context_chars = 1200
allow_oracle_metrics_for_decision = False
memory_source = "episode-local"
expose_sim_pose_online = False
```

`VLNState` must separate:

```text
online_metrics:
  only agent-accessible signals for controller / critic / memory manager

diagnostics:
  oracle metrics and evaluation-only signals, logger-only
```

Recommended `VLNState` fields:

```text
scene_id
episode_id
instruction
step_id
current_image
online_metrics
diagnostics
pose
diagnostic_pose
last_action
```

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_harness_types.py -q
```

Expected: pass.

---

## Task 2: Task-Level Memory

**Files:**
- Create: `ClawNav/src/harness/memory/__init__.py`
- Create: `ClawNav/src/harness/memory/task_memory.py`
- Test: `ClawNav/tests/test_task_memory.py`

**Step 1: Write failing tests**

Create `tests/test_task_memory.py`:

```python
from harness.memory.task_memory import TaskMemory


def test_single_subgoal_mode_uses_full_instruction():
    tm = TaskMemory()
    tm.reset("ep1", "go to the kitchen", mode="single")
    assert tm.current_subgoal.text == "go to the kitchen"
    assert tm.task_state.global_instruction == "go to the kitchen"


def test_rule_based_subgoal_mode_splits_instruction():
    tm = TaskMemory()
    tm.reset(
        "ep1",
        "Walk down the hallway and turn left at the sofa, then enter the kitchen and stop near the sink.",
        mode="rule",
    )
    texts = [sg.text for sg in tm.task_state.pending_subgoals]
    assert any("hallway" in text for text in texts)
    assert any("kitchen" in text for text in texts)


def test_task_memory_advances_subgoals():
    tm = TaskMemory()
    tm.reset("ep1", "go to kitchen", subgoals=["find hallway", "enter kitchen"])
    assert tm.current_subgoal.text == "find hallway"
    tm.mark_current_complete(reason="hallway visible")
    assert tm.current_subgoal.text == "enter kitchen"
    assert len(tm.task_state.completed_subgoals) == 1


def test_task_memory_records_failure_and_recovery():
    tm = TaskMemory()
    tm.reset("ep1", "go to kitchen", subgoals=["enter kitchen"])
    tm.mark_current_failed("stuck")
    tm.record_recovery_attempt("recall_memory")
    assert tm.task_state.failure_reason == "stuck"
    assert tm.task_state.recovery_attempts == ["recall_memory"]
```

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_task_memory.py -q
```

Expected: fail because `TaskMemory` does not exist.

**Step 3: Implement `TaskMemory`**

Implement:

```text
reset(episode_id, global_instruction, mode="single", subgoals=None)
decompose_instruction_rule_based(instruction)
mark_current_complete(reason)
mark_current_failed(reason)
record_recovery_attempt(action)
should_advance_subgoal()
```

Rule-based split should be intentionally simple:

```text
then
and then
turn
enter
stop
near
after
```

Do not call LLM planner in Phase 1.

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_task_memory.py -q
```

Expected: pass.

---

## Task 3: Working Memory Without Oracle Signals

**Files:**
- Create: `ClawNav/src/harness/memory/working_memory.py`
- Test: `ClawNav/tests/test_working_memory.py`

**Step 1: Write failing tests**

Create `tests/test_working_memory.py`:

```python
from harness.memory.working_memory import WorkingMemory


def test_recent_frames_are_bounded():
    wm = WorkingMemory(max_recent_frames=2)
    wm.append_frame("f0")
    wm.append_frame("f1")
    wm.append_frame("f2")
    assert wm.get_recent_frames(10) == ["f1", "f2"]


def test_repeated_turns_mark_possible_stuck():
    wm = WorkingMemory()
    for _ in range(5):
        wm.append_action("TURN_LEFT")
    assert wm.has_action_oscillation()


def test_low_displacement_is_non_oracle():
    wm = WorkingMemory()
    wm.append_pose([0.0, 0.0, 0.0])
    wm.append_pose([0.01, 0.0, 0.0])
    wm.append_pose([0.02, 0.0, 0.0])
    assert wm.has_low_displacement(threshold=0.05)


def test_oracle_metrics_are_not_used_for_decision():
    wm = WorkingMemory()
    wm.append_diagnostics({"distance_to_goal": 1.0, "success": True})
    assert wm.decision_metrics() == {}
    assert wm.diagnostic_metrics() == {"distance_to_goal": 1.0, "success": True}
```

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_working_memory.py -q
```

Expected: fail.

**Step 3: Implement `WorkingMemory`**

Implement:

```text
append_frame
append_action
append_pose
append_online_metrics
append_diagnostics
get_recent_frames
has_action_oscillation
has_low_displacement
has_repeated_observation
should_promote_keyframe
decision_metrics
diagnostic_metrics
```

Keep oracle metrics out of `decision_metrics`.

Pose notes:

```text
append_pose is online only for agent-accessible pose / odometry.
Simulator-only pose must be stored through diagnostics or diagnostic_pose.
```

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_working_memory.py -q
```

Expected: pass.

---

## Task 4: SpatialMemory Clients

**Files:**
- Create: `ClawNav/src/harness/memory/spatial_memory_client.py`
- Test: `ClawNav/tests/test_spatial_memory_client.py`

**Step 1: Write failing tests**

Create `tests/test_spatial_memory_client.py`:

```python
from harness.memory.spatial_memory_client import FakeSpatialMemoryClient, SpatialMemoryHttpClient


def test_fake_spatial_memory_returns_episode_local_hits():
    client = FakeSpatialMemoryClient(memory_source="episode-local")
    hits = client.query_semantic("kitchen", n_results=2)
    assert len(hits) <= 2
    assert hits[0].memory_source == "episode-local"


def test_http_client_maps_results(monkeypatch):
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "m1",
                        "memory_type": "place",
                        "name": "kitchen",
                        "target_pose": {"x": 1.0, "y": 2.0},
                        "confidence": 0.9,
                        "source": "test",
                        "timestamp": 1.0,
                        "evidence": {"note": "kitchen entrance", "image_path": "/tmp/a.jpg"},
                    }
                ]
            }

    def fake_post(url, json, timeout):
        return Resp()

    monkeypatch.setattr("requests.post", fake_post)
    client = SpatialMemoryHttpClient("http://memory", memory_source="scene-prior")
    hits = client.query_semantic("kitchen")
    assert hits[0].memory_id == "m1"
    assert hits[0].target_pose["x"] == 1.0
    assert hits[0].memory_source == "scene-prior"
```

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_spatial_memory_client.py -q
```

Expected: fail.

**Step 3: Implement clients**

Implement:

```text
BaseSpatialMemoryClient
FakeSpatialMemoryClient
SpatialMemoryHttpClient
```

HTTP client must communicate only through HTTP API. Do not import ABot-Claw modules.

Endpoints:

```text
GET  /health
POST /query/object
POST /query/place
POST /query/position
POST /query/semantic/text
POST /query/unified
POST /memory/semantic/ingest
```

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_spatial_memory_client.py -q
```

Expected: pass.

---

## Task 5: Memory Manager

**Files:**
- Create: `ClawNav/src/harness/memory/memory_manager.py`
- Test: `ClawNav/tests/test_memory_manager.py`

**Step 1: Write failing tests**

Create `tests/test_memory_manager.py`:

```python
from harness.config import HarnessConfig
from harness.memory.memory_manager import MemoryManager
from harness.memory.spatial_memory_client import FakeSpatialMemoryClient


def test_memory_manager_splits_contexts():
    manager = MemoryManager(FakeSpatialMemoryClient(), HarnessConfig())
    result = manager.recall(text="kitchen", step_id=0, reason="initial")
    assert result.policy_context["memory_context_text"]
    assert "hits" in result.control_context
    assert "target_poses" in result.executor_context


def test_recall_interval_blocks_overcalling():
    cfg = HarnessConfig(recall_interval_steps=5)
    manager = MemoryManager(FakeSpatialMemoryClient(), cfg)
    assert manager.should_recall(step_id=0, reason="initial") is True
    manager.mark_recalled(step_id=0)
    assert manager.should_recall(step_id=2, reason="periodic") is False


def test_memory_manager_proposes_write_without_side_effect():
    manager = MemoryManager(FakeSpatialMemoryClient(), HarnessConfig())
    decision = manager.propose_write(step_id=0, image_path="frame.jpg", note="start")
    assert decision["should_write"] is True
    assert decision["write_type"] == "episodic_keyframe"
```

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_memory_manager.py -q
```

Expected: fail.

**Step 3: Implement `MemoryManager`**

Responsibilities:

```text
decide should_recall
construct query payload
construct policy_context
construct control_context
construct executor_context
propose write payloads
```

Boundary:

```text
MemoryManager does not perform side-effectful storage.
MemoryWriteSkill performs storage.
```

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_memory_manager.py -q
```

Expected: pass.

---

## Task 6: Skill Base and Registry

**Files:**
- Create: `ClawNav/src/harness/skills/__init__.py`
- Create: `ClawNav/src/harness/skills/base.py`
- Create: `ClawNav/src/harness/skill_registry.py`
- Test: `ClawNav/tests/test_skill_registry.py`

**Step 1: Write failing tests**

Create tests for:

- register skill by name
- duplicate skill fails
- missing skill returns `SkillResult.error_result`
- skill exception is converted to structured error

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_skill_registry.py -q
```

Expected: fail.

**Step 3: Implement registry**

Implement:

```text
Skill
SkillRegistry.register
SkillRegistry.get
SkillRegistry.run
SkillRegistry.names
```

`run()` must catch exceptions and return structured `SkillResult`.

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_skill_registry.py -q
```

Expected: pass.

---

## Task 7: NavigationPolicySkill

**Files:**
- Create: `ClawNav/src/harness/skills/navigation_policy.py`
- Test: `ClawNav/tests/test_navigation_policy_skill.py`

**Step 1: Write failing tests**

Create `tests/test_navigation_policy_skill.py`:

```python
from harness.skills.navigation_policy import NavigationPolicySkill
from harness.types import VLNState


class FakeModel:
    def __init__(self):
        self.calls = []

    def call_model(self, images, task, step_id):
        self.calls.append((images, task, step_id))
        return ["MOVE_FORWARD"]


def test_navigation_policy_augments_instruction_but_not_model_architecture():
    model = FakeModel()
    skill = NavigationPolicySkill(model=model, num_history=8, max_memory_images=0)
    state = VLNState(
        scene_id="s",
        episode_id="e",
        instruction="go to kitchen",
        step_id=3,
        current_image="cur",
        online_metrics={},
        diagnostics={},
    )
    result = skill.run(
        state,
        {
            "recent_frames": ["r1", "r2"],
            "memory_images": ["m1"],
            "active_subgoal": "enter kitchen",
            "memory_context_text": "Remembered kitchen entrance.",
        },
    )
    assert result.ok
    assert result.payload["action_text"] == "MOVE_FORWARD"
    images, task, step_id = model.calls[0]
    assert images[-1] == "cur"
    assert "Current subgoal: enter kitchen" in task
    assert "Remembered kitchen entrance." in task
    assert "m1" not in images
```

Frame selection rule:

```text
recent_frames excludes current_image.
NavigationPolicySkill appends state.current_image before calling model.call_model.
```

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_navigation_policy_skill.py -q
```

Expected: fail.

**Step 3: Implement skill**

Rules:

- Do not modify model architecture.
- Inject memory text and active subgoal by instruction augmentation.
- Use existing recent-frame interface.
- Pass memory images only if `max_memory_images > 0`.
- Default Phase 1 should use `max_memory_images=0`.
- `recent_frames` must not contain current image; the skill owns appending `state.current_image`.

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_navigation_policy_skill.py -q
```

Expected: pass.

---

## Task 8: Memory Query and Write Skills

**Files:**
- Create: `ClawNav/src/harness/skills/memory_query.py`
- Create: `ClawNav/src/harness/skills/memory_write.py`
- Test: `ClawNav/tests/test_memory_skills.py`

**Step 1: Write failing tests**

Test:

- `MemoryQuerySkill` returns `memory_hits`.
- `MemoryWriteSkill` stores episodic records through a provided store/client.
- `MemoryWriteSkill` accepts proposed write payload from `MemoryManager`.
- write skill validates `memory_source`.

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_memory_skills.py -q
```

Expected: fail.

**Step 3: Implement skills**

`MemoryQuerySkill` delegates to `MemoryManager.recall()`.

`MemoryWriteSkill` is the only module that performs side effects:

```text
append episodic JSON record
or call SpatialMemoryHttpClient.ingest_semantic
```

Do not let `MemoryManager` write files or call ingest endpoints directly.

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_memory_skills.py -q
```

Expected: pass.

---

## Task 9: Non-Oracle Progress Critic and Replanner

**Files:**
- Create: `ClawNav/src/harness/skills/progress_critic.py`
- Create: `ClawNav/src/harness/skills/replanner.py`
- Test: `ClawNav/tests/test_progress_critic_replanner.py`

**Step 1: Write failing tests**

Test:

- repeated turns produce `possible_stuck`.
- low displacement produces `low_displacement`.
- STOP plus poor semantic/memory consistency produces `risky_stop`.
- critic ignores `distance_to_goal`.
- replanner creates recovery subgoal from active subgoal and failure reason.

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_progress_critic_replanner.py -q
```

Expected: fail.

**Step 3: Implement critic/replanner**

Critic allowed signals:

```text
recent action pattern
pose displacement
visual repetition
policy output pattern
memory consistency
elapsed steps / max_steps
active subgoal status
```

Critic forbidden signals:

```text
distance_to_goal
success
SPL
oracle path
future observations
```

Replanner output:

```text
active_subgoal
memory_query_hint
reason
```

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_progress_critic_replanner.py -q
```

Expected: pass.

---

## Task 10: Harness Controller

**Files:**
- Create: `ClawNav/src/harness/controller.py`
- Test: `ClawNav/tests/test_harness_controller.py`

**Step 1: Write failing tests**

Test:

- `act_only` calls only `NavigationPolicySkill`.
- first step in `memory_recall` recalls before acting.
- controller obeys `max_internal_calls_per_step`.
- skill exception falls back to baseline navigation action.
- controller never passes `distance_to_goal` into critic decision payload.
- controller does not pass diagnostics to MemoryQuerySkill, ProgressCriticSkill, or ReplannerSkill payloads.
- risky stop triggers verify/recall/replan path without oracle metric.

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_harness_controller.py -q
```

Expected: fail.

**Step 3: Implement controller**

Modes:

```text
act_only
memory_recall
memory_critic
full
```

Phase-1 controller policy:

```text
if risky_stop:
  verify_progress or recall_memory
elif stuck:
  recall_memory -> replan if recall does not resolve uncertainty
elif should_write_keyframe:
  write_memory -> act
elif recall_interval reached and active_subgoal needs landmark:
  recall_memory -> act
else:
  act
```

Fallback:

```text
If any non-navigation skill fails, timeout, returns invalid output, or exceeds budget,
fall back to NavigationPolicySkill for that step.
If NavigationPolicySkill fails, return STOP and log fallback=true.
```

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_harness_controller.py -q
```

Expected: pass.

---

## Task 11: Harness Logger

**Files:**
- Create: `ClawNav/src/harness/logging/__init__.py`
- Create: `ClawNav/src/harness/logging/harness_logger.py`
- Test: `ClawNav/tests/test_harness_logger.py`

**Step 1: Write failing tests**

Test:

- JSONL record is appended.
- parent directory is created.
- diagnostics include oracle metrics only under `diagnostics`.
- record contains `oracle_metrics_used_for_decision=false`.
- record contains `memory_source`.
- record contains `decision_inputs`.
- record contains `oracle_guard_passed=true`.

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_harness_logger.py -q
```

Expected: fail.

**Step 3: Implement logger**

Output:

```text
results/.../harness_trace_rank{rank}.jsonl
```

Each record should include:

```text
scene_id
episode_id
step_id
intent
skill
reason
memory_backend
num_memory_hits
active_subgoal
subgoal_status
action_text
fallback
diagnostics
decision_inputs
oracle_guard_passed
```

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_harness_logger.py -q
```

Expected: pass.

---

## Task 12: Habitat VLN Adapter

**Files:**
- Create: `ClawNav/src/harness/env_adapters/__init__.py`
- Create: `ClawNav/src/harness/env_adapters/habitat_vln_adapter.py`
- Test: `ClawNav/tests/test_habitat_vln_adapter.py`

**Step 1: Write failing tests**

Use fake env/episode objects; do not require Habitat import in unit tests.

Test:

- extracts RGB as current image.
- extracts instruction.
- extracts pose if available.
- separates online pose from diagnostic simulator pose.
- stores oracle metrics only in diagnostics / raw metrics, not decision metrics.

Add a test like:

```python
def test_adapter_separates_online_pose_from_diagnostic_pose():
    state = adapter.build_state(..., expose_pose_online=False)
    assert state.pose is None
    assert "sim_position" in state.diagnostics
```

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_habitat_vln_adapter.py -q
```

Expected: fail.

**Step 3: Implement adapter**

Rules:

- Avoid importing Habitat at module import time.
- Use duck typing for `env.sim.get_agent_state()`.
- Populate `VLNState.online_metrics` only with agent-accessible signals.
- Put `distance_to_goal`, `success`, simulator-only pose, and other evaluation-only fields in `VLNState.diagnostics`.
- If `expose_pose_online=False`, simulator pose goes to `diagnostics["sim_position"]`, not `state.pose`.

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_habitat_vln_adapter.py -q
```

Expected: pass.

---

## Task 13: Evaluation Harness Entrypoint

**Files:**
- Create: `ClawNav/src/evaluation_harness.py`
- Test: `ClawNav/tests/test_evaluation_harness_imports.py`

**Step 1: Write failing tests**

Test:

- `src/evaluation_harness.py` exists.
- it can be imported without running evaluation.
- parser helper includes harness args.
- no sparse / slow-fast flags are required for harness operation.
- importing the module does not initialize distributed mode.
- importing the module does not load model weights.

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_evaluation_harness_imports.py -q
```

Expected: fail.

**Step 3: Implement `evaluation_harness.py`**

Use original `evaluation.py` as reference, but keep changes scoped:

- Create parser helper, e.g. `build_parser()`.
- Add debug arg `--harness_debug_max_episodes`, default `None`.
- Reuse `JanusVLN_Inference` if safe.
- Build `HarnessConfig`.
- Build task memory, working memory, memory client, memory manager, skill registry, controller, adapter, logger.
- In episode loop:

```python
state = adapter.build_state(env, observations, episode, step_id, last_action)
action_text = harness.step(state)
action = actions2idx[action_text]
observations = env.step(action)
```

Do not install sparse attention.

Import-time constraints:

```text
No Habitat import at module import time if possible.
No model load at module import time.
No distributed init at module import time.
main() should be guarded by if __name__ == "__main__".
```

**Step 4: Run import test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_evaluation_harness_imports.py -q
```

Expected: pass.

---

## Task 14: Low-Memory Harness Script

**Files:**
- Create: `ClawNav/scripts/evaluation_lowmem_harness.sh`
- Modify: `ClawNav/tests/test_evaluation_scripts.py`

**Step 1: Write failing script tests**

Extend `test_evaluation_scripts.py`:

```python
def test_lowmem_harness_script_uses_no_sparse_lowmem_config():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "evaluation_lowmem_harness.sh"
    contents = script.read_text(encoding="utf-8")

    assert "src/evaluation_harness.py" in contents
    assert "max_pixels=401408" in contents
    assert "kv_start_size=8" in contents
    assert "kv_recent_size=24" in contents
    assert "num_history=8" in contents
    assert "--use_llm_adaptive_sparse_attention" not in contents
    assert "--enable_slow_fast" not in contents
```

**Step 2: Run test to verify failure**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_evaluation_scripts.py -q
```

Expected: fail because script does not exist.

**Step 3: Create script**

Copy structure from:

```text
scripts/evaluation_lowmem_no_llm_sparse.sh
```

Use:

```text
src/evaluation_harness.py
OUTPUT_PATH="results/janusvln_extra_lowmem_401408_start8_recent24_history8_clawnav"
```

Add:

```bash
--harness_mode memory_recall
--harness_max_internal_calls 3
--harness_recall_interval 5
--harness_memory_backend fake
--harness_memory_source episode-local
--harness_debug_max_episodes 1
```

Do not add sparse or slow-fast flags.

**Step 4: Run test**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_evaluation_scripts.py -q
```

Expected: pass.

---

## Task 15: Unit Test Sweep and Compile Check

**Files:**
- No new files unless fixes are required.

**Step 1: Run new harness tests**

Run:

```bash
cd ClawNav
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

**Step 2: Run existing lightweight tests**

Run:

```bash
cd ClawNav
PYTHONPATH=src pytest tests/test_llm_visual_pruner.py tests/test_adaptive_sparse_attention.py tests/test_evaluation_scripts.py -q
```

Expected: pass.

**Step 3: Compile new files**

Run:

```bash
cd ClawNav
python -m py_compile \
  src/harness/types.py \
  src/harness/config.py \
  src/harness/controller.py \
  src/harness/skill_registry.py \
  src/harness/memory/task_memory.py \
  src/harness/memory/working_memory.py \
  src/harness/memory/spatial_memory_client.py \
  src/harness/memory/memory_manager.py \
  src/harness/skills/base.py \
  src/harness/skills/navigation_policy.py \
  src/harness/skills/memory_query.py \
  src/harness/skills/memory_write.py \
  src/harness/skills/progress_critic.py \
  src/harness/skills/replanner.py \
  src/harness/env_adapters/habitat_vln_adapter.py \
  src/harness/logging/harness_logger.py \
  src/evaluation_harness.py
```

Expected: exit 0.

---

## Task 16: Short Smoke Runs

**Files:**
- No code files unless bugs are found.

**Step 1: Act-only smoke**

Run only if model/data/GPU are available:

```bash
cd ClawNav
CUDA_VISIBLE_DEVICES=4 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20331 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_smoke_act_only \
  --harness_mode act_only \
  --harness_memory_backend fake \
  --harness_memory_source episode-local
```

Expected:

- evaluation starts
- no sparse attention install log
- no slow-fast path
- trace file exists

**Step 2: Memory-recall smoke**

Run:

```bash
cd ClawNav
CUDA_VISIBLE_DEVICES=4 /ssd/dingmuhe/anaconda3/envs/janusvln/bin/torchrun --nproc_per_node=1 \
  --master_port=20332 \
  src/evaluation_harness.py \
  --model_path /ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra \
  --habitat_config_path config/vln_r2r.yaml \
  --num_history 8 \
  --max_pixels 401408 \
  --kv_start_size 8 \
  --kv_recent_size 24 \
  --max_steps 30 \
  --harness_debug_max_episodes 1 \
  --output_path results/clawnav_smoke_memory_recall \
  --harness_mode memory_recall \
  --harness_memory_backend fake \
  --harness_memory_source episode-local
```

Expected:

- `harness_trace_rank0.jsonl` includes `recall_memory`
- diagnostics include `oracle_metrics_used_for_decision=false`
- no crash

---

## Phase 1 Completion Criteria

Phase 1 is complete when:

- All new harness tests pass.
- Existing lightweight tests still pass.
- `evaluation_lowmem_no_llm_sparse.sh` remains unchanged.
- `evaluation_lowmem_harness.sh` exists and uses lowmem no-sparse settings.
- `evaluation_harness.py` runs at least an act-only smoke test.
- Harness trace includes:
  - intent
  - skill
  - reason
  - active subgoal
  - memory source
  - fallback
  - diagnostics
  - `oracle_metrics_used_for_decision=false`
- Online controller never uses oracle metrics for decisions.
- Scene memory usage is tagged as `episode-local`, `scene-prior`, or `train-scene-only`.
