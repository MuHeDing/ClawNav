# OpenClaw VLN Research System

## Purpose

Phase 4 extends the Phase 3 repo-local OpenClaw VLN runtime into a research-system boundary. The runtime remains import-safe without external OpenClaw, robot, or LLM/VLM services, while optional gateways can be enabled for experiments.

This document distinguishes runnable smoke verification from evidence for navigation claims. A one-episode smoke only proves that the runtime path executes; research claims require a planned ablation matrix, validated no-leakage memory manifests, and summarized navigation plus harness-level metrics.

## External OpenClaw Gateway

The default planner remains local and rule based. External OpenClaw planning is optional through the gateway planner backend and must be selected explicitly with `openclaw_planner_backend="gateway"` plus an `openclaw_gateway_url`.

Gateway requests serialize the online VLN state and non-oracle runtime context. The gateway response is converted back into an `OpenClawPlanDecision`. If a gateway is not configured, `evaluation_harness.py` must still import and run without an external OpenClaw package.

## Robot Executor

Habitat remains the default executor for VLN evaluation. The robot executor abstraction exists so a real robot or robot gateway can consume the same discrete action text without changing JanusVLN model internals.

The optional HTTP robot executor posts only the selected `action_text` to the robot gateway. It does not receive oracle metrics, future observations, or Habitat-only diagnostics used for offline evaluation.

## Scene-Prior Memory

Scene-Prior Memory uses an explicit manifest describing allowed memory records for a scene before evaluation starts. The manifest validator rejects records that expose online oracle values or evaluation-target labels as decision inputs.

This mode is intended to test whether train-time or prebuilt scene knowledge can improve the OpenClaw-style runtime while keeping online decisions constrained to permitted information.

## Train-Scene-Only Memory

Train-Scene-Only Memory is a stricter long-term memory setting where entries must come from approved train-scene sources. Evaluation scenes must not contribute future observations, success labels, SPL, distance-to-goal, or oracle shortest-path information.

The experiment scripts build and validate manifests before runtime use so the memory source is auditable independently from the navigation run.

## LLM/VLM Subagents

LLM/VLM Subagents are optional planner, critic, and memory-curator backends behind a repo-local protocol. The fake client is the default test double, and no external LLM/VLM service is required for imports or unit tests.

Subagent inputs are sanitized before dispatch. Oracle fields are stripped from context, and returned decisions are normalized into the same runtime metadata used by the local planner path.

## Ablation Matrix

The ablation matrix must include at least:

- low-memory baseline
- Phase 2 memory recall
- Phase 3 OpenClaw bridge
- scene-prior memory
- train-scene-only memory
- subagent planner

All full evaluation commands preserve the low-memory settings:

```text
--num_history 8
--max_pixels 401408
--kv_start_size 8
--kv_recent_size 24
```

The matrix must not enable LLM adaptive sparse attention or slow-fast active memory reuse.

## No Oracle Decision Inputs

Online planner, controller, critic, memory curator, memory query, and executor decisions must not use:

- `distance_to_goal`
- `success`
- `SPL`
- oracle shortest path
- oracle shortest path action
- future observations

Oracle values may be recorded as diagnostics only. Harness traces must keep `oracle_metrics_used_for_decision` false for valid Phase 4 evidence.

## Evidence Required Before Research Claims

Research claims require more than smoke tests. At minimum, provide:

- passing Phase 4 unit and regression tests
- py_compile coverage for changed runtime and script modules
- validated memory manifests for scene-prior and train-scene-only experiments
- ablation output directories with `summary.json` and harness traces
- summarized navigation metrics, trace step counts, memory recall counts, and oracle leakage counts
- enough episodes and seeds to support the claimed navigation trend

One-episode smoke results may be used only to confirm runtime operability.
