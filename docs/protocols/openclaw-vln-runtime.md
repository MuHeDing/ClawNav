# OpenClaw VLN Runtime

## Purpose

Phase 3 runs ClawNav through an OpenClaw-style planner, tool, and executor runtime. The runtime keeps the Phase 2 tool adapter and skill registry as the local execution boundary, then adds a small planner/executor loop for VLN decisions.

This mode is intended to exercise the same ClawNav navigation and memory skills through an OpenClaw-compatible runtime shape without changing JanusVLN model internals.

## No Required OpenClaw Python Dependency

External OpenClaw, ABot-Claw, and RoboClaw Python packages are optional integrations. The default Phase 3 implementation uses a repo-local bridge under `harness.openclaw` and must remain import-safe when those packages are unavailable.

If a real OpenClaw gateway is added later, it should sit behind an optional runtime adapter rather than becoming a required import path for `evaluation_harness.py`.

## Runtime Modes

`phase2` is the default compatibility path. It runs the existing harness controller and preserves the Phase 2 no-OpenClaw baseline behavior.

`openclaw_bridge` is the Phase 3 opt-in path. It builds `OpenClawVLNRuntime`, lists skills through `OpenClawToolAdapter`, calls planned tools through `SkillRegistry`, and converts navigation actions through `HabitatOpenClawExecutor`.

## Tool Flow

The local bridge follows this flow:

```text
RuleOpenClawPlanner -> OpenClawToolAdapter -> SkillRegistry -> HabitatOpenClawExecutor
```

At each runtime step, the planner chooses an intent and tool. Memory recall uses `MemoryQuerySkill`; navigation action selection uses `NavigationPolicySkill`; the executor maps the action text to a Habitat discrete command.

Harness traces preserve Phase 2 fields and add OpenClaw runtime metadata such as `runtime_mode`, `planner_backend`, `planned_intent`, `planned_tool`, `runtime_executor`, and summarized `tool_calls`.

## SpatialMemory

The bridge can read ABot-Claw-style service metadata from `SERVICE.md` through `OpenClawServiceRegistry`. When `--harness_memory_backend spatial_http` and `--openclaw_service_registry_path` are both provided, the SpatialMemory base URL from the registry overrides `--spatial_memory_url`.

Without an explicit service registry path, existing `spatial_http` behavior is unchanged.

## No Oracle Decision Inputs

Online planner, controller, critic, and executor decisions must not use oracle metrics. Forbidden decision inputs include:

- `distance_to_goal`
- `success`
- `SPL`
- oracle shortest path
- oracle shortest path action
- future observations

Oracle metrics may be logged as diagnostics only. They must not be forwarded into planner context, skill arguments, or executor commands used for online decisions.
