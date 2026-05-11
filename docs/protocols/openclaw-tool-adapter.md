# OpenClaw Tool Adapter

## Purpose

Expose Phase 1 harness skills through an OpenClaw-compatible tool interface
without requiring OpenClaw runtime at import or evaluation time.

The adapter is an interface layer over the existing Python `SkillRegistry`.
It does not replace the Phase 1 controller or change `Skill.run()` behavior.

## APIs

`list_tools()`

Returns every registered skill manifest as a plain dictionary.

`get_tool_schema(name)`

Returns one tool schema by name, or `None` when the skill is not registered.

`call_tool(name, arguments, state)`

Runs a registered skill through the existing registry and returns a structured
runtime-compatible result envelope.

## Example Tool Schema

```json
{
  "schema_version": "phase2.skill_manifest.v1",
  "name": "MemoryQuerySkill",
  "description": "Query spatial memory for instruction, subgoal, or failure-recovery context.",
  "input_schema": {
    "type": "object",
    "properties": {
      "text": {"type": "string"},
      "reason": {"type": "string"},
      "n_results": {"type": "integer"}
    }
  },
  "output_schema": {"type": "object"},
  "timeout_ms": 5000,
  "side_effects": false,
  "oracle_safe": true,
  "callable_from_runtime": true
}
```

## Example Call Result

```json
{
  "ok": true,
  "result_type": "memory_query",
  "payload": {
    "policy_context": {},
    "control_context": {},
    "executor_context": {}
  },
  "confidence": 0.8,
  "error": null,
  "tool_name": "MemoryQuerySkill",
  "tool_schema_version": "phase2.skill_manifest.v1",
  "runtime_status": "completed",
  "latency_ms": 1.25,
  "error_type": ""
}
```

## No OpenClaw Runtime Dependency

The adapter depends only on harness `SkillRegistry`, `SkillManifest`, and
`SkillResult` contracts. Importing it must not import Habitat, load model
weights, initialize distributed mode, or require OpenClaw packages.

## No Oracle Decision Inputs

Tool calls reject online arguments containing `distance_to_goal`, `success`,
`SPL`, oracle path, oracle shortest path action, or future observations.

Rejected calls return `ok=false` and `error_type=oracle_input_rejected`.
