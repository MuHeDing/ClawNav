import json
from pathlib import Path
from typing import Any, Dict, Optional

from harness.types import VLNState


ORACLE_KEYS = {
    "distance_to_goal",
    "success",
    "SPL",
    "spl",
    "oracle_path",
    "oracle_shortest_path",
    "oracle_shortest_path_action",
    "oracle_action",
}


class HarnessLogger:
    def __init__(self, path: str, rank: Optional[int] = None) -> None:
        trace_path = Path(path)
        if trace_path.suffix != ".jsonl":
            filename = f"harness_trace_rank{0 if rank is None else rank}.jsonl"
            trace_path = trace_path / filename
        self.path = trace_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        state: VLNState,
        intent: str,
        skill: str,
        reason: str = "",
        memory_backend: str = "",
        memory_source: str = "episode-local",
        num_memory_hits: int = 0,
        active_subgoal: str = "",
        subgoal_status: str = "",
        action_text: str = "",
        fallback: bool = False,
        diagnostics: Optional[Dict[str, Any]] = None,
        decision_inputs: Optional[Dict[str, Any]] = None,
        runtime: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        decision_inputs = decision_inputs or {}
        diagnostics = diagnostics if diagnostics is not None else state.diagnostics
        oracle_metrics_used = any(key in decision_inputs for key in ORACLE_KEYS)
        record: Dict[str, Any] = {
            "scene_id": state.scene_id,
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "intent": intent,
            "skill": skill,
            "reason": reason,
            "memory_backend": memory_backend,
            "memory_source": memory_source,
            "num_memory_hits": num_memory_hits,
            "active_subgoal": active_subgoal,
            "subgoal_status": subgoal_status,
            "action_text": action_text,
            "fallback": fallback,
            "diagnostics": diagnostics,
            "decision_inputs": decision_inputs,
            "oracle_metrics_used_for_decision": oracle_metrics_used,
            "oracle_guard_passed": not oracle_metrics_used,
        }
        runtime_fields: Dict[str, Any] = {
            "trace_schema_version": "phase2.harness_trace.v2",
            "skill_call_id": "",
            "parent_call_id": "",
            "tool_schema_version": "",
            "latency_ms": None,
            "input_summary": {},
            "output_summary": {},
            "error_type": "",
            "runtime_status": "fallback" if fallback else "completed",
            "runtime_mode": "",
            "planner_backend": "",
            "planned_intent": "",
            "planned_tool": "",
            "planner_reason": "",
            "runtime_executor": "",
            "tool_calls": [],
        }
        if runtime:
            runtime_fields.update(runtime)
        record.update(runtime_fields)
        if extra:
            record.update(extra)

        record = self._json_safe(record)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def _json_safe(self, value: Any, seen: Optional[set] = None) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if seen is None:
            seen = set()
        object_id = id(value)
        if object_id in seen:
            return str(value)
        seen.add(object_id)
        try:
            return self._json_safe_container(value, seen)
        finally:
            seen.discard(object_id)

    def _json_safe_container(self, value: Any, seen: set) -> Any:
        if all(hasattr(value, attr) for attr in ("w", "x", "y", "z")):
            return [
                self._json_safe(getattr(value, "w"), seen),
                self._json_safe(getattr(value, "x"), seen),
                self._json_safe(getattr(value, "y"), seen),
                self._json_safe(getattr(value, "z"), seen),
            ]
        if isinstance(value, dict):
            return {
                str(key): self._json_safe(nested, seen)
                for key, nested in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item, seen) for item in value]
        if hasattr(value, "item"):
            try:
                converted = value.item()
                if converted is not value:
                    return self._json_safe(converted, seen)
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                converted = value.tolist()
                if converted is not value:
                    return self._json_safe(converted, seen)
            except Exception:
                pass
        return str(value)
