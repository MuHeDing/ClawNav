from typing import Any, Dict

from harness.skills.base import Skill
from harness.types import SkillResult, VLNState


class ReplannerSkill(Skill):
    name = "ReplannerSkill"
    description = "Create a recovery subgoal from non-oracle failure signals and memory hits."
    input_schema = {
        "type": "object",
        "properties": {
            "task_memory": {"type": "object"},
            "failure_reason": {"type": "string"},
            "memory_hits": {"type": "array"},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "active_subgoal": {"type": "string"},
            "memory_query_hint": {"type": "string"},
            "reason": {"type": "string"},
            "memory_hits": {"type": "array"},
        },
        "required": ["active_subgoal"],
    }
    oracle_safe = True

    def run(self, state: VLNState, payload: Dict[str, Any]) -> SkillResult:
        task_memory = payload.get("task_memory")
        failure_reason = payload.get("failure_reason") or "unknown_failure"
        current_text = self._current_subgoal_text(task_memory) or state.instruction
        memory_query_hint = current_text
        recovery_subgoal = f"Recover from {failure_reason}: {current_text}"

        return SkillResult.ok_result(
            "replan",
            {
                "active_subgoal": recovery_subgoal,
                "memory_query_hint": memory_query_hint,
                "reason": failure_reason,
                "memory_hits": payload.get("memory_hits", []),
            },
            confidence=0.7,
        )

    def _current_subgoal_text(self, task_memory: Any) -> str:
        if task_memory is None:
            return ""
        current = getattr(task_memory, "current_subgoal", None)
        if current is None:
            return ""
        return getattr(current, "text", "") or ""
