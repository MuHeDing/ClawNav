from typing import Any, Dict

from harness.memory.memory_manager import MemoryManager
from harness.skills.base import Skill
from harness.types import SkillResult


class MemoryQuerySkill(Skill):
    name = "MemoryQuerySkill"
    description = "Query spatial memory for instruction, subgoal, or failure-recovery context."
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "query": {"type": "string"},
            "step_id": {"type": "integer"},
            "reason": {"type": "string"},
            "n_results": {"type": "integer"},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "memory_hits": {"type": "array"},
            "query": {"type": "string"},
            "backend": {"type": "string"},
            "policy_context": {"type": "object"},
            "control_context": {"type": "object"},
            "executor_context": {"type": "object"},
        },
    }
    oracle_safe = True

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager

    def run(self, state: Any, payload: Dict[str, Any]) -> SkillResult:
        text = payload.get("text") or payload.get("query") or ""
        step_id = int(payload.get("step_id", getattr(state, "step_id", 0)))
        reason = payload.get("reason", "")
        n_results = int(payload.get("n_results", 5))
        recall = self.memory_manager.recall(
            text=text,
            step_id=step_id,
            reason=reason,
            n_results=n_results,
        )
        return SkillResult.ok_result(
            "memory_query",
            {
                "memory_hits": recall.hits,
                "query": recall.query,
                "backend": recall.backend,
                "policy_context": recall.policy_context,
                "control_context": recall.control_context,
                "executor_context": recall.executor_context,
            },
            confidence=recall.control_context.get("confidence", 0.0),
        )
