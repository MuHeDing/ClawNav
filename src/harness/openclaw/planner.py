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
