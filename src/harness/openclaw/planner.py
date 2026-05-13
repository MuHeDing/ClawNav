from dataclasses import dataclass, field
from typing import Any, Dict

from harness.openclaw.subagents import SubagentRequest
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


class SubagentOpenClawPlanner:
    def __init__(self, client) -> None:
        self.client = client

    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> OpenClawPlanDecision:
        response = self.client.call(
            SubagentRequest(
                role="planner",
                instruction=state.instruction,
                context=runtime_context,
            )
        )
        return OpenClawPlanDecision(
            intent=str(response.get("intent", "act")),
            tool_name=str(response.get("tool_name", "NavigationPolicySkill")),
            arguments=dict(response.get("arguments", {})),
            reason=str(response.get("reason", "subagent")),
            planner_backend="subagent",
        )
