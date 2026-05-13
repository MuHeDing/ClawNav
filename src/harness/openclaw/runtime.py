from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.tool_adapter import OpenClawToolAdapter
from harness.skill_registry import SkillRegistry
from harness.types import VLNState


@dataclass
class OpenClawRuntimeStepResult:
    ok: bool
    action_text: str
    executor_command: Dict[str, Any] = field(default_factory=dict)
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class OpenClawPlannerProtocol(Protocol):
    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> Any:
        ...


class OpenClawVLNRuntime:
    def __init__(
        self,
        tool_registry: SkillRegistry,
        planner: OpenClawPlannerProtocol,
        executor: HabitatOpenClawExecutor,
    ) -> None:
        self.tool_adapter = OpenClawToolAdapter(tool_registry)
        self.planner = planner
        self.executor = executor

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.tool_adapter.list_tools()

    def step(
        self,
        state: VLNState,
        payload: Dict[str, Any],
    ) -> OpenClawRuntimeStepResult:
        decision = self.planner.plan(state, runtime_context=payload)
        tool_calls: List[Dict[str, Any]] = []

        if decision.intent == "recall_memory":
            recall = self.tool_adapter.call_tool(
                decision.tool_name,
                decision.arguments,
                state=state,
            )
            tool_calls.append(recall)

        nav_result = self.tool_adapter.call_tool(
            "NavigationPolicySkill",
            dict(payload),
            state=state,
        )
        tool_calls.append(nav_result)

        metadata = self._metadata(decision, tool_calls)
        if not nav_result.get("ok"):
            return OpenClawRuntimeStepResult(
                ok=False,
                action_text="STOP",
                runtime_metadata=metadata,
                error=nav_result.get("error") or "navigation_failed",
            )

        action_text = str(nav_result.get("payload", {}).get("action_text") or "STOP")
        return OpenClawRuntimeStepResult(
            ok=True,
            action_text=action_text,
            executor_command=self.executor.command_for_action(action_text),
            runtime_metadata=metadata,
        )

    def _metadata(self, decision, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "runtime_mode": "openclaw_bridge",
            "planner_backend": decision.planner_backend,
            "planned_intent": decision.intent,
            "planned_tool": decision.tool_name,
            "planner_reason": decision.reason,
            "tool_calls": [self._summarize_tool_call(call) for call in tool_calls],
        }

    def _summarize_tool_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = call.get("payload")
        payload_summary: Dict[str, Any] = {}
        if isinstance(payload, dict):
            payload_summary = {
                "keys": sorted(str(key) for key in payload.keys()),
            }
            for key in ("action_text", "images_used", "instruction"):
                if key in payload:
                    payload_summary[key] = payload[key]

        return {
            "ok": call.get("ok"),
            "tool_name": call.get("tool_name", ""),
            "result_type": call.get("result_type", ""),
            "runtime_status": call.get("runtime_status", ""),
            "latency_ms": call.get("latency_ms"),
            "error_type": call.get("error_type", ""),
            "error": call.get("error"),
            "payload_summary": payload_summary,
        }
