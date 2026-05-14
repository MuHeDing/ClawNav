from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.tool_adapter import OpenClawToolAdapter
from harness.skill_registry import SkillRegistry
from harness.types import VLNState


PRE_ACTION_INTENTS = {
    "recall_memory",
    "write_memory",
    "verify_progress",
    "replan",
}


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
        fallback_planner: OpenClawPlannerProtocol = None,
    ) -> None:
        self.tool_adapter = OpenClawToolAdapter(tool_registry)
        self.planner = planner
        self.executor = executor
        self.fallback_planner = fallback_planner

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.tool_adapter.list_tools()

    def step(
        self,
        state: VLNState,
        payload: Dict[str, Any],
    ) -> OpenClawRuntimeStepResult:
        planner_error = ""
        planner_fallback = False
        try:
            decision = self.planner.plan(state, runtime_context=payload)
        except Exception as exc:
            planner_error = str(exc)
            if self.fallback_planner is None:
                return OpenClawRuntimeStepResult(
                    ok=False,
                    action_text="STOP",
                    runtime_metadata={
                        "runtime_mode": "openclaw_bridge",
                        "planner_fallback": False,
                        "planner_error": planner_error,
                    },
                    error=planner_error,
                )
            decision = self.fallback_planner.plan(state, runtime_context=payload)
            planner_fallback = True
        tool_calls: List[Dict[str, Any]] = []

        if decision.intent in PRE_ACTION_INTENTS:
            arguments = dict(decision.arguments)
            if decision.intent == "write_memory":
                candidate = payload.get("keyframe_candidate") or {}
                arguments = {**candidate, **arguments}
                arguments.setdefault("write_type", "episodic_keyframe")
            tool_result = self.tool_adapter.call_tool(
                decision.tool_name,
                arguments,
                state=state,
            )
            tool_calls.append(tool_result)

        nav_result = self.tool_adapter.call_tool(
            "NavigationPolicySkill",
            dict(payload),
            state=state,
        )
        tool_calls.append(nav_result)

        metadata = self._metadata(decision, tool_calls)
        if planner_error:
            metadata["planner_error"] = planner_error
        metadata["planner_fallback"] = planner_fallback
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
