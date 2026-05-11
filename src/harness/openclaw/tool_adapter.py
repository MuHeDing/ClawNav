import time
from typing import Any, Dict, List, Optional

from harness.skill_registry import SkillRegistry
from harness.types import SkillResult, VLNState


ORACLE_DECISION_KEYS = {
    "distance_to_goal",
    "success",
    "SPL",
    "spl",
    "oracle_path",
    "oracle_shortest_path",
    "oracle_shortest_path_action",
    "oracle_action",
}


class OpenClawToolAdapter:
    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.registry.export_tool_schemas()

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        for tool in self.list_tools():
            if tool["name"] == name:
                return tool
        return None

    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        state: VLNState,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        tool_schema = self.get_tool_schema(name) or {}
        rejected_keys = sorted(set(arguments) & ORACLE_DECISION_KEYS)
        if rejected_keys:
            latency_ms = (time.perf_counter() - start) * 1000.0
            result = SkillResult.error_result(
                f"Oracle decision inputs rejected: {', '.join(rejected_keys)}"
            )
            return self._result_to_dict(
                result,
                tool_name=name,
                tool_schema_version=tool_schema.get("schema_version", ""),
                latency_ms=latency_ms,
                error_type="oracle_input_rejected",
            )

        result = self.registry.run(name, state, arguments)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return self._result_to_dict(
            result,
            tool_name=name,
            tool_schema_version=tool_schema.get("schema_version", ""),
            latency_ms=latency_ms,
        )

    def _result_to_dict(
        self,
        result: SkillResult,
        tool_name: str,
        tool_schema_version: str,
        latency_ms: float,
        error_type: str = "",
    ) -> Dict[str, Any]:
        return {
            "ok": result.ok,
            "result_type": result.result_type,
            "payload": result.payload,
            "confidence": result.confidence,
            "error": result.error,
            "tool_name": tool_name,
            "tool_schema_version": tool_schema_version,
            "runtime_status": "completed" if result.ok else "failed",
            "latency_ms": latency_ms,
            "error_type": error_type,
        }
