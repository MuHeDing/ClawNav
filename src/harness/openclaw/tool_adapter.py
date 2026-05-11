from typing import Any, Dict, List, Optional

from harness.skill_registry import SkillRegistry
from harness.types import SkillResult, VLNState


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
        result = self.registry.run(name, state, arguments)
        return self._result_to_dict(result)

    def _result_to_dict(self, result: SkillResult) -> Dict[str, Any]:
        return {
            "ok": result.ok,
            "result_type": result.result_type,
            "payload": result.payload,
            "confidence": result.confidence,
            "error": result.error,
        }
