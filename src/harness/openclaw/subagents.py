from dataclasses import dataclass
from typing import Any, Dict, List

from harness.logging.harness_logger import ORACLE_KEYS


@dataclass
class SubagentRequest:
    role: str
    instruction: str
    context: Dict[str, Any]


def sanitize_subagent_context(context: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in context.items() if key not in ORACLE_KEYS}


class FakeSubagentClient:
    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        self.requests: List[SubagentRequest] = []

    def call(self, request: SubagentRequest) -> Dict[str, Any]:
        sanitized = SubagentRequest(
            role=request.role,
            instruction=request.instruction,
            context=sanitize_subagent_context(request.context),
        )
        self.requests.append(sanitized)
        return dict(self.response)
