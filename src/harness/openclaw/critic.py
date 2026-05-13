from typing import Any, Dict

from harness.openclaw.subagents import SubagentRequest
from harness.types import VLNState


class SubagentProgressCritic:
    def __init__(self, client) -> None:
        self.client = client

    def evaluate(self, state: VLNState, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.call(
            SubagentRequest(
                role="critic",
                instruction=state.instruction,
                context=context,
            )
        )
