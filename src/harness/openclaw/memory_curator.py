from typing import Any, Dict

from harness.openclaw.subagents import SubagentRequest
from harness.types import VLNState


class SubagentMemoryCurator:
    def __init__(self, client) -> None:
        self.client = client

    def should_write(self, state: VLNState, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.call(
            SubagentRequest(
                role="memory_curator",
                instruction=state.instruction,
                context=context,
            )
        )
