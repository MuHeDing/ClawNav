from typing import Any, Dict, List, Optional

from harness.types import VLNState


class BaseEmbodimentAdapter:
    def build_state(
        self,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        metrics: Optional[Dict[str, Any]],
        step_id: int,
        last_action: Optional[str] = None,
    ) -> VLNState:
        raise NotImplementedError

    def action_space(self) -> List[str]:
        raise NotImplementedError

    def action_to_executor_command(self, action_text: str) -> Dict[str, Any]:
        raise NotImplementedError
