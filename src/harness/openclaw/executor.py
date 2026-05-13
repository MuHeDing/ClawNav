from typing import Any, Dict

from harness.env_adapters.base import BaseEmbodimentAdapter


class HabitatOpenClawExecutor:
    def __init__(self, adapter: BaseEmbodimentAdapter) -> None:
        self.adapter = adapter

    def command_for_action(self, action_text: str) -> Dict[str, Any]:
        command = self.adapter.action_to_executor_command(action_text)
        command["runtime_executor"] = "openclaw_habitat"
        return command
