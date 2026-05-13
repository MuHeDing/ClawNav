from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests


@dataclass
class RobotExecutorCommand:
    executor: str
    action_text: str
    runtime_executor: str
    command_id: str = ""
    accepted: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executor": self.executor,
            "action_text": self.action_text,
            "runtime_executor": self.runtime_executor,
            "command_id": self.command_id,
            "accepted": self.accepted,
        }


class FakeRobotExecutor:
    def __init__(self) -> None:
        self.commands: List[str] = []

    def command_for_action(self, action_text: str) -> Dict[str, Any]:
        self.commands.append(action_text)
        return RobotExecutorCommand(
            executor="robot_http",
            action_text=action_text,
            runtime_executor="openclaw_robot",
        ).to_dict()


class RobotHttpExecutor:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        post_json: Optional[Callable[[str, Dict[str, Any], float], Dict[str, Any]]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.post_json = post_json

    def command_for_action(self, action_text: str) -> Dict[str, Any]:
        response = self._post(f"{self.base_url}/action", {"action_text": action_text})
        return RobotExecutorCommand(
            executor="robot_http",
            action_text=action_text,
            runtime_executor="openclaw_robot",
            command_id=str(response.get("command_id", "")),
            accepted=bool(response.get("accepted", True)),
        ).to_dict()

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.post_json is not None:
            return self.post_json(url, payload, self.timeout_s)
        response = requests.post(url, json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return {}
        return data
