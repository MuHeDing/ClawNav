from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import requests

from harness.logging.harness_logger import ORACLE_KEYS
from harness.openclaw.planner import OpenClawPlanDecision
from harness.types import VLNState


class OpenClawGatewayError(RuntimeError):
    pass


def strip_oracle_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if key not in ORACLE_KEYS}


@dataclass
class OpenClawGatewayClient:
    base_url: str
    timeout_s: float = 5.0
    post_json: Optional[Callable[[str, Dict[str, Any], float], Dict[str, Any]]] = None

    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> OpenClawPlanDecision:
        payload = {
            "state": {
                "scene_id": state.scene_id,
                "episode_id": state.episode_id,
                "instruction": state.instruction,
                "step_id": state.step_id,
                "last_action": state.last_action,
            },
            "runtime_context": strip_oracle_fields(runtime_context),
        }
        response = self._post(f"{self.base_url.rstrip('/')}/plan", payload)
        return gateway_response_to_decision(response)

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.post_json is not None:
            return self.post_json(url, payload, self.timeout_s)
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise OpenClawGatewayError(str(exc)) from exc
        if not isinstance(data, dict):
            raise OpenClawGatewayError("gateway response must be a JSON object")
        return data


class FakeOpenClawGatewayClient:
    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response

    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> OpenClawPlanDecision:
        return gateway_response_to_decision(self.response)


def gateway_response_to_decision(response: Dict[str, Any]) -> OpenClawPlanDecision:
    missing = [key for key in ("intent", "tool_name") if key not in response]
    if missing:
        raise OpenClawGatewayError(f"gateway response missing {', '.join(missing)}")
    arguments = response.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise OpenClawGatewayError("gateway arguments must be an object")
    return OpenClawPlanDecision(
        intent=str(response["intent"]),
        tool_name=str(response["tool_name"]),
        arguments=strip_oracle_fields(arguments),
        reason=str(response.get("reason", "")),
        planner_backend="gateway",
    )
