from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import requests

from harness.logging.harness_logger import ORACLE_KEYS
from harness.openclaw.instruction_stages import (
    InstructionStagePlan,
    StageFallbackCategory,
    parse_instruction_stage_plan,
)
from harness.openclaw.planner import OpenClawPlanDecision
from harness.types import VLNState


class OpenClawGatewayError(RuntimeError):
    pass


@dataclass(frozen=True)
class InstructionSegmentationResult:
    stage_plan: InstructionStagePlan
    runtime_metadata: Dict[str, Any]


GATEWAY_FORBIDDEN_KEYS = ORACLE_KEYS | {
    "future_observations",
    "future_trajectory_frames",
}


def strip_oracle_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    stripped: Dict[str, Any] = {}
    for key, value in data.items():
        if key in GATEWAY_FORBIDDEN_KEYS:
            continue
        if isinstance(value, dict):
            stripped[key] = strip_oracle_fields(value)
        elif isinstance(value, list):
            stripped[key] = [
                strip_oracle_fields(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            stripped[key] = value
    return stripped


def json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if not value:
            return []
        safe_items = []
        for item in value:
            safe_item = json_safe_value(item)
            if safe_item is not None:
                safe_items.append(safe_item)
        return safe_items or None
    if isinstance(value, dict):
        return json_safe_dict(value)
    return None


def json_safe_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in data.items():
        safe_value = json_safe_value(value)
        if safe_value is not None:
            safe[str(key)] = safe_value
    return safe


@dataclass
class OpenClawGatewayClient:
    base_url: str
    timeout_s: float = 5.0
    segmentation_timeout_s: float = 120.0
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
            "runtime_context": json_safe_dict(strip_oracle_fields(runtime_context)),
        }
        response = self._post(f"{self.base_url.rstrip('/')}/plan", payload)
        return gateway_response_to_decision(response)

    def segment_instruction(
        self,
        scene_id: str,
        episode_id: str,
        instruction: str,
    ) -> InstructionSegmentationResult:
        if not self._base_url_is_loopback():
            raise OpenClawGatewayError(
                "instruction segmentation requires a loopback gateway URL"
            )
        if not isinstance(instruction, str) or not instruction.strip():
            raise OpenClawGatewayError("instruction must be a non-empty string")
        if len(instruction.strip()) > 4096:
            raise OpenClawGatewayError("instruction exceeds 4096 characters")
        payload = {
            "scene_id": str(scene_id),
            "episode_id": str(episode_id),
            "instruction": instruction.strip(),
        }
        response = self._post(
            f"{self.base_url.rstrip('/')}/segment_instruction",
            payload,
            timeout_s=self.segmentation_timeout_s,
        )
        if set(response) != {"stage_plan", "runtime_metadata"}:
            raise OpenClawGatewayError("segmentation response fields mismatch")
        stage_payload = response.get("stage_plan")
        metadata = response.get("runtime_metadata")
        if not isinstance(stage_payload, dict) or not isinstance(metadata, dict):
            raise OpenClawGatewayError("segmentation response must contain objects")
        plan = parse_instruction_stage_plan(stage_payload, instruction.strip())
        fallback_value = str(metadata.get("fallback_category") or "none")
        try:
            fallback_category = StageFallbackCategory(fallback_value)
        except ValueError as exc:
            raise OpenClawGatewayError(
                "segmentation fallback category is invalid"
            ) from exc
        if (
            plan.fallback_category is not StageFallbackCategory.NONE
            and fallback_category is StageFallbackCategory.NONE
        ):
            raise OpenClawGatewayError("segmentation stage plan is invalid")
        plan = replace(
            plan,
            segmentation_source=str(metadata.get("segmentation_source") or "qwen"),
            fallback_category=fallback_category,
        )
        return InstructionSegmentationResult(
            stage_plan=plan,
            runtime_metadata=strip_oracle_fields(metadata),
        )

    def _base_url_is_loopback(self) -> bool:
        hostname = (urlparse(self.base_url).hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "::1"} or hostname.startswith(
            "127."
        )

    def _post(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        request_timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        if self.post_json is not None:
            return self.post_json(url, payload, request_timeout)
        try:
            session = requests.Session()
            session.trust_env = False
            response = session.post(url, json=payload, timeout=request_timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise OpenClawGatewayError(str(exc)) from exc
        if not isinstance(data, dict):
            raise OpenClawGatewayError("gateway response must be a JSON object")
        return data


class FakeOpenClawGatewayClient:
    def __init__(
        self,
        response: Dict[str, Any],
        segmentation_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.response = response
        self.segmentation_response = segmentation_response

    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> OpenClawPlanDecision:
        return gateway_response_to_decision(self.response)

    def segment_instruction(
        self,
        scene_id: str,
        episode_id: str,
        instruction: str,
    ) -> InstructionSegmentationResult:
        if self.segmentation_response is None:
            raise OpenClawGatewayError("fake segmentation response is not configured")
        client = OpenClawGatewayClient(
            base_url="http://127.0.0.1",
            post_json=lambda *_: self.segmentation_response or {},
        )
        return client.segment_instruction(scene_id, episode_id, instruction)


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
        runtime_metadata=strip_oracle_fields(
            response.get("runtime_metadata")
            if isinstance(response.get("runtime_metadata"), dict)
            else {}
        ),
    )
