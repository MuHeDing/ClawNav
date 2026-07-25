import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests

from harness.openclaw.gateway_server import GatewayRequestError, make_gateway_server
from harness.openclaw.instruction_stages import (
    INSTRUCTION_STAGE_SCHEMA_VERSION,
    StageFallbackCategory,
    parse_instruction_stage_plan,
    single_stage_fallback,
)
from harness.openclaw.visual_analyzer import OpenClawVisualAnalyzer


OpenClawRunner = Callable[[List[str], float], subprocess.CompletedProcess]


INTENT_TOOL_NAMES = {
    "act": "NavigationPolicySkill",
    "recall_memory": "MemoryQuerySkill",
    "write_memory": "MemoryWriteSkill",
    "verify_progress": "ProgressCriticSkill",
    "replan": "ReplannerSkill",
}

ACTION_TEXTS = {
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
}

ACTION_ALIASES = {
    "FORWARD": "MOVE_FORWARD",
    "MOVE": "MOVE_FORWARD",
    "LEFT": "TURN_LEFT",
    "RIGHT": "TURN_RIGHT",
}

PROMPT_STATE_KEYS = ("scene_id", "episode_id", "instruction", "step_id", "last_action")
PROMPT_RUNTIME_CONTEXT_KEYS = (
    "run_id",
    "policy_action",
    "current_image_path",
    "recent_keyframe_paths",
    "memory_context_text",
    "memory_images",
    "retrieved_memory_image_paths",
    "retrieved_memory_images",
    "policy_input",
    "control_context",
    "evidence_context",
    "input_regime",
    "map_assist_mode",
    "map_frame_interval_steps",
    "motion_feedback_enabled",
    "forward_stall_odometry_enabled",
    "map_collision_overlay_enabled",
    "motion_feedback",
    "map_context",
    "blocked_stop_feedback",
    "forward_stall_feedback",
    "stop_verification_feedback",
    "route_progress",
    "task_state",
    "recent_step_summary",
    "retrieved_memory_ids",
    "visual_evidence_registry",
    "visual_recovery_active",
    "visual_recovery_reason",
    "visual_recovery_phase",
    "turn_loop_recovery_active",
    "turn_loop_feedback",
    "active_stage_id",
    "stage_state",
    "trigger_reasons",
    "requested_evidence",
)
PROMPT_KEYFRAME_KEYS = ("step_id", "reason", "image_path")
PROMPT_MAP_CONTEXT_KEYS = (
    "mode",
    "input_regime",
    "map_frame_interval_steps",
    "map_step_id",
    "map_frame_due",
    "map_available",
    "map_source",
    "pose_source",
    "map_safety",
    "visited_trail_count",
    "collision_overlay_enabled",
    "collision_point_count",
    "map_image_label",
    "map_generation_error",
    "cached_map_available",
    "cached_map_step_id",
    "map_age_steps",
    "map_interval_due",
    "map_view_scope",
    "map_refresh_reason",
)
PROMPT_TASK_STATE_KEYS = (
    "scene_id",
    "episode_id",
    "instruction",
    "current_step_id",
    "last_action_text",
    "last_planner_reason",
    "last_ok",
    "last_error",
    "last_image_path",
)
MAX_PROMPT_RECENT_KEYFRAME_PATHS = 2
MAX_PROMPT_MEMORY_CONTEXT_CHARS = 300
MAX_PROMPT_MEMORY_IMAGES = 2
MAX_DIRECT_RETRIEVED_MEMORY_IMAGES = 3
MAX_DIRECT_RECENT_CURRENT_IMAGES = 4
MAX_PROMPT_RECENT_STEP_SUMMARY_CHARS = 500
MAX_PROMPT_RETRIEVED_MEMORY_IDS = 3
PLAN_MAX_TOTAL_TOKENS = 6000
QWEN_HARD_MAX_INPUT_TOKENS = 50000
OPENCLAW_SESSION_MODE = "fresh_per_step"
LOCAL_POLICY_FAST_MODES = {"memory_guided_policy_fast", "local_policy"}
JANUS_POLICY_BACKEND = "janus_policy"
QWEN_DIRECT_POLICY_BACKEND = "qwen_direct"
ROUTE_V2_REQUIRED_FIELDS = {
    "action_text",
    "confidence",
    "visual_summary",
    "progress_state",
    "route_stage",
    "confirmed_landmarks",
    "current_target",
    "target_relation",
    "stop_evidence",
    "semantic_stop_state",
    "reason",
}
ROUTE_V2_MAX_RESPONSE_CHARS = 2048
ROUTE_V2_STRING_LIMITS = {
    "visual_summary": 240,
    "progress_state": 120,
    "current_target": 120,
    "target_relation": 120,
    "reason": 160,
}
ROUTE_V2_MAX_LANDMARKS = 8
ROUTE_V2_MAX_LANDMARK_CHARS = 80
ROUTE_V2_FIELD_LIMITS_TEXT = (
    "Field limits: visual_summary<=240 chars; progress_state<=120 chars; "
    "current_target<=120 chars; target_relation<=120 chars; reason<=160 chars; "
    "confirmed_landmarks<=8 items and each item<=80 chars."
)
ROUTE_V3_STAGED_REQUIRED_FIELDS = ROUTE_V2_REQUIRED_FIELDS | {
    "active_stage_id",
    "stage_complete_candidate",
    "stage_relation",
    "stage_evidence_refs",
}
ROUTE_V3_STAGED_MAX_RESPONSE_CHARS = 3072
ROUTE_V3_STAGED_MAX_EVIDENCE_REFS = 6
ROUTE_V3_STAGED_MAX_EVIDENCE_REF_CHARS = 120
ROUTE_V3_STAGED_LOCAL_REASON_DEFAULT = "qwen_direct_local_schema_default"
ROUTE_V3_STAGED_RELATIONS = {
    "before",
    "at",
    "inside",
    "past",
    "outside",
    "unknown",
}
ROUTE_V3_STAGED_SCHEMA_LINE = (
    'Schema: {"action_text":"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT",'
    '"confidence":0.0,"visual_summary":"short","progress_state":"short",'
    '"route_stage":"start|en_route|intermediate_landmark|post_landmark_transition|approaching_target|verifying_target|complete|unknown",'
    '"confirmed_landmarks":["short"],"current_target":"short",'
    '"target_relation":"short","stop_evidence":"none|visible_goal|instruction_complete",'
    '"semantic_stop_state":"not_ready|visible_not_reached|approaching_target|at_or_inside_target|beside_target|instruction_complete_at_target",'
    '"reason":"short","active_stage_id":"stage_00",'
    '"stage_complete_candidate":false,'
    '"stage_relation":"before|at|inside|past|outside|unknown",'
    '"stage_evidence_refs":["current"]}'
)
ROUTE_STAGES = {
    "start",
    "en_route",
    "intermediate_landmark",
    "post_landmark_transition",
    "approaching_target",
    "verifying_target",
    "complete",
    "unknown",
}
STOP_EVIDENCE_VALUES = {"none", "visible_goal", "instruction_complete"}
SEMANTIC_STOP_STATES = {
    "not_ready",
    "visible_not_reached",
    "approaching_target",
    "at_or_inside_target",
    "beside_target",
    "instruction_complete_at_target",
}
VISUAL_IMAGE_ROLES = {
    "map_view",
    "current",
    "left_scan",
    "center_scan",
    "right_scan",
    "keyframe",
    "confirmed_landmark",
    "target_candidate",
    "stuck_before_keyframe",
}
DYNAMIC_VISUAL_ROLE_ORDERS = {
    "normal": ("map_view", "confirmed_landmark", "keyframe", "current"),
    "stuck": (
        "map_view",
        "stuck_before_keyframe",
        "left_scan",
        "center_scan",
        "right_scan",
        "current",
    ),
    "stop_blocked": (
        "map_view",
        "confirmed_landmark",
        "target_candidate",
        "current",
    ),
}
DIRECT_PROMPT_OMIT_KEYS = {
    "run_id",
    "current_image_path",
    "recent_keyframe_paths",
    "memory_images",
    "retrieved_memory_image_paths",
    "image_path",
    "map_image_path",
    "map_image_hash",
    "internal_only",
    "_map_internal_context",
    "raw_pose",
    "pose_history",
    "sim_position",
    "sim_rotation",
    "diagnostic_pose",
    "full_pose_history",
    "local_artifact_path",
    "visual_evidence_registry",
}


class QwenApiRequestError(RuntimeError):
    def __init__(self, message: str, audit_metadata: Dict[str, Any]) -> None:
        super().__init__(message)
        self.audit_metadata = dict(audit_metadata)


class QwenApiModelClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        run_openclaw: Optional[OpenClawRunner] = None,
        openclaw_profile: str = "",
        max_retries: Optional[int] = None,
        retry_backoff_s: Optional[float] = None,
        thinking_mode: str = "auto",
        thinking_budget: Optional[int] = None,
        transport_mode: str = "sync",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.run_openclaw = run_openclaw or run_openclaw_command
        self.openclaw_profile = openclaw_profile
        self.session = requests.Session()
        self._loaded_openclaw_config = False
        self.max_retries = self._retry_int(
            max_retries,
            "OPENCLAW_QWEN_API_RETRIES",
            0,
        )
        self.retry_backoff_s = self._retry_float(
            retry_backoff_s,
            "OPENCLAW_QWEN_API_RETRY_BACKOFF_S",
            1.0,
        )
        if thinking_mode not in {"auto", "off", "on"}:
            raise ValueError("thinking_mode must be one of auto, off, on")
        if thinking_budget is not None and thinking_budget <= 0:
            raise ValueError("thinking_budget must be a positive integer")
        if transport_mode != "sync":
            raise ValueError("only synchronous Qwen transport is currently supported")
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget
        self.transport_mode = transport_mode

    def run(
        self,
        prompt: str,
        image_paths: List[str],
        model: str,
        timeout_s: float,
        image_labels: Optional[List[str]] = None,
        thinking_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_config(timeout_s)
        if not self.api_key:
            raise RuntimeError(
                "Qwen API key is not configured; set OPENCLAW_QWEN_API_KEY, "
                "DASHSCOPE_API_KEY, QWEN_API_KEY, or configure OpenClaw provider qwen"
            )
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        base_request_json = {
            "model": self._api_model_name(model),
            "temperature": 0,
        }
        request_image_paths = list(image_paths)
        request_image_labels = (
            list(image_labels or [])
            if len(image_labels or []) == len(request_image_paths)
            else []
        )
        original_image_count = len(request_image_paths)
        effective_thinking_mode = thinking_mode or self.thinking_mode
        if effective_thinking_mode not in {"auto", "off", "on"}:
            raise ValueError("thinking_mode must be one of auto, off, on")
        thinking_enabled: Optional[bool] = None
        if effective_thinking_mode == "on":
            thinking_enabled = True
        elif effective_thinking_mode == "off":
            thinking_enabled = False
        request_attempts = 0
        retry_count = 0
        capability_fallback = False
        degradation_attempted = False
        degradation_mode: Optional[str] = None
        degradation_trigger_detail: Optional[str] = None
        last_error_detail: Optional[str] = None
        last_error_fingerprint: Optional[str] = None
        request_started_at = time.monotonic()
        while True:
            request_json = dict(base_request_json)
            request_json["messages"] = [
                {
                    "role": "user",
                    "content": self._message_content(
                        prompt,
                        request_image_paths,
                        request_image_labels,
                    ),
                }
            ]
            if thinking_enabled is not None:
                request_json["enable_thinking"] = thinking_enabled
                if thinking_enabled and self.thinking_budget is not None:
                    request_json["thinking_budget"] = self.thinking_budget
            request_attempts += 1
            try:
                response = self.session.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_json,
                    timeout=timeout_s,
                )
            except requests.exceptions.RequestException as exc:
                if retry_count >= self.max_retries:
                    metadata = self._failure_metadata(
                        status_code=None,
                        error_code="network_error",
                        category="network_error",
                        retryable=True,
                        request_attempts=request_attempts,
                        retry_count=retry_count,
                        capability_fallback=capability_fallback,
                        error_detail="network_error",
                        degradation_attempted=degradation_attempted,
                        degradation_mode=degradation_mode,
                        degradation_trigger_detail=degradation_trigger_detail,
                        original_image_count=original_image_count,
                        final_image_count=len(request_image_paths),
                        thinking_enabled=thinking_enabled,
                    )
                    raise QwenApiRequestError(
                        "Qwen API request failed: category=network_error",
                        metadata,
                    ) from exc
                self._sleep_before_retry(retry_count)
                retry_count += 1
                continue
            if response.status_code >= 400:
                error_code, error_message = self._provider_error(response)
                category = self._provider_error_category(
                    response.status_code,
                    error_code,
                    error_message,
                    thinking_enabled,
                )
                error_detail = self._provider_error_detail(error_code, error_message)
                error_fingerprint = self._provider_error_fingerprint(
                    error_code,
                    error_message,
                )
                last_error_detail = error_detail
                last_error_fingerprint = error_fingerprint
                if category == "thinking_transport_incompatible":
                    metadata = self._failure_metadata(
                        response.status_code,
                        error_code,
                        category,
                        False,
                        request_attempts,
                        retry_count,
                        capability_fallback,
                        error_detail=error_detail,
                        error_fingerprint=error_fingerprint,
                        degradation_attempted=degradation_attempted,
                        degradation_mode=degradation_mode,
                        degradation_trigger_detail=degradation_trigger_detail,
                        original_image_count=original_image_count,
                        final_image_count=len(request_image_paths),
                        thinking_enabled=thinking_enabled,
                    )
                    raise QwenApiRequestError(
                        self._sanitized_error_message(
                            response.status_code,
                            error_code,
                            category,
                            error_detail,
                        ),
                        metadata,
                    )
                if not degradation_attempted:
                    candidate_mode = self._provider_degradation_mode(
                        category,
                        error_detail,
                        thinking_enabled,
                        request_image_paths,
                    )
                    if candidate_mode == "thinking_disabled":
                        degradation_attempted = True
                        degradation_mode = candidate_mode
                        degradation_trigger_detail = error_detail
                        capability_fallback = category == "thinking_unsupported"
                        thinking_enabled = False
                        continue
                    if candidate_mode == "map_current_only":
                        reduced_paths, reduced_labels = self._reduced_visual_inputs(
                            request_image_paths,
                            request_image_labels,
                        )
                        if len(reduced_paths) < len(request_image_paths):
                            degradation_attempted = True
                            degradation_mode = candidate_mode
                            degradation_trigger_detail = error_detail
                            request_image_paths = reduced_paths
                            request_image_labels = reduced_labels
                            continue
                retryable = self._retryable_status(response.status_code)
                if retryable and retry_count < self.max_retries:
                    self._sleep_before_retry(retry_count)
                    retry_count += 1
                    continue
                metadata = self._failure_metadata(
                    response.status_code,
                    error_code,
                    category,
                    retryable,
                    request_attempts,
                    retry_count,
                    capability_fallback,
                    error_detail=error_detail,
                    error_fingerprint=error_fingerprint,
                    degradation_attempted=degradation_attempted,
                    degradation_mode=degradation_mode,
                    degradation_trigger_detail=degradation_trigger_detail,
                    original_image_count=original_image_count,
                    final_image_count=len(request_image_paths),
                    thinking_enabled=thinking_enabled,
                )
                raise QwenApiRequestError(
                    self._sanitized_error_message(
                        response.status_code,
                        error_code,
                        category,
                        error_detail,
                    ),
                    metadata,
                )
            try:
                response_data = response.json()
            except (ValueError, TypeError) as exc:
                metadata = self._failure_metadata(
                    response.status_code,
                    "invalid_json",
                    "invalid_response",
                    False,
                    request_attempts,
                    retry_count,
                    capability_fallback,
                    error_detail="invalid_response_json",
                    degradation_attempted=degradation_attempted,
                    degradation_mode=degradation_mode,
                    degradation_trigger_detail=degradation_trigger_detail,
                    original_image_count=original_image_count,
                    final_image_count=len(request_image_paths),
                    thinking_enabled=thinking_enabled,
                )
                raise QwenApiRequestError(
                    self._sanitized_error_message(
                        response.status_code,
                        "invalid_json",
                        "invalid_response",
                    ),
                    metadata,
                ) from exc
            normalized = self._normalize_response(
                response_data,
                thinking_enabled=thinking_enabled,
                capability_fallback=capability_fallback,
                degradation_attempted=degradation_attempted,
                degradation_mode=degradation_mode,
                degradation_trigger_detail=degradation_trigger_detail,
                error_detail=last_error_detail,
                error_fingerprint=last_error_fingerprint,
                original_image_count=original_image_count,
                final_image_count=len(request_image_paths),
            )
            normalized["request_attempts"] = request_attempts
            normalized["retry_count"] = retry_count
            provider_metadata = normalized.get("provider_metadata")
            if isinstance(provider_metadata, dict):
                provider_metadata["provider_latency_ms"] = round(
                    (time.monotonic() - request_started_at) * 1000.0,
                    3,
                )
            return normalized

    @staticmethod
    def _provider_error(response: Any) -> tuple[str, str]:
        try:
            data = response.json()
        except (ValueError, TypeError):
            return "unknown", ""
        if not isinstance(data, dict):
            return "unknown", ""
        error = data.get("error")
        if not isinstance(error, dict):
            error = data
        raw_code = str(error.get("code") or "unknown")
        code = (
            raw_code if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", raw_code) else "unknown"
        )
        return code, str(error.get("message") or "")

    @staticmethod
    def _provider_error_category(
        status_code: int,
        error_code: str,
        error_message: str,
        thinking_enabled: Optional[bool],
    ) -> str:
        transient = f"{error_code} {error_message}".lower()
        if thinking_enabled is True and (
            "only support stream" in transient
            or "stream call" in transient
            or "streaming required" in transient
        ):
            return "thinking_transport_incompatible"
        if (
            thinking_enabled is True
            and "thinking" in transient
            and (
                "not supported" in transient
                or "unsupported" in transient
                or "unknown parameter" in transient
                or "unrecognized" in transient
            )
        ):
            return "thinking_unsupported"
        if status_code == 429:
            return "rate_limited"
        if status_code >= 500:
            return "server_error"
        return "client_error"

    @staticmethod
    def _provider_error_detail(error_code: str, error_message: str) -> str:
        error_text = f"{error_code} {error_message}".lower()
        if "thinking_budget" in error_text or "thinking budget" in error_text:
            return "thinking_budget_invalid"
        if "thinking" in error_text and (
            "not supported" in error_text
            or "unsupported" in error_text
            or "unknown parameter" in error_text
            or "unrecognized" in error_text
            or "invalid" in error_text
        ):
            return "thinking_parameter_invalid"
        if (
            "context length" in error_text
            or "input length" in error_text
            or "input tokens" in error_text
            or "maximum token" in error_text
            or "token limit" in error_text
            or "too many tokens" in error_text
        ):
            return "input_length_invalid"
        if "image" in error_text and (
            "invalid" in error_text
            or "format" in error_text
            or "size" in error_text
            or "resolution" in error_text
            or "too many" in error_text
            or "unsupported" in error_text
            or "image_url" in error_text
            or "image url" in error_text
        ):
            return "image_input_invalid"
        if "model" in error_text and (
            "not found" in error_text
            or "unsupported" in error_text
            or "invalid" in error_text
        ):
            return "model_parameter_invalid"
        if "invalidparameter" in error_text or "invalid_parameter" in error_text:
            return "request_parameter_invalid"
        return "unclassified"

    @staticmethod
    def _provider_error_fingerprint(error_code: str, error_message: str) -> str:
        normalized = " ".join(f"{error_code} {error_message}".lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _provider_degradation_mode(
        category: str,
        error_detail: str,
        thinking_enabled: Optional[bool],
        image_paths: List[str],
    ) -> Optional[str]:
        if thinking_enabled is True and (
            category == "thinking_unsupported"
            or error_detail in {"thinking_budget_invalid", "thinking_parameter_invalid"}
        ):
            return "thinking_disabled"
        if (
            error_detail in {"image_input_invalid", "input_length_invalid"}
            and len(image_paths) > 2
        ):
            return "map_current_only"
        return None

    @staticmethod
    def _reduced_visual_inputs(
        image_paths: List[str],
        image_labels: List[str],
    ) -> Tuple[List[str], List[str]]:
        if len(image_paths) <= 2:
            return list(image_paths), list(image_labels)
        if len(image_labels) == len(image_paths):
            map_index = next(
                (
                    index
                    for index, label in enumerate(image_labels)
                    if label == "map_view"
                ),
                0,
            )
            current_index = next(
                (
                    index
                    for index in range(len(image_labels) - 1, -1, -1)
                    if image_labels[index] == "current"
                ),
                len(image_paths) - 1,
            )
            selected_indices = [map_index]
            if current_index != map_index:
                selected_indices.append(current_index)
            return (
                [image_paths[index] for index in selected_indices],
                [image_labels[index] for index in selected_indices],
            )
        return [image_paths[0], image_paths[-1]], []

    @staticmethod
    def _sanitized_error_message(
        status_code: int,
        error_code: str,
        category: str,
        error_detail: Optional[str] = None,
    ) -> str:
        detail = f" detail={error_detail}" if error_detail else ""
        return (
            f"Qwen API request failed: HTTP {status_code} code={error_code} "
            f"category={category}{detail}"
        )

    def _failure_metadata(
        self,
        status_code: Optional[int],
        error_code: str,
        category: str,
        retryable: bool,
        request_attempts: int,
        retry_count: int,
        capability_fallback: bool,
        error_detail: Optional[str] = None,
        error_fingerprint: Optional[str] = None,
        degradation_attempted: bool = False,
        degradation_mode: Optional[str] = None,
        degradation_trigger_detail: Optional[str] = None,
        original_image_count: int = 0,
        final_image_count: int = 0,
        thinking_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return {
            "provider_error_status": status_code,
            "provider_error_code": error_code,
            "provider_error_category": category,
            "provider_error_retryable": retryable,
            "provider_error_detail": error_detail,
            "provider_error_fingerprint": error_fingerprint,
            "provider_degradation_attempted": degradation_attempted,
            "provider_degradation_mode": degradation_mode,
            "provider_degradation_trigger_detail": degradation_trigger_detail,
            "provider_original_image_count": original_image_count,
            "provider_final_image_count": final_image_count,
            "qwen_api_request_attempts": request_attempts,
            "qwen_api_retry_count": retry_count,
            "qwen_thinking_enabled": thinking_enabled,
            "qwen_thinking_exercised": None,
            "thinking_model_supported": False
            if category == "thinking_unsupported"
            else None,
            "thinking_capability_fallback": capability_fallback,
            "thinking_transport_incompatible": category
            == "thinking_transport_incompatible",
        }

    def _sleep_before_retry(self, attempt_index: int) -> None:
        if self.retry_backoff_s <= 0:
            return
        time.sleep(self.retry_backoff_s * (attempt_index + 1))

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        return status_code == 429 or status_code >= 500

    @staticmethod
    def _retry_int(value: Optional[int], env_name: str, default: int) -> int:
        raw = os.environ.get(env_name) if value is None else value
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _retry_float(value: Optional[float], env_name: str, default: float) -> float:
        raw = os.environ.get(env_name) if value is None else value
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return default

    def _ensure_config(self, timeout_s: float) -> None:
        if not self.base_url:
            self.base_url = os.environ.get(
                "OPENCLAW_QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        if not self.api_key:
            self.api_key = (
                os.environ.get("OPENCLAW_QWEN_API_KEY")
                or os.environ.get("DASHSCOPE_API_KEY")
                or os.environ.get("QWEN_API_KEY")
                or ""
            )
        if self.api_key or self._loaded_openclaw_config:
            return
        self.api_key = self._openclaw_auth_store_api_key()
        if self.api_key:
            return
        self._loaded_openclaw_config = True
        provider = self._openclaw_qwen_provider(timeout_s)
        if not isinstance(provider, dict):
            return
        self.base_url = str(
            provider.get("baseUrl") or provider.get("base_url") or self.base_url
        )
        api_key = (
            provider.get("apiKey")
            or provider.get("api_key")
            or provider.get("key")
            or ""
        )
        self.api_key = (
            self._resolve_secret_value(str(api_key)) if api_key else self.api_key
        )
        if not self.api_key:
            self.api_key = self._openclaw_auth_store_api_key()

    @staticmethod
    def _openclaw_auth_store_api_key() -> str:
        store_path = Path(
            os.environ.get(
                "OPENCLAW_AUTH_PROFILES_PATH",
                str(
                    Path.home()
                    / ".openclaw"
                    / "agents"
                    / "main"
                    / "agent"
                    / "auth-profiles.json"
                ),
            )
        )
        if not store_path.exists():
            return ""
        try:
            data = json.loads(store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        profiles = data.get("profiles") if isinstance(data, dict) else None
        if not isinstance(profiles, dict):
            return ""
        preferred = profiles.get("qwen:default")
        key = QwenApiModelClient._api_key_from_auth_profile(preferred)
        if key:
            return key
        for profile in profiles.values():
            key = QwenApiModelClient._api_key_from_auth_profile(profile)
            if key:
                return key
        return ""

    @staticmethod
    def _api_key_from_auth_profile(profile: Any) -> str:
        if not isinstance(profile, dict):
            return ""
        if profile.get("provider") not in (None, "qwen"):
            return ""
        key = (
            profile.get("key") or profile.get("apiKey") or profile.get("api_key") or ""
        )
        return str(key).strip()

    def _openclaw_qwen_provider(self, timeout_s: float) -> Dict[str, Any]:
        command = ["openclaw"]
        if self.openclaw_profile:
            command.extend(["--profile", self.openclaw_profile])
        command.extend(["config", "get", "models.providers.qwen", "--json"])
        result = self.run_openclaw(command, min(timeout_s, 10.0))
        if result.returncode != 0:
            return {}
        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict) and isinstance(data.get("value"), dict):
            return data["value"]
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _resolve_secret_value(value: str) -> str:
        stripped = value.strip()
        if stripped.startswith("${") and stripped.endswith("}"):
            return os.environ.get(stripped[2:-1], "")
        if stripped.startswith("$") and len(stripped) > 1:
            return os.environ.get(stripped[1:], "")
        return stripped

    @staticmethod
    def _api_model_name(model: str) -> str:
        if model.startswith("qwen/"):
            return model.split("/", 1)[1]
        return model

    def _message_content(
        self,
        prompt: str,
        image_paths: List[str],
        image_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        labels = image_labels if len(image_labels or []) == len(image_paths) else []
        for index, image_path in enumerate(image_paths):
            if labels:
                label = str(labels[index])
                if label not in VISUAL_IMAGE_ROLES:
                    raise ValueError(f"unsupported provider image label: {label}")
                content.append({"type": "text", "text": f"Image label: {label}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image_path)},
                }
            )
        content.append({"type": "text", "text": prompt})
        return content

    @staticmethod
    def _image_data_url(image_path: str) -> str:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _normalize_response(
        data: Dict[str, Any],
        thinking_enabled: Optional[bool] = None,
        capability_fallback: bool = False,
        degradation_attempted: bool = False,
        degradation_mode: Optional[str] = None,
        degradation_trigger_detail: Optional[str] = None,
        error_detail: Optional[str] = None,
        error_fingerprint: Optional[str] = None,
        original_image_count: int = 0,
        final_image_count: int = 0,
    ) -> Dict[str, Any]:
        text = ""
        reasoning_content_present = False
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            message = (
                choices[0].get("message") if isinstance(choices[0], dict) else None
            )
            if isinstance(message, dict):
                reasoning = message.get("reasoning_content")
                reasoning_content_present = isinstance(reasoning, str) and bool(
                    reasoning.strip()
                )
                content = message.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "\n".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and isinstance(item.get("text"), str)
                    )
        usage = data.get("usage") if isinstance(data, dict) else {}
        normalized_usage: Dict[str, Any] = {}
        reasoning_tokens: Optional[int] = None
        if isinstance(usage, dict):
            normalized_usage = {
                "input": usage.get("prompt_tokens")
                or usage.get("input")
                or usage.get("input_tokens"),
                "output": (
                    usage.get("completion_tokens")
                    or usage.get("output")
                    or usage.get("output_tokens")
                ),
                "totalTokens": (
                    usage.get("total_tokens")
                    or usage.get("totalTokens")
                    or usage.get("total_tokens")
                ),
            }
            details = usage.get("completion_tokens_details")
            if (
                isinstance(details, dict)
                and details.get("reasoning_tokens") is not None
            ):
                try:
                    reasoning_tokens = max(0, int(details.get("reasoning_tokens")))
                except (TypeError, ValueError):
                    reasoning_tokens = None
            normalized_usage = {
                key: value
                for key, value in normalized_usage.items()
                if value is not None
            }
        thinking_exercised: Optional[bool] = None
        if reasoning_content_present or (
            reasoning_tokens is not None and reasoning_tokens > 0
        ):
            thinking_exercised = True
        elif reasoning_tokens == 0:
            thinking_exercised = False
        returned_model_id = (
            str(data.get("model") or "") if isinstance(data, dict) else ""
        )
        return {
            "ok": True,
            "outputs": [{"text": text}],
            "usage": normalized_usage,
            "provider_metadata": {
                "returned_model_id": returned_model_id or None,
                "qwen_thinking_enabled": thinking_enabled,
                "qwen_thinking_exercised": thinking_exercised,
                "thinking_model_supported": (
                    False
                    if capability_fallback
                    else True
                    if thinking_enabled is True
                    else None
                ),
                "thinking_capability_fallback": capability_fallback,
                "thinking_transport_incompatible": False,
                "reasoning_tokens": reasoning_tokens,
                "provider_error_detail": error_detail,
                "provider_error_fingerprint": error_fingerprint,
                "provider_degradation_attempted": degradation_attempted,
                "provider_degradation_mode": degradation_mode,
                "provider_degradation_trigger_detail": degradation_trigger_detail,
                "provider_original_image_count": original_image_count,
                "provider_final_image_count": final_image_count,
            },
        }


def run_openclaw_command(
    args: List[str], timeout_s: float
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=timeout_s,
    )


class OpenClawCliPlanPlanner:
    def __init__(
        self,
        recall_interval_steps: int = 5,
        run_openclaw: OpenClawRunner = run_openclaw_command,
        timeout_s: float = 5.0,
        gateway_url: str = "",
        planner_mode: str = "heuristic",
        agent_id: str = "main",
        agent_timeout_s: float = 60.0,
        openclaw_profile: str = "",
        openclaw_model: str = "",
        openclaw_model_provider: str = "qwen_api",
        openclaw_model_max_images: int = 3,
        openclaw_model_image_interval_steps: int = 20,
        openclaw_model_fast_mode: str = "qwen_text_only",
        openclaw_model_fast_use_memory_context: bool = True,
        model_client: Any = None,
        agent_session_id: str = "",
        openclaw_visual_mode: str = "path",
        openclaw_visual_max_images: int = 2,
        openclaw_visual_interval_steps: int = 1,
        openclaw_visual_timeout_ms: int = 30000,
        openclaw_visual_model: str = "",
        visual_analyzer: Any = None,
        agent_session_timestamp: str = "",
        agent_max_input_tokens: int = 10000,
        openclaw_session_dir: str = "",
        policy_backend: str = JANUS_POLICY_BACKEND,
        dynamic_visual_context_enabled: bool = False,
        qwen_thinking_mode: str = "auto",
        qwen_thinking_interval_steps: int = 1,
        qwen_output_schema: str = "legacy",
        qwen_thinking_budget: Optional[int] = None,
        qwen_transport_mode: str = "sync",
        segmentation_timeout_s: float = 120.0,
        stage_plan_manifest_path: str = "",
    ) -> None:
        self.recall_interval_steps = max(1, recall_interval_steps)
        self.run_openclaw = run_openclaw
        self.timeout_s = timeout_s
        self.gateway_url = gateway_url
        self.planner_mode = planner_mode
        self.agent_id = agent_id
        self.agent_timeout_s = agent_timeout_s
        self.openclaw_profile = openclaw_profile
        self.openclaw_model = openclaw_model
        self.openclaw_model_provider = openclaw_model_provider or "qwen_api"
        self.openclaw_model_max_images = max(0, int(openclaw_model_max_images or 0))
        self.openclaw_model_image_interval_steps = max(
            1,
            int(openclaw_model_image_interval_steps or 1),
        )
        self.openclaw_model_fast_mode = openclaw_model_fast_mode or "qwen_text_only"
        self.openclaw_model_fast_use_memory_context = bool(
            openclaw_model_fast_use_memory_context
        )
        self.model_client = model_client
        self._visual_memory_cache: Dict[str, Dict[str, Any]] = {}
        self.agent_session_id = agent_session_id or "clawnav"
        self.agent_session_timestamp = (
            self._safe_session_part(agent_session_timestamp)
            if agent_session_timestamp
            else datetime.now().strftime("%m%d%H%M")
        )
        self.agent_max_input_tokens = max(0, int(agent_max_input_tokens or 0))
        self.openclaw_session_dir = openclaw_session_dir
        self.policy_backend = policy_backend or JANUS_POLICY_BACKEND
        if qwen_thinking_mode not in {"auto", "off", "on"}:
            raise ValueError("qwen_thinking_mode must be one of auto, off, on")
        if int(qwen_thinking_interval_steps) <= 0:
            raise ValueError("qwen_thinking_interval_steps must be a positive integer")
        if qwen_output_schema not in {"legacy", "route_v2", "route_v3_staged"}:
            raise ValueError(
                "qwen_output_schema must be one of legacy, route_v2, route_v3_staged"
            )
        if qwen_thinking_budget is not None and qwen_thinking_budget <= 0:
            raise ValueError("qwen_thinking_budget must be a positive integer")
        if qwen_transport_mode != "sync":
            raise ValueError("only synchronous Qwen transport is currently supported")
        if openclaw_model_provider == "openclaw_cli" and qwen_thinking_mode != "auto":
            raise ValueError(
                "explicit Qwen thinking control is only supported by qwen_api"
            )
        if (
            openclaw_model_provider == "openclaw_cli"
            and int(qwen_thinking_interval_steps) != 1
        ):
            raise ValueError("periodic Qwen thinking is only supported by qwen_api")
        self.dynamic_visual_context_enabled = bool(dynamic_visual_context_enabled)
        self.qwen_thinking_mode = qwen_thinking_mode
        self.qwen_thinking_interval_steps = int(qwen_thinking_interval_steps)
        self.qwen_output_schema = qwen_output_schema
        self.qwen_thinking_budget = qwen_thinking_budget
        self.qwen_transport_mode = qwen_transport_mode
        if float(segmentation_timeout_s) <= 0:
            raise ValueError("segmentation_timeout_s must be positive")
        self.segmentation_timeout_s = float(segmentation_timeout_s)
        self.stage_plan_manifest_path = str(stage_plan_manifest_path or "")
        self._stage_plan_manifest_rows: Optional[Dict[str, Dict[str, Any]]] = None
        self._agent_token_guard: Dict[str, Any] = {}
        self.openclaw_visual_mode = openclaw_visual_mode
        self.openclaw_visual_max_images = max(0, openclaw_visual_max_images)
        self.openclaw_visual_interval_steps = max(1, openclaw_visual_interval_steps)
        self.visual_analyzer = visual_analyzer
        if self.visual_analyzer is None and self.openclaw_visual_mode == "describe":
            self.visual_analyzer = OpenClawVisualAnalyzer(
                run_openclaw=run_openclaw,
                model=openclaw_visual_model,
                timeout_ms=openclaw_visual_timeout_ms,
            )

    def health_payload(self) -> Dict[str, Any]:
        health = self._gateway_health()
        return {
            "ok": True,
            "service": "openclaw_cli_plan_gateway",
            "planner_mode": self.planner_mode,
            "agent_id": self.agent_id,
            "agent_max_input_tokens": self.agent_max_input_tokens,
            "agent_token_guard": self._agent_token_guard,
            "timeout_budget": self._timeout_budget_payload(),
            "openclaw_gateway": health,
            "instruction_segmentation": True,
            "stage_schema_versions": [INSTRUCTION_STAGE_SCHEMA_VERSION],
            "action_schema_versions": ["legacy", "route_v2", "route_v3_staged"],
            "active_stage_schema": INSTRUCTION_STAGE_SCHEMA_VERSION,
            "active_action_schema": self.qwen_output_schema,
        }

    def segment_instruction_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) != {
            "scene_id",
            "episode_id",
            "instruction",
        }:
            raise GatewayRequestError("invalid_request_fields")
        instruction = payload.get("instruction")
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or len(instruction.strip()) > 4096
        ):
            raise GatewayRequestError("invalid_instruction")
        instruction = instruction.strip()
        scene_id = str(payload.get("scene_id") or "")
        episode_id = str(payload.get("episode_id") or "")
        if self.stage_plan_manifest_path:
            return self._frozen_stage_plan_response(
                scene_id,
                episode_id,
                instruction,
            )

        metadata: Dict[str, Any] = {
            "segmentation_source": "qwen",
            "fallback_category": StageFallbackCategory.NONE.value,
            "repair_attempted": False,
            "repair_succeeded": False,
        }
        prompt = self._instruction_segmentation_prompt(instruction)
        try:
            stdout = self._model_run_stdout(
                prompt,
                {"paths": [], "provider_image_labels": []},
                timeout_s=self.segmentation_timeout_s,
            )
        except Exception:
            plan = single_stage_fallback(
                instruction,
                StageFallbackCategory.PROVIDER_FAILURE,
            )
            metadata.update(
                {
                    "segmentation_source": "fallback",
                    "fallback_category": StageFallbackCategory.PROVIDER_FAILURE.value,
                }
            )
        else:
            plan = self._parse_or_repair_instruction_stages(
                stdout,
                instruction,
                metadata,
            )
        metadata.update(
            {
                "instruction_sha256": plan.instruction_sha256,
                "stage_plan_sha256": plan.stage_plan_sha256,
            }
        )
        return {
            "stage_plan": self._provider_stage_plan_payload(plan),
            "runtime_metadata": metadata,
        }

    def _parse_or_repair_instruction_stages(
        self,
        stdout: str,
        instruction: str,
        metadata: Dict[str, Any],
    ):
        try:
            candidate = self._extract_instruction_stage_json(
                self._model_visible_text(stdout)
            )
            plan = parse_instruction_stage_plan(candidate, instruction)
            if plan.fallback_category is not StageFallbackCategory.NONE:
                raise RuntimeError("instruction stage schema validation failed")
            return plan
        except Exception:
            metadata["repair_attempted"] = True
            try:
                repair_stdout = self._model_run_stdout(
                    self._instruction_segmentation_repair_prompt(instruction),
                    {"paths": [], "provider_image_labels": []},
                    timeout_s=self.segmentation_timeout_s,
                )
                repair_candidate = self._extract_instruction_stage_json(
                    self._model_visible_text(repair_stdout)
                )
                plan = parse_instruction_stage_plan(repair_candidate, instruction)
                if plan.fallback_category is not StageFallbackCategory.NONE:
                    raise RuntimeError("instruction stage repair validation failed")
                metadata["repair_succeeded"] = True
            except Exception:
                plan = single_stage_fallback(
                    instruction,
                    StageFallbackCategory.SCHEMA_ERROR,
                )
                metadata.update(
                    {
                        "segmentation_source": "fallback",
                        "fallback_category": StageFallbackCategory.SCHEMA_ERROR.value,
                        "repair_succeeded": False,
                    }
                )
            return plan

    def _frozen_stage_plan_response(
        self,
        scene_id: str,
        episode_id: str,
        instruction: str,
    ) -> Dict[str, Any]:
        rows = self._load_stage_plan_manifest_rows()
        episode_key = f"{scene_id}:{episode_id}"
        row = rows.get(episode_key)
        if row is None:
            raise RuntimeError(f"frozen stage manifest missing episode {episode_key}")
        stages = row.get("stages")
        if not isinstance(stages, list):
            raise RuntimeError("frozen stage manifest stages are invalid")
        provider_stages = []
        for stage in stages:
            if not isinstance(stage, dict):
                raise RuntimeError("frozen stage manifest stage is invalid")
            provider_stages.append(
                {key: value for key, value in stage.items() if key != "stage_id"}
            )
        stage_payload = {
            "schema_version": INSTRUCTION_STAGE_SCHEMA_VERSION,
            "stages": provider_stages,
        }
        plan = parse_instruction_stage_plan(stage_payload, instruction)
        if plan.fallback_category is not StageFallbackCategory.NONE:
            raise RuntimeError("frozen stage manifest plan is invalid")
        if row.get("generation_status") != "ok":
            raise RuntimeError("frozen stage manifest generation status is invalid")
        if row.get("instruction_sha256") != plan.instruction_sha256:
            raise RuntimeError("frozen stage manifest instruction hash mismatch")
        if row.get("stage_plan_sha256") != plan.stage_plan_sha256:
            raise RuntimeError("frozen stage manifest stage hash mismatch")
        return {
            "stage_plan": stage_payload,
            "runtime_metadata": {
                "segmentation_source": "frozen_manifest",
                "fallback_category": StageFallbackCategory.NONE.value,
                "repair_attempted": False,
                "repair_succeeded": False,
                "instruction_sha256": plan.instruction_sha256,
                "stage_plan_sha256": plan.stage_plan_sha256,
                "model_prompt_schema_fingerprint": str(
                    row.get("model_prompt_schema_fingerprint") or ""
                ),
            },
        }

    def _load_stage_plan_manifest_rows(self) -> Dict[str, Dict[str, Any]]:
        if self._stage_plan_manifest_rows is not None:
            return self._stage_plan_manifest_rows
        try:
            raw = json.loads(
                Path(self.stage_plan_manifest_path).read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RuntimeError("frozen stage manifest cannot be loaded") from exc
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != "staged_stage_plan_manifest_v1"
            or not isinstance(raw.get("rows"), list)
        ):
            raise RuntimeError("frozen stage manifest schema is invalid")
        rows: Dict[str, Dict[str, Any]] = {}
        for row in raw["rows"]:
            if not isinstance(row, dict):
                raise RuntimeError("frozen stage manifest row is invalid")
            key = str(row.get("episode_key") or "")
            if not key or key in rows:
                raise RuntimeError("frozen stage manifest episode keys are invalid")
            rows[key] = row
        self._stage_plan_manifest_rows = rows
        return rows

    @staticmethod
    def _instruction_segmentation_prompt(instruction: str) -> str:
        return "\n".join(
            [
                "Segment the immutable navigation instruction into ordered route stages.",
                "Return exactly one JSON object and no other text.",
                'Top-level JSON shape: {"schema_version":"instruction_stages_v1",'
                '"stages":[...]}.',
                "Do not emit markdown, chain-of-thought, reasoning_content, images, metrics, target coordinates, or reference paths.",
                "Use 1-12 contiguous zero-based stages. Only the last stage has final_stage=true.",
                "Each stage has exactly: order, route_clause, transition_type, expected_landmarks, completion_cues, final_stage.",
                "transition_type must be turn, approach, pass, enter, exit, traverse, or final_arrival.",
                "Instruction:",
                json.dumps(instruction, ensure_ascii=True),
            ]
        )

    @classmethod
    def _instruction_segmentation_repair_prompt(cls, instruction: str) -> str:
        return "\n".join(
            [
                "The previous instruction stage response was invalid.",
                "Repair it by returning only one strict instruction_stages_v1 JSON object.",
                cls._instruction_segmentation_prompt(instruction),
            ]
        )

    @staticmethod
    def _extract_instruction_stage_json(text: str) -> Dict[str, Any]:
        if len(text) > 8192:
            raise RuntimeError("instruction stage response is too long")
        try:
            candidate = json.loads(text.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError("instruction stage response is invalid JSON") from exc
        if not isinstance(candidate, dict):
            raise RuntimeError("instruction stage response must be an object")
        return candidate

    @staticmethod
    def _provider_stage_plan_payload(plan) -> Dict[str, Any]:
        stages = []
        for stage in plan.stages:
            serialized = stage.to_dict()
            serialized.pop("stage_id", None)
            stages.append(serialized)
        return {
            "schema_version": INSTRUCTION_STAGE_SCHEMA_VERSION,
            "stages": stages,
        }

    def _timeout_budget_payload(self) -> Dict[str, Any]:
        uses_direct_qwen = (
            self.planner_mode == "model" and self.openclaw_model_provider == "qwen_api"
        )
        qwen_retries = self._qwen_api_retries() if uses_direct_qwen else 0
        qwen_retry_backoff_s = (
            self._qwen_api_retry_backoff_s() if uses_direct_qwen else 0.0
        )
        qwen_attempts = qwen_retries + 1
        retry_backoff_total_s = qwen_retry_backoff_s * (
            qwen_retries * (qwen_retries + 1) / 2
        )
        estimated_qwen_wall_timeout_s = (
            self.agent_timeout_s * qwen_attempts + retry_backoff_total_s
        )
        recommended_gateway_timeout_s = int(
            max(240.0, estimated_qwen_wall_timeout_s + 60.0) + 0.999
        )
        return {
            "planner_mode": self.planner_mode,
            "model_provider": self.openclaw_model_provider,
            "agent_timeout_s": self.agent_timeout_s,
            "qwen_api_retries": qwen_retries,
            "qwen_api_retry_backoff_s": qwen_retry_backoff_s,
            "estimated_qwen_wall_timeout_s": int(estimated_qwen_wall_timeout_s)
            if estimated_qwen_wall_timeout_s.is_integer()
            else estimated_qwen_wall_timeout_s,
            "recommended_gateway_timeout_s": recommended_gateway_timeout_s,
        }

    def _qwen_api_retries(self) -> int:
        if isinstance(self.model_client, QwenApiModelClient):
            return self.model_client.max_retries
        raw = os.environ.get("OPENCLAW_QWEN_API_RETRIES")
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def _qwen_api_retry_backoff_s(self) -> float:
        if isinstance(self.model_client, QwenApiModelClient):
            return self.model_client.retry_backoff_s
        raw = os.environ.get("OPENCLAW_QWEN_API_RETRY_BACKOFF_S")
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 1.0

    def plan_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.planner_mode == "model":
            try:
                return self._model_plan(payload)
            except Exception as exc:
                if self._is_qwen_direct_policy():
                    return self._direct_policy_failure_decision(
                        str(exc),
                        reason="qwen_direct_model_exception",
                    )
                fallback = self._heuristic_plan(payload)
                fallback["reason"] = f"openclaw_cli_model_fallback:{fallback['reason']}"
                fallback["arguments"] = dict(fallback.get("arguments") or {})
                fallback["arguments"]["planner_error"] = str(exc)
                return fallback

        if self.planner_mode == "agent":
            try:
                return self._agent_plan(payload)
            except Exception as exc:
                if self._is_qwen_direct_policy():
                    return self._direct_policy_failure_decision(
                        str(exc),
                        reason="qwen_direct_agent_exception",
                    )
                fallback = self._heuristic_plan(payload)
                fallback["reason"] = f"openclaw_cli_agent_fallback:{fallback['reason']}"
                fallback["arguments"] = dict(fallback.get("arguments") or {})
                fallback["arguments"]["planner_error"] = str(exc)
                return fallback

        self._gateway_health()
        return self._heuristic_plan(payload)

    def _is_qwen_direct_policy(self) -> bool:
        return self.policy_backend == QWEN_DIRECT_POLICY_BACKEND

    def _model_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt_payload = self._prompt_payload(payload)
        step_mode = self._planner_step_mode(prompt_payload)
        effective_thinking_mode, thinking_due = self._qwen_thinking_mode_for_step(
            prompt_payload
        )
        step_mode["qwen_thinking_effective_mode"] = effective_thinking_mode
        step_mode["qwen_thinking_due"] = thinking_due
        self._apply_model_step_context(prompt_payload, step_mode)
        model_images = self._model_image_files(
            prompt_payload, step_mode["planner_step_mode"]
        )
        self._attach_direct_model_image_order_context(
            prompt_payload,
            model_images,
            step_mode["planner_step_mode"],
        )
        prompt = self._model_prompt_from_prompt_payload(
            prompt_payload,
            qwen_thinking_mode=effective_thinking_mode,
        )
        assembled_prompt_tokens = self._estimate_tokens(prompt)
        if self._should_use_local_policy_fast_step(step_mode):
            return self._memory_guided_policy_fast_decision(
                prompt_payload,
                step_mode,
                prompt,
                model_images,
                assembled_prompt_tokens,
            )
        if assembled_prompt_tokens > QWEN_HARD_MAX_INPUT_TOKENS:
            context_audit = self._context_audit(
                prompt=prompt,
                usage={},
                session_id="",
                assembled_prompt_tokens=assembled_prompt_tokens,
                context_profile="plan_model",
                session_mode="stateless_model",
                model_images=model_images,
            )
            self._update_context_audit_for_model_step(context_audit, step_mode)
            context_audit["model_provider"] = self.openclaw_model_provider
            token_guard = {
                "tripped": True,
                "input_tokens": assembled_prompt_tokens,
                "max_input_tokens": QWEN_HARD_MAX_INPUT_TOKENS,
                "total_tokens": assembled_prompt_tokens,
                "output_tokens": 0,
                "qwen_hard_limit_exceeded": True,
            }
            if self._is_qwen_direct_policy():
                context_audit["qwen_model_called"] = False
                return self._direct_policy_failure_decision(
                    "qwen input token limit exceeded",
                    context_audit=context_audit,
                    reason="qwen_direct_input_token_limit",
                )
            return self._agent_token_limit_decision(
                token_guard,
                context_audit=context_audit,
            )

        try:
            model_stdout = self._model_run_stdout(
                prompt,
                model_images,
                thinking_mode=effective_thinking_mode,
            )
        except Exception as exc:
            context_audit = self._context_audit(
                prompt=prompt,
                usage={},
                session_id="",
                assembled_prompt_tokens=assembled_prompt_tokens,
                context_profile="plan_model",
                session_mode="stateless_model",
                model_images=model_images,
            )
            self._update_context_audit_for_model_step(context_audit, step_mode)
            context_audit["model_provider"] = self.openclaw_model_provider
            context_audit["qwen_model_called"] = True
            provider_error_metadata = getattr(exc, "audit_metadata", None)
            if isinstance(provider_error_metadata, dict):
                context_audit.update(provider_error_metadata)
            context_audit[
                "visual_memory_update_status"
            ] = self._mark_visual_memory_error(
                prompt_payload,
                step_mode,
                str(exc),
            )
            if self._is_qwen_direct_policy():
                return self._direct_policy_failure_decision(
                    str(exc),
                    context_audit=context_audit,
                    reason="qwen_direct_provider_error",
                    episode_invalid=True,
                )
            return self._model_fallback_decision(
                payload,
                str(exc),
                context_audit=context_audit,
            )
        usage = self._agent_usage_metadata(model_stdout)
        context_audit = self._context_audit(
            prompt=prompt,
            usage=usage,
            session_id="",
            assembled_prompt_tokens=assembled_prompt_tokens,
            context_profile="plan_model",
            session_mode="stateless_model",
            model_images=model_images,
        )
        self._update_context_audit_for_model_step(context_audit, step_mode)
        context_audit["model_provider"] = self.openclaw_model_provider
        context_audit["qwen_model_called"] = True
        context_audit.update(self._model_request_metadata(model_stdout))
        strict_route_schema = (
            self._is_qwen_direct_policy()
            and self.qwen_output_schema
            in {
                "route_v2",
                "route_v3_staged",
            }
        )
        if strict_route_schema:
            context_audit["qwen_output_repair_attempted"] = False
            context_audit["qwen_output_repair_succeeded"] = False
        model_text = self._model_visible_text(model_stdout)
        parse_model_text = model_text
        local_boundary_sanitized = False
        if self.qwen_output_schema == "route_v3_staged":
            (
                parse_model_text,
                local_boundary_sanitized,
            ) = self._strip_single_json_code_fence(model_text)
        try:
            if strict_route_schema:
                decision = self._extract_strict_route_json_object(parse_model_text)
            else:
                decision = self._extract_json_object(model_text)
            if self.qwen_output_schema == "route_v3_staged":
                (
                    decision,
                    locally_sanitized_fields,
                ) = self._sanitize_route_v3_primary_text_candidate(decision)
                if local_boundary_sanitized:
                    locally_sanitized_fields.append("json_boundary")
                    locally_sanitized_fields.sort()
                if locally_sanitized_fields:
                    context_audit[
                        "qwen_output_local_sanitized_fields"
                    ] = locally_sanitized_fields
            normalized = self._normalize_model_decision(decision, prompt_payload)
            if self._is_qwen_direct_policy():
                context_audit["qwen_output_json_valid"] = True
        except RuntimeError as exc:
            if self._is_qwen_direct_policy():
                context_audit["qwen_output_json_valid"] = False
            if strict_route_schema:
                context_audit["qwen_output_repair_attempted"] = True
                context_audit["qwen_output_repair_thinking_enabled"] = False
                context_audit[
                    "qwen_output_repair_error_category"
                ] = self._strict_route_error_category(exc)
                context_audit["qwen_output_repair_error_detail"] = str(exc)[:500]
                repair_error: Optional[Exception] = None
                try:
                    runtime_context = prompt_payload.get("runtime_context")
                    expected_active_stage_id = (
                        str(runtime_context.get("active_stage_id") or "")
                        if isinstance(runtime_context, dict)
                        else ""
                    )
                    repair_stdout = self._model_run_stdout(
                        self._strict_route_repair_prompt(
                            model_text,
                            exc,
                            expected_active_stage_id=expected_active_stage_id,
                        ),
                        {"paths": [], "provider_image_labels": []},
                        thinking_mode="off",
                    )
                    repair_metadata = self._model_request_metadata(repair_stdout)
                    repair_usage = self._agent_usage_metadata(repair_stdout)
                    for source_key, output_key in (
                        ("input", "qwen_output_repair_input_tokens"),
                        ("output", "qwen_output_repair_output_tokens"),
                        ("totalTokens", "qwen_output_repair_total_tokens"),
                    ):
                        value = self._usage_int(repair_usage, source_key)
                        if value is not None:
                            context_audit[output_key] = value
                    if "qwen_api_request_attempts" in repair_metadata:
                        context_audit[
                            "qwen_output_repair_request_attempts"
                        ] = repair_metadata["qwen_api_request_attempts"]
                    if "qwen_api_retry_count" in repair_metadata:
                        context_audit[
                            "qwen_output_repair_retry_count"
                        ] = repair_metadata["qwen_api_retry_count"]
                    repair_text = self._model_visible_text(repair_stdout)
                    repair_parse_text = repair_text
                    repair_boundary_sanitized = False
                    if self.qwen_output_schema == "route_v3_staged":
                        (
                            repair_parse_text,
                            repair_boundary_sanitized,
                        ) = self._strip_single_json_code_fence(repair_text)
                    repair_candidate = self._extract_strict_route_json_object(
                        repair_parse_text
                    )
                    (
                        repair_candidate,
                        sanitized_fields,
                    ) = self._sanitize_strict_route_repair_candidate(repair_candidate)
                    if repair_boundary_sanitized:
                        sanitized_fields.append("json_boundary")
                        sanitized_fields.sort()
                    if sanitized_fields:
                        context_audit[
                            "qwen_output_repair_sanitized_fields"
                        ] = sanitized_fields
                    normalized = self._normalize_model_decision(
                        repair_candidate,
                        prompt_payload,
                    )
                except Exception as repair_exc:
                    repair_error = repair_exc
                if repair_error is None:
                    context_audit["qwen_output_json_valid"] = True
                    context_audit["qwen_output_repair_succeeded"] = True
                else:
                    context_audit["qwen_output_repair_succeeded"] = False
                    context_audit[
                        "qwen_output_repair_final_error_category"
                    ] = self._strict_route_error_category(repair_error)
                    context_audit["qwen_output_repair_final_error_detail"] = str(
                        repair_error
                    )[:500]
                    context_audit[
                        "visual_memory_update_status"
                    ] = self._mark_visual_memory_error(
                        prompt_payload,
                        step_mode,
                        str(repair_error),
                    )
                    return self._direct_policy_failure_decision(
                        str(repair_error),
                        context_audit=context_audit,
                        reason="qwen_direct_invalid_episode_abort",
                        episode_invalid=True,
                    )
            else:
                context_audit[
                    "visual_memory_update_status"
                ] = self._mark_visual_memory_error(
                    prompt_payload,
                    step_mode,
                    str(exc),
                )
                if self._is_qwen_direct_policy():
                    return self._direct_policy_failure_decision(
                        str(exc),
                        context_audit=context_audit,
                        reason="qwen_direct_parse_error",
                    )
                return self._model_fallback_decision(
                    payload,
                    str(exc),
                    context_audit=context_audit,
                )
        self._enrich_write_memory_with_visual_observation(normalized, prompt_payload)
        context_audit["visual_memory_update_status"] = self._update_visual_memory_cache(
            prompt_payload,
            normalized,
            model_images,
            step_mode,
        )
        visual_analysis = self._visual_analysis_metadata(prompt_payload)
        runtime_metadata = dict(normalized.get("runtime_metadata") or {})
        if visual_analysis:
            runtime_metadata["visual_analysis"] = visual_analysis
        runtime_metadata["context_audit"] = context_audit
        normalized["runtime_metadata"] = runtime_metadata
        return normalized

    def _direct_policy_failure_decision(
        self,
        error: str,
        context_audit: Optional[Dict[str, Any]] = None,
        reason: str = "qwen_direct_failure",
        episode_invalid: bool = False,
    ) -> Dict[str, Any]:
        audit = dict(context_audit or {})
        audit.setdefault("planner_authority", "qwen")
        audit.setdefault("qwen_candidate_requested", True)
        audit.setdefault("qwen_model_called", False)
        audit.setdefault("qwen_provider", self.openclaw_model_provider)
        audit.setdefault("qwen_api_called", self.openclaw_model_provider == "qwen_api")
        audit["policy_backend"] = QWEN_DIRECT_POLICY_BACKEND
        audit["qwen_failure"] = True
        audit["qwen_failure_reason"] = error
        return {
            "intent": "act",
            "tool_name": "QwenDirectPolicy",
            "arguments": {
                "action_text": "STOP",
                "confidence": 0.0,
                "qwen_failure": True,
                "qwen_failure_reason": error,
                "candidate_action": None,
                "fallback_policy": (
                    "invalid_episode_abort" if episode_invalid else "hard_failure_stop"
                ),
                "episode_invalid": episode_invalid,
            },
            "reason": reason,
            "runtime_metadata": {"context_audit": audit},
        }

    def _model_fallback_decision(
        self,
        payload: Dict[str, Any],
        error: str,
        context_audit: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        fallback = self._heuristic_plan(payload)
        fallback["reason"] = f"openclaw_cli_model_fallback:{fallback['reason']}"
        fallback["arguments"] = dict(fallback.get("arguments") or {})
        fallback["arguments"]["planner_error"] = error
        if context_audit is not None:
            fallback["runtime_metadata"] = {"context_audit": context_audit}
        return fallback

    def _should_use_local_policy_fast_step(self, step_mode: Dict[str, Any]) -> bool:
        if self._is_qwen_direct_policy():
            return False
        return (
            step_mode.get("planner_step_mode") == "fast_text"
            and self.openclaw_model_fast_mode in LOCAL_POLICY_FAST_MODES
        )

    def _memory_guided_policy_fast_decision(
        self,
        prompt_payload: Dict[str, Any],
        step_mode: Dict[str, Any],
        prompt: str,
        model_images: Dict[str, Any],
        assembled_prompt_tokens: int,
    ) -> Dict[str, Any]:
        context_audit = self._context_audit(
            prompt=prompt,
            usage={},
            session_id="",
            assembled_prompt_tokens=assembled_prompt_tokens,
            context_profile="plan_model_fast_skip",
            session_mode="local_policy_fast",
            model_images=model_images,
        )
        self._update_context_audit_for_model_step(context_audit, step_mode)
        context_audit["model_provider"] = self.openclaw_model_provider
        context_audit["planner_authority"] = "local_policy"
        context_audit["qwen_api_called"] = False
        context_audit["model_call_skipped"] = True
        context_audit["model_skip_reason"] = "memory_guided_policy_fast"
        context_audit["fast_policy_mode"] = self.openclaw_model_fast_mode
        context_audit["provider_usage_source"] = "not_called"
        return {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": self._memory_guided_policy_arguments(
                prompt_payload, step_mode
            ),
            "reason": "openclaw_memory_guided_policy_fast",
            "runtime_metadata": {"context_audit": context_audit},
        }

    def _memory_guided_policy_arguments(
        self,
        prompt_payload: Dict[str, Any],
        step_mode: Dict[str, Any],
    ) -> Dict[str, Any]:
        memory = step_mode.get("visual_memory")
        if not isinstance(memory, dict):
            memory = {}
        arguments: Dict[str, Any] = {}
        subgoal = str(memory.get("last_suggested_subgoal") or "").strip()
        if subgoal:
            arguments["active_subgoal"] = subgoal
        lines: List[str] = []
        summary = str(memory.get("last_visual_summary") or "").strip()
        if summary:
            lines.append(f"Cached visual memory: {summary}")
        if subgoal:
            lines.append(f"Suggested subgoal: {subgoal}")
        reason = str(memory.get("last_qwen_reason") or "").strip()
        if reason:
            lines.append(f"Last Qwen reason: {reason}")
        runtime_context = prompt_payload.get("runtime_context") or {}
        if isinstance(runtime_context, dict):
            context_note = str(runtime_context.get("memory_context_text") or "").strip()
            if context_note:
                lines.append(f"Retrieved memory: {context_note}")
        if lines:
            arguments["memory_context_text"] = "\n".join(lines)[
                :MAX_PROMPT_MEMORY_CONTEXT_CHARS
            ]
        return arguments

    def _model_run_stdout(
        self,
        prompt: str,
        model_images: Dict[str, Any],
        timeout_s: Optional[float] = None,
        thinking_mode: Optional[str] = None,
    ) -> str:
        request_timeout_s = (
            self.agent_timeout_s if timeout_s is None else float(timeout_s)
        )
        image_paths = list(model_images.get("paths") or [])
        if self.openclaw_model_provider == "qwen_api":
            if self.model_client is None:
                self.model_client = QwenApiModelClient(
                    run_openclaw=self.run_openclaw,
                    openclaw_profile=self.openclaw_profile,
                    thinking_mode=self.qwen_thinking_mode,
                    thinking_budget=self.qwen_thinking_budget,
                    transport_mode=self.qwen_transport_mode,
                )
            run_kwargs = {
                "prompt": prompt,
                "image_paths": image_paths,
                "model": self.openclaw_model,
                "timeout_s": request_timeout_s,
                "image_labels": list(
                    model_images.get("provider_image_labels") or []
                ),
            }
            if isinstance(self.model_client, QwenApiModelClient):
                run_kwargs["thinking_mode"] = thinking_mode
            response = self.model_client.run(**run_kwargs)
            return json.dumps(response, ensure_ascii=True)

        args = self._openclaw_args("capability", "model", "run", "--json")
        if self.openclaw_model:
            args.extend(["--model", self.openclaw_model])
        for image_path in image_paths:
            args.extend(["--file", image_path])
        args.extend(["--prompt", prompt])
        result = self.run_openclaw(args, request_timeout_s)
        if result.returncode != 0:
            message = (
                result.stderr or result.stdout or "openclaw model run failed"
            ).strip()
            raise RuntimeError(message)
        return result.stdout or ""

    def _heuristic_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state") or {}
        instruction = str(state.get("instruction") or "")
        step_id = int(state.get("step_id") or 0)

        if step_id == 0:
            return self._memory_recall(
                instruction,
                step_id,
                "openclaw_cli_initial_recall",
            )
        if step_id % self.recall_interval_steps == 0:
            return self._memory_recall(
                instruction,
                step_id,
                "openclaw_cli_interval_recall",
            )
        return {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {},
            "reason": "openclaw_cli_default_act",
        }

    def _agent_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._agent_token_guard:
            return self._agent_token_limit_decision(self._agent_token_guard)
        prompt_payload = self._prompt_payload(payload)
        prompt = self._agent_prompt_from_prompt_payload(prompt_payload)
        session_id = self._agent_session_id_for_payload(prompt_payload)
        assembled_prompt_tokens = self._estimate_tokens(prompt)
        if assembled_prompt_tokens > QWEN_HARD_MAX_INPUT_TOKENS:
            context_audit = self._context_audit(
                prompt=prompt,
                usage={},
                session_id=session_id,
                assembled_prompt_tokens=assembled_prompt_tokens,
            )
            token_guard = {
                "tripped": True,
                "input_tokens": assembled_prompt_tokens,
                "max_input_tokens": QWEN_HARD_MAX_INPUT_TOKENS,
                "total_tokens": assembled_prompt_tokens,
                "output_tokens": 0,
                "qwen_hard_limit_exceeded": True,
            }
            return self._agent_token_limit_decision(
                token_guard,
                context_audit=context_audit,
            )
        args = self._openclaw_args(
            "agent",
            "--agent",
            self.agent_id,
            "--json",
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--timeout",
            str(int(self.agent_timeout_s)),
        )
        result = self.run_openclaw(args, self.agent_timeout_s)
        if result.returncode != 0:
            message = (
                result.stderr or result.stdout or "openclaw agent failed"
            ).strip()
            raise RuntimeError(message)
        usage = self._agent_usage_metadata(result.stdout or "")
        if not usage:
            usage = self._agent_usage_from_session_file(session_id)
        context_audit = self._context_audit(
            prompt=prompt,
            usage=usage,
            session_id=session_id,
            assembled_prompt_tokens=assembled_prompt_tokens,
        )
        token_guard = self._agent_token_guard_metadata(usage)
        if token_guard:
            token_guard["context_audit"] = context_audit
            self._agent_token_guard = token_guard
            return self._agent_token_limit_decision(
                token_guard, context_audit=context_audit
            )
        agent_text = self._agent_visible_text(result.stdout or "")
        try:
            decision = self._extract_json_object(agent_text)
            normalized = self._normalize_decision(decision)
        except RuntimeError as exc:
            fallback = self._heuristic_plan(payload)
            fallback["reason"] = f"openclaw_cli_agent_fallback:{fallback['reason']}"
            fallback["arguments"] = dict(fallback.get("arguments") or {})
            fallback["arguments"]["planner_error"] = str(exc)
            fallback["runtime_metadata"] = {"context_audit": context_audit}
            return fallback
        self._enrich_write_memory_with_visual_observation(normalized, prompt_payload)
        visual_analysis = self._visual_analysis_metadata(prompt_payload)
        runtime_metadata = dict(normalized.get("runtime_metadata") or {})
        if visual_analysis:
            runtime_metadata["visual_analysis"] = visual_analysis
        runtime_metadata["context_audit"] = context_audit
        normalized["runtime_metadata"] = runtime_metadata
        return normalized

    def _agent_usage_metadata(self, stdout: str) -> Dict[str, Any]:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return {}
        last_call_usage = self._agent_last_call_usage(data)
        if last_call_usage:
            return last_call_usage
        usage = self._find_usage_dict(data)
        return dict(usage) if usage else {}

    def _agent_last_call_usage(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        meta = data.get("meta")
        if not isinstance(meta, dict):
            return {}
        agent_meta = meta.get("agentMeta")
        if not isinstance(agent_meta, dict):
            return {}
        usage = agent_meta.get("lastCallUsage")
        if isinstance(usage, dict):
            return dict(usage)
        return {}

    def _find_usage_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            usage = value.get("usage")
            if isinstance(usage, dict) and (
                "input" in usage or "totalTokens" in usage or "output" in usage
            ):
                return usage
            for nested in value.values():
                found = self._find_usage_dict(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = self._find_usage_dict(item)
                if found:
                    return found
        return {}

    def _agent_usage_from_session_file(self, session_id: str) -> Dict[str, Any]:
        session_file = self._agent_session_file(session_id)
        if not session_file.exists():
            return {}
        last_usage: Dict[str, Any] = {}
        for line in session_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message") if isinstance(data, dict) else None
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            usage = message.get("usage")
            if isinstance(usage, dict):
                last_usage = usage
        return dict(last_usage)

    def _agent_session_file(self, session_id: str) -> Path:
        if self.openclaw_session_dir:
            return Path(self.openclaw_session_dir) / f"{session_id}.jsonl"
        return (
            Path.home()
            / ".openclaw"
            / "agents"
            / self.agent_id
            / "sessions"
            / f"{session_id}.jsonl"
        )

    def _agent_token_guard_metadata(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        if self.agent_max_input_tokens <= 0 or not usage:
            return {}
        input_tokens = int(usage.get("input") or 0)
        if input_tokens < self.agent_max_input_tokens:
            return {}
        return {
            "tripped": True,
            "input_tokens": input_tokens,
            "max_input_tokens": self.agent_max_input_tokens,
            "total_tokens": int(usage.get("totalTokens") or 0),
            "output_tokens": int(usage.get("output") or 0),
        }

    def _agent_token_limit_decision(
        self,
        token_guard: Dict[str, Any],
        context_audit: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        input_tokens = int(token_guard.get("input_tokens") or 0)
        max_input_tokens = int(
            token_guard.get("max_input_tokens") or self.agent_max_input_tokens
        )
        planner_error = (
            "openclaw_agent_input_token_limit:"
            f" input_tokens={input_tokens} max_input_tokens={max_input_tokens}"
        )
        return {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {
                "action_text": "STOP",
                "planner_error": planner_error,
                "input_tokens": input_tokens,
                "max_input_tokens": max_input_tokens,
            },
            "reason": "openclaw_agent_input_token_limit",
            "runtime_metadata": {
                "agent_token_guard": token_guard,
                "context_audit": context_audit
                or token_guard.get("context_audit")
                or {},
            },
        }

    def _agent_prompt(self, payload: Dict[str, Any]) -> str:
        prompt_payload = self._prompt_payload(payload)
        return self._agent_prompt_from_prompt_payload(prompt_payload)

    def _agent_prompt_from_prompt_payload(self, prompt_payload: Dict[str, Any]) -> str:
        compact_payload = json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True)
        visual_context_lines = self._visual_context_lines(prompt_payload)
        return "\n".join(
            [
                "You are the OpenClaw LLM planner brain for ClawNav.",
                "Return only one JSON object. Do not wrap it in markdown.",
                "Allowed intents and tools:",
                "- act -> NavigationPolicySkill",
                "- recall_memory -> MemoryQuerySkill",
                "- write_memory -> MemoryWriteSkill",
                "- verify_progress -> ProgressCriticSkill",
                "- replan -> ReplannerSkill",
                "Schema:",
                '{"intent":"act|recall_memory|write_memory|verify_progress|replan",'
                '"tool_name":"...",'
                '"arguments":{},'
                '"reason":"short_reason"}',
                "For act/replan, when the next discrete action is known, put it in arguments.action_text.",
                "arguments.action_text must be one of STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT.",
                "For replan, also put the recovery instruction in arguments.active_subgoal.",
                "Do not bury mandatory action commands only in reason.",
                "For write_memory add visual fields and write_gate.",
                "write_gate has candidate_reason, curator_decision write|skip, curator_reason, confidence.",
                "Use episode memory for current; scene memory only approved prior. No future/oracle evidence.",
                "Use act unless memory recall/write, progress verification, or replanning is useful before the next navigation action.",
                *visual_context_lines,
                "Payload:",
                compact_payload,
            ]
        )

    def _model_prompt_from_prompt_payload(
        self,
        prompt_payload: Dict[str, Any],
        qwen_thinking_mode: Optional[str] = None,
    ) -> str:
        if self._is_qwen_direct_policy():
            return self._direct_model_prompt_from_prompt_payload(
                prompt_payload,
                qwen_thinking_mode=qwen_thinking_mode,
            )
        compact_payload = json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True)
        runtime_context = prompt_payload.get("runtime_context") or {}
        planner_step_mode = ""
        if isinstance(runtime_context, dict):
            planner_step_mode = str(runtime_context.get("planner_step_mode") or "")
        image_instruction = (
            "No image is attached for this step. Use the cached visual summary from the last visual_update as visual memory."
            if planner_step_mode == "fast_text"
            else "Attached image files correspond to payload image paths. If images are attached, refresh arguments.visual_summary and arguments.suggested_subgoal when useful."
        )
        return "\n".join(
            [
                "Return one JSON object for ClawNav navigation planning.",
                'Schema: {"intent":"act|recall_memory|write_memory|verify_progress|replan","tool_name":"...","arguments":{},"reason":"short"}',
                "For act/replan set arguments.action_text to STOP, MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT when known.",
                image_instruction,
                "Payload:",
                compact_payload,
            ]
        )

    def _direct_model_prompt_from_prompt_payload(
        self,
        prompt_payload: Dict[str, Any],
        qwen_thinking_mode: Optional[str] = None,
    ) -> str:
        compact_payload = json.dumps(
            self._direct_model_prompt_payload(prompt_payload),
            ensure_ascii=True,
            sort_keys=True,
        )
        runtime_context = prompt_payload.get("runtime_context") or {}
        planner_step_mode = ""
        if isinstance(runtime_context, dict):
            planner_step_mode = str(runtime_context.get("planner_step_mode") or "")
        attached_order = (
            runtime_context.get("attached_image_order")
            if isinstance(runtime_context, dict)
            else None
        )
        has_map_view = bool(
            isinstance(attached_order, dict)
            and "map_view" in (attached_order.get("sources") or [])
        )
        image_instruction = (
            "No image is attached for this step. Use compact cached visual and memory evidence only."
            if planner_step_mode == "fast_text"
            else (
                "Attached images are ordered as optional map_view floorplan first, then retrieved "
                "history memory, then recent_current frames from oldest to newest, with the final "
                "image as the current observation. Use map_view for coarse spatial layout and "
                "visited-trail context only; base immediate action feasibility on the final current RGB image."
                if has_map_view
                else (
                    "Attached images are ordered as retrieved history memory first, then recent_current "
                    "frames from oldest to newest, with the final image as the current t observation. "
                    "Base the action on the final current image and use earlier images only for progress context."
                )
            )
        )
        if (
            self.dynamic_visual_context_enabled
            and isinstance(attached_order, dict)
            and attached_order.get("sources")
        ):
            image_instruction = (
                "Attached images carry controlled purpose labels in this exact order: "
                + ", ".join(str(value) for value in attached_order.get("sources") or [])
                + ". Use each image only for its labeled purpose; current is the final image and controls immediate action feasibility."
            )
        if self.qwen_output_schema == "route_v3_staged":
            schema_line = ROUTE_V3_STAGED_SCHEMA_LINE
        elif self.qwen_output_schema == "route_v2":
            schema_line = 'Schema: {"action_text":"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT","confidence":0.0,"visual_summary":"short","progress_state":"short","route_stage":"start|en_route|intermediate_landmark|post_landmark_transition|approaching_target|verifying_target|complete|unknown","confirmed_landmarks":["short"],"current_target":"short","target_relation":"short","stop_evidence":"none|visible_goal|instruction_complete","semantic_stop_state":"not_ready|visible_not_reached|approaching_target|at_or_inside_target|beside_target|instruction_complete_at_target","reason":"short"}'
        else:
            schema_line = 'Schema: {"action_text":"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT","confidence":0.0,"visual_summary":"short","progress_state":"short","stop_evidence":"none|visible_goal|instruction_complete","current_target":"short","target_relation":"short","semantic_stop_state":"not_ready|visible_not_reached|approaching_target|at_or_inside_target|beside_target|instruction_complete_at_target","reason":"short"}'
        field_limits_line = (
            ROUTE_V2_FIELD_LIMITS_TEXT
            if self.qwen_output_schema in {"route_v2", "route_v3_staged"}
            else ""
        )
        staged_lines = []
        if self.qwen_output_schema == "route_v3_staged":
            staged_lines = [
                "The immutable full instruction and controller-owned active/completed/pending stages are separate fields in the payload.",
                "Copy active_stage_id exactly. Qwen may propose stage_complete_candidate but must not mutate stage state.",
                "Evaluate stage_complete_candidate only against the controller-owned active stage, not against later stages or the full instruction.",
                "When attached visual evidence satisfies the active stage completion_cues, set stage_complete_candidate=true even though later stages remain; otherwise set it false.",
                "Use stage_relation for the visually grounded relation to the active stage and cite only attached stage_evidence_refs.",
                "Current RGB is final and authoritative for immediate feasibility; map_view is goal-free coarse topology; historical images are evidence only.",
                "A visible landmark alone cannot prove pass, enter, exit, or final arrival.",
            ]
        thinking_lines = []
        if (qwen_thinking_mode or self.qwen_thinking_mode) == "on":
            thinking_lines = [
                "Use Qwen thinking to reason internally before producing the final JSON.",
                "In that internal reasoning, inspect map_view for coarse topology, agent heading, visited trail, target direction, and route-stage consistency before choosing an action.",
                "Treat collision marks as evidence only when map_context says collision_overlay_enabled=true, and never invent map content that is not visible.",
                "Do not include chain-of-thought, reasoning_content, markdown, or prose in the final answer; output only the short JSON object.",
            ]
        return "\n".join(
            [
                "Return one JSON object for QwenDirectPolicy.",
                "QwenDirectPolicy is the only action-producing policy in this mode.",
                schema_line,
                field_limits_line,
                "Do not name external tools. Do not emit markdown.",
                *staged_lines,
                *thinking_lines,
                "Action rules:",
                "Use TURN_LEFT or TURN_RIGHT when the intended route or landmark is off-center, missing from the current view, or requires reorientation.",
                "Embodiment/action scale: MOVE_FORWARD advances 0.25 meters; TURN_LEFT and TURN_RIGHT rotate 15 degrees each.",
                "MOVE_FORWARD is valid only when the intended route remains centered, visible, and unobstructed in the current visual evidence.",
                "A path clear alone is not enough after repeated forward actions; require progress evidence or choose a turn/STOP.",
                "STOP requires semantic arrival, not just seeing the target.",
                "Do not STOP if one more 0.25-meter MOVE_FORWARD is likely needed to be inside an archway, doorway, entrance, room threshold, or final wait location.",
                "For STOP, semantic_stop_state must be at_or_inside_target, beside_target, or instruction_complete_at_target.",
                "Do not STOP for visible_not_reached, approaching_target, not_ready, near/vicinity, edge, visible ahead, directly ahead, or path-clear-only evidence.",
                "For multi-step routes, do not STOP just because the final object is visible; visual_summary must mention observed intermediate landmarks such as objects or rooms passed on the route, not only progress_state or reason claims.",
                "For archway, doorway, or entrance targets, beside_target or next-to evidence is too loose. Do not STOP on the outside face of an archway, doorway, or entrance; if the opening is centered and traversable, choose MOVE_FORWARD and let the runtime controller confirm arrival.",
                "Use current_target for the current final landmark/room/object and target_relation for the agent-to-target relation in the final current image.",
                "When motion_feedback reports actual_effect=blocked or recommended_constraint=avoid_forward, avoid MOVE_FORWARD unless the final current RGB image clearly shows a newly aligned open path.",
                "When blocked_stop_feedback is present, do not choose STOP. Use MOVE_FORWARD only when the current image clearly shows the route continues forward; otherwise use TURN_LEFT or TURN_RIGHT to recheck alignment or target evidence.",
                "When forward_stall_feedback is present, do not choose MOVE_FORWARD; choose only TURN_LEFT or TURN_RIGHT and explain the corrective visual reason.",
                "When turn_loop_feedback is present, stop repeating the previous turn pattern. Use map_view visited trail plus left_scan, center_scan, and right_scan to choose the least-visited traversable direction. Collect a missing scan with the corresponding turn; once scans are available, align the final current RGB with the chosen open direction and prefer MOVE_FORWARD to leave the loop.",
                "When stop_verification_feedback is present, do not choose MOVE_FORWARD. Recheck the same current observation and choose STOP only if the agent is already at or inside the structural target boundary; otherwise choose TURN_LEFT or TURN_RIGHT to inspect.",
                "When route_progress is present, treat negated waypoint mentions as unseen. Do not claim a walk-past waypoint is complete until it appears in passed_waypoints, and do not choose final STOP while turn_round_completed is false or required_waypoints are not passed.",
                image_instruction,
                "Payload:",
                compact_payload,
            ]
        )

    def _direct_model_prompt_payload(
        self, prompt_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        sanitized = self._sanitize_direct_prompt_value(prompt_payload)
        return sanitized if isinstance(sanitized, dict) else {}

    def _sanitize_direct_prompt_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            for key, nested in value.items():
                if key in DIRECT_PROMPT_OMIT_KEYS:
                    continue
                sanitized_value = self._sanitize_direct_prompt_value(nested)
                if sanitized_value in ({}, [], None, ""):
                    continue
                sanitized[key] = sanitized_value
            return sanitized
        if isinstance(value, list):
            sanitized_items = [
                self._sanitize_direct_prompt_value(item) for item in value
            ]
            return [item for item in sanitized_items if item not in ({}, [], None, "")]
        return value

    def _planner_step_mode(self, prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
        state = prompt_payload.get("state") or {}
        if not isinstance(state, dict):
            state = {}
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            runtime_context = {}
        step_id = self._step_id(state)
        cache_key = self._visual_memory_cache_key(prompt_payload)
        memory = self._visual_memory_cache.get(cache_key) if cache_key else None
        age_steps = self._visual_memory_age_steps(step_id, memory)
        mode = "visual_update"
        reason = "every_step_visual_update"
        if step_id == 0:
            reason = "initial_step"
        elif self._has_keyframe_candidate(runtime_context):
            reason = "keyframe_candidate"
        elif self._force_visual_refresh_requested(runtime_context):
            reason = "forced_visual_refresh"
        elif self._is_qwen_direct_policy():
            reason = "qwen_direct_visual_update"
        elif not self._model_fast_mode_enabled():
            reason = "fast_mode_disabled"
        elif not memory:
            reason = "missing_visual_memory"
        elif memory.get("last_update_error"):
            reason = "previous_visual_update_failed"
        elif age_steps >= self.openclaw_model_image_interval_steps:
            reason = "image_interval"
        else:
            mode = "fast_text"
            reason = "cached_visual_memory"
        map_context = runtime_context.get("map_context")
        if not isinstance(map_context, dict):
            map_context = {}
        return {
            "planner_step_mode": mode,
            "fast_break_reason": reason,
            "cache_key": cache_key,
            "visual_memory": memory or {},
            "visual_memory_age_steps": age_steps,
            "visual_memory_valid_until_step": (
                memory.get("valid_until_step") if isinstance(memory, dict) else None
            ),
            "memory_context_used": bool(
                mode == "fast_text"
                and self.openclaw_model_fast_use_memory_context
                and isinstance(memory, dict)
                and memory.get("last_visual_summary")
            ),
            "input_regime": runtime_context.get("input_regime")
            or map_context.get("input_regime"),
            "map_assist_mode": runtime_context.get("map_assist_mode")
            or map_context.get("mode"),
            "map_frame_interval_steps": runtime_context.get("map_frame_interval_steps")
            or map_context.get("map_frame_interval_steps"),
            "map_frame_due": map_context.get("map_frame_due"),
            "map_step_id": map_context.get("map_step_id"),
            "map_available": map_context.get("map_available"),
            "map_view_scope": map_context.get("map_view_scope"),
            "map_refresh_reason": map_context.get("map_refresh_reason"),
            "turn_loop_recovery_active": runtime_context.get(
                "turn_loop_recovery_active"
            ),
            "visual_recovery_phase": runtime_context.get("visual_recovery_phase"),
            "motion_feedback_enabled": runtime_context.get("motion_feedback_enabled"),
            "forward_stall_odometry_enabled": runtime_context.get(
                "forward_stall_odometry_enabled"
            ),
            "map_collision_overlay_enabled": runtime_context.get(
                "map_collision_overlay_enabled"
            ),
        }

    def _force_visual_refresh_requested(self, runtime_context: Dict[str, Any]) -> bool:
        if runtime_context.get("force_visual_refresh") is True:
            return True
        control_context = runtime_context.get("control_context")
        return (
            isinstance(control_context, dict)
            and control_context.get("force_visual_refresh") is True
        )

    def _model_fast_mode_enabled(self) -> bool:
        return self.openclaw_model_fast_mode == "qwen_text_only" or (
            self.openclaw_model_fast_mode in LOCAL_POLICY_FAST_MODES
        )

    def _apply_model_step_context(
        self,
        prompt_payload: Dict[str, Any],
        step_mode: Dict[str, Any],
    ) -> None:
        runtime_context = prompt_payload.setdefault("runtime_context", {})
        if not isinstance(runtime_context, dict):
            return
        runtime_context["planner_step_mode"] = step_mode["planner_step_mode"]
        runtime_context["visual_memory_backend"] = "adapter_episode_local"
        runtime_context["visual_memory_age_steps"] = step_mode.get(
            "visual_memory_age_steps"
        )
        if step_mode.get("visual_memory_valid_until_step") is not None:
            runtime_context["visual_memory_valid_until_step"] = step_mode.get(
                "visual_memory_valid_until_step"
            )
        if step_mode["planner_step_mode"] != "fast_text":
            return
        runtime_context.pop("current_image_path", None)
        runtime_context.pop("recent_keyframe_paths", None)
        runtime_context.pop("keyframe_candidate", None)
        runtime_context.pop("memory_images", None)
        runtime_context[
            "model_visual_context_note"
        ] = "No image is attached for this step. Use cached visual memory from the last visual_update."
        if not self.openclaw_model_fast_use_memory_context:
            return
        memory = step_mode.get("visual_memory")
        if not isinstance(memory, dict) or not memory.get("last_visual_summary"):
            return
        runtime_context["cached_visual_memory"] = {
            "last_visual_step_id": memory.get("last_visual_step_id"),
            "last_visual_summary": memory.get("last_visual_summary"),
            "last_suggested_subgoal": memory.get("last_suggested_subgoal"),
            "last_qwen_reason": memory.get("last_qwen_reason"),
            "last_action_text": memory.get("last_action_text"),
            "valid_until_step": memory.get("valid_until_step"),
            "source_image_paths": memory.get("source_image_paths") or [],
        }

    def _model_image_files(
        self,
        prompt_payload: Dict[str, Any],
        planner_step_mode: str = "visual_update",
    ) -> Dict[str, Any]:
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            return {"paths": [], "missing_paths": []}
        dynamic_qwen_selection = (
            self._is_qwen_direct_policy() and self.dynamic_visual_context_enabled
        )
        if self.openclaw_model_max_images <= 0 and not dynamic_qwen_selection:
            return {"paths": [], "missing_paths": []}
        if planner_step_mode == "fast_text":
            return {"paths": [], "missing_paths": []}
        candidates = self._model_visual_update_image_candidates(runtime_context)
        paths: List[str] = []
        missing_paths: List[str] = []
        selected_candidates: List[Dict[str, Any]] = []
        missing_candidates: List[Dict[str, Any]] = []
        for candidate in candidates:
            image_path = str(candidate.get("path") or "")
            if not image_path:
                continue
            if Path(image_path).is_file():
                paths.append(image_path)
                selected_candidates.append(candidate)
            else:
                missing_paths.append(image_path)
                missing_candidates.append(candidate)
        model_images = {
            "paths": paths,
            "missing_paths": missing_paths,
        }
        model_images.update(
            self._model_image_selection_metadata(
                selected_candidates,
                missing_candidates,
            )
        )
        if self._is_qwen_direct_policy() and self.dynamic_visual_context_enabled:
            model_images.update(
                self._dynamic_visual_selection_metadata(
                    runtime_context,
                    selected_candidates,
                    missing_candidates,
                )
            )
            model_images["provider_image_labels"] = [
                str(candidate.get("source") or "") for candidate in selected_candidates
            ]
        return model_images

    def _model_visual_update_image_paths(
        self, runtime_context: Dict[str, Any]
    ) -> List[str]:
        return [
            candidate["path"]
            for candidate in self._model_visual_update_image_candidates(runtime_context)
            if candidate.get("path")
        ]

    def _model_visual_update_image_candidates(
        self,
        runtime_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self._is_qwen_direct_policy():
            return self._qwen_direct_model_image_candidates(runtime_context)
        return self._default_model_image_candidates(runtime_context)

    def _default_model_image_candidates(
        self,
        runtime_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        selected: List[str] = []
        current_image_path = runtime_context.get("current_image_path")
        if isinstance(current_image_path, str) and current_image_path:
            selected.append(current_image_path)
        keyframe_candidate = runtime_context.get("keyframe_candidate")
        if isinstance(keyframe_candidate, dict):
            candidate_path = keyframe_candidate.get("image_path")
            if isinstance(candidate_path, str) and candidate_path:
                selected.append(candidate_path)
        recent_keyframe_paths = runtime_context.get("recent_keyframe_paths") or []
        if isinstance(recent_keyframe_paths, list):
            selected.extend(
                path
                for path in reversed(recent_keyframe_paths)
                if isinstance(path, str) and path
            )
        deduped: List[str] = []
        for path in selected:
            if path not in deduped:
                deduped.append(path)
        return [
            {
                "path": path,
                "source": self._default_model_image_source(path, runtime_context),
            }
            for path in deduped[: self.openclaw_model_max_images]
        ]

    def _default_model_image_source(
        self,
        path: str,
        runtime_context: Dict[str, Any],
    ) -> str:
        if path == runtime_context.get("current_image_path"):
            return "current"
        keyframe_candidate = runtime_context.get("keyframe_candidate")
        if isinstance(keyframe_candidate, dict) and path == keyframe_candidate.get(
            "image_path"
        ):
            return "keyframe_candidate"
        return "recent_keyframe"

    def _qwen_direct_model_image_candidates(
        self,
        runtime_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self.dynamic_visual_context_enabled:
            return self._dynamic_qwen_direct_model_image_candidates(runtime_context)
        current_image_path = runtime_context.get("current_image_path")
        current_path = current_image_path if isinstance(current_image_path, str) else ""
        current_budget = 1 if current_path else 0
        map_candidate = self._map_view_image_candidate(runtime_context)
        map_candidates = [map_candidate] if map_candidate else []
        pinned_budget = current_budget + len(map_candidates)
        recent_candidates = self._recent_current_frame_candidates(
            current_path,
            limit=MAX_DIRECT_RECENT_CURRENT_IMAGES,
        )
        excluded_paths = {
            str(candidate.get("path") or "")
            for candidate in map_candidates
            if candidate.get("path")
        }
        recent_paths = {
            str(candidate.get("path") or "")
            for candidate in recent_candidates
            if candidate.get("path") and candidate.get("path") not in excluded_paths
        }
        history_budget = max(0, self.openclaw_model_max_images - pinned_budget)
        history_budget = min(MAX_DIRECT_RETRIEVED_MEMORY_IMAGES, history_budget)
        memory_candidates = self._retrieved_memory_image_candidates(
            runtime_context,
            current_path=current_path,
            exclude_paths=recent_paths | excluded_paths,
        )[:history_budget]
        recent_budget = max(
            0,
            self.openclaw_model_max_images
            - len(map_candidates)
            - len(memory_candidates)
            - current_budget,
        )
        if recent_budget < len(recent_candidates):
            recent_candidates = (
                recent_candidates[-recent_budget:] if recent_budget else []
            )
        candidates = list(map_candidates)
        candidates.extend(memory_candidates)
        candidates.extend(recent_candidates[:recent_budget])
        if current_path:
            candidates.append({"path": current_path, "source": "current"})
        return self._dedupe_model_image_candidates(candidates)

    def _dynamic_qwen_direct_model_image_candidates(
        self,
        runtime_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        mode = self._dynamic_visual_context_mode(runtime_context)
        role_order = DYNAMIC_VISUAL_ROLE_ORDERS[mode]
        registry = runtime_context.get("visual_evidence_registry")
        records = [record for record in registry or [] if isinstance(record, dict)]
        current_path = str(runtime_context.get("current_image_path") or "")
        candidates: List[Dict[str, Any]] = []
        for role in role_order:
            if role == "map_view":
                candidate = self._map_view_image_candidate(runtime_context)
            elif role == "current":
                candidate = (
                    {"path": current_path, "source": "current"}
                    if current_path
                    else None
                )
            elif role == "keyframe":
                candidate = self._ranked_keyframe_candidate(records)
            else:
                candidate = self._latest_registry_role_candidate(
                    records,
                    role,
                )
            if candidate:
                candidates.append(candidate)
        return self._dedupe_dynamic_candidates_current_last(candidates)

    @staticmethod
    def _dynamic_visual_context_mode(runtime_context: Dict[str, Any]) -> str:
        if runtime_context.get("blocked_stop_feedback") or runtime_context.get(
            "stop_verification_feedback"
        ):
            return "stop_blocked"
        if (
            runtime_context.get("forward_stall_feedback")
            or runtime_context.get("visual_recovery_active") is True
        ):
            return "stuck"
        return "normal"

    def _latest_registry_role_candidate(
        self,
        records: List[Dict[str, Any]],
        role: str,
    ) -> Optional[Dict[str, Any]]:
        eligible = [
            record
            for record in records
            if role in (record.get("roles") or [])
            and str(record.get("image_path") or "")
        ]
        if not eligible:
            return None
        record = max(
            eligible, key=lambda value: self._nonnegative_int(value.get("step_id"))
        )
        return {
            "path": str(record.get("image_path") or ""),
            "source": role,
            "capture_route_stage": record.get("capture_route_stage"),
            "capture_current_target": record.get("capture_current_target"),
        }

    def _ranked_keyframe_candidate(
        self,
        records: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        candidates = [
            record
            for record in records
            if "keyframe" in (record.get("roles") or [])
            and str(record.get("image_path") or "")
        ]
        if not candidates:
            return None
        references = [
            record
            for record in records
            if record.get("capture_route_stage") not in (None, "", "unknown")
            or record.get("capture_current_target")
        ]
        reference = (
            max(
                references,
                key=lambda value: self._nonnegative_int(value.get("step_id")),
            )
            if references
            else {}
        )
        stage = str(reference.get("capture_route_stage") or "")
        target = str(reference.get("capture_current_target") or "")
        record = max(
            candidates,
            key=lambda value: (
                int(
                    bool(
                        stage
                        and stage != "unknown"
                        and value.get("capture_route_stage") == stage
                    )
                ),
                int(bool(target and value.get("capture_current_target") == target)),
                self._nonnegative_int(value.get("step_id")),
            ),
        )
        return {
            "path": str(record.get("image_path") or ""),
            "source": "keyframe",
            "capture_route_stage": record.get("capture_route_stage"),
            "capture_current_target": record.get("capture_current_target"),
        }

    @staticmethod
    def _dedupe_dynamic_candidates_current_last(
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_path: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for raw_candidate in candidates:
            candidate = dict(raw_candidate)
            path = str(candidate.get("path") or "")
            source = str(candidate.get("source") or "")
            if not path:
                continue
            if path in by_path:
                existing = by_path[path]
                aliases = set(existing.get("aliases") or [])
                if existing.get("source") != source:
                    aliases.add(str(existing.get("source") or ""))
                    aliases.add(source)
                existing["aliases"] = sorted(alias for alias in aliases if alias)
                if source == "current":
                    existing["source"] = "current"
                    existing["aliases"] = [
                        alias for alias in existing["aliases"] if alias != "current"
                    ]
                    order.remove(path)
                    order.append(path)
                continue
            candidate["aliases"] = list(candidate.get("aliases") or [])
            by_path[path] = candidate
            order.append(path)
        return [by_path[path] for path in order]

    def _dynamic_visual_selection_metadata(
        self,
        runtime_context: Dict[str, Any],
        selected_candidates: List[Dict[str, Any]],
        missing_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        mode = self._dynamic_visual_context_mode(runtime_context)
        requested = list(DYNAMIC_VISUAL_ROLE_ORDERS[mode])
        selected_roles = set()
        aliases: Dict[str, List[str]] = {}
        for candidate in selected_candidates:
            source = str(candidate.get("source") or "")
            if source:
                selected_roles.add(source)
            candidate_aliases = [str(value) for value in candidate.get("aliases") or []]
            selected_roles.update(candidate_aliases)
            if candidate_aliases:
                aliases[str(candidate.get("path") or "")] = candidate_aliases
        return {
            "visual_context_mode": mode,
            "requested_image_roles": requested,
            "selected_image_roles": [
                role for role in requested if role in selected_roles
            ],
            "missing_image_roles": [
                role for role in requested if role not in selected_roles
            ],
            "image_role_aliases": aliases,
        }

    def _map_view_image_candidate(
        self,
        runtime_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        map_context = runtime_context.get("map_context")
        if not isinstance(map_context, dict):
            return None
        fresh = bool(
            map_context.get("map_frame_due") and map_context.get("map_available")
        )
        interval = (
            self._nonnegative_int(map_context.get("map_frame_interval_steps")) or 1
        )
        age_steps = self._int_or_none(map_context.get("map_age_steps"))
        cached = bool(
            map_context.get("cached_map_available")
            and age_steps is not None
            and 0 <= age_steps < interval
        )
        if not fresh and not cached:
            return None
        internal = runtime_context.get("_map_internal_context")
        if not isinstance(internal, dict):
            internal = map_context.get("internal_only")
        if not isinstance(internal, dict):
            internal = {}
        path = internal.get("map_image_path") or map_context.get("map_image_path")
        if not isinstance(path, str) or not path:
            return None
        return {
            "path": path,
            "source": "map_view",
            "map_step_id": (
                map_context.get("map_step_id")
                if fresh
                else map_context.get("cached_map_step_id")
            ),
            "map_age_steps": 0 if fresh else age_steps,
            "map_reuse": "fresh" if fresh else "cached",
            "map_image_hash": internal.get("map_image_hash")
            or map_context.get("map_image_hash"),
        }

    def _recent_current_frame_candidates(
        self,
        current_path: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        return [
            {"path": path, "source": "recent_current"}
            for path in self._recent_current_frame_paths(current_path, limit=limit)
        ]

    @staticmethod
    def _recent_current_frame_paths(current_path: str, limit: int) -> List[str]:
        if not current_path or limit <= 0:
            return []
        current = Path(current_path)
        match = re.match(r"^(step_)(\d+)(\.[^.]+)$", current.name)
        if not match:
            return []
        prefix, step_text, suffix = match.groups()
        try:
            current_index = int(step_text)
        except ValueError:
            return []
        start_index = max(0, current_index - limit)
        return [
            str(current.with_name(f"{prefix}{index:0{len(step_text)}d}{suffix}"))
            for index in range(start_index, current_index)
        ]

    def _retrieved_memory_image_candidates(
        self,
        runtime_context: Dict[str, Any],
        current_path: str = "",
        exclude_paths: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        excluded = exclude_paths or set()
        structured = runtime_context.get("retrieved_memory_images")
        if isinstance(structured, list) and structured:
            candidates = []
            for item in structured:
                if not isinstance(item, dict):
                    continue
                image_path = item.get("image_path")
                if not isinstance(image_path, str) or not image_path:
                    continue
                if (
                    current_path and image_path == current_path
                ) or image_path in excluded:
                    continue
                candidates.append(
                    {
                        "path": image_path,
                        "source": str(
                            item.get("source") or "openclaw_retrieved_memory"
                        ),
                        "memory_id": str(
                            item.get("memory_id") or item.get("keyframe_id") or ""
                        ),
                    }
                )
            return self._dedupe_model_image_candidates(candidates)

        raw_paths = runtime_context.get("retrieved_memory_image_paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raw_paths = runtime_context.get("memory_images")
        memory_ids = runtime_context.get("retrieved_memory_ids")
        if not isinstance(memory_ids, list):
            memory_ids = []
        candidates = []
        if isinstance(raw_paths, list):
            for index, image_path in enumerate(raw_paths):
                if not isinstance(image_path, str) or not image_path:
                    continue
                if (
                    current_path and image_path == current_path
                ) or image_path in excluded:
                    continue
                memory_id = memory_ids[index] if index < len(memory_ids) else ""
                candidates.append(
                    {
                        "path": image_path,
                        "source": "openclaw_retrieved_memory",
                        "memory_id": str(memory_id or ""),
                    }
                )
        return self._dedupe_model_image_candidates(candidates)

    @staticmethod
    def _dedupe_model_image_candidates(
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            path = candidate.get("path")
            if not isinstance(path, str) or not path or path in seen:
                continue
            seen.add(path)
            deduped.append(candidate)
        return deduped

    def _model_image_selection_metadata(
        self,
        selected_candidates: List[Dict[str, Any]],
        missing_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        paths = [str(candidate.get("path") or "") for candidate in selected_candidates]
        metadata: Dict[str, Any] = {
            "selected_image_count": len(paths),
            "model_image_sources": [
                str(candidate.get("source") or "unknown")
                for candidate in selected_candidates
            ],
            "current_image_last": bool(
                selected_candidates
                and selected_candidates[-1].get("source") == "current"
            ),
        }
        if not self._is_qwen_direct_policy():
            return metadata

        map_view = [
            candidate
            for candidate in selected_candidates
            if candidate.get("source") == "map_view"
        ]
        missing_map_view = [
            candidate
            for candidate in missing_candidates
            if candidate.get("source") == "map_view"
        ]
        retrieved = [
            candidate
            for candidate in selected_candidates
            if candidate.get("source") not in ("current", "recent_current", "map_view")
        ]
        missing_retrieved = [
            candidate
            for candidate in missing_candidates
            if candidate.get("source") not in ("current", "recent_current", "map_view")
        ]
        recent_current = [
            candidate
            for candidate in selected_candidates
            if candidate.get("source") == "recent_current"
        ]
        missing_recent_current = [
            candidate
            for candidate in missing_candidates
            if candidate.get("source") == "recent_current"
        ]
        retrieved_ids = [
            str(candidate.get("memory_id") or "")
            for candidate in retrieved
            if candidate.get("memory_id")
        ]
        metadata["history_frame_source"] = "openclaw_retrieved_memory"
        metadata["map_view_image_count"] = len(map_view)
        metadata["map_view_image_paths"] = [
            str(candidate.get("path") or "") for candidate in map_view
        ]
        metadata["map_view_missing_image_paths"] = [
            str(candidate.get("path") or "") for candidate in missing_map_view
        ]
        if map_view and map_view[0].get("map_image_hash"):
            metadata["map_image_hash"] = str(map_view[0].get("map_image_hash") or "")
        if map_view:
            metadata["map_age_steps"] = map_view[0].get("map_age_steps")
            metadata["map_reuse"] = map_view[0].get("map_reuse")
        metadata["retrieved_memory_image_count"] = len(retrieved)
        metadata["retrieved_memory_image_paths"] = [
            str(candidate.get("path") or "") for candidate in retrieved
        ]
        metadata["retrieved_memory_missing_image_paths"] = [
            str(candidate.get("path") or "") for candidate in missing_retrieved
        ]
        metadata["recent_current_image_count"] = len(recent_current)
        metadata["recent_current_image_paths"] = [
            str(candidate.get("path") or "") for candidate in recent_current
        ]
        metadata["recent_current_missing_image_paths"] = [
            str(candidate.get("path") or "") for candidate in missing_recent_current
        ]
        if retrieved_ids:
            metadata["retrieved_memory_ids"] = retrieved_ids
        return metadata

    def _attach_direct_model_image_order_context(
        self,
        prompt_payload: Dict[str, Any],
        model_images: Dict[str, Any],
        planner_step_mode: str,
    ) -> None:
        if not self._is_qwen_direct_policy() or planner_step_mode == "fast_text":
            return
        runtime_context = prompt_payload.get("runtime_context")
        if not isinstance(runtime_context, dict):
            return
        sources = [
            str(source)
            for source in model_images.get("model_image_sources", [])
            if isinstance(source, str) and source
        ]
        if not sources:
            return
        runtime_context["attached_image_order"] = {
            "total": len(sources),
            "sources": sources,
            "semantic_order": (
                "map_view_first_when_present_then_retrieved_history_then_recent_current_oldest_to_newest_current_last"
                if "map_view" in sources
                else "retrieved_history_first_then_recent_current_oldest_to_newest_current_last"
            ),
            "current_image_last": bool(model_images.get("current_image_last")),
            "visual_context_mode": model_images.get("visual_context_mode"),
            "image_role_aliases": model_images.get("image_role_aliases") or {},
        }

    def _update_context_audit_for_model_step(
        self,
        context_audit: Dict[str, Any],
        step_mode: Dict[str, Any],
    ) -> None:
        context_audit["planner_step_mode"] = step_mode.get("planner_step_mode")
        context_audit["planner_authority"] = "qwen"
        context_audit["qwen_api_called"] = self.openclaw_model_provider == "qwen_api"
        context_audit["qwen_candidate_requested"] = True
        context_audit.setdefault("qwen_model_called", True)
        context_audit["qwen_provider"] = self.openclaw_model_provider
        context_audit["policy_backend"] = self.policy_backend
        context_audit["visual_memory_backend"] = "adapter_episode_local"
        context_audit["visual_memory_age_steps"] = step_mode.get(
            "visual_memory_age_steps"
        )
        context_audit["visual_memory_valid_until_step"] = step_mode.get(
            "visual_memory_valid_until_step"
        )
        context_audit["memory_context_used"] = bool(
            step_mode.get("memory_context_used")
        )
        context_audit["fast_break_reason"] = step_mode.get("fast_break_reason")
        context_audit["visual_memory_update_status"] = "not_applicable"
        context_audit["qwen_thinking_interval_steps"] = (
            self.qwen_thinking_interval_steps
        )
        context_audit["qwen_thinking_due"] = step_mode.get("qwen_thinking_due")
        effective_thinking_mode = step_mode.get("qwen_thinking_effective_mode")
        if effective_thinking_mode in {"auto", "off", "on"}:
            context_audit["qwen_thinking_enabled"] = (
                True
                if effective_thinking_mode == "on"
                else False
                if effective_thinking_mode == "off"
                else None
            )
        for key in (
            "input_regime",
            "map_assist_mode",
            "map_frame_interval_steps",
            "map_frame_due",
            "map_step_id",
            "map_available",
            "map_view_scope",
            "map_refresh_reason",
            "turn_loop_recovery_active",
            "visual_recovery_phase",
            "motion_feedback_enabled",
            "forward_stall_odometry_enabled",
            "map_collision_overlay_enabled",
        ):
            if step_mode.get(key) is not None:
                context_audit[key] = step_mode.get(key)

    def _update_visual_memory_cache(
        self,
        prompt_payload: Dict[str, Any],
        decision: Dict[str, Any],
        model_images: Dict[str, Any],
        step_mode: Dict[str, Any],
    ) -> str:
        if step_mode.get("planner_step_mode") != "visual_update":
            return "not_applicable"
        cache_key = step_mode.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            return "missing_cache_key"
        state = prompt_payload.get("state") or {}
        step_id = self._step_id(state if isinstance(state, dict) else {})
        arguments = decision.get("arguments") if isinstance(decision, dict) else {}
        if not isinstance(arguments, dict):
            arguments = {}
        summary = self._visual_summary_from_decision(decision, arguments)
        if not summary:
            self._visual_memory_cache[cache_key] = {
                "last_visual_step_id": step_id,
                "last_update_error": "missing_summary",
            }
            return "missing_summary"
        self._visual_memory_cache[cache_key] = {
            "last_visual_step_id": step_id,
            "last_visual_summary": summary,
            "last_suggested_subgoal": (
                arguments.get("suggested_subgoal")
                or arguments.get("active_subgoal")
                or arguments.get("subgoal")
                or ""
            ),
            "last_qwen_reason": decision.get("reason") or "",
            "last_action_text": arguments.get("action_text") or "",
            "valid_until_step": step_id + self.openclaw_model_image_interval_steps,
            "source_image_paths": list(model_images.get("paths") or []),
            "last_update_error": "",
        }
        return "updated"

    def _mark_visual_memory_error(
        self,
        prompt_payload: Dict[str, Any],
        step_mode: Dict[str, Any],
        error: str,
    ) -> str:
        if step_mode.get("planner_step_mode") != "visual_update":
            return "not_applicable"
        cache_key = step_mode.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            return "missing_cache_key"
        state = prompt_payload.get("state") or {}
        step_id = self._step_id(state if isinstance(state, dict) else {})
        self._visual_memory_cache[cache_key] = {
            "last_visual_step_id": step_id,
            "last_update_error": error or "visual_update_failed",
        }
        return "error"

    def _visual_summary_from_decision(
        self,
        decision: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> str:
        for key in ("visual_summary", "visual_observation", "caption", "summary"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        reason = decision.get("reason") if isinstance(decision, dict) else ""
        return str(reason).strip() if reason else ""

    def _visual_memory_cache_key(self, prompt_payload: Dict[str, Any]) -> str:
        state = prompt_payload.get("state") or {}
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(state, dict):
            state = {}
        if not isinstance(runtime_context, dict):
            runtime_context = {}
        return "|".join(
            [
                str(runtime_context.get("run_id") or ""),
                str(state.get("scene_id") or ""),
                str(state.get("episode_id") or ""),
            ]
        )

    def _visual_memory_age_steps(
        self,
        step_id: int,
        memory: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        if not isinstance(memory, dict) or memory.get("last_visual_step_id") is None:
            return None
        try:
            return max(0, step_id - int(memory.get("last_visual_step_id")))
        except (TypeError, ValueError):
            return None

    def _has_keyframe_candidate(self, runtime_context: Dict[str, Any]) -> bool:
        keyframe_candidate = runtime_context.get("keyframe_candidate")
        return bool(
            isinstance(keyframe_candidate, dict)
            and keyframe_candidate.get("image_path")
        )

    def _step_id(self, state: Dict[str, Any]) -> int:
        try:
            return int(state.get("step_id") or 0)
        except (TypeError, ValueError):
            return 0

    def _qwen_thinking_mode_for_step(
        self,
        prompt_payload: Dict[str, Any],
    ) -> Tuple[str, Optional[bool]]:
        if self.qwen_thinking_mode == "off":
            return "off", False
        if self.qwen_thinking_interval_steps == 1:
            return (
                self.qwen_thinking_mode,
                True if self.qwen_thinking_mode == "on" else None,
            )
        state = prompt_payload.get("state") or {}
        step_id = self._step_id(state if isinstance(state, dict) else {})
        thinking_due = step_id % self.qwen_thinking_interval_steps == 0
        return ("on" if thinking_due else "off"), thinking_due

    @staticmethod
    def _nonnegative_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _int_or_none(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _agent_session_id_for_payload(self, prompt_payload: Dict[str, Any]) -> str:
        state = prompt_payload.get("state") or {}
        if not isinstance(state, dict):
            state = {}
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            runtime_context = {}
        parts = [
            self.agent_session_id,
            runtime_context.get("run_id") or "",
            state.get("scene_id") or "scene",
            state.get("episode_id") or "episode",
            f"step_{state.get('step_id') if state.get('step_id') is not None else 'unknown'}",
            self.agent_session_timestamp,
        ]
        safe_parts = [self._safe_session_part(part) for part in parts]
        safe_parts = [part for part in safe_parts if part]
        return "-".join(safe_parts)

    @staticmethod
    def _safe_session_part(part: Any) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(part)).strip("_")
        return safe[:80]

    def _context_audit(
        self,
        prompt: str,
        usage: Dict[str, Any],
        session_id: str,
        assembled_prompt_tokens: Optional[int] = None,
        context_profile: str = "plan",
        session_mode: str = OPENCLAW_SESSION_MODE,
        model_images: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assembled = (
            int(assembled_prompt_tokens)
            if assembled_prompt_tokens is not None
            else self._estimate_tokens(prompt)
        )
        provider_input = self._usage_int(usage, "input")
        hidden_history = (
            provider_input - assembled if provider_input is not None else None
        )
        token_limit_exceeded = bool(
            assembled > QWEN_HARD_MAX_INPUT_TOKENS
            or (
                provider_input is not None
                and provider_input > QWEN_HARD_MAX_INPUT_TOKENS
            )
            or (
                provider_input is not None
                and self.agent_max_input_tokens > 0
                and provider_input >= self.agent_max_input_tokens
            )
        )
        qwen_hard_limit_exceeded = bool(
            assembled > QWEN_HARD_MAX_INPUT_TOKENS
            or (
                provider_input is not None
                and provider_input > QWEN_HARD_MAX_INPUT_TOKENS
            )
        )
        provider_usage_source = (
            "reported" if provider_input is not None else "unavailable"
        )
        audit = {
            "context_profile": context_profile,
            "assembled_prompt_tokens": assembled,
            "estimated_provider_input_tokens": assembled,
            "provider_input_tokens": provider_input,
            "provider_usage_source": provider_usage_source,
            "hidden_history_tokens_estimate": hidden_history,
            "max_total_tokens": PLAN_MAX_TOTAL_TOKENS,
            "agent_max_input_tokens": self.agent_max_input_tokens,
            "qwen_hard_max_input_tokens": QWEN_HARD_MAX_INPUT_TOKENS,
            "prompt_char_count": len(prompt),
            "profile_budget_exceeded": assembled > PLAN_MAX_TOTAL_TOKENS,
            "qwen_hard_limit_exceeded": qwen_hard_limit_exceeded,
            "token_limit_exceeded": token_limit_exceeded,
            "history_tokens": 0,
            "openclaw_session_mode": session_mode,
            "openclaw_session_id": session_id,
            "dynamic_visual_context_enabled": self.dynamic_visual_context_enabled,
            "qwen_thinking_enabled": (
                True
                if self.qwen_thinking_mode == "on"
                else False
                if self.qwen_thinking_mode == "off"
                else None
            ),
            "qwen_thinking_exercised": None,
            "qwen_thinking_interval_steps": self.qwen_thinking_interval_steps,
            "qwen_thinking_due": None,
            "thinking_model_supported": None,
            "qwen_output_json_valid": None,
            "qwen_output_schema": self.qwen_output_schema,
            "configured_model_id": self.openclaw_model or None,
            "configured_model_id_canonical": self._canonical_model_id(
                self.openclaw_model
            ),
            "returned_model_id": None,
            "returned_model_id_canonical": None,
            "qwen_transport_mode": self.qwen_transport_mode,
            "qwen_temperature": 0,
            "openclaw_model_max_images": self.openclaw_model_max_images,
            "model_image_count_policy": (
                "role_adaptive"
                if self._is_qwen_direct_policy() and self.dynamic_visual_context_enabled
                else "legacy_cap"
            ),
            "openclaw_model_max_images_applied": not (
                self._is_qwen_direct_policy() and self.dynamic_visual_context_enabled
            ),
            "openclaw_model_image_interval_steps": self.openclaw_model_image_interval_steps,
            "qwen_thinking_budget": self.qwen_thinking_budget,
        }
        if model_images is not None:
            image_paths = [
                str(path)
                for path in model_images.get("paths", [])
                if isinstance(path, str) and path
            ]
            missing_paths = [
                str(path)
                for path in model_images.get("missing_paths", [])
                if isinstance(path, str) and path
            ]
            audit["model_image_count"] = len(image_paths)
            audit["model_image_paths"] = image_paths
            audit["model_missing_image_paths"] = missing_paths
            for key in (
                "selected_image_count",
                "model_image_sources",
                "current_image_last",
                "history_frame_source",
                "map_view_image_count",
                "map_view_image_paths",
                "map_view_missing_image_paths",
                "map_image_hash",
                "retrieved_memory_image_count",
                "retrieved_memory_image_paths",
                "retrieved_memory_missing_image_paths",
                "recent_current_image_count",
                "recent_current_image_paths",
                "recent_current_missing_image_paths",
                "retrieved_memory_ids",
                "visual_context_mode",
                "requested_image_roles",
                "selected_image_roles",
                "missing_image_roles",
                "image_role_aliases",
                "map_age_steps",
                "map_reuse",
            ):
                if key in model_images:
                    audit[key] = model_images[key]
        return audit

    @staticmethod
    def _canonical_model_id(model: Any) -> Optional[str]:
        value = str(model or "").strip()
        if not value:
            return None
        return value.split("/", 1)[1] if value.startswith("qwen/") else value

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _usage_int(self, usage: Dict[str, Any], key: str) -> Optional[int]:
        if not isinstance(usage, dict) or key not in usage:
            return None
        try:
            return int(usage.get(key))
        except (TypeError, ValueError):
            return None

    def _enrich_write_memory_with_visual_observation(
        self,
        decision: Dict[str, Any],
        prompt_payload: Dict[str, Any],
    ) -> None:
        if decision.get("intent") != "write_memory":
            return
        arguments = decision.get("arguments")
        if not isinstance(arguments, dict):
            return
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            return
        observations = runtime_context.get("visual_observations") or []
        if not isinstance(observations, list):
            return
        image_path = str(
            arguments.get("image_path")
            or runtime_context.get("current_image_path")
            or ""
        )
        selected: Dict[str, Any] = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            if not selected:
                selected = observation
            if image_path and observation.get("image_path") == image_path:
                selected = observation
                break
        if not selected:
            return
        for key in (
            "image_path",
            "caption",
            "visual_observation",
            "objects",
            "landmarks",
            "spatial_cues",
            "navigation_relevance",
            "confidence",
        ):
            if key in selected and key not in arguments:
                arguments[key] = selected[key]

    def _prompt_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state") or {}
        prompt_state = self._copy_keys(state, PROMPT_STATE_KEYS)
        runtime_context = payload.get("runtime_context") or {}
        prompt_runtime_context = self._copy_keys(
            runtime_context,
            PROMPT_RUNTIME_CONTEXT_KEYS,
        )
        prompt_runtime_context.pop("map_context", None)
        recent_paths = prompt_runtime_context.get("recent_keyframe_paths")
        if isinstance(recent_paths, list):
            prompt_runtime_context["recent_keyframe_paths"] = recent_paths[
                -MAX_PROMPT_RECENT_KEYFRAME_PATHS:
            ]
        memory_context_text = prompt_runtime_context.get("memory_context_text")
        if isinstance(memory_context_text, str):
            prompt_runtime_context["memory_context_text"] = memory_context_text[
                :MAX_PROMPT_MEMORY_CONTEXT_CHARS
            ]
        memory_image_limit = (
            MAX_DIRECT_RETRIEVED_MEMORY_IMAGES
            if self._is_qwen_direct_policy()
            else MAX_PROMPT_MEMORY_IMAGES
        )
        memory_images = prompt_runtime_context.get("memory_images")
        if isinstance(memory_images, list):
            prompt_runtime_context["memory_images"] = [
                path
                for path in memory_images[:memory_image_limit]
                if isinstance(path, str) and path
            ]
        retrieved_image_paths = prompt_runtime_context.get(
            "retrieved_memory_image_paths"
        )
        if isinstance(retrieved_image_paths, list):
            prompt_runtime_context["retrieved_memory_image_paths"] = [
                path
                for path in retrieved_image_paths[:memory_image_limit]
                if isinstance(path, str) and path
            ]
        retrieved_images = prompt_runtime_context.get("retrieved_memory_images")
        if isinstance(retrieved_images, list):
            prompt_runtime_context["retrieved_memory_images"] = [
                self._copy_keys(item, ("memory_id", "image_path", "source", "step_id"))
                for item in retrieved_images[:memory_image_limit]
                if isinstance(item, dict) and item.get("image_path")
            ]
        task_state = prompt_runtime_context.get("task_state")
        if isinstance(task_state, dict):
            prompt_runtime_context["task_state"] = self._copy_keys(
                task_state,
                PROMPT_TASK_STATE_KEYS,
            )
        recent_step_summary = prompt_runtime_context.get("recent_step_summary")
        if isinstance(recent_step_summary, str):
            prompt_runtime_context["recent_step_summary"] = recent_step_summary[
                :MAX_PROMPT_RECENT_STEP_SUMMARY_CHARS
            ]
        retrieved_memory_ids = prompt_runtime_context.get("retrieved_memory_ids")
        if isinstance(retrieved_memory_ids, list):
            retrieved_memory_id_limit = (
                MAX_DIRECT_RETRIEVED_MEMORY_IMAGES
                if self._is_qwen_direct_policy()
                else MAX_PROMPT_RETRIEVED_MEMORY_IDS
            )
            prompt_runtime_context["retrieved_memory_ids"] = [
                str(memory_id)
                for memory_id in retrieved_memory_ids[:retrieved_memory_id_limit]
                if memory_id
            ]
        keyframe_candidate = runtime_context.get("keyframe_candidate")
        if isinstance(keyframe_candidate, dict):
            prompt_keyframe = self._copy_keys(keyframe_candidate, PROMPT_KEYFRAME_KEYS)
            if prompt_keyframe:
                prompt_runtime_context["keyframe_candidate"] = prompt_keyframe
        map_context = runtime_context.get("map_context")
        if isinstance(map_context, dict):
            prompt_map_context = self._copy_keys(
                map_context,
                PROMPT_MAP_CONTEXT_KEYS,
            )
            if prompt_map_context:
                prompt_runtime_context["map_context"] = prompt_map_context
            internal_map_context = map_context.get("internal_only")
            if isinstance(internal_map_context, dict):
                prompt_runtime_context["_map_internal_context"] = self._copy_keys(
                    internal_map_context,
                    ("map_image_path", "map_image_hash"),
                )
        if (
            self.openclaw_visual_mode == "describe"
            and prompt_runtime_context
            and self._should_analyze_visual(prompt_state, prompt_runtime_context)
        ):
            visual_observations = self._visual_observations(prompt_runtime_context)
            if visual_observations:
                prompt_runtime_context["visual_observations"] = visual_observations
        prompt_payload: Dict[str, Any] = {"state": prompt_state}
        if prompt_runtime_context:
            prompt_payload["runtime_context"] = prompt_runtime_context
        return prompt_payload

    def _copy_keys(self, data: Any, keys) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        copied: Dict[str, Any] = {}
        for key in keys:
            value = data.get(key)
            if value is not None and value != "":
                copied[key] = value
        return copied

    def _should_analyze_visual(
        self,
        state: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> bool:
        if self.openclaw_visual_interval_steps <= 1:
            return True
        keyframe_candidate = runtime_context.get("keyframe_candidate")
        if isinstance(keyframe_candidate, dict) and keyframe_candidate.get(
            "image_path"
        ):
            return True
        step_id = state.get("step_id")
        if not isinstance(step_id, int):
            try:
                step_id = int(step_id)
            except (TypeError, ValueError):
                return True
        return step_id % self.openclaw_visual_interval_steps == 0

    def _visual_observations(
        self, runtime_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if self.visual_analyzer is None:
            return []
        image_paths = self._visual_analysis_paths(runtime_context)
        if not image_paths:
            return []
        return self.visual_analyzer.analyze(image_paths)

    def _visual_context_lines(self, payload: Dict[str, Any]) -> List[str]:
        runtime_context = payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            return []
        deduped = self._visual_image_paths(runtime_context)
        if not deduped:
            return []
        return [
            "Visual context image paths: " + json.dumps(deduped, ensure_ascii=True),
        ]

    def _visual_image_paths(self, runtime_context: Dict[str, Any]) -> List[str]:
        current_image_path = runtime_context.get("current_image_path")
        recent_keyframe_paths = runtime_context.get("recent_keyframe_paths") or []
        image_paths: List[str] = []
        if isinstance(current_image_path, str) and current_image_path:
            image_paths.append(current_image_path)
        if isinstance(recent_keyframe_paths, list):
            image_paths.extend(
                path for path in recent_keyframe_paths if isinstance(path, str) and path
            )
        deduped: List[str] = []
        for path in image_paths:
            if path not in deduped:
                deduped.append(path)
        return deduped

    def _visual_analysis_paths(self, runtime_context: Dict[str, Any]) -> List[str]:
        image_paths = self._visual_image_paths(runtime_context)
        if self.openclaw_visual_max_images <= 0:
            return []
        if len(image_paths) <= self.openclaw_visual_max_images:
            return image_paths
        current_path = runtime_context.get("current_image_path")
        selected: List[str] = []
        if isinstance(current_path, str) and current_path:
            selected.append(current_path)
        remaining_slots = self.openclaw_visual_max_images - len(selected)
        if remaining_slots > 0:
            recent_paths = [path for path in image_paths if path not in selected]
            selected.extend(recent_paths[-remaining_slots:])
        return selected[: self.openclaw_visual_max_images]

    def _visual_analysis_metadata(
        self, prompt_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            return {}
        observations = runtime_context.get("visual_observations") or []
        if not isinstance(observations, list) or not observations:
            return {}
        image_paths: List[str] = []
        observation_summaries: List[Dict[str, Any]] = []
        vlm_latency_ms = 0.0
        cache_hits = 0
        failures = 0
        model = ""
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            summary = self._visual_observation_summary(observation)
            if summary:
                observation_summaries.append(summary)
            image_path = observation.get("image_path")
            if isinstance(image_path, str) and image_path:
                image_paths.append(image_path)
            metadata = observation.get("analysis_metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata.get("cache_hit"):
                cache_hits += 1
            else:
                vlm_latency_ms += float(metadata.get("vlm_latency_ms") or 0.0)
            if metadata.get("error") or observation.get("error"):
                failures += 1
            if not model and metadata.get("visual_model"):
                model = str(metadata.get("visual_model"))
        return {
            "ran": True,
            "num_images": len(image_paths),
            "image_paths": image_paths,
            "vlm_latency_ms": round(vlm_latency_ms, 3),
            "cache_hits": cache_hits,
            "failures": failures,
            "model": model,
            "observations": observation_summaries[: self.openclaw_visual_max_images],
        }

    def _visual_observation_summary(
        self, observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key in (
            "image_path",
            "caption",
            "visual_observation",
            "objects",
            "landmarks",
            "place_category",
            "spatial_cues",
            "navigation_relevance",
            "confidence",
        ):
            value = observation.get(key)
            if value not in (None, "", []):
                summary[key] = value
        return summary

    def _agent_visible_text(self, stdout: str) -> str:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout
        if not isinstance(data, dict):
            return stdout
        for key in ("finalAssistantVisibleText", "finalAssistantRawText", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        payloads = data.get("payloads")
        if isinstance(payloads, list):
            texts = [
                item.get("text", "")
                for item in payloads
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            joined = "\n".join(text for text in texts if text.strip())
            if joined.strip():
                return joined
        return stdout

    def _model_visible_text(self, stdout: str) -> str:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout
        if not isinstance(data, dict):
            return stdout
        outputs = data.get("outputs")
        if isinstance(outputs, list):
            texts = [
                item.get("text", "")
                for item in outputs
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            joined = "\n".join(text for text in texts if text.strip())
            if joined.strip():
                return joined
        for key in ("text", "finalAssistantVisibleText", "finalAssistantRawText"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return stdout

    def _model_request_metadata(self, stdout: str) -> Dict[str, Any]:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        metadata: Dict[str, Any] = {}
        request_attempts = data.get("request_attempts")
        retry_count = data.get("retry_count")
        if request_attempts is not None:
            try:
                metadata["qwen_api_request_attempts"] = int(request_attempts)
            except (TypeError, ValueError):
                pass
        if retry_count is not None:
            try:
                metadata["qwen_api_retry_count"] = int(retry_count)
            except (TypeError, ValueError):
                pass
        provider_metadata = data.get("provider_metadata")
        if isinstance(provider_metadata, dict):
            for key in (
                "qwen_thinking_enabled",
                "qwen_thinking_exercised",
                "thinking_model_supported",
                "thinking_capability_fallback",
                "thinking_transport_incompatible",
                "reasoning_tokens",
                "provider_latency_ms",
                "provider_error_detail",
                "provider_error_fingerprint",
                "provider_degradation_attempted",
                "provider_degradation_mode",
                "provider_degradation_trigger_detail",
                "provider_original_image_count",
                "provider_final_image_count",
            ):
                if key in provider_metadata:
                    metadata[key] = provider_metadata[key]
            returned_model_id = provider_metadata.get("returned_model_id")
            if isinstance(returned_model_id, str) and returned_model_id:
                metadata["returned_model_id"] = returned_model_id
                metadata["returned_model_id_canonical"] = self._canonical_model_id(
                    returned_model_id
                )
        return metadata

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise RuntimeError("openclaw agent did not return a JSON object")

    def _extract_route_v2_json_object(self, text: str) -> Dict[str, Any]:
        if len(text) > ROUTE_V2_MAX_RESPONSE_CHARS:
            raise RuntimeError(
                f"qwen route_v2 response exceeds {ROUTE_V2_MAX_RESPONSE_CHARS} characters"
            )
        stripped = text.strip()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "qwen route_v2 response must be exactly one JSON object"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError("qwen route_v2 response must be a JSON object")
        return value

    def _extract_route_v3_staged_json_object(self, text: str) -> Dict[str, Any]:
        if len(text) > ROUTE_V3_STAGED_MAX_RESPONSE_CHARS:
            raise RuntimeError(
                "qwen route_v3_staged response exceeds "
                f"{ROUTE_V3_STAGED_MAX_RESPONSE_CHARS} characters"
            )
        stripped = text.strip()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "qwen route_v3_staged response must be exactly one JSON object"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError("qwen route_v3_staged response must be a JSON object")
        return value

    @staticmethod
    def _strip_single_json_code_fence(text: str) -> Tuple[str, bool]:
        """Strip only an otherwise-whole single JSON code fence."""
        stripped = text.strip()
        lines = stripped.splitlines()
        if len(lines) < 3:
            return text, False
        if lines[0].strip().lower() not in {"```", "```json"}:
            return text, False
        if lines[-1].strip() != "```":
            return text, False
        candidate = "\n".join(lines[1:-1]).strip()
        if not candidate:
            return text, False
        return candidate, True

    def _extract_strict_route_json_object(self, text: str) -> Dict[str, Any]:
        if self.qwen_output_schema == "route_v3_staged":
            return self._extract_route_v3_staged_json_object(text)
        return self._extract_route_v2_json_object(text)

    def _route_v2_repair_prompt(self, model_text: str, error: Exception) -> str:
        candidate = model_text[:ROUTE_V2_MAX_RESPONSE_CHARS]
        return "\n".join(
            [
                "Repair the candidate into exactly one valid QwenDirectPolicy JSON object.",
                "Return JSON only. Do not include markdown, prose, chain-of-thought, or reasoning_content.",
                "Preserve the candidate action and navigation evidence when valid; only fix schema, types, boundaries, and overlong fields.",
                f"Validation error category: {self._route_v2_error_category(error)}.",
                ROUTE_V2_FIELD_LIMITS_TEXT,
                'Schema: {"action_text":"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT","confidence":0.0,"visual_summary":"short","progress_state":"short","route_stage":"start|en_route|intermediate_landmark|post_landmark_transition|approaching_target|verifying_target|complete|unknown","confirmed_landmarks":["short"],"current_target":"short","target_relation":"short","stop_evidence":"none|visible_goal|instruction_complete","semantic_stop_state":"not_ready|visible_not_reached|approaching_target|at_or_inside_target|beside_target|instruction_complete_at_target","reason":"short"}',
                "Candidate JSON text follows as a JSON-escaped string:",
                json.dumps(candidate, ensure_ascii=True),
            ]
        )

    def _strict_route_repair_prompt(
        self,
        model_text: str,
        error: Exception,
        *,
        expected_active_stage_id: str,
    ) -> str:
        if self.qwen_output_schema == "route_v2":
            return self._route_v2_repair_prompt(model_text, error)
        candidate = model_text[:ROUTE_V3_STAGED_MAX_RESPONSE_CHARS]
        expected_stage_line = (
            "Expected active_stage_id: "
            f"{json.dumps(expected_active_stage_id, ensure_ascii=True)}. "
            "Copy it exactly."
        )
        return "\n".join(
            [
                "Repair the candidate into exactly one valid route_v3_staged JSON object.",
                "Return JSON only. Do not include markdown, prose, chain-of-thought, or reasoning_content.",
                "Preserve the candidate action and grounded evidence when valid; only fix schema, types, and boundaries.",
                f"Validation error category: {self._strict_route_error_category(error)}.",
                expected_stage_line,
                ROUTE_V2_FIELD_LIMITS_TEXT,
                "active_stage_id<=64 chars; stage_evidence_refs<=6 items and each<=120 chars.",
                ROUTE_V3_STAGED_SCHEMA_LINE,
                "Candidate JSON text follows as a JSON-escaped string:",
                json.dumps(candidate, ensure_ascii=True),
            ]
        )

    @staticmethod
    def _sanitize_route_v2_repair_candidate(
        candidate: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        sanitized = dict(candidate)
        sanitized_fields: List[str] = []
        for key, limit in ROUTE_V2_STRING_LIMITS.items():
            value = sanitized.get(key)
            if isinstance(value, str) and len(value) > limit:
                sanitized[key] = value[:limit]
                sanitized_fields.append(key)
        landmarks = sanitized.get("confirmed_landmarks")
        if isinstance(landmarks, list):
            clipped_landmarks = [
                item[:ROUTE_V2_MAX_LANDMARK_CHARS] if isinstance(item, str) else item
                for item in landmarks[:ROUTE_V2_MAX_LANDMARKS]
            ]
            if clipped_landmarks != landmarks:
                sanitized["confirmed_landmarks"] = clipped_landmarks
                sanitized_fields.append("confirmed_landmarks")
        return sanitized, sorted(sanitized_fields)

    @staticmethod
    def _sanitize_route_v3_primary_text_candidate(
        candidate: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Apply conservative defaults while keeping action/stage control strict."""
        sanitized = dict(candidate)
        sanitized_fields: List[str] = []
        defaults = {
            "confidence": 0.0,
            "visual_summary": "",
            "progress_state": "",
            "route_stage": "unknown",
            "confirmed_landmarks": [],
            "current_target": "",
            "target_relation": "unknown",
            "stop_evidence": "none",
            "semantic_stop_state": "not_ready",
            "reason": ROUTE_V3_STAGED_LOCAL_REASON_DEFAULT,
        }
        for key, default in defaults.items():
            if key not in sanitized:
                sanitized[key] = default
                sanitized_fields.append(key)
        for key, limit in ROUTE_V2_STRING_LIMITS.items():
            value = sanitized.get(key)
            if isinstance(value, str) and len(value) > limit:
                sanitized[key] = value[:limit]
                sanitized_fields.append(key)
        return sanitized, sorted(sanitized_fields)

    def _sanitize_strict_route_repair_candidate(
        self,
        candidate: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        sanitized, sanitized_fields = self._sanitize_route_v2_repair_candidate(
            candidate
        )
        if self.qwen_output_schema != "route_v3_staged":
            return sanitized, sanitized_fields
        sanitized, local_fields = self._sanitize_route_v3_primary_text_candidate(
            sanitized
        )
        sanitized_fields.extend(local_fields)
        return sanitized, sorted(set(sanitized_fields))

    @staticmethod
    def _route_v2_error_category(error: Exception) -> str:
        message = str(error)
        categories = (
            ("exceeds", "response_too_long"),
            ("exactly one JSON object", "invalid_json_boundary"),
            ("must be a JSON object", "invalid_json_type"),
            ("fields mismatch", "schema_fields"),
            ("action_text", "invalid_action_text"),
            ("confidence", "invalid_confidence"),
            ("visual_summary", "invalid_visual_summary"),
            ("progress_state", "invalid_progress_state"),
            ("current_target", "invalid_current_target"),
            ("target_relation", "invalid_target_relation"),
            ("reason", "invalid_reason"),
            ("route_stage", "invalid_route_stage"),
            ("confirmed_landmarks", "invalid_confirmed_landmarks"),
            ("stop_evidence", "invalid_stop_evidence"),
            ("semantic_stop_state", "invalid_semantic_stop_state"),
        )
        for marker, category in categories:
            if marker in message:
                return category
        return "provider_or_unknown_error"

    def _strict_route_error_category(self, error: Exception) -> str:
        if self.qwen_output_schema == "route_v2":
            return self._route_v2_error_category(error)
        message = str(error)
        for marker, category in (
            ("active_stage_id mismatch", "active_stage_mismatch"),
            ("fields mismatch", "schema_fields"),
            ("stage_complete_candidate", "invalid_stage_complete_candidate"),
            ("stage_relation", "invalid_stage_relation"),
            ("stage_evidence_refs", "invalid_stage_evidence_refs"),
            ("response exceeds", "response_too_long"),
            ("exactly one JSON object", "invalid_json_boundary"),
        ):
            if marker in message:
                return category
        return self._route_v2_error_category(error)

    def _normalize_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if self._is_qwen_direct_policy():
            return self._normalize_direct_policy_decision(decision)
        intent = str(decision.get("intent") or "act")
        if intent not in INTENT_TOOL_NAMES:
            raise RuntimeError(f"openclaw agent returned unsupported intent: {intent}")
        arguments = decision.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise RuntimeError("openclaw agent returned non-object arguments")
        arguments = dict(arguments)
        action_text = self._normalize_action_text(arguments.get("action_text"))
        if action_text:
            arguments["action_text"] = action_text
        elif intent in {"act", "replan"}:
            inferred_action = self._infer_action_text(str(decision.get("reason") or ""))
            if inferred_action:
                arguments["action_text"] = inferred_action
        return {
            "intent": intent,
            "tool_name": str(decision.get("tool_name") or INTENT_TOOL_NAMES[intent]),
            "arguments": arguments,
            "reason": str(decision.get("reason") or "openclaw_agent"),
        }

    def _normalize_model_decision(
        self,
        decision: Dict[str, Any],
        prompt_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = self._normalize_decision(decision)
        if not (
            self._is_qwen_direct_policy()
            and self.qwen_output_schema == "route_v3_staged"
        ):
            return normalized
        runtime_context = prompt_payload.get("runtime_context")
        expected_stage_id = (
            str(runtime_context.get("active_stage_id") or "")
            if isinstance(runtime_context, dict)
            else ""
        )
        actual_stage_id = str(
            (normalized.get("arguments") or {}).get("active_stage_id") or ""
        )
        if not expected_stage_id or actual_stage_id != expected_stage_id:
            raise RuntimeError("qwen route_v3_staged active_stage_id mismatch")
        return normalized

    def _normalize_direct_policy_decision(
        self, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.qwen_output_schema == "route_v2":
            return self._normalize_route_v2_decision(decision)
        if self.qwen_output_schema == "route_v3_staged":
            return self._normalize_route_v3_staged_decision(decision)
        if "arguments" in decision and isinstance(decision.get("arguments"), dict):
            arguments = dict(decision.get("arguments") or {})
            reason = str(
                decision.get("reason") or arguments.get("reason") or "qwen_direct"
            )
        else:
            arguments = {
                key: decision.get(key)
                for key in (
                    "action_text",
                    "confidence",
                    "visual_summary",
                    "progress_state",
                    "stop_evidence",
                    "current_target",
                    "target_relation",
                    "semantic_stop_state",
                )
                if key in decision
            }
            reason = str(decision.get("reason") or "qwen_direct")
        action_text = self._normalize_action_text(arguments.get("action_text"))
        if not action_text:
            raise RuntimeError("qwen direct response missing supported action_text")
        arguments["action_text"] = action_text
        arguments["reason"] = reason
        return {
            "intent": "act",
            "tool_name": "QwenDirectPolicy",
            "arguments": arguments,
            "reason": reason,
        }

    def _normalize_route_v2_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        fields = set(decision)
        if fields != ROUTE_V2_REQUIRED_FIELDS:
            missing = sorted(ROUTE_V2_REQUIRED_FIELDS - fields)
            extra = sorted(fields - ROUTE_V2_REQUIRED_FIELDS)
            raise RuntimeError(
                f"qwen route_v2 fields mismatch missing={missing} extra={extra}"
            )
        action_text = decision.get("action_text")
        if action_text not in ACTION_TEXTS:
            raise RuntimeError("qwen route_v2 action_text is invalid")
        confidence = decision.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise RuntimeError("qwen route_v2 confidence is invalid")
        for key, limit in ROUTE_V2_STRING_LIMITS.items():
            value = decision.get(key)
            if not isinstance(value, str) or len(value) > limit:
                raise RuntimeError(f"qwen route_v2 {key} is invalid")
        if decision.get("route_stage") not in ROUTE_STAGES:
            raise RuntimeError("qwen route_v2 route_stage is invalid")
        landmarks = decision.get("confirmed_landmarks")
        if (
            not isinstance(landmarks, list)
            or len(landmarks) > ROUTE_V2_MAX_LANDMARKS
            or any(
                not isinstance(item, str) or len(item) > ROUTE_V2_MAX_LANDMARK_CHARS
                for item in landmarks
            )
        ):
            raise RuntimeError("qwen route_v2 confirmed_landmarks is invalid")
        if decision.get("stop_evidence") not in STOP_EVIDENCE_VALUES:
            raise RuntimeError("qwen route_v2 stop_evidence is invalid")
        if decision.get("semantic_stop_state") not in SEMANTIC_STOP_STATES:
            raise RuntimeError("qwen route_v2 semantic_stop_state is invalid")
        arguments = dict(decision)
        reason = arguments["reason"]
        return {
            "intent": "act",
            "tool_name": "QwenDirectPolicy",
            "arguments": arguments,
            "reason": reason,
        }

    def _normalize_route_v3_staged_decision(
        self,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        fields = set(decision)
        if fields != ROUTE_V3_STAGED_REQUIRED_FIELDS:
            missing = sorted(ROUTE_V3_STAGED_REQUIRED_FIELDS - fields)
            extra = sorted(fields - ROUTE_V3_STAGED_REQUIRED_FIELDS)
            raise RuntimeError(
                "qwen route_v3_staged fields mismatch "
                f"missing={missing} extra={extra}"
            )
        base_decision = {
            key: value
            for key, value in decision.items()
            if key in ROUTE_V2_REQUIRED_FIELDS
        }
        normalized = self._normalize_route_v2_decision(base_decision)
        active_stage_id = decision.get("active_stage_id")
        if (
            not isinstance(active_stage_id, str)
            or not active_stage_id
            or len(active_stage_id) > 64
        ):
            raise RuntimeError("qwen route_v3_staged active_stage_id is invalid")
        complete_candidate = decision.get("stage_complete_candidate")
        if not isinstance(complete_candidate, bool):
            raise RuntimeError(
                "qwen route_v3_staged stage_complete_candidate is invalid"
            )
        relation = decision.get("stage_relation")
        if relation not in ROUTE_V3_STAGED_RELATIONS:
            raise RuntimeError("qwen route_v3_staged stage_relation is invalid")
        evidence_refs = decision.get("stage_evidence_refs")
        if (
            not isinstance(evidence_refs, list)
            or len(evidence_refs) > ROUTE_V3_STAGED_MAX_EVIDENCE_REFS
            or any(
                not isinstance(reference, str)
                or not reference
                or len(reference) > ROUTE_V3_STAGED_MAX_EVIDENCE_REF_CHARS
                for reference in evidence_refs
            )
            or len(set(evidence_refs)) != len(evidence_refs)
        ):
            raise RuntimeError("qwen route_v3_staged stage_evidence_refs is invalid")
        normalized["arguments"].update(
            {
                "active_stage_id": active_stage_id,
                "stage_complete_candidate": complete_candidate,
                "stage_relation": relation,
                "stage_evidence_refs": list(evidence_refs),
            }
        )
        return normalized

    def _normalize_action_text(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        normalized = ACTION_ALIASES.get(normalized, normalized)
        return normalized if normalized in ACTION_TEXTS else ""

    def _infer_action_text(self, text: str) -> str:
        upper_text = text.upper()
        patterns = [
            r"\bMUST\s+(?:BE\s+)?(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\b",
            r"\b(?:ONLY|CORRECT)\s+(?:ACTION|COMMAND)\s+(?:IS\s+)?(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\b",
            r"\b(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\s+(?:IS\s+)?(?:THE\s+)?ONLY\b",
            r"\b(?:NEXT\s+ACTION|ACTION|COMMAND)\s*(?:IS|:)\s*(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, upper_text)
            if match:
                return self._normalize_action_text(match.group(1))
        return ""

    def _gateway_health(self) -> Dict[str, Any]:
        args = self._openclaw_args("gateway", "call", "health", "--json")
        if self.gateway_url:
            args.extend(["--url", self.gateway_url])
        args.extend(["--timeout", str(int(self.timeout_s * 1000))])
        result = self.run_openclaw(args, self.timeout_s)
        if result.returncode != 0:
            message = (
                result.stderr or result.stdout or "openclaw gateway health failed"
            ).strip()
            raise RuntimeError(message)
        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("openclaw gateway health returned invalid JSON") from exc
        if not isinstance(data, dict) or not data.get("ok"):
            raise RuntimeError("openclaw gateway health is not ok")
        return data

    def _openclaw_args(self, *args: str) -> List[str]:
        command = ["openclaw"]
        if self.openclaw_profile:
            command.extend(["--profile", self.openclaw_profile])
        command.extend(args)
        return command

    def _memory_recall(
        self,
        instruction: str,
        step_id: int,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "intent": "recall_memory",
            "tool_name": "MemoryQuerySkill",
            "arguments": {
                "text": instruction,
                "step_id": step_id,
                "reason": reason,
            },
            "reason": reason,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP /plan adapter backed by the OpenClaw CLI WebSocket gateway"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--recall_interval_steps", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--openclaw_gateway_url", default="")
    parser.add_argument(
        "--planner_mode", choices=("heuristic", "agent", "model"), default="heuristic"
    )
    parser.add_argument(
        "--policy_backend",
        choices=(JANUS_POLICY_BACKEND, QWEN_DIRECT_POLICY_BACKEND),
        default=JANUS_POLICY_BACKEND,
    )
    parser.add_argument("--agent_id", default="main")
    parser.add_argument("--agent_timeout", type=float, default=60.0)
    parser.add_argument("--agent_max_input_tokens", type=int, default=10000)
    parser.add_argument("--openclaw_profile", default="")
    parser.add_argument("--openclaw_model", default="")
    parser.add_argument(
        "--openclaw_model_provider",
        choices=("qwen_api", "openclaw_cli"),
        default="qwen_api",
    )
    parser.add_argument("--openclaw_model_max_images", type=int, default=3)
    parser.add_argument("--openclaw_model_image_interval_steps", type=int, default=20)
    parser.add_argument("--openclaw_model_fast_mode", default="qwen_text_only")
    parser.add_argument("--openclaw_model_fast_use_memory_context", type=int, default=1)
    parser.add_argument(
        "--dynamic_visual_context_enabled", type=int, choices=(0, 1), default=0
    )
    parser.add_argument(
        "--qwen_thinking_mode",
        choices=("auto", "off", "on"),
        default="auto",
    )
    parser.add_argument("--qwen_thinking_interval_steps", type=int, default=1)
    parser.add_argument(
        "--qwen_output_schema",
        choices=("legacy", "route_v2", "route_v3_staged"),
        default="legacy",
    )
    parser.add_argument("--segmentation_timeout", type=float, default=120.0)
    parser.add_argument("--stage_plan_manifest_path", default="")
    parser.add_argument("--qwen_thinking_budget", type=int, default=None)
    parser.add_argument(
        "--qwen_transport_mode",
        choices=("sync",),
        default="sync",
    )
    parser.add_argument("--agent_session_id", default="")
    parser.add_argument(
        "--openclaw_visual_mode", choices=("path", "describe"), default="path"
    )
    parser.add_argument("--openclaw_visual_max_images", type=int, default=2)
    parser.add_argument("--openclaw_visual_interval_steps", type=int, default=1)
    parser.add_argument("--openclaw_visual_timeout_ms", type=int, default=30000)
    parser.add_argument("--openclaw_visual_model", default="")
    args = parser.parse_args()

    planner = OpenClawCliPlanPlanner(
        recall_interval_steps=args.recall_interval_steps,
        timeout_s=args.timeout,
        gateway_url=args.openclaw_gateway_url,
        planner_mode=args.planner_mode,
        policy_backend=args.policy_backend,
        agent_id=args.agent_id,
        agent_timeout_s=args.agent_timeout,
        agent_max_input_tokens=args.agent_max_input_tokens,
        openclaw_profile=args.openclaw_profile,
        openclaw_model=args.openclaw_model,
        openclaw_model_provider=args.openclaw_model_provider,
        openclaw_model_max_images=args.openclaw_model_max_images,
        openclaw_model_image_interval_steps=args.openclaw_model_image_interval_steps,
        openclaw_model_fast_mode=args.openclaw_model_fast_mode,
        openclaw_model_fast_use_memory_context=bool(
            args.openclaw_model_fast_use_memory_context
        ),
        dynamic_visual_context_enabled=bool(args.dynamic_visual_context_enabled),
        qwen_thinking_mode=args.qwen_thinking_mode,
        qwen_thinking_interval_steps=args.qwen_thinking_interval_steps,
        qwen_output_schema=args.qwen_output_schema,
        qwen_thinking_budget=args.qwen_thinking_budget,
        qwen_transport_mode=args.qwen_transport_mode,
        segmentation_timeout_s=args.segmentation_timeout,
        stage_plan_manifest_path=args.stage_plan_manifest_path,
        agent_session_id=args.agent_session_id,
        openclaw_visual_mode=args.openclaw_visual_mode,
        openclaw_visual_max_images=args.openclaw_visual_max_images,
        openclaw_visual_interval_steps=args.openclaw_visual_interval_steps,
        openclaw_visual_timeout_ms=args.openclaw_visual_timeout_ms,
        openclaw_visual_model=args.openclaw_visual_model,
    )
    server = make_gateway_server(args.host, args.port, planner)
    print(
        f"OpenClaw CLI /plan adapter listening on http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
