import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import requests

from harness.openclaw.gateway_server import make_gateway_server
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
    "blocked_stop_feedback",
    "forward_stall_feedback",
    "task_state",
    "recent_step_summary",
    "retrieved_memory_ids",
)
PROMPT_KEYFRAME_KEYS = ("step_id", "reason", "image_path")
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
DIRECT_PROMPT_OMIT_KEYS = {
    "run_id",
    "current_image_path",
    "recent_keyframe_paths",
    "memory_images",
    "retrieved_memory_image_paths",
    "image_path",
}


class QwenApiModelClient:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        run_openclaw: Optional[OpenClawRunner] = None,
        openclaw_profile: str = "",
        max_retries: Optional[int] = None,
        retry_backoff_s: Optional[float] = None,
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

    def run(
        self,
        prompt: str,
        image_paths: List[str],
        model: str,
        timeout_s: float,
    ) -> Dict[str, Any]:
        self._ensure_config(timeout_s)
        if not self.api_key:
            raise RuntimeError(
                "Qwen API key is not configured; set OPENCLAW_QWEN_API_KEY, "
                "DASHSCOPE_API_KEY, QWEN_API_KEY, or configure OpenClaw provider qwen"
            )
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        request_json = {
            "model": self._api_model_name(model),
            "messages": [
                {
                    "role": "user",
                    "content": self._message_content(prompt, image_paths),
                }
            ],
            "temperature": 0,
        }
        attempts = self.max_retries + 1
        last_error = ""
        for attempt_index in range(attempts):
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
                last_error = str(exc)
                if attempt_index >= self.max_retries:
                    raise RuntimeError(
                        "Qwen API request failed after "
                        f"{attempt_index + 1} attempt(s): {last_error}"
                    ) from exc
                self._sleep_before_retry(attempt_index)
                continue
            if response.status_code >= 400:
                message = f"HTTP {response.status_code} {response.text}"
                last_error = message
                if self._retryable_status(response.status_code) and attempt_index < self.max_retries:
                    self._sleep_before_retry(attempt_index)
                    continue
                raise RuntimeError(f"Qwen API request failed: {message}")
            normalized = self._normalize_response(response.json())
            normalized["request_attempts"] = attempt_index + 1
            normalized["retry_count"] = attempt_index
            return normalized
        raise RuntimeError(f"Qwen API request failed after {attempts} attempt(s): {last_error}")

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
        self.base_url = str(provider.get("baseUrl") or provider.get("base_url") or self.base_url)
        api_key = provider.get("apiKey") or provider.get("api_key") or provider.get("key") or ""
        self.api_key = self._resolve_secret_value(str(api_key)) if api_key else self.api_key
        if not self.api_key:
            self.api_key = self._openclaw_auth_store_api_key()

    @staticmethod
    def _openclaw_auth_store_api_key() -> str:
        store_path = Path(
            os.environ.get(
                "OPENCLAW_AUTH_PROFILES_PATH",
                str(Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"),
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
        key = profile.get("key") or profile.get("apiKey") or profile.get("api_key") or ""
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

    def _message_content(self, prompt: str, image_paths: List[str]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for image_path in image_paths:
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
    def _normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
        text = ""
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
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
        if isinstance(usage, dict):
            normalized_usage = {
                "input": usage.get("prompt_tokens") or usage.get("input") or usage.get("input_tokens"),
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
            normalized_usage = {key: value for key, value in normalized_usage.items() if value is not None}
        return {
            "ok": True,
            "outputs": [{"text": text}],
            "usage": normalized_usage,
            "raw": data,
        }


def run_openclaw_command(args: List[str], timeout_s: float) -> subprocess.CompletedProcess:
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
        self.openclaw_model_fast_use_memory_context = bool(openclaw_model_fast_use_memory_context)
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
        }

    def _timeout_budget_payload(self) -> Dict[str, Any]:
        uses_direct_qwen = (
            self.planner_mode == "model" and self.openclaw_model_provider == "qwen_api"
        )
        qwen_retries = self._qwen_api_retries() if uses_direct_qwen else 0
        qwen_retry_backoff_s = self._qwen_api_retry_backoff_s() if uses_direct_qwen else 0.0
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
        self._apply_model_step_context(prompt_payload, step_mode)
        model_images = self._model_image_files(prompt_payload, step_mode["planner_step_mode"])
        self._attach_direct_model_image_order_context(
            prompt_payload,
            model_images,
            step_mode["planner_step_mode"],
        )
        prompt = self._model_prompt_from_prompt_payload(prompt_payload)
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
            model_stdout = self._model_run_stdout(prompt, model_images)
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
            context_audit["visual_memory_update_status"] = self._mark_visual_memory_error(
                prompt_payload,
                step_mode,
                str(exc),
            )
            if self._is_qwen_direct_policy():
                return self._direct_policy_failure_decision(
                    str(exc),
                    context_audit=context_audit,
                    reason="qwen_direct_provider_error",
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
        model_text = self._model_visible_text(model_stdout)
        try:
            decision = self._extract_json_object(model_text)
            normalized = self._normalize_decision(decision)
        except RuntimeError as exc:
            context_audit["visual_memory_update_status"] = self._mark_visual_memory_error(
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
                "fallback_policy": "hard_failure_stop",
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
            "arguments": self._memory_guided_policy_arguments(prompt_payload, step_mode),
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
            arguments["memory_context_text"] = "\n".join(lines)[:MAX_PROMPT_MEMORY_CONTEXT_CHARS]
        return arguments

    def _model_run_stdout(self, prompt: str, model_images: Dict[str, Any]) -> str:
        image_paths = list(model_images.get("paths") or [])
        if self.openclaw_model_provider == "qwen_api":
            if self.model_client is None:
                self.model_client = QwenApiModelClient(
                    run_openclaw=self.run_openclaw,
                    openclaw_profile=self.openclaw_profile,
                )
            response = self.model_client.run(
                prompt=prompt,
                image_paths=image_paths,
                model=self.openclaw_model,
                timeout_s=self.agent_timeout_s,
            )
            return json.dumps(response, ensure_ascii=True)

        args = self._openclaw_args("capability", "model", "run", "--json")
        if self.openclaw_model:
            args.extend(["--model", self.openclaw_model])
        for image_path in image_paths:
            args.extend(["--file", image_path])
        args.extend(["--prompt", prompt])
        result = self.run_openclaw(args, self.agent_timeout_s)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "openclaw model run failed").strip()
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
            message = (result.stderr or result.stdout or "openclaw agent failed").strip()
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
            return self._agent_token_limit_decision(token_guard, context_audit=context_audit)
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
        max_input_tokens = int(token_guard.get("max_input_tokens") or self.agent_max_input_tokens)
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
                "context_audit": context_audit or token_guard.get("context_audit") or {},
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

    def _model_prompt_from_prompt_payload(self, prompt_payload: Dict[str, Any]) -> str:
        if self._is_qwen_direct_policy():
            return self._direct_model_prompt_from_prompt_payload(prompt_payload)
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

    def _direct_model_prompt_from_prompt_payload(self, prompt_payload: Dict[str, Any]) -> str:
        compact_payload = json.dumps(
            self._direct_model_prompt_payload(prompt_payload),
            ensure_ascii=True,
            sort_keys=True,
        )
        runtime_context = prompt_payload.get("runtime_context") or {}
        planner_step_mode = ""
        if isinstance(runtime_context, dict):
            planner_step_mode = str(runtime_context.get("planner_step_mode") or "")
        image_instruction = (
            "No image is attached for this step. Use compact cached visual and memory evidence only."
            if planner_step_mode == "fast_text"
            else (
                "Attached images are ordered as retrieved history memory first, then recent_current "
                "frames from oldest to newest, with the final image as the current t observation. "
                "Base the action on the final current image and use earlier images only for progress context."
            )
        )
        return "\n".join(
            [
                "Return one JSON object for QwenDirectPolicy.",
                "QwenDirectPolicy is the only action-producing policy in this mode.",
                'Schema: {"action_text":"STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT","confidence":0.0,"visual_summary":"short","progress_state":"short","stop_evidence":"none|visible_goal|instruction_complete","current_target":"short","target_relation":"short","semantic_stop_state":"not_ready|visible_not_reached|approaching_target|at_or_inside_target|beside_target|instruction_complete_at_target","reason":"short"}',
                "Do not name external tools. Do not emit markdown.",
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
                "When blocked_stop_feedback is present, do not choose STOP. Use MOVE_FORWARD only when the current image clearly shows the route continues forward; otherwise use TURN_LEFT or TURN_RIGHT to recheck alignment or target evidence.",
                "When forward_stall_feedback is present, do not choose MOVE_FORWARD; choose only TURN_LEFT or TURN_RIGHT and explain the corrective visual reason.",
                image_instruction,
                "Payload:",
                compact_payload,
            ]
        )

    def _direct_model_prompt_payload(self, prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
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
                self._sanitize_direct_prompt_value(item)
                for item in value
            ]
            return [
                item
                for item in sanitized_items
                if item not in ({}, [], None, "")
            ]
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
        return {
            "planner_step_mode": mode,
            "fast_break_reason": reason,
            "cache_key": cache_key,
            "visual_memory": memory or {},
            "visual_memory_age_steps": age_steps,
            "visual_memory_valid_until_step": (
                memory.get("valid_until_step")
                if isinstance(memory, dict)
                else None
            ),
            "memory_context_used": bool(
                mode == "fast_text"
                and self.openclaw_model_fast_use_memory_context
                and isinstance(memory, dict)
                and memory.get("last_visual_summary")
            ),
        }

    def _force_visual_refresh_requested(self, runtime_context: Dict[str, Any]) -> bool:
        if runtime_context.get("force_visual_refresh") is True:
            return True
        control_context = runtime_context.get("control_context")
        return isinstance(control_context, dict) and control_context.get("force_visual_refresh") is True

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
        runtime_context["visual_memory_age_steps"] = step_mode.get("visual_memory_age_steps")
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
        runtime_context["model_visual_context_note"] = (
            "No image is attached for this step. Use cached visual memory from the last visual_update."
        )
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
        if not isinstance(runtime_context, dict) or self.openclaw_model_max_images <= 0:
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
        return model_images

    def _model_visual_update_image_paths(self, runtime_context: Dict[str, Any]) -> List[str]:
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
        if isinstance(keyframe_candidate, dict) and path == keyframe_candidate.get("image_path"):
            return "keyframe_candidate"
        return "recent_keyframe"

    def _qwen_direct_model_image_candidates(
        self,
        runtime_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        current_image_path = runtime_context.get("current_image_path")
        current_path = current_image_path if isinstance(current_image_path, str) else ""
        current_budget = 1 if current_path else 0
        recent_candidates = self._recent_current_frame_candidates(
            current_path,
            limit=MAX_DIRECT_RECENT_CURRENT_IMAGES,
        )
        recent_paths = {
            str(candidate.get("path") or "")
            for candidate in recent_candidates
            if candidate.get("path")
        }
        history_budget = max(0, self.openclaw_model_max_images - current_budget)
        history_budget = min(MAX_DIRECT_RETRIEVED_MEMORY_IMAGES, history_budget)
        memory_candidates = self._retrieved_memory_image_candidates(
            runtime_context,
            current_path=current_path,
            exclude_paths=recent_paths,
        )[:history_budget]
        recent_budget = max(
            0,
            self.openclaw_model_max_images - len(memory_candidates) - current_budget,
        )
        if recent_budget < len(recent_candidates):
            recent_candidates = recent_candidates[-recent_budget:] if recent_budget else []
        candidates = list(memory_candidates)
        candidates.extend(recent_candidates[:recent_budget])
        if current_path:
            candidates.append({"path": current_path, "source": "current"})
        return self._dedupe_model_image_candidates(candidates)

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
                if (current_path and image_path == current_path) or image_path in excluded:
                    continue
                candidates.append(
                    {
                        "path": image_path,
                        "source": str(item.get("source") or "openclaw_retrieved_memory"),
                        "memory_id": str(
                            item.get("memory_id")
                            or item.get("keyframe_id")
                            or ""
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
                if (current_path and image_path == current_path) or image_path in excluded:
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

        retrieved = [
            candidate
            for candidate in selected_candidates
            if candidate.get("source") not in ("current", "recent_current")
        ]
        missing_retrieved = [
            candidate
            for candidate in missing_candidates
            if candidate.get("source") not in ("current", "recent_current")
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
        metadata["retrieved_memory_image_count"] = len(retrieved)
        metadata["retrieved_memory_image_paths"] = [
            str(candidate.get("path") or "")
            for candidate in retrieved
        ]
        metadata["retrieved_memory_missing_image_paths"] = [
            str(candidate.get("path") or "")
            for candidate in missing_retrieved
        ]
        metadata["recent_current_image_count"] = len(recent_current)
        metadata["recent_current_image_paths"] = [
            str(candidate.get("path") or "")
            for candidate in recent_current
        ]
        metadata["recent_current_missing_image_paths"] = [
            str(candidate.get("path") or "")
            for candidate in missing_recent_current
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
                "retrieved_history_first_then_recent_current_oldest_to_newest_current_last"
            ),
            "current_image_last": bool(model_images.get("current_image_last")),
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
        context_audit["visual_memory_age_steps"] = step_mode.get("visual_memory_age_steps")
        context_audit["visual_memory_valid_until_step"] = step_mode.get(
            "visual_memory_valid_until_step"
        )
        context_audit["memory_context_used"] = bool(step_mode.get("memory_context_used"))
        context_audit["fast_break_reason"] = step_mode.get("fast_break_reason")
        context_audit["visual_memory_update_status"] = "not_applicable"

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
            provider_input - assembled
            if provider_input is not None
            else None
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
        provider_usage_source = "reported" if provider_input is not None else "unavailable"
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
                "retrieved_memory_image_count",
                "retrieved_memory_image_paths",
                "retrieved_memory_missing_image_paths",
                "recent_current_image_count",
                "recent_current_image_paths",
                "recent_current_missing_image_paths",
                "retrieved_memory_ids",
            ):
                if key in model_images:
                    audit[key] = model_images[key]
        return audit

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
        image_path = str(arguments.get("image_path") or runtime_context.get("current_image_path") or "")
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
        retrieved_image_paths = prompt_runtime_context.get("retrieved_memory_image_paths")
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
        if isinstance(keyframe_candidate, dict) and keyframe_candidate.get("image_path"):
            return True
        step_id = state.get("step_id")
        if not isinstance(step_id, int):
            try:
                step_id = int(step_id)
            except (TypeError, ValueError):
                return True
        return step_id % self.openclaw_visual_interval_steps == 0

    def _visual_observations(self, runtime_context: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            recent_paths = [
                path
                for path in image_paths
                if path not in selected
            ]
            selected.extend(recent_paths[-remaining_slots:])
        return selected[: self.openclaw_visual_max_images]

    def _visual_analysis_metadata(self, prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
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

    def _visual_observation_summary(self, observation: Dict[str, Any]) -> Dict[str, Any]:
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

    def _normalize_direct_policy_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if "arguments" in decision and isinstance(decision.get("arguments"), dict):
            arguments = dict(decision.get("arguments") or {})
            reason = str(decision.get("reason") or arguments.get("reason") or "qwen_direct")
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
        return {
            "intent": "act",
            "tool_name": "QwenDirectPolicy",
            "arguments": arguments,
            "reason": reason,
        }

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
            message = (result.stderr or result.stdout or "openclaw gateway health failed").strip()
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
    parser.add_argument("--planner_mode", choices=("heuristic", "agent", "model"), default="heuristic")
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
    parser.add_argument("--agent_session_id", default="")
    parser.add_argument("--openclaw_visual_mode", choices=("path", "describe"), default="path")
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
        openclaw_model_fast_use_memory_context=bool(args.openclaw_model_fast_use_memory_context),
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
