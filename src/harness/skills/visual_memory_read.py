import json
import os
import re
from typing import Any, Dict, List, Optional

from harness.skills.base import Skill
from harness.types import SkillResult, VLNState
from harness.visual_readback.readback import (
    build_readback_request,
    normalize_visual_readback_response,
    validate_readback_images,
)


class NoopVisualReadbackAdapter:
    def read(self, request: Dict[str, Any]) -> Dict[str, Any]:
        del request
        return {
            "verifier_labels": ["insufficient_evidence"],
            "readback_confidence": 0.0,
            "verifier_confidence": 0.0,
            "visual_grounding_status": "unverified",
        }


class QwenDirectVisualReadbackAdapter:
    def __init__(
        self,
        model_client: Any = None,
        model: str = "",
        timeout_ms: Optional[int] = None,
    ) -> None:
        self.model_client = model_client
        self.model = model or os.environ.get(
            "OPENCLAW_VISUAL_READBACK_MODEL",
            os.environ.get("OPENCLAW_VISUAL_MODEL", "qwen/qwen3.5-flash"),
        )
        self.timeout_ms = int(
            timeout_ms
            if timeout_ms is not None
            else os.environ.get("OPENCLAW_VISUAL_READBACK_TIMEOUT_MS", "90000")
        )

    def read(self, request: Dict[str, Any]) -> Dict[str, Any]:
        image_paths = list(request.get("actually_read_image_paths") or [])
        memory_ids = list(request.get("attached_memory_ids") or [])
        try:
            response = self._client().run(
                prompt=self._prompt(request, memory_ids),
                image_paths=image_paths,
                model=self.model,
                timeout_s=max(1.0, self.timeout_ms / 1000.0),
            )
            text = self._response_text(response)
            parsed = self._extract_json_object(text)
            if parsed is not None:
                parsed = self._normalize_parsed_payload(parsed)
                parsed.setdefault("retrieval_confidence", request.get("retrieval_confidence", 0.0))
                parsed.setdefault(
                    "grounding_eval_protocol",
                    {
                        "adapter": "qwen_direct_api",
                        "model": self.model,
                    },
                )
                return parsed
            return self._fallback_payload(
                request,
                visual_evidence=text,
                error_type="parse_failure",
            )
        except Exception as exc:
            return self._fallback_payload(
                request,
                visual_evidence="",
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def _client(self) -> Any:
        if self.model_client is None:
            from harness.openclaw.openclaw_cli_plan_gateway import QwenApiModelClient

            self.model_client = QwenApiModelClient()
        return self.model_client

    def _prompt(self, request: Dict[str, Any], memory_ids: List[str]) -> str:
        memory_id_text = ", ".join(memory_ids) if memory_ids else "<none>"
        expected_memory_ids_json = json.dumps(memory_ids)
        image_count = 1 + len(memory_ids)
        return (
            "You are a visual memory readback verifier for embodied navigation. "
            f"There are exactly {image_count} attached images.\n"
            "Image 1 is the current view. Images 2..N are retrieved memory images "
            "in the same order as attached_memory_ids.\n"
            f"candidate_action={request.get('candidate_action', '')}\n"
            f"trigger_rule={request.get('trigger_rule', '')}\n"
            f"instruction={request.get('instruction', '')}\n"
            f"attached_memory_ids={memory_id_text}\n"
            "Important: matched_memory_ids means memory image IDs that you actually "
            "read/used. It does not mean the visual content matches the current view.\n"
            "If every memory image is visible and used, set "
            f'"matched_memory_ids": {expected_memory_ids_json}. '
            "If a memory image is unreadable, omit only that image's ID.\n"
            "Use verifier_labels from this closed set only: route_conflict, "
            "goal_not_visible, insufficient_evidence. If current and memory views "
            "show conflicting landmarks/layouts, use route_conflict.\n"
            "Return only compact JSON with keys: verifier_labels, matched_memory_ids, "
            "visual_evidence, audit_action_hint, readback_confidence, "
            "verifier_confidence, visual_grounding_status."
            )

    @staticmethod
    def _normalize_parsed_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload)
        status = str(normalized.get("visual_grounding_status") or "").strip().lower()
        if status in {
            "mismatch",
            "conflict",
            "conflict_detected",
            "different",
            "difference",
        }:
            normalized["visual_grounding_status"] = "grounded"
        return normalized

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if not isinstance(response, dict):
            return ""
        outputs = response.get("outputs")
        if isinstance(outputs, list):
            texts = [
                item.get("text", "")
                for item in outputs
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            joined = "\n".join(text for text in texts if text.strip())
            if joined.strip():
                return joined
        raw = response.get("raw")
        if isinstance(raw, dict):
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return str(message["content"])
        for key in ("text", "finalAssistantVisibleText", "finalAssistantRawText"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            data = json.loads(stripped)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None

    @staticmethod
    def _fallback_payload(
        request: Dict[str, Any],
        *,
        visual_evidence: str,
        error_type: str,
        error: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "verifier_labels": ["insufficient_evidence"],
            "matched_memory_ids": [],
            "visual_evidence": visual_evidence,
            "readback_confidence": 0.0,
            "retrieval_confidence": request.get("retrieval_confidence", 0.0),
            "verifier_confidence": 0.0,
            "visual_grounding_status": "unverified",
            "grounding_eval_protocol": {
                "adapter": "qwen_direct_api",
                "error_type": error_type,
            },
        }
        if error:
            payload["grounding_eval_protocol"]["error"] = error
        return payload


class VisualMemoryReadSkill(Skill):
    name = "VisualMemoryReadSkill"
    description = "Read current and retrieved memory images into structured verifier labels."
    input_schema = {
        "type": "object",
        "properties": {
            "current_image_path": {"type": "string"},
            "memory_hits": {"type": "array"},
            "candidate_action": {"type": "string"},
            "trigger_rule": {"type": "string"},
        },
        "required": ["current_image_path", "candidate_action", "trigger_rule"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "read_status": {"type": "string"},
            "actually_read_image_paths": {"type": "array"},
            "model_image_count": {"type": "integer"},
            "verifier_labels": {"type": "array"},
            "matched_memory_ids": {"type": "array"},
            "readback_confidence": {"type": "number"},
            "verifier_confidence": {"type": "number"},
        },
    }
    oracle_safe = True

    def __init__(self, adapter: Any = None) -> None:
        self.adapter = adapter or QwenDirectVisualReadbackAdapter()

    def run(self, state: VLNState, payload: Dict[str, Any]) -> SkillResult:
        request = build_readback_request(
            current_image_path=str(payload.get("current_image_path") or ""),
            memory_hits=payload.get("memory_hits") or [],
            candidate_action=str(payload.get("candidate_action") or ""),
            trigger_rule=str(payload.get("trigger_rule") or ""),
            instruction=getattr(state, "instruction", ""),
        )
        missing_reason = validate_readback_images(request)
        if missing_reason:
            result = normalize_visual_readback_response("{}", request)
            result["read_status"] = "failed"
            result["error_type"] = missing_reason
            result["verifier_labels"] = ["insufficient_evidence"]
            result["readback_confidence"] = 0.0
            result["verifier_confidence"] = 0.0
            return SkillResult.ok_result("visual_memory_read", result, confidence=0.0)

        response = self.adapter.read(request)
        result = normalize_visual_readback_response(response, request)
        return SkillResult.ok_result(
            "visual_memory_read",
            result,
            confidence=float(result.get("readback_confidence") or 0.0),
        )
