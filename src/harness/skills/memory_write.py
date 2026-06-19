from typing import Any, Dict, Optional, Set

from harness.skills.base import Skill
from harness.types import SkillResult


DEFAULT_ALLOWED_MEMORY_SOURCES = {
    "episode-local",
    "scene-prior",
    "train-scene-only",
}


class MemoryWriteSkill(Skill):
    name = "MemoryWriteSkill"
    description = "Write approved keyframe or semantic memory records to the configured memory backend."
    input_schema = {
        "type": "object",
        "properties": {
            "should_write": {"type": "boolean"},
            "memory_source": {"type": "string"},
            "write_type": {"type": "string"},
            "step_id": {"type": "integer"},
            "scene_id": {"type": "string"},
            "episode_id": {"type": "string"},
            "image_path": {"type": "string"},
            "note": {"type": "string"},
            "caption": {"type": "string"},
            "visual_observation": {"type": "string"},
            "landmarks": {"type": "array"},
            "objects": {"type": "array"},
            "place_category": {"type": "string"},
            "spatial_cues": {"type": "array"},
            "navigation_relevance": {"type": "string"},
            "retrieval_text": {"type": "string"},
            "memory_scope": {"type": "string"},
            "memory_namespace": {"type": "string"},
            "source_image_role": {"type": "string"},
            "run_id": {"type": "string"},
            "action_context": {"type": "string"},
            "write_gate": {"type": "object"},
            "metadata": {"type": "object"},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "written": {"type": "boolean"},
            "skipped": {"type": "boolean"},
            "record": {"type": "object"},
        },
        "required": ["written"],
    }
    side_effects = True
    oracle_safe = True

    def __init__(
        self,
        store: Optional[Any] = None,
        client: Optional[Any] = None,
        allowed_sources: Optional[Set[str]] = None,
    ) -> None:
        self.store = store
        self.client = client
        self.allowed_sources = allowed_sources or set(DEFAULT_ALLOWED_MEMORY_SOURCES)

    def run(self, state: Any, payload: Dict[str, Any]) -> SkillResult:
        memory_source = payload.get("memory_source", "episode-local")
        if not payload.get("should_write", False):
            skip_reason = self._write_gate_skip_reason(payload) or "should_write_false"
            return SkillResult.ok_result(
                "memory_write",
                self._skip_payload(state, payload, memory_source, skip_reason),
            )

        if memory_source not in self.allowed_sources:
            return SkillResult.error_result(
                f"Invalid memory_source for online write: {memory_source}",
                result_type="error",
            )
        skip_reason = self._write_gate_skip_reason(payload)
        if skip_reason:
            return SkillResult.ok_result(
                "memory_write",
                self._skip_payload(state, payload, memory_source, skip_reason),
            )
        contract_error = self._event_gated_keyframe_contract_error(
            state,
            payload,
        )
        if contract_error:
            failed_payload = self._skip_payload(
                state,
                payload,
                memory_source,
                contract_error,
            )
            failed_payload["error_type"] = "invalid_event_gated_keyframe_write"
            return SkillResult.ok_result("memory_write", failed_payload)

        record = self._build_record(state, payload, memory_source)
        if self.client is not None:
            try:
                response = self.client.ingest_semantic(record)
            except Exception as exc:
                return SkillResult.ok_result(
                    "memory_write",
                    {
                        "written": False,
                        "skipped": False,
                        "error_type": "memory_write_failed",
                        "error": str(exc),
                        "record": record,
                    },
                )
            if isinstance(response, dict) and response.get("memory_id"):
                record["memory_id"] = str(response["memory_id"])
        if self.store is not None:
            self.store.append(record)

        payload_out = {"written": True, "record": record}
        if record.get("memory_id"):
            payload_out["memory_id"] = record["memory_id"]
        return SkillResult.ok_result(
            "memory_write",
            payload_out,
            confidence=1.0,
        )

    def _build_record(
        self,
        state: Any,
        payload: Dict[str, Any],
        memory_source: str,
    ) -> Dict[str, Any]:
        metadata = self._record_metadata(state, payload, memory_source)
        return {
            "write_type": payload.get("write_type", "episodic_keyframe"),
            "step_id": payload.get("step_id", getattr(state, "step_id", None)),
            "scene_id": payload.get("scene_id", getattr(state, "scene_id", None)),
            "episode_id": payload.get("episode_id", getattr(state, "episode_id", None)),
            "run_id": payload.get("run_id") or metadata.get("run_id", ""),
            "image_path": payload.get("image_path"),
            "note": payload.get("note", ""),
            "caption": payload.get("caption", ""),
            "visual_observation": payload.get("visual_observation", ""),
            "objects": self._string_list(payload.get("objects")),
            "landmarks": self._string_list(payload.get("landmarks")),
            "place_category": payload.get("place_category", ""),
            "spatial_cues": self._string_list(payload.get("spatial_cues")),
            "navigation_relevance": payload.get("navigation_relevance", ""),
            "retrieval_text": payload.get("retrieval_text")
            or self._build_retrieval_text(payload),
            "memory_scope": payload.get("memory_scope", "episode"),
            "memory_namespace": payload.get("memory_namespace")
            or self._default_namespace(state, payload, memory_source),
            "source_image_role": payload.get("source_image_role", "keyframe"),
            "write_gate": dict(payload.get("write_gate") or {}),
            "memory_source": memory_source,
            "metadata": metadata,
        }

    def _write_gate_skip_reason(self, payload: Dict[str, Any]) -> str:
        write_gate = payload.get("write_gate")
        if not isinstance(write_gate, dict):
            return ""
        if str(write_gate.get("curator_decision") or "").lower() != "skip":
            return ""
        return str(write_gate.get("curator_reason") or "write_gate_skip")

    def _skip_payload(
        self,
        state: Any,
        payload: Dict[str, Any],
        memory_source: str,
        skip_reason: str,
    ) -> Dict[str, Any]:
        return {
            "written": False,
            "skipped": True,
            "skip_reason": skip_reason,
            "image_path": payload.get("image_path"),
            "memory_scope": payload.get("memory_scope", "episode"),
            "memory_namespace": payload.get("memory_namespace")
            or self._default_namespace(state, payload, memory_source),
            "memory_source": memory_source,
            "write_gate": dict(payload.get("write_gate") or {}),
            "caption": payload.get("caption", ""),
            "visual_observation": payload.get("visual_observation", ""),
        }

    def _event_gated_keyframe_contract_error(
        self,
        state: Any,
        payload: Dict[str, Any],
    ) -> str:
        if str(payload.get("source_image_role") or "") != "event_gated_keyframe":
            return ""
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        keyframe_gate = metadata.get("keyframe_gate")
        required_values = {
            "step_id": payload.get("step_id", getattr(state, "step_id", None)),
            "scene_id": payload.get("scene_id", getattr(state, "scene_id", None)),
            "episode_id": payload.get("episode_id", getattr(state, "episode_id", None)),
            "image_path": payload.get("image_path"),
            "memory_scope": payload.get("memory_scope"),
            "memory_namespace": payload.get("memory_namespace"),
            "run_id": payload.get("run_id") or metadata.get("run_id"),
        }
        missing = [
            key
            for key, value in required_values.items()
            if value is None or str(value) == ""
        ]
        if missing:
            return "missing required event_gated_keyframe fields: " + ", ".join(missing)
        if not isinstance(keyframe_gate, dict) or not keyframe_gate:
            return "missing required event_gated_keyframe fields: metadata.keyframe_gate"
        if not (
            payload.get("retrieval_text")
            or payload.get("action_context")
            or payload.get("visual_observation")
            or payload.get("caption")
            or payload.get("note")
        ):
            return (
                "missing required event_gated_keyframe fields: "
                "retrieval_text_or_action_context"
            )
        return ""

    def _record_metadata(
        self,
        state: Any,
        payload: Dict[str, Any],
        memory_source: str,
    ) -> Dict[str, Any]:
        metadata = payload.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        fields = {
            "run_id": payload.get("run_id") or metadata.get("run_id", ""),
            "step_id": payload.get("step_id", getattr(state, "step_id", None)),
            "scene_id": payload.get("scene_id", getattr(state, "scene_id", None)),
            "episode_id": payload.get("episode_id", getattr(state, "episode_id", None)),
            "image_path": payload.get("image_path"),
            "memory_scope": payload.get("memory_scope", "episode"),
            "memory_namespace": payload.get("memory_namespace")
            or self._default_namespace(state, payload, memory_source),
            "source_image_role": payload.get("source_image_role", "keyframe"),
        }
        for key, value in fields.items():
            if value is not None and str(value) != "":
                metadata.setdefault(key, value)
        return metadata

    def _build_retrieval_text(self, payload: Dict[str, Any]) -> str:
        parts = []
        for key in (
            "instruction_context",
            "active_subgoal",
            "caption",
            "visual_observation",
            "place_category",
            "navigation_relevance",
            "action_context",
            "note",
        ):
            value = payload.get(key)
            if value:
                parts.append(str(value))
        for key in ("objects", "landmarks", "spatial_cues"):
            values = self._string_list(payload.get(key))
            if values:
                parts.append(", ".join(values))
        return "\n".join(parts)

    def _default_namespace(
        self,
        state: Any,
        payload: Dict[str, Any],
        memory_source: str,
    ) -> str:
        scope = str(payload.get("memory_scope") or "episode")
        scene_id = payload.get("scene_id", getattr(state, "scene_id", ""))
        episode_id = payload.get("episode_id", getattr(state, "episode_id", ""))
        if scope == "scene":
            return f"scene:{scene_id}:{memory_source}"
        if scope == "task":
            return f"task:{memory_source}"
        return f"episode:{scene_id}:{episode_id}"

    def _string_list(self, value: Any) -> list:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item)]
