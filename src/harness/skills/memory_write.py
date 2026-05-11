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
        if not payload.get("should_write", False):
            return SkillResult.ok_result("memory_write", {"written": False, "skipped": True})

        memory_source = payload.get("memory_source", "episode-local")
        if memory_source not in self.allowed_sources:
            return SkillResult.error_result(
                f"Invalid memory_source for online write: {memory_source}",
                result_type="error",
            )

        record = self._build_record(state, payload, memory_source)
        if self.store is not None:
            self.store.append(record)
        if self.client is not None:
            self.client.ingest_semantic(record)

        return SkillResult.ok_result(
            "memory_write",
            {"written": True, "record": record},
            confidence=1.0,
        )

    def _build_record(
        self,
        state: Any,
        payload: Dict[str, Any],
        memory_source: str,
    ) -> Dict[str, Any]:
        return {
            "write_type": payload.get("write_type", "episodic_keyframe"),
            "step_id": payload.get("step_id", getattr(state, "step_id", None)),
            "scene_id": payload.get("scene_id", getattr(state, "scene_id", None)),
            "episode_id": payload.get("episode_id", getattr(state, "episode_id", None)),
            "image_path": payload.get("image_path"),
            "note": payload.get("note", ""),
            "memory_source": memory_source,
            "metadata": payload.get("metadata", {}),
        }
