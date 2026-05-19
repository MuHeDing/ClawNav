import re
from typing import Any, Dict, Set, Tuple

from harness.skills.base import Skill
from harness.types import SkillResult


class VisualMemoryCuratorSkill(Skill):
    name = "VisualMemoryCuratorSkill"
    description = "Gate visual memory writes using non-oracle visual evidence."
    input_schema = {
        "type": "object",
        "properties": {
            "caption": {"type": "string"},
            "visual_observation": {"type": "string"},
            "landmarks": {"type": "array"},
            "objects": {"type": "array"},
            "place_category": {"type": "string"},
            "spatial_cues": {"type": "array"},
            "navigation_relevance": {"type": "string"},
            "confidence": {"type": "number"},
            "write_gate": {"type": "object"},
            "recent_visual_memories": {"type": "array"},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "should_write": {"type": "boolean"},
            "write_gate": {"type": "object"},
        },
    }
    oracle_safe = True

    def run(self, state: Any, payload: Dict[str, Any]) -> SkillResult:
        confidence = self._confidence(payload.get("confidence"))
        reason = self._curator_reason(payload)
        novelty_score, duplicate_of_memory_id = self._novelty(payload)
        should_write = self._has_navigation_value(payload, reason, confidence)
        if duplicate_of_memory_id:
            should_write = False
        decision = "write" if should_write else "skip"
        curator_reason = reason if should_write else "generic or low-confidence visual observation"
        if duplicate_of_memory_id:
            curator_reason = "duplicate visual memory"
        return SkillResult.ok_result(
            "visual_memory_curator",
            {
                "should_write": should_write,
                "write_gate": {
                    "candidate_reason": self._candidate_reason(payload),
                    "curator_decision": decision,
                    "curator_reason": curator_reason,
                    "novelty_score": round(novelty_score, 4),
                    "duplicate_of_memory_id": duplicate_of_memory_id or None,
                    "confidence": confidence,
                },
            },
        )

    def _candidate_reason(self, payload: Dict[str, Any]) -> str:
        write_gate = payload.get("write_gate")
        if isinstance(write_gate, dict) and write_gate.get("candidate_reason"):
            return str(write_gate["candidate_reason"])
        if payload.get("landmarks") or payload.get("objects") or payload.get("spatial_cues"):
            return "landmark"
        return "planner_request"

    def _curator_reason(self, payload: Dict[str, Any]) -> str:
        for key in ("navigation_relevance", "visual_observation", "caption", "note"):
            value = payload.get(key)
            if value:
                return str(value)
        return "visual memory candidate"

    def _has_navigation_value(
        self,
        payload: Dict[str, Any],
        reason: str,
        confidence: float,
    ) -> bool:
        if confidence and confidence < 0.35:
            return False
        if payload.get("landmarks") or payload.get("objects") or payload.get("spatial_cues"):
            return True
        text = " ".join(
            str(payload.get(key) or "")
            for key in (
                "caption",
                "visual_observation",
                "place_category",
                "navigation_relevance",
                "note",
                "curator_reason",
            )
        ).lower()
        navigation_terms = (
            "door",
            "doorway",
            "hall",
            "corridor",
            "stairs",
            "room",
            "turn",
            "left",
            "right",
            "landmark",
            "anchor",
            "entrance",
            "exit",
            "kitchen",
            "sofa",
            "table",
        )
        if any(term in text for term in navigation_terms):
            generic_terms = ("blank wall", "generic wall", "no navigation cue")
            return not any(term in text for term in generic_terms)
        return bool(reason and confidence >= 0.65)

    def _confidence(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _novelty(self, payload: Dict[str, Any]) -> Tuple[float, str]:
        current = self._signature_terms(payload)
        if not current:
            return 0.0, ""
        best_similarity = 0.0
        duplicate_id = ""
        recent_memories = payload.get("recent_visual_memories")
        if not isinstance(recent_memories, list):
            return 1.0, ""
        for memory in recent_memories:
            if not isinstance(memory, dict):
                continue
            previous = self._signature_terms(memory)
            if not previous:
                continue
            similarity = len(current & previous) / max(1, len(current | previous))
            if similarity > best_similarity:
                best_similarity = similarity
                duplicate_id = str(memory.get("memory_id") or memory.get("id") or "")
        if best_similarity >= 0.75:
            return 1.0 - best_similarity, duplicate_id
        return 1.0 - best_similarity, ""

    def _signature_terms(self, payload: Dict[str, Any]) -> Set[str]:
        terms: Set[str] = set()
        for key in (
            "caption",
            "visual_observation",
            "place_category",
            "navigation_relevance",
        ):
            terms.update(self._terms(str(payload.get(key) or "")))
        for key in ("landmarks", "objects", "spatial_cues"):
            values = payload.get(key)
            if isinstance(values, list):
                for value in values:
                    terms.update(self._terms(str(value)))
        return terms

    def _terms(self, text: str) -> Set[str]:
        return {
            term
            for term in re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if len(term) > 2
        }
