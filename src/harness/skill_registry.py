from typing import Any, Dict, List

from harness.skills.base import Skill, SkillManifest
from harness.types import SkillResult


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        name = skill.name or skill.__class__.__name__
        if name in self._skills:
            raise ValueError(f"Skill already registered: {name}")
        self._skills[name] = skill

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def names(self) -> List[str]:
        return sorted(self._skills.keys())

    def list_manifests(self) -> List[SkillManifest]:
        return [self._skills[name].manifest() for name in self.names()]

    def export_tool_schemas(self) -> List[Dict[str, Any]]:
        return [manifest.to_dict() for manifest in self.list_manifests()]

    def run(self, name: str, state: Any, payload: Dict[str, Any]) -> SkillResult:
        skill = self._skills.get(name)
        if skill is None:
            return SkillResult.error_result(f"Skill not registered: {name}")

        try:
            result = skill.run(state, payload)
        except Exception as exc:
            return SkillResult.error_result(f"{name} failed: {exc}")

        if not isinstance(result, SkillResult):
            return SkillResult.error_result(
                f"{name} returned invalid result type: {type(result).__name__}"
            )
        return result
