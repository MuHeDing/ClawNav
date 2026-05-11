from typing import Any, Dict

from harness.types import SkillResult


class Skill:
    name: str = ""

    def run(self, state: Any, payload: Dict[str, Any]) -> SkillResult:
        raise NotImplementedError

