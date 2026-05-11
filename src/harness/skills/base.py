from dataclasses import dataclass, field
from typing import Any, Dict

from harness.types import SkillResult


@dataclass
class SkillManifest:
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 5000
    side_effects: bool = False
    oracle_safe: bool = True
    callable_from_runtime: bool = True
    schema_version: str = "phase2.skill_manifest.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "timeout_ms": self.timeout_ms,
            "side_effects": self.side_effects,
            "oracle_safe": self.oracle_safe,
            "callable_from_runtime": self.callable_from_runtime,
        }


class Skill:
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    timeout_ms: int = 5000
    side_effects: bool = False
    oracle_safe: bool = True
    callable_from_runtime: bool = True

    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name=self.name or self.__class__.__name__,
            description=self.description,
            input_schema=dict(self.input_schema),
            output_schema=dict(self.output_schema),
            timeout_ms=self.timeout_ms,
            side_effects=self.side_effects,
            oracle_safe=self.oracle_safe,
            callable_from_runtime=self.callable_from_runtime,
        )

    def run(self, state: Any, payload: Dict[str, Any]) -> SkillResult:
        raise NotImplementedError
