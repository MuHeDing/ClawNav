from harness.skills.base import Skill, SkillManifest
from harness.types import SkillResult


class EchoSkill(Skill):
    name = "EchoSkill"
    description = "Echoes a text payload."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    output_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    timeout_ms = 1000
    side_effects = False
    oracle_safe = True
    callable_from_runtime = True

    def run(self, state, payload):
        return SkillResult.ok_result("echo", {"text": payload["text"]})


def test_skill_manifest_is_structured():
    manifest = EchoSkill().manifest()

    assert isinstance(manifest, SkillManifest)
    assert manifest.name == "EchoSkill"
    assert manifest.description == "Echoes a text payload."
    assert manifest.input_schema["required"] == ["text"]
    assert manifest.output_schema["required"] == ["text"]
    assert manifest.timeout_ms == 1000
    assert manifest.side_effects is False
    assert manifest.oracle_safe is True
    assert manifest.callable_from_runtime is True


def test_skill_manifest_exports_plain_dict():
    data = EchoSkill().manifest().to_dict()

    assert data["name"] == "EchoSkill"
    assert data["schema_version"] == "phase2.skill_manifest.v1"
    assert data["input_schema"]["type"] == "object"
    assert data["output_schema"]["type"] == "object"
