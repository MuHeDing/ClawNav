from harness.skills.base import Skill, SkillManifest
from harness.memory.memory_manager import MemoryManager
from harness.memory.spatial_memory_client import FakeSpatialMemoryClient
from harness.skills.memory_query import MemoryQuerySkill
from harness.skills.memory_write import MemoryWriteSkill
from harness.skills.navigation_policy import NavigationPolicySkill
from harness.skills.progress_critic import ProgressCriticSkill
from harness.skills.replanner import ReplannerSkill
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


def test_builtin_skills_have_runtime_manifests():
    memory_manager = MemoryManager(FakeSpatialMemoryClient())
    skills = [
        NavigationPolicySkill(model=None),
        MemoryQuerySkill(memory_manager),
        MemoryWriteSkill(FakeSpatialMemoryClient()),
        ProgressCriticSkill(),
        ReplannerSkill(),
    ]

    for skill in skills:
        manifest = skill.manifest()
        assert manifest.name
        assert manifest.description
        assert manifest.input_schema["type"] == "object"
        assert manifest.output_schema["type"] == "object"
        assert manifest.oracle_safe is True


def test_memory_write_manifest_declares_side_effects():
    manifest = MemoryWriteSkill(FakeSpatialMemoryClient()).manifest()

    assert manifest.side_effects is True
