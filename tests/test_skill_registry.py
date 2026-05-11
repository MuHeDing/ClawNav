from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.types import SkillResult


class EchoSkill(Skill):
    name = "EchoSkill"

    def run(self, state, payload):
        return SkillResult.ok_result("echo", {"payload": payload})


class ExplodingSkill(Skill):
    name = "ExplodingSkill"

    def run(self, state, payload):
        raise RuntimeError("boom")


class NamedEchoSkill(Skill):
    description = "Echoes payload."
    input_schema = {"type": "object"}
    output_schema = {"type": "object"}

    def __init__(self, name):
        self.name = name

    def run(self, state, payload):
        return SkillResult.ok_result("echo", {"payload": payload})


def test_register_skill_by_name():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    assert registry.get("EchoSkill").name == "EchoSkill"
    assert registry.names() == ["EchoSkill"]


def test_duplicate_skill_fails():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    try:
        registry.register(EchoSkill())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate registration should fail")


def test_missing_skill_returns_structured_error():
    registry = SkillRegistry()
    result = registry.run("MissingSkill", state=None, payload={})
    assert result.ok is False
    assert result.result_type == "error"
    assert "not registered" in result.error


def test_skill_exception_is_structured_error():
    registry = SkillRegistry()
    registry.register(ExplodingSkill())
    result = registry.run("ExplodingSkill", state=None, payload={})
    assert result.ok is False
    assert result.result_type == "error"
    assert "boom" in result.error


def test_registered_skill_runs_successfully():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    result = registry.run("EchoSkill", state=None, payload={"x": 1})
    assert result.ok is True
    assert result.payload["payload"] == {"x": 1}


def test_registry_lists_skill_manifests():
    registry = SkillRegistry()
    registry.register(EchoSkill())

    manifests = registry.list_manifests()

    assert len(manifests) == 1
    assert manifests[0].name == "EchoSkill"


def test_registry_exports_manifest_dicts_sorted_by_name():
    registry = SkillRegistry()
    registry.register(NamedEchoSkill(name="ZSkill"))
    registry.register(NamedEchoSkill(name="ASkill"))

    data = registry.export_tool_schemas()

    assert [item["name"] for item in data] == ["ASkill", "ZSkill"]
    assert data[0]["schema_version"] == "phase2.skill_manifest.v1"
