from harness.openclaw.tool_adapter import OpenClawToolAdapter
from harness.skill_registry import SkillRegistry
from harness.skills.base import Skill
from harness.types import SkillResult, VLNState


class EchoSkill(Skill):
    name = "EchoSkill"
    description = "Echoes text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    output_schema = {"type": "object"}

    def run(self, state, payload):
        return SkillResult.ok_result("echo", {"text": payload["text"]})


def make_state():
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=0,
        current_image=None,
    )


def test_tool_adapter_lists_tools_from_registry():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = OpenClawToolAdapter(registry)

    tools = adapter.list_tools()

    assert tools[0]["name"] == "EchoSkill"
    assert tools[0]["input_schema"]["required"] == ["text"]


def test_tool_adapter_calls_registered_skill():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = OpenClawToolAdapter(registry)

    result = adapter.call_tool("EchoSkill", {"text": "hello"}, state=make_state())

    assert result["ok"] is True
    assert result["result_type"] == "echo"
    assert result["payload"]["text"] == "hello"


def test_tool_adapter_returns_structured_error_for_missing_tool():
    adapter = OpenClawToolAdapter(SkillRegistry())

    result = adapter.call_tool("MissingSkill", {}, state=make_state())

    assert result["ok"] is False
    assert "not registered" in result["error"]


def test_tool_adapter_rejects_oracle_decision_inputs():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = OpenClawToolAdapter(registry)

    result = adapter.call_tool(
        "EchoSkill",
        {"text": "hello", "distance_to_goal": 1.0},
        state=make_state(),
    )

    assert result["ok"] is False
    assert result["error_type"] == "oracle_input_rejected"


def test_tool_adapter_adds_call_metadata():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    adapter = OpenClawToolAdapter(registry)

    result = adapter.call_tool("EchoSkill", {"text": "hello"}, state=make_state())

    assert result["tool_name"] == "EchoSkill"
    assert result["runtime_status"] == "completed"
    assert isinstance(result["latency_ms"], float)
    assert result["tool_schema_version"] == "phase2.skill_manifest.v1"
