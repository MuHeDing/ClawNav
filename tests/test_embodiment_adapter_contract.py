from harness.env_adapters.base import BaseEmbodimentAdapter
from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter


def test_habitat_adapter_implements_base_contract():
    adapter = HabitatVLNAdapter()

    assert isinstance(adapter, BaseEmbodimentAdapter)
    assert adapter.action_space() == ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]


def test_action_to_executor_command_is_structured():
    adapter = HabitatVLNAdapter()

    command = adapter.action_to_executor_command("TURN_LEFT")

    assert command["action_text"] == "TURN_LEFT"
    assert command["action_index"] == 2
    assert command["executor"] == "habitat_discrete"
