from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.openclaw.executor import HabitatOpenClawExecutor


def test_executor_converts_action_text_to_habitat_command():
    executor = HabitatOpenClawExecutor(HabitatVLNAdapter())

    command = executor.command_for_action("TURN_LEFT")

    assert command["executor"] == "habitat_discrete"
    assert command["action_text"] == "TURN_LEFT"
    assert command["action_index"] == 2
    assert command["runtime_executor"] == "openclaw_habitat"


def test_executor_rejects_unsupported_action():
    executor = HabitatOpenClawExecutor(HabitatVLNAdapter())

    try:
        executor.command_for_action("JUMP")
    except ValueError as exc:
        assert "Unsupported Habitat action" in str(exc)
    else:
        raise AssertionError("expected ValueError")
