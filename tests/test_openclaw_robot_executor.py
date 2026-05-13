from harness.openclaw.robot_executor import (
    FakeRobotExecutor,
    RobotExecutorCommand,
    RobotHttpExecutor,
)


def test_fake_robot_executor_records_command():
    executor = FakeRobotExecutor()

    command = executor.command_for_action("TURN_LEFT")

    assert command["executor"] == "robot_http"
    assert command["action_text"] == "TURN_LEFT"
    assert command["runtime_executor"] == "openclaw_robot"
    assert executor.commands == ["TURN_LEFT"]


def test_robot_http_executor_posts_action_without_oracle_fields():
    captured = {}

    def post_json(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"command_id": "cmd-1", "accepted": True}

    executor = RobotHttpExecutor(
        base_url="http://robot",
        post_json=post_json,
        timeout_s=2.0,
    )

    command = executor.command_for_action("MOVE_FORWARD")

    assert captured["url"] == "http://robot/action"
    assert captured["payload"] == {"action_text": "MOVE_FORWARD"}
    assert command["command_id"] == "cmd-1"
    assert command["runtime_executor"] == "openclaw_robot"


def test_robot_executor_command_is_dict_compatible():
    command = RobotExecutorCommand(
        executor="robot_http",
        action_text="STOP",
        runtime_executor="openclaw_robot",
        command_id="cmd-2",
    )

    assert command.to_dict()["action_text"] == "STOP"
