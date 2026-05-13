from harness.openclaw.critic import SubagentProgressCritic
from harness.openclaw.memory_curator import SubagentMemoryCurator
from harness.openclaw.planner import SubagentOpenClawPlanner
from harness.openclaw.subagents import FakeSubagentClient
from harness.types import VLNState


def make_state():
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=3,
        current_image=None,
    )


def test_subagent_planner_returns_plan_decision():
    planner = SubagentOpenClawPlanner(
        FakeSubagentClient(
            {
                "intent": "act",
                "tool_name": "NavigationPolicySkill",
                "arguments": {},
                "reason": "subagent",
            }
        )
    )

    decision = planner.plan(make_state(), {"success": True})

    assert decision.planner_backend == "subagent"
    assert decision.reason == "subagent"


def test_subagent_critic_returns_non_oracle_feedback():
    critic = SubagentProgressCritic(
        FakeSubagentClient({"status": "stuck", "reason": "repeated_turns"})
    )

    result = critic.evaluate(
        make_state(),
        {"distance_to_goal": 1.0, "actions": ["TURN_LEFT"]},
    )

    assert result["status"] == "stuck"
    assert "distance_to_goal" not in critic.client.requests[0].context


def test_subagent_memory_curator_returns_write_decision():
    curator = SubagentMemoryCurator(
        FakeSubagentClient({"should_write": True, "reason": "landmark"})
    )

    result = curator.should_write(
        make_state(),
        {"success": True, "visual_novelty": 0.7},
    )

    assert result["should_write"] is True
    assert curator.client.requests[0].context == {"visual_novelty": 0.7}
