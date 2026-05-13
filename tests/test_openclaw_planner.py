from harness.openclaw.planner import (
    OpenClawPlanDecision,
    RuleOpenClawPlanner,
)
from harness.types import VLNState


def make_state(step_id=0, instruction="go to the kitchen"):
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction=instruction,
        step_id=step_id,
        current_image=None,
    )


def test_rule_planner_starts_with_memory_recall():
    planner = RuleOpenClawPlanner(recall_interval_steps=5)

    decision = planner.plan(make_state(step_id=0), runtime_context={})

    assert isinstance(decision, OpenClawPlanDecision)
    assert decision.intent == "recall_memory"
    assert decision.tool_name == "MemoryQuerySkill"
    assert decision.arguments["text"] == "go to the kitchen"
    assert decision.reason == "initial_recall"


def test_rule_planner_acts_between_recall_intervals():
    planner = RuleOpenClawPlanner(recall_interval_steps=5)

    decision = planner.plan(make_state(step_id=2), runtime_context={})

    assert decision.intent == "act"
    assert decision.tool_name == "NavigationPolicySkill"


def test_rule_planner_never_forwards_oracle_context():
    planner = RuleOpenClawPlanner(recall_interval_steps=5)

    decision = planner.plan(
        make_state(step_id=0),
        runtime_context={"distance_to_goal": 1.0, "success": True},
    )

    assert "distance_to_goal" not in decision.arguments
    assert "success" not in decision.arguments
