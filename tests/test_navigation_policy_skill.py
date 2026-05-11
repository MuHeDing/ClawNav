from harness.skills.navigation_policy import NavigationPolicySkill
from harness.types import VLNState


class FakeModel:
    def __init__(self):
        self.calls = []

    def call_model(self, images, task, step_id):
        self.calls.append((images, task, step_id))
        return ["MOVE_FORWARD"]


def test_navigation_policy_augments_instruction_but_not_model_architecture():
    model = FakeModel()
    skill = NavigationPolicySkill(model=model, num_history=8, max_memory_images=0)
    state = VLNState(
        scene_id="s",
        episode_id="e",
        instruction="go to kitchen",
        step_id=3,
        current_image="cur",
        online_metrics={},
        diagnostics={},
    )
    result = skill.run(
        state,
        {
            "recent_frames": ["r1", "r2"],
            "memory_images": ["m1"],
            "active_subgoal": "enter kitchen",
            "memory_context_text": "Remembered kitchen entrance.",
        },
    )
    assert result.ok
    assert result.payload["action_text"] == "MOVE_FORWARD"
    images, task, step_id = model.calls[0]
    assert images[-1] == "cur"
    assert "Current subgoal: enter kitchen" in task
    assert "Remembered kitchen entrance." in task
    assert "m1" not in images
