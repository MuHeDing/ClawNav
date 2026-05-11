from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter


class FakeInstruction:
    instruction_text = "go to the kitchen"


class FakeEpisode:
    scene_id = "scene1"
    episode_id = "episode1"
    instruction = FakeInstruction()


class FakeAgentState:
    position = [1.0, 2.0, 3.0]
    rotation = [0.0, 0.0, 0.0, 1.0]


class FakeSim:
    def get_agent_state(self):
        return FakeAgentState()


class FakeEnv:
    sim = FakeSim()


def test_adapter_extracts_rgb_and_instruction():
    adapter = HabitatVLNAdapter()
    state = adapter.build_state(
        env=FakeEnv(),
        episode=FakeEpisode(),
        observations={"rgb": "rgb-frame"},
        metrics={},
        step_id=0,
    )
    assert state.current_image == "rgb-frame"
    assert state.instruction == "go to the kitchen"
    assert state.scene_id == "scene1"
    assert state.episode_id == "episode1"


def test_adapter_extracts_online_pose_when_allowed():
    adapter = HabitatVLNAdapter(expose_pose_online=True)
    state = adapter.build_state(
        env=FakeEnv(),
        episode=FakeEpisode(),
        observations={"rgb": "rgb-frame"},
        metrics={},
        step_id=0,
    )
    assert state.pose["position"] == [1.0, 2.0, 3.0]
    assert "sim_position" not in state.diagnostics


def test_adapter_separates_online_pose_from_diagnostic_pose():
    adapter = HabitatVLNAdapter(expose_pose_online=False)
    state = adapter.build_state(
        env=FakeEnv(),
        episode=FakeEpisode(),
        observations={"rgb": "rgb-frame"},
        metrics={},
        step_id=0,
    )
    assert state.pose is None
    assert state.diagnostics["sim_position"] == [1.0, 2.0, 3.0]


def test_oracle_metrics_are_diagnostics_not_online_metrics():
    adapter = HabitatVLNAdapter()
    state = adapter.build_state(
        env=FakeEnv(),
        episode=FakeEpisode(),
        observations={"rgb": "rgb-frame"},
        metrics={
            "distance_to_goal": 1.0,
            "success": False,
            "SPL": 0.0,
            "collisions": {"is_collision": True},
        },
        step_id=3,
    )
    assert state.online_metrics["collision"] is True
    assert "distance_to_goal" not in state.online_metrics
    assert state.diagnostics["distance_to_goal"] == 1.0
    assert state.diagnostics["success"] is False
    assert state.diagnostics["SPL"] == 0.0
