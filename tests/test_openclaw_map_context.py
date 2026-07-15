from types import SimpleNamespace

from harness.openclaw.map_context import (
    FloorplanMapContextProvider,
    MAP_ASSIST_FLOORPLAN,
)


def fake_state(position, collision=False, last_action=None):
    return SimpleNamespace(
        diagnostic_pose={
            "position": list(position),
            "rotation": [1.0, 0.0, 0.0, 0.0],
        },
        pose=None,
        diagnostics={},
        online_metrics={"collision": collision},
        last_action=last_action,
    )


def test_floorplan_map_context_writes_only_on_interval_steps(tmp_path):
    calls = []

    def renderer(**kwargs):
        calls.append(kwargs)
        return b"fake-png"

    provider = FloorplanMapContextProvider(
        output_root=tmp_path,
        mode=MAP_ASSIST_FLOORPLAN,
        frame_interval_steps=5,
        renderer=renderer,
    )

    first = provider.build_context(
        env=SimpleNamespace(),
        state=fake_state((0.0, 0.0, 0.0)),
        step_id=0,
        scene_id="scene/a",
        episode_id="ep:1",
    )
    middle = provider.build_context(
        env=SimpleNamespace(),
        state=fake_state((0.1, 0.0, 0.0)),
        step_id=1,
        scene_id="scene/a",
        episode_id="ep:1",
    )
    next_due = provider.build_context(
        env=SimpleNamespace(),
        state=fake_state((0.2, 0.0, 0.0)),
        step_id=5,
        scene_id="scene/a",
        episode_id="ep:1",
    )

    assert first["map_frame_due"] is True
    assert first["map_available"] is True
    assert first["map_image_label"] == "map_view"
    assert first["internal_only"]["map_image_path"].endswith(
        "openclaw_map_frames/scene_a/ep_1/step_000000.png"
    )
    assert middle["map_frame_due"] is False
    assert middle["map_available"] is False
    assert "internal_only" not in middle
    assert next_due["internal_only"]["map_image_path"].endswith("step_000005.png")
    assert len(calls) == 2
    assert (tmp_path / "openclaw_map_frames" / "scene_a" / "ep_1" / "step_000000.png").exists()
    assert (tmp_path / "openclaw_map_frames" / "scene_a" / "ep_1" / "step_000005.png").exists()


def test_floorplan_map_context_records_non_oracle_safety_contract(tmp_path):
    provider = FloorplanMapContextProvider(
        output_root=tmp_path,
        mode=MAP_ASSIST_FLOORPLAN,
        frame_interval_steps=5,
        collision_overlay_enabled=True,
        renderer=lambda **_: b"fake-png",
    )

    initial = provider.build_context(
        env=SimpleNamespace(),
        state=fake_state((0.0, 0.0, 0.0)),
        step_id=0,
        scene_id="scene-a",
        episode_id="episode-1",
    )
    progressed_collision = provider.build_context(
        env=SimpleNamespace(),
        state=fake_state(
            (0.0, 0.0, 0.25),
            collision=True,
            last_action="MOVE_FORWARD",
        ),
        step_id=1,
        scene_id="scene-a",
        episode_id="episode-1",
    )
    blocked_collision = provider.build_context(
        env=SimpleNamespace(),
        state=fake_state(
            (0.0, 0.0, 0.26),
            collision=True,
            last_action="MOVE_FORWARD",
        ),
        step_id=2,
        scene_id="scene-a",
        episode_id="episode-1",
    )

    assert initial["map_source"] == "habitat_pathfinder_navmesh"
    assert initial["pose_source"] == "sim_agent_state_odometry"
    assert initial["collision_point_count"] == 0
    assert progressed_collision["collision_point_count"] == 0
    assert blocked_collision["collision_point_count"] == 1
    assert all(value is False for value in initial["map_safety"].values())
    assert "goal" not in initial["internal_only"]
    assert "reference_path" not in initial["internal_only"]
