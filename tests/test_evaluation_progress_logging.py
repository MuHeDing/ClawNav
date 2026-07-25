import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_debug_utils import (
    apply_episode_validity,
    format_episode_progress_line,
    should_force_stop_for_episode_limit,
)


def test_format_episode_progress_line_includes_episode_position():
    line = format_episode_progress_line(
        status="start",
        rank=0,
        current=3,
        total=100,
        scene_id="zsNo4HB9uLZ",
        episode_id=14,
    )

    assert line == (
        "episode_progress status=start rank=0 episode=3/100 "
        "scene_id=zsNo4HB9uLZ episode_id=14"
    )


def test_format_episode_progress_line_includes_finish_metrics():
    line = format_episode_progress_line(
        status="finish",
        rank=1,
        current=4,
        total=50,
        scene_id="TbHJrupSAjP",
        episode_id="24",
        steps=17,
        metrics={"success": 1.0, "spl": 0.5, "oracle_success": 1.0, "distance_to_goal": 0.25},
    )

    assert line == (
        "episode_progress status=finish rank=1 episode=4/50 "
        "scene_id=TbHJrupSAjP episode_id=24 steps=17 "
        "success=1.0 spl=0.5 os=1.0 ne=0.25"
    )


def test_apply_episode_validity_zeroes_navigation_success_for_policy_abort():
    model = type(
        "InvalidModel",
        (),
        {"episode_invalid": True, "episode_invalid_reason": "route_v2 repair failed"},
    )()

    metrics, invalid, reason = apply_episode_validity(
        {
            "success": 1.0,
            "spl": 0.75,
            "oracle_success": 1.0,
            "distance_to_goal": 0.2,
        },
        model,
    )

    assert invalid is True
    assert reason == "route_v2 repair failed"
    assert metrics == {
        "success": 0.0,
        "spl": 0.0,
        "oracle_success": 0.0,
        "distance_to_goal": 0.2,
    }


def test_episode_step_limit_reserves_final_step_for_controller_stop():
    assert should_force_stop_for_episode_limit(step_id=10, max_steps=12) is False
    assert should_force_stop_for_episode_limit(step_id=11, max_steps=12) is True


def test_episode_step_limit_rejects_non_positive_limit():
    try:
        should_force_stop_for_episode_limit(step_id=0, max_steps=0)
    except ValueError as exc:
        assert "max_steps" in str(exc)
    else:
        raise AssertionError("expected ValueError")
