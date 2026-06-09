import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_debug_utils import format_episode_progress_line


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
