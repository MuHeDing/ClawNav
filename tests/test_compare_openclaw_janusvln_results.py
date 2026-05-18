import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_openclaw_janusvln_results.py"


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_compare_results_outputs_table_in_openclaw_order(tmp_path):
    openclaw = tmp_path / "openclaw.jsonl"
    janus = tmp_path / "janus.jsonl"
    write_jsonl(
        openclaw,
        [
            {"episode_id": 2, "scene_id": "s", "success": 0.0, "spl": 0.0, "ne": 5.0},
            {"episode_id": 1, "scene_id": "s", "success": 1.0, "spl": 0.9, "ne": 0.5},
        ],
    )
    write_jsonl(
        janus,
        [
            {"episode_id": 1, "scene_id": "s", "success": 0.0, "spl": 0.0, "ne": 4.0},
            {"episode_id": 2, "scene_id": "s", "success": 1.0, "spl": 0.8, "ne": 0.7},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--openclaw-result",
            str(openclaw),
            "--janus-result",
            str(janus),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "OpenClaw+JanusVLN success: 1/2 = 0.5000" in result.stdout
    assert "JanusVLN matched success: 1/2 = 0.5000" in result.stdout
    assert "| episode_id | scene_id | OpenClaw+JanusVLN success | JanusVLN success |" in result.stdout
    assert "SPL" not in result.stdout
    assert "NE" not in result.stdout
    first_row = result.stdout.index("| 2 |")
    second_row = result.stdout.index("| 1 |")
    assert first_row < second_row


def test_compare_results_fails_when_baseline_missing_episode(tmp_path):
    openclaw = tmp_path / "openclaw.jsonl"
    janus = tmp_path / "janus.jsonl"
    write_jsonl(openclaw, [{"episode_id": 9, "success": 1.0}])
    write_jsonl(janus, [{"episode_id": 8, "success": 1.0}])

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--openclaw-result",
            str(openclaw),
            "--janus-result",
            str(janus),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing episode_id in JanusVLN result: 9" in result.stderr
