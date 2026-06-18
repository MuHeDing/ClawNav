import json
from pathlib import Path

from scripts.check_visual_readback_runtime_smoke import check_runtime_smoke


def write_trace(run_dir: Path, rows):
    trace_dir = run_dir / "harness_traces"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "harness_trace_rank0.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def controlled_row(**overrides):
    row = {
        "step_id": 0,
        "action_text": "TURN_LEFT",
        "fallback": False,
        "runtime_status": "completed",
        "planner_fallback": False,
        "tool_calls": [
            {"tool_name": "NavigationPolicySkill"},
            {"tool_name": "MemoryWriteSkill"},
            {"tool_name": "MemoryQuerySkill"},
            {"tool_name": "VisualMemoryReadSkill"},
        ],
        "recall_usage": [
            {
                "num_hits": 1,
                "hit_image_paths": ["/tmp/memory.png"],
                "used_by_policy": False,
            }
        ],
        "visual_readback": {
            "read_status": "completed",
            "trigger_source": "controlled_smoke_seed",
            "actually_read_image_paths": ["/tmp/current.png", "/tmp/memory.png"],
            "matched_memory_ids": ["image-backed-local-0"],
            "used_by_policy": False,
            "final_policy_payload_has_memory": False,
        },
        "visual_readback_config": {
            "visual_readback_mode": "image_read_controller",
            "visual_readback_smoke_seed_memory": True,
        },
    }
    row.update(overrides)
    return row


def test_controlled_seed_smoke_requires_query_readback_and_matched_images(tmp_path):
    write_trace(tmp_path, [controlled_row()])

    summary = check_runtime_smoke(tmp_path, mode="controlled_seed")

    assert summary["passed"] is True
    assert summary["trace_rows"] == 1
    assert summary["tool_counts"]["MemoryQuerySkill"] == 1
    assert summary["completed_with_current_plus_memory"] == 1
    assert summary["completed_with_matched_memory_ids"] == 1
    assert summary["completed_used_by_policy_count"] == 0
    assert summary["issues"] == []


def test_controlled_seed_smoke_fails_when_matched_ids_are_empty(tmp_path):
    row = controlled_row(
        visual_readback={
            "read_status": "completed",
            "trigger_source": "controlled_smoke_seed",
            "actually_read_image_paths": ["/tmp/current.png", "/tmp/memory.png"],
            "matched_memory_ids": [],
            "used_by_policy": False,
            "final_policy_payload_has_memory": False,
        }
    )
    write_trace(tmp_path, [row])

    summary = check_runtime_smoke(tmp_path, mode="controlled_seed")

    assert summary["passed"] is False
    assert any("matched_memory_ids" in issue for issue in summary["issues"])


def test_no_seed_diagnostic_classifies_no_memory_hit_without_failing_health(tmp_path):
    row = controlled_row(
        tool_calls=[
            {"tool_name": "NavigationPolicySkill"},
            {"tool_name": "VisualMemoryReadSkill"},
        ],
        recall_usage=[],
        visual_readback={
            "read_status": "skipped",
            "trigger_source": "online_controller",
            "skip_reason": "no_memory_hit",
            "actually_read_image_paths": [],
            "matched_memory_ids": [],
            "used_by_policy": False,
            "final_policy_payload_has_memory": False,
        },
        visual_readback_config={
            "visual_readback_mode": "image_read_controller",
            "visual_readback_smoke_seed_memory": False,
        },
    )
    write_trace(tmp_path, [row])

    summary = check_runtime_smoke(tmp_path, mode="no_seed_diagnostic")

    assert summary["passed"] is True
    assert summary["natural_image_hit_ready"] is False
    assert summary["diagnostic_status"] == "no_memory_hit"
    assert summary["skip_reasons"]["no_memory_hit"] == 1
