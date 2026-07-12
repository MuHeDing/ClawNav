import json
from pathlib import Path

from scripts.compare_qwen_direct_policy_results import compare_result_files


def _write_jsonl(path: Path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_compare_qwen_direct_policy_results_aligns_shared_episode_keys(tmp_path):
    qwen = tmp_path / "qwen.jsonl"
    janus = tmp_path / "janus.jsonl"
    _write_jsonl(
        qwen,
        [
            {"scene_id": "s1", "episode_id": "1", "success": 1.0, "spl": 0.5, "oracle_success": 1.0},
            {"scene_id": "s1", "episode_id": "2", "success": 0.0, "spl": 0.0, "oracle_success": 1.0},
            {"summary": True, "success": 99.0},
        ],
    )
    _write_jsonl(
        janus,
        [
            {"scene_id": "s1", "episode_id": "1", "success": 1.0, "spl": 0.7, "oracle_success": 1.0},
            {"scene_id": "s1", "episode_id": "2", "success": 1.0, "spl": 0.3, "oracle_success": 1.0},
            {"summary": True},
        ],
    )

    report = compare_result_files(qwen, janus)

    assert report["common_key_count"] == 2
    assert report["qwen_metrics"]["success_rate"] == 0.5
    assert report["janus_metrics"]["success_rate"] == 1.0
    assert report["overlap_buckets"] == {
        "both_success": 1,
        "qwen_only": 0,
        "janus_only": 1,
        "both_fail": 0,
    }
    assert report["missing_in_qwen"] == []
    assert report["missing_in_janus"] == []


def test_compare_qwen_direct_policy_results_reports_missing_and_duplicates(tmp_path):
    qwen = tmp_path / "qwen.jsonl"
    janus = tmp_path / "janus.jsonl"
    _write_jsonl(
        qwen,
        [
            {"scene_id": "s1", "episode_id": "1", "success": 1.0},
            {"scene_id": "s1", "episode_id": "1", "success": 0.0},
            {"scene_id": "s1", "episode_id": "3", "success": 1.0},
        ],
    )
    _write_jsonl(
        janus,
        [
            {"scene_id": "s1", "episode_id": "1", "success": 1.0},
            {"scene_id": "s1", "episode_id": "2", "success": 0.0},
        ],
    )

    report = compare_result_files(qwen, janus)

    assert report["common_key_count"] == 1
    assert report["duplicate_qwen_keys"] == ["s1:1"]
    assert report["duplicate_janus_keys"] == []
    assert report["missing_in_qwen"] == ["s1:2"]
    assert report["missing_in_janus"] == ["s1:3"]


def test_compare_qwen_direct_policy_trace_preflight_rejects_janus_calls(tmp_path):
    qwen = tmp_path / "qwen.jsonl"
    janus = tmp_path / "janus.jsonl"
    trace = tmp_path / "trace.jsonl"
    _write_jsonl(qwen, [{"scene_id": "s1", "episode_id": "1", "success": 1.0}])
    _write_jsonl(janus, [{"scene_id": "s1", "episode_id": "1", "success": 1.0}])
    _write_jsonl(
        trace,
        [
            {
                "policy_backend": "qwen_direct",
                "janus_loaded": False,
                "navigation_policy_skill_called": True,
                "planned_tool": "NavigationPolicySkill",
            }
        ],
    )

    report = compare_result_files(qwen, janus, qwen_trace_path=trace)

    assert report["trace_preflight"]["valid"] is False
    assert report["trace_preflight"]["navigation_policy_skill_called_count"] == 1
    assert report["trace_preflight"]["planned_navigation_policy_skill_count"] == 1


def test_compare_qwen_direct_policy_trace_preflight_reports_action_diagnostics(tmp_path):
    qwen = tmp_path / "qwen.jsonl"
    janus = tmp_path / "janus.jsonl"
    trace = tmp_path / "trace.jsonl"
    _write_jsonl(qwen, [{"scene_id": "2azQ1b91cZZ", "episode_id": "11", "success": 0.0}])
    _write_jsonl(janus, [{"scene_id": "2azQ1b91cZZ", "episode_id": "11", "success": 1.0}])
    _write_jsonl(
        trace,
        [
            {
                "scene_id": "2azQ1b91cZZ",
                "episode_id": "11",
                "policy_backend": "qwen_direct",
                "direct_policy": True,
                "planned_tool": "QwenDirectPolicy",
                "candidate_action": "MOVE_FORWARD",
                "final_action": "MOVE_FORWARD",
                "final_action_source": "qwen",
                "context_audit": {
                    "policy_backend": "qwen_direct",
                    "planner_step_mode": "visual_update",
                    "qwen_failure": False,
                },
            },
            {
                "scene_id": "2azQ1b91cZZ",
                "episode_id": "11",
                "policy_backend": "qwen_direct",
                "direct_policy": True,
                "planned_tool": "QwenDirectPolicy",
                "candidate_action": "MOVE_FORWARD",
                "final_action": "MOVE_FORWARD",
                "final_action_source": "qwen",
                "forward_stall_gate_decision": "passed",
                "context_audit": {
                    "policy_backend": "qwen_direct",
                    "planner_step_mode": "fast_text",
                    "qwen_failure": False,
                },
            },
        ],
    )

    report = compare_result_files(qwen, janus, qwen_trace_path=trace)
    preflight = report["trace_preflight"]

    assert preflight["valid"] is True
    assert preflight["action_counts"] == {"MOVE_FORWARD": 2}
    assert preflight["candidate_action_counts"] == {"MOVE_FORWARD": 2}
    assert preflight["final_action_source_counts"] == {"qwen": 2}
    assert preflight["planner_step_mode_counts"] == {"fast_text": 1, "visual_update": 1}
    assert preflight["gate_counts"] == {"forward_stall_gate_decision": 1}
    assert preflight["qwen_failure_count"] == 0
    assert preflight["behaviorally_suspicious"] is True
    assert preflight["behavior_warnings"] == [
        {"code": "all_forward_actions", "count": 2}
    ]
