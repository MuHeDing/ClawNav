import json
from pathlib import Path

from scripts.compare_qwen_direct_policy_results import (
    compare_ablation_2x2,
    compare_result_files,
)


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


def _ablation_trace_row(*, dynamic, thinking, exercised, success_action="MOVE_FORWARD"):
    return {
        "scene_id": "s1",
        "episode_id": "1",
        "policy_backend": "qwen_direct",
        "janus_loaded": False,
        "navigation_policy_skill_called": False,
        "planned_tool": "QwenDirectPolicy",
        "candidate_action": success_action,
        "final_action": success_action,
        "final_action_source": "qwen",
        "qwen_route_stage": "en_route",
        "context_audit": {
            "policy_backend": "qwen_direct",
            "qwen_model_called": True,
            "dynamic_visual_context_enabled": dynamic,
            "qwen_thinking_enabled": thinking,
            "qwen_thinking_exercised": exercised,
            "thinking_model_supported": True if thinking else None,
            "thinking_capability_fallback": False,
            "thinking_transport_incompatible": False,
            "qwen_output_json_valid": True,
            "qwen_output_schema": "route_v2",
            "configured_model_id": "qwen/qwen3.5-flash-2026-02-23",
            "configured_model_id_canonical": "qwen3.5-flash-2026-02-23",
            "returned_model_id": "qwen3.5-flash-2026-02-23",
            "returned_model_id_canonical": "qwen3.5-flash-2026-02-23",
            "qwen_transport_mode": "sync",
            "qwen_temperature": 0,
            "openclaw_model_max_images": 3 if dynamic else 8,
            "model_image_count_policy": "role_adaptive" if dynamic else "legacy_cap",
            "openclaw_model_max_images_applied": not dynamic,
            "openclaw_model_image_interval_steps": 1,
            "qwen_thinking_budget": 1024,
            "map_assist_mode": "floorplan_map_assisted",
            "map_frame_interval_steps": 5,
            "motion_feedback_enabled": True,
            "forward_stall_odometry_enabled": True,
            "map_collision_overlay_enabled": True,
            "selected_image_count": 3,
            "selected_image_roles": ["map_view", "keyframe", "current"],
            "missing_image_roles": ["confirmed_landmark"],
            "reasoning_tokens": 12 if exercised else 0,
            "provider_latency_ms": 125.0,
        },
    }


def test_compare_ablation_2x2_uses_exact_keys_and_sr_first_outcomes(tmp_path):
    arms = {}
    settings = {
        "fixed_off": (False, False, False, 0.0, 0.0),
        "dynamic_off": (True, False, False, 1.0, 0.4),
        "fixed_on": (False, True, True, 0.0, 0.0),
        "dynamic_on": (True, True, True, 1.0, 0.5),
    }
    for name, (dynamic, thinking, exercised, success, spl) in settings.items():
        result = tmp_path / f"{name}.jsonl"
        trace = tmp_path / f"{name}.trace.jsonl"
        _write_jsonl(
            result,
            [{"scene_id": "s1", "episode_id": "1", "success": success, "spl": spl}],
        )
        _write_jsonl(
            trace,
            [_ablation_trace_row(dynamic=dynamic, thinking=thinking, exercised=exercised)],
        )
        arms[name] = {"result": result, "trace": trace}

    report = compare_ablation_2x2(arms)

    assert report["valid"] is True
    assert report["exact_episode_keys"] == ["s1:1"]
    assert report["comparisons"]["phase3_dynamic_visual"]["outcome"] == "navigation_improvement"
    assert report["comparisons"]["phase4_thinking"]["outcome"] == "efficiency_improvement_only"
    assert report["arms"]["dynamic_on"]["trace_audit"]["thinking_exercised_true_count"] == 1
    assert report["arms"]["dynamic_on"]["trace_audit"]["selected_image_count_distribution"] == {"3": 1}


def test_compare_ablation_2x2_invalidates_unexercised_thinking_and_mutable_model(tmp_path):
    arms = {}
    for name, dynamic, thinking in (
        ("fixed_off", False, False),
        ("dynamic_off", True, False),
        ("fixed_on", False, True),
        ("dynamic_on", True, True),
    ):
        result = tmp_path / f"{name}.jsonl"
        trace = tmp_path / f"{name}.trace.jsonl"
        row = _ablation_trace_row(dynamic=dynamic, thinking=thinking, exercised=False)
        if name == "fixed_off":
            row["context_audit"]["configured_model_id"] = "qwen/qwen3.5-flash"
            row["context_audit"]["configured_model_id_canonical"] = "qwen3.5-flash"
        _write_jsonl(result, [{"scene_id": "s1", "episode_id": "1", "success": 1, "spl": 1}])
        _write_jsonl(trace, [row])
        arms[name] = {"result": result, "trace": trace}

    report = compare_ablation_2x2(arms)

    assert report["valid"] is False
    assert "mutable_or_wrong_configured_model" in report["arms"]["fixed_off"]["invalid_reasons"]
    assert "thinking_not_exercised" in report["arms"]["dynamic_on"]["invalid_reasons"]
    assert report["comparisons"]["phase3_dynamic_visual"]["outcome"] == "invalid_comparison"
    assert report["comparisons"]["phase4_thinking"]["outcome"] == "invalid_comparison"
