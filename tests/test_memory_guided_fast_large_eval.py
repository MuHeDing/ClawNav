import json
from pathlib import Path

from scripts.run_memory_guided_fast_large_eval import (
    LargeEvalConfig,
    build_adapter_env,
    build_eval_env,
    collect_run_summary,
    render_markdown_report,
    validate_large_eval_gate,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _trace_row(step_mode: str, **overrides):
    audit = {
        "planner_step_mode": step_mode,
        "planner_authority": "qwen" if step_mode == "visual_update" else "local_policy",
        "qwen_api_called": step_mode == "visual_update",
        "model_call_skipped": step_mode == "fast_text",
        "model_image_count": 2 if step_mode == "visual_update" else 0,
        "model_image_paths": ["a.png", "b.png"] if step_mode == "visual_update" else [],
        "visual_memory_update_status": "updated"
        if step_mode == "visual_update"
        else "not_applicable",
        "provider_input_tokens": 1350 if step_mode == "visual_update" else None,
        "assembled_prompt_tokens": 640,
    }
    audit.update(overrides.pop("context_audit", {}))
    row = {
        "scene_id": "2azQ1b91cZZ",
        "episode_id": "11",
        "step_id": 0,
        "fallback": False,
        "planner_fallback": False,
        "planner_reason": "ok",
        "context_audit": audit,
    }
    row.update(overrides)
    return row


def test_collect_summary_accepts_valid_memory_guided_fast_trace(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            _trace_row("visual_update", step_id=0),
            _trace_row("fast_text", step_id=1),
            _trace_row("fast_text", step_id=2),
            _trace_row("visual_update", step_id=10),
        ],
    )
    _write_jsonl(
        run_dir / "result.json",
        [
            {"scene_id": "2azQ1b91cZZ", "episode_id": 11, "success": 1.0, "spl": 1.0},
            {"scene_id": "2azQ1b91cZZ", "episode_id": 10, "success": 0.0, "spl": 0.0},
        ],
    )

    summary = collect_run_summary(run_dir, expected_episodes=2, model_max_images=2)

    assert summary["result_count"] == 2
    assert summary["trace_rows"] == 4
    assert summary["mode_counts"] == {"fast_text": 2, "visual_update": 2}
    assert summary["qwen_api_called"] == 2
    assert summary["fast_qwen_api_called"] == 0
    assert summary["visual_qwen_api_called"] == 2
    assert summary["max_model_image_count"] == 2
    assert summary["success_sum"] == 1.0
    assert validate_large_eval_gate(summary) == []


def test_gate_rejects_planner_fallback_and_keeps_context(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            _trace_row("visual_update", step_id=0),
            _trace_row(
                "visual_update",
                step_id=20,
                episode_id="43",
                planner_fallback=True,
                planner_reason="openclaw_cli_model_fallback:openclaw_cli_interval_recall",
                context_audit={
                    "visual_memory_update_status": "error",
                    "planner_error": "Qwen API read timeout",
                },
            ),
        ],
    )
    _write_jsonl(
        run_dir / "result.json",
        [{"scene_id": "2azQ1b91cZZ", "episode_id": 11, "success": 1.0, "spl": 1.0}],
    )

    summary = collect_run_summary(run_dir, expected_episodes=2, model_max_images=2)
    failures = validate_large_eval_gate(summary)

    assert any(failure["code"] == "expected_episode_count" for failure in failures)
    planner_failure = next(
        failure for failure in failures if failure["code"] == "planner_fallback"
    )
    assert planner_failure["count"] == 1
    assert planner_failure["examples"][0]["episode_id"] == "43"
    assert "openclaw_cli_model_fallback" in planner_failure["examples"][0]["planner_reason"]
    assert "Qwen API read timeout" in planner_failure["examples"][0]["planner_error"]


def test_gate_rejects_fast_qwen_calls_and_visual_image_over_budget(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "harness_traces" / "harness_trace_rank0.jsonl",
        [
            _trace_row(
                "fast_text",
                step_id=1,
                context_audit={
                    "planner_authority": "qwen",
                    "qwen_api_called": True,
                    "model_call_skipped": False,
                },
            ),
            _trace_row(
                "visual_update",
                step_id=10,
                context_audit={
                    "model_image_count": 3,
                    "model_image_paths": ["a.png", "b.png", "c.png"],
                },
            ),
        ],
    )
    _write_jsonl(
        run_dir / "result.json",
        [{"scene_id": "2azQ1b91cZZ", "episode_id": 11, "success": 1.0, "spl": 1.0}],
    )

    summary = collect_run_summary(run_dir, expected_episodes=1, model_max_images=2)
    failures = validate_large_eval_gate(summary)

    assert {failure["code"] for failure in failures} == {
        "fast_qwen_api_called",
        "fast_not_local_policy",
        "fast_model_not_skipped",
        "visual_model_image_over_budget",
    }


def test_command_env_defaults_lock_memory_guided_fast_contract(tmp_path):
    config = LargeEvalConfig(
        gpu="1",
        port=8011,
        episodes=30,
        image_interval_steps=10,
        model_max_images=1,
        smoke_output=tmp_path / "smoke",
        eval_output=tmp_path / "eval",
    )

    adapter_env = build_adapter_env(config)
    smoke_env = build_eval_env(
        config,
        output_path=config.smoke_output,
        episodes=1,
        master_port=20501,
        episode_keys=config.smoke_episode_key,
    )
    eval_env = build_eval_env(
        config,
        output_path=config.eval_output,
        episodes=config.episodes,
        master_port=20502,
    )

    assert adapter_env["OPENCLAW_MODEL_FAST_MODE"] == "memory_guided_policy_fast"
    assert adapter_env["OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS"] == "10"
    assert adapter_env["OPENCLAW_MODEL_MAX_IMAGES"] == "1"
    assert adapter_env["OPENCLAW_QWEN_API_RETRIES"] == "1"
    assert smoke_env["OPENCLAW_GATEWAY_TIMEOUT"] == "300"
    assert smoke_env["OPENCLAW_ENFORCE_TIMEOUT_BUDGET"] == "1"
    assert smoke_env["HARNESS_DEBUG_MAX_EPISODES"] == "1"
    assert smoke_env["HARNESS_EPISODE_KEYS"] == "2azQ1b91cZZ:11"
    assert eval_env["HARNESS_DEBUG_MAX_EPISODES"] == "30"
    assert "HARNESS_EPISODE_KEYS" not in eval_env


def test_report_marks_invalid_runs_as_diagnostic_only(tmp_path):
    summary = {
        "run_dir": str(tmp_path / "run"),
        "result_count": 8,
        "expected_episodes": 30,
        "trace_rows": 387,
        "mode_counts": {"visual_update": 44, "fast_text": 343},
        "fallback_count": 0,
        "planner_fallback_count": 1,
        "qwen_api_called": 44,
        "fast_qwen_api_called": 0,
        "visual_qwen_api_called": 44,
        "max_model_image_count": 2,
        "success_sum": 6.0,
        "spl_sum": 6.0,
        "failures": [
            {
                "code": "planner_fallback",
                "message": "planner_fallback rows were present",
                "count": 1,
                "examples": [
                    {
                        "scene_id": "2azQ1b91cZZ",
                        "episode_id": "43",
                        "step_id": 20,
                        "planner_reason": "openclaw_cli_model_fallback:openclaw_cli_interval_recall",
                    }
                ],
            }
        ],
    }

    report = render_markdown_report(summary, label="eval")

    assert "Status: INVALID_DIAGNOSTIC_ONLY" in report
    assert "planner_fallback" in report
    assert "2azQ1b91cZZ / 43 / step 20" in report
