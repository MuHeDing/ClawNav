import json

from scripts.summarize_openclaw_context_audit import (
    format_markdown,
    summarize_audit_file,
)


def write_audit(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_summarize_audit_file_passes_stable_fresh_per_step_trace(tmp_path):
    audit_path = tmp_path / "context_audit.jsonl"
    write_audit(
        audit_path,
        [
            {
                "step_id": step,
                "assembled_prompt_tokens": 500 + (step % 2),
                "provider_input_tokens": 900 + (step % 2),
                "hidden_history_tokens_estimate": 400,
                "history_tokens": 0,
                "openclaw_session_mode": "fresh_per_step",
                "openclaw_session_id": f"session-step-{step}",
                "token_limit_exceeded": False,
                "qwen_hard_limit_exceeded": False,
                "prompt_contains_step0_marker": step == 0,
            }
            for step in range(100)
        ],
    )

    summary = summarize_audit_file(audit_path)

    assert summary["status"] == "pass"
    assert summary["total_steps"] == 100
    assert summary["step10_provider_input_tokens"] == 900
    assert summary["last_provider_input_tokens"] == 901
    assert summary["provider_growth_from_step10_to_last"] == 1
    assert summary["unique_session_ids"] == 100
    assert summary["checks"]["provider_tokens_stable"]["passed"] is True
    assert summary["checks"]["no_history_tokens"]["passed"] is True
    assert summary["checks"]["step0_marker_not_leaked_after_step0"]["passed"] is True


def test_summarize_audit_file_fails_linear_provider_growth(tmp_path):
    audit_path = tmp_path / "context_audit.jsonl"
    write_audit(
        audit_path,
        [
            {
                "step_id": step,
                "assembled_prompt_tokens": 500,
                "provider_input_tokens": 1000 + step * 100,
                "hidden_history_tokens_estimate": 500 + step * 100,
                "history_tokens": 0,
                "openclaw_session_mode": "fresh_per_step",
                "openclaw_session_id": f"session-step-{step}",
                "token_limit_exceeded": False,
                "qwen_hard_limit_exceeded": False,
                "prompt_contains_step0_marker": step == 0,
            }
            for step in range(100)
        ],
    )

    summary = summarize_audit_file(audit_path)

    assert summary["status"] == "fail"
    assert summary["checks"]["provider_tokens_stable"]["passed"] is False
    assert summary["provider_growth_from_step10_to_last"] > 1000


def test_format_markdown_reports_checks_and_token_metrics(tmp_path):
    audit_path = tmp_path / "context_audit.jsonl"
    write_audit(
        audit_path,
        [
            {
                "step_id": 0,
                "assembled_prompt_tokens": 500,
                "provider_input_tokens": 900,
                "history_tokens": 0,
                "openclaw_session_mode": "fresh_per_step",
                "openclaw_session_id": "session-step-0",
                "prompt_contains_step0_marker": True,
            },
            {
                "step_id": 1,
                "assembled_prompt_tokens": 501,
                "provider_input_tokens": 901,
                "history_tokens": 0,
                "openclaw_session_mode": "fresh_per_step",
                "openclaw_session_id": "session-step-1",
                "prompt_contains_step0_marker": False,
            },
        ],
    )

    markdown = format_markdown(summarize_audit_file(audit_path))

    assert "# OpenClaw Context Audit Report" in markdown
    assert "| provider_tokens_stable |" in markdown
    assert "provider_growth_from_step10_to_last" in markdown
