import json
import subprocess

from scripts.stress_openclaw_context_engine import run_context_stress


def test_run_context_stress_writes_100_step_audit_and_summary(tmp_path):
    output_dir = tmp_path / "stress"

    summary = run_context_stress(output_dir=output_dir, steps=100, mode="fake")

    audit_path = output_dir / "context_audit.jsonl"
    summary_path = output_dir / "summary.json"
    assert audit_path.exists()
    assert summary_path.exists()
    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 100
    assert summary["total_steps"] == 100
    assert summary["status"] == "pass"
    assert records[0]["step_id"] == 0
    assert records[-1]["step_id"] == 99
    assert records[0]["prompt_contains_step0_marker"] is True
    assert all(
        not record["prompt_contains_step0_marker"]
        for record in records
        if record["step_id"] > 0
    )
    assert len({record["openclaw_session_id"] for record in records}) == 100


def test_run_context_stress_can_fail_when_fake_provider_growth_is_linear(tmp_path):
    output_dir = tmp_path / "stress"

    summary = run_context_stress(
        output_dir=output_dir,
        steps=100,
        mode="fake",
        fake_growth_per_step=100,
    )

    assert summary["status"] == "fail"
    assert summary["checks"]["provider_tokens_stable"]["passed"] is False


def test_run_context_stress_model_mode_writes_stateless_audit_without_provider_usage(
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "stress"

    def fake_openclaw(args, timeout_s):
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "capability": "model.run",
                    "outputs": [
                        {
                            "text": json.dumps(
                                {
                                    "intent": "act",
                                    "tool_name": "NavigationPolicySkill",
                                    "arguments": {"action_text": "MOVE_FORWARD"},
                                    "reason": "model stress",
                                }
                            )
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "scripts.stress_openclaw_context_engine.run_openclaw_command",
        fake_openclaw,
    )

    summary = run_context_stress(
        output_dir=output_dir,
        steps=100,
        mode="model",
        openclaw_model="qwen/qwen3.5-flash",
    )

    records = [
        json.loads(line)
        for line in (output_dir / "context_audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["status"] == "pass"
    assert summary["checks"]["stateless_model"]["passed"] is True
    assert summary["checks"]["provider_usage_optional"]["passed"] is True
    assert all(record["openclaw_session_mode"] == "stateless_model" for record in records)
    assert all(record["provider_input_tokens"] is None for record in records)
    assert all(record["provider_usage_source"] == "unavailable" for record in records)
