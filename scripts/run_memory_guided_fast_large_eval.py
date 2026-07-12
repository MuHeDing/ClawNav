#!/usr/bin/env python
"""Run and audit memory-guided OpenClaw/Qwen large evaluations.

This runner deliberately keeps the planner implementation unchanged.  It wraps
the existing adapter and evaluation scripts with the gates needed for a valid
large-scale benchmark: one smoke episode first, no planner fallback, slow
visual updates only on Qwen calls, and local-policy fast steps in between.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_SMOKE_EPISODE_KEY = "2azQ1b91cZZ:11"
DEFAULT_MODEL_PATH = (
    "/ssd/dingmuhe/Embodied-task/JanusVLN/JanusVLN_Model/misstl/JanusVLN_Extra"
)
TRACE_RELATIVE_PATH = Path("harness_traces") / "harness_trace_rank0.jsonl"
SUMMARY_JSON_NAME = "large_eval_summary.json"
REPORT_MD_NAME = "large_eval_report.md"


@dataclass
class LargeEvalConfig:
    gpu: str = "1"
    host: str = "127.0.0.1"
    port: int = 8011
    episodes: int = 30
    max_steps: int = 400
    data_path: Optional[str] = None
    model_path: str = DEFAULT_MODEL_PATH
    model: str = "qwen/qwen3.5-flash"
    model_provider: str = "qwen_api"
    fast_mode: str = "memory_guided_policy_fast"
    image_interval_steps: int = 20
    model_max_images: int = 2
    qwen_retries: int = 1
    qwen_retry_backoff_s: float = 2.0
    agent_timeout_s: int = 90
    agent_max_input_tokens: int = 50000
    adapter_timeout_s: int = 30
    gateway_timeout_s: int = 300
    master_port_smoke: int = 20501
    master_port_eval: int = 20502
    smoke_episode_key: str = DEFAULT_SMOKE_EPISODE_KEY
    smoke_output: Path = field(
        default_factory=lambda: Path(
            "results/clawnav_openclaw_qwen_memory_guided_fast_smoke_largeeval"
        )
    )
    eval_output: Path = field(
        default_factory=lambda: Path(
            "results/clawnav_openclaw_qwen_memory_guided_fast_largeeval"
        )
    )
    poll_seconds: float = 20.0
    startup_timeout_s: float = 90.0
    abort_on_invalid: bool = True
    reuse_adapter: bool = False
    skip_smoke: bool = False
    dry_run: bool = False

    @property
    def gateway_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _stringify_env(env: Dict[str, Any]) -> Dict[str, str]:
    return {key: str(value) for key, value in env.items() if value is not None}


def build_adapter_env(config: LargeEvalConfig) -> Dict[str, str]:
    """Environment for scripts/start_openclaw_cli_plan_gateway.sh."""
    return _stringify_env(
        {
            "HOST": config.host,
            "PORT": config.port,
            "OPENCLAW_QWEN_API_RETRIES": config.qwen_retries,
            "OPENCLAW_QWEN_API_RETRY_BACKOFF_S": _format_number(
                config.qwen_retry_backoff_s
            ),
            "OPENCLAW_PLANNER_MODE": "model",
            "OPENCLAW_MODEL_PROVIDER": config.model_provider,
            "OPENCLAW_MODEL": config.model,
            "OPENCLAW_MODEL_MAX_IMAGES": config.model_max_images,
            "OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS": config.image_interval_steps,
            "OPENCLAW_MODEL_FAST_MODE": config.fast_mode,
            "OPENCLAW_MODEL_FAST_USE_MEMORY_CONTEXT": 1,
            "OPENCLAW_AGENT_TIMEOUT": config.agent_timeout_s,
            "OPENCLAW_AGENT_MAX_INPUT_TOKENS": config.agent_max_input_tokens,
            "OPENCLAW_GATEWAY_TIMEOUT": config.adapter_timeout_s,
            "NO_PROXY": "127.0.0.1,localhost,::1",
            "no_proxy": "127.0.0.1,localhost,::1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )


def build_eval_env(
    config: LargeEvalConfig,
    *,
    output_path: Path,
    episodes: int,
    master_port: int,
    episode_keys: str = "",
) -> Dict[str, str]:
    """Environment for scripts/evaluation_openclaw_gateway.sh."""
    env: Dict[str, Any] = {
        "MODEL_PATH": config.model_path,
        "OUTPUT_PATH": output_path,
        "CUDA_VISIBLE_DEVICES": config.gpu,
        "MASTER_PORT": master_port,
        "OPENCLAW_GATEWAY_URL": config.gateway_url,
        "OPENCLAW_GATEWAY_TIMEOUT": config.gateway_timeout_s,
        "OPENCLAW_ENFORCE_TIMEOUT_BUDGET": 1,
        "CHECK_GATEWAY": 1,
        "REQUIRE_GATEWAY": 1,
        "REQUIRE_OPENCLAW_CLI_ADAPTER": 1,
        "HARNESS_DEBUG_MAX_EPISODES": episodes,
        "MAX_STEPS": config.max_steps,
        "NO_PROXY": "127.0.0.1,localhost,::1",
        "no_proxy": "127.0.0.1,localhost,::1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if episode_keys:
        env["HARNESS_EPISODE_KEYS"] = episode_keys
    if config.data_path:
        env["DATA_PATH"] = config.data_path
        if not episode_keys:
            env["HARNESS_USE_DEFAULT_EPISODE_KEYS"] = 0
    return _stringify_env(env)


def build_adapter_command() -> List[str]:
    return ["bash", "scripts/start_openclaw_cli_plan_gateway.sh"]


def build_eval_command() -> List[str]:
    return ["bash", "scripts/evaluation_openclaw_gateway.sh"]


def collect_run_summary(
    run_dir: Path | str,
    *,
    expected_episodes: Optional[int] = None,
    model_max_images: Optional[int] = None,
    allow_partial_jsonl: bool = False,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    trace_path = _find_trace_path(run_path)
    trace_rows, trace_errors = _load_jsonl(trace_path, allow_partial=allow_partial_jsonl)
    result_rows, result_errors = _load_jsonl(
        run_path / "result.json", allow_partial=allow_partial_jsonl
    )

    mode_counts: Counter[str] = Counter()
    fallback_examples: List[Dict[str, Any]] = []
    planner_fallback_examples: List[Dict[str, Any]] = []
    cli_fallback_examples: List[Dict[str, Any]] = []
    runtime_error_examples: List[Dict[str, Any]] = []
    fast_qwen_examples: List[Dict[str, Any]] = []
    fast_not_local_policy_examples: List[Dict[str, Any]] = []
    fast_model_not_skipped_examples: List[Dict[str, Any]] = []
    fast_model_image_examples: List[Dict[str, Any]] = []
    visual_not_qwen_examples: List[Dict[str, Any]] = []
    visual_not_qwen_authority_examples: List[Dict[str, Any]] = []
    visual_no_image_examples: List[Dict[str, Any]] = []
    visual_image_over_budget_examples: List[Dict[str, Any]] = []
    direct_janus_loaded_examples: List[Dict[str, Any]] = []
    direct_navigation_policy_skill_examples: List[Dict[str, Any]] = []
    direct_planned_navigation_policy_skill_examples: List[Dict[str, Any]] = []
    qwen_candidate_final_action_diff_examples: List[Dict[str, Any]] = []
    fast_assembled_tokens: List[float] = []
    visual_assembled_tokens: List[float] = []
    visual_provider_tokens: List[float] = []
    policy_backend_counts: Counter[str] = Counter()
    final_action_source_counts: Counter[str] = Counter()

    qwen_api_called = 0
    qwen_candidate_requested = 0
    qwen_model_called = 0
    qwen_failure_count = 0
    qwen_missing_confidence_count = 0
    qwen_missing_stop_evidence_count = 0
    qwen_visual_summary_present_count = 0
    qwen_candidate_final_action_diff_count = 0
    direct_policy_rows = 0
    janus_loaded_count = 0
    navigation_policy_skill_called_count = 0
    fast_qwen_api_called = 0
    visual_qwen_api_called = 0
    max_model_image_count = 0
    fallback_count = 0
    planner_fallback_count = 0
    cli_fallback_count = 0
    runtime_error_count = 0

    for row in trace_rows:
        audit = row.get("context_audit") if isinstance(row.get("context_audit"), dict) else {}
        step_mode = str(audit.get("planner_step_mode") or row.get("planner_step_mode") or "")
        policy_backend = str(row.get("policy_backend") or audit.get("policy_backend") or "")
        if policy_backend:
            policy_backend_counts[policy_backend] += 1
        direct_policy = row.get("direct_policy") is True or policy_backend == "qwen_direct"
        if direct_policy:
            direct_policy_rows += 1
            if row.get("qwen_confidence") in (None, ""):
                qwen_missing_confidence_count += 1
            if row.get("qwen_stop_evidence") in (None, ""):
                qwen_missing_stop_evidence_count += 1
            if row.get("qwen_visual_summary_present") is True:
                qwen_visual_summary_present_count += 1
            candidate_action = str(row.get("candidate_action") or "")
            final_action = str(row.get("final_action") or "")
            if candidate_action and final_action and candidate_action != final_action:
                qwen_candidate_final_action_diff_count += 1
                _append_example(qwen_candidate_final_action_diff_examples, row)
        final_action_source = str(row.get("final_action_source") or "")
        if final_action_source:
            final_action_source_counts[final_action_source] += 1
        if row.get("janus_loaded") is True or audit.get("janus_loaded") is True:
            janus_loaded_count += 1
            if direct_policy:
                _append_example(direct_janus_loaded_examples, row)
        planned_tool = str(row.get("planned_tool") or row.get("skill") or "")
        navigation_policy_called = (
            row.get("navigation_policy_skill_called") is True
            or planned_tool == "NavigationPolicySkill"
            and direct_policy
        )
        if navigation_policy_called:
            navigation_policy_skill_called_count += 1
        if direct_policy and row.get("navigation_policy_skill_called") is True:
            _append_example(direct_navigation_policy_skill_examples, row)
        if direct_policy and planned_tool == "NavigationPolicySkill":
            _append_example(direct_planned_navigation_policy_skill_examples, row)
        if audit.get("qwen_candidate_requested") is True:
            qwen_candidate_requested += 1
        if audit.get("qwen_model_called") is True:
            qwen_model_called += 1
        if row.get("qwen_failure") is True or audit.get("qwen_failure") is True:
            qwen_failure_count += 1
        if step_mode:
            mode_counts[step_mode] += 1
        planner_reason = str(row.get("planner_reason") or row.get("reason") or "")
        image_count = _coerce_int(audit.get("model_image_count"), default=0)
        max_model_image_count = max(max_model_image_count, image_count)
        qwen_called = audit.get("qwen_api_called") is True
        if qwen_called:
            qwen_api_called += 1

        if row.get("fallback") is True:
            fallback_count += 1
            _append_example(fallback_examples, row)

        is_cli_fallback = planner_reason.startswith("openclaw_cli_")
        is_planner_fallback = row.get("planner_fallback") is True or is_cli_fallback
        if is_planner_fallback:
            planner_fallback_count += 1
            _append_example(planner_fallback_examples, row)
        if is_cli_fallback:
            cli_fallback_count += 1
            _append_example(cli_fallback_examples, row)
        if _is_openclaw_runtime_error(row, planner_reason):
            runtime_error_count += 1
            _append_example(runtime_error_examples, row)

        assembled_tokens = _coerce_float(audit.get("assembled_prompt_tokens"))
        provider_tokens = _coerce_float(audit.get("provider_input_tokens"))

        if step_mode == "fast_text":
            if assembled_tokens is not None:
                fast_assembled_tokens.append(assembled_tokens)
            if qwen_called:
                fast_qwen_api_called += 1
                _append_example(fast_qwen_examples, row)
            if audit.get("planner_authority") != "local_policy":
                _append_example(fast_not_local_policy_examples, row)
            if audit.get("model_call_skipped") is not True:
                _append_example(fast_model_not_skipped_examples, row)
            if image_count != 0:
                _append_example(fast_model_image_examples, row)
        elif step_mode == "visual_update":
            if assembled_tokens is not None:
                visual_assembled_tokens.append(assembled_tokens)
            if provider_tokens is not None:
                visual_provider_tokens.append(provider_tokens)
            if qwen_called:
                visual_qwen_api_called += 1
            else:
                _append_example(visual_not_qwen_examples, row)
            if audit.get("planner_authority") != "qwen":
                _append_example(visual_not_qwen_authority_examples, row)
            if image_count <= 0:
                _append_example(visual_no_image_examples, row)
            if model_max_images is not None and image_count > model_max_images:
                _append_example(visual_image_over_budget_examples, row)

    success_sum = sum(_coerce_float(row.get("success")) or 0.0 for row in result_rows)
    spl_sum = sum(_coerce_float(row.get("spl")) or 0.0 for row in result_rows)
    result_count = len(result_rows)

    return {
        "run_dir": str(run_path),
        "trace_path": str(trace_path) if trace_path else "",
        "trace_exists": bool(trace_path and trace_path.exists()),
        "result_path": str(run_path / "result.json"),
        "expected_episodes": expected_episodes,
        "model_max_images": model_max_images,
        "result_count": result_count,
        "trace_rows": len(trace_rows),
        "mode_counts": dict(sorted(mode_counts.items())),
        "policy_backend_counts": dict(sorted(policy_backend_counts.items())),
        "final_action_source_counts": dict(sorted(final_action_source_counts.items())),
        "direct_policy_rows": direct_policy_rows,
        "janus_loaded_count": janus_loaded_count,
        "navigation_policy_skill_called_count": navigation_policy_skill_called_count,
        "qwen_candidate_requested": qwen_candidate_requested,
        "qwen_model_called": qwen_model_called,
        "qwen_failure_count": qwen_failure_count,
        "qwen_missing_confidence_count": qwen_missing_confidence_count,
        "qwen_missing_stop_evidence_count": qwen_missing_stop_evidence_count,
        "qwen_visual_summary_present_count": qwen_visual_summary_present_count,
        "qwen_candidate_final_action_diff_count": qwen_candidate_final_action_diff_count,
        "fallback_count": fallback_count,
        "planner_fallback_count": planner_fallback_count,
        "cli_fallback_count": cli_fallback_count,
        "runtime_error_count": runtime_error_count,
        "qwen_api_called": qwen_api_called,
        "fast_qwen_api_called": fast_qwen_api_called,
        "visual_qwen_api_called": visual_qwen_api_called,
        "max_model_image_count": max_model_image_count,
        "success_sum": round(success_sum, 6),
        "spl_sum": round(spl_sum, 6),
        "success_rate": round(success_sum / result_count, 6) if result_count else None,
        "spl_rate": round(spl_sum / result_count, 6) if result_count else None,
        "token_summary": {
            "fast_assembled_prompt_tokens": _numeric_summary(fast_assembled_tokens),
            "visual_assembled_prompt_tokens": _numeric_summary(visual_assembled_tokens),
            "visual_provider_input_tokens": _numeric_summary(visual_provider_tokens),
        },
        "fallback_examples": fallback_examples,
        "planner_fallback_examples": planner_fallback_examples,
        "cli_fallback_examples": cli_fallback_examples,
        "runtime_error_examples": runtime_error_examples,
        "fast_qwen_examples": fast_qwen_examples,
        "fast_not_local_policy_examples": fast_not_local_policy_examples,
        "fast_model_not_skipped_examples": fast_model_not_skipped_examples,
        "fast_model_image_examples": fast_model_image_examples,
        "visual_not_qwen_examples": visual_not_qwen_examples,
        "visual_not_qwen_authority_examples": visual_not_qwen_authority_examples,
        "visual_no_image_examples": visual_no_image_examples,
        "visual_image_over_budget_examples": visual_image_over_budget_examples,
        "direct_janus_loaded_examples": direct_janus_loaded_examples,
        "direct_navigation_policy_skill_examples": direct_navigation_policy_skill_examples,
        "direct_planned_navigation_policy_skill_examples": direct_planned_navigation_policy_skill_examples,
        "qwen_candidate_final_action_diff_examples": qwen_candidate_final_action_diff_examples,
        "trace_parse_errors": trace_errors,
        "result_parse_errors": result_errors,
    }


def validate_large_eval_gate(
    summary: Dict[str, Any], *, require_complete: bool = True
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    expected = summary.get("expected_episodes")
    if require_complete and expected is not None and summary.get("result_count") != expected:
        failures.append(
            {
                "code": "expected_episode_count",
                "message": "result.json does not contain the expected number of episodes",
                "expected": expected,
                "actual": summary.get("result_count"),
            }
        )
    if require_complete and summary.get("trace_rows", 0) <= 0:
        failures.append(
            {
                "code": "missing_trace_rows",
                "message": "harness trace is missing or empty",
                "trace_path": summary.get("trace_path", ""),
            }
        )
    _append_count_failure(
        failures,
        summary,
        key="runtime_error_count",
        code="openclaw_runtime_error",
        message="openclaw_runtime_error rows were present; stop the eval immediately",
        examples_key="runtime_error_examples",
    )
    _append_count_failure(
        failures,
        summary,
        key="fallback_count",
        code="runtime_fallback",
        message="top-level runtime fallback rows were present",
        examples_key="fallback_examples",
    )
    _append_count_failure(
        failures,
        summary,
        key="planner_fallback_count",
        code="planner_fallback",
        message="planner_fallback rows were present",
        examples_key="planner_fallback_examples",
    )
    _append_count_failure(
        failures,
        summary,
        key="cli_fallback_count",
        code="openclaw_cli_fallback",
        message="OpenClaw CLI fallback reason rows were present",
        examples_key="cli_fallback_examples",
    )
    _append_example_failure(
        failures,
        summary,
        key="fast_qwen_examples",
        code="fast_qwen_api_called",
        message="fast_text rows called Qwen; memory_guided_policy_fast should skip Qwen",
    )
    _append_example_failure(
        failures,
        summary,
        key="fast_not_local_policy_examples",
        code="fast_not_local_policy",
        message="fast_text rows were not under local_policy authority",
    )
    _append_example_failure(
        failures,
        summary,
        key="fast_model_not_skipped_examples",
        code="fast_model_not_skipped",
        message="fast_text rows did not mark model_call_skipped=true",
    )
    _append_example_failure(
        failures,
        summary,
        key="fast_model_image_examples",
        code="fast_model_image_count",
        message="fast_text rows carried model images",
    )
    _append_example_failure(
        failures,
        summary,
        key="visual_not_qwen_examples",
        code="visual_not_qwen_called",
        message="visual_update rows did not call Qwen",
    )
    _append_example_failure(
        failures,
        summary,
        key="visual_not_qwen_authority_examples",
        code="visual_not_qwen_authority",
        message="visual_update rows were not under Qwen authority",
    )
    _append_example_failure(
        failures,
        summary,
        key="visual_no_image_examples",
        code="visual_without_image",
        message="visual_update rows had no model image",
    )
    _append_example_failure(
        failures,
        summary,
        key="visual_image_over_budget_examples",
        code="visual_model_image_over_budget",
        message="visual_update rows exceeded OPENCLAW_MODEL_MAX_IMAGES",
    )
    _append_count_failure(
        failures,
        summary,
        key="janus_loaded_count",
        code="direct_janus_loaded",
        message="direct-policy trace rows reported Janus as loaded",
        examples_key="direct_janus_loaded_examples",
    )
    _append_count_failure(
        failures,
        summary,
        key="navigation_policy_skill_called_count",
        code="direct_navigation_policy_skill_called",
        message="direct-policy trace rows called NavigationPolicySkill",
        examples_key="direct_navigation_policy_skill_examples",
    )
    _append_example_failure(
        failures,
        summary,
        key="direct_planned_navigation_policy_skill_examples",
        code="direct_planned_navigation_policy_skill",
        message="direct-policy trace rows planned NavigationPolicySkill",
    )
    if require_complete:
        if summary.get("trace_parse_errors"):
            failures.append(
                {
                    "code": "trace_parse_errors",
                    "message": "trace JSONL had parse errors",
                    "examples": summary.get("trace_parse_errors", [])[:5],
                }
            )
        if summary.get("result_parse_errors"):
            failures.append(
                {
                    "code": "result_parse_errors",
                    "message": "result JSONL had parse errors",
                    "examples": summary.get("result_parse_errors", [])[:5],
                }
            )
    return failures


def render_markdown_report(summary: Dict[str, Any], *, label: str) -> str:
    failures = summary.get("failures")
    if failures is None:
        failures = validate_large_eval_gate(summary)
    status = "VALID_BENCHMARK" if not failures else "INVALID_DIAGNOSTIC_ONLY"
    lines = [
        f"# Memory-Guided Fast {label} Report",
        "",
        f"Status: {status}",
        f"Run dir: `{summary.get('run_dir', '')}`",
        f"Results: {summary.get('result_count')} / expected {summary.get('expected_episodes')}",
        f"Trace rows: {summary.get('trace_rows')}",
        f"Mode counts: `{json.dumps(summary.get('mode_counts', {}), sort_keys=True)}`",
        (
            "Fallbacks: "
            f"runtime={summary.get('fallback_count')}, "
            f"planner={summary.get('planner_fallback_count')}, "
            f"cli={summary.get('cli_fallback_count')}, "
            f"runtime_errors={summary.get('runtime_error_count', 0)}"
        ),
        (
            "Qwen calls: "
            f"total={summary.get('qwen_api_called')}, "
            f"visual_update={summary.get('visual_qwen_api_called')}, "
            f"fast_text={summary.get('fast_qwen_api_called')}"
        ),
        f"Max model images: {summary.get('max_model_image_count')}",
        (
            "Metrics: "
            f"success_sum={summary.get('success_sum')}, "
            f"success_rate={summary.get('success_rate')}, "
            f"spl_rate={summary.get('spl_rate')}"
        ),
        "",
        "## Token Summary",
        "",
    ]
    token_summary = summary.get("token_summary") or {}
    for key in (
        "fast_assembled_prompt_tokens",
        "visual_assembled_prompt_tokens",
        "visual_provider_input_tokens",
    ):
        lines.append(f"- {key}: `{json.dumps(token_summary.get(key, {}), sort_keys=True)}`")

    lines.extend(["", "## Gate Failures", ""])
    if not failures:
        lines.append("- none")
    else:
        for failure in failures:
            lines.append(
                f"- {failure.get('code')}: {failure.get('message')} "
                f"(count={failure.get('count', failure.get('actual', 'n/a'))})"
            )
            for example in failure.get("examples", [])[:3]:
                lines.append(f"  - {_format_example_for_report(example)}")
    lines.append("")
    return "\n".join(lines)


def write_run_audit(
    summary: Dict[str, Any], *, label: str, output_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    run_dir = output_dir or Path(summary["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_with_failures = dict(summary)
    summary_with_failures["failures"] = validate_large_eval_gate(summary)
    summary_with_failures["status"] = (
        "valid_benchmark" if not summary_with_failures["failures"] else "invalid_diagnostic_only"
    )
    json_path = run_dir / SUMMARY_JSON_NAME
    md_path = run_dir / REPORT_MD_NAME
    json_path.write_text(
        json.dumps(summary_with_failures, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown_report(summary_with_failures, label=label),
        encoding="utf-8",
    )
    return json_path, md_path


def run_suite(config: LargeEvalConfig) -> int:
    if config.dry_run:
        _print_dry_run(config)
        return 0

    adapter_process: Optional[subprocess.Popen[Any]] = None
    adapter_log = config.eval_output / "runner_logs" / "adapter.log"
    try:
        if config.reuse_adapter:
            wait_for_gateway(config.gateway_url, timeout_s=config.startup_timeout_s)
        else:
            if _gateway_responds(config.gateway_url):
                raise RuntimeError(
                    f"{config.gateway_url} is already responding. "
                    "Pass --reuse-adapter to use it intentionally, or choose another --port."
                )
            adapter_process = _launch_process(
                build_adapter_command(),
                build_adapter_env(config),
                log_path=adapter_log,
                label="adapter",
            )
            wait_for_gateway(config.gateway_url, timeout_s=config.startup_timeout_s)

        if not config.skip_smoke:
            smoke_code, smoke_summary = _run_eval_phase(
                config,
                label="smoke",
                output_path=config.smoke_output,
                episodes=1,
                master_port=config.master_port_smoke,
                episode_keys=config.smoke_episode_key,
            )
            if smoke_code != 0 or smoke_summary.get("failures"):
                print(
                    f"Smoke failed gate; see {config.smoke_output / REPORT_MD_NAME}",
                    file=sys.stderr,
                )
                return 2

        eval_code, eval_summary = _run_eval_phase(
            config,
            label="eval",
            output_path=config.eval_output,
            episodes=config.episodes,
            master_port=config.master_port_eval,
            episode_keys="",
        )
        if eval_code != 0 or eval_summary.get("failures"):
            print(
                f"Eval is diagnostic-only; see {config.eval_output / REPORT_MD_NAME}",
                file=sys.stderr,
            )
            return 3
        print(f"Eval passed gate; see {config.eval_output / REPORT_MD_NAME}")
        return 0
    finally:
        if adapter_process is not None:
            _terminate_process(adapter_process)


def wait_for_gateway(gateway_url: str, *, timeout_s: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 - local gateway health endpoint.
                f"{gateway_url.rstrip('/')}/health", timeout=5
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
            if data.get("ok") and data.get("service") == "openclaw_cli_plan_gateway":
                return data
            last_error = RuntimeError(f"unexpected health response: {data}")
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"gateway did not become healthy within {timeout_s}s: {last_error}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a smoke-gated memory_guided_policy_fast OpenClaw/Qwen eval and "
            "audit traces for benchmark validity."
        )
    )
    parser.add_argument("--audit-only", type=Path, default=None)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default="qwen/qwen3.5-flash")
    parser.add_argument("--model-max-images", type=int, default=2)
    parser.add_argument("--image-interval-steps", type=int, default=20)
    parser.add_argument("--qwen-retries", type=int, default=1)
    parser.add_argument("--qwen-retry-backoff-s", type=float, default=2.0)
    parser.add_argument("--agent-timeout-s", type=int, default=90)
    parser.add_argument("--agent-max-input-tokens", type=int, default=50000)
    parser.add_argument("--adapter-timeout-s", type=int, default=30)
    parser.add_argument("--gateway-timeout-s", type=int, default=300)
    parser.add_argument("--master-port-smoke", type=int, default=20501)
    parser.add_argument("--master-port-eval", type=int, default=20502)
    parser.add_argument("--smoke-episode-key", default=DEFAULT_SMOKE_EPISODE_KEY)
    parser.add_argument("--smoke-output", type=Path, default=None)
    parser.add_argument("--eval-output", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--startup-timeout-s", type=float, default=90.0)
    parser.add_argument("--reuse-adapter", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--no-abort-on-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> LargeEvalConfig:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_output = args.smoke_output or Path(
        f"results/clawnav_openclaw_qwen_memory_guided_fast_smoke_{timestamp}"
    )
    eval_output = args.eval_output or Path(
        f"results/clawnav_openclaw_qwen_memory_guided_fast_{args.episodes}_{timestamp}"
    )
    return LargeEvalConfig(
        gpu=args.gpu,
        host=args.host,
        port=args.port,
        episodes=args.episodes,
        max_steps=args.max_steps,
        data_path=args.data_path,
        model_path=args.model_path,
        model=args.model,
        image_interval_steps=args.image_interval_steps,
        model_max_images=args.model_max_images,
        qwen_retries=args.qwen_retries,
        qwen_retry_backoff_s=args.qwen_retry_backoff_s,
        agent_timeout_s=args.agent_timeout_s,
        agent_max_input_tokens=args.agent_max_input_tokens,
        adapter_timeout_s=args.adapter_timeout_s,
        gateway_timeout_s=args.gateway_timeout_s,
        master_port_smoke=args.master_port_smoke,
        master_port_eval=args.master_port_eval,
        smoke_episode_key=args.smoke_episode_key,
        smoke_output=smoke_output,
        eval_output=eval_output,
        poll_seconds=args.poll_seconds,
        startup_timeout_s=args.startup_timeout_s,
        abort_on_invalid=not args.no_abort_on_invalid,
        reuse_adapter=args.reuse_adapter,
        skip_smoke=args.skip_smoke,
        dry_run=args.dry_run,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.audit_only is not None:
        expected = args.expected_episodes if args.expected_episodes is not None else args.episodes
        summary = collect_run_summary(
            args.audit_only,
            expected_episodes=expected,
            model_max_images=args.model_max_images,
        )
        summary["failures"] = validate_large_eval_gate(summary)
        write_run_audit(summary, label="audit")
        print(render_markdown_report(summary, label="audit"))
        return 0 if not summary["failures"] else 2
    return run_suite(config_from_args(args))


def _run_eval_phase(
    config: LargeEvalConfig,
    *,
    label: str,
    output_path: Path,
    episodes: int,
    master_port: int,
    episode_keys: str,
) -> Tuple[int, Dict[str, Any]]:
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "runner_logs" / f"{label}.log"
    env = build_eval_env(
        config,
        output_path=output_path,
        episodes=episodes,
        master_port=master_port,
        episode_keys=episode_keys,
    )
    process = _launch_process(
        build_eval_command(),
        env,
        log_path=log_path,
        label=label,
        mirror_stdout=True,
        mirror_line_substrings=("episode_progress",),
    )
    aborted = False
    abort_code = ""
    try:
        while process.poll() is None:
            time.sleep(config.poll_seconds)
            summary = collect_run_summary(
                output_path,
                expected_episodes=episodes,
                model_max_images=config.model_max_images,
                allow_partial_jsonl=True,
            )
            failures = validate_large_eval_gate(summary, require_complete=False)
            has_runtime_error = any(
                failure.get("code") == "openclaw_runtime_error" for failure in failures
            )
            if has_runtime_error or (failures and config.abort_on_invalid):
                aborted = True
                abort_code = "openclaw_runtime_error" if has_runtime_error else "invalid_trace"
                _terminate_process(process)
                break
        returncode = process.poll()
        if returncode is None:
            returncode = process.wait()
        _close_process_log(process)
    finally:
        if process.poll() is None:
            _terminate_process(process)

    summary = collect_run_summary(
        output_path,
        expected_episodes=episodes,
        model_max_images=config.model_max_images,
    )
    failures = validate_large_eval_gate(summary)
    if aborted:
        runtime_abort = abort_code == "openclaw_runtime_error"
        failures.insert(
            0,
            {
                "code": "aborted_on_runtime_error"
                if runtime_abort
                else "aborted_on_invalid_trace",
                "message": (
                    "runner stopped the phase after openclaw_runtime_error appeared in the live trace"
                    if runtime_abort
                    else "runner stopped the phase after live trace gate failure"
                ),
                "count": 1,
            },
        )
    if returncode != 0 and not aborted:
        failures.append(
            {
                "code": "process_failed",
                "message": f"{label} command exited non-zero",
                "returncode": returncode,
                "log_path": str(log_path),
            }
        )
    summary["failures"] = failures
    summary["status"] = "valid_benchmark" if not failures else "invalid_diagnostic_only"
    summary["process_returncode"] = returncode
    summary["log_path"] = str(log_path)
    write_run_audit(summary, label=label)
    return returncode, summary


def _launch_process(
    command: List[str],
    env_updates: Dict[str, str],
    *,
    log_path: Path,
    label: str,
    mirror_stdout: bool = False,
    mirror_line_substrings: Optional[Tuple[str, ...]] = None,
) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(env_updates)
    log_file = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=subprocess.PIPE if mirror_stdout else log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except Exception:
        log_file.close()
        raise
    process._clawnav_log_file = log_file  # type: ignore[attr-defined]
    if mirror_stdout:
        if process.stdout is None:
            log_file.close()
            raise RuntimeError(f"failed to capture {label} stdout for mirroring")
        thread = threading.Thread(
            target=_mirror_process_output,
            args=(process.stdout, log_file, mirror_line_substrings),
            daemon=True,
        )
        thread.start()
        process._clawnav_output_thread = thread  # type: ignore[attr-defined]
    print(f"Started {label}: pid={process.pid}, log={log_path}")
    return process


def _mirror_process_output(
    stream: Any,
    log_file: Any,
    mirror_line_substrings: Optional[Tuple[str, ...]],
) -> None:
    for line in iter(stream.readline, b""):
        log_file.write(line)
        log_file.flush()
        if _should_mirror_line(line, mirror_line_substrings):
            _write_stdout_bytes(line)


def _should_mirror_line(
    line: bytes,
    mirror_line_substrings: Optional[Tuple[str, ...]],
) -> bool:
    if mirror_line_substrings is None:
        return True
    return any(token.encode("utf-8") in line for token in mirror_line_substrings)


def _write_stdout_bytes(chunk: bytes) -> None:
    stdout_buffer = getattr(sys.stdout, "buffer", None)
    if stdout_buffer is not None:
        stdout_buffer.write(chunk)
        stdout_buffer.flush()
        return
    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
    sys.stdout.flush()


def _terminate_process(process: subprocess.Popen[Any], *, grace_s: float = 20.0) -> None:
    if process.poll() is not None:
        _close_process_log(process)
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)
    finally:
        _close_process_log(process)


def _close_process_log(process: subprocess.Popen[Any]) -> None:
    output_thread = getattr(process, "_clawnav_output_thread", None)
    if output_thread is not None:
        output_thread.join(timeout=5)
    stdout_pipe = getattr(process, "stdout", None)
    if stdout_pipe is not None and not stdout_pipe.closed:
        stdout_pipe.close()
    log_file = getattr(process, "_clawnav_log_file", None)
    if log_file is not None and not log_file.closed:
        log_file.close()


def _gateway_responds(gateway_url: str) -> bool:
    try:
        wait_for_gateway(gateway_url, timeout_s=2)
        return True
    except Exception:
        return False


def _find_trace_path(run_path: Path) -> Optional[Path]:
    default = run_path / TRACE_RELATIVE_PATH
    if default.exists():
        return default
    trace_dir = run_path / "harness_traces"
    if trace_dir.exists():
        matches = sorted(trace_dir.glob("harness_trace_rank*.jsonl"))
        if matches:
            return matches[0]
    return default


def _load_jsonl(path: Optional[Path], *, allow_partial: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if path is None or not path.exists():
        return [], []
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if allow_partial and index == len(lines):
                continue
            errors.append({"line": index, "error": str(exc), "text": line[:240]})
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            errors.append({"line": index, "error": "row is not an object"})
    return rows, errors


def _append_count_failure(
    failures: List[Dict[str, Any]],
    summary: Dict[str, Any],
    *,
    key: str,
    code: str,
    message: str,
    examples_key: str,
) -> None:
    count = int(summary.get(key) or 0)
    if count <= 0:
        return
    failures.append(
        {
            "code": code,
            "message": message,
            "count": count,
            "examples": summary.get(examples_key, [])[:5],
        }
    )


def _append_example_failure(
    failures: List[Dict[str, Any]],
    summary: Dict[str, Any],
    *,
    key: str,
    code: str,
    message: str,
) -> None:
    examples = summary.get(key) or []
    if not examples:
        return
    failures.append(
        {
            "code": code,
            "message": message,
            "count": len(examples),
            "examples": examples[:5],
        }
    )


def _is_openclaw_runtime_error(row: Dict[str, Any], planner_reason: str) -> bool:
    error_type = str(row.get("error_type") or "")
    if error_type == "openclaw_runtime_error":
        return True
    runtime_status = str(row.get("runtime_status") or "")
    return (
        runtime_status == "failed"
        and planner_reason.startswith("openclaw_cli_model_fallback:")
    )


def _append_example(examples: List[Dict[str, Any]], row: Dict[str, Any], *, limit: int = 5) -> None:
    if len(examples) >= limit:
        return
    examples.append(_row_example(row))


def _row_example(row: Dict[str, Any]) -> Dict[str, Any]:
    audit = row.get("context_audit") if isinstance(row.get("context_audit"), dict) else {}
    return {
        "scene_id": row.get("scene_id"),
        "episode_id": row.get("episode_id"),
        "step_id": row.get("step_id"),
        "planner_reason": row.get("planner_reason") or row.get("reason", ""),
        "runtime_status": row.get("runtime_status"),
        "error_type": row.get("error_type"),
        "planner_error": audit.get("planner_error")
        or audit.get("error")
        or audit.get("model_error")
        or row.get("error_type", ""),
        "planner_step_mode": audit.get("planner_step_mode"),
        "planner_authority": audit.get("planner_authority"),
        "qwen_api_called": audit.get("qwen_api_called"),
        "model_call_skipped": audit.get("model_call_skipped"),
        "model_image_count": audit.get("model_image_count"),
        "visual_memory_update_status": audit.get("visual_memory_update_status"),
        "provider_input_tokens": audit.get("provider_input_tokens"),
        "assembled_prompt_tokens": audit.get("assembled_prompt_tokens"),
    }


def _format_example_for_report(example: Dict[str, Any]) -> str:
    location = (
        f"{example.get('scene_id')} / {example.get('episode_id')} / "
        f"step {example.get('step_id')}"
    )
    reason = example.get("planner_reason") or ""
    error = example.get("planner_error") or ""
    suffix = f" - {reason}" if reason else ""
    if error:
        suffix += f" - {error}"
    return location + suffix


def _numeric_summary(values: Iterable[float]) -> Dict[str, Any]:
    numbers = list(values)
    if not numbers:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(numbers),
        "min": round(min(numbers), 3),
        "max": round(max(numbers), 3),
        "avg": round(sum(numbers) / len(numbers), 3),
    }


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _print_dry_run(config: LargeEvalConfig) -> None:
    plan = {
        "adapter": {
            "command": build_adapter_command(),
            "env": build_adapter_env(config),
            "log": str(config.eval_output / "runner_logs" / "adapter.log"),
        },
        "smoke": {
            "command": build_eval_command(),
            "env": build_eval_env(
                config,
                output_path=config.smoke_output,
                episodes=1,
                master_port=config.master_port_smoke,
                episode_keys=config.smoke_episode_key,
            ),
        },
        "eval": {
            "command": build_eval_command(),
            "env": build_eval_env(
                config,
                output_path=config.eval_output,
                episodes=config.episodes,
                master_port=config.master_port_eval,
            ),
        },
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
