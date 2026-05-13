import importlib
import sys


def test_evaluation_harness_import_has_no_side_effects(monkeypatch):
    calls = []

    def fake_init(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "utils.dist",
        type("FakeDist", (), {"init_distributed_mode": fake_init})(),
    )
    module = importlib.import_module("evaluation_harness")
    assert hasattr(module, "build_parser")
    assert calls == []


def test_parser_helper_includes_harness_args():
    module = importlib.import_module("evaluation_harness")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model_path",
            "model",
            "--output_path",
            "out",
            "--harness_mode",
            "full",
            "--harness_memory_backend",
            "fake",
            "--memory_manifest_path",
            "",
            "--harness_debug_max_episodes",
            "1",
        ]
    )
    assert args.harness_mode == "full"
    assert args.harness_memory_backend == "fake"
    assert args.memory_manifest_path == ""
    assert args.harness_debug_max_episodes == 1


def test_harness_parser_does_not_require_sparse_or_slow_fast_flags():
    module = importlib.import_module("evaluation_harness")
    args = module.build_parser().parse_args(["--model_path", "model", "--output_path", "out"])
    assert not hasattr(args, "use_llm_adaptive_sparse_attention")
    assert not hasattr(args, "slow_fast_active_memory_reuse")


def test_module_exposes_component_builder_without_loading_model():
    module = importlib.import_module("evaluation_harness")
    assert hasattr(module, "build_harness_components")


def test_openclaw_adapter_import_has_no_runtime_dependency():
    import harness.openclaw.tool_adapter  # noqa: F401


def test_parser_includes_openclaw_runtime_args():
    module = importlib.import_module("evaluation_harness")
    args = module.build_parser().parse_args(
        [
            "--model_path",
            "model",
            "--output_path",
            "out",
            "--harness_runtime",
            "openclaw_bridge",
            "--openclaw_workspace_path",
            "/tmp/openclaw",
            "--openclaw_service_host",
            "127.0.0.1",
            "--openclaw_planner_backend",
            "rule",
        ]
    )

    assert args.harness_runtime == "openclaw_bridge"
    assert args.openclaw_workspace_path == "/tmp/openclaw"
    assert args.openclaw_service_host == "127.0.0.1"
    assert args.openclaw_planner_backend == "rule"


def test_openclaw_runtime_import_has_no_external_dependency():
    import harness.openclaw.runtime  # noqa: F401
