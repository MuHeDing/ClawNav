import importlib
import sys

import pytest


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
            "--harness_episode_keys",
            "2azQ1b91cZZ:11,zsNo4HB9uLZ:2",
        ]
    )
    assert args.harness_mode == "full"
    assert args.harness_memory_backend == "fake"
    assert args.memory_manifest_path == ""
    assert args.harness_debug_max_episodes == 1
    assert args.harness_episode_keys == "2azQ1b91cZZ:11,zsNo4HB9uLZ:2"
    assert args.harness_stream_video is False


def test_parser_can_enable_harness_stream_video_separately_from_save_video():
    module = importlib.import_module("evaluation_harness")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model_path",
            "model",
            "--output_path",
            "out",
            "--save_video",
            "--harness_stream_video",
        ]
    )

    assert args.save_video is True
    assert args.harness_stream_video is True


def test_parser_allows_qwen_direct_without_model_path():
    module = importlib.import_module("evaluation_harness")
    parser = module.build_parser()

    args = parser.parse_args(
        [
            "--policy_backend",
            "qwen_direct",
            "--output_path",
            "out",
        ]
    )

    module.validate_args(args)
    assert args.policy_backend == "qwen_direct"
    assert args.model_path == ""


def test_validate_args_requires_model_path_for_janus_policy():
    module = importlib.import_module("evaluation_harness")
    parser = module.build_parser()
    args = parser.parse_args(["--output_path", "out"])

    with pytest.raises(SystemExit):
        module.validate_args(args)


def test_parser_rejects_non_positive_max_steps():
    module = importlib.import_module("evaluation_harness")
    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model_path",
                "model",
                "--output_path",
                "out",
                "--max_steps",
                "0",
            ]
        )


def test_filter_harness_episodes_by_scene_episode_keys():
    module = importlib.import_module("evaluation_harness")

    class Episode:
        def __init__(self, scene_id, episode_id):
            self.scene_id = scene_id
            self.episode_id = episode_id

    first = Episode("data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb", 11)
    second = Episode("data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb", 2)
    skipped = Episode("data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb", 10)

    selected = module.filter_harness_episodes_by_keys(
        [skipped, second, first],
        "2azQ1b91cZZ:11,zsNo4HB9uLZ:2",
    )

    assert selected == [first, second]


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
