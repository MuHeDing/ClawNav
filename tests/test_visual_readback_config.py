from types import SimpleNamespace

import pytest

from evaluation_harness import build_harness_config, build_parser
from harness.config import HarnessConfig, validate_visual_readback_config


def make_args(**overrides):
    data = {
        "harness_mode": "memory_recall",
        "harness_memory_backend": "fake",
        "spatial_memory_url": "http://127.0.0.1:8022",
        "memory_manifest_path": "",
        "harness_memory_source": "episode-local",
        "harness_max_internal_calls": 3,
        "harness_recall_interval_steps": 5,
        "expose_sim_pose_online": False,
        "harness_runtime": "phase2",
        "openclaw_workspace_path": "",
        "openclaw_service_registry_path": "",
        "openclaw_service_host": "127.0.0.1",
        "openclaw_planner_backend": "rule",
        "openclaw_gateway_url": "",
        "openclaw_gateway_timeout": 5.0,
        "openclaw_executor_backend": "habitat",
        "openclaw_robot_executor_url": "",
        "openclaw_subagent_backend": "fake",
        "openclaw_enable_subagent_planner": False,
        "openclaw_enable_subagent_critic": False,
        "openclaw_enable_subagent_memory_curator": False,
        "openclaw_allow_planner_action_override": False,
        "visual_readback_mode": None,
        "visual_readback_trigger_policy": None,
        "visual_readback_top_k": None,
        "visual_readback_timeout_ms": None,
        "visual_readback_low_confidence": None,
        "visual_readback_high_confidence": None,
        "visual_readback_shuffle_scope": None,
        "visual_readback_shuffle_seed": None,
        "visual_readback_control_only": None,
        "visual_readback_fixed_case_manifest": None,
        "visual_readback_stop_fallback_policy": None,
        "visual_readback_smoke_seed_memory": None,
        "visual_readback_max_action_overrides_per_episode": None,
        "keyframe_policy_mode": None,
        "keyframe_min_gap_steps": None,
        "keyframe_episode_cap": None,
        "keyframe_coverage_gap_steps": None,
        "keyframe_debug_save_all_eligible": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_visual_readback_defaults_are_off_and_log_only():
    config = build_harness_config(make_args())

    assert config.visual_readback_mode == "off"
    assert config.visual_readback_trigger_policy == "sparse_action_override"
    assert config.visual_readback_top_k == 3
    assert config.visual_readback_timeout_ms == 90000
    assert config.visual_readback_low_confidence == 0.6
    assert config.visual_readback_high_confidence == 0.8
    assert config.visual_readback_shuffle_scope == "same_scene_other_episode"
    assert config.visual_readback_shuffle_seed == ""
    assert config.visual_readback_control_only is True
    assert config.visual_readback_fixed_case_manifest_path == ""
    assert config.visual_readback_stop_fallback_policy == "log_only"
    assert getattr(config, "visual_readback_smoke_seed_memory", None) is False
    assert config.visual_readback_max_action_overrides_per_episode == 1
    assert config.keyframe_policy_mode == "interval"
    assert config.keyframe_min_gap_steps == 5
    assert config.keyframe_episode_cap == 64
    assert config.keyframe_coverage_gap_steps == 20
    assert config.keyframe_debug_save_all_eligible is False


def test_visual_readback_env_values_are_used_when_cli_is_absent(monkeypatch):
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_MODE", "image_read_controller")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_TRIGGER_POLICY", "dense_action_override")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_TOP_K", "5")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_TIMEOUT_MS", "45000")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_LOW_CONFIDENCE", "0.25")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_HIGH_CONFIDENCE", "0.75")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_CONTROL_ONLY", "true")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_STOP_FALLBACK_POLICY", "log_only")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_SMOKE_SEED_MEMORY", "true")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_MAX_ACTION_OVERRIDES_PER_EPISODE", "3")
    monkeypatch.setenv("OPENCLAW_KEYFRAME_POLICY_MODE", "event_gated_smoke")
    monkeypatch.setenv("OPENCLAW_KEYFRAME_MIN_GAP_STEPS", "7")
    monkeypatch.setenv("OPENCLAW_KEYFRAME_EPISODE_CAP", "11")
    monkeypatch.setenv("OPENCLAW_KEYFRAME_COVERAGE_GAP_STEPS", "13")
    monkeypatch.setenv("OPENCLAW_KEYFRAME_DEBUG_SAVE_ALL_ELIGIBLE", "true")

    config = build_harness_config(make_args(harness_memory_backend="spatial_http"))

    assert config.visual_readback_mode == "image_read_controller"
    assert config.visual_readback_trigger_policy == "dense_action_override"
    assert config.visual_readback_top_k == 5
    assert config.visual_readback_timeout_ms == 45000
    assert config.visual_readback_low_confidence == 0.25
    assert config.visual_readback_high_confidence == 0.75
    assert config.visual_readback_control_only is True
    assert config.visual_readback_stop_fallback_policy == "log_only"
    assert getattr(config, "visual_readback_smoke_seed_memory", None) is True
    assert config.visual_readback_max_action_overrides_per_episode == 3
    assert config.keyframe_policy_mode == "event_gated_smoke"
    assert config.keyframe_min_gap_steps == 7
    assert config.keyframe_episode_cap == 11
    assert config.keyframe_coverage_gap_steps == 13
    assert config.keyframe_debug_save_all_eligible is True


def test_visual_readback_env_accepts_adaptive_action_override_budget(monkeypatch):
    monkeypatch.setenv(
        "OPENCLAW_VISUAL_READBACK_MAX_ACTION_OVERRIDES_PER_EPISODE",
        "adaptive",
    )

    config = build_harness_config(make_args())

    assert config.visual_readback_max_action_overrides_per_episode == "adaptive"


def test_visual_readback_cli_values_override_environment(monkeypatch):
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_MODE", "image_read_controller")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_TRIGGER_POLICY", "dense_action_override")
    monkeypatch.setenv("OPENCLAW_VISUAL_READBACK_TOP_K", "5")
    monkeypatch.setenv("OPENCLAW_KEYFRAME_POLICY_MODE", "event_gated_smoke")
    monkeypatch.setenv("OPENCLAW_KEYFRAME_MIN_GAP_STEPS", "9")

    config = build_harness_config(
        make_args(
            harness_memory_backend="spatial_http",
            visual_readback_mode="current_only_controller",
            visual_readback_trigger_policy="sparse_action_override",
            visual_readback_top_k=2,
            keyframe_policy_mode="interval",
            keyframe_min_gap_steps=3,
        )
    )

    assert config.visual_readback_mode == "current_only_controller"
    assert config.visual_readback_trigger_policy == "sparse_action_override"
    assert config.visual_readback_top_k == 2
    assert config.keyframe_policy_mode == "interval"
    assert config.keyframe_min_gap_steps == 3


def test_visual_readback_parser_exposes_cli_flags():
    args = build_parser().parse_args(
        [
            "--model_path",
            "model",
            "--output_path",
            "out",
            "--visual_readback_mode",
            "current_only_controller",
            "--visual_readback_trigger_policy",
            "dense_action_override",
            "--visual_readback_top_k",
            "4",
            "--visual_readback_control_only",
            "false",
            "--visual_readback_stop_fallback_policy",
            "previous_non_stop_else_move_forward",
            "--visual_readback_max_action_overrides_per_episode",
            "adaptive",
            "--keyframe_policy_mode",
            "event_gated_smoke",
            "--keyframe_min_gap_steps",
            "6",
            "--keyframe_episode_cap",
            "12",
            "--keyframe_coverage_gap_steps",
            "18",
            "--keyframe_debug_save_all_eligible",
            "true",
        ]
    )
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert args.visual_readback_mode == "current_only_controller"
    assert args.visual_readback_trigger_policy == "dense_action_override"
    assert args.visual_readback_top_k == 4
    assert args.visual_readback_control_only is False
    assert args.visual_readback_stop_fallback_policy == "previous_non_stop_else_move_forward"
    assert args.visual_readback_max_action_overrides_per_episode == "adaptive"
    assert "--visual_readback_smoke_seed_memory" in options
    assert "--visual_readback_max_action_overrides_per_episode" in options
    assert "--visual_readback_trigger_policy" in options
    assert args.keyframe_policy_mode == "event_gated_smoke"
    assert args.keyframe_min_gap_steps == 6
    assert args.keyframe_episode_cap == 12
    assert args.keyframe_coverage_gap_steps == 18
    assert args.keyframe_debug_save_all_eligible is True
    assert "--keyframe_policy_mode" in options


def smoke_seed_config(mode):
    config = HarnessConfig(
        memory_backend="image_backed_local",
        visual_readback_mode=mode,
    )
    config.visual_readback_smoke_seed_memory = True
    return config


@pytest.mark.parametrize(
    "config,error_text",
    [
        (HarnessConfig(visual_readback_mode="unknown"), "unknown visual_readback_mode"),
        (
            HarnessConfig(visual_readback_trigger_policy="unknown"),
            "unknown visual_readback_trigger_policy",
        ),
        (
            HarnessConfig(
                visual_readback_low_confidence=0.9,
                visual_readback_high_confidence=0.8,
            ),
            "low_confidence",
        ),
        (
            HarnessConfig(
                memory_backend="spatial_http",
                visual_readback_mode="image_read_controller",
                visual_readback_control_only=False,
            ),
            "control_only",
        ),
        (
            HarnessConfig(
                memory_backend="spatial_http",
                visual_readback_mode="current_only_controller",
                visual_readback_control_only=False,
            ),
            "control_only",
        ),
        (
            HarnessConfig(visual_readback_mode="direct_policy_images"),
            "direct_policy_images",
        ),
        (
            HarnessConfig(
                memory_backend="spatial_http",
                visual_readback_mode="shuffled_image_read_controller",
                visual_readback_shuffle_seed="",
            ),
            "shuffle_seed",
        ),
        (
            HarnessConfig(
                memory_backend="fake",
                visual_readback_mode="image_read_controller",
            ),
            "image-backed memory",
        ),
        (
            smoke_seed_config("current_only_controller"),
            "smoke_seed_memory",
        ),
        (
            HarnessConfig(keyframe_policy_mode="event_gated_smoke_gate"),
            "use keyframe_policy_mode=event_gated_smoke",
        ),
        (
            HarnessConfig(keyframe_policy_mode="event_gated_smoke_audit"),
            "use keyframe_policy_mode=event_gated_smoke",
        ),
        (
            HarnessConfig(keyframe_policy_mode="unknown"),
            "unknown keyframe_policy_mode",
        ),
        (
            HarnessConfig(keyframe_min_gap_steps=0),
            "keyframe_min_gap_steps",
        ),
        (
            HarnessConfig(visual_readback_max_action_overrides_per_episode=0),
            "max_action_overrides",
        ),
        (
            HarnessConfig(keyframe_episode_cap=0),
            "keyframe_episode_cap",
        ),
        (
            HarnessConfig(keyframe_coverage_gap_steps=0),
            "keyframe_coverage_gap_steps",
        ),
    ],
)
def test_visual_readback_validation_rejects_invalid_contracts(config, error_text):
    with pytest.raises(ValueError, match=error_text):
        validate_visual_readback_config(config)


def test_event_gated_smoke_config_does_not_require_v1_only_fields():
    config = HarnessConfig(
        memory_backend="image_backed_local",
        visual_readback_mode="off",
        keyframe_policy_mode="event_gated_smoke",
    )

    validate_visual_readback_config(config)


def test_image_read_replan_prompt_is_accepted_as_phase2_mode():
    config = HarnessConfig(
        memory_backend="image_backed_local",
        visual_readback_mode="image_read_replan_prompt",
    )

    validate_visual_readback_config(config)


def test_image_read_action_override_is_accepted_as_phase3_mode():
    config = HarnessConfig(
        memory_backend="image_backed_local",
        visual_readback_mode="image_read_action_override",
    )

    validate_visual_readback_config(config)


def test_visual_readback_validation_can_require_fixed_manifest():
    config = HarnessConfig(
        memory_backend="spatial_http",
        visual_readback_mode="image_read_controller",
        visual_readback_fixed_case_manifest_path="",
    )

    with pytest.raises(ValueError, match="fixed_case_manifest"):
        validate_visual_readback_config(config, require_fixed_manifest=True)
