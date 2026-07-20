from harness.config import HarnessConfig
from harness.types import HarnessDecision, MemoryHit, SkillResult, VLNState


def test_skill_result_helpers():
    ok = SkillResult.ok_result(
        "action", {"action_text": "MOVE_FORWARD"}, confidence=0.7
    )
    assert ok.ok is True
    assert ok.result_type == "action"
    assert ok.payload["action_text"] == "MOVE_FORWARD"
    assert ok.confidence == 0.7

    err = SkillResult.error_result("boom")
    assert err.ok is False
    assert err.error == "boom"


def test_harness_decision_is_structured():
    decision = HarnessDecision(
        intent="recall_memory",
        skill_name="MemoryQuerySkill",
        reason="initial_recall",
        payload={"text": "kitchen"},
    )
    assert decision.intent == "recall_memory"
    assert decision.payload["text"] == "kitchen"


def test_memory_hit_keeps_policy_and_control_fields():
    hit = MemoryHit(
        memory_id="m1",
        memory_type="place",
        name="kitchen",
        confidence=0.8,
        target_pose={"x": 1.0, "y": 2.0},
        evidence_text="kitchen entrance",
        image_path="/tmp/kitchen.jpg",
        memory_source="episode-local",
    )
    assert hit.target_pose["x"] == 1.0
    assert hit.memory_source == "episode-local"


def test_vln_state_separates_online_metrics_and_diagnostics():
    state = VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=0,
        current_image="frame",
        online_metrics={"collision": False},
        diagnostics={"distance_to_goal": 1.0, "success": False},
    )
    assert "distance_to_goal" not in state.online_metrics
    assert state.diagnostics["distance_to_goal"] == 1.0


def test_harness_config_defaults_are_bounded_and_non_oracle():
    cfg = HarnessConfig()
    assert cfg.max_internal_calls_per_step == 3
    assert cfg.recall_interval_steps >= 1
    assert cfg.memory_backend == "fake"
    assert cfg.allow_oracle_metrics_for_decision is False
    assert cfg.staged_visual_memory_enabled is False
    assert cfg.staged_memory_treatment == "on"
    assert cfg.stage_min_translation_m == 0.25
    assert cfg.stage_min_heading_change_deg == 15.0
    assert cfg.staged_memory_event_cap == 64


def test_harness_config_rejects_ablation_without_staged_mode():
    try:
        HarnessConfig(staged_memory_treatment="off_ablation")
    except ValueError as exc:
        assert "staged_visual_memory_enabled" in str(exc)
    else:
        raise AssertionError("expected staged ablation validation failure")


def test_harness_config_accepts_staged_ablation_with_bounded_settings():
    cfg = HarnessConfig(
        policy_backend="qwen_direct",
        dynamic_visual_context_enabled=True,
        staged_visual_memory_enabled=True,
        staged_memory_treatment="off_ablation",
        staged_memory_event_cap=1,
        staged_recovery_retrigger_steps=1,
    )

    assert cfg.staged_memory_treatment == "off_ablation"
    assert cfg.staged_memory_event_cap == 1


def test_harness_config_rejects_staged_mode_outside_qwen_dynamic_contract():
    for overrides in (
        {"staged_visual_memory_enabled": True},
        {
            "policy_backend": "qwen_direct",
            "staged_visual_memory_enabled": True,
        },
    ):
        try:
            HarnessConfig(**overrides)
        except ValueError as exc:
            assert "requires" in str(exc)
        else:
            raise AssertionError("expected staged controller contract failure")
