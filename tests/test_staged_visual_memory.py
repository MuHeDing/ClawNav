from harness.openclaw.staged_visual_memory import (
    StagedMemoryTrigger,
    StagedVisualMemoryCoordinator,
    build_stage_memory_query,
    run_forced_memory_operations,
)
from harness.openclaw.instruction_stages import InstructionStage, TransitionType


def test_trigger_reasons_coalesce_in_controller_priority_order():
    coordinator = StagedVisualMemoryCoordinator()
    event = coordinator.observe(
        "scene::episode",
        4,
        stage_entry=True,
        no_progress=True,
        turn_loop=True,
    )
    event = coordinator.observe(
        "scene::episode",
        4,
        stage_completion_candidate=True,
        stop_candidate=True,
    )

    assert event.event_id == "scene::episode:4:1"
    assert event.trigger_reasons == (
        StagedMemoryTrigger.STOP_CANDIDATE,
        StagedMemoryTrigger.STAGE_COMPLETION_CANDIDATE,
        StagedMemoryTrigger.NO_PROGRESS,
        StagedMemoryTrigger.TURN_LOOP,
        StagedMemoryTrigger.STAGE_ENTRY,
    )


def test_sticky_recovery_retriggers_after_cooldown_but_not_same_step():
    coordinator = StagedVisualMemoryCoordinator(recovery_retrigger_steps=3)

    first = coordinator.observe("s::e", 1, no_progress=True)
    assert coordinator.observe("s::e", 1, no_progress=True) is first
    assert coordinator.observe("s::e", 2, no_progress=True) is None
    assert coordinator.observe("s::e", 3, no_progress=True) is None
    second = coordinator.observe("s::e", 4, no_progress=True)

    assert second is not None
    assert second.event_id == "s::e:4:2"


def test_event_cap_returns_suppression_record_without_hiding_candidate():
    coordinator = StagedVisualMemoryCoordinator(event_cap=1)
    assert coordinator.observe("s::e", 0, stage_entry=True).selected is True

    suppressed = coordinator.observe("s::e", 1, stop_candidate=True)

    assert suppressed.selected is False
    assert suppressed.status == "suppressed:event_cap"
    assert suppressed.trigger_reasons == (StagedMemoryTrigger.STOP_CANDIDATE,)


def test_stage_query_is_bounded_and_uses_stage_not_full_instruction():
    stage = InstructionStage(
        stage_id="stage_02",
        order=2,
        route_clause="pass the red sofa",
        transition_type=TransitionType.PASS,
        expected_landmarks=("red sofa",),
        completion_cues=("sofa behind the agent",),
    )

    query = build_stage_memory_query(
        stage,
        [StagedMemoryTrigger.STAGE_COMPLETION_CANDIDATE],
        visual_summary="red sofa is on the right",
        stage_relation="at",
    )

    assert "pass the red sofa" in query
    assert "red sofa" in query
    assert "stage_completion_candidate" in query
    assert "the complete secret instruction" not in query
    assert len(query) <= 800


def test_forced_operations_attempt_registry_and_query_independently_and_retry_error():
    calls = {"registry": 0, "query": 0}

    def registry_call():
        calls["registry"] += 1
        raise RuntimeError("registry unavailable")

    def query_call():
        calls["query"] += 1
        if calls["query"] == 1:
            raise TimeoutError("transient")
        return {"ok": True, "payload": {"memory_hits": []}}

    result = run_forced_memory_operations(
        registry_call=registry_call,
        query_call=query_call,
        treatment="on",
    )

    assert calls == {"registry": 1, "query": 2}
    assert result.registry_status == "failed"
    assert result.query_status == "empty"
    assert result.query_attempts == 2


def test_schema_empty_query_is_not_retried_and_ablation_calls_neither_layer():
    calls = {"registry": 0, "query": 0}

    def registry_call():
        calls["registry"] += 1
        return None

    def query_call():
        calls["query"] += 1
        return {"ok": True, "payload": {"memory_hits": []}}

    empty = run_forced_memory_operations(
        registry_call=registry_call,
        query_call=query_call,
        treatment="on",
    )
    disabled = run_forced_memory_operations(
        registry_call=registry_call,
        query_call=query_call,
        treatment="off_ablation",
    )

    assert empty.query_attempts == 1
    assert calls == {"registry": 1, "query": 1}
    assert disabled.registry_status == "disabled_ablation"
    assert disabled.query_status == "disabled_ablation"
