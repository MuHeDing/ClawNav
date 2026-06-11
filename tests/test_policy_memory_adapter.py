from harness.memory.policy_memory_adapter import (
    contains_stop_semantics,
    distill_memory,
    filter_stop_semantics,
    to_navigation_arguments,
)


def test_stop_semantics_are_filtered_from_policy_context():
    result = distill_memory(
        {
            "last_visual_summary": "You reached the destination. Doorway ahead.",
            "last_suggested_subgoal": "stop at the final location",
            "last_qwen_reason": "I think we should stop now.",
        },
        {},
        mode="safe_cue",
        should_filter_stop_semantics=True,
    )

    arguments = to_navigation_arguments(result)

    assert "stop" not in arguments.get("memory_context_text", "").lower()
    assert "destination" not in arguments.get("memory_context_text", "").lower()
    assert result.control_context.requires_stop_verification is True
    assert set(result.control_context.filtered_stop_terms) >= {"stop", "destination"}


def test_original_instruction_stop_words_are_not_filtered():
    result = distill_memory(
        {"last_visual_summary": "arched doorway ahead"},
        {"instruction": "Stop at the chair after entering the room."},
        mode="safe_cue",
        should_filter_stop_semantics=True,
    )

    assert "Stop at the chair" in result.evidence_context.runtime_context.get("instruction", "")
    assert "stop" not in " ".join(result.control_context.filtered_stop_terms).lower()


def test_no_reason_mode_excludes_last_qwen_reason():
    result = distill_memory(
        {
            "last_visual_summary": "red hallway with doorway ahead",
            "last_suggested_subgoal": "continue toward the doorway",
            "last_qwen_reason": "therefore turn right",
        },
        {},
        mode="no_reason",
        should_filter_stop_semantics=False,
    )

    text = to_navigation_arguments(result)["memory_context_text"]

    assert "Cached visual memory: red hallway" in text
    assert "Suggested subgoal: continue toward the doorway" in text
    assert "Last Qwen reason" not in text
    assert result.raw_reason_included is False


def test_safe_cue_mode_uses_landmark_and_direction_only():
    result = distill_memory(
        {
            "last_visual_summary": "A long hallway continues forward toward the arched doorway.",
            "last_suggested_subgoal": "continue toward the arched doorway",
            "last_qwen_reason": "therefore move forward",
        },
        {},
        mode="safe_cue",
        should_filter_stop_semantics=True,
    )

    arguments = to_navigation_arguments(result)

    assert arguments["memory_context_text"].startswith("Navigation cue:")
    assert "arched doorway" in arguments["memory_context_text"]
    assert "Direction hint: forward" in arguments["memory_context_text"]
    assert arguments["active_subgoal"] == "continue toward the arched doorway"
    assert "therefore" not in arguments["memory_context_text"]


def test_safe_cue_drops_unparseable_text_to_evidence_only():
    result = distill_memory(
        {"last_visual_summary": "This looks promising and I am unsure what to do."},
        {},
        mode="safe_cue",
        should_filter_stop_semantics=True,
    )

    assert to_navigation_arguments(result) == {}
    assert result.policy_context.memory_context_text == ""
    assert result.evidence_context.raw_visual_summary.startswith("This looks promising")


def test_safe_cue_filters_action_and_arrival_synonyms():
    filtered, terms = filter_stop_semantics(
        "Turn left and move forward. The goal reached sign is visible. Hallway ahead."
    )

    assert "Turn left" not in filtered
    assert "goal reached" not in filtered
    assert "Hallway ahead" in filtered
    assert "turn left" in terms
    assert "goal reached" in terms


def test_safe_cue_active_subgoal_requires_landmark_or_direction():
    result = distill_memory(
        {"last_suggested_subgoal": "continue carefully"},
        {},
        mode="safe_cue",
        should_filter_stop_semantics=True,
    )

    assert "active_subgoal" not in to_navigation_arguments(result)


def test_off_mode_returns_empty_navigation_arguments():
    result = distill_memory(
        {"last_visual_summary": "doorway ahead", "last_suggested_subgoal": "continue toward doorway"},
        {"memory_context_text": "retrieved doorway"},
        mode="off",
        should_filter_stop_semantics=True,
    )

    assert to_navigation_arguments(result) == {}
    assert result.policy_context_used is False
    assert result.mode == "off"


def test_raw_mode_preserves_current_labels_and_ignores_filter_flag():
    result = distill_memory(
        {
            "last_visual_summary": "reached the destination",
            "last_suggested_subgoal": "stop here",
            "last_qwen_reason": "I think stop is correct",
        },
        {"memory_context_text": "retrieved prior cue"},
        mode="raw",
        should_filter_stop_semantics=True,
    )

    text = to_navigation_arguments(result)["memory_context_text"]

    assert "Cached visual memory: reached the destination" in text
    assert "Suggested subgoal: stop here" in text
    assert "Last Qwen reason: I think stop is correct" in text
    assert "Retrieved memory: retrieved prior cue" in text
    assert result.raw_reason_included is True
    assert result.stop_semantics_filtered is False


def test_control_context_retains_stop_evidence_without_oracle_metrics():
    result = distill_memory(
        {
            "last_visual_summary": "destination ahead",
            "distance_to_goal": 0.2,
            "success": 1,
        },
        {},
        mode="safe_cue",
        should_filter_stop_semantics=True,
    )

    assert result.control_context.requires_stop_verification is True
    assert "distance_to_goal" not in result.control_context.to_dict()
    assert "success" not in result.control_context.to_dict()


def test_policy_context_is_bounded():
    result = distill_memory(
        {
            "last_visual_summary": " ".join(["doorway ahead"] * 80),
            "last_suggested_subgoal": "continue toward the doorway",
        },
        {},
        mode="raw",
        should_filter_stop_semantics=False,
    )

    assert len(to_navigation_arguments(result)["memory_context_text"]) <= 240


def test_contains_stop_semantics_uses_full_contract_terms():
    assert contains_stop_semantics("The task complete marker is visible.")
    assert contains_stop_semantics("Please wait near the doorway.")
    assert contains_stop_semantics("move forward through the room")
