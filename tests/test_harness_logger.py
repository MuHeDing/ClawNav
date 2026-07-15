import json

from harness.logging.harness_logger import HarnessLogger
from harness.types import VLNState


class FakeQuaternion:
    w = 1.0
    x = 0.0
    y = 0.0
    z = 0.0


class SelfListingQuaternion(FakeQuaternion):
    def tolist(self):
        return self


def make_state():
    return VLNState(
        scene_id="scene1",
        episode_id="episode1",
        instruction="go to kitchen",
        step_id=3,
        current_image="frame",
        online_metrics={"collision": False},
        diagnostics={"distance_to_goal": 1.0, "success": False, "SPL": 0.0},
    )


def test_jsonl_record_is_appended_and_parent_created(tmp_path):
    path = tmp_path / "nested" / "harness_trace_rank0.jsonl"
    logger = HarnessLogger(path)
    logger.log_step(make_state(), intent="act", skill="NavigationPolicySkill")
    logger.log_step(make_state(), intent="act", skill="NavigationPolicySkill")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["scene_id"] == "scene1"


def test_oracle_metrics_are_only_under_diagnostics(tmp_path):
    path = tmp_path / "trace.jsonl"
    logger = HarnessLogger(path)
    logger.log_step(
        make_state(),
        intent="verify_progress",
        skill="ProgressCriticSkill",
        decision_inputs={"used_action_history": True},
    )
    record = json.loads(path.read_text().strip())
    assert record["diagnostics"]["distance_to_goal"] == 1.0
    assert "distance_to_goal" not in record["decision_inputs"]
    assert record["oracle_metrics_used_for_decision"] is False
    assert record["oracle_guard_passed"] is True


def test_logger_serializes_quaternion_like_values_before_json_write(tmp_path):
    logger = HarnessLogger(tmp_path, rank=0)
    state = VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go",
        step_id=1,
        current_image=None,
        diagnostics={"sim_rotation": FakeQuaternion()},
    )

    record = logger.log_step(
        state,
        intent="act",
        skill="QwenDirectPolicy",
        runtime={"context_engine": {"pose": {"rotation": FakeQuaternion()}}},
    )

    assert record["diagnostics"]["sim_rotation"] == [1.0, 0.0, 0.0, 0.0]
    assert record["context_engine"]["pose"]["rotation"] == [1.0, 0.0, 0.0, 0.0]
    saved = json.loads((tmp_path / "harness_trace_rank0.jsonl").read_text())
    assert saved["diagnostics"]["sim_rotation"] == [1.0, 0.0, 0.0, 0.0]


def test_logger_does_not_recurse_when_tolist_returns_self(tmp_path):
    logger = HarnessLogger(tmp_path, rank=0)
    state = VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go",
        step_id=1,
        current_image=None,
        diagnostics={"sim_rotation": SelfListingQuaternion()},
    )

    record = logger.log_step(state, intent="act", skill="QwenDirectPolicy")

    assert record["diagnostics"]["sim_rotation"] == [1.0, 0.0, 0.0, 0.0]


def test_record_contains_memory_source_and_decision_inputs(tmp_path):
    path = tmp_path / "trace.jsonl"
    logger = HarnessLogger(path)
    logger.log_step(
        make_state(),
        intent="recall_memory",
        skill="MemoryQuerySkill",
        memory_backend="fake",
        memory_source="episode-local",
        num_memory_hits=2,
        decision_inputs={"used_memory_consistency": True},
    )
    record = json.loads(path.read_text().strip())
    assert record["memory_source"] == "episode-local"
    assert record["memory_backend"] == "fake"
    assert record["num_memory_hits"] == 2
    assert record["decision_inputs"]["used_memory_consistency"] is True


def test_logger_writes_runtime_trace_metadata(tmp_path):
    logger = HarnessLogger(tmp_path, rank=0)
    state = VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=1,
        current_image=None,
    )

    record = logger.log_step(
        state,
        intent="recall_memory",
        skill="MemoryQuerySkill",
        reason="initial_recall",
        runtime={
            "skill_call_id": "call-1",
            "parent_call_id": "root",
            "tool_schema_version": "phase2.skill_manifest.v1",
            "latency_ms": 2.5,
            "runtime_status": "completed",
        },
    )

    assert record["trace_schema_version"] == "phase2.harness_trace.v2"
    assert record["skill_call_id"] == "call-1"
    assert record["parent_call_id"] == "root"
    assert record["tool_schema_version"] == "phase2.skill_manifest.v1"
    assert record["latency_ms"] == 2.5
    assert record["runtime_status"] == "completed"


def test_logger_writes_openclaw_runtime_metadata(tmp_path):
    logger = HarnessLogger(tmp_path, rank=0)
    state = VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=1,
        current_image=None,
    )

    record = logger.log_step(
        state,
        intent="act",
        skill="NavigationPolicySkill",
        runtime={
            "runtime_mode": "openclaw_bridge",
            "planner_backend": "rule",
            "planned_intent": "act",
            "planned_tool": "NavigationPolicySkill",
            "runtime_executor": "openclaw_habitat",
        },
    )

    assert record["trace_schema_version"] == "phase2.harness_trace.v2"
    assert record["runtime_mode"] == "openclaw_bridge"
    assert record["planner_backend"] == "rule"
    assert record["planned_tool"] == "NavigationPolicySkill"
    assert record["runtime_executor"] == "openclaw_habitat"


def test_logger_writes_openclaw_runtime_defaults(tmp_path):
    logger = HarnessLogger(tmp_path, rank=0)

    record = logger.log_step(
        make_state(),
        intent="act",
        skill="NavigationPolicySkill",
    )

    assert record["runtime_mode"] == ""
    assert record["planner_backend"] == ""
    assert record["planned_intent"] == ""
    assert record["planned_tool"] == ""
    assert record["planner_reason"] == ""
    assert record["runtime_executor"] == ""
    assert record["tool_calls"] == []


def test_logger_writes_recall_usage_metadata(tmp_path):
    logger = HarnessLogger(tmp_path, rank=0)

    record = logger.log_step(
        make_state(),
        intent="recall_memory",
        skill="MemoryQuerySkill",
        runtime={
            "recall_usage": [
                {
                    "event_type": "memory_recall",
                    "query_text": "go to kitchen",
                    "allowed_scopes": ["episode"],
                    "selected_namespace": "episode:scene1:episode1",
                    "num_hits": 1,
                    "hit_ids": ["m1"],
                    "hit_image_paths": ["/tmp/keyframe.png"],
                    "used_by_planner": True,
                    "used_by_policy": False,
                    "used_by_critic": True,
                    "action_changed_after_recall": True,
                }
            ],
            "memory_writes": [
                {
                    "memory_scope": "episode",
                    "memory_namespace": "episode:scene1:episode1",
                    "write_gate": {"curator_decision": "write"},
                }
            ],
        },
    )

    event = record["recall_usage"][0]
    assert event["event_type"] == "memory_recall"
    assert event["used_by_planner"] is True
    assert event["action_changed_after_recall"] is True
    assert record["memory_writes"][0]["memory_namespace"] == "episode:scene1:episode1"


def test_logger_writes_visual_memory_trace_contract(tmp_path):
    logger = HarnessLogger(tmp_path, rank=0)

    record = logger.log_step(
        make_state(),
        intent="recall_memory",
        skill="MemoryQuerySkill",
        runtime={
            "visual_analysis": {
                "ran": True,
                "vlm_latency_ms": 37.5,
                "num_images": 1,
                "image_paths": ["/tmp/keyframe.png"],
            },
            "memory_writes": [
                {
                    "memory_scope": "episode",
                    "memory_namespace": "episode:scene1:episode1",
                    "write_gate": {
                        "curator_decision": "write",
                        "novelty_score": 0.71,
                        "duplicate_of_memory_id": None,
                    },
                }
            ],
            "recall_usage": [
                {
                    "event_type": "memory_recall",
                    "planner_intent_before_recall": "recall_memory",
                    "planner_intent_after_recall": "replan",
                    "action_changed_after_recall": True,
                    "selected_namespace": "episode:scene1:episode1",
                }
            ],
        },
    )

    assert record["visual_analysis"]["ran"] is True
    assert record["visual_analysis"]["vlm_latency_ms"] == 37.5
    assert record["memory_writes"][0]["write_gate"]["novelty_score"] == 0.71
    assert record["recall_usage"][0]["planner_intent_before_recall"] == "recall_memory"
    assert record["recall_usage"][0]["planner_intent_after_recall"] == "replan"
    assert record["recall_usage"][0]["selected_namespace"].startswith("episode:")
    serialized = json.dumps(record)
    assert "image_bytes" not in serialized
    assert record["oracle_guard_passed"] is True
