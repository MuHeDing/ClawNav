import json

from harness.logging.harness_logger import HarnessLogger
from harness.types import VLNState


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
