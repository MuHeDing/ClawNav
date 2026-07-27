from types import SimpleNamespace

from evaluation_harness import build_harness_components
from harness.skills.visual_memory_read import (
    QwenDirectVisualReadbackAdapter,
    VisualMemoryReadSkill,
)
from harness.types import MemoryHit, VLNState


class FakeReadbackAdapter:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def read(self, request):
        self.calls.append(request)
        return self.response


class FakeQwenModelClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def run(self, prompt, image_paths, model, timeout_s):
        self.calls.append(
            {
                "prompt": prompt,
                "image_paths": list(image_paths),
                "model": model,
                "timeout_s": timeout_s,
            }
        )
        return self.response


def make_state():
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to the kitchen",
        step_id=4,
        current_image=None,
    )


def make_args(tmp_path, **overrides):
    data = {
        "harness_mode": "memory_recall",
        "harness_memory_backend": "image_backed_local",
        "spatial_memory_url": "http://127.0.0.1:8022",
        "memory_manifest_path": "",
        "harness_memory_source": "episode-local",
        "harness_max_internal_calls": 3,
        "harness_recall_interval_steps": 5,
        "harness_trace_rank": 0,
        "output_path": str(tmp_path),
        "num_history": 8,
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
        "visual_readback_mode": "image_read_controller",
        "visual_readback_top_k": None,
        "visual_readback_timeout_ms": None,
        "visual_readback_low_confidence": None,
        "visual_readback_high_confidence": None,
        "visual_readback_shuffle_scope": None,
        "visual_readback_shuffle_seed": None,
        "visual_readback_control_only": None,
        "visual_readback_fixed_case_manifest": None,
        "visual_readback_stop_fallback_policy": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_build_components_registers_visual_memory_read_skill_when_enabled(tmp_path):
    components = build_harness_components(make_args(tmp_path), model=None)

    assert "VisualMemoryReadSkill" in components["skill_registry"].names()


def test_default_visual_memory_read_skill_uses_direct_qwen_flash_adapter(monkeypatch):
    monkeypatch.delenv("OPENCLAW_VISUAL_READBACK_MODEL", raising=False)

    skill = VisualMemoryReadSkill()

    assert isinstance(skill.adapter, QwenDirectVisualReadbackAdapter)
    assert skill.adapter.model == "qwen/qwen3.5-flash"


def test_qwen_direct_adapter_reads_attached_images_and_returns_json(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    client = FakeQwenModelClient(
        {
            "outputs": [
                {
                    "text": """```json
{
  "verifier_labels": ["route_conflict"],
  "matched_memory_ids": ["m1"],
  "visual_evidence": "The current and memory views have different landmarks.",
  "readback_confidence": 0.87,
  "verifier_confidence": 0.84,
  "visual_grounding_status": "grounded"
}
```"""
                }
            ]
        }
    )
    adapter = QwenDirectVisualReadbackAdapter(
        model_client=client,
        model="qwen/qwen3.5-flash",
        timeout_ms=120000,
    )
    skill = VisualMemoryReadSkill(adapter=adapter)

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [
                {
                    "memory_id": "m1",
                    "image_path": str(memory),
                    "confidence": 0.7,
                }
            ],
            "candidate_action": "TURN_RIGHT",
            "trigger_rule": "decision_point",
        },
    )

    assert result.payload["read_status"] == "completed"
    assert result.payload["matched_memory_ids"] == ["m1"]
    assert result.payload["verifier_labels"] == ["route_conflict"]
    assert result.payload["visual_grounding_status"] == "grounded"
    assert result.payload["readback_confidence"] == 0.87
    assert client.calls[0]["image_paths"] == [str(current), str(memory)]
    assert client.calls[0]["model"] == "qwen/qwen3.5-flash"
    assert client.calls[0]["timeout_s"] == 120.0
    assert "There are exactly 2 attached images" in client.calls[0]["prompt"]
    assert '"matched_memory_ids": ["m1"]' in client.calls[0]["prompt"]


def test_visual_memory_read_skill_round_trips_instruction_stage_progress(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    adapter = FakeReadbackAdapter(
        {
            "verifier_labels": ["route_conflict"],
            "matched_memory_ids": ["m1"],
            "visual_evidence": "The current view has reached the hallway.",
            "candidate_action_valid": False,
            "should_override": True,
            "recommended_action": "TURN_LEFT",
            "decision_scope": "immediate_next_action",
            "action_confidence": 0.9,
            "current_view_candidate_invalid": True,
            "current_view_recommended_action_supported": True,
            "evidence_sources": [
                {
                    "field_name": "visual_evidence",
                    "item_index": 0,
                    "evidence_source_type": "current",
                    "readback_slot_id": 0,
                }
            ],
            "readback_confidence": 0.9,
            "verifier_confidence": 0.9,
            "predicted_current_stage_id": 1,
            "completed_stage_ids": [0],
            "next_stage_id": 2,
            "stage_evidence": "current image shows the hallway stage",
        }
    )
    skill = VisualMemoryReadSkill(adapter=adapter)
    instruction_stages = [
        {"stage_id": 0, "stage_text": "leave the closet", "status": "active"},
        {"stage_id": 1, "stage_text": "enter the hallway", "status": "pending"},
    ]

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [
                MemoryHit(
                    memory_id="m1",
                    memory_type="semantic_frame",
                    name="closet",
                    confidence=1.0,
                    image_path=str(memory),
                )
            ],
            "candidate_action": "MOVE_FORWARD",
            "trigger_rule": "motion_consistency_check",
            "instruction_stages": instruction_stages,
            "current_stage_id": 0,
        },
    )

    assert adapter.calls[0]["instruction_stages"] == instruction_stages
    assert adapter.calls[0]["current_stage_id"] == 0
    assert result.payload["instruction_stages"] == instruction_stages
    assert result.payload["current_stage_id"] == 0
    assert result.payload["predicted_current_stage_id"] == 1
    assert result.payload["completed_stage_ids"] == [0]
    assert result.payload["next_stage_id"] == 2
    assert result.payload["stage_evidence"] == "current image shows the hallway stage"
    assert result.payload["current_view_candidate_invalid"] is True
    assert result.payload["current_view_recommended_action_supported"] is True


def test_qwen_direct_adapter_normalizes_grounded_mismatch_status(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    client = FakeQwenModelClient(
        {
            "outputs": [
                {
                    "text": """{
  "verifier_labels": ["route_conflict"],
  "matched_memory_ids": ["m1"],
  "visual_evidence": "The two images show conflicting layouts.",
  "readback_confidence": 0.9,
  "verifier_confidence": 0.9,
  "visual_grounding_status": "mismatch"
}"""
                }
            ]
        }
    )
    skill = VisualMemoryReadSkill(
        adapter=QwenDirectVisualReadbackAdapter(model_client=client)
    )

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [{"memory_id": "m1", "image_path": str(memory)}],
            "candidate_action": "TURN_RIGHT",
            "trigger_rule": "decision_point",
        },
    )

    assert result.payload["visual_grounding_status"] == "grounded"


def test_qwen_direct_adapter_normalizes_conflict_detected_status(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    client = FakeQwenModelClient(
        {
            "outputs": [
                {
                    "text": """{
  "verifier_labels": ["route_conflict"],
  "matched_memory_ids": ["m1"],
  "visual_evidence": "A visual conflict was detected across the two images.",
  "readback_confidence": 0.9,
  "verifier_confidence": 0.9,
  "visual_grounding_status": "conflict_detected"
}"""
                }
            ]
        }
    )
    skill = VisualMemoryReadSkill(
        adapter=QwenDirectVisualReadbackAdapter(model_client=client)
    )

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [{"memory_id": "m1", "image_path": str(memory)}],
            "candidate_action": "TURN_RIGHT",
            "trigger_rule": "decision_point",
        },
    )

    assert result.payload["visual_grounding_status"] == "grounded"


def test_visual_memory_read_skill_normalizes_successful_grounding_aliases(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")

    for status in ("Grounded", "clear", "complete", "confirmed", "verified", "successful"):
        skill = VisualMemoryReadSkill(
            adapter=FakeReadbackAdapter(
                {
                    "verifier_labels": ["route_conflict"],
                    "matched_memory_ids": ["m1"],
                    "visual_evidence": "Images were read.",
                    "readback_confidence": 0.9,
                    "verifier_confidence": 0.9,
                    "visual_grounding_status": status,
                }
            )
        )

        result = skill.run(
            make_state(),
            {
                "current_image_path": str(current),
                "memory_hits": [{"memory_id": "m1", "image_path": str(memory)}],
                "candidate_action": "TURN_RIGHT",
                "trigger_rule": "decision_point",
            },
        )

        assert result.payload["visual_grounding_status"] == "grounded"


def test_visual_memory_read_skill_normalizes_valid_adapter_payload(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    adapter = FakeReadbackAdapter(
        {
            "verifier_labels": ["route_conflict"],
            "matched_memory_ids": ["m1", "not-attached"],
            "visual_evidence": "The recalled image shows the left opening.",
            "audit_action_hint": "TURN_LEFT",
            "candidate_action_valid": False,
            "should_override": True,
            "invalid_reason": "route_conflict",
            "recommended_action": "TURN_LEFT",
            "decision_scope": "immediate_next_action",
            "action_confidence": 0.82,
            "readback_confidence": 0.81,
            "retrieval_confidence": 0.72,
            "verifier_confidence": 0.8,
        }
    )
    skill = VisualMemoryReadSkill(adapter=adapter)

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [
                MemoryHit(
                    memory_id="m1",
                    memory_type="semantic_frame",
                    name="left opening",
                    confidence=0.72,
                    image_path=str(memory),
                )
            ],
            "candidate_action": "TURN_RIGHT",
            "trigger_rule": "decision_point",
        },
    )

    payload = result.payload
    assert result.ok is True
    assert payload["read_status"] == "completed"
    assert payload["actually_read_image_paths"] == [str(current), str(memory)]
    assert payload["model_image_count"] == 2
    assert payload["matched_memory_ids"] == ["m1"]
    assert payload["verifier_labels"] == ["route_conflict"]
    assert payload["audit_action_hint"] == "TURN_LEFT"
    assert payload["candidate_action_valid"] is False
    assert payload["should_override"] is True
    assert payload["invalid_reason"] == "route_conflict"
    assert payload["recommended_action"] == "TURN_LEFT"
    assert payload["decision_scope"] == "immediate_next_action"
    assert payload["action_confidence"] == 0.82
    assert payload["readback_confidence"] == 0.81
    assert adapter.calls[0]["candidate_action"] == "TURN_RIGHT"
    assert adapter.calls[0]["trigger_rule"] == "decision_point"


def test_visual_memory_read_skill_counts_valid_memory_evidence_source(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    skill = VisualMemoryReadSkill(
        adapter=FakeReadbackAdapter(
            {
                "verifier_labels": ["route_conflict"],
                "matched_memory_ids": ["m1"],
                "visual_evidence": "The retrieved memory shows the turn.",
                "evidence_sources": [
                    {
                        "field_name": "visual_evidence",
                        "item_index": 0,
                        "evidence_source_type": "memory",
                        "memory_id": "m1",
                    }
                ],
            }
        )
    )

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [{"memory_id": "m1", "image_path": str(memory)}],
            "candidate_action": "TURN_RIGHT",
            "trigger_rule": "decision_point",
        },
    )

    assert result.payload["memory_evidence_used_count"] == 1
    assert result.payload["current_only_evidence_count"] == 0
    assert result.payload["ambiguous_evidence_count"] == 0
    assert result.payload["invalid_evidence_source_count"] == 0


def test_visual_memory_read_skill_counts_current_only_evidence_source(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    skill = VisualMemoryReadSkill(
        adapter=FakeReadbackAdapter(
            {
                "verifier_labels": ["goal_not_visible"],
                "visual_evidence": "The current view does not show the goal.",
                "evidence_sources": [
                    {
                        "field_name": "visual_evidence",
                        "item_index": 0,
                        "evidence_source_type": "current",
                        "readback_slot_id": 0,
                    }
                ],
            }
        )
    )

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [{"memory_id": "m1", "image_path": str(memory)}],
            "candidate_action": "STOP",
            "trigger_rule": "risky_stop",
        },
    )

    assert result.payload["memory_evidence_used_count"] == 0
    assert result.payload["current_only_evidence_count"] == 1
    assert result.payload["ambiguous_evidence_count"] == 0


def test_visual_memory_read_skill_counts_legacy_string_evidence_as_ambiguous(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    skill = VisualMemoryReadSkill(
        adapter=FakeReadbackAdapter(
            {
                "verifier_labels": ["route_conflict"],
                "visual_evidence": "A previous view shows the doorway.",
            }
        )
    )

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [{"memory_id": "m1", "image_path": str(memory)}],
            "candidate_action": "TURN_LEFT",
            "trigger_rule": "decision_point",
        },
    )

    assert result.payload["memory_evidence_used_count"] == 0
    assert result.payload["ambiguous_evidence_count"] == 1


def test_visual_memory_read_skill_rejects_unattached_or_duplicate_sources(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    skill = VisualMemoryReadSkill(
        adapter=FakeReadbackAdapter(
            {
                "verifier_labels": ["route_conflict"],
                "visual_evidence": "The retrieved image supports this.",
                "evidence_sources": [
                    {
                        "field_name": "visual_evidence",
                        "item_index": 0,
                        "evidence_source_type": "memory",
                        "memory_id": "not-attached",
                    },
                    {
                        "field_name": "visual_evidence",
                        "item_index": 1,
                        "evidence_source_type": "memory",
                        "memory_id": "m1",
                    },
                    {
                        "field_name": "visual_evidence",
                        "item_index": 1,
                        "evidence_source_type": "memory",
                        "memory_id": "m1",
                    },
                ],
            }
        )
    )

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [{"memory_id": "m1", "image_path": str(memory)}],
            "candidate_action": "TURN_LEFT",
            "trigger_rule": "decision_point",
        },
    )

    assert result.payload["memory_evidence_used_count"] == 1
    assert result.payload["invalid_evidence_source_count"] == 2


def test_visual_memory_read_skill_fails_closed_on_non_json_response(tmp_path):
    current = tmp_path / "current.png"
    memory = tmp_path / "memory.png"
    current.write_text("current", encoding="utf-8")
    memory.write_text("memory", encoding="utf-8")
    skill = VisualMemoryReadSkill(adapter=FakeReadbackAdapter("not json"))

    result = skill.run(
        make_state(),
        {
            "current_image_path": str(current),
            "memory_hits": [
                {
                    "memory_id": "m1",
                    "image_path": str(memory),
                    "confidence": 0.6,
                }
            ],
            "candidate_action": "STOP",
            "trigger_rule": "risky_stop",
        },
    )

    assert result.ok is True
    assert result.payload["read_status"] == "failed"
    assert result.payload["verifier_labels"] == ["insufficient_evidence"]
    assert result.payload["matched_memory_ids"] == []
    assert result.payload["readback_confidence"] == 0.0
    assert result.payload["error_type"] == "parse_failure"
