import json
import subprocess

import requests

from harness.openclaw.openclaw_cli_plan_gateway import OpenClawCliPlanPlanner, QwenApiModelClient


class FakeOpenClawRunner:
    def __init__(self, returncode=0, stdout=None, stderr=""):
        self.returncode = returncode
        self.stdout = stdout or json.dumps({"ok": True})
        self.stderr = stderr
        self.calls = []

    def __call__(self, args, timeout_s):
        self.calls.append((args, timeout_s))
        return subprocess.CompletedProcess(
            args,
            self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


class FakeVisualAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze(self, image_paths):
        self.calls.append(list(image_paths))
        return [
            {
                "image_path": path,
                "caption": f"caption for {path}",
                "visual_observation": f"doorway visible in {path}",
                "landmarks": ["doorway"],
                "objects": ["door"],
                "spatial_cues": ["doorway ahead"],
                "navigation_relevance": "stable landmark",
                "confidence": 0.8,
                "analysis_metadata": {
                    "vlm_latency_ms": 12.5,
                    "visual_model": "qwen/qwen3.5-flash",
                    "visual_mode": "describe",
                    "cache_hit": path.endswith("key1.png"),
                    "error": False,
                },
            }
            for path in image_paths
        ]


class FakeHttpResponse:
    def __init__(self, data, status_code=200, text=""):
        self._data = data
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._data


class SequenceSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.posts = []

    def post(self, *args, **kwargs):
        self.posts.append({"args": args, "kwargs": kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeModelClient:
    def __init__(self, response=None):
        self.responses = list(response) if isinstance(response, list) else None
        self.response = response or {
            "ok": True,
            "outputs": [
                {
                    "text": json.dumps(
                        {
                            "intent": "act",
                            "tool_name": "NavigationPolicySkill",
                            "arguments": {"action_text": "TURN_RIGHT"},
                            "reason": "direct qwen api",
                        }
                    )
                }
            ],
            "usage": {"input": 321, "output": 12, "totalTokens": 333},
        }
        self.calls = []

    def run(self, prompt, image_paths, model, timeout_s):
        response = self.responses.pop(0) if self.responses else self.response
        self.calls.append(
            {
                "prompt": prompt,
                "image_paths": list(image_paths),
                "model": model,
                "timeout_s": timeout_s,
            }
        )
        return response


class FailingModelClient:
    def __init__(self, error="qwen request timed out"):
        self.error = error
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
        raise RuntimeError(self.error)


def model_response(arguments=None, reason="direct qwen api"):
    return {
        "ok": True,
        "outputs": [
            {
                "text": json.dumps(
                    {
                        "intent": "act",
                        "tool_name": "NavigationPolicySkill",
                        "arguments": arguments or {"action_text": "TURN_RIGHT"},
                        "reason": reason,
                    }
                )
            }
        ],
        "usage": {"input": 321, "output": 12, "totalTokens": 333},
    }


def test_cli_plan_gateway_health_uses_openclaw_gateway_call():
    runner = FakeOpenClawRunner(stdout=json.dumps({"ok": True, "defaultAgentId": "main"}))
    planner = OpenClawCliPlanPlanner(run_openclaw=runner)

    payload = planner.health_payload()

    assert payload["ok"] is True
    assert payload["service"] == "openclaw_cli_plan_gateway"
    assert runner.calls[0][0][:4] == ["openclaw", "gateway", "call", "health"]


def test_cli_plan_gateway_plan_requires_openclaw_health_and_returns_plan_intent():
    runner = FakeOpenClawRunner()
    planner = OpenClawCliPlanPlanner(run_openclaw=runner, recall_interval_steps=5)

    decision = planner.plan_payload(
        {
            "state": {
                "instruction": "go to kitchen",
                "step_id": 0,
            }
        }
    )

    assert decision["intent"] == "recall_memory"
    assert decision["tool_name"] == "MemoryQuerySkill"
    assert decision["arguments"]["text"] == "go to kitchen"
    assert runner.calls[0][0][:4] == ["openclaw", "gateway", "call", "health"]


def test_cli_plan_gateway_raises_when_openclaw_gateway_is_unavailable():
    runner = FakeOpenClawRunner(returncode=1, stderr="connection refused")
    planner = OpenClawCliPlanPlanner(run_openclaw=runner)

    try:
        planner.plan_payload({"state": {"instruction": "go", "step_id": 1}})
    except RuntimeError as exc:
        assert "connection refused" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_cli_plan_gateway_agent_mode_calls_openclaw_agent_and_parses_json_text():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "write_memory",
                                "tool_name": "MemoryWriteSkill",
                                "arguments": {"summary": "saw a kitchen"},
                                "reason": "landmark_keyframe",
                            }
                        )
                    }
                ],
                "meta": {"agentMeta": {"provider": "qwen", "model": "qwen3.5-flash"}},
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_id="main",
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go to kitchen", "step_id": 3},
            "runtime_context": {"keyframe_candidate": {"image_path": "frame.png"}},
        }
    )

    assert decision["intent"] == "write_memory"
    assert decision["tool_name"] == "MemoryWriteSkill"
    assert decision["arguments"]["summary"] == "saw a kitchen"
    assert runner.calls[0][0][:5] == ["openclaw", "agent", "--agent", "main", "--json"]
    assert "--session-id" in runner.calls[0][0]


def test_cli_plan_gateway_model_mode_calls_openclaw_model_run_and_parses_json_text():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "ok": True,
                "capability": "model.run",
                "provider": "qwen",
                "model": "qwen3.5-flash",
                "outputs": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "forward"},
                                "reason": "stateless model",
                            }
                        )
                    }
                ],
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="model",
        openclaw_model_provider="openclaw_cli",
        openclaw_model="qwen/qwen3.5-flash",
    )

    decision = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go to kitchen",
                "step_id": 3,
            },
            "runtime_context": {
                "run_id": "results/run",
                "current_image_path": "/tmp/current.png",
            },
        }
    )

    assert decision["intent"] == "act"
    assert decision["arguments"]["action_text"] == "MOVE_FORWARD"
    args = runner.calls[0][0]
    assert args[:5] == ["openclaw", "capability", "model", "run", "--json"]
    assert args[args.index("--model") + 1] == "qwen/qwen3.5-flash"
    prompt = args[args.index("--prompt") + 1]
    assert "Payload:" in prompt
    assert "Allowed intents and tools:" not in prompt
    assert "openclaw_session" not in " ".join(args)

    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["context_profile"] == "plan_model"
    assert audit["openclaw_session_mode"] == "stateless_model"
    assert audit["openclaw_session_id"] == ""
    assert audit["provider_input_tokens"] is None
    assert audit["provider_usage_source"] == "unavailable"
    assert audit["estimated_provider_input_tokens"] == audit["assembled_prompt_tokens"]


def test_cli_plan_gateway_model_mode_can_call_direct_qwen_api_without_openclaw_cli(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    runner = FakeOpenClawRunner()
    model_client = FakeModelClient()
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model="qwen/qwen3.5-flash",
        model_client=model_client,
    )

    decision = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go to kitchen",
                "step_id": 3,
            },
            "runtime_context": {"current_image_path": str(current)},
        }
    )

    assert decision["intent"] == "act"
    assert decision["arguments"]["action_text"] == "TURN_RIGHT"
    assert runner.calls == []
    assert model_client.calls[0]["model"] == "qwen/qwen3.5-flash"
    assert model_client.calls[0]["image_paths"] == [str(current)]
    assert "Payload:" in model_client.calls[0]["prompt"]

    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["model_provider"] == "qwen_api"
    assert audit["context_profile"] == "plan_model"
    assert audit["openclaw_session_mode"] == "stateless_model"
    assert audit["openclaw_session_id"] == ""
    assert audit["provider_input_tokens"] == 321


def test_cli_plan_gateway_qwen_direct_attaches_map_view_before_history_and_current_last(tmp_path):
    map_view = tmp_path / "map.png"
    memory0 = tmp_path / "memory0.png"
    memory1 = tmp_path / "memory1.png"
    current_dir = tmp_path / "openclaw_current_frames" / "scene-a" / "episode-1"
    current = current_dir / "step_000005.png"
    for path in (map_view, memory0, memory1, current):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient()
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model="qwen/qwen3.5-flash",
        openclaw_model_max_images=4,
        model_client=model_client,
        policy_backend="qwen_direct",
    )

    decision = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene-a",
                "episode_id": "episode-1",
                "instruction": "go to kitchen",
                "step_id": 5,
            },
            "runtime_context": {
                "current_image_path": str(current),
                "retrieved_memory_image_paths": [str(memory0), str(memory1)],
                "map_assist_mode": "floorplan_map_assisted",
                "map_frame_interval_steps": 5,
                "input_regime": "rgb_plus_privileged_floorplan_pose",
                "map_context": {
                    "mode": "floorplan_map_assisted",
                    "input_regime": "rgb_plus_privileged_floorplan_pose",
                    "map_frame_interval_steps": 5,
                    "map_step_id": 5,
                    "map_frame_due": True,
                    "map_available": True,
                    "map_source": "habitat_pathfinder_navmesh",
                    "pose_source": "sim_agent_state_odometry",
                    "map_safety": {"contains_goal": False},
                    "map_image_label": "map_view",
                    "internal_only": {
                        "map_image_path": str(map_view),
                        "map_image_hash": "abc123",
                    },
                },
            },
        }
    )

    assert decision["intent"] == "act"
    assert model_client.calls[0]["image_paths"] == [
        str(map_view),
        str(memory0),
        str(memory1),
        str(current),
    ]
    prompt = model_client.calls[0]["prompt"]
    assert "map_view floorplan" in prompt
    assert str(map_view) not in prompt
    assert "internal_only" not in prompt
    assert "map_image_path" not in prompt
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["map_assist_mode"] == "floorplan_map_assisted"
    assert audit["map_frame_due"] is True
    assert audit["map_view_image_count"] == 1
    assert audit["model_image_sources"] == [
        "map_view",
        "openclaw_retrieved_memory",
        "openclaw_retrieved_memory",
        "current",
    ]
    assert audit["current_image_last"] is True


def test_cli_plan_gateway_qwen_direct_does_not_attach_stale_map_on_non_due_step(tmp_path):
    map_view = tmp_path / "map.png"
    current = tmp_path / "openclaw_current_frames" / "scene-a" / "episode-1" / "step_000006.png"
    for path in (map_view, current):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient()
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model="qwen/qwen3.5-flash",
        openclaw_model_max_images=4,
        model_client=model_client,
        policy_backend="qwen_direct",
    )

    decision = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene-a",
                "episode_id": "episode-1",
                "instruction": "go to kitchen",
                "step_id": 6,
            },
            "runtime_context": {
                "current_image_path": str(current),
                "map_assist_mode": "floorplan_map_assisted",
                "map_frame_interval_steps": 5,
                "map_context": {
                    "mode": "floorplan_map_assisted",
                    "map_frame_interval_steps": 5,
                    "map_step_id": 6,
                    "map_frame_due": False,
                    "map_available": False,
                    "internal_only": {"map_image_path": str(map_view)},
                },
            },
        }
    )

    assert decision["intent"] == "act"
    assert model_client.calls[0]["image_paths"] == [str(current)]
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["map_frame_due"] is False
    assert audit["map_view_image_count"] == 0
    assert audit["model_image_sources"] == ["current"]
    assert audit["provider_usage_source"] == "reported"


def test_cli_plan_gateway_qwen_direct_normalizes_concise_action_json(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    response = {
        "ok": True,
        "outputs": [
            {
                "text": json.dumps(
                    {
                        "action_text": "turn left",
                        "confidence": 0.82,
                        "visual_summary": "hallway with doorway ahead",
                        "progress_state": "following_instruction",
                        "stop_evidence": "none",
                        "current_target": "doorway",
                        "target_relation": "doorway centered ahead",
                        "semantic_stop_state": "not_ready",
                        "reason": "turn toward the visible doorway",
                    }
                )
            }
        ],
        "usage": {"input": 321, "output": 12, "totalTokens": 333},
    }
    model_client = FakeModelClient(response=response)
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        openclaw_model="qwen/qwen3.5-flash",
        model_client=model_client,
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go to kitchen", "step_id": 3},
            "runtime_context": {"current_image_path": str(current)},
        }
    )

    assert decision["intent"] == "act"
    assert decision["tool_name"] == "QwenDirectPolicy"
    assert decision["arguments"]["action_text"] == "TURN_LEFT"
    assert decision["arguments"]["confidence"] == 0.82
    assert decision["arguments"]["stop_evidence"] == "none"
    assert decision["arguments"]["current_target"] == "doorway"
    assert decision["arguments"]["target_relation"] == "doorway centered ahead"
    assert decision["arguments"]["semantic_stop_state"] == "not_ready"
    assert decision["arguments"]["reason"] == "turn toward the visible doorway"
    prompt = model_client.calls[0]["prompt"]
    assert "QwenDirectPolicy" in prompt
    assert "NavigationPolicySkill" not in prompt
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["policy_backend"] == "qwen_direct"
    assert audit["planner_authority"] == "qwen"
    assert audit["qwen_candidate_requested"] is True
    assert audit["qwen_model_called"] is True
    assert audit["qwen_provider"] == "qwen_api"
    assert audit["qwen_api_called"] is True


def test_cli_plan_gateway_qwen_direct_prompt_defines_action_selection_rules(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response={
            "ok": True,
            "outputs": [
                {
                    "text": json.dumps(
                        {
                            "action_text": "move forward",
                            "confidence": 0.7,
                            "visual_summary": "hallway ahead",
                            "progress_state": "following_instruction",
                            "stop_evidence": "none",
                            "reason": "test prompt contract",
                        }
                    )
                }
            ],
            "usage": {"input": 111, "output": 9, "totalTokens": 120},
        }
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "walk across the floor and wait the archway", "step_id": 4},
            "runtime_context": {"current_image_path": str(current)},
        }
    )

    prompt = model_client.calls[0]["prompt"]
    assert "semantic_stop_state" in prompt
    assert "current_target" in prompt
    assert "target_relation" in prompt
    assert "TURN_LEFT or TURN_RIGHT" in prompt
    assert "route or landmark is off-center" in prompt
    assert "MOVE_FORWARD is valid only" in prompt
    assert "path clear alone is not enough" in prompt
    assert "MOVE_FORWARD advances 0.25 meters" in prompt
    assert "TURN_LEFT and TURN_RIGHT rotate 15 degrees" in prompt
    assert "one more 0.25-meter MOVE_FORWARD" in prompt
    assert "STOP requires semantic arrival" in prompt
    assert "visible_not_reached" in prompt
    assert "at_or_inside_target" in prompt
    assert "beside_target" in prompt
    assert "outside face of an archway, doorway, or entrance" in prompt
    assert "visual_summary must mention observed intermediate landmarks" in prompt
    assert "QwenDirectPolicy is the only action-producing policy" in prompt


def test_cli_plan_gateway_qwen_direct_prompt_handles_blocked_stop_feedback(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response={
            "ok": True,
            "outputs": [
                {
                    "text": json.dumps(
                        {
                            "action_text": "turn left",
                            "confidence": 0.7,
                            "visual_summary": "archway is off center",
                            "progress_state": "needs_reorientation",
                            "stop_evidence": "none",
                            "reason": "blocked stop requires a non-stop correction",
                        }
                    )
                }
            ],
            "usage": {"input": 111, "output": 9, "totalTokens": 120},
        }
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "walk across the floor and wait the archway", "step_id": 4},
            "runtime_context": {
                "current_image_path": str(current),
                "blocked_stop_feedback": {
                    "blocked_action": "STOP",
                    "allowed_actions": [
                        "MOVE_FORWARD",
                        "TURN_LEFT",
                        "TURN_RIGHT",
                    ],
                    "reason": "STOP lacked current arrival evidence.",
                },
            },
        }
    )

    prompt = model_client.calls[0]["prompt"]
    assert "When blocked_stop_feedback is present" in prompt
    assert "do not choose STOP" in prompt
    assert "Use MOVE_FORWARD only when the current image clearly shows" in prompt
    assert "otherwise use TURN_LEFT or TURN_RIGHT" in prompt


def test_cli_plan_gateway_qwen_direct_prompt_handles_forward_stall_feedback(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response={
            "ok": True,
            "outputs": [
                {
                    "text": json.dumps(
                        {
                            "action_text": "turn right",
                            "confidence": 0.7,
                            "visual_summary": "archway is off center",
                            "progress_state": "reorienting after no progress",
                            "stop_evidence": "none",
                            "reason": "blocked forward requires a turn",
                        }
                    )
                }
            ],
            "usage": {"input": 111, "output": 9, "totalTokens": 120},
        }
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "walk across the floor and wait the archway", "step_id": 4},
            "runtime_context": {
                "current_image_path": str(current),
                "forward_stall_feedback": {
                    "blocked_action": "MOVE_FORWARD",
                    "allowed_actions": ["TURN_LEFT", "TURN_RIGHT"],
                    "reason": "Repeated forward actions showed no visual progress.",
                },
            },
        }
    )

    prompt = model_client.calls[0]["prompt"]
    assert "When forward_stall_feedback is present" in prompt
    assert "do not choose MOVE_FORWARD" in prompt
    assert "choose only TURN_LEFT or TURN_RIGHT" in prompt


def test_cli_plan_gateway_qwen_direct_prompt_handles_stop_verification_feedback(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient()
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "wait at the archway", "step_id": 15},
            "runtime_context": {
                "current_image_path": str(current),
                "stop_verification_feedback": {
                    "blocked_action": "STOP",
                    "allowed_actions": ["STOP", "TURN_LEFT", "TURN_RIGHT"],
                    "reason": "Verify structural arrival without translating.",
                },
            },
        }
    )

    prompt = model_client.calls[0]["prompt"]
    assert "When stop_verification_feedback is present" in prompt
    assert "do not choose MOVE_FORWARD" in prompt
    assert "choose STOP only" in prompt
    assert "otherwise choose TURN_LEFT or TURN_RIGHT" in prompt


def test_cli_plan_gateway_qwen_direct_prompt_includes_route_progress_contract(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient()
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {
                "instruction": (
                    "turn round and walk past the billiard table, then stop by the window"
                ),
                "step_id": 3,
            },
            "runtime_context": {
                "current_image_path": str(current),
                "route_progress": {
                    "turn_round_required": True,
                    "turn_round_completed": False,
                    "heading_change_from_start_deg": 45.0,
                    "required_waypoints": ["billiard table"],
                    "positively_seen_waypoints": [],
                    "passed_waypoints": [],
                    "negated_waypoint_mentions": ["billiard table"],
                },
            },
        }
    )

    prompt = model_client.calls[0]["prompt"]
    assert "When route_progress is present" in prompt
    assert "negated waypoint mentions as unseen" in prompt
    assert "passed_waypoints" in prompt
    assert '"turn_round_completed": false' in prompt
    assert '"negated_waypoint_mentions": ["billiard table"]' in prompt


def test_cli_plan_gateway_qwen_direct_prompt_includes_motion_feedback_without_raw_pose(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient()
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "walk past the billiard table", "step_id": 8},
            "runtime_context": {
                "current_image_path": str(current),
                "motion_feedback_enabled": True,
                "motion_feedback": {
                    "last_action": "MOVE_FORWARD",
                    "expected_effect": "advance about 0.25m",
                    "actual_effect": "blocked",
                    "last_forward_delta_m": 0.02,
                    "distance_source": "rounded_local_odometry",
                    "collision_recent": True,
                    "consecutive_no_progress_forward": 2,
                    "recommended_constraint": "avoid_forward",
                    "sim_position": [1.0, 2.0, 3.0],
                    "sim_rotation": [1.0, 0.0, 0.0, 0.0],
                },
            },
        }
    )

    prompt = model_client.calls[0]["prompt"]
    assert '"motion_feedback"' in prompt
    assert '"actual_effect": "blocked"' in prompt
    assert "When motion_feedback reports actual_effect=blocked" in prompt
    assert "sim_position" not in prompt
    assert "sim_rotation" not in prompt


def test_cli_plan_gateway_qwen_direct_prompt_omits_run_id_and_image_paths(tmp_path):
    run_dir = tmp_path / "qwen_direct_policy_stop_gate_test_case"
    run_dir.mkdir()
    current = run_dir / "current.png"
    memory = run_dir / "memory.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    memory.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response={
            "ok": True,
            "outputs": [
                {
                    "text": json.dumps(
                        {
                            "action_text": "turn right",
                            "confidence": 0.7,
                            "visual_summary": "archway is off center",
                            "progress_state": "needs_reorientation",
                            "stop_evidence": "none",
                            "reason": "current view requires reorientation",
                        }
                    )
                }
            ],
            "usage": {"input": 111, "output": 9, "totalTokens": 120},
        }
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        openclaw_model_max_images=2,
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "walk across the floor and wait the archway", "step_id": 4},
            "runtime_context": {
                "run_id": str(run_dir),
                "current_image_path": str(current),
                "retrieved_memory_images": [
                    {
                        "memory_id": "mem1",
                        "image_path": str(memory),
                        "source": "openclaw_retrieved_memory",
                        "step_id": 0,
                    }
                ],
            },
        }
    )

    prompt = model_client.calls[0]["prompt"]
    assert model_client.calls[0]["image_paths"] == [str(memory), str(current)]
    assert "stop_gate_test" not in prompt
    assert str(run_dir) not in prompt
    assert str(current) not in prompt
    assert str(memory) not in prompt
    assert '"memory_id": "mem1"' in prompt
    assert '"source": "openclaw_retrieved_memory"' in prompt


def test_cli_plan_gateway_qwen_direct_uses_retrieved_memory_images_before_current(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    memory_paths = []
    for index in range(9):
        path = tmp_path / f"memory{index}.png"
        path.write_bytes(b"not-a-real-png-for-command-shape-test")
        memory_paths.append(path)
    response = {
        "ok": True,
        "outputs": [
            {
                "text": json.dumps(
                    {
                        "action_text": "move forward",
                        "confidence": 0.74,
                        "visual_summary": "retrieved doorway context plus current view",
                        "progress_state": "following_instruction",
                        "stop_evidence": "none",
                        "reason": "current view still supports forward progress",
                    }
                )
            }
        ],
        "usage": {"input": 321, "output": 12, "totalTokens": 333},
    }
    model_client = FakeModelClient(response=response)
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        openclaw_model_max_images=8,
        model_client=model_client,
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go to kitchen", "step_id": 3},
            "runtime_context": {
                "current_image_path": str(current),
                "memory_images": [str(path) for path in memory_paths],
                "retrieved_memory_ids": [f"mem{index}" for index in range(9)],
            },
        }
    )

    expected_history = [str(path) for path in memory_paths[:3]]
    assert model_client.calls[0]["image_paths"] == expected_history + [str(current)]
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["model_image_count"] == 4
    assert audit["selected_image_count"] == 4
    assert audit["model_image_sources"] == ["openclaw_retrieved_memory"] * 3 + ["current"]
    assert audit["retrieved_memory_image_count"] == 3
    assert audit["retrieved_memory_ids"] == [f"mem{index}" for index in range(3)]
    assert audit["retrieved_memory_image_paths"] == expected_history
    assert audit["recent_current_image_count"] == 0
    assert audit["recent_current_image_paths"] == []
    assert audit["history_frame_source"] == "openclaw_retrieved_memory"
    assert audit["current_image_last"] is True


def test_cli_plan_gateway_qwen_direct_uses_memory_and_recent_current_sequence(tmp_path):
    frame_dir = tmp_path / "openclaw_current_frames" / "scene" / "episode"
    frame_dir.mkdir(parents=True)
    frames = []
    for index in range(7):
        path = frame_dir / f"step_{index:06d}.png"
        path.write_bytes(b"not-a-real-png-for-command-shape-test")
        frames.append(path)
    memory_paths = []
    for index in range(5):
        path = tmp_path / f"memory{index}.png"
        path.write_bytes(b"not-a-real-png-for-command-shape-test")
        memory_paths.append(path)
    model_client = FakeModelClient(
        response=model_response(
            {
                "action_text": "MOVE_FORWARD",
                "visual_summary": "history memory plus temporal current sequence",
                "reason": "current view stays aligned",
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        openclaw_model_max_images=8,
        model_client=model_client,
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go to kitchen", "step_id": 6},
            "runtime_context": {
                "current_image_path": str(frames[6]),
                "memory_images": [str(path) for path in memory_paths],
                "retrieved_memory_ids": [f"mem{index}" for index in range(5)],
            },
        }
    )

    expected_history = [str(path) for path in memory_paths[:3]]
    expected_recent = [str(path) for path in frames[2:6]]
    assert model_client.calls[0]["image_paths"] == (
        expected_history + expected_recent + [str(frames[6])]
    )
    prompt = model_client.calls[0]["prompt"]
    assert "attached_image_order" in prompt
    expected_sources = ["openclaw_retrieved_memory"] * 3 + ["recent_current"] * 4 + ["current"]
    assert f'"sources": {json.dumps(expected_sources)}' in prompt

    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["model_image_count"] == 8
    assert audit["selected_image_count"] == 8
    assert audit["model_image_sources"] == expected_sources
    assert audit["retrieved_memory_image_count"] == 3
    assert audit["retrieved_memory_ids"] == ["mem0", "mem1", "mem2"]
    assert audit["retrieved_memory_image_paths"] == expected_history
    assert audit["recent_current_image_count"] == 4
    assert audit["recent_current_image_paths"] == expected_recent
    assert audit["history_frame_source"] == "openclaw_retrieved_memory"
    assert audit["current_image_last"] is True


def test_cli_plan_gateway_qwen_direct_forces_visual_update_every_step(tmp_path):
    current0 = tmp_path / "current0.png"
    current1 = tmp_path / "current1.png"
    current0.write_bytes(b"not-a-real-png-for-command-shape-test")
    current1.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response=[
            model_response(
                {
                    "action_text": "TURN_RIGHT",
                    "visual_summary": "fireplace centered, doorway to the right",
                    "reason": "initial visual action",
                }
            ),
            model_response(
                {
                    "action_text": "MOVE_FORWARD",
                    "visual_summary": "current doorway is centered",
                    "reason": "fresh current image action",
                }
            ),
        ]
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        openclaw_model_max_images=2,
        openclaw_model_image_interval_steps=8,
        openclaw_model_fast_mode="qwen_text_only",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {
                "scene_id": "2azQ1b91cZZ",
                "episode_id": "70",
                "instruction": "go through the doorway",
                "step_id": 0,
            },
            "runtime_context": {"current_image_path": str(current0), "run_id": "run-a"},
        }
    )
    decision = planner.plan_payload(
        {
            "state": {
                "scene_id": "2azQ1b91cZZ",
                "episode_id": "70",
                "instruction": "go through the doorway",
                "step_id": 1,
            },
            "runtime_context": {"current_image_path": str(current1), "run_id": "run-a"},
        }
    )

    assert model_client.calls[1]["image_paths"] == [str(current1)]
    assert "No image is attached for this step." not in model_client.calls[1]["prompt"]
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["policy_backend"] == "qwen_direct"
    assert audit["planner_step_mode"] == "visual_update"
    assert audit["fast_break_reason"] == "qwen_direct_visual_update"
    assert audit["model_image_count"] == 1
    assert audit["memory_context_used"] is False


def test_cli_plan_gateway_qwen_direct_invalid_json_returns_qwen_failure():
    model_client = FakeModelClient(
        response={
            "ok": True,
            "outputs": [{"text": "not json"}],
            "usage": {"input": 10, "output": 2, "totalTokens": 12},
        }
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        model_client=model_client,
    )

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    assert decision["intent"] == "act"
    assert decision["tool_name"] == "QwenDirectPolicy"
    assert decision["arguments"]["action_text"] == "STOP"
    assert decision["arguments"]["qwen_failure"] is True
    assert decision["arguments"]["fallback_policy"] == "hard_failure_stop"
    assert "JSON object" in decision["arguments"]["qwen_failure_reason"]
    assert decision["runtime_metadata"]["context_audit"]["qwen_failure"] is True


def test_cli_plan_gateway_qwen_direct_provider_error_returns_qwen_failure():
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        policy_backend="qwen_direct",
        openclaw_model_provider="qwen_api",
        model_client=FailingModelClient("qwen request timed out"),
    )

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    assert decision["tool_name"] == "QwenDirectPolicy"
    assert decision["arguments"]["action_text"] == "STOP"
    assert decision["arguments"]["qwen_failure"] is True
    assert decision["arguments"]["qwen_failure_reason"] == "qwen request timed out"
    assert decision["runtime_metadata"]["context_audit"]["qwen_model_called"] is True


def test_qwen_api_model_client_reads_openclaw_auth_store_without_cli(tmp_path, monkeypatch):
    auth_dir = tmp_path / ".openclaw" / "agents" / "main" / "agent"
    auth_dir.mkdir(parents=True)
    (auth_dir / "auth-profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "qwen:default": {
                        "type": "api_key",
                        "provider": "qwen",
                        "key": "sk-test",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENCLAW_QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    runner = FakeOpenClawRunner()
    client = QwenApiModelClient(run_openclaw=runner)

    client._ensure_config(5)

    assert client.api_key == "sk-test"
    assert runner.calls == []


def test_qwen_api_model_client_retries_dashscope_read_timeout():
    session = SequenceSession(
        [
            requests.exceptions.ReadTimeout("read timed out"),
            FakeHttpResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "intent": "act",
                                        "tool_name": "NavigationPolicySkill",
                                        "arguments": {"action_text": "MOVE_FORWARD"},
                                        "reason": "retry success",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                }
            ),
        ]
    )
    client = QwenApiModelClient(
        api_key="sk-test",
        base_url="https://dashscope.example/v1",
        max_retries=1,
        retry_backoff_s=0,
    )
    client.session = session

    response = client.run(
        prompt="return json",
        image_paths=[],
        model="qwen/qwen3.5-flash",
        timeout_s=90,
    )

    assert len(session.posts) == 2
    assert response["request_attempts"] == 2
    assert response["retry_count"] == 1
    assert response["usage"]["input"] == 12
    assert "MOVE_FORWARD" in response["outputs"][0]["text"]


def test_cli_plan_gateway_health_reports_recommended_timeout_budget(monkeypatch):
    monkeypatch.setenv("OPENCLAW_QWEN_API_RETRIES", "1")
    monkeypatch.setenv("OPENCLAW_QWEN_API_RETRY_BACKOFF_S", "2")
    planner = OpenClawCliPlanPlanner(
        run_openclaw=FakeOpenClawRunner(stdout=json.dumps({"ok": True})),
        planner_mode="model",
        agent_timeout_s=90,
        openclaw_model_provider="qwen_api",
    )

    health = planner.health_payload()

    budget = health["timeout_budget"]
    assert budget["agent_timeout_s"] == 90
    assert budget["qwen_api_retries"] == 1
    assert budget["qwen_api_retry_backoff_s"] == 2
    assert budget["estimated_qwen_wall_timeout_s"] == 182
    assert budget["recommended_gateway_timeout_s"] == 242
    assert budget["recommended_gateway_timeout_s"] > 120


def test_cli_plan_gateway_model_mode_records_reported_usage_when_available():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "ok": True,
                "outputs": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "TURN_LEFT"},
                                "reason": "reported usage",
                            }
                        )
                    }
                ],
                "usage": {"input": 1200, "output": 20, "totalTokens": 1220},
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="model",
        openclaw_model_provider="openclaw_cli",
        openclaw_model="qwen/qwen3.5-flash",
    )

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    audit = decision["runtime_metadata"]["context_audit"]
    assert decision["arguments"]["action_text"] == "TURN_LEFT"
    assert audit["provider_input_tokens"] == 1200
    assert audit["provider_usage_source"] == "reported"
    assert audit["hidden_history_tokens_estimate"] == (
        audit["provider_input_tokens"] - audit["assembled_prompt_tokens"]
    )


def test_cli_plan_gateway_model_mode_attaches_existing_visual_image_files(tmp_path):
    current = tmp_path / "current.png"
    key0 = tmp_path / "key0.png"
    key1 = tmp_path / "key1.png"
    missing = tmp_path / "missing.png"
    for path in (current, key0, key1):
        path.write_bytes(b"not-a-real-png-for-command-shape-test")
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "ok": True,
                "outputs": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "uses attached images",
                            }
                        )
                    }
                ],
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="model",
        openclaw_model_provider="openclaw_cli",
        openclaw_model="qwen/qwen3.5-flash",
        openclaw_model_max_images=3,
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 2},
            "runtime_context": {
                "current_image_path": str(current),
                "recent_keyframe_paths": [str(missing), str(key0), str(key1)],
            },
        }
    )

    args = runner.calls[0][0]
    file_paths = [
        args[index + 1]
        for index, arg in enumerate(args)
        if arg == "--file"
    ]
    assert file_paths == [str(current), str(key1), str(key0)]
    prompt = args[args.index("--prompt") + 1]
    assert "Attached image files correspond to payload image paths." in prompt

    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["model_image_count"] == 3
    assert audit["model_image_paths"] == [str(current), str(key1), str(key0)]
    assert audit["model_missing_image_paths"] == []


def test_cli_plan_gateway_model_mode_does_not_attach_missing_image_files(tmp_path):
    missing = tmp_path / "missing.png"
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "ok": True,
                "outputs": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "missing image skipped",
                            }
                        )
                    }
                ],
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="model",
        openclaw_model_provider="openclaw_cli",
        openclaw_model="qwen/qwen3.5-flash",
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 2},
            "runtime_context": {"current_image_path": str(missing)},
        }
    )

    args = runner.calls[0][0]
    assert "--file" not in args
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["model_image_count"] == 0
    assert audit["model_image_paths"] == []
    assert audit["model_missing_image_paths"] == [str(missing)]


def test_cli_plan_gateway_model_mode_gates_images_and_uses_cached_visual_memory(tmp_path):
    current0 = tmp_path / "current0.png"
    current1 = tmp_path / "current1.png"
    current0.write_bytes(b"not-a-real-png-for-command-shape-test")
    current1.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response=[
            model_response(
                {
                    "action_text": "TURN_RIGHT",
                    "visual_summary": "red hallway with doorway ahead",
                    "suggested_subgoal": "continue toward the doorway",
                },
                reason="visual update",
            ),
            model_response({"action_text": "MOVE_FORWARD"}, reason="fast text"),
        ]
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model="qwen/qwen3.5-flash",
        openclaw_model_max_images=2,
        openclaw_model_image_interval_steps=8,
        openclaw_model_fast_mode="qwen_text_only",
        model_client=model_client,
    )

    first = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go to kitchen",
                "step_id": 0,
            },
            "runtime_context": {"current_image_path": str(current0), "run_id": "run-a"},
        }
    )
    second = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go to kitchen",
                "step_id": 1,
            },
            "runtime_context": {"current_image_path": str(current1), "run_id": "run-a"},
        }
    )

    assert first["arguments"]["action_text"] == "TURN_RIGHT"
    assert second["arguments"]["action_text"] == "MOVE_FORWARD"
    assert model_client.calls[0]["image_paths"] == [str(current0)]
    assert model_client.calls[1]["image_paths"] == []
    assert "red hallway with doorway ahead" in model_client.calls[1]["prompt"]
    assert "No image is attached for this step." in model_client.calls[1]["prompt"]

    first_audit = first["runtime_metadata"]["context_audit"]
    second_audit = second["runtime_metadata"]["context_audit"]
    assert first_audit["planner_step_mode"] == "visual_update"
    assert first_audit["planner_authority"] == "qwen"
    assert first_audit["qwen_api_called"] is True
    assert first_audit["model_image_count"] == 1
    assert first_audit["visual_memory_backend"] == "adapter_episode_local"
    assert first_audit["visual_memory_update_status"] == "updated"
    assert second_audit["planner_step_mode"] == "fast_text"
    assert second_audit["planner_authority"] == "qwen"
    assert second_audit["qwen_api_called"] is True
    assert second_audit["model_image_count"] == 0
    assert second_audit["memory_context_used"] is True
    assert second_audit["visual_memory_age_steps"] == 1


def test_cli_plan_gateway_control_context_can_force_visual_refresh(tmp_path):
    current0 = tmp_path / "current0.png"
    current1 = tmp_path / "current1.png"
    current0.write_bytes(b"not-a-real-png-for-command-shape-test")
    current1.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response=[
            model_response(
                {"action_text": "TURN_RIGHT", "visual_summary": "cached hallway"},
                reason="visual update",
            ),
            model_response(
                {"action_text": "MOVE_FORWARD", "visual_summary": "forced refresh view"},
                reason="forced visual refresh",
            ),
        ]
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model_max_images=2,
        openclaw_model_image_interval_steps=8,
        openclaw_model_fast_mode="qwen_text_only",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"scene_id": "scene/A", "episode_id": "ep:1", "step_id": 0},
            "runtime_context": {"current_image_path": str(current0), "run_id": "run-a"},
        }
    )
    decision = planner.plan_payload(
        {
            "state": {"scene_id": "scene/A", "episode_id": "ep:1", "step_id": 1},
            "runtime_context": {
                "current_image_path": str(current1),
                "run_id": "run-a",
                "control_context": {"force_visual_refresh": True},
            },
        }
    )

    assert model_client.calls[1]["image_paths"] == [str(current1)]
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["planner_step_mode"] == "visual_update"
    assert audit["fast_break_reason"] == "forced_visual_refresh"
    assert audit["model_image_count"] == 1


def test_cli_plan_gateway_model_mode_records_qwen_retry_metadata():
    response = model_response({"action_text": "MOVE_FORWARD"}, reason="retry metadata")
    response["request_attempts"] = 2
    response["retry_count"] = 1
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model="qwen/qwen3.5-flash",
        model_client=FakeModelClient(response=response),
    )

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 0}})

    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["qwen_api_request_attempts"] == 2
    assert audit["qwen_api_retry_count"] == 1


def test_cli_plan_gateway_model_mode_memory_guided_policy_fast_skips_qwen_on_fast_steps(tmp_path):
    current0 = tmp_path / "current0.png"
    current1 = tmp_path / "current1.png"
    current0.write_bytes(b"not-a-real-png-for-command-shape-test")
    current1.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response=[
            model_response(
                {
                    "action_text": "TURN_RIGHT",
                    "visual_summary": "red hallway with doorway ahead",
                    "suggested_subgoal": "continue toward the doorway",
                },
                reason="visual update",
            ),
        ]
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model="qwen/qwen3.5-flash",
        openclaw_model_max_images=2,
        openclaw_model_image_interval_steps=10,
        openclaw_model_fast_mode="memory_guided_policy_fast",
        model_client=model_client,
    )

    first = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go to kitchen",
                "step_id": 0,
            },
            "runtime_context": {"current_image_path": str(current0), "run_id": "run-a"},
        }
    )
    second = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go to kitchen",
                "step_id": 1,
            },
            "runtime_context": {"current_image_path": str(current1), "run_id": "run-a"},
        }
    )

    assert first["arguments"]["action_text"] == "TURN_RIGHT"
    assert len(model_client.calls) == 1
    assert model_client.calls[0]["image_paths"] == [str(current0)]
    assert second["intent"] == "act"
    assert second["tool_name"] == "NavigationPolicySkill"
    assert "action_text" not in second["arguments"]
    assert second["arguments"]["active_subgoal"] == "continue toward the doorway"
    assert "red hallway with doorway ahead" in second["arguments"]["memory_context_text"]

    audit = second["runtime_metadata"]["context_audit"]
    assert audit["planner_step_mode"] == "fast_text"
    assert audit["planner_authority"] == "local_policy"
    assert audit["qwen_api_called"] is False
    assert audit["model_call_skipped"] is True
    assert audit["model_image_count"] == 0
    assert audit["memory_context_used"] is True
    assert audit["fast_policy_mode"] == "memory_guided_policy_fast"


def test_cli_plan_gateway_model_mode_preserves_context_audit_on_model_error(tmp_path):
    current = tmp_path / "current.png"
    current.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FailingModelClient("qwen request timed out")
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model="qwen/qwen3.5-flash",
        openclaw_model_max_images=2,
        openclaw_model_image_interval_steps=10,
        model_client=model_client,
    )

    decision = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go to kitchen",
                "step_id": 0,
            },
            "runtime_context": {"current_image_path": str(current), "run_id": "run-a"},
        }
    )

    assert decision["reason"] == "openclaw_cli_model_fallback:openclaw_cli_initial_recall"
    assert decision["arguments"]["planner_error"] == "qwen request timed out"
    assert model_client.calls[0]["image_paths"] == [str(current)]
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["planner_step_mode"] == "visual_update"
    assert audit["fast_break_reason"] == "initial_step"
    assert audit["model_image_count"] == 1
    assert audit["model_provider"] == "qwen_api"
    assert audit["qwen_api_called"] is True
    assert audit["visual_memory_update_status"] == "error"


def test_cli_plan_gateway_model_mode_keyframe_forces_visual_update_with_current_and_keyframe(tmp_path):
    current0 = tmp_path / "current0.png"
    current1 = tmp_path / "current1.png"
    keyframe = tmp_path / "keyframe.png"
    for path in (current0, current1, keyframe):
        path.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response=[
            model_response(
                {"action_text": "TURN_RIGHT", "visual_summary": "cached hallway"},
                reason="visual update",
            ),
            model_response(
                {"action_text": "MOVE_FORWARD", "visual_summary": "new doorway"},
                reason="keyframe update",
            ),
        ]
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model_max_images=2,
        openclaw_model_image_interval_steps=8,
        openclaw_model_fast_mode="qwen_text_only",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"scene_id": "scene/A", "episode_id": "ep:1", "step_id": 0},
            "runtime_context": {"current_image_path": str(current0), "run_id": "run-a"},
        }
    )
    decision = planner.plan_payload(
        {
            "state": {"scene_id": "scene/A", "episode_id": "ep:1", "step_id": 1},
            "runtime_context": {
                "current_image_path": str(current1),
                "recent_keyframe_paths": [str(keyframe)],
                "keyframe_candidate": {
                    "step_id": 1,
                    "reason": "novel view",
                    "image_path": str(keyframe),
                },
                "run_id": "run-a",
            },
        }
    )

    assert model_client.calls[1]["image_paths"] == [str(current1), str(keyframe)]
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["planner_step_mode"] == "visual_update"
    assert audit["fast_break_reason"] == "keyframe_candidate"
    assert audit["model_image_count"] == 2


def test_cli_plan_gateway_model_mode_visual_memory_is_episode_local(tmp_path):
    ep1_image = tmp_path / "ep1.png"
    ep2_image = tmp_path / "ep2.png"
    ep1_image.write_bytes(b"not-a-real-png-for-command-shape-test")
    ep2_image.write_bytes(b"not-a-real-png-for-command-shape-test")
    model_client = FakeModelClient(
        response=[
            model_response(
                {"action_text": "TURN_RIGHT", "visual_summary": "episode one memory"},
                reason="visual update",
            ),
            model_response(
                {"action_text": "MOVE_FORWARD", "visual_summary": "episode two memory"},
                reason="missing memory update",
            ),
        ]
    )
    planner = OpenClawCliPlanPlanner(
        planner_mode="model",
        openclaw_model_provider="qwen_api",
        openclaw_model_image_interval_steps=8,
        openclaw_model_fast_mode="qwen_text_only",
        model_client=model_client,
    )

    planner.plan_payload(
        {
            "state": {"scene_id": "scene/A", "episode_id": "ep:1", "step_id": 0},
            "runtime_context": {"current_image_path": str(ep1_image), "run_id": "run-a"},
        }
    )
    decision = planner.plan_payload(
        {
            "state": {"scene_id": "scene/A", "episode_id": "ep:2", "step_id": 1},
            "runtime_context": {"current_image_path": str(ep2_image), "run_id": "run-a"},
        }
    )

    assert model_client.calls[1]["image_paths"] == [str(ep2_image)]
    assert "episode one memory" not in model_client.calls[1]["prompt"]
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["planner_step_mode"] == "visual_update"
    assert audit["fast_break_reason"] == "missing_visual_memory"


def test_cli_plan_gateway_model_prompt_is_shorter_than_agent_prompt():
    planner = OpenClawCliPlanPlanner(planner_mode="model")
    prompt_payload = planner._prompt_payload(
        {
            "state": {"instruction": "go to kitchen", "step_id": 2},
            "runtime_context": {
                "run_id": "results/run",
                "current_image_path": "/tmp/current.png",
                "memory_context_text": "m" * 300,
            },
        }
    )

    agent_prompt = planner._agent_prompt_from_prompt_payload(prompt_payload)
    model_prompt = planner._model_prompt_from_prompt_payload(prompt_payload)

    assert len(model_prompt) < len(agent_prompt)
    assert "Allowed intents and tools:" in agent_prompt
    assert "Allowed intents and tools:" not in model_prompt
    assert "write_gate" not in model_prompt


def test_cli_plan_gateway_skips_visual_describe_between_interval_steps():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "interval skip",
                            }
                        )
                    }
                ]
            }
        )
    )
    visual_analyzer = FakeVisualAnalyzer()
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        openclaw_visual_mode="describe",
        openclaw_visual_interval_steps=5,
        visual_analyzer=visual_analyzer,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 3},
            "runtime_context": {"current_image_path": "frame3.png"},
        }
    )

    assert visual_analyzer.calls == []
    prompt = runner.calls[0][0][runner.calls[0][0].index("--message") + 1]
    assert "visual_observations" not in prompt


def test_cli_plan_gateway_describes_visuals_on_interval_steps():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "interval describe",
                            }
                        )
                    }
                ]
            }
        )
    )
    visual_analyzer = FakeVisualAnalyzer()
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        openclaw_visual_mode="describe",
        openclaw_visual_interval_steps=5,
        visual_analyzer=visual_analyzer,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 5},
            "runtime_context": {"current_image_path": "frame5.png"},
        }
    )

    assert visual_analyzer.calls == [["frame5.png"]]
    prompt = runner.calls[0][0][runner.calls[0][0].index("--message") + 1]
    assert "visual_observations" in prompt


def test_cli_plan_gateway_agent_mode_uses_fresh_openclaw_session_per_planner():
    runner_a = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "fresh session",
                            }
                        )
                    }
                ]
            }
        )
    )
    runner_b = FakeOpenClawRunner(stdout=runner_a.stdout)
    planner_a = OpenClawCliPlanPlanner(run_openclaw=runner_a, planner_mode="agent")
    planner_b = OpenClawCliPlanPlanner(run_openclaw=runner_b, planner_mode="agent")

    planner_a.plan_payload({"state": {"instruction": "go", "step_id": 2}})
    planner_b.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    args_a = runner_a.calls[0][0]
    args_b = runner_b.calls[0][0]
    session_id_a = args_a[args_a.index("--session-id") + 1]
    session_id_b = args_b[args_b.index("--session-id") + 1]
    assert session_id_a.startswith("clawnav-scene-")
    assert session_id_b.startswith("clawnav-scene-")
    assert len(session_id_a.rsplit("-", 1)[-1]) == 8


def test_cli_plan_gateway_agent_mode_uses_fresh_session_per_plan_step():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "fresh request",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_session_id="base-session",
        agent_session_timestamp="05221040",
        agent_max_input_tokens=50000,
    )

    planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go",
                "step_id": 2,
            },
            "runtime_context": {"run_id": "results/run"},
        }
    )
    planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go",
                "step_id": 3,
            },
            "runtime_context": {"run_id": "results/run"},
        }
    )

    first_args = runner.calls[0][0]
    second_args = runner.calls[1][0]
    first_session_id = first_args[first_args.index("--session-id") + 1]
    second_session_id = second_args[second_args.index("--session-id") + 1]
    assert first_session_id == "base-session-results_run-scene_A-ep:1-step_2-05221040"
    assert second_session_id == "base-session-results_run-scene_A-ep:1-step_3-05221040"
    assert first_session_id != second_session_id

    first_message = first_args[first_args.index("--message") + 1]
    first_payload = json.loads(first_message.split("Payload:\n", 1)[1])
    assert first_payload["state"]["scene_id"] == "scene/A"
    assert first_payload["state"]["episode_id"] == "ep:1"


def test_cli_plan_gateway_agent_mode_isolates_session_by_scene():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "fresh request",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_session_id="base-session",
        agent_session_timestamp="05221040",
        agent_max_input_tokens=50000,
    )

    payload = {
        "state": {
            "episode_id": "ep:1",
            "instruction": "go",
            "step_id": 2,
        }
    }
    planner.plan_payload({"state": {**payload["state"], "scene_id": "scene/A"}})
    planner.plan_payload({"state": {**payload["state"], "scene_id": "scene/B"}})

    first_args = runner.calls[0][0]
    second_args = runner.calls[1][0]
    first_session_id = first_args[first_args.index("--session-id") + 1]
    second_session_id = second_args[second_args.index("--session-id") + 1]
    assert first_session_id == "base-session-scene_A-ep:1-step_2-05221040"
    assert second_session_id == "base-session-scene_B-ep:1-step_2-05221040"
    assert first_session_id != second_session_id


def test_cli_plan_gateway_agent_mode_normalizes_action_text_argument():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "forward"},
                                "reason": "move through doorway",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(run_openclaw=runner, planner_mode="agent")

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    assert decision["arguments"]["action_text"] == "MOVE_FORWARD"


def test_cli_plan_gateway_agent_mode_infers_clear_action_from_reason():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "replan",
                                "tool_name": "ReplannerSkill",
                                "arguments": {},
                                "reason": "Step 2 MUST be MOVE_FORWARD to exit the loop. Do not issue TURN_LEFT.",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(run_openclaw=runner, planner_mode="agent")

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    assert decision["arguments"]["action_text"] == "MOVE_FORWARD"


def test_cli_plan_gateway_agent_prompt_includes_visual_image_paths():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "use visual context",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(run_openclaw=runner, planner_mode="agent")

    planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 2},
            "runtime_context": {
                "current_image_path": "/tmp/current.png",
                "recent_keyframe_paths": [
                    "/tmp/key0.png",
                    "/tmp/key1.png",
                    "/tmp/key2.png",
                ],
            },
        }
    )

    message = runner.calls[0][0][runner.calls[0][0].index("--message") + 1]
    assert "Visual context image paths" in message
    assert "/tmp/current.png" in message
    assert "/tmp/key1.png" in message
    assert "/tmp/key2.png" in message
    assert "/tmp/key0.png" not in message


def test_cli_plan_gateway_agent_prompt_includes_visual_observations_in_describe_mode():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "use visual observations",
                            }
                        )
                    }
                ]
            }
        )
    )
    visual_analyzer = FakeVisualAnalyzer()
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        openclaw_visual_mode="describe",
        openclaw_visual_max_images=2,
        visual_analyzer=visual_analyzer,
    )

    planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 2},
            "runtime_context": {
                "current_image_path": "/tmp/current.png",
                "recent_keyframe_paths": [
                    "/tmp/key0.png",
                    "/tmp/key1.png",
                ],
            },
        }
    )

    assert visual_analyzer.calls == [["/tmp/current.png", "/tmp/key1.png"]]
    message = runner.calls[0][0][runner.calls[0][0].index("--message") + 1]
    payload_text = message.split("Payload:\n", 1)[1]
    prompt_payload = json.loads(payload_text)
    observations = prompt_payload["runtime_context"]["visual_observations"]
    assert [item["image_path"] for item in observations] == [
        "/tmp/current.png",
        "/tmp/key1.png",
    ]
    assert observations[0]["visual_observation"] == "doorway visible in /tmp/current.png"
    assert observations[0]["landmarks"] == ["doorway"]


def test_cli_plan_gateway_returns_visual_analysis_runtime_metadata():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "use visual observations",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        openclaw_visual_mode="describe",
        openclaw_visual_max_images=2,
        visual_analyzer=FakeVisualAnalyzer(),
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 2},
            "runtime_context": {
                "current_image_path": "/tmp/current.png",
                "recent_keyframe_paths": ["/tmp/key1.png"],
            },
        }
    )

    visual = decision["runtime_metadata"]["visual_analysis"]
    assert visual["ran"] is True
    assert visual["num_images"] == 2
    assert visual["image_paths"] == ["/tmp/current.png", "/tmp/key1.png"]
    assert visual["vlm_latency_ms"] == 12.5
    assert visual["cache_hits"] == 1
    assert visual["failures"] == 0


def test_cli_plan_gateway_agent_prompt_requests_write_gate_for_visual_memory():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "schema check",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(run_openclaw=runner, planner_mode="agent")

    planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    message = runner.calls[0][0][runner.calls[0][0].index("--message") + 1]
    assert "write_gate" in message
    assert "curator_decision" in message
    assert "episode memory" in message
    assert "scene memory" in message


def test_cli_plan_gateway_enriches_write_memory_with_visual_observation_defaults():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "write_memory",
                                "tool_name": "MemoryWriteSkill",
                                "arguments": {"note": "planner wants memory"},
                                "reason": "visual landmark",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        openclaw_visual_mode="describe",
        visual_analyzer=FakeVisualAnalyzer(),
    )

    decision = planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 2},
            "runtime_context": {"current_image_path": "/tmp/current.png"},
        }
    )

    assert decision["intent"] == "write_memory"
    assert decision["arguments"]["caption"] == "caption for /tmp/current.png"
    assert decision["arguments"]["visual_observation"] == "doorway visible in /tmp/current.png"
    assert decision["arguments"]["landmarks"] == ["doorway"]


def test_cli_plan_gateway_agent_prompt_uses_bounded_recall_context():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "minimal context",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(run_openclaw=runner, planner_mode="agent")

    planner.plan_payload(
        {
            "state": {
                "instruction": "go",
                "step_id": 2,
                "last_action": "TURN_LEFT",
                "success": True,
            },
            "runtime_context": {
                "run_id": "results/run",
                "policy_action": "TURN_LEFT",
                "current_image_path": "/tmp/current.png",
                "recent_keyframe_paths": ["/tmp/key0.png"],
                "keyframe_candidate": {
                    "step_id": 2,
                    "reason": "interval",
                    "image_path": "/tmp/current.png",
                    "raw_frame": "x" * 2000,
                },
                "memory_context_text": "m" * 2000,
                "memory_images": ["/tmp/memory0.png", "/tmp/memory1.png", "/tmp/memory2.png"],
                "recent_frames": ["frame"] * 20,
                "distance_to_goal": 1.0,
            },
        }
    )

    message = runner.calls[0][0][runner.calls[0][0].index("--message") + 1]
    payload_text = message.split("Payload:\n", 1)[1]
    prompt_payload = json.loads(payload_text)
    assert prompt_payload == {
        "state": {
            "instruction": "go",
            "last_action": "TURN_LEFT",
            "step_id": 2,
        },
        "runtime_context": {
            "current_image_path": "/tmp/current.png",
            "keyframe_candidate": {
                "image_path": "/tmp/current.png",
                "reason": "interval",
                "step_id": 2,
            },
            "policy_action": "TURN_LEFT",
            "run_id": "results/run",
            "recent_keyframe_paths": ["/tmp/key0.png"],
            "memory_context_text": "m" * 300,
            "memory_images": ["/tmp/memory0.png", "/tmp/memory1.png"],
        },
    }
    assert len(message) < 1900
    assert "distance_to_goal" not in message
    assert "raw_frame" not in message


def test_cli_plan_gateway_agent_prompt_includes_bounded_context_engine_fields():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {},
                                "reason": "ok",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(run_openclaw=runner, planner_mode="agent")

    planner.plan_payload(
        {
            "state": {"instruction": "go", "step_id": 3},
            "runtime_context": {
                "run_id": "results/run",
                "task_state": {
                    "scene_id": "s1",
                    "episode_id": "e1",
                    "instruction": "go",
                    "current_step_id": 3,
                    "last_action_text": "TURN_LEFT",
                    "last_planner_reason": "r" * 1000,
                    "unbounded_internal_note": "x" * 1000,
                },
                "recent_step_summary": "s" * 1000,
                "retrieved_memory_ids": ["mem1", "mem2", "mem3", "mem4"],
            },
        }
    )

    message = runner.calls[0][0][runner.calls[0][0].index("--message") + 1]
    payload_text = message.split("Payload:\n", 1)[1]
    prompt_payload = json.loads(payload_text)
    runtime_context = prompt_payload["runtime_context"]
    assert runtime_context["task_state"]["last_action_text"] == "TURN_LEFT"
    assert "unbounded_internal_note" not in runtime_context["task_state"]
    assert len(runtime_context["recent_step_summary"]) == 500
    assert runtime_context["retrieved_memory_ids"] == ["mem1", "mem2", "mem3"]


def test_cli_plan_gateway_agent_mode_passes_openclaw_profile():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "profile",
                            }
                        )
                    }
                ]
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        openclaw_profile="clawnav-vln-min",
    )

    planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    assert runner.calls[0][0][:3] == ["openclaw", "--profile", "clawnav-vln-min"]
    assert runner.calls[0][0][3:6] == ["agent", "--agent", "main"]


def test_cli_plan_gateway_agent_mode_falls_back_to_heuristic_when_agent_fails():
    runner = FakeOpenClawRunner(returncode=1, stderr="qwen unavailable")
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        recall_interval_steps=5,
    )

    decision = planner.plan_payload(
        {
            "state": {
                "instruction": "go to kitchen",
                "step_id": 5,
            }
        }
    )

    assert decision["intent"] == "recall_memory"
    assert decision["tool_name"] == "MemoryQuerySkill"
    assert decision["reason"] == "openclaw_cli_agent_fallback:openclaw_cli_interval_recall"
    assert "qwen unavailable" in decision["arguments"]["planner_error"]


def test_cli_plan_gateway_agent_mode_preserves_context_audit_on_parse_fallback():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [{"text": "I cannot act as a navigation planner."}],
                "meta": {
                    "agentMeta": {
                        "lastCallUsage": {
                            "input": 16273,
                            "output": 293,
                            "totalTokens": 16566,
                        }
                    }
                },
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_session_id="base-session",
        agent_session_timestamp="05221040",
        agent_max_input_tokens=50000,
    )

    decision = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go",
                "step_id": 50,
            },
            "runtime_context": {"run_id": "results/run"},
        }
    )

    assert decision["reason"] == "openclaw_cli_agent_fallback:openclaw_cli_interval_recall"
    assert "openclaw agent did not return a JSON object" in decision["arguments"]["planner_error"]
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["provider_input_tokens"] == 16273
    assert audit["openclaw_session_mode"] == "fresh_per_step"
    assert audit["openclaw_session_id"] == (
        "base-session-results_run-scene_A-ep:1-step_50-05221040"
    )


def test_cli_plan_gateway_agent_mode_stops_when_input_token_limit_is_reached():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "would move",
                            }
                        )
                    }
                ],
                "usage": {"input": 10000, "output": 10, "totalTokens": 10010},
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_max_input_tokens=10000,
    )

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})
    second = planner.plan_payload({"state": {"instruction": "go", "step_id": 3}})

    assert decision["intent"] == "act"
    assert decision["tool_name"] == "NavigationPolicySkill"
    assert decision["arguments"]["action_text"] == "STOP"
    assert decision["reason"] == "openclaw_agent_input_token_limit"
    assert decision["arguments"]["input_tokens"] == 10000
    assert decision["arguments"]["max_input_tokens"] == 10000
    assert decision["runtime_metadata"]["agent_token_guard"]["tripped"] is True
    assert second["arguments"]["action_text"] == "STOP"
    assert len(runner.calls) == 1


def test_cli_plan_gateway_agent_mode_refuses_prompt_above_qwen_hard_limit(monkeypatch):
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "should not be called",
                            }
                        )
                    }
                ],
                "usage": {"input": 10, "output": 1, "totalTokens": 11},
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_max_input_tokens=0,
    )
    monkeypatch.setattr(planner, "_estimate_tokens", lambda text: 50001)

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    assert decision["intent"] == "act"
    assert decision["tool_name"] == "NavigationPolicySkill"
    assert decision["arguments"]["action_text"] == "STOP"
    assert decision["reason"] == "openclaw_agent_input_token_limit"
    assert decision["arguments"]["input_tokens"] == 50001
    assert decision["arguments"]["max_input_tokens"] == 50000
    assert runner.calls == []
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["assembled_prompt_tokens"] == 50001
    assert audit["provider_input_tokens"] is None
    assert audit["token_limit_exceeded"] is True
    assert audit["qwen_hard_limit_exceeded"] is True


def test_cli_plan_gateway_agent_mode_records_profile_budget_overrun(monkeypatch):
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "budget audit",
                            }
                        )
                    }
                ],
                "usage": {"input": 7100, "output": 12, "totalTokens": 7112},
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_max_input_tokens=10000,
    )
    monkeypatch.setattr(planner, "_estimate_tokens", lambda text: 7000)

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 2}})

    assert decision["arguments"]["action_text"] == "MOVE_FORWARD"
    assert len(runner.calls) == 1
    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["assembled_prompt_tokens"] == 7000
    assert audit["provider_input_tokens"] == 7100
    assert audit["profile_budget_exceeded"] is True
    assert audit["qwen_hard_limit_exceeded"] is False
    assert audit["token_limit_exceeded"] is False


def test_cli_plan_gateway_agent_mode_records_context_audit_metadata():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "use bounded context",
                            }
                        )
                    }
                ],
                "usage": {"input": 900, "output": 12, "totalTokens": 912},
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_session_id="base-session",
        agent_session_timestamp="05221040",
        agent_max_input_tokens=10000,
    )

    decision = planner.plan_payload(
        {
            "state": {
                "scene_id": "scene/A",
                "episode_id": "ep:1",
                "instruction": "go",
                "step_id": 2,
            },
            "runtime_context": {"run_id": "results/run"},
        }
    )

    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["context_profile"] == "plan"
    assert audit["openclaw_session_mode"] == "fresh_per_step"
    assert audit["openclaw_session_id"] == (
        "base-session-results_run-scene_A-ep:1-step_2-05221040"
    )
    assert audit["history_tokens"] == 0
    assert audit["assembled_prompt_tokens"] > 0
    assert audit["provider_input_tokens"] == 900
    assert audit["hidden_history_tokens_estimate"] == (
        audit["provider_input_tokens"] - audit["assembled_prompt_tokens"]
    )
    assert audit["agent_max_input_tokens"] == 10000
    assert audit["qwen_hard_max_input_tokens"] == 50000
    assert audit["token_limit_exceeded"] is False


def test_cli_plan_gateway_agent_mode_prefers_last_call_usage_for_context_audit():
    runner = FakeOpenClawRunner(
        stdout=json.dumps(
            {
                "payloads": [
                    {
                        "text": json.dumps(
                            {
                                "intent": "act",
                                "tool_name": "NavigationPolicySkill",
                                "arguments": {"action_text": "MOVE_FORWARD"},
                                "reason": "last call usage",
                            }
                        )
                    }
                ],
                "meta": {
                    "agentMeta": {
                        "usage": {
                            "input": 83000,
                            "output": 300,
                            "totalTokens": 83300,
                        },
                        "lastCallUsage": {
                            "input": 17000,
                            "output": 40,
                            "totalTokens": 17040,
                        },
                    }
                },
            }
        )
    )
    planner = OpenClawCliPlanPlanner(
        run_openclaw=runner,
        planner_mode="agent",
        agent_max_input_tokens=50000,
    )

    decision = planner.plan_payload({"state": {"instruction": "go", "step_id": 10}})

    audit = decision["runtime_metadata"]["context_audit"]
    assert audit["provider_input_tokens"] == 17000
    assert audit["qwen_hard_limit_exceeded"] is False
    assert audit["token_limit_exceeded"] is False
    assert decision["arguments"]["action_text"] == "MOVE_FORWARD"
