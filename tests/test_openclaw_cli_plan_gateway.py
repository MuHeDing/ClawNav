import json
import subprocess

from harness.openclaw.openclaw_cli_plan_gateway import OpenClawCliPlanPlanner


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
                    "visual_model": "qwen/qwen3.5-vl",
                    "visual_mode": "describe",
                    "cache_hit": path.endswith("key1.png"),
                    "error": False,
                },
            }
            for path in image_paths
        ]


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
                "meta": {"agentMeta": {"provider": "qwen", "model": "qwen3.5-plus"}},
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
    assert session_id_a.startswith("clawnav-")
    assert session_id_b.startswith("clawnav-")
    assert session_id_a != session_id_b


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


def test_cli_plan_gateway_agent_prompt_uses_minimal_payload():
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
            "recent_keyframe_paths": ["/tmp/key0.png"],
        },
    }
    assert len(message) < 1500
    assert "distance_to_goal" not in message
    assert "memory_context_text" not in message
    assert "raw_frame" not in message


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
