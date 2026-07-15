import json
from pathlib import Path
from types import SimpleNamespace

from evaluation_harness import (
    HarnessModelProxy,
    QwenDirectPolicyProxy,
    build_harness_components,
)


class FakeBaseModel:
    model = object()

    def __init__(self):
        self.calls = []

    def call_model(self, images, task, step_id):
        self.calls.append((images, task, step_id))
        return ["TURN_LEFT"]

    def consume_last_visual_prune_profile(self):
        return None


class SaveableFrame:
    def __init__(self, text):
        self.text = text

    def save(self, path):
        path.write_text(self.text, encoding="utf-8")


class FakeAgentState:
    def __init__(self, position, rotation):
        self.position = position
        self.rotation = rotation


class FakeSim:
    def __init__(self, position, rotation):
        self._state = FakeAgentState(position, rotation)

    def get_agent_state(self):
        return self._state


class FakeEnv:
    def __init__(self, position, rotation):
        self.sim = FakeSim(position, rotation)


class FakeDirectRuntimeResult:
    ok = True
    action_text = "MOVE_FORWARD"
    executor_command = {"action_index": 1, "runtime_executor": "openclaw_habitat"}
    error = ""
    runtime_metadata = {
        "planned_intent": "act",
        "planned_tool": "QwenDirectPolicy",
        "planner_reason": "direct test",
        "policy_backend": "qwen_direct",
        "direct_policy": True,
        "janus_loaded": False,
    }


class FakeDirectRuntime:
    def __init__(self):
        self.calls = []

    def step(self, state, payload):
        self.calls.append((state, dict(payload)))
        return FakeDirectRuntimeResult()


def make_args(tmp_path, **overrides):
    data = {
        "harness_mode": "memory_recall",
        "harness_memory_backend": "fake",
        "spatial_memory_url": "http://127.0.0.1:8022",
        "memory_manifest_path": "",
        "harness_memory_source": "episode-local",
        "harness_max_internal_calls": 3,
        "harness_recall_interval_steps": 5,
        "harness_trace_rank": 0,
        "output_path": str(tmp_path),
        "num_history": 8,
        "expose_sim_pose_online": False,
        "keyframe_policy_mode": "interval",
        "keyframe_min_gap_steps": 5,
        "keyframe_episode_cap": 64,
        "keyframe_coverage_gap_steps": 20,
        "keyframe_debug_save_all_eligible": False,
        "map_assist_mode": "off",
        "map_frame_interval_steps": 5,
        "motion_feedback_enabled": False,
        "forward_stall_odometry_enabled": False,
        "map_collision_overlay_enabled": False,
        "harness_runtime": "openclaw_bridge",
        "openclaw_workspace_path": "",
        "openclaw_service_registry_path": "",
        "openclaw_service_host": "127.0.0.1",
        "openclaw_planner_backend": "rule",
        "openclaw_gateway_url": "",
        "openclaw_executor_backend": "habitat",
        "openclaw_robot_executor_url": "",
        "openclaw_subagent_backend": "fake",
        "openclaw_enable_subagent_planner": False,
        "openclaw_enable_subagent_critic": False,
        "openclaw_enable_subagent_memory_curator": False,
        "openclaw_allow_planner_action_override": False,
        "policy_backend": "janus_policy",
        "save_video": False,
        "save_video_ratio": 0.0,
        "harness_stream_video": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_build_components_creates_openclaw_runtime_when_requested(tmp_path):
    components = build_harness_components(make_args(tmp_path), model=FakeBaseModel())

    assert components["openclaw_runtime"] is not None
    assert components["config"].harness_runtime == "openclaw_bridge"


def test_build_components_qwen_direct_leaves_navigation_policy_unregistered(tmp_path):
    components = build_harness_components(
        make_args(
            tmp_path,
            policy_backend="qwen_direct",
            openclaw_planner_backend="gateway",
            openclaw_gateway_url="http://127.0.0.1:8011",
        ),
        model=None,
    )

    assert "NavigationPolicySkill" not in components["skill_registry"].names()
    assert components["openclaw_runtime"] is not None


def test_qwen_direct_proxy_requires_openclaw_runtime(tmp_path):
    components = build_harness_components(
        make_args(tmp_path, policy_backend="qwen_direct", harness_runtime="phase2"),
        model=None,
    )

    try:
        QwenDirectPolicyProxy(components)
    except ValueError as exc:
        assert "openclaw_runtime" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_qwen_direct_proxy_exposes_evaluator_compatibility_attrs(tmp_path):
    components = build_harness_components(
        make_args(
            tmp_path,
            policy_backend="qwen_direct",
            openclaw_planner_backend="gateway",
            openclaw_gateway_url="http://127.0.0.1:8011",
        ),
        model=None,
    )

    proxy = QwenDirectPolicyProxy(components)

    assert proxy.processor is None
    assert proxy.tokenizer is None
    assert proxy.model.past_key_values_vggt is None
    assert proxy.model.config is not None


def test_qwen_direct_proxy_uses_runtime_without_base_model(tmp_path):
    components = build_harness_components(
        make_args(
            tmp_path,
            policy_backend="qwen_direct",
            openclaw_planner_backend="gateway",
            openclaw_gateway_url="http://127.0.0.1:8011",
        ),
        model=None,
    )
    runtime = FakeDirectRuntime()
    components["openclaw_runtime"] = runtime
    proxy = QwenDirectPolicyProxy(components)
    proxy.start_episode("scene-a", "episode-1")

    action = proxy.call_model([SaveableFrame("frame0")], "go to kitchen", step_id=0)

    assert action == ["MOVE_FORWARD"]
    assert proxy.consume_last_visual_prune_profile() is None
    state, payload = runtime.calls[0]
    assert state.scene_id == "scene-a"
    assert state.episode_id == "episode-1"
    assert payload["run_id"] == str(tmp_path)
    trace_path = tmp_path / "harness_traces" / "harness_trace_rank0.jsonl"
    record = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert record["policy_backend"] == "qwen_direct"
    assert record["direct_policy"] is True
    assert record["janus_loaded"] is False
    assert record["planned_tool"] == "QwenDirectPolicy"


def test_qwen_direct_proxy_injects_habitat_pose_into_state_without_prompt_leak(tmp_path):
    components = build_harness_components(
        make_args(
            tmp_path,
            policy_backend="qwen_direct",
            openclaw_planner_backend="gateway",
            openclaw_gateway_url="http://127.0.0.1:8011",
        ),
        model=None,
    )
    runtime = FakeDirectRuntime()
    components["openclaw_runtime"] = runtime
    proxy = QwenDirectPolicyProxy(components)
    proxy.start_episode("scene-a", "episode-1")
    episode = SimpleNamespace(
        scene_id="/tmp/scene-a/scene.glb",
        episode_id="episode-1",
        instruction=SimpleNamespace(instruction_text="go to kitchen"),
    )

    proxy.observe_environment_state(
        env=FakeEnv(
            position=[1.0, 2.0, 3.0],
            rotation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
        episode=episode,
        observations={"rgb": "raw-frame"},
        metrics={"collisions": {"is_collision": True}, "distance_to_goal": 0.1},
        step_id=0,
    )
    action = proxy.call_model([SaveableFrame("frame0")], "go to kitchen", step_id=0)

    assert action == ["MOVE_FORWARD"]
    state, payload = runtime.calls[0]
    assert state.diagnostic_pose == {
        "position": [1.0, 2.0, 3.0],
        "rotation": [1.0, 0.0, 0.0, 0.0],
    }
    assert state.diagnostics["sim_position"] == [1.0, 2.0, 3.0]
    assert "raw_metrics" not in state.diagnostics
    assert state.pose is None
    assert payload["control_context"]["non_oracle_metrics"] == {"collision": True}
    prompt_payload_text = json.dumps(
        {
            "policy_input": payload["policy_input"],
            "control_context": payload["control_context"],
            "evidence_context": payload["evidence_context"],
        },
        sort_keys=True,
    )
    assert "sim_position" not in prompt_payload_text
    assert "sim_rotation" not in prompt_payload_text
    assert "distance_to_goal" not in prompt_payload_text


def test_build_components_registers_visual_memory_curator_when_enabled(tmp_path):
    components = build_harness_components(
        make_args(tmp_path, openclaw_enable_subagent_memory_curator=True),
        model=FakeBaseModel(),
    )

    assert "VisualMemoryCuratorSkill" in components["skill_registry"].names()


def test_build_components_leaves_runtime_off_by_default(tmp_path):
    components = build_harness_components(
        make_args(tmp_path, harness_runtime="phase2"),
        model=FakeBaseModel(),
    )

    assert components["openclaw_runtime"] is None


def test_proxy_uses_openclaw_runtime_and_logs_metadata(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(make_args(tmp_path), model=base_model)
    proxy = HarnessModelProxy(base_model, components)
    proxy.start_episode("scene-a", "episode-1")

    action = proxy.call_model(["frame0"], "go to kitchen", step_id=0)

    assert action == ["TURN_LEFT"]
    trace_path = tmp_path / "harness_traces" / "harness_trace_rank0.jsonl"
    record = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert record["scene_id"] == "scene-a"
    assert record["episode_id"] == "episode-1"
    assert record["runtime_mode"] == "openclaw_bridge"
    assert record["planned_intent"] == "recall_memory"
    assert record["runtime_executor"] == "openclaw_habitat"
    assert record["oracle_metrics_used_for_decision"] is False


def test_proxy_start_episode_resets_last_action_and_updates_state_identity(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(make_args(tmp_path), model=base_model)
    proxy = HarnessModelProxy(base_model, components)

    proxy.start_episode("scene-a", "episode-1")
    proxy.call_model(["frame0"], "go to kitchen", step_id=0)
    proxy.start_episode("scene-b", "episode-2")
    proxy.call_model(["frame1"], "go to bedroom", step_id=0)

    trace_path = tmp_path / "harness_traces" / "harness_trace_rank0.jsonl"
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["scene_id"] == "scene-a"
    assert records[0]["episode_id"] == "episode-1"
    assert records[1]["scene_id"] == "scene-b"
    assert records[1]["episode_id"] == "episode-2"


def test_proxy_save_video_does_not_enable_duplicate_harness_stream_video_by_default(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(
        make_args(tmp_path, save_video=True, save_video_ratio=1.0),
        model=base_model,
    )
    proxy = HarnessModelProxy(base_model, components)

    proxy.start_episode("scene-a", "episode-1")

    assert proxy._episode_save_video is False


def test_proxy_can_enable_harness_stream_video_explicitly(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(
        make_args(
            tmp_path,
            save_video=True,
            save_video_ratio=1.0,
            harness_stream_video=True,
        ),
        model=base_model,
    )
    proxy = HarnessModelProxy(base_model, components)

    proxy.start_episode("scene-a", "episode-1")

    assert proxy._episode_save_video is True


def test_service_registry_can_supply_spatial_memory_url(tmp_path):
    service_doc = tmp_path / "SERVICE.md"
    service_doc.write_text(
        """
| Service | Purpose | IP / Host | Port | Base URL | Main Endpoint |
|---|---|---|---|---|---|
| SpatialMemory | Memory | `<SERVICE_HOST>` | `8012` | `http://<SERVICE_HOST>:8012` | `/health` |
""",
        encoding="utf-8",
    )

    components = build_harness_components(
        make_args(
            tmp_path,
            harness_memory_backend="spatial_http",
            openclaw_service_registry_path=str(service_doc),
            openclaw_service_host="localhost",
        ),
        model=FakeBaseModel(),
    )

    assert components["config"].spatial_memory_url == "http://localhost:8012"


def test_gateway_backend_requires_gateway_url(tmp_path):
    try:
        build_harness_components(
            make_args(
                tmp_path,
                harness_runtime="openclaw_bridge",
                openclaw_planner_backend="gateway",
                openclaw_gateway_url="",
            ),
            model=FakeBaseModel(),
        )
    except ValueError as exc:
        assert "openclaw_gateway_url" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_robot_executor_requires_url_when_selected(tmp_path):
    try:
        build_harness_components(
            make_args(
                tmp_path,
                harness_runtime="openclaw_bridge",
                openclaw_executor_backend="robot_http",
                openclaw_robot_executor_url="",
            ),
            model=FakeBaseModel(),
        )
    except ValueError as exc:
        assert "openclaw_robot_executor_url" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_proxy_payload_exposes_keyframe_candidate_without_image_object(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(make_args(tmp_path), model=base_model)
    proxy = HarnessModelProxy(base_model, components)

    payload = proxy._runtime_payload(["frame0"], step_id=0)

    assert payload["run_id"] == str(tmp_path)
    assert payload["keyframe_candidate"]["step_id"] == 0
    assert "image" not in payload["keyframe_candidate"]
    assert payload["keyframe_candidate"]["reason"] == "interval"


def test_proxy_payload_exposes_current_image_path_and_recent_keyframe_paths(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(make_args(tmp_path), model=base_model)
    proxy = HarnessModelProxy(base_model, components)
    proxy.start_episode("scene-a", "episode-1")

    first_payload = proxy._runtime_payload([SaveableFrame("first")], step_id=0)
    second_payload = proxy._runtime_payload([SaveableFrame("second")], step_id=1)

    assert first_payload["current_image_path"]
    assert first_payload["current_image_path"] == first_payload["keyframe_candidate"]["image_path"]
    assert first_payload["recent_keyframe_paths"] == [first_payload["current_image_path"]]
    assert second_payload["current_image_path"]
    assert second_payload["current_image_path"] != first_payload["current_image_path"]
    assert second_payload["recent_keyframe_paths"] == [first_payload["current_image_path"]]
    assert Path(first_payload["current_image_path"]).exists()
    assert Path(second_payload["current_image_path"]).exists()
    assert first_payload["current_image_path"].endswith("step_000000.png")
    assert second_payload["current_image_path"].endswith("step_000001.png")
    assert "map_context" not in first_payload
    assert "map_assist_mode" not in first_payload


def test_proxy_payload_uses_absolute_image_paths_for_gateway(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_model = FakeBaseModel()
    components = build_harness_components(
        make_args("relative-run-output"),
        model=base_model,
    )
    proxy = HarnessModelProxy(base_model, components)
    proxy.start_episode("scene-a", "episode-1")

    payload = proxy._runtime_payload([SaveableFrame("first")], step_id=0)

    assert Path(payload["current_image_path"]).is_absolute()
    assert Path(payload["keyframe_candidate"]["image_path"]).is_absolute()
    assert Path(payload["current_image_path"]).exists()


def test_qwen_direct_proxy_payload_exposes_map_context_when_enabled(tmp_path):
    components = build_harness_components(
        make_args(
            tmp_path,
            policy_backend="qwen_direct",
            openclaw_planner_backend="gateway",
            openclaw_gateway_url="http://127.0.0.1:8011",
            map_assist_mode="floorplan_map_assisted",
            map_frame_interval_steps=5,
        ),
        model=None,
    )
    runtime = FakeDirectRuntime()
    components["openclaw_runtime"] = runtime
    proxy = QwenDirectPolicyProxy(components)
    proxy.start_episode("scene-a", "episode-1")
    proxy._map_context_provider._renderer = lambda **_: b"fake-png"
    episode = SimpleNamespace(
        scene_id="/tmp/scene-a/scene.glb",
        episode_id="episode-1",
        instruction=SimpleNamespace(instruction_text="go to kitchen"),
    )

    proxy.observe_environment_state(
        FakeEnv([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        episode,
        {"rgb": SaveableFrame("current")},
        {},
        step_id=0,
    )
    first_payload = proxy._runtime_payload([SaveableFrame("current")], step_id=0)
    proxy.observe_environment_state(
        FakeEnv([0.1, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        episode,
        {"rgb": SaveableFrame("current")},
        {},
        step_id=1,
    )
    second_payload = proxy._runtime_payload([SaveableFrame("current")], step_id=1)

    assert first_payload["map_assist_mode"] == "floorplan_map_assisted"
    assert first_payload["input_regime"] == "rgb_plus_privileged_floorplan_pose"
    assert first_payload["map_context"]["map_frame_due"] is True
    assert first_payload["map_context"]["map_available"] is True
    assert first_payload["map_context"]["internal_only"]["map_image_path"].endswith(
        "openclaw_map_frames/scene-a/episode-1/step_000000.png"
    )
    assert Path(first_payload["map_context"]["internal_only"]["map_image_path"]).exists()
    assert second_payload["map_context"]["map_frame_due"] is False
    assert second_payload["map_context"]["map_available"] is False
    assert "internal_only" not in second_payload["map_context"]


def test_qwen_direct_proxy_payload_includes_structured_control_context(tmp_path):
    components = build_harness_components(
        make_args(
            tmp_path,
            policy_backend="qwen_direct",
            openclaw_planner_backend="gateway",
            openclaw_gateway_url="http://127.0.0.1:8011",
        ),
        model=None,
    )
    proxy = QwenDirectPolicyProxy(components)
    proxy.start_episode("scene-a", "episode-1")
    working_memory = components["working_memory"]
    for _ in range(4):
        working_memory.append_action("MOVE_FORWARD")
    working_memory.append_online_metrics(
        {
            "distance_to_goal": 0.2,
            "collision": True,
        }
    )

    payload = proxy._runtime_payload([SaveableFrame("current")], step_id=4)

    assert payload["policy_input"]["policy_backend"] == "qwen_direct"
    assert payload["policy_input"]["action_scale"]["MOVE_FORWARD"]["distance_m"] == 0.25
    assert payload["policy_input"]["action_scale"]["TURN_LEFT"]["angle_degrees"] == 15
    assert payload["policy_input"]["action_scale"]["TURN_RIGHT"]["angle_degrees"] == 15
    assert "distance_to_goal" not in json.dumps(payload["policy_input"])
    assert payload["control_context"]["recent_action_counts"]["MOVE_FORWARD"] == 4
    assert payload["control_context"]["recent_forward_count"] == 4
    assert payload["control_context"]["non_oracle_metrics"] == {"collision": True}
    assert "distance_to_goal" not in json.dumps(payload["control_context"])
    assert payload["evidence_context"]["has_current_image"] is True
    assert payload["evidence_context"]["current_image_path"] == payload["current_image_path"]


def test_proxy_start_episode_resets_episode_local_working_memory(tmp_path):
    components = build_harness_components(
        make_args(
            tmp_path,
            policy_backend="qwen_direct",
            openclaw_planner_backend="gateway",
            openclaw_gateway_url="http://127.0.0.1:8011",
        ),
        model=None,
    )
    proxy = QwenDirectPolicyProxy(components)
    proxy.start_episode("scene-a", "episode-1")
    working_memory = components["working_memory"]
    for _ in range(14):
        working_memory.append_action("MOVE_FORWARD")

    proxy.start_episode("scene-a", "episode-2")
    payload = proxy._runtime_payload([SaveableFrame("current")], step_id=0)

    assert payload["recent_actions"] == []
    assert payload["control_context"]["recent_forward_count"] == 0
    assert payload["control_context"]["recent_action_count"] == 0


def test_proxy_saves_keyframe_artifact_for_write_memory(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(make_args(tmp_path), model=base_model)
    proxy = HarnessModelProxy(base_model, components)

    path = proxy._save_keyframe_if_needed("frame0", step_id=0)

    assert path.endswith("keyframes/step_000000.txt")
    assert (tmp_path / "keyframes" / "step_000000.txt").exists()
