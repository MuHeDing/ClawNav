import json
from pathlib import Path
from types import SimpleNamespace

from evaluation_harness import HarnessModelProxy, build_harness_components


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
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_build_components_creates_openclaw_runtime_when_requested(tmp_path):
    components = build_harness_components(make_args(tmp_path), model=FakeBaseModel())

    assert components["openclaw_runtime"] is not None
    assert components["config"].harness_runtime == "openclaw_bridge"


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


def test_proxy_smoke_mode_stages_current_frame_without_interval_keyframe(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(
        make_args(tmp_path, keyframe_policy_mode="event_gated_smoke"),
        model=base_model,
    )
    proxy = HarnessModelProxy(base_model, components)
    proxy.start_episode("scene-a", "episode-1")

    payload = proxy._runtime_payload([SaveableFrame("first")], step_id=0)

    assert payload["current_image_path"].endswith(
        "openclaw_current_frames/scene-a/episode-1/step_000000.png"
    )
    assert payload["keyframe_target_path"].endswith(
        "keyframes/scene-a/episode-1/step_000000.png"
    )
    assert "keyframe_candidate" not in payload
    assert payload["recent_keyframe_paths"] == []
    assert Path(payload["current_image_path"]).exists()
    assert not Path(payload["keyframe_target_path"]).exists()


def test_proxy_updates_recent_keyframes_only_after_runtime_promoted_written(tmp_path):
    class FakeRuntime:
        def __init__(self, metadata):
            self.metadata = metadata

        def step(self, state, payload):
            del state, payload
            return SimpleNamespace(
                ok=True,
                action_text="MOVE_FORWARD",
                executor_command={},
                runtime_metadata=self.metadata,
                error=None,
            )

    promoted_path = str(tmp_path / "keyframes" / "scene-a" / "episode-1" / "step_000000.png")
    components = build_harness_components(
        make_args(tmp_path, keyframe_policy_mode="event_gated_smoke"),
        model=FakeBaseModel(),
    )
    components["openclaw_runtime"] = FakeRuntime(
        {
            "planned_intent": "act",
            "keyframe_gate": {
                "promotion_status": "promoted",
                "write_status": "written",
                "promoted_image_path": promoted_path,
            },
        }
    )
    proxy = HarnessModelProxy(FakeBaseModel(), components)
    proxy.start_episode("scene-a", "episode-1")

    proxy.call_model([SaveableFrame("first")], "go", step_id=0)

    assert proxy.recent_keyframe_paths == [promoted_path]

    components["openclaw_runtime"] = FakeRuntime(
        {
            "planned_intent": "act",
            "keyframe_gate": {
                "promotion_status": "promoted",
                "write_status": "failed",
                "promoted_image_path": str(tmp_path / "failed.png"),
            },
        }
    )
    proxy.call_model([SaveableFrame("second")], "go", step_id=1)

    assert proxy.recent_keyframe_paths == [promoted_path]


def test_proxy_saves_keyframe_artifact_for_write_memory(tmp_path):
    base_model = FakeBaseModel()
    components = build_harness_components(make_args(tmp_path), model=base_model)
    proxy = HarnessModelProxy(base_model, components)

    path = proxy._save_keyframe_if_needed("frame0", step_id=0)

    assert path.endswith("keyframes/step_000000.txt")
    assert (tmp_path / "keyframes" / "step_000000.txt").exists()
