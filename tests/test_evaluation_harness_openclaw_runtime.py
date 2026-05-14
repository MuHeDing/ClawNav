import json
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
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_build_components_creates_openclaw_runtime_when_requested(tmp_path):
    components = build_harness_components(make_args(tmp_path), model=FakeBaseModel())

    assert components["openclaw_runtime"] is not None
    assert components["config"].harness_runtime == "openclaw_bridge"


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

    action = proxy.call_model(["frame0"], "go to kitchen", step_id=0)

    assert action == ["TURN_LEFT"]
    trace_path = tmp_path / "harness_traces" / "harness_trace_rank0.jsonl"
    record = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert record["runtime_mode"] == "openclaw_bridge"
    assert record["planned_intent"] == "recall_memory"
    assert record["runtime_executor"] == "openclaw_habitat"
    assert record["oracle_metrics_used_for_decision"] is False


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

    assert payload["keyframe_candidate"]["step_id"] == 0
    assert "image" not in payload["keyframe_candidate"]
    assert payload["keyframe_candidate"]["reason"] == "interval"
