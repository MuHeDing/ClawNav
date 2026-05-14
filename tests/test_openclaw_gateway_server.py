import json
import threading
import urllib.request

from harness.openclaw.gateway import OpenClawGatewayClient
from harness.openclaw.gateway_server import LocalOpenClawGatewayPlanner, make_gateway_server
from harness.types import VLNState


def make_state(step_id=0):
    return VLNState(
        scene_id="s1",
        episode_id="e1",
        instruction="go to kitchen",
        step_id=step_id,
        current_image=None,
    )


def start_test_server():
    server = make_gateway_server(
        host="127.0.0.1",
        port=0,
        planner=LocalOpenClawGatewayPlanner(recall_interval_steps=5),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_gateway_server_health_endpoint():
    server = start_test_server()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["ok"] is True
        assert payload["service"] == "clawnav_openclaw_gateway"
    finally:
        server.shutdown()
        server.server_close()


def test_gateway_server_serves_plan_endpoint_for_client():
    server = start_test_server()
    try:
        host, port = server.server_address
        client = OpenClawGatewayClient(base_url=f"http://{host}:{port}")

        decision = client.plan(make_state(step_id=0), runtime_context={"success": True})

        assert decision.intent == "recall_memory"
        assert decision.tool_name == "MemoryQuerySkill"
        assert decision.arguments["text"] == "go to kitchen"
        assert decision.planner_backend == "gateway"
    finally:
        server.shutdown()
        server.server_close()
