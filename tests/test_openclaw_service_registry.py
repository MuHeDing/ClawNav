from harness.config import HarnessConfig
from harness.openclaw.service_registry import (
    OpenClawServiceRegistry,
    parse_service_table,
)


def test_harness_config_has_runtime_defaults():
    config = HarnessConfig()

    assert config.harness_runtime == "phase2"
    assert config.openclaw_workspace_path == ""
    assert config.openclaw_planner_backend == "rule"
    assert config.openclaw_service_registry_path == ""


def test_parse_service_table_extracts_spatial_memory():
    text = """
| Service | Purpose | IP / Host | Port | Base URL | Main Endpoint |
|---|---|---|---|---|---|
| SpatialMemory | Robot memory write / query / retrieval | `<SERVICE_HOST>` | `8012` | `http://<SERVICE_HOST>:8012` | `/health`, `/query/*`, `/memory/*` |
"""

    services = parse_service_table(text, service_host="127.0.0.1")

    assert services["SpatialMemory"].name == "SpatialMemory"
    assert services["SpatialMemory"].port == 8012
    assert services["SpatialMemory"].base_url == "http://127.0.0.1:8012"


def test_service_registry_loads_markdown_file(tmp_path):
    doc = tmp_path / "SERVICE.md"
    doc.write_text(
        """
| Service | Purpose | IP / Host | Port | Base URL | Main Endpoint |
|---|---|---|---|---|---|
| SpatialMemory | Memory | `<SERVICE_HOST>` | `8012` | `http://<SERVICE_HOST>:8012` | `/health` |
""",
        encoding="utf-8",
    )

    registry = OpenClawServiceRegistry.from_file(doc, service_host="localhost")

    assert registry.get("SpatialMemory").base_url == "http://localhost:8012"
    assert registry.spatial_memory_url() == "http://localhost:8012"
