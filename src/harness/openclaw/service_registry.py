from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class OpenClawService:
    name: str
    purpose: str
    host: str
    port: int
    base_url: str
    endpoint: str


class OpenClawServiceRegistry:
    def __init__(self, services: Dict[str, OpenClawService]) -> None:
        self.services = services

    @classmethod
    def from_file(
        cls,
        path: Path,
        service_host: str = "127.0.0.1",
    ) -> "OpenClawServiceRegistry":
        return cls(parse_service_table(path.read_text(encoding="utf-8"), service_host))

    def get(self, name: str) -> Optional[OpenClawService]:
        return self.services.get(name)

    def spatial_memory_url(self) -> str:
        service = self.get("SpatialMemory")
        return service.base_url if service is not None else ""


def parse_service_table(
    text: str,
    service_host: str = "127.0.0.1",
) -> Dict[str, OpenClawService]:
    services: Dict[str, OpenClawService] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line or "Service" in line:
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) < 6:
            continue

        name, purpose, host, port_text, base_url, endpoint = cells[:6]
        host = host.replace("<SERVICE_HOST>", service_host)
        base_url = base_url.replace("<SERVICE_HOST>", service_host)
        try:
            port = int(port_text)
        except ValueError:
            continue

        services[name] = OpenClawService(
            name=name,
            purpose=purpose,
            host=host,
            port=port,
            base_url=base_url,
            endpoint=endpoint,
        )
    return services
