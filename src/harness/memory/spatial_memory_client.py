from typing import Any, Dict, List, Optional

import requests

from harness.types import MemoryHit


class BaseSpatialMemoryClient:
    def __init__(self, memory_source: str = "episode-local") -> None:
        self.memory_source = memory_source

    def health(self) -> bool:
        return True

    def query_semantic(self, text: str, n_results: int = 5) -> List[MemoryHit]:
        raise NotImplementedError

    def query_object(self, name: str, n_results: int = 5) -> List[MemoryHit]:
        return self.query_unified({"query_type": "object", "text": name, "n_results": n_results})

    def query_place(self, name: str, n_results: int = 5) -> List[MemoryHit]:
        return self.query_unified({"query_type": "place", "text": name, "n_results": n_results})

    def query_position(self, pose: Dict[str, Any], n_results: int = 5) -> List[MemoryHit]:
        return self.query_unified({"query_type": "position", "pose": pose, "n_results": n_results})

    def query_unified(self, payload: Dict[str, Any]) -> List[MemoryHit]:
        text = payload.get("text") or payload.get("query") or ""
        n_results = int(payload.get("n_results", 5))
        return self.query_semantic(text, n_results=n_results)

    def ingest_semantic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "payload": payload}


class FakeSpatialMemoryClient(BaseSpatialMemoryClient):
    def query_semantic(self, text: str, n_results: int = 5) -> List[MemoryHit]:
        query = (text or "memory").strip() or "memory"
        hits = [
            MemoryHit(
                memory_id=f"fake-{idx}",
                memory_type="place",
                name=query,
                confidence=max(0.1, 0.8 - idx * 0.1),
                target_pose={"x": float(idx), "y": float(idx + 1)},
                evidence_text=f"Possible remembered landmark for {query}",
                note=f"fake memory hit {idx}",
                memory_source=self.memory_source,
            )
            for idx in range(max(1, n_results))
        ]
        return hits[:n_results]


class SpatialMemoryHttpClient(BaseSpatialMemoryClient):
    def __init__(
        self,
        base_url: str,
        memory_source: str = "episode-local",
        timeout: float = 5.0,
    ) -> None:
        super().__init__(memory_source=memory_source)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return True

    def query_semantic(self, text: str, n_results: int = 5) -> List[MemoryHit]:
        return self._post_query(
            "/query/semantic/text",
            {"text": text, "n_results": n_results},
        )

    def query_object(self, name: str, n_results: int = 5) -> List[MemoryHit]:
        return self._post_query("/query/object", {"text": name, "n_results": n_results})

    def query_place(self, name: str, n_results: int = 5) -> List[MemoryHit]:
        return self._post_query("/query/place", {"text": name, "n_results": n_results})

    def query_position(self, pose: Dict[str, Any], n_results: int = 5) -> List[MemoryHit]:
        return self._post_query("/query/position", {"pose": pose, "n_results": n_results})

    def query_unified(self, payload: Dict[str, Any]) -> List[MemoryHit]:
        return self._post_query("/query/unified", payload)

    def ingest_semantic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/memory/semantic/ingest",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post_query(self, endpoint: str, payload: Dict[str, Any]) -> List[MemoryHit]:
        response = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return [self._memory_hit_from_result(item) for item in data.get("results", [])]

    def _memory_hit_from_result(self, item: Dict[str, Any]) -> MemoryHit:
        evidence = item.get("evidence") or {}
        source = self.memory_source or item.get("source") or "episode-local"
        return MemoryHit(
            memory_id=str(item.get("id") or item.get("memory_id") or ""),
            memory_type=str(item.get("memory_type") or "semantic"),
            name=str(item.get("name") or item.get("text") or ""),
            confidence=float(item.get("confidence", 0.0)),
            target_pose=item.get("target_pose"),
            evidence_text=str(
                item.get("evidence_text")
                or evidence.get("note")
                or evidence.get("text")
                or ""
            ),
            image_path=item.get("image_path") or evidence.get("image_path"),
            note=str(item.get("note") or evidence.get("note") or ""),
            timestamp=item.get("timestamp"),
            memory_source=source,
            metadata={key: value for key, value in item.items() if key not in {"evidence"}},
        )
