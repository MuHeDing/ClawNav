import json
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional


OpenClawRunner = Callable[[List[str], float], subprocess.CompletedProcess]


def run_openclaw_command(args: List[str], timeout_s: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=timeout_s,
    )


class OpenClawVisualAnalyzer:
    def __init__(
        self,
        run_openclaw: OpenClawRunner = run_openclaw_command,
        model: str = "",
        prompt: str = "",
        timeout_ms: int = 30000,
    ) -> None:
        self.run_openclaw = run_openclaw
        self.model = model
        self.prompt = prompt or (
            "Describe navigation-relevant visual evidence. Focus on stable "
            "landmarks, objects, place category, spatial layout, and cues useful "
            "for future embodied navigation memory."
        )
        self.timeout_ms = timeout_ms
        self._cache: Dict[str, Dict[str, Any]] = {}

    def analyze(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        requested = [path for path in image_paths if isinstance(path, str) and path]
        cached_before = {path for path in requested if path in self._cache}
        missing = [path for path in requested if path not in self._cache]
        if missing:
            self._describe_and_cache(missing)
        observations = []
        for path in requested:
            observation = dict(self._cache[path])
            metadata = dict(observation.get("analysis_metadata") or {})
            if path in cached_before:
                metadata["cache_hit"] = True
                metadata["vlm_latency_ms"] = 0.0
            observation["analysis_metadata"] = metadata
            observations.append(observation)
        return observations

    def _describe_and_cache(self, image_paths: List[str]) -> None:
        args = [
            "openclaw",
            "capability",
            "image",
            "describe-many",
        ]
        for path in image_paths:
            args.extend(["--file", path])
        args.extend(["--prompt", self.prompt, "--json"])
        if self.model:
            args.extend(["--model", self.model])
        args.extend(["--timeout-ms", str(int(self.timeout_ms))])

        timeout_s = max(1.0, self.timeout_ms / 1000.0)
        start = time.perf_counter()
        result = self.run_openclaw(args, timeout_s)
        latency_ms = (time.perf_counter() - start) * 1000.0
        if result.returncode != 0:
            error = (result.stderr or result.stdout or "openclaw image describe failed").strip()
            for path in image_paths:
                self._cache[path] = self._with_analysis_metadata(
                    self._fallback_observation(path, error=error),
                    latency_ms=latency_ms,
                    error=True,
                )
            return

        parsed = self._parse_stdout(result.stdout or "", image_paths)
        for path in image_paths:
            observation = parsed.get(path) or self._fallback_observation(path)
            self._cache[path] = self._with_analysis_metadata(
                observation,
                latency_ms=latency_ms,
                error=bool(observation.get("error")),
            )

    def _parse_stdout(
        self,
        stdout: str,
        image_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            if len(image_paths) == 1:
                return {
                    image_paths[0]: self._observation_from_text(image_paths[0], stdout.strip())
                }
            return {}

        items = self._result_items(data)
        parsed: Dict[str, Dict[str, Any]] = {}
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            path = self._item_path(item)
            if not path and index < len(image_paths):
                path = image_paths[index]
            if not path:
                continue
            parsed[path] = self._observation_from_item(path, item)
        if not parsed and len(image_paths) == 1 and isinstance(data, dict):
            parsed[image_paths[0]] = self._observation_from_item(image_paths[0], data)
        return parsed

    def _result_items(self, data: Any) -> List[Any]:
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        for key in ("results", "descriptions", "items", "payloads"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return []

    def _item_path(self, item: Dict[str, Any]) -> str:
        for key in ("image_path", "file", "path"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _observation_from_item(self, image_path: str, item: Dict[str, Any]) -> Dict[str, Any]:
        caption = self._first_text(item)
        return {
            "image_path": image_path,
            "caption": caption,
            "visual_observation": str(item.get("visual_observation") or caption),
            "landmarks": self._list_value(item.get("landmarks")),
            "objects": self._list_value(item.get("objects")),
            "spatial_cues": self._list_value(item.get("spatial_cues")),
            "navigation_relevance": str(item.get("navigation_relevance") or ""),
            "confidence": item.get("confidence"),
        }

    def _observation_from_text(self, image_path: str, text: str) -> Dict[str, Any]:
        observation = self._fallback_observation(image_path)
        observation["caption"] = text
        observation["visual_observation"] = text
        return observation

    def _fallback_observation(
        self,
        image_path: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        observation: Dict[str, Any] = {
            "image_path": image_path,
            "caption": "",
            "visual_observation": "",
            "landmarks": [],
            "objects": [],
            "spatial_cues": [],
            "navigation_relevance": "",
            "confidence": None,
        }
        if error:
            observation["error"] = error
        return observation

    def _with_analysis_metadata(
        self,
        observation: Dict[str, Any],
        latency_ms: float,
        error: bool = False,
    ) -> Dict[str, Any]:
        enriched = dict(observation)
        enriched["analysis_metadata"] = {
            "vlm_latency_ms": round(float(latency_ms), 3),
            "visual_model": self.model,
            "visual_mode": "describe",
            "cache_hit": False,
            "error": error,
        }
        return enriched

    def _first_text(self, item: Dict[str, Any]) -> str:
        for key in ("caption", "description", "text", "finalAssistantVisibleText"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _list_value(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item)]
