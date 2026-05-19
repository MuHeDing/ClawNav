import json
import subprocess

from harness.openclaw.visual_analyzer import OpenClawVisualAnalyzer


class FakeOpenClawRunner:
    def __init__(self, returncode=0, stdout=None, stderr=""):
        self.returncode = returncode
        self.stdout = stdout or json.dumps(
            {
                "results": [
                    {
                        "file": "/tmp/current.png",
                        "description": "A hallway with a doorway ahead.",
                    }
                ]
            }
        )
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


def test_visual_analyzer_describes_images_with_openclaw_capability():
    runner = FakeOpenClawRunner()
    analyzer = OpenClawVisualAnalyzer(
        run_openclaw=runner,
        model="qwen/qwen3.5-vl",
        timeout_ms=30000,
    )

    observations = analyzer.analyze(["/tmp/current.png"])

    assert observations == [
        {
            "image_path": "/tmp/current.png",
            "caption": "A hallway with a doorway ahead.",
            "visual_observation": "A hallway with a doorway ahead.",
            "landmarks": [],
            "objects": [],
            "spatial_cues": [],
            "navigation_relevance": "",
            "confidence": None,
            "analysis_metadata": {
                "vlm_latency_ms": observations[0]["analysis_metadata"]["vlm_latency_ms"],
                "visual_model": "qwen/qwen3.5-vl",
                "visual_mode": "describe",
                "cache_hit": False,
                "error": False,
            },
        }
    ]
    assert observations[0]["analysis_metadata"]["vlm_latency_ms"] >= 0.0
    command = runner.calls[0][0]
    assert command[:4] == ["openclaw", "capability", "image", "describe-many"]
    assert command.count("--file") == 1
    assert "--json" in command
    assert command[command.index("--model") + 1] == "qwen/qwen3.5-vl"
    assert command[command.index("--timeout-ms") + 1] == "30000"


def test_visual_analyzer_caches_by_image_path():
    runner = FakeOpenClawRunner()
    analyzer = OpenClawVisualAnalyzer(run_openclaw=runner)

    first = analyzer.analyze(["/tmp/current.png"])
    second = analyzer.analyze(["/tmp/current.png"])

    assert first[0]["analysis_metadata"]["cache_hit"] is False
    assert second[0]["analysis_metadata"]["cache_hit"] is True
    assert second[0]["analysis_metadata"]["vlm_latency_ms"] == 0.0
    first_without_cache = dict(first[0])
    second_without_cache = dict(second[0])
    first_without_cache.pop("analysis_metadata")
    second_without_cache.pop("analysis_metadata")
    assert first_without_cache == second_without_cache
    assert len(runner.calls) == 1


def test_visual_analyzer_fails_soft_when_openclaw_fails():
    runner = FakeOpenClawRunner(returncode=1, stderr="qwen unavailable")
    analyzer = OpenClawVisualAnalyzer(run_openclaw=runner)

    observations = analyzer.analyze(["/tmp/current.png"])

    assert observations[0]["image_path"] == "/tmp/current.png"
    assert observations[0]["caption"] == ""
    assert observations[0]["error"] == "qwen unavailable"
    assert observations[0]["analysis_metadata"]["error"] is True


def test_visual_analyzer_records_latency_and_cache_status():
    runner = FakeOpenClawRunner()
    analyzer = OpenClawVisualAnalyzer(run_openclaw=runner)

    observations = analyzer.analyze(["/tmp/current.png"])
    cached = analyzer.analyze(["/tmp/current.png"])

    assert observations[0]["analysis_metadata"]["vlm_latency_ms"] >= 0.0
    assert observations[0]["analysis_metadata"]["cache_hit"] is False
    assert cached[0]["analysis_metadata"]["cache_hit"] is True
    assert cached[0]["analysis_metadata"]["vlm_latency_ms"] == 0.0
