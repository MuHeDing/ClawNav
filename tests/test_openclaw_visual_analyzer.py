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
        }
    ]
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

    assert first == second
    assert len(runner.calls) == 1


def test_visual_analyzer_fails_soft_when_openclaw_fails():
    runner = FakeOpenClawRunner(returncode=1, stderr="qwen unavailable")
    analyzer = OpenClawVisualAnalyzer(run_openclaw=runner)

    observations = analyzer.analyze(["/tmp/current.png"])

    assert observations[0]["image_path"] == "/tmp/current.png"
    assert observations[0]["caption"] == ""
    assert observations[0]["error"] == "qwen unavailable"
