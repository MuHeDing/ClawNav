import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


class MemoryManifestError(ValueError):
    pass


@dataclass
class MemoryManifest:
    memory_source: str
    entries: List[Dict[str, Any]] = field(default_factory=list)


def load_memory_manifest(path: Path) -> MemoryManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = MemoryManifest(
        memory_source=str(data.get("memory_source", "")),
        entries=list(data.get("entries", [])),
    )
    validate_memory_manifest(manifest)
    return manifest


def validate_memory_manifest(manifest: MemoryManifest) -> None:
    if manifest.memory_source not in {"episode-local", "scene-prior", "train-scene-only"}:
        raise MemoryManifestError(f"unsupported memory_source {manifest.memory_source}")

    for entry in manifest.entries:
        if entry.get("uses_target_instruction"):
            raise MemoryManifestError("leakage: target instruction used to build memory")
        if entry.get("uses_oracle_path"):
            raise MemoryManifestError("leakage: oracle path used to build memory")
        if entry.get("uses_future_observations"):
            raise MemoryManifestError("leakage: future observations used to build memory")
        if manifest.memory_source == "scene-prior" and entry.get("split") != "exploration":
            raise MemoryManifestError(
                "leakage: scene-prior memory must come from exploration split"
            )
        if manifest.memory_source == "train-scene-only" and entry.get("split") != "train":
            raise MemoryManifestError(
                "leakage: train-scene-only memory must come from train split"
            )
