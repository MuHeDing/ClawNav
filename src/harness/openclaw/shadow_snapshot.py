from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from harness.openclaw.instruction_stages import FORBIDDEN_ORACLE_KEYS


SHADOW_SNAPSHOT_SCHEMA_VERSION = "staged_shadow_input_v1"
_FORBIDDEN_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "reasoning",
    "raw_error",
    "api_key",
    "access_token",
    "token",
    "image_base64",
    "image_bytes",
)
_RUNTIME_CONTEXT_ALLOWLIST = frozenset(
    {
        "active_stage_id",
        "stage_state",
        "trigger_reasons",
        "requested_evidence",
        "policy_input",
        "control_context",
        "evidence_context",
        "local_control_context",
        "motion_feedback",
        "route_progress",
        "map_context",
        "current_image_path",
        "retrieved_memory_images",
        "retrieved_memory_image_paths",
        "retrieved_memory_ids",
        "recent_keyframe_paths",
    }
)


@dataclass(frozen=True)
class ShadowSnapshotSelection:
    selected: bool
    reason: str
    snapshot_path: Optional[Path] = None


def sanitize_shadow_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if value.lstrip().lower().startswith("data:image/"):
            raise ValueError("inline image bytes are forbidden in shadow snapshots")
        return value[:4096]
    if isinstance(value, Mapping):
        sanitized: Dict[str, Any] = {}
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_ORACLE_KEYS or any(
                part in normalized for part in _FORBIDDEN_KEY_PARTS
            ):
                continue
            sanitized[str(key)] = sanitize_shadow_value(nested)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_shadow_value(item) for item in value[:64]]
    return str(value)[:512]


def build_image_role_manifests(
    runtime_context: Mapping[str, Any],
) -> Tuple[list[Dict[str, str]], list[Dict[str, str]]]:
    manifest: list[Dict[str, str]] = []
    map_context = runtime_context.get("map_context")
    if isinstance(map_context, Mapping):
        map_path = str(
            map_context.get("map_image_path") or map_context.get("image_path") or ""
        )
        if map_path:
            manifest.append(
                {
                    "image_path": map_path,
                    "image_role": "map_view",
                    "memory_id": "",
                    "source": "map_view",
                }
            )
    memories = runtime_context.get("retrieved_memory_images")
    if isinstance(memories, list):
        for item in memories[:8]:
            if not isinstance(item, Mapping):
                continue
            image_path = str(item.get("image_path") or "")
            if not image_path:
                continue
            manifest.append(
                {
                    "image_path": image_path,
                    "image_role": "retrieved_memory",
                    "memory_id": str(item.get("memory_id") or ""),
                    "source": str(item.get("source") or "retrieved_memory"),
                }
            )
    current_path = str(runtime_context.get("current_image_path") or "")
    if current_path:
        manifest.append(
            {
                "image_path": current_path,
                "image_role": "current",
                "memory_id": "",
                "source": "current",
            }
        )
    deduped: list[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in manifest:
        key = (item["image_role"], item["image_path"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    masked = [
        dict(item) for item in deduped if item["image_role"] != "retrieved_memory"
    ]
    return deduped, masked


class ShadowSnapshotWriter:
    def __init__(
        self,
        output_path: Path | str,
        selection_manifest_path: Path | str,
        max_events_per_episode: int = 5,
    ) -> None:
        self.output_path = Path(output_path)
        self.selection_manifest_path = Path(selection_manifest_path)
        if int(max_events_per_episode) <= 0:
            raise ValueError("max_events_per_episode must be positive")
        self.max_events_per_episode = int(max_events_per_episode)
        self._selection = self._load_selection_manifest()
        self._selected_triggers: Dict[str, set[str]] = {}
        self._selected_counts: Dict[str, int] = {}
        self.trace_dir = self.output_path / "harness_traces"
        self.snapshot_dir = self.trace_dir / "shadow_inputs"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.trace_dir / "shadow_manifest.jsonl"

    def consider(
        self,
        *,
        scene_id: str,
        episode_id: str,
        step_id: int,
        memory_event_id: str,
        trigger_reasons: Sequence[str],
        provider_call_id: str,
        stage_state: Mapping[str, Any],
        runtime_context: Mapping[str, Any],
    ) -> ShadowSnapshotSelection:
        episode_key = f"{scene_id}:{episode_id}"
        requested = self._selection.get(episode_key)
        selected_before = self._selected_triggers.setdefault(episode_key, set())
        matching = [
            str(reason)
            for reason in trigger_reasons
            if requested is not None
            and str(reason) in requested
            and str(reason) not in selected_before
        ]
        if requested is None:
            reason = "episode_not_predeclared"
        elif not any(str(reason) in requested for reason in trigger_reasons):
            reason = "trigger_not_predeclared"
        elif not matching:
            reason = "trigger_already_selected"
        elif self._selected_counts.get(episode_key, 0) >= self.max_events_per_episode:
            reason = "episode_selection_cap"
        else:
            reason = "selected"
        selected = reason == "selected"
        snapshot_path: Optional[Path] = None
        if selected:
            selected_before.update(matching)
            self._selected_counts[episode_key] = (
                self._selected_counts.get(episode_key, 0) + 1
            )
            snapshot_path = self._write_snapshot(
                scene_id=scene_id,
                episode_id=episode_id,
                step_id=step_id,
                memory_event_id=memory_event_id,
                trigger_reasons=trigger_reasons,
                provider_call_id=provider_call_id,
                stage_state=stage_state,
                runtime_context=runtime_context,
            )
        row = {
            "schema_version": SHADOW_SNAPSHOT_SCHEMA_VERSION,
            "episode_key": episode_key,
            "step_id": int(step_id),
            "memory_event_id": str(memory_event_id),
            "trigger_reasons": [str(value) for value in trigger_reasons],
            "provider_call_id": str(provider_call_id),
            "selected": selected,
            "selection_reason": reason,
            "matched_trigger_classes": matching,
            "snapshot_path": str(snapshot_path or ""),
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return ShadowSnapshotSelection(selected, reason, snapshot_path)

    def _write_snapshot(
        self,
        **values: Any,
    ) -> Path:
        runtime_context = values["runtime_context"]
        full_manifest, masked_manifest = build_image_role_manifests(runtime_context)
        safe_context = {
            key: runtime_context[key]
            for key in _RUNTIME_CONTEXT_ALLOWLIST
            if key in runtime_context
        }
        payload = sanitize_shadow_value(
            {
                "schema_version": SHADOW_SNAPSHOT_SCHEMA_VERSION,
                "scene_id": values["scene_id"],
                "episode_id": values["episode_id"],
                "step_id": int(values["step_id"]),
                "memory_event_id": values["memory_event_id"],
                "provider_call_id": values["provider_call_id"],
                "selection_basis": {
                    "episode_key": (f"{values['scene_id']}:{values['episode_id']}"),
                    "trigger_reasons": list(values["trigger_reasons"]),
                },
                "stage_state": values["stage_state"],
                "input_context": safe_context,
                "full_image_role_manifest": full_manifest,
                "non_memory_image_role_manifest": masked_manifest,
            }
        )
        digest = hashlib.sha256(
            (f"{values['memory_event_id']}:{values['provider_call_id']}").encode(
                "utf-8"
            )
        ).hexdigest()[:20]
        path = self.snapshot_dir / f"{digest}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _load_selection_manifest(self) -> Dict[str, set[str]]:
        if not self.selection_manifest_path.is_file():
            raise ValueError("shadow selection manifest does not exist")
        selection: Dict[str, set[str]] = {}
        for line in self.selection_manifest_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("shadow selection row must be an object")
            episode_key = str(row.get("episode_key") or "")
            trigger_classes = row.get("trigger_classes")
            if not episode_key or not isinstance(trigger_classes, list):
                raise ValueError("shadow selection row fields are invalid")
            selection[episode_key] = {
                str(value) for value in trigger_classes if str(value)
            }
        return selection
