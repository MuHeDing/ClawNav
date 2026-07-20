#!/usr/bin/env python
"""Replay frozen staged-memory shadow inputs without mutating navigation state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import requests


def replay_shadow_snapshots(
    snapshot_paths: Iterable[Path | str],
    post_plan: Callable[[Dict[str, Any]], Dict[str, Any]],
    *,
    replicas: int = 2,
    output_dir: Optional[Path | str] = None,
    primary_result_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    if int(replicas) < 2:
        raise ValueError("shadow replay requires at least two replicas")
    if output_dir is not None and primary_result_path is not None:
        output = Path(output_dir).resolve()
        primary = Path(primary_result_path).resolve()
        if output == primary or primary in output.parents:
            raise ValueError("shadow output must be outside the primary result path")
    snapshots = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in sorted((Path(path) for path in snapshot_paths), key=str)
    ]
    rows = []
    for arm in ("memory_on", "memory_off"):
        for snapshot in snapshots:
            for replica in range(int(replicas)):
                payload = _plan_payload(snapshot, memory_enabled=arm == "memory_on")
                try:
                    response = post_plan(payload)
                    status = "ok"
                    error_category = ""
                except Exception as exc:
                    response = {}
                    status = "failed"
                    error_category = exc.__class__.__name__
                response_hash = hashlib.sha256(
                    json.dumps(
                        response, sort_keys=True, separators=(",", ":"), default=str
                    ).encode("utf-8")
                ).hexdigest()
                rows.append(
                    {
                        "memory_event_id": snapshot.get("memory_event_id", ""),
                        "arm": arm,
                        "replica": replica,
                        "status": status,
                        "error_category": error_category,
                        "response_sha256": response_hash,
                        "response": response,
                    }
                )
    report = _summarize(rows, len(snapshots))
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "shadow_replay_rows.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        (target / "shadow_summary.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _plan_payload(snapshot: Dict[str, Any], *, memory_enabled: bool) -> Dict[str, Any]:
    context = dict(snapshot.get("input_context") or {})
    if not memory_enabled:
        for key in (
            "retrieved_memory_images",
            "retrieved_memory_image_paths",
            "retrieved_memory_ids",
            "memory_images",
            "stage_attachment_manifest",
        ):
            context.pop(key, None)
    stage_state = snapshot.get("stage_state") or {}
    return {
        "state": {
            "scene_id": str(snapshot.get("scene_id") or ""),
            "episode_id": str(snapshot.get("episode_id") or ""),
            "instruction": str(
                stage_state.get("full_instruction")
                or stage_state.get("original_instruction")
                or ""
            ),
            "step_id": int(snapshot.get("step_id") or 0),
            "last_action": "",
        },
        "runtime_context": context,
    }


def _summarize(rows: list[Dict[str, Any]], event_count: int) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    for arm in ("memory_on", "memory_off"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        grouped: Dict[str, list[Dict[str, Any]]] = {}
        for row in arm_rows:
            grouped.setdefault(str(row["memory_event_id"]), []).append(row)
        stable = sum(
            len({row["response_sha256"] for row in values}) == 1
            and all(row["status"] == "ok" for row in values)
            for values in grouped.values()
        )
        arms[arm] = {
            "row_count": len(arm_rows),
            "event_count": len(grouped),
            "stable_event_count": stable,
            "unstable_event_count": len(grouped) - stable,
            "provider_failure_count": sum(row["status"] != "ok" for row in arm_rows),
        }
    return {
        "schema_version": "staged_shadow_replay_summary_v1",
        "row_count": len(rows),
        "event_count": event_count,
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-manifest", type=Path, required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-result-path", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=660.0)
    args = parser.parse_args()
    manifest_rows = [
        json.loads(line)
        for line in args.shadow_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    snapshot_paths = [
        Path(row["snapshot_path"])
        for row in manifest_rows
        if row.get("selected") is True and row.get("snapshot_path")
    ]
    session = requests.Session()
    session.trust_env = False

    def post(payload: Dict[str, Any]) -> Dict[str, Any]:
        response = session.post(
            f"{args.gateway_url.rstrip('/')}/plan",
            json=payload,
            timeout=args.timeout,
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("plan response must be an object")
        return value

    report = replay_shadow_snapshots(
        snapshot_paths,
        post,
        replicas=args.replicas,
        output_dir=args.output_dir,
        primary_result_path=args.primary_result_path,
    )
    return (
        0
        if all(arm["unstable_event_count"] == 0 for arm in report["arms"].values())
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
