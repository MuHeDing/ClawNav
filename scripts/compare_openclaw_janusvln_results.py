#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_OPENCLAW_RESULT = (
    "/ssd/dingmuhe/Embodied-task/Navigation_Claw/ClawNav/results/my_run2/result.json"
)
DEFAULT_JANUS_RESULT = (
    "/ssd/dingmuhe/Embodied-task/JanusVLN/results/"
    "janusvln_extra_lowmem_1605632_start8_recent24_2/result.json"
)


def load_jsonl(path: Path, require_episode_id: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if "episode_id" not in row and require_episode_id:
                raise ValueError(f"{path}:{line_number}: missing episode_id")
            if "episode_id" in row:
                rows.append(row)
    return rows


def episode_key(row: Dict[str, Any]) -> str:
    return str(row["episode_id"])


def index_by_episode(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for row in rows:
        key = episode_key(row)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"duplicate episode_id in {label} result: {duplicate_text}")
    return indexed


def is_success(row: Dict[str, Any]) -> bool:
    return float(row.get("success", 0.0)) == 1.0


def build_table(openclaw_rows: List[Dict[str, Any]], janus_by_episode: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| episode_id | scene_id | OpenClaw+JanusVLN success | JanusVLN success |",
        "|---:|---|---:|---:|",
    ]
    for openclaw_row in openclaw_rows:
        key = episode_key(openclaw_row)
        janus_row = janus_by_episode.get(key)
        if janus_row is None:
            raise ValueError(f"missing episode_id in JanusVLN result: {key}")
        lines.append(
            "| {episode_id} | {scene_id} | {open_success:.1f} | {janus_success:.1f} |".format(
                episode_id=key,
                scene_id=openclaw_row.get("scene_id", ""),
                open_success=float(openclaw_row.get("success", 0.0)),
                janus_success=float(janus_row.get("success", 0.0)),
            )
        )
    return "\n".join(lines)


def compare(openclaw_result: Path, janus_result: Path) -> str:
    openclaw_rows = load_jsonl(openclaw_result)
    janus_rows = load_jsonl(janus_result, require_episode_id=False)
    janus_by_episode = index_by_episode(janus_rows, "JanusVLN")

    missing = [episode_key(row) for row in openclaw_rows if episode_key(row) not in janus_by_episode]
    if missing:
        raise ValueError(
            "missing episode_id in JanusVLN result: " + ", ".join(missing)
        )

    open_success = sum(1 for row in openclaw_rows if is_success(row))
    janus_success = sum(1 for row in openclaw_rows if is_success(janus_by_episode[episode_key(row)]))
    total = len(openclaw_rows)

    lines = [
        f"OpenClaw+JanusVLN result: {openclaw_result}",
        f"JanusVLN result: {janus_result}",
        f"Matched episodes: {total}",
        f"OpenClaw+JanusVLN success: {open_success}/{total} = {(open_success / total if total else 0.0):.4f}",
        f"JanusVLN matched success: {janus_success}/{total} = {(janus_success / total if total else 0.0):.4f}",
        "",
        build_table(openclaw_rows, janus_by_episode),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare OpenClaw+JanusVLN and JanusVLN JSONL results by episode_id."
    )
    parser.add_argument("--openclaw-result", default=DEFAULT_OPENCLAW_RESULT)
    parser.add_argument("--janus-result", default=DEFAULT_JANUS_RESULT)
    args = parser.parse_args()

    try:
        print(compare(Path(args.openclaw_result), Path(args.janus_result)))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
