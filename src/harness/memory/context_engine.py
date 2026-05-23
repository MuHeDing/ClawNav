import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


MAX_RECENT_STEP_SUMMARY_CHARS = 500
MAX_MEMORY_CONTEXT_CHARS = 1500
MAX_RETRIEVED_MEMORIES = 3
REVIEW_INTERVAL_STEPS = 10


@dataclass
class ContextMemoryRecord:
    memory_id: str
    text: str
    scene_id: str
    episode_id: str
    step_id: int
    image_path: str = ""
    tags: List[str] = None
    importance: float = 0.3
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags or [])
        return data


class MemoryAwareContextEngine:
    """Budgeted context source for OpenClaw planner calls.

    The engine keeps full local artifacts for audit, but only returns compact
    task state, recent summaries, and episode-scoped retrieved memories.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.memory_dir = self.root_dir / "memory"
        self.task_state_path = self.root_dir / "task_state.json"
        self.summary_path = self.root_dir / "running_summary.md"
        self.decision_log_path = self.root_dir / "decision_log.jsonl"
        self.error_fixes_path = self.root_dir / "error_fixes.jsonl"
        self.memory_index_path = self.root_dir / "memory_index.jsonl"
        self.review_log_path = self.root_dir / "review_log.md"
        self.dreams_path = self.root_dir / "DREAMS.md"
        self.review_interval_steps = REVIEW_INTERVAL_STEPS
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def prepare_plan_context(
        self,
        run_id: str,
        scene_id: str,
        episode_id: str,
        instruction: str,
        step_id: int,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_state = self._load_task_state()
        if not task_state:
            task_state = {
                "run_id": run_id,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "instruction": self._bounded_text(instruction, 500),
            }
        task_state.update(
            {
                "current_step_id": step_id,
                "scene_id": scene_id,
                "episode_id": episode_id,
            }
        )

        context: Dict[str, Any] = {"task_state": self._bounded_task_state(task_state)}
        recent_summary = self._recent_step_summary()
        if recent_summary:
            context["recent_step_summary"] = recent_summary

        query = str(
            payload.get("memory_query")
            or payload.get("active_subgoal")
            or instruction
            or ""
        )
        retrieved = self.retrieve(
            query=query,
            scene_id=scene_id,
            episode_id=episode_id,
            top_k=MAX_RETRIEVED_MEMORIES,
        )
        if retrieved:
            context["retrieved_memory_ids"] = [
                record.memory_id for record in retrieved
            ]
            context["memory_context_text"] = self._memory_context_text(retrieved)
        return context

    def record_step(
        self,
        run_id: str,
        scene_id: str,
        episode_id: str,
        instruction: str,
        step_id: int,
        payload: Dict[str, Any],
        action_text: str,
        planner_reason: str,
        ok: bool,
        error: str = "",
    ) -> Dict[str, Any]:
        now = self._now()
        summary = self._step_summary(step_id, action_text, planner_reason, ok, error)
        task_state = {
            "run_id": run_id,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "instruction": self._bounded_text(instruction, 500),
            "current_step_id": step_id,
            "last_action_text": action_text,
            "last_planner_reason": self._bounded_text(planner_reason, 240),
            "last_ok": ok,
            "last_error": self._bounded_text(error, 240),
            "last_image_path": str(payload.get("current_image_path") or ""),
            "updated_at": now,
        }
        self._write_json(self.task_state_path, task_state)
        self._append_text(self.summary_path, summary + "\n")
        self._append_jsonl(
            self.decision_log_path,
            {
                "created_at": now,
                "run_id": run_id,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "step_id": step_id,
                "action_text": action_text,
                "planner_reason": planner_reason,
                "ok": ok,
                "error": error,
            },
        )
        if error:
            self._append_jsonl(
                self.error_fixes_path,
                {
                    "created_at": now,
                    "step_id": step_id,
                    "error": error,
                    "recovery_hint": planner_reason or action_text or "inspect trace",
                },
            )
        return {
            "recorded": True,
            "root_dir": str(self.root_dir),
            "task_state_path": str(self.task_state_path),
            "running_summary_path": str(self.summary_path),
            "decision_log_path": str(self.decision_log_path),
        }

    def add_memory(
        self,
        text: str,
        scene_id: str,
        episode_id: str,
        step_id: int,
        image_path: str = "",
        tags: Optional[List[str]] = None,
        importance: float = 0.3,
    ) -> str:
        existing = self._load_memories()
        memory_id = f"mem_{len(existing) + 1:06d}"
        record = ContextMemoryRecord(
            memory_id=memory_id,
            text=self._bounded_text(text, 800),
            scene_id=scene_id,
            episode_id=episode_id,
            step_id=int(step_id),
            image_path=image_path,
            tags=list(tags or []),
            importance=float(importance),
            created_at=self._now(),
        )
        self._append_jsonl(self.memory_index_path, record.to_dict())
        self._append_text(
            self.memory_dir / f"episode_{self._safe_name(episode_id)}.md",
            f"- {record.memory_id} step={step_id} image={image_path} {record.text}\n",
        )
        return memory_id

    def retrieve(
        self,
        query: str,
        scene_id: str,
        episode_id: str,
        top_k: int = MAX_RETRIEVED_MEMORIES,
    ) -> List[ContextMemoryRecord]:
        query_terms = self._terms(query)
        scored: List[tuple[float, ContextMemoryRecord]] = []
        for record in self._load_memories():
            if record.scene_id != scene_id or record.episode_id != episode_id:
                continue
            text_terms = self._terms(" ".join([record.text, " ".join(record.tags or [])]))
            overlap = len(query_terms & text_terms)
            if query_terms and overlap == 0:
                continue
            score = overlap * 2.0 + float(record.importance) + min(record.step_id, 100) / 1000.0
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[: max(0, top_k)]]

    def review_and_compact(self, current_step_id: int) -> Dict[str, Any]:
        memories = self._load_memories()
        kept: List[ContextMemoryRecord] = []
        seen: Set[str] = set()
        removed_count = 0
        for record in memories:
            key = self._dedupe_key(record)
            old_low_value = (
                current_step_id - int(record.step_id) >= self.review_interval_steps
                and float(record.importance) < 0.1
            )
            if key in seen or old_low_value:
                removed_count += 1
                continue
            seen.add(key)
            kept.append(record)

        self._rewrite_jsonl(self.memory_index_path, [record.to_dict() for record in kept])
        line = (
            f"{self._now()} review current_step={current_step_id} "
            f"kept={len(kept)} removed={removed_count}\n"
        )
        self._append_text(self.review_log_path, line)
        if removed_count or kept:
            self._append_text(
                self.dreams_path,
                "- Review unresolved navigation evidence before promoting episode memory.\n",
            )
        return {
            "reviewed": True,
            "removed_count": removed_count,
            "kept_memory_ids": [record.memory_id for record in kept],
            "review_log_path": str(self.review_log_path),
            "dreams_path": str(self.dreams_path),
        }

    def _load_task_state(self) -> Dict[str, Any]:
        if not self.task_state_path.exists():
            return {}
        try:
            data = json.loads(self.task_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _load_memories(self) -> List[ContextMemoryRecord]:
        records: List[ContextMemoryRecord] = []
        if not self.memory_index_path.exists():
            return records
        for line in self.memory_index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            records.append(
                ContextMemoryRecord(
                    memory_id=str(data.get("memory_id") or ""),
                    text=str(data.get("text") or ""),
                    scene_id=str(data.get("scene_id") or ""),
                    episode_id=str(data.get("episode_id") or ""),
                    step_id=int(data.get("step_id") or 0),
                    image_path=str(data.get("image_path") or ""),
                    tags=list(data.get("tags") or []),
                    importance=float(data.get("importance") or 0.0),
                    created_at=str(data.get("created_at") or ""),
                )
            )
        return records

    def _recent_step_summary(self) -> str:
        if not self.summary_path.exists():
            return ""
        lines = [
            line.strip()
            for line in self.summary_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = "\n".join(lines[-5:])
        return self._bounded_text(summary, MAX_RECENT_STEP_SUMMARY_CHARS)

    def _memory_context_text(self, records: List[ContextMemoryRecord]) -> str:
        lines = [
            f"{record.memory_id}: {record.text}"
            for record in records
            if record.text
        ]
        return self._bounded_text("\n".join(lines), MAX_MEMORY_CONTEXT_CHARS)

    def _bounded_task_state(self, task_state: Dict[str, Any]) -> Dict[str, Any]:
        allowed = (
            "run_id",
            "scene_id",
            "episode_id",
            "instruction",
            "current_step_id",
            "last_action_text",
            "last_planner_reason",
            "last_ok",
            "last_error",
            "last_image_path",
        )
        bounded: Dict[str, Any] = {}
        for key in allowed:
            value = task_state.get(key)
            if isinstance(value, str):
                bounded[key] = self._bounded_text(value, 500)
            elif value is not None:
                bounded[key] = value
        return bounded

    def _step_summary(
        self,
        step_id: int,
        action_text: str,
        planner_reason: str,
        ok: bool,
        error: str,
    ) -> str:
        status = "ok" if ok else "failed"
        reason = self._bounded_text(planner_reason or error or "", 160)
        return f"Step {step_id}: {action_text or 'STOP'} status={status} reason={reason}"

    def _dedupe_key(self, record: ContextMemoryRecord) -> str:
        text = re.sub(r"\s+", " ", record.text.lower()).strip()
        return "|".join([record.scene_id, record.episode_id, record.image_path, text])

    def _terms(self, text: str) -> Set[str]:
        return {
            term
            for term in re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if len(term) > 1
        }

    def _append_jsonl(self, path: Path, row: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def _rewrite_jsonl(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _append_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def _bounded_text(self, text: str, limit: int) -> str:
        value = str(text or "")
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 3)] + "..."

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).strip("_") or "default"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
