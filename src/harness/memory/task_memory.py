import re
from typing import Iterable, List, Optional

from harness.types import SubgoalState, TaskState


class TaskMemory:
    def __init__(self) -> None:
        self.task_state = TaskState()

    @property
    def current_subgoal(self) -> Optional[SubgoalState]:
        return self.task_state.active_subgoal

    def reset(
        self,
        episode_id: str,
        global_instruction: str,
        mode: str = "single",
        subgoals: Optional[Iterable[str]] = None,
    ) -> None:
        if subgoals is not None:
            subgoal_texts = [text.strip() for text in subgoals if text and text.strip()]
        elif mode == "rule":
            subgoal_texts = self.decompose_instruction_rule_based(global_instruction)
        else:
            subgoal_texts = [global_instruction.strip()]

        if not subgoal_texts:
            subgoal_texts = [global_instruction.strip()]

        pending = [SubgoalState(text=text) for text in subgoal_texts]
        active = pending[0] if pending else None
        self.task_state = TaskState(
            episode_id=episode_id,
            global_instruction=global_instruction,
            active_subgoal=active,
            pending_subgoals=pending,
            completed_subgoals=[],
            failed_subgoals=[],
            failure_reason=None,
            recovery_attempts=[],
        )

    def decompose_instruction_rule_based(self, instruction: str) -> List[str]:
        text = instruction.strip().strip(".")
        if not text:
            return []

        pattern = r"\b(?:and then|then|and|after)\b|(?=\b(?:turn|enter|stop|near)\b)"
        parts = [part.strip(" ,.;") for part in re.split(pattern, text, flags=re.IGNORECASE)]
        return [part for part in parts if part]

    def mark_current_complete(self, reason: str = "") -> None:
        current = self.task_state.active_subgoal
        if current is None:
            return
        current.status = "completed"
        current.reason = reason
        self.task_state.completed_subgoals.append(current)
        self._advance_to_next_pending()

    def mark_current_failed(self, reason: str) -> None:
        current = self.task_state.active_subgoal
        self.task_state.failure_reason = reason
        if current is None:
            return
        current.status = "failed"
        current.reason = reason
        self.task_state.failed_subgoals.append(current)

    def record_recovery_attempt(self, action: str) -> None:
        self.task_state.recovery_attempts.append(action)

    def should_advance_subgoal(self) -> bool:
        current = self.task_state.active_subgoal
        return current is not None and current.status == "completed"

    def _advance_to_next_pending(self) -> None:
        pending = self.task_state.pending_subgoals
        while pending and pending[0].status != "pending":
            pending.pop(0)
        self.task_state.active_subgoal = pending[0] if pending else None
