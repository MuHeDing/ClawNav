from typing import Any, Dict, List

from harness.skills.base import Skill
from harness.types import SkillResult, VLNState


class NavigationPolicySkill(Skill):
    name = "NavigationPolicySkill"

    def __init__(
        self,
        model: Any,
        num_history: int = 8,
        max_memory_images: int = 0,
        max_prompt_context_chars: int = 1200,
    ) -> None:
        self.model = model
        self.num_history = num_history
        self.max_memory_images = max_memory_images
        self.max_prompt_context_chars = max_prompt_context_chars

    def run(self, state: VLNState, payload: Dict[str, Any]) -> SkillResult:
        recent_frames = list(payload.get("recent_frames") or [])
        memory_images = list(payload.get("memory_images") or [])
        images = self._select_images(recent_frames, memory_images, state.current_image)
        task = self._augment_instruction(
            state.instruction,
            active_subgoal=payload.get("active_subgoal"),
            memory_context_text=payload.get("memory_context_text"),
        )
        action = self.model.call_model(images, task, state.step_id)[0]
        return SkillResult.ok_result(
            "action",
            {
                "action_text": action,
                "images_used": len(images),
                "instruction": task,
            },
        )

    def _select_images(
        self,
        recent_frames: List[Any],
        memory_images: List[Any],
        current_image: Any,
    ) -> List[Any]:
        selected = list(recent_frames[-self.num_history :])
        if self.max_memory_images > 0:
            selected.extend(memory_images[: self.max_memory_images])
        selected.append(current_image)
        return selected

    def _augment_instruction(
        self,
        instruction: str,
        active_subgoal: Any = None,
        memory_context_text: Any = None,
    ) -> str:
        additions = []
        if active_subgoal:
            additions.append(f"Current subgoal: {active_subgoal}")
        if memory_context_text:
            memory_text = str(memory_context_text)
            if len(memory_text) > self.max_prompt_context_chars:
                memory_text = memory_text[: self.max_prompt_context_chars]
            additions.append(f"Relevant memory:\n{memory_text}")
        if not additions:
            return instruction
        return instruction + "\n\n" + "\n".join(additions)
