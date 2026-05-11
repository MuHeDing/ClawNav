from typing import Any, Dict, List

from harness.skills.base import Skill
from harness.types import SkillResult, VLNState


class ProgressCriticSkill(Skill):
    name = "ProgressCriticSkill"
    description = "Evaluate non-oracle progress signals for stuck, risky stop, or replan conditions."
    input_schema = {
        "type": "object",
        "properties": {
            "working_memory": {"type": "object"},
            "policy_action": {"type": "string"},
            "semantic_alignment": {"type": "number"},
            "memory_consistency": {"type": "number"},
            "max_steps": {"type": "integer"},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "possible_stuck": {"type": "boolean"},
            "low_displacement": {"type": "boolean"},
            "repeated_observation": {"type": "boolean"},
            "risky_stop": {"type": "boolean"},
            "near_step_budget": {"type": "boolean"},
            "should_recall": {"type": "boolean"},
            "should_replan": {"type": "boolean"},
            "signals": {"type": "array"},
            "decision_inputs": {"type": "object"},
            "used_oracle_metrics": {"type": "boolean"},
        },
    }
    oracle_safe = True

    def __init__(
        self,
        low_alignment_threshold: float = 0.3,
        low_memory_consistency_threshold: float = 0.3,
    ) -> None:
        self.low_alignment_threshold = low_alignment_threshold
        self.low_memory_consistency_threshold = low_memory_consistency_threshold

    def run(self, state: VLNState, payload: Dict[str, Any]) -> SkillResult:
        working_memory = payload.get("working_memory")
        policy_action = payload.get("policy_action") or state.last_action
        semantic_alignment = float(payload.get("semantic_alignment", 1.0))
        memory_consistency = float(payload.get("memory_consistency", 1.0))
        max_steps = payload.get("max_steps")

        signals: List[str] = []
        possible_stuck = False
        low_displacement = False
        repeated_observation = False

        if working_memory is not None:
            possible_stuck = bool(working_memory.has_action_oscillation())
            low_displacement = bool(working_memory.has_low_displacement())
            repeated_observation = bool(working_memory.has_repeated_observation())

        if possible_stuck:
            signals.append("possible_stuck")
        if low_displacement:
            signals.append("low_displacement")
        if repeated_observation:
            signals.append("repeated_observation")

        risky_stop = (
            policy_action == "STOP"
            and (
                semantic_alignment < self.low_alignment_threshold
                or memory_consistency < self.low_memory_consistency_threshold
            )
        )
        if risky_stop:
            signals.append("risky_stop")

        near_step_budget = False
        if max_steps is not None:
            near_step_budget = state.step_id >= int(max_steps) - 5
            if near_step_budget:
                signals.append("near_step_budget")

        decision_inputs = {
            "used_action_history": working_memory is not None,
            "used_pose_displacement": working_memory is not None,
            "used_visual_repetition": working_memory is not None,
            "policy_action": policy_action,
            "semantic_alignment": semantic_alignment,
            "memory_consistency": memory_consistency,
            "step_id": state.step_id,
            "max_steps": max_steps,
        }

        return SkillResult.ok_result(
            "progress_critic",
            {
                "possible_stuck": possible_stuck,
                "low_displacement": low_displacement,
                "repeated_observation": repeated_observation,
                "risky_stop": risky_stop,
                "near_step_budget": near_step_budget,
                "should_recall": possible_stuck or low_displacement or risky_stop,
                "should_replan": possible_stuck or near_step_budget,
                "signals": signals,
                "decision_inputs": decision_inputs,
                "used_oracle_metrics": False,
            },
            confidence=1.0 if signals else 0.5,
        )
