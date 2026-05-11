from typing import Any, Dict, Optional

from harness.config import HarnessConfig
from harness.skill_registry import SkillRegistry
from harness.types import SkillResult, VLNState


ORACLE_PAYLOAD_KEYS = {
    "diagnostics",
    "distance_to_goal",
    "success",
    "SPL",
    "spl",
    "oracle_path",
    "oracle_shortest_path",
    "oracle_shortest_path_action",
    "oracle_action",
}


class HarnessController:
    def __init__(self, skill_registry: SkillRegistry, config: Optional[HarnessConfig] = None) -> None:
        self.skill_registry = skill_registry
        self.config = config or HarnessConfig()
        self.last_trace: Dict[str, Any] = {}

    def run_step(self, state: VLNState, payload: Optional[Dict[str, Any]] = None) -> SkillResult:
        payload = payload or {}
        self.last_trace = {
            "mode": self.config.harness_mode,
            "calls": [],
            "fallback": False,
            "fallback_reason": "",
        }

        try:
            if self.config.harness_mode == "act_only":
                return self._act(state, payload)

            memory_context: Dict[str, Any] = {}
            if self.config.harness_mode in {"memory_recall", "full"} and state.step_id == 0:
                memory_result = self._call_skill(
                    "MemoryQuerySkill",
                    state,
                    self._memory_payload(state, payload, reason="initial"),
                    decision_skill=True,
                )
                if not memory_result.ok:
                    return self._fallback_to_navigation(state, payload, memory_result.error or "memory_failed")
                memory_context = self._extract_memory_context(memory_result)

            critic_result: Optional[SkillResult] = None
            if self.config.harness_mode in {"memory_critic", "full"}:
                critic_result = self._call_skill(
                    "ProgressCriticSkill",
                    state,
                    self._critic_payload(state, payload),
                    decision_skill=True,
                )
                if not critic_result.ok:
                    return self._fallback_to_navigation(state, payload, critic_result.error or "critic_failed")

            if critic_result is not None:
                critic_payload = critic_result.payload
                risky_stop = bool(critic_payload.get("risky_stop"))
                possible_stuck = bool(critic_payload.get("possible_stuck"))
                if risky_stop or possible_stuck:
                    memory_result = self._call_skill(
                        "MemoryQuerySkill",
                        state,
                        self._memory_payload(
                            state,
                            payload,
                            reason="risky_stop" if risky_stop else "stuck",
                        ),
                        decision_skill=True,
                    )
                    if not memory_result.ok:
                        return self._fallback_to_navigation(
                            state,
                            payload,
                            memory_result.error or "memory_failed",
                        )
                    memory_context = self._extract_memory_context(memory_result)

                    replan_result = self._call_skill(
                        "ReplannerSkill",
                        state,
                        self._replan_payload(
                            payload,
                            failure_reason="risky_stop" if risky_stop else "possible_stuck",
                            memory_result=memory_result,
                        ),
                        decision_skill=True,
                    )
                    if not replan_result.ok:
                        return self._fallback_to_navigation(
                            state,
                            payload,
                            replan_result.error or "replan_failed",
                        )
                    active_subgoal = replan_result.payload.get("active_subgoal")
                    if active_subgoal:
                        payload = dict(payload)
                        payload["active_subgoal"] = active_subgoal

            nav_payload = dict(payload)
            nav_payload.update(memory_context)
            return self._act(state, nav_payload)
        except BudgetExceeded:
            return self._fallback_to_navigation(state, payload, "internal_call_budget_exceeded")

    def _act(self, state: VLNState, payload: Dict[str, Any]) -> SkillResult:
        result = self._call_skill(
            "NavigationPolicySkill",
            state,
            self._navigation_payload(payload),
            decision_skill=False,
        )
        if result.ok:
            return result
        return SkillResult.ok_result(
            "action",
            {"action_text": "STOP", "fallback": True, "error": result.error},
            confidence=0.0,
        )

    def _fallback_to_navigation(
        self,
        state: VLNState,
        payload: Dict[str, Any],
        reason: str,
    ) -> SkillResult:
        self.last_trace["fallback"] = True
        self.last_trace["fallback_reason"] = reason
        result = self.skill_registry.run(
            "NavigationPolicySkill",
            state,
            self._navigation_payload(payload),
        )
        self.last_trace["calls"].append("NavigationPolicySkill")
        if result.ok:
            return result
        return SkillResult.ok_result(
            "action",
            {"action_text": "STOP", "fallback": True, "error": result.error},
            confidence=0.0,
        )

    def _call_skill(
        self,
        name: str,
        state: VLNState,
        payload: Dict[str, Any],
        decision_skill: bool,
    ) -> SkillResult:
        if len(self.last_trace["calls"]) >= self.config.max_internal_calls_per_step:
            raise BudgetExceeded
        safe_payload = self._sanitize_payload(payload) if decision_skill else dict(payload)
        result = self.skill_registry.run(name, state, safe_payload)
        self.last_trace["calls"].append(name)
        return result

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in ORACLE_PAYLOAD_KEYS
        }

    def _navigation_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed_keys = {
            "recent_frames",
            "memory_images",
            "active_subgoal",
            "memory_context_text",
        }
        return {key: value for key, value in payload.items() if key in allowed_keys}

    def _memory_payload(self, state: VLNState, payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "text": payload.get("active_subgoal") or state.instruction,
            "step_id": state.step_id,
            "reason": reason,
        }

    def _critic_payload(self, state: VLNState, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "working_memory": payload.get("working_memory"),
            "policy_action": payload.get("policy_action") or state.last_action,
            "semantic_alignment": payload.get("semantic_alignment", 1.0),
            "memory_consistency": payload.get("memory_consistency", 1.0),
            "max_steps": payload.get("max_steps"),
        }

    def _replan_payload(
        self,
        payload: Dict[str, Any],
        failure_reason: str,
        memory_result: SkillResult,
    ) -> Dict[str, Any]:
        return {
            "task_memory": payload.get("task_memory"),
            "failure_reason": failure_reason,
            "memory_hits": memory_result.payload.get("memory_hits", []),
        }

    def _extract_memory_context(self, memory_result: SkillResult) -> Dict[str, Any]:
        policy_context = memory_result.payload.get("policy_context") or {}
        return {
            "memory_context_text": policy_context.get("memory_context_text", ""),
            "memory_images": policy_context.get("memory_images", []),
        }


class BudgetExceeded(Exception):
    pass
