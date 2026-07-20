from dataclasses import dataclass, field
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Protocol

from harness.memory.context_engine import MemoryAwareContextEngine
from harness.memory.episode_visual_store import EpisodeVisualMemoryStore
from harness.openclaw.control_gates import (
    QwenDirectControlGates,
    ROUTE_WAYPOINT_PASS_FORWARD_ACTIONS,
    TURN_ROUND_COMPLETION_THRESHOLD_DEG,
)
from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.keyframe_gate import (
    EventGatedKeyframeGate,
    candidate_action_status_from_tool_result,
)
from harness.openclaw.motion_feedback import build_motion_feedback
from harness.openclaw.tool_adapter import OpenClawToolAdapter
from harness.skill_registry import SkillRegistry
from harness.types import VLNState


PRE_ACTION_INTENTS = {
    "recall_memory",
    "write_memory",
    "verify_progress",
    "replan",
}

NAVIGATION_CONTEXT_KEYS = {
    "active_subgoal",
    "memory_context_text",
    "recent_frames",
}

ACTION_TEXTS = {
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
}

ACTION_ARGUMENT_KEYS = (
    "action_text",
    "planned_action",
    "preferred_action",
    "forced_action_text",
)

OPENCLAW_CLI_AGENT_FALLBACK_REASON_PREFIX = "openclaw_cli_agent_fallback:"
OPENCLAW_CLI_MODEL_FALLBACK_REASON_PREFIX = "openclaw_cli_model_fallback:"
QWEN_DIRECT_POLICY_BACKEND = "qwen_direct"
ODOMETRY_PROGRESS_THRESHOLD_M = 0.05
ODOMETRY_TURN_PROGRESS_THRESHOLD_DEG = 5.0
TURN_LOOP_WINDOW_STEPS = 8
TURN_LOOP_MIN_TURNS = 6
TURN_LOOP_SAME_DIRECTION_TURNS = 8
TURN_LOOP_MAX_FORWARD_ACTIONS = 1
TURN_LOOP_MAX_TRANSLATION_SPAN_M = 0.35


@dataclass
class OpenClawRuntimeStepResult:
    ok: bool
    action_text: str
    executor_command: Dict[str, Any] = field(default_factory=dict)
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class OpenClawPlannerProtocol(Protocol):
    def plan(
        self,
        state: VLNState,
        runtime_context: Dict[str, Any],
    ) -> Any:
        ...


class OpenClawVLNRuntime:
    def __init__(
        self,
        tool_registry: SkillRegistry,
        planner: OpenClawPlannerProtocol,
        executor: HabitatOpenClawExecutor,
        fallback_planner: OpenClawPlannerProtocol = None,
        allow_planner_action_override: bool = True,
        policy_backend: str = "janus_policy",
        keyframe_policy_mode: str = "interval",
        keyframe_min_gap_steps: int = 5,
        keyframe_episode_cap: int = 64,
        keyframe_coverage_gap_steps: int = 20,
        keyframe_debug_save_all_eligible: bool = False,
        dynamic_visual_context_enabled: bool = False,
        staged_visual_memory_enabled: bool = False,
        episode_visual_store: Optional[EpisodeVisualMemoryStore] = None,
    ) -> None:
        self.tool_adapter = OpenClawToolAdapter(tool_registry)
        self.planner = planner
        self.executor = executor
        self.fallback_planner = fallback_planner
        self.allow_planner_action_override = allow_planner_action_override
        self.policy_backend = policy_backend
        self.qwen_direct_gates = QwenDirectControlGates()
        self.keyframe_policy_mode = str(keyframe_policy_mode or "interval")
        self.keyframe_gate = EventGatedKeyframeGate(
            min_gap_steps=int(keyframe_min_gap_steps),
            episode_cap=int(keyframe_episode_cap),
            coverage_gap_steps=int(keyframe_coverage_gap_steps),
            debug_save_all_eligible=bool(keyframe_debug_save_all_eligible),
        )
        self.recent_visual_memories: List[Dict[str, Any]] = []
        self.max_recent_visual_memories = 10
        self.context_engines: Dict[str, MemoryAwareContextEngine] = {}
        self.qwen_direct_episode_state: Dict[str, Dict[str, Any]] = {}
        self.odometry_episode_state: Dict[str, Dict[str, Any]] = {}
        self.dynamic_visual_context_enabled = bool(dynamic_visual_context_enabled)
        self.staged_visual_memory_enabled = bool(staged_visual_memory_enabled)
        self.episode_visual_store = episode_visual_store

    def reset_episode(self, scene_id: str = "", episode_id: str = "") -> None:
        self.qwen_direct_episode_state.clear()
        self.odometry_episode_state.clear()
        if self.episode_visual_store is not None:
            self.episode_visual_store.reset_episode()
            if scene_id or episode_id:
                self.episode_visual_store.start_episode(scene_id, episode_id)

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.tool_adapter.list_tools()

    def step(
        self,
        state: VLNState,
        payload: Dict[str, Any],
    ) -> OpenClawRuntimeStepResult:
        planner_error = ""
        planner_fallback = False
        runtime_payload = dict(payload)
        tool_calls: List[Dict[str, Any]] = []
        context_engine = self._context_engine_for_payload(runtime_payload)

        pre_planner_recall = self._auto_recall_memory(
            state,
            runtime_payload,
            reason="pre_planner_visual_memory_recall",
        )
        if pre_planner_recall:
            tool_calls.append(pre_planner_recall)
            self._merge_tool_navigation_context(runtime_payload, pre_planner_recall)

        if context_engine is not None:
            self._merge_context_engine_plan_context(
                context_engine,
                state,
                runtime_payload,
            )

        self._attach_pre_planner_local_control_context(state, runtime_payload)
        if self._dynamic_visual_context_active(runtime_payload):
            self._record_visual_evidence_frame(state, runtime_payload)
            self._attach_visual_evidence_registry(state, runtime_payload)
        image_paths_used = self._image_paths_used(runtime_payload)
        try:
            decision = self.planner.plan(state, runtime_context=runtime_payload)
        except Exception as exc:
            planner_error = str(exc)
            if self.fallback_planner is None:
                metadata = {
                    "runtime_mode": "openclaw_bridge",
                    "planner_fallback": False,
                    "planner_error": planner_error,
                }
                self._record_context_engine_step(
                    metadata,
                    context_engine,
                    state,
                    runtime_payload,
                    "STOP",
                    planner_error,
                    False,
                    planner_error,
                )
                return OpenClawRuntimeStepResult(
                    ok=False,
                    action_text="STOP",
                    runtime_metadata=metadata,
                    error=planner_error,
                )
            decision = self.fallback_planner.plan(
                state, runtime_context=runtime_payload
            )
            planner_fallback = True
        causal_recall: Dict[str, Any] = {}
        self._merge_planner_visual_observations(runtime_payload, decision)

        cli_fallback_kind = self._cli_fallback_kind(decision)
        if cli_fallback_kind:
            metadata = self._metadata(decision, tool_calls, image_paths_used)
            metadata[f"planner_{cli_fallback_kind}_fallback"] = True
            metadata["planner_fallback"] = True
            planner_error = str(
                decision.arguments.get("planner_error") or decision.reason
            )
            if planner_error:
                metadata["planner_error"] = planner_error
            self._record_context_engine_step(
                metadata,
                context_engine,
                state,
                runtime_payload,
                "STOP",
                metadata.get("planner_reason", ""),
                False,
                planner_error or f"openclaw_cli_{cli_fallback_kind}_fallback",
                tool_calls=tool_calls,
            )
            return OpenClawRuntimeStepResult(
                ok=False,
                action_text="STOP",
                executor_command=self.executor.command_for_action("STOP"),
                runtime_metadata=metadata,
                error=planner_error or f"openclaw_cli_{cli_fallback_kind}_fallback",
            )

        nav_payload = self._navigation_payload(runtime_payload)

        if decision.intent in PRE_ACTION_INTENTS:
            arguments = dict(decision.arguments)
            self._merge_navigation_context(
                nav_payload,
                arguments,
                include_memory_images=False,
            )
            if decision.intent == "recall_memory":
                arguments = self._memory_query_arguments(
                    state, runtime_payload, arguments
                )
            if decision.intent == "write_memory":
                arguments = self._memory_write_arguments(runtime_payload, arguments)
            if decision.intent == "write_memory":
                arguments.setdefault(
                    "recent_visual_memories",
                    list(self.recent_visual_memories),
                )
                curator_result = self._curate_memory_write(state, arguments)
                if curator_result:
                    tool_calls.append(curator_result)
                    arguments = self._apply_curator_result(arguments, curator_result)
            tool_result = self.tool_adapter.call_tool(
                decision.tool_name,
                arguments,
                state=state,
            )
            tool_calls.append(tool_result)
            if decision.intent == "write_memory":
                self._remember_written_visual_memory(tool_result)
            self._merge_tool_navigation_context(
                nav_payload,
                tool_result,
                include_memory_images=False,
            )
            if decision.intent == "recall_memory":
                after_decision = self._after_recall_decision(
                    state, runtime_payload, nav_payload
                )
                if after_decision is not None:
                    causal_recall = {
                        "before_decision": decision,
                        "after_decision": after_decision,
                        "action_before": self._normalize_action_text(
                            payload.get("policy_action")
                        )
                        or "",
                    }
                    if after_decision.intent in {"act", "replan"}:
                        decision = after_decision
                        self._merge_navigation_context(
                            nav_payload,
                            decision.arguments,
                            include_memory_images=False,
                        )

        if decision.intent != "write_memory":
            auto_write_calls = self._auto_write_visual_memory(
                state, runtime_payload, decision
            )
            if auto_write_calls:
                tool_calls.extend(auto_write_calls)
                written = any(
                    call.get("tool_name") == "MemoryWriteSkill"
                    and bool((call.get("payload") or {}).get("written"))
                    for call in auto_write_calls
                )
                if written:
                    after_write_recall = self._auto_recall_memory(
                        state,
                        runtime_payload,
                        reason="after_visual_memory_write_recall",
                    )
                    if after_write_recall:
                        tool_calls.append(after_write_recall)
                        self._merge_tool_navigation_context(
                            nav_payload,
                            after_write_recall,
                            include_memory_images=False,
                        )

        self._merge_navigation_context(
            nav_payload,
            decision.arguments,
            include_memory_images=False,
        )
        if (
            decision.intent == "replan"
            and not nav_payload.get("active_subgoal")
            and decision.reason
        ):
            nav_payload["active_subgoal"] = decision.reason

        planned_action_text = self._planned_action_text(decision.arguments)
        if self.policy_backend == QWEN_DIRECT_POLICY_BACKEND:
            self._promote_visual_semantic_evidence(
                state,
                runtime_payload,
                decision,
            )
            self._attach_visual_evidence_registry(state, runtime_payload)
            self._record_qwen_direct_observation(state, decision.arguments)
            self._attach_route_progress_context(state, runtime_payload)
            gate_context = self._qwen_direct_gate_context(
                runtime_payload,
                decision,
                state=state,
            )
            gate_result = self.qwen_direct_gates.apply(decision.arguments, gate_context)
            requery_metadata: Dict[str, Any] = {}
            if self._qwen_direct_should_requery_gate(runtime_payload, gate_result):
                requery_result = self._qwen_direct_requery_after_gate(
                    state,
                    runtime_payload,
                    decision,
                    gate_result,
                )
                if requery_result is not None:
                    requery_decision, requery_payload = requery_result
                    requery_metadata = self._qwen_direct_requery_metadata(
                        decision,
                        gate_result,
                    )
                    decision = requery_decision
                    self._merge_planner_visual_observations(runtime_payload, decision)
                    self._promote_visual_semantic_evidence(
                        state,
                        requery_payload,
                        decision,
                    )
                    self._attach_visual_evidence_registry(state, requery_payload)
                    self._record_qwen_direct_observation(state, decision.arguments)
                    self._attach_route_progress_context(state, requery_payload)
                    gate_context = self._qwen_direct_gate_context(
                        requery_payload,
                        decision,
                        state=state,
                    )
                    gate_result = self.qwen_direct_gates.apply(
                        decision.arguments,
                        gate_context,
                    )
            metadata = self._metadata(
                decision,
                tool_calls,
                image_paths_used,
                state=state,
                runtime_context=runtime_payload,
                action_text=gate_result.final_action,
                causal_recall=causal_recall,
            )
            metadata.update(
                self._qwen_direct_schema_metadata(decision.arguments, decision.reason)
            )
            metadata.update(gate_result.metadata)
            metadata.update(requery_metadata)
            metadata["policy_backend"] = QWEN_DIRECT_POLICY_BACKEND
            metadata["direct_policy"] = True
            metadata.setdefault("janus_loaded", False)
            metadata.setdefault("navigation_policy_skill_called", False)
            keyframe_gate = self._run_keyframe_gate_for_action(
                state=state,
                runtime_payload=runtime_payload,
                raw_candidate_action=planned_action_text or gate_result.final_action,
                candidate_action_status="ok",
            )
            if keyframe_gate:
                metadata["keyframe_gate"] = keyframe_gate
                self._record_promoted_keyframe_evidence(
                    state,
                    runtime_payload,
                    keyframe_gate,
                )
            if planner_error:
                metadata["planner_error"] = planner_error
            metadata["planner_fallback"] = planner_fallback
            self._record_context_engine_step(
                metadata,
                context_engine,
                state,
                runtime_payload,
                gate_result.final_action,
                metadata.get("planner_reason", ""),
                True,
                tool_calls=tool_calls,
            )
            return OpenClawRuntimeStepResult(
                ok=True,
                action_text=gate_result.final_action,
                executor_command=self.executor.command_for_action(
                    gate_result.final_action
                ),
                runtime_metadata=metadata,
            )
        if planned_action_text and self.allow_planner_action_override:
            metadata = self._metadata(
                decision,
                tool_calls,
                image_paths_used,
                state=state,
                runtime_context=runtime_payload,
                action_text=planned_action_text,
                causal_recall=causal_recall,
            )
            metadata["planner_action_override"] = planned_action_text
            metadata["policy_skipped"] = True
            keyframe_gate = self._run_keyframe_gate_for_action(
                state=state,
                runtime_payload=runtime_payload,
                raw_candidate_action=planned_action_text,
                candidate_action_status="ok",
            )
            if keyframe_gate:
                metadata["keyframe_gate"] = keyframe_gate
            if planner_error:
                metadata["planner_error"] = planner_error
            metadata["planner_fallback"] = planner_fallback
            self._record_context_engine_step(
                metadata,
                context_engine,
                state,
                runtime_payload,
                planned_action_text,
                metadata.get("planner_reason", ""),
                True,
                tool_calls=tool_calls,
            )
            return OpenClawRuntimeStepResult(
                ok=True,
                action_text=planned_action_text,
                executor_command=self.executor.command_for_action(planned_action_text),
                runtime_metadata=metadata,
            )
        nav_result = self.tool_adapter.call_tool(
            "NavigationPolicySkill",
            nav_payload,
            state=state,
        )
        tool_calls.append(nav_result)

        action_text = str(nav_result.get("payload", {}).get("action_text") or "STOP")
        keyframe_gate = self._run_keyframe_gate_for_action(
            state=state,
            runtime_payload=runtime_payload,
            raw_candidate_action=action_text,
            candidate_action_status=candidate_action_status_from_tool_result(
                nav_result
            ),
        )
        metadata = self._metadata(
            decision,
            tool_calls,
            image_paths_used,
            state=state,
            runtime_context=runtime_payload,
            action_text=action_text,
            causal_recall=causal_recall,
        )
        if planner_error:
            metadata["planner_error"] = planner_error
        metadata["planner_fallback"] = planner_fallback
        if planned_action_text and not self.allow_planner_action_override:
            metadata["planner_action_guidance"] = planned_action_text
        if keyframe_gate:
            metadata["keyframe_gate"] = keyframe_gate
        if not nav_result.get("ok"):
            self._record_context_engine_step(
                metadata,
                context_engine,
                state,
                runtime_payload,
                "STOP",
                metadata.get("planner_reason", ""),
                False,
                nav_result.get("error") or "navigation_failed",
                tool_calls=tool_calls,
            )
            return OpenClawRuntimeStepResult(
                ok=False,
                action_text="STOP",
                runtime_metadata=metadata,
                error=nav_result.get("error") or "navigation_failed",
            )

        self._record_context_engine_step(
            metadata,
            context_engine,
            state,
            runtime_payload,
            action_text,
            metadata.get("planner_reason", ""),
            True,
            tool_calls=tool_calls,
        )
        return OpenClawRuntimeStepResult(
            ok=True,
            action_text=action_text,
            executor_command=self.executor.command_for_action(action_text),
            runtime_metadata=metadata,
        )

    def _run_keyframe_gate_for_action(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
        raw_candidate_action: str,
        candidate_action_status: str,
    ) -> Dict[str, Any]:
        if self.keyframe_policy_mode != "event_gated_smoke":
            return {}
        gate = self.keyframe_gate.evaluate(
            scene_id=state.scene_id,
            episode_id=state.episode_id,
            step_id=state.step_id,
            raw_candidate_action=raw_candidate_action,
            candidate_action_status=candidate_action_status,
            current_image_path=str(runtime_payload.get("current_image_path") or ""),
            keyframe_target_path=str(runtime_payload.get("keyframe_target_path") or ""),
        )
        if gate.get("save_decision") == "save":
            self._promote_keyframe(gate)
            self._attach_promoted_keyframe_candidate(runtime_payload, gate)
        runtime_payload["keyframe_gate"] = gate
        return gate

    def _promote_keyframe(self, gate: Dict[str, Any]) -> None:
        source = Path(str(gate.get("current_image_path") or ""))
        target = Path(str(gate.get("keyframe_target_path") or ""))
        if not source.exists():
            gate["promotion_status"] = "failed"
            gate["failure_reason"] = "missing_current_image"
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        except OSError as exc:
            gate["promotion_status"] = "failed"
            gate["failure_reason"] = f"promotion_failed:{exc.__class__.__name__}"
            return
        gate["promotion_status"] = "promoted"
        gate["promoted_image_path"] = str(target)

    def _attach_promoted_keyframe_candidate(
        self,
        runtime_payload: Dict[str, Any],
        gate: Dict[str, Any],
    ) -> None:
        if gate.get("promotion_status") != "promoted":
            return
        image_path = str(
            gate.get("promoted_image_path") or gate.get("keyframe_target_path") or ""
        )
        if not image_path:
            return
        runtime_payload["keyframe_candidate"] = {
            "step_id": gate.get("step_id"),
            "reason": gate.get("save_reason"),
            "event_type": gate.get("candidate_event_type"),
            "image_path": image_path,
            "keyframe_gate": dict(gate),
        }

    def _record_promoted_keyframe_evidence(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
        gate: Dict[str, Any],
    ) -> None:
        if not self._dynamic_visual_context_active(runtime_payload):
            return
        if gate.get("promotion_status") != "promoted":
            return
        promoted_path = str(
            gate.get("promoted_image_path") or gate.get("keyframe_target_path") or ""
        )
        current_path = str(runtime_payload.get("current_image_path") or "")
        if not promoted_path or not current_path:
            return
        episode_state = self._qwen_direct_state_for_episode(state)
        records = episode_state.get("visual_evidence_registry")
        if not isinstance(records, list):
            return
        source = next(
            (
                record
                for record in reversed(records)
                if isinstance(record, dict) and record.get("image_path") == current_path
            ),
            None,
        )
        if not isinstance(source, dict):
            return
        promoted = dict(source)
        promoted["image_path"] = promoted_path
        promoted["keyframe"] = True
        promoted["roles"] = sorted(
            {
                str(role)
                for role in source.get("roles") or []
                if isinstance(role, str) and role
            }
            | {"keyframe"}
        )
        records[:] = [
            record
            for record in records
            if not (
                isinstance(record, dict) and record.get("image_path") == promoted_path
            )
        ]
        records.append(promoted)
        self._mirror_visual_record_to_episode_store(
            state,
            runtime_payload,
            promoted,
            provenance="runtime:keyframe_promotion",
        )
        self._evict_visual_evidence_records(episode_state)

    def _qwen_direct_should_requery_gate(
        self,
        runtime_payload: Dict[str, Any],
        gate_result,
    ) -> bool:
        if runtime_payload.get("_qwen_direct_gate_requery"):
            return False
        return bool(self._qwen_direct_requery_reason(gate_result))

    def _qwen_direct_requery_after_gate(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
        decision,
        gate_result,
    ) -> Optional[tuple[Any, Dict[str, Any]]]:
        requery_reason = self._qwen_direct_requery_reason(gate_result)
        if requery_reason == "blocked_stop_gate":
            feedback_key = "blocked_stop_feedback"
            feedback = self._qwen_direct_blocked_stop_feedback(decision, gate_result)
            control_flags = {"force_non_stop_action": True}
        elif requery_reason == "forward_stall_gate":
            feedback_key = "forward_stall_feedback"
            feedback = self._qwen_direct_forward_stall_feedback(
                decision,
                gate_result,
                runtime_payload,
            )
            control_flags = {"force_non_forward_action": True}
        elif requery_reason == "structural_stop_verification":
            feedback_key = "stop_verification_feedback"
            feedback = self._qwen_direct_stop_verification_feedback(
                decision,
                gate_result,
            )
            control_flags = {
                "force_non_forward_action": True,
                "force_stop_verification": True,
            }
        else:
            return None
        requery_payload = dict(runtime_payload)
        requery_payload["_qwen_direct_gate_requery"] = True
        requery_payload[feedback_key] = feedback
        control_context = dict(requery_payload.get("control_context") or {})
        control_context.update(control_flags)
        control_context["allowed_actions"] = list(feedback["allowed_actions"])
        control_context[feedback_key] = feedback
        requery_payload["control_context"] = control_context
        if requery_reason == "structural_stop_verification":
            requery_payload["structural_stop_verification_active"] = True
        try:
            decision = self.planner.plan(state, runtime_context=requery_payload)
        except Exception:
            return None
        return decision, requery_payload

    def _qwen_direct_requery_reason(self, gate_result) -> str:
        metadata = getattr(gate_result, "metadata", {}) or {}
        if (
            metadata.get("stop_gate_decision") == "blocked"
            and metadata.get("blocked_action") == "STOP"
        ):
            if (
                metadata.get("stop_gate_block_reason")
                == "structural_stop_confirmation_required"
            ):
                return "structural_stop_verification"
            return "blocked_stop_gate"
        if (
            metadata.get("forward_stall_gate_decision") == "blocked"
            and metadata.get("blocked_action") == "MOVE_FORWARD"
        ):
            return "forward_stall_gate"
        return ""

    def _qwen_direct_blocked_stop_feedback(
        self,
        decision,
        gate_result,
    ) -> Dict[str, Any]:
        arguments = getattr(decision, "arguments", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        feedback = {
            "gate_decision": "blocked",
            "blocked_action": "STOP",
            "allowed_actions": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
            "reason": (
                "STOP was rejected because current evidence did not prove "
                "arrival, route progress, or wait completion. Choose a non-STOP "
                "corrective action. Use MOVE_FORWARD only when the current "
                "image clearly shows the route continues forward; otherwise "
                "turn to recheck alignment or target evidence."
            ),
            "fallback_action": gate_result.final_action,
            "stop_evidence": self._bounded_metadata_text(
                arguments.get("stop_evidence")
            ),
            "current_target": self._bounded_metadata_text(
                arguments.get("current_target")
            ),
            "target_relation": self._bounded_metadata_text(
                arguments.get("target_relation")
            ),
            "semantic_stop_state": self._bounded_metadata_text(
                arguments.get("semantic_stop_state")
            ),
            "visual_summary": self._bounded_metadata_text(
                arguments.get("visual_summary")
            ),
            "progress_state": self._bounded_metadata_text(
                arguments.get("progress_state")
            ),
            "qwen_reason": self._bounded_metadata_text(
                arguments.get("reason") or getattr(decision, "reason", "")
            ),
        }
        gate_metadata = getattr(gate_result, "metadata", {}) or {}
        for key in (
            "stop_gate_block_reason",
            "stop_gate_missing_waypoints",
            "stop_gate_missing_route_waypoints",
            "stop_gate_unpassed_route_waypoints",
            "stop_gate_route_missing_reasons",
            "stop_gate_min_step_id",
            "stop_gate_current_step_id",
            "stop_gate_min_forward_actions",
            "stop_gate_forward_action_count",
            "stop_gate_required_route_waypoints",
            "stop_gate_positively_seen_waypoints",
            "stop_gate_passed_waypoints",
            "stop_gate_negated_waypoint_mentions",
            "stop_gate_turn_round_required",
            "stop_gate_turn_round_completed",
            "stop_gate_heading_change_from_start_deg",
        ):
            value = gate_metadata.get(key)
            if value not in (None, "", []):
                feedback[key] = value
        return feedback

    def _qwen_direct_stop_verification_feedback(
        self,
        decision,
        gate_result,
    ) -> Dict[str, Any]:
        arguments = getattr(decision, "arguments", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        return {
            "gate_decision": "verify",
            "blocked_action": "STOP",
            "allowed_actions": ["STOP", "TURN_LEFT", "TURN_RIGHT"],
            "reason": (
                "Verify structural arrival from the same observation without "
                "translating. Choose STOP only if the agent is already at or "
                "inside the target boundary; otherwise turn to inspect. "
                "MOVE_FORWARD is forbidden during this verification."
            ),
            "fallback_action": gate_result.final_action,
            "stop_evidence": self._bounded_metadata_text(
                arguments.get("stop_evidence")
            ),
            "current_target": self._bounded_metadata_text(
                arguments.get("current_target")
            ),
            "target_relation": self._bounded_metadata_text(
                arguments.get("target_relation")
            ),
            "semantic_stop_state": self._bounded_metadata_text(
                arguments.get("semantic_stop_state")
            ),
            "visual_summary": self._bounded_metadata_text(
                arguments.get("visual_summary")
            ),
            "qwen_reason": self._bounded_metadata_text(
                arguments.get("reason") or getattr(decision, "reason", "")
            ),
        }

    def _qwen_direct_forward_stall_feedback(
        self,
        decision,
        gate_result,
        runtime_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        arguments = getattr(decision, "arguments", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        recent_actions = runtime_payload.get("recent_actions") or []
        if not isinstance(recent_actions, list):
            recent_actions = []
        feedback = {
            "gate_decision": "blocked",
            "blocked_action": "MOVE_FORWARD",
            "allowed_actions": ["TURN_LEFT", "TURN_RIGHT"],
            "reason": (
                "MOVE_FORWARD was rejected because recent forward actions did "
                "not show visual progress. Choose a turn to regain alignment."
            ),
            "recent_actions": [
                self._normalize_action_text(action)
                for action in recent_actions[-4:]
                if self._normalize_action_text(action)
            ],
            "visual_summary": self._bounded_metadata_text(
                arguments.get("visual_summary")
            ),
            "progress_state": self._bounded_metadata_text(
                arguments.get("progress_state")
            ),
            "qwen_reason": self._bounded_metadata_text(
                arguments.get("reason") or getattr(decision, "reason", "")
            ),
        }
        motion_feedback = runtime_payload.get("motion_feedback")
        if isinstance(motion_feedback, dict) and motion_feedback:
            feedback["motion_feedback"] = dict(motion_feedback)
        local_control_context = runtime_payload.get("local_control_context")
        if isinstance(local_control_context, dict):
            odometry = local_control_context.get("odometry")
            if isinstance(odometry, dict) and odometry:
                feedback["odometry"] = {
                    "last_forward_delta_m": odometry.get("last_forward_delta_m"),
                    "collision": odometry.get("collision"),
                    "consecutive_no_progress_forward": odometry.get(
                        "consecutive_no_progress_forward"
                    ),
                }
        return feedback

    def _qwen_direct_requery_metadata(
        self,
        decision,
        gate_result,
    ) -> Dict[str, Any]:
        arguments = getattr(decision, "arguments", {}) or {}
        if not isinstance(arguments, dict):
            arguments = {}
        requery_reason = self._qwen_direct_requery_reason(gate_result)
        metadata = {
            "qwen_direct_requery_triggered": True,
            "qwen_direct_requery_reason": requery_reason,
            "qwen_direct_initial_candidate_action": (
                self._normalize_action_text(arguments.get("action_text"))
                or gate_result.metadata.get("candidate_action")
                or ""
            ),
            "qwen_direct_initial_fallback_action": gate_result.final_action,
            "qwen_direct_initial_final_action_source": gate_result.metadata.get(
                "final_action_source"
            ),
        }
        if requery_reason == "blocked_stop_gate":
            for key in (
                "stop_gate_block_reason",
                "stop_gate_missing_waypoints",
                "stop_gate_missing_route_waypoints",
                "stop_gate_unpassed_route_waypoints",
                "stop_gate_route_missing_reasons",
                "stop_gate_min_step_id",
                "stop_gate_current_step_id",
                "stop_gate_min_forward_actions",
                "stop_gate_forward_action_count",
                "stop_gate_required_route_waypoints",
                "stop_gate_positively_seen_waypoints",
                "stop_gate_passed_waypoints",
                "stop_gate_negated_waypoint_mentions",
                "stop_gate_turn_round_required",
                "stop_gate_turn_round_completed",
                "stop_gate_heading_change_from_start_deg",
            ):
                value = gate_result.metadata.get(key)
                if value not in (None, "", []):
                    metadata[key] = value
            metadata.update(
                {
                    "qwen_direct_initial_stop_evidence": self._bounded_metadata_text(
                        arguments.get("stop_evidence")
                    ),
                    "qwen_direct_initial_current_target": self._bounded_metadata_text(
                        arguments.get("current_target")
                    ),
                    "qwen_direct_initial_target_relation": self._bounded_metadata_text(
                        arguments.get("target_relation")
                    ),
                    "qwen_direct_initial_semantic_stop_state": (
                        self._bounded_metadata_text(
                            arguments.get("semantic_stop_state")
                        )
                    ),
                    "stop_gate_decision": "blocked",
                    "blocked_action": "STOP",
                }
            )
        elif requery_reason == "forward_stall_gate":
            metadata.update(
                {
                    "forward_stall_gate_decision": "blocked",
                    "blocked_action": "MOVE_FORWARD",
                }
            )
            for key in (
                "forward_stall_evidence_source",
                "forward_stall_gate_reason",
                "blocked_forward_by_odometry_gate",
                "forward_stall_odometry_consecutive_no_progress_forward",
                "forward_stall_odometry_last_forward_delta_m",
                "forward_stall_odometry_collision",
            ):
                value = gate_result.metadata.get(key)
                if value is not None:
                    metadata[key] = value
        elif requery_reason == "structural_stop_verification":
            metadata.update(
                {
                    "qwen_direct_initial_stop_evidence": self._bounded_metadata_text(
                        arguments.get("stop_evidence")
                    ),
                    "qwen_direct_initial_current_target": self._bounded_metadata_text(
                        arguments.get("current_target")
                    ),
                    "qwen_direct_initial_target_relation": self._bounded_metadata_text(
                        arguments.get("target_relation")
                    ),
                    "qwen_direct_initial_semantic_stop_state": (
                        self._bounded_metadata_text(
                            arguments.get("semantic_stop_state")
                        )
                    ),
                    "qwen_direct_initial_stop_gate_block_reason": (
                        gate_result.metadata.get("stop_gate_block_reason")
                    ),
                }
            )
        return metadata

    def _context_engine_for_payload(
        self,
        payload: Dict[str, Any],
    ) -> Optional[MemoryAwareContextEngine]:
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return None
        root = Path(run_id) / "openclaw_context_engine"
        key = str(root)
        engine = self.context_engines.get(key)
        if engine is None:
            engine = MemoryAwareContextEngine(root)
            self.context_engines[key] = engine
        return engine

    def _merge_context_engine_plan_context(
        self,
        context_engine: MemoryAwareContextEngine,
        state: VLNState,
        payload: Dict[str, Any],
    ) -> None:
        context = context_engine.prepare_plan_context(
            run_id=str(payload.get("run_id") or ""),
            scene_id=state.scene_id,
            episode_id=state.episode_id,
            instruction=state.instruction,
            step_id=state.step_id,
            payload=payload,
        )
        for key in (
            "task_state",
            "recent_step_summary",
            "retrieved_memory_ids",
            "retrieved_memory_image_paths",
            "retrieved_memory_images",
            "memory_images",
        ):
            value = context.get(key)
            if value:
                payload[key] = value
        memory_context_text = context.get("memory_context_text")
        if memory_context_text and not payload.get("memory_context_text"):
            payload["memory_context_text"] = memory_context_text

    def _record_context_engine_step(
        self,
        metadata: Dict[str, Any],
        context_engine: Optional[MemoryAwareContextEngine],
        state: VLNState,
        payload: Dict[str, Any],
        action_text: str,
        planner_reason: str,
        ok: bool,
        error: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if context_engine is None:
            return
        mirrored_memory_ids = self._mirror_memory_writes_to_context_engine(
            context_engine,
            state,
            tool_calls or [],
        )
        seeded_keyframe_memory_ids = self._seed_keyframe_memory_to_context_engine(
            context_engine,
            state,
            payload,
            planner_reason=planner_reason,
            action_text=action_text,
        )
        record = context_engine.record_step(
            run_id=str(payload.get("run_id") or ""),
            scene_id=state.scene_id,
            episode_id=state.episode_id,
            instruction=state.instruction,
            step_id=state.step_id,
            payload=payload,
            action_text=action_text,
            planner_reason=planner_reason,
            ok=ok,
            error=error,
        )
        if mirrored_memory_ids:
            record["mirrored_memory_ids"] = mirrored_memory_ids
        if seeded_keyframe_memory_ids:
            record["seeded_keyframe_memory_ids"] = seeded_keyframe_memory_ids
        if (
            state.step_id > 0
            and state.step_id % context_engine.review_interval_steps == 0
        ):
            record["review"] = context_engine.review_and_compact(
                current_step_id=state.step_id
            )
        metadata["context_engine"] = record

    def _seed_keyframe_memory_to_context_engine(
        self,
        context_engine: MemoryAwareContextEngine,
        state: VLNState,
        payload: Dict[str, Any],
        planner_reason: str,
        action_text: str,
    ) -> List[str]:
        candidate = payload.get("keyframe_candidate")
        if not isinstance(candidate, dict):
            return []
        image_path = str(candidate.get("image_path") or "")
        if not image_path:
            return []
        if context_engine.has_memory_image(
            image_path=image_path,
            scene_id=state.scene_id,
            episode_id=state.episode_id,
        ):
            return []
        text_parts = [
            state.instruction,
            candidate.get("reason"),
            candidate.get("summary"),
            candidate.get("caption"),
            planner_reason,
            action_text,
        ]
        text = "\n".join(str(part) for part in text_parts if part)
        tags = ["openclaw_keyframe", "episode_keyframe"]
        for key in ("tags", "landmarks", "objects", "spatial_cues"):
            values = candidate.get(key)
            if isinstance(values, list):
                tags.extend(str(value) for value in values if str(value))
        memory_id = context_engine.add_memory(
            text=text or state.instruction or "OpenClaw saved keyframe",
            scene_id=state.scene_id,
            episode_id=state.episode_id,
            step_id=int(candidate.get("step_id") or state.step_id),
            image_path=image_path,
            tags=tags,
            importance=0.7,
        )
        return [memory_id]

    def _mirror_memory_writes_to_context_engine(
        self,
        context_engine: MemoryAwareContextEngine,
        state: VLNState,
        tool_calls: List[Dict[str, Any]],
    ) -> List[str]:
        memory_ids: List[str] = []
        for call in tool_calls:
            if call.get("tool_name") != "MemoryWriteSkill":
                continue
            payload = call.get("payload")
            if not isinstance(payload, dict) or not payload.get("written"):
                continue
            record = payload.get("record")
            if not isinstance(record, dict):
                continue
            text = str(
                record.get("retrieval_text")
                or record.get("visual_observation")
                or record.get("caption")
                or record.get("note")
                or ""
            )
            if not text:
                continue
            tags = []
            for key in ("objects", "landmarks", "spatial_cues"):
                values = record.get(key)
                if isinstance(values, list):
                    tags.extend(str(value) for value in values if str(value))
            memory_ids.append(
                context_engine.add_memory(
                    text=text,
                    scene_id=str(record.get("scene_id") or state.scene_id),
                    episode_id=str(record.get("episode_id") or state.episode_id),
                    step_id=int(record.get("step_id") or state.step_id),
                    image_path=str(record.get("image_path") or ""),
                    tags=tags,
                    importance=0.5,
                )
            )
        return memory_ids

    def _metadata(
        self,
        decision,
        tool_calls: List[Dict[str, Any]],
        image_paths_used: Optional[List[str]] = None,
        state: Optional[VLNState] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        action_text: str = "",
        causal_recall: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = {
            "runtime_mode": "openclaw_bridge",
            "planner_backend": decision.planner_backend,
            "planned_intent": decision.intent,
            "planned_tool": decision.tool_name,
            "planner_reason": decision.reason,
            "tool_calls": [self._summarize_tool_call(call) for call in tool_calls],
        }
        if image_paths_used:
            metadata["image_paths_used"] = image_paths_used
        planner_runtime_metadata = getattr(decision, "runtime_metadata", {}) or {}
        if isinstance(planner_runtime_metadata, dict):
            for key in ("context_audit", "agent_token_guard"):
                value = planner_runtime_metadata.get(key)
                if isinstance(value, dict):
                    metadata[key] = value
        planner_visual_analysis = {}
        if isinstance(planner_runtime_metadata, dict):
            candidate = planner_runtime_metadata.get("visual_analysis")
            if isinstance(candidate, dict):
                planner_visual_analysis = candidate
        memory_writes = self._memory_writes(tool_calls)
        if memory_writes:
            metadata["memory_writes"] = memory_writes
        visual_analysis = self._visual_analysis(tool_calls)
        if planner_visual_analysis:
            metadata["visual_analysis"] = planner_visual_analysis
            if visual_analysis.get("latency_ms") is not None:
                metadata["visual_analysis"][
                    "memory_write_latency_ms"
                ] = visual_analysis.get("latency_ms")
        elif visual_analysis:
            metadata["visual_analysis"] = visual_analysis
        recall_usage = self._recall_usage(
            tool_calls,
            decision=decision,
            state=state,
            runtime_context=runtime_context or {},
            action_text=action_text,
            causal_recall=causal_recall or {},
        )
        if recall_usage:
            metadata["recall_usage"] = recall_usage
        self._attach_local_control_metadata(metadata, runtime_context or {})
        return metadata

    def _update_local_odometry_context(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        odometry = self._odometry_metadata_for_state(state)
        if not odometry:
            return {}
        local_control_context = runtime_payload.get("local_control_context")
        if not isinstance(local_control_context, dict):
            local_control_context = {}
        local_control_context["odometry"] = odometry
        runtime_payload["local_control_context"] = local_control_context
        return odometry

    def _attach_pre_planner_local_control_context(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
    ) -> None:
        odometry = self._update_local_odometry_context(state, runtime_payload)
        self._update_route_progress_from_odometry(state, odometry)
        self._attach_route_progress_context(state, runtime_payload)
        if not odometry or runtime_payload.get("motion_feedback_enabled") is not True:
            return
        motion_feedback = build_motion_feedback(odometry)
        if not motion_feedback:
            return
        runtime_payload["motion_feedback"] = motion_feedback
        control_context = runtime_payload.get("control_context")
        if not isinstance(control_context, dict):
            control_context = {}
        control_context["motion_feedback"] = motion_feedback
        runtime_payload["control_context"] = control_context

    def _attach_local_control_metadata(
        self,
        metadata: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> None:
        motion_feedback = runtime_context.get("motion_feedback")
        if isinstance(motion_feedback, dict) and motion_feedback:
            metadata["motion_feedback"] = motion_feedback
            actual_effect = motion_feedback.get("actual_effect")
            if actual_effect:
                metadata["motion_feedback_actual_effect"] = actual_effect
        route_progress = runtime_context.get("route_progress")
        if isinstance(route_progress, dict) and route_progress:
            metadata["route_progress"] = dict(route_progress)
            for key in (
                "required_waypoints",
                "positively_seen_waypoints",
                "passed_waypoints",
                "negated_waypoint_mentions",
                "heading_change_from_start_deg",
                "turn_round_required",
                "turn_round_completed",
            ):
                if key in route_progress:
                    metadata[f"route_progress_{key}"] = route_progress[key]
        for key in (
            "visual_recovery_active",
            "visual_recovery_reason",
            "visual_recovery_phase",
            "turn_loop_recovery_active",
            "turn_loop_feedback",
        ):
            if key in runtime_context:
                metadata[key] = runtime_context[key]
        local_control_context = runtime_context.get("local_control_context")
        if not isinstance(local_control_context, dict) or not local_control_context:
            return
        metadata["local_control_context"] = local_control_context
        odometry = local_control_context.get("odometry")
        if not isinstance(odometry, dict):
            return
        for key, value in odometry.items():
            metadata[f"odometry_{key}"] = value

    def _odometry_metadata_for_state(self, state: VLNState) -> Dict[str, Any]:
        pose = self._pose_from_state(state)
        if pose is None:
            return {}
        episode_state = self._odometry_state_for_episode(state)
        previous_pose = episode_state.get("last_pose")
        initial_rotation = episode_state.get("initial_rotation")
        if initial_rotation is None:
            initial_rotation = pose.get("rotation")
            episode_state["initial_rotation"] = initial_rotation
        previous_action = self._normalize_action_text(state.last_action) or ""
        position_delta = None
        rotation_delta = None
        if isinstance(previous_pose, dict):
            position_delta = self._position_delta_m(
                previous_pose.get("position"),
                pose.get("position"),
            )
            rotation_delta = self._rotation_delta_deg(
                previous_pose.get("rotation"),
                pose.get("rotation"),
            )

        had_progress = self._last_action_had_progress(
            previous_action,
            position_delta,
            rotation_delta,
        )
        if previous_action == "MOVE_FORWARD" and had_progress is False:
            consecutive_no_progress_forward = (
                self._nonnegative_int(
                    episode_state.get("consecutive_no_progress_forward")
                )
                + 1
            )
        elif previous_action == "MOVE_FORWARD" and had_progress is True:
            consecutive_no_progress_forward = 0
        else:
            consecutive_no_progress_forward = self._nonnegative_int(
                episode_state.get("consecutive_no_progress_forward")
            )

        collision = self._collision_from_state(state)
        if collision is True:
            consecutive_collision = (
                self._nonnegative_int(episode_state.get("consecutive_collision")) + 1
            )
        elif collision is False:
            consecutive_collision = 0
        else:
            consecutive_collision = self._nonnegative_int(
                episode_state.get("consecutive_collision")
            )

        episode_state["last_pose"] = pose
        episode_state[
            "consecutive_no_progress_forward"
        ] = consecutive_no_progress_forward
        episode_state["consecutive_collision"] = consecutive_collision
        heading_change_from_start = self._rotation_delta_deg(
            initial_rotation,
            pose.get("rotation"),
        )

        return {
            "available": True,
            "previous_action": previous_action,
            "position_delta_m": self._round_float(position_delta),
            "rotation_delta_deg": self._round_float(rotation_delta),
            "last_forward_delta_m": self._round_float(
                position_delta if previous_action == "MOVE_FORWARD" else None
            ),
            "last_turn_delta_deg": self._round_float(
                rotation_delta
                if previous_action in {"TURN_LEFT", "TURN_RIGHT"}
                else None
            ),
            "last_action_had_progress": had_progress,
            "consecutive_no_progress_forward": consecutive_no_progress_forward,
            "collision": collision,
            "consecutive_collision": consecutive_collision,
            "progress_threshold_m": ODOMETRY_PROGRESS_THRESHOLD_M,
            "turn_progress_threshold_deg": ODOMETRY_TURN_PROGRESS_THRESHOLD_DEG,
            "heading_change_from_start_deg": self._round_float(
                heading_change_from_start
            ),
        }

    def _odometry_state_for_episode(self, state: VLNState) -> Dict[str, Any]:
        key = f"{state.scene_id}::{state.episode_id}"
        return self.odometry_episode_state.setdefault(key, {})

    def _pose_from_state(self, state: VLNState) -> Optional[Dict[str, Any]]:
        candidates = [state.diagnostic_pose, state.pose]
        diagnostics = state.diagnostics if isinstance(state.diagnostics, dict) else {}
        if diagnostics:
            candidates.append(
                {
                    "position": diagnostics.get("sim_position"),
                    "rotation": diagnostics.get("sim_rotation"),
                }
            )
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            position = self._numeric_list(candidate.get("position"))
            rotation = self._numeric_list(candidate.get("rotation"))
            if position is None and rotation is None:
                continue
            return {"position": position, "rotation": rotation}
        return None

    @staticmethod
    def _numeric_list(value: Any) -> Optional[List[float]]:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if all(hasattr(value, attr) for attr in ("w", "x", "y", "z")):
            value = [value.w, value.x, value.y, value.z]
        if not isinstance(value, (list, tuple)):
            return None
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _position_delta_m(
        previous_position: Any,
        current_position: Any,
    ) -> Optional[float]:
        previous = OpenClawVLNRuntime._numeric_list(previous_position)
        current = OpenClawVLNRuntime._numeric_list(current_position)
        if previous is None or current is None:
            return None
        dims = min(len(previous), len(current))
        if dims == 0:
            return None
        return math.sqrt(
            sum((current[index] - previous[index]) ** 2 for index in range(dims))
        )

    @staticmethod
    def _rotation_delta_deg(
        previous_rotation: Any,
        current_rotation: Any,
    ) -> Optional[float]:
        previous = OpenClawVLNRuntime._quaternion_wxyz(previous_rotation)
        current = OpenClawVLNRuntime._quaternion_wxyz(current_rotation)
        if previous is None or current is None:
            return None
        dot = sum(previous[index] * current[index] for index in range(4))
        dot = max(-1.0, min(1.0, abs(dot)))
        return math.degrees(2.0 * math.acos(dot))

    @staticmethod
    def _quaternion_wxyz(value: Any) -> Optional[List[float]]:
        values = OpenClawVLNRuntime._numeric_list(value)
        if values is None or len(values) != 4:
            return None
        if abs(values[0]) >= abs(values[3]):
            ordered = [values[0], values[1], values[2], values[3]]
        else:
            ordered = [values[3], values[0], values[1], values[2]]
        norm = math.sqrt(sum(item * item for item in ordered))
        if norm <= 0:
            return None
        return [item / norm for item in ordered]

    @staticmethod
    def _last_action_had_progress(
        previous_action: str,
        position_delta_m: Optional[float],
        rotation_delta_deg: Optional[float],
    ) -> Optional[bool]:
        if previous_action == "MOVE_FORWARD":
            if position_delta_m is None:
                return None
            return position_delta_m >= ODOMETRY_PROGRESS_THRESHOLD_M
        if previous_action in {"TURN_LEFT", "TURN_RIGHT"}:
            if rotation_delta_deg is None:
                return None
            return rotation_delta_deg >= ODOMETRY_TURN_PROGRESS_THRESHOLD_DEG
        return None

    @staticmethod
    def _collision_from_state(state: VLNState) -> Optional[bool]:
        metrics = state.online_metrics if isinstance(state.online_metrics, dict) else {}
        if "collision" in metrics:
            return bool(metrics.get("collision"))
        collisions = metrics.get("collisions")
        if isinstance(collisions, dict) and "is_collision" in collisions:
            return bool(collisions.get("is_collision"))
        return None

    @staticmethod
    def _round_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), 6)

    def _qwen_direct_gate_context(
        self,
        runtime_payload: Dict[str, Any],
        decision,
        state: Optional[VLNState] = None,
    ) -> Dict[str, Any]:
        gate_context = dict(runtime_payload)
        if state is not None:
            gate_context.setdefault("current_step_id", state.step_id)
            if state.instruction:
                gate_context.setdefault("instruction", state.instruction)
            self._attach_qwen_direct_episode_gate_context(gate_context, state)
        planner_runtime_metadata = getattr(decision, "runtime_metadata", {}) or {}
        if isinstance(planner_runtime_metadata, dict):
            context_audit = planner_runtime_metadata.get("context_audit")
            if isinstance(context_audit, dict):
                for key in (
                    "planner_step_mode",
                    "visual_memory_age_steps",
                    "model_image_count",
                    "current_image_last",
                ):
                    if key in context_audit and key not in gate_context:
                        gate_context[key] = context_audit[key]
                if context_audit.get("planner_step_mode") == "visual_update":
                    gate_context.setdefault("current_visual_evidence", True)
        control_context = gate_context.get("control_context")
        if isinstance(control_context, dict):
            for key in (
                "force_visual_refresh",
                "recent_forward_count",
                "visual_age_steps",
            ):
                if key in control_context and key not in gate_context:
                    gate_context[key] = control_context[key]
        return gate_context

    def _attach_qwen_direct_episode_gate_context(
        self,
        gate_context: Dict[str, Any],
        state: VLNState,
    ) -> None:
        episode_state = self._qwen_direct_state_for_episode(state)
        observed_visual_summaries = episode_state.get("observed_visual_summaries")
        if isinstance(observed_visual_summaries, list) and observed_visual_summaries:
            existing = gate_context.get("observed_visual_summaries")
            if isinstance(existing, list):
                gate_context["observed_visual_summaries"] = [
                    *observed_visual_summaries,
                    *existing,
                ]
            else:
                gate_context["observed_visual_summaries"] = list(
                    observed_visual_summaries
                )
        self._attach_route_progress_context(state, gate_context)

    def _record_qwen_direct_observation(
        self,
        state: VLNState,
        arguments: Dict[str, Any],
    ) -> None:
        episode_state = self._qwen_direct_state_for_episode(state)
        visual_summary = self._bounded_metadata_text(arguments.get("visual_summary"))
        if visual_summary:
            summaries = list(episode_state.get("observed_visual_summaries") or [])
            if not summaries or summaries[-1] != visual_summary:
                summaries.append(visual_summary)
            episode_state["observed_visual_summaries"] = summaries[-24:]
        required_waypoints = (
            self.qwen_direct_gates.required_intermediate_route_waypoints(
                state.instruction
            )
        )
        if not visual_summary or not required_waypoints:
            return
        waypoint_states = episode_state.setdefault("route_waypoint_states", {})
        for waypoint in required_waypoints:
            status = self.qwen_direct_gates.classify_waypoint_observation(
                visual_summary,
                waypoint,
            )
            waypoint_state = waypoint_states.setdefault(
                waypoint,
                {
                    "positively_seen": False,
                    "effective_forward_after_seen": 0,
                    "passed": False,
                    "negated_mentions": 0,
                },
            )
            if status == "positive":
                waypoint_state["positively_seen"] = True
            elif status == "negated":
                waypoint_state["negated_mentions"] = (
                    self._nonnegative_int(waypoint_state.get("negated_mentions")) + 1
                )

    def _update_route_progress_from_odometry(
        self,
        state: VLNState,
        odometry: Dict[str, Any],
    ) -> None:
        if not odometry:
            return
        if (
            self._normalize_action_text(odometry.get("previous_action"))
            != "MOVE_FORWARD"
        ):
            return
        if odometry.get("last_action_had_progress") is not True:
            return
        episode_state = self._qwen_direct_state_for_episode(state)
        waypoint_states = episode_state.get("route_waypoint_states")
        if not isinstance(waypoint_states, dict):
            return
        for waypoint_state in waypoint_states.values():
            if not isinstance(waypoint_state, dict):
                continue
            if waypoint_state.get("positively_seen") is not True:
                continue
            if waypoint_state.get("passed") is True:
                continue
            forward_count = (
                self._nonnegative_int(
                    waypoint_state.get("effective_forward_after_seen")
                )
                + 1
            )
            waypoint_state["effective_forward_after_seen"] = forward_count
            if forward_count >= ROUTE_WAYPOINT_PASS_FORWARD_ACTIONS:
                waypoint_state["passed"] = True

    def _attach_route_progress_context(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
    ) -> None:
        required_waypoints = (
            self.qwen_direct_gates.required_intermediate_route_waypoints(
                state.instruction
            )
        )
        turn_round_required = self.qwen_direct_gates.route_requires_turn_round(
            state.instruction
        )
        if not required_waypoints and not turn_round_required:
            return
        existing_route_progress = runtime_payload.get("route_progress")
        if not isinstance(existing_route_progress, dict):
            existing_route_progress = {}
        local_control_context = runtime_payload.get("local_control_context")
        odometry = (
            local_control_context.get("odometry")
            if isinstance(local_control_context, dict)
            else {}
        )
        if not isinstance(odometry, dict):
            odometry = {}
        heading_change = self._float_or_none(
            odometry.get("heading_change_from_start_deg")
        )
        if heading_change is None:
            heading_change = self._float_or_none(
                existing_route_progress.get("heading_change_from_start_deg")
            )
        episode_state = self._qwen_direct_state_for_episode(state)
        waypoint_states = episode_state.get("route_waypoint_states")
        if not isinstance(waypoint_states, dict):
            waypoint_states = {}
        serialized_states: Dict[str, Dict[str, Any]] = {}
        positively_seen: List[str] = []
        passed: List[str] = []
        negated: List[str] = []
        for waypoint in required_waypoints:
            state_value = waypoint_states.get(waypoint)
            if not isinstance(state_value, dict):
                existing_states = existing_route_progress.get("waypoint_states")
                state_value = (
                    existing_states.get(waypoint)
                    if isinstance(existing_states, dict)
                    and isinstance(existing_states.get(waypoint), dict)
                    else {}
                )
            existing_seen = existing_route_progress.get("positively_seen_waypoints")
            existing_passed = existing_route_progress.get("passed_waypoints")
            existing_negated = existing_route_progress.get("negated_waypoint_mentions")
            serialized = {
                "positively_seen": (
                    state_value.get("positively_seen") is True
                    or (isinstance(existing_seen, list) and waypoint in existing_seen)
                ),
                "effective_forward_after_seen": self._nonnegative_int(
                    state_value.get("effective_forward_after_seen")
                ),
                "passed": (
                    state_value.get("passed") is True
                    or (
                        isinstance(existing_passed, list)
                        and waypoint in existing_passed
                    )
                ),
                "negated_mentions": self._nonnegative_int(
                    state_value.get("negated_mentions")
                )
                or int(
                    isinstance(existing_negated, list) and waypoint in existing_negated
                ),
            }
            serialized_states[waypoint] = serialized
            if serialized["positively_seen"]:
                positively_seen.append(waypoint)
            if serialized["passed"]:
                passed.append(waypoint)
            if serialized["negated_mentions"] > 0:
                negated.append(waypoint)
        turn_round_completed = (
            not turn_round_required
            or episode_state.get("turn_round_completed") is True
            or (
                heading_change is not None
                and heading_change >= TURN_ROUND_COMPLETION_THRESHOLD_DEG
            )
            or (
                heading_change is None
                and existing_route_progress.get("turn_round_completed") is True
            )
        )
        if turn_round_required and turn_round_completed:
            episode_state["turn_round_completed"] = True
        route_progress = {
            "required_waypoints": required_waypoints,
            "positively_seen_waypoints": positively_seen,
            "passed_waypoints": passed,
            "negated_waypoint_mentions": negated,
            "waypoint_states": serialized_states,
            "waypoint_pass_forward_actions": ROUTE_WAYPOINT_PASS_FORWARD_ACTIONS,
            "heading_change_from_start_deg": self._round_float(heading_change),
            "turn_round_required": turn_round_required,
            "turn_round_completion_threshold_deg": (
                TURN_ROUND_COMPLETION_THRESHOLD_DEG
            ),
            "turn_round_completed": turn_round_completed,
        }
        runtime_payload["route_progress"] = route_progress
        control_context = runtime_payload.get("control_context")
        if not isinstance(control_context, dict):
            control_context = {}
        else:
            control_context = dict(control_context)
        control_context["route_progress"] = route_progress
        runtime_payload["control_context"] = control_context

    def _dynamic_visual_context_active(self, runtime_payload: Dict[str, Any]) -> bool:
        return (
            self.staged_visual_memory_enabled
            or self.dynamic_visual_context_enabled
            or (runtime_payload.get("dynamic_visual_context_enabled") is True)
        )

    def _record_visual_evidence_frame(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
    ) -> None:
        image_path = str(runtime_payload.get("current_image_path") or "")
        if not image_path:
            return
        if (
            self.staged_visual_memory_enabled
            and self.episode_visual_store is not None
            and state.last_action
            and state.step_id > 0
        ):
            self.episode_visual_store.start_episode(state.scene_id, state.episode_id)
            self.episode_visual_store.record_action_after_capture(
                state.step_id - 1,
                str(state.last_action),
            )
        episode_state = self._qwen_direct_state_for_episode(state)
        records = episode_state.setdefault("visual_evidence_registry", [])
        if not isinstance(records, list):
            records = []
            episode_state["visual_evidence_registry"] = records
        heading_deg = self._heading_deg_for_state(state)
        previous_action = self._normalize_action_text(state.last_action) or ""
        keyframe_candidate = runtime_payload.get("keyframe_candidate")
        keyframe_path = (
            str(keyframe_candidate.get("image_path") or "")
            if isinstance(keyframe_candidate, dict)
            else ""
        )
        recent_keyframes = {
            str(path)
            for path in runtime_payload.get("recent_keyframe_paths") or []
            if isinstance(path, str) and path
        }
        record = {
            "step_id": int(state.step_id),
            "image_path": image_path,
            "previous_executed_action": previous_action,
            "heading_deg": self._round_float(heading_deg),
            "keyframe": image_path == keyframe_path or image_path in recent_keyframes,
            "roles": [],
            "confirmed_landmarks": [],
            "capture_route_stage": "unknown",
            "capture_current_target": "",
            "confirmation_basis": "",
        }
        records[:] = [
            existing
            for existing in records
            if not (
                isinstance(existing, dict)
                and (
                    existing.get("image_path") == image_path
                    or existing.get("step_id") == state.step_id
                )
            )
        ]
        records.append(record)
        self._mirror_visual_record_to_episode_store(
            state,
            runtime_payload,
            record,
            provenance="runtime:current_rgb",
        )
        self._record_turn_loop_pose_sample(episode_state, state)
        self._update_visual_recovery_state(episode_state, records, runtime_payload)
        self._evict_visual_evidence_records(episode_state)

    def _record_turn_loop_pose_sample(
        self,
        episode_state: Dict[str, Any],
        state: VLNState,
    ) -> None:
        pose = self._pose_from_state(state)
        position = (
            self._numeric_list(pose.get("position")) if isinstance(pose, dict) else None
        )
        if position is None or len(position) < 3:
            return
        samples = episode_state.setdefault("turn_loop_pose_history", [])
        if not isinstance(samples, list):
            samples = []
            episode_state["turn_loop_pose_history"] = samples
        samples.append(
            {
                "step_id": int(state.step_id),
                "position": position[:3],
            }
        )
        del samples[: max(0, len(samples) - (TURN_LOOP_WINDOW_STEPS + 1))]

    def _update_visual_recovery_state(
        self,
        episode_state: Dict[str, Any],
        records: List[Dict[str, Any]],
        runtime_payload: Dict[str, Any],
    ) -> None:
        current = records[-1] if records else None
        if not isinstance(current, dict):
            return
        local_control = runtime_payload.get("local_control_context")
        odometry = (
            local_control.get("odometry")
            if isinstance(local_control, dict)
            and isinstance(local_control.get("odometry"), dict)
            else {}
        )
        previous_action = self._normalize_action_text(odometry.get("previous_action"))
        had_progress = odometry.get("last_action_had_progress")
        if previous_action == "MOVE_FORWARD" and had_progress is True:
            episode_state.pop("visual_recovery", None)
            return
        if isinstance(episode_state.get("visual_recovery"), dict):
            return
        if previous_action == "MOVE_FORWARD" and had_progress is False:
            if not isinstance(episode_state.get("visual_recovery"), dict):
                prior = records[-2] if len(records) >= 2 else None
                episode_state["visual_recovery"] = {
                    "stuck_before_path": (
                        str(prior.get("image_path") or "")
                        if isinstance(prior, dict)
                        else ""
                    ),
                    "center_path": str(current.get("image_path") or ""),
                    "anchor_heading_deg": current.get("heading_deg"),
                    "anchor_step_id": current.get("step_id"),
                    "reason": "forward_stall",
                }
            return
        detection = self._turn_loop_detection(episode_state, runtime_payload)
        if not detection:
            return
        window_start_step = self._nonnegative_int(detection.get("window_start_step_id"))
        prior = next(
            (
                record
                for record in reversed(records[:-1])
                if isinstance(record, dict)
                and self._nonnegative_int(record.get("step_id")) <= window_start_step
            ),
            records[-2] if len(records) >= 2 else None,
        )
        episode_state["visual_recovery"] = {
            "stuck_before_path": (
                str(prior.get("image_path") or "") if isinstance(prior, dict) else ""
            ),
            "center_path": str(current.get("image_path") or ""),
            "anchor_heading_deg": current.get("heading_deg"),
            "anchor_step_id": current.get("step_id"),
            "reason": "turn_loop",
            **detection,
        }

    def _turn_loop_detection(
        self,
        episode_state: Dict[str, Any],
        runtime_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if runtime_payload.get("forward_stall_odometry_enabled") is not True:
            return {}
        recent_actions = runtime_payload.get("recent_actions")
        if not isinstance(recent_actions, list):
            return {}
        tail = [
            action
            for action in (
                self._normalize_action_text(value)
                for value in recent_actions[-TURN_LOOP_WINDOW_STEPS:]
            )
            if action
        ]
        if len(tail) < TURN_LOOP_MIN_TURNS:
            return {}
        turns = [action for action in tail if action in {"TURN_LEFT", "TURN_RIGHT"}]
        forward_count = sum(action == "MOVE_FORWARD" for action in tail)
        if (
            len(turns) < TURN_LOOP_MIN_TURNS
            or forward_count > TURN_LOOP_MAX_FORWARD_ACTIONS
        ):
            return {}
        left_turns = turns.count("TURN_LEFT")
        right_turns = turns.count("TURN_RIGHT")
        oscillating = left_turns >= 2 and right_turns >= 2
        same_direction_spin = len(turns) >= TURN_LOOP_SAME_DIRECTION_TURNS and (
            left_turns == 0 or right_turns == 0
        )
        route_progress = runtime_payload.get("route_progress")
        legitimate_turn_round = (
            isinstance(route_progress, dict)
            and route_progress.get("turn_round_required") is True
            and route_progress.get("turn_round_completed") is not True
            and not oscillating
        )
        if legitimate_turn_round or not (oscillating or same_direction_spin):
            return {}
        samples = episode_state.get("turn_loop_pose_history")
        if not isinstance(samples, list) or len(samples) < TURN_LOOP_MIN_TURNS:
            return {}
        window = [
            sample for sample in samples[-(len(tail) + 1) :] if isinstance(sample, dict)
        ]
        positions = [self._numeric_list(sample.get("position")) for sample in window]
        positions = [
            position
            for position in positions
            if position is not None and len(position) >= 3
        ]
        if len(positions) < TURN_LOOP_MIN_TURNS:
            return {}
        x_values = [position[0] for position in positions]
        z_values = [position[2] for position in positions]
        translation_span = math.hypot(
            max(x_values) - min(x_values),
            max(z_values) - min(z_values),
        )
        if translation_span > TURN_LOOP_MAX_TRANSLATION_SPAN_M:
            return {}
        return {
            "window_start_step_id": self._nonnegative_int(window[0].get("step_id")),
            "turn_count": len(turns),
            "left_turn_count": left_turns,
            "right_turn_count": right_turns,
            "forward_count": forward_count,
            "translation_span_m": self._round_float(translation_span),
        }

    def _attach_visual_evidence_registry(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
    ) -> None:
        if not self._dynamic_visual_context_active(runtime_payload):
            return
        episode_state = self._qwen_direct_state_for_episode(state)
        records = episode_state.get("visual_evidence_registry")
        if not isinstance(records, list):
            runtime_payload["visual_evidence_registry"] = []
            return
        role_paths = self._visual_role_paths(episode_state, records)
        current_path = str(runtime_payload.get("current_image_path") or "")
        serialized: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            path = str(record.get("image_path") or "")
            roles = {
                str(role)
                for role in record.get("roles") or []
                if isinstance(role, str) and role
            }
            if record.get("keyframe"):
                roles.add("keyframe")
            if path == current_path:
                roles.add("current")
            for role, role_path in role_paths.items():
                if path and path == role_path:
                    roles.add(role)
            item = dict(record)
            item["roles"] = sorted(roles)
            serialized.append(item)
        runtime_payload["visual_evidence_registry"] = serialized
        recovery = episode_state.get("visual_recovery")
        recovery_active = isinstance(recovery, dict)
        runtime_payload["visual_recovery_active"] = recovery_active
        runtime_payload["turn_loop_recovery_active"] = bool(
            recovery_active and recovery.get("reason") == "turn_loop"
        )
        if not recovery_active:
            return
        reason = str(recovery.get("reason") or "")
        runtime_payload["visual_recovery_reason"] = reason
        requested_scan_roles = (
            [] if reason == "turn_loop" else ["left_scan", "right_scan"]
        )
        missing_scan_roles = [
            role for role in requested_scan_roles if not role_paths.get(role)
        ]
        phase = "collect_scans" if missing_scan_roles else "choose_escape"
        runtime_payload["visual_recovery_phase"] = phase
        if reason != "turn_loop":
            return
        feedback = {
            "reason": "repeated turning with insufficient translation",
            "phase": phase,
            "missing_scan_roles": missing_scan_roles,
            "turn_count": self._nonnegative_int(recovery.get("turn_count")),
            "left_turn_count": self._nonnegative_int(recovery.get("left_turn_count")),
            "right_turn_count": self._nonnegative_int(recovery.get("right_turn_count")),
            "forward_count": self._nonnegative_int(recovery.get("forward_count")),
            "translation_span_m": self._round_float(
                self._float_or_none(recovery.get("translation_span_m"))
            ),
            "recovery_goal": (
                "Use the current RGB, local map, and any already available scans to "
                "choose one visibly open escape direction. Do not alternate turns just "
                "to collect missing scans; move forward once the open path is aligned."
            ),
        }
        runtime_payload["turn_loop_feedback"] = feedback
        control_context = runtime_payload.get("control_context")
        if not isinstance(control_context, dict):
            control_context = {}
        else:
            control_context = dict(control_context)
        control_context["turn_loop_feedback"] = feedback
        control_context["force_visual_refresh"] = True
        runtime_payload["control_context"] = control_context

    def _visual_role_paths(
        self,
        episode_state: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        recovery = episode_state.get("visual_recovery")
        if not isinstance(recovery, dict):
            return {}
        role_paths = {
            "stuck_before_keyframe": str(recovery.get("stuck_before_path") or ""),
            "center_scan": str(recovery.get("center_path") or ""),
        }
        anchor_heading = self._float_or_none(recovery.get("anchor_heading_deg"))
        if anchor_heading is None:
            return role_paths
        directional: Dict[str, List[tuple[float, int, str]]] = {
            "left_scan": [],
            "right_scan": [],
        }
        anchor_step = self._nonnegative_int(recovery.get("anchor_step_id"))
        for record in records:
            if not isinstance(record, dict):
                continue
            step_id = self._nonnegative_int(record.get("step_id"))
            if step_id <= anchor_step:
                continue
            heading = self._float_or_none(record.get("heading_deg"))
            if heading is None:
                continue
            delta = self._signed_heading_delta(anchor_heading, heading)
            if not 15.0 <= abs(delta) <= 90.0:
                continue
            role = "left_scan" if delta > 0 else "right_scan"
            directional[role].append(
                (abs(abs(delta) - 45.0), -step_id, str(record.get("image_path") or ""))
            )
        for role, candidates in directional.items():
            if candidates:
                role_paths[role] = min(candidates)[2]
        return role_paths

    def _promote_visual_semantic_evidence(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
        decision: Any,
    ) -> None:
        if not self._dynamic_visual_context_active(runtime_payload):
            return
        runtime_metadata = getattr(decision, "runtime_metadata", {}) or {}
        context_audit = (
            runtime_metadata.get("context_audit")
            if isinstance(runtime_metadata, dict)
            else {}
        )
        if (
            not isinstance(context_audit, dict)
            or context_audit.get("qwen_output_json_valid") is not True
        ):
            return
        arguments = getattr(decision, "arguments", {}) or {}
        if not isinstance(arguments, dict):
            return
        image_path = str(runtime_payload.get("current_image_path") or "")
        episode_state = self._qwen_direct_state_for_episode(state)
        records = episode_state.get("visual_evidence_registry")
        if not isinstance(records, list):
            return
        current_record = next(
            (
                record
                for record in reversed(records)
                if isinstance(record, dict) and record.get("image_path") == image_path
            ),
            None,
        )
        if not isinstance(current_record, dict):
            return
        visual_summary = self._bounded_metadata_text(
            arguments.get("visual_summary"),
            limit=240,
        )
        confirmed = arguments.get("confirmed_landmarks")
        if not isinstance(confirmed, list):
            confirmed = []
        positive_landmarks = []
        for landmark in confirmed[:8]:
            normalized = self._normalize_semantic_text(landmark, 80)
            if not normalized:
                continue
            if (
                self.qwen_direct_gates.classify_waypoint_observation(
                    visual_summary,
                    normalized,
                )
                == "positive"
            ):
                positive_landmarks.append(normalized)
        route_stage = self._normalize_semantic_text(arguments.get("route_stage"), 40)
        if route_stage not in {
            "start",
            "en_route",
            "intermediate_landmark",
            "post_landmark_transition",
            "approaching_target",
            "verifying_target",
            "complete",
            "unknown",
        }:
            route_stage = "unknown"
        current_target = self._normalize_semantic_text(
            arguments.get("current_target"), 120
        )
        current_record["capture_route_stage"] = route_stage
        current_record["capture_current_target"] = current_target
        roles = {
            str(role)
            for role in current_record.get("roles") or []
            if isinstance(role, str)
        }
        if positive_landmarks:
            self._replace_latest_semantic_role(records, "confirmed_landmark")
            roles.add("confirmed_landmark")
            current_record["confirmed_landmarks"] = positive_landmarks
            current_record[
                "confirmation_basis"
            ] = "schema_valid_positive_visual_summary"
        if (
            current_target
            and self.qwen_direct_gates.classify_waypoint_observation(
                visual_summary,
                current_target,
            )
            == "positive"
        ):
            self._replace_latest_semantic_role(records, "target_candidate")
            roles.add("target_candidate")
            if not current_record.get("confirmation_basis"):
                current_record["confirmation_basis"] = "schema_valid_target_candidate"
        current_record["roles"] = sorted(roles)
        self._mirror_visual_record_to_episode_store(
            state,
            runtime_payload,
            current_record,
            visual_summary=visual_summary,
            landmarks=positive_landmarks,
            provenance="runtime:qwen_semantic_promotion",
        )
        self._evict_visual_evidence_records(episode_state)

    def _mirror_visual_record_to_episode_store(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
        record: Dict[str, Any],
        *,
        visual_summary: str = "",
        landmarks: Optional[List[str]] = None,
        provenance: str,
    ) -> None:
        if not self.staged_visual_memory_enabled or self.episode_visual_store is None:
            return
        self.episode_visual_store.start_episode(state.scene_id, state.episode_id)
        local_control = runtime_payload.get("local_control_context")
        odometry = (
            local_control.get("odometry")
            if isinstance(local_control, dict)
            and isinstance(local_control.get("odometry"), dict)
            else {}
        )
        active_stage_id = str(runtime_payload.get("active_stage_id") or "")
        stage_state = runtime_payload.get("stage_state")
        if not active_stage_id and isinstance(stage_state, dict):
            active_stage_id = str(stage_state.get("active_stage_id") or "")
        record_roles = [
            str(role)
            for role in record.get("roles") or []
            if isinstance(role, str) and role
        ]
        if record.get("keyframe") and "keyframe" not in record_roles:
            record_roles.append("keyframe")
        if (
            str(record.get("image_path") or "")
            == str(runtime_payload.get("current_image_path") or "")
            and "current" not in record_roles
        ):
            record_roles.append("current")
        self.episode_visual_store.add_observation(
            image_path=str(record.get("image_path") or ""),
            step_id=int(record.get("step_id", state.step_id)),
            stage_id=active_stage_id,
            visual_summary=visual_summary,
            landmarks=landmarks or record.get("confirmed_landmarks") or [],
            action_before_capture=str(
                record.get("previous_executed_action") or state.last_action or ""
            ),
            odometry_snapshot=odometry,
            image_roles=record_roles,
            provenance=[provenance],
            importance=1.0 if "confirmed_landmark" in record_roles else 0.5,
        )

    @staticmethod
    def _replace_latest_semantic_role(
        records: List[Dict[str, Any]],
        role: str,
    ) -> None:
        for record in records:
            if not isinstance(record, dict):
                continue
            roles = record.get("roles")
            if isinstance(roles, list) and role in roles:
                record["roles"] = [value for value in roles if value != role]

    def _evict_visual_evidence_records(self, episode_state: Dict[str, Any]) -> None:
        records = episode_state.get("visual_evidence_registry")
        if not isinstance(records, list):
            return
        while len(records) > 32:
            pinned_paths = self._pinned_visual_evidence_paths(episode_state, records)
            remove_index = next(
                (
                    index
                    for index, record in enumerate(records[:-1])
                    if isinstance(record, dict)
                    and str(record.get("image_path") or "") not in pinned_paths
                ),
                0,
            )
            records.pop(remove_index)

    def _pinned_visual_evidence_paths(
        self,
        episode_state: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> set[str]:
        pinned = (
            {str(records[-1].get("image_path") or "")}
            if records and isinstance(records[-1], dict)
            else set()
        )
        recovery = episode_state.get("visual_recovery")
        if isinstance(recovery, dict):
            pinned.update(
                str(recovery.get(key) or "")
                for key in ("stuck_before_path", "center_path")
            )
        for role in ("confirmed_landmark", "target_candidate"):
            for record in reversed(records):
                if isinstance(record, dict) and role in (record.get("roles") or []):
                    pinned.add(str(record.get("image_path") or ""))
                    break
        return {path for path in pinned if path}

    @staticmethod
    def _normalize_semantic_text(value: Any, limit: int) -> str:
        return " ".join(str(value or "").strip().lower().split())[:limit]

    def _heading_deg_for_state(self, state: VLNState) -> Optional[float]:
        pose = self._pose_from_state(state)
        if not isinstance(pose, dict):
            return None
        quaternion = self._quaternion_wxyz(pose.get("rotation"))
        if quaternion is None:
            return None
        w, x, y, z = quaternion
        return math.degrees(
            math.atan2(
                2.0 * (w * y + x * z),
                1.0 - 2.0 * (y * y + z * z),
            )
        )

    @staticmethod
    def _signed_heading_delta(anchor_deg: float, current_deg: float) -> float:
        return (float(current_deg) - float(anchor_deg) + 180.0) % 360.0 - 180.0

    def _qwen_direct_state_for_episode(self, state: VLNState) -> Dict[str, Any]:
        key = f"{state.scene_id}::{state.episode_id}"
        return self.qwen_direct_episode_state.setdefault(key, {})

    def _nonnegative_int(self, value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _qwen_direct_schema_metadata(
        self,
        arguments: Dict[str, Any],
        decision_reason: str = "",
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        confidence = self._float_or_none(arguments.get("confidence"))
        if confidence is not None:
            metadata["qwen_confidence"] = confidence
        progress_state = self._bounded_metadata_text(arguments.get("progress_state"))
        if progress_state:
            metadata["qwen_progress_state"] = progress_state
        route_stage = self._bounded_metadata_text(
            arguments.get("route_stage"), limit=40
        )
        if route_stage:
            metadata["qwen_route_stage"] = route_stage
        confirmed_landmarks = arguments.get("confirmed_landmarks")
        if isinstance(confirmed_landmarks, list):
            bounded_landmarks = [
                self._bounded_metadata_text(value, limit=80)
                for value in confirmed_landmarks[:8]
            ]
            metadata["qwen_confirmed_landmarks"] = [
                value for value in bounded_landmarks if value
            ]
        stop_evidence = self._bounded_metadata_text(arguments.get("stop_evidence"))
        if stop_evidence:
            metadata["qwen_stop_evidence"] = stop_evidence
        current_target = self._bounded_metadata_text(arguments.get("current_target"))
        if current_target:
            metadata["qwen_current_target"] = current_target
        target_relation = self._bounded_metadata_text(arguments.get("target_relation"))
        if target_relation:
            metadata["qwen_target_relation"] = target_relation
        semantic_stop_state = self._bounded_metadata_text(
            arguments.get("semantic_stop_state")
        )
        if semantic_stop_state:
            metadata["qwen_semantic_stop_state"] = semantic_stop_state
        visual_summary = self._bounded_metadata_text(arguments.get("visual_summary"))
        metadata["qwen_visual_summary_present"] = bool(visual_summary)
        if visual_summary:
            metadata["qwen_visual_summary"] = visual_summary
        reason = self._bounded_metadata_text(arguments.get("reason") or decision_reason)
        if reason:
            metadata["qwen_reason"] = reason
        return metadata

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bounded_metadata_text(value: Any, limit: int = 500) -> str:
        if value is None:
            return ""
        return str(value).strip()[:limit]

    def _after_recall_decision(
        self,
        state: VLNState,
        original_payload: Dict[str, Any],
        nav_payload: Dict[str, Any],
    ) -> Optional[Any]:
        if original_payload.get("_after_recall_replan"):
            return None
        after_payload = {
            **original_payload,
            **nav_payload,
            "_after_recall_replan": True,
        }
        try:
            return self.planner.plan(state, runtime_context=after_payload)
        except Exception:
            return None

    def _curate_memory_write(
        self,
        state: VLNState,
        arguments: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.tool_adapter.get_tool_schema("VisualMemoryCuratorSkill") is None:
            return None
        return self.tool_adapter.call_tool(
            "VisualMemoryCuratorSkill",
            arguments,
            state=state,
        )

    def _apply_curator_result(
        self,
        arguments: Dict[str, Any],
        curator_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = curator_result.get("payload")
        if not isinstance(payload, dict):
            return arguments
        enriched = dict(arguments)
        if "should_write" in payload:
            enriched["should_write"] = bool(payload.get("should_write"))
        write_gate = payload.get("write_gate")
        if isinstance(write_gate, dict):
            enriched["write_gate"] = write_gate
        return enriched

    def _merge_planner_visual_observations(
        self, payload: Dict[str, Any], decision
    ) -> None:
        planner_runtime_metadata = getattr(decision, "runtime_metadata", {}) or {}
        if not isinstance(planner_runtime_metadata, dict):
            return
        visual_analysis = planner_runtime_metadata.get("visual_analysis") or {}
        if not isinstance(visual_analysis, dict):
            return
        observations = visual_analysis.get("observations") or []
        if not isinstance(observations, list) or not observations:
            return
        existing = payload.get("visual_observations")
        if isinstance(existing, list) and existing:
            return
        payload["visual_observations"] = [
            observation for observation in observations if isinstance(observation, dict)
        ]

    def _auto_write_visual_memory(
        self,
        state: VLNState,
        payload: Dict[str, Any],
        decision,
    ) -> List[Dict[str, Any]]:
        if self.tool_adapter.get_tool_schema("MemoryWriteSkill") is None:
            return []
        if not self._planner_visual_analysis_ran(decision):
            return []
        observation = self._matching_visual_observation(
            payload,
            str(payload.get("current_image_path") or ""),
        )
        if not observation:
            return []
        arguments = self._memory_write_arguments(
            payload,
            {
                "memory_source": "episode-local",
                "note": str(getattr(decision, "reason", "") or ""),
                "write_gate": {
                    "candidate_reason": "openclaw_visual_analysis",
                    "curator_decision": "write",
                    "curator_reason": str(
                        observation.get("navigation_relevance")
                        or observation.get("visual_observation")
                        or observation.get("caption")
                        or "OpenClaw visual analysis"
                    ),
                    "confidence": observation.get("confidence"),
                },
            },
        )
        arguments.setdefault(
            "recent_visual_memories", list(self.recent_visual_memories)
        )
        tool_calls: List[Dict[str, Any]] = []
        curator_result = self._curate_memory_write(state, arguments)
        if curator_result:
            tool_calls.append(curator_result)
            arguments = self._apply_curator_result(arguments, curator_result)
        write_result = self.tool_adapter.call_tool(
            "MemoryWriteSkill",
            arguments,
            state=state,
        )
        tool_calls.append(write_result)
        self._remember_written_visual_memory(write_result)
        return tool_calls

    def _auto_recall_memory(
        self,
        state: VLNState,
        payload: Dict[str, Any],
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        if self.tool_adapter.get_tool_schema("MemoryQuerySkill") is None:
            return None
        if not self.recent_visual_memories:
            return None
        arguments = self._memory_query_arguments(
            state,
            payload,
            {
                "text": state.instruction,
                "step_id": state.step_id,
                "reason": reason,
                "n_results": 3,
                "planner_reason": str(payload.get("active_subgoal") or ""),
            },
        )
        return self.tool_adapter.call_tool(
            "MemoryQuerySkill",
            arguments,
            state=state,
        )

    def _planner_visual_analysis_ran(self, decision) -> bool:
        planner_runtime_metadata = getattr(decision, "runtime_metadata", {}) or {}
        if not isinstance(planner_runtime_metadata, dict):
            return False
        visual_analysis = planner_runtime_metadata.get("visual_analysis") or {}
        if not isinstance(visual_analysis, dict):
            return False
        return bool(visual_analysis.get("ran"))

    def _image_paths_used(self, payload: Dict[str, Any]) -> List[str]:
        paths: List[str] = []
        for value in [payload.get("current_image_path")]:
            if isinstance(value, str) and value:
                paths.append(value)
        recent_paths = payload.get("recent_keyframe_paths") or []
        if isinstance(recent_paths, list):
            for value in recent_paths:
                if isinstance(value, str) and value:
                    paths.append(value)
        deduped: List[str] = []
        for path in paths:
            if path not in deduped:
                deduped.append(path)
        return deduped

    def _memory_write_arguments(
        self,
        payload: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate = payload.get("keyframe_candidate") or {}
        if not isinstance(candidate, dict):
            candidate = {}
        enriched = {**candidate, **arguments}
        if not enriched.get("image_path"):
            current_image_path = payload.get("current_image_path")
            if isinstance(current_image_path, str) and current_image_path:
                enriched["image_path"] = current_image_path
        visual_observation = self._matching_visual_observation(
            payload,
            str(enriched.get("image_path") or ""),
        )
        if visual_observation:
            for key in (
                "caption",
                "visual_observation",
                "objects",
                "landmarks",
                "place_category",
                "spatial_cues",
                "navigation_relevance",
                "confidence",
            ):
                if key in visual_observation and key not in enriched:
                    enriched[key] = visual_observation[key]
        if "write_gate" not in enriched:
            enriched["write_gate"] = self._default_write_gate(
                payload, visual_observation
            )
        enriched.setdefault("should_write", True)
        enriched.setdefault("write_type", "episodic_keyframe")
        return enriched

    def _memory_query_arguments(
        self,
        state: VLNState,
        payload: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = dict(arguments)
        if not enriched.get("visual_observation"):
            visual_observation = self._matching_visual_observation(
                payload,
                str(payload.get("current_image_path") or ""),
            )
            if visual_observation.get("visual_observation"):
                enriched["visual_observation"] = visual_observation[
                    "visual_observation"
                ]
        enriched.setdefault("allowed_scopes", ["episode"])
        if not enriched.get("memory_namespace"):
            enriched[
                "memory_namespace"
            ] = f"episode:{state.scene_id}:{state.episode_id}"
        return enriched

    def _matching_visual_observation(
        self,
        payload: Dict[str, Any],
        image_path: str,
    ) -> Dict[str, Any]:
        observations = payload.get("visual_observations")
        if not isinstance(observations, list):
            return {}
        fallback: Dict[str, Any] = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            if not fallback:
                fallback = observation
            if image_path and observation.get("image_path") == image_path:
                return observation
        return fallback

    def _default_write_gate(
        self,
        payload: Dict[str, Any],
        visual_observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate_reason = "planner_request"
        if payload.get("keyframe_candidate"):
            candidate_reason = "keyframe_candidate"
        return {
            "candidate_reason": candidate_reason,
            "curator_decision": "write",
            "curator_reason": str(
                visual_observation.get("navigation_relevance")
                or visual_observation.get("visual_observation")
                or "planner requested visual memory write"
            ),
            "confidence": visual_observation.get("confidence"),
        }

    def _cli_fallback_kind(self, decision) -> str:
        reason = getattr(decision, "reason", "")
        if not isinstance(reason, str):
            return ""
        if reason.startswith(OPENCLAW_CLI_AGENT_FALLBACK_REASON_PREFIX):
            return "agent"
        if reason.startswith(OPENCLAW_CLI_MODEL_FALLBACK_REASON_PREFIX):
            return "model"
        return ""

    def _navigation_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key in NAVIGATION_CONTEXT_KEYS
        }

    def _merge_navigation_context(
        self,
        nav_payload: Dict[str, Any],
        context: Dict[str, Any],
        include_memory_images: bool = True,
    ) -> None:
        for key in ("active_subgoal", "memory_context_text"):
            value = context.get(key)
            if value:
                nav_payload[key] = value
        memory_images = context.get("memory_images")
        if include_memory_images and memory_images:
            nav_payload["memory_images"] = memory_images

    def _merge_tool_navigation_context(
        self,
        nav_payload: Dict[str, Any],
        tool_result: Dict[str, Any],
        include_memory_images: bool = True,
    ) -> None:
        payload = tool_result.get("payload")
        if not isinstance(payload, dict):
            return
        self._merge_navigation_context(
            nav_payload,
            payload,
            include_memory_images=include_memory_images,
        )
        policy_context = payload.get("policy_context")
        if isinstance(policy_context, dict):
            self._merge_navigation_context(
                nav_payload,
                policy_context,
                include_memory_images=include_memory_images,
            )

    def _planned_action_text(self, arguments: Dict[str, Any]) -> Optional[str]:
        for key in ACTION_ARGUMENT_KEYS:
            action_text = self._normalize_action_text(arguments.get(key))
            if action_text:
                return action_text
        return None

    def _normalize_action_text(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "FORWARD": "MOVE_FORWARD",
            "MOVE": "MOVE_FORWARD",
            "LEFT": "TURN_LEFT",
            "RIGHT": "TURN_RIGHT",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in ACTION_TEXTS:
            return normalized
        return None

    def _summarize_tool_call(self, call: Dict[str, Any]) -> Dict[str, Any]:
        payload = call.get("payload")
        payload_summary: Dict[str, Any] = {}
        if isinstance(payload, dict):
            payload_summary = {
                "keys": sorted(str(key) for key in payload.keys()),
            }
            for key in (
                "action_text",
                "images_used",
                "image_path",
                "caption",
                "visual_observation",
                "write_gate",
                "instruction",
                "active_subgoal",
                "memory_context_text",
            ):
                if key in payload:
                    payload_summary[key] = payload[key]

        return {
            "ok": call.get("ok"),
            "tool_name": call.get("tool_name", ""),
            "result_type": call.get("result_type", ""),
            "runtime_status": call.get("runtime_status", ""),
            "latency_ms": call.get("latency_ms"),
            "error_type": call.get("error_type", ""),
            "error": call.get("error"),
            "payload_summary": payload_summary,
        }

    def _memory_writes(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        writes: List[Dict[str, Any]] = []
        for call in tool_calls:
            if call.get("tool_name") != "MemoryWriteSkill":
                continue
            payload = call.get("payload") or {}
            record = payload.get("record") if isinstance(payload, dict) else {}
            if not isinstance(record, dict):
                record = {}
            writes.append(
                {
                    "written": bool(payload.get("written"))
                    if isinstance(payload, dict)
                    else False,
                    "skipped": bool(payload.get("skipped"))
                    if isinstance(payload, dict)
                    else False,
                    "skip_reason": payload.get("skip_reason", "")
                    if isinstance(payload, dict)
                    else "",
                    "image_path": record.get("image_path") or payload.get("image_path"),
                    "memory_scope": record.get("memory_scope")
                    or payload.get("memory_scope"),
                    "memory_namespace": record.get("memory_namespace")
                    or payload.get("memory_namespace"),
                    "memory_source": record.get("memory_source")
                    or payload.get("memory_source"),
                    "write_gate": record.get("write_gate")
                    or payload.get("write_gate", {}),
                }
            )
        return writes

    def _remember_written_visual_memory(self, tool_result: Dict[str, Any]) -> None:
        payload = tool_result.get("payload") or {}
        if not isinstance(payload, dict) or not payload.get("written"):
            return
        record = payload.get("record")
        if not isinstance(record, dict):
            return
        memory = {
            "memory_id": str(
                record.get("memory_id")
                or record.get("id")
                or record.get("image_path")
                or record.get("step_id")
                or ""
            ),
            "image_path": record.get("image_path", ""),
            "caption": record.get("caption", ""),
            "visual_observation": record.get("visual_observation", ""),
            "objects": record.get("objects", []),
            "landmarks": record.get("landmarks", []),
            "spatial_cues": record.get("spatial_cues", []),
        }
        if not any(
            memory.get(key)
            for key in (
                "memory_id",
                "caption",
                "visual_observation",
                "objects",
                "landmarks",
                "spatial_cues",
            )
        ):
            return
        self.recent_visual_memories.append(memory)
        self.recent_visual_memories = self.recent_visual_memories[
            -self.max_recent_visual_memories :
        ]

    def _visual_analysis(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        for call in tool_calls:
            if call.get("tool_name") != "MemoryWriteSkill":
                continue
            payload = call.get("payload") or {}
            record = payload.get("record") if isinstance(payload, dict) else {}
            if not isinstance(record, dict):
                record = {}
            visual_payload = record or payload
            if visual_payload.get("caption") or visual_payload.get(
                "visual_observation"
            ):
                return {
                    "ran": True,
                    "image_path": visual_payload.get("image_path", ""),
                    "caption": visual_payload.get("caption", ""),
                    "visual_observation": visual_payload.get("visual_observation", ""),
                    "latency_ms": call.get("latency_ms"),
                }
        return {}

    def _recall_usage(
        self,
        tool_calls: List[Dict[str, Any]],
        decision,
        state: Optional[VLNState],
        runtime_context: Dict[str, Any],
        action_text: str,
        causal_recall: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        causal_recall = causal_recall or {}
        before_decision = causal_recall.get("before_decision") or decision
        after_decision = causal_recall.get("after_decision") or decision
        for call in tool_calls:
            if call.get("tool_name") != "MemoryQuerySkill":
                continue
            payload = call.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            hits = payload.get("memory_hits") or []
            hit_ids = []
            hit_image_paths = []
            hit_captions = []
            for hit in hits:
                memory_id = getattr(hit, "memory_id", None)
                image_path = getattr(hit, "image_path", None)
                metadata = getattr(hit, "metadata", {}) or {}
                caption = metadata.get("caption", "")
                if memory_id:
                    hit_ids.append(memory_id)
                if image_path:
                    hit_image_paths.append(image_path)
                if caption:
                    hit_captions.append(caption)
            policy_context = payload.get("policy_context") or {}
            control_context = payload.get("control_context") or {}
            action_before = (
                causal_recall.get("action_before")
                or self._normalize_action_text(runtime_context.get("policy_action"))
                or ""
            )
            action_after = self._normalize_action_text(action_text) or ""
            decision_arguments = getattr(before_decision, "arguments", {}) or {}
            if not isinstance(decision_arguments, dict):
                decision_arguments = {}
            events.append(
                {
                    "event_type": "memory_recall",
                    "step_id": payload.get(
                        "step_id",
                        decision_arguments.get(
                            "step_id",
                            state.step_id if state is not None else None,
                        ),
                    ),
                    "query_text": payload.get("query", ""),
                    "allowed_scopes": payload.get("allowed_scopes")
                    or decision_arguments.get("allowed_scopes", []),
                    "selected_namespace": payload.get("memory_namespace")
                    or decision_arguments.get("memory_namespace", ""),
                    "num_hits": len(hits),
                    "hit_ids": hit_ids,
                    "hit_image_paths": hit_image_paths,
                    "hit_captions": hit_captions,
                    "best_hit_id": hit_ids[0] if hit_ids else "",
                    "recall_confidence": control_context.get(
                        "recall_confidence",
                        control_context.get("confidence", 0.0),
                    ),
                    "used_by_planner": bool(causal_recall.get("after_decision"))
                    or before_decision.intent == "recall_memory",
                    "used_by_policy": bool(
                        policy_context.get("memory_context_text")
                        or policy_context.get("memory_images")
                    ),
                    "used_by_critic": bool(control_context),
                    "planner_intent_before_recall": before_decision.intent,
                    "planner_intent_after_recall": after_decision.intent,
                    "action_before_recall": action_before,
                    "action_after_recall": action_after,
                    "action_changed_after_recall": bool(
                        action_before and action_after and action_before != action_after
                    ),
                    "stop_blocked_after_recall": False,
                    "replan_created_after_recall": after_decision.intent == "replan",
                }
            )
        return events
