from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from harness.config import HarnessConfig
from harness.memory.context_engine import MemoryAwareContextEngine
from harness.openclaw.executor import HabitatOpenClawExecutor
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
        config: Optional[HarnessConfig] = None,
    ) -> None:
        self.tool_adapter = OpenClawToolAdapter(tool_registry)
        self.planner = planner
        self.executor = executor
        self.fallback_planner = fallback_planner
        self.allow_planner_action_override = allow_planner_action_override
        self.config = config or HarnessConfig()
        self.recent_visual_memories: List[Dict[str, Any]] = []
        self.max_recent_visual_memories = 10
        self.context_engines: Dict[str, MemoryAwareContextEngine] = {}
        self._stop_state_episode_key = ("", "")
        self.last_executed_non_stop_action = ""
        self.last_executed_non_stop_action_age = 0

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.tool_adapter.list_tools()

    def step(
        self,
        state: VLNState,
        payload: Dict[str, Any],
    ) -> OpenClawRuntimeStepResult:
        self._reset_stop_state_if_episode_changed(state)
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
            decision = self.fallback_planner.plan(state, runtime_context=runtime_payload)
            planner_fallback = True
        causal_recall: Dict[str, Any] = {}
        self._merge_planner_visual_observations(runtime_payload, decision)

        cli_fallback_kind = self._cli_fallback_kind(decision)
        if cli_fallback_kind:
            metadata = self._metadata(decision, tool_calls, image_paths_used)
            metadata[f"planner_{cli_fallback_kind}_fallback"] = True
            metadata["planner_fallback"] = True
            planner_error = str(decision.arguments.get("planner_error") or decision.reason)
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
                arguments = self._memory_query_arguments(state, runtime_payload, arguments)
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
                after_decision = self._after_recall_decision(state, runtime_payload, nav_payload)
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
            auto_write_calls = self._auto_write_visual_memory(state, runtime_payload, decision)
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
        if decision.intent == "replan" and not nav_payload.get("active_subgoal") and decision.reason:
            nav_payload["active_subgoal"] = decision.reason

        planned_action_text = self._planned_action_text(decision.arguments)
        if planned_action_text and self.allow_planner_action_override:
            metadata = self._metadata(
                decision,
                tool_calls,
                image_paths_used,
                state=state,
                runtime_context=runtime_payload,
                action_text=planned_action_text,
                causal_recall=causal_recall,
                policy_payload_has_memory=False,
            )
            metadata["planner_action_override"] = planned_action_text
            metadata["policy_skipped"] = True
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
            self._update_stop_fallback_state(planned_action_text)
            return OpenClawRuntimeStepResult(
                ok=True,
                action_text=planned_action_text,
                executor_command=self.executor.command_for_action(planned_action_text),
                runtime_metadata=metadata,
            )
        policy_context_available = self._navigation_payload_has_memory(nav_payload)
        context_engine_context_available = bool(runtime_payload.get("memory_context_text"))
        if self._control_only_visual_readback_enabled():
            nav_payload = self._clean_policy_payload(nav_payload)
        final_policy_payload_has_memory = self._navigation_payload_has_memory(nav_payload)

        nav_result = self.tool_adapter.call_tool(
            "NavigationPolicySkill",
            nav_payload,
            state=state,
        )
        tool_calls.append(nav_result)

        action_text = str(nav_result.get("payload", {}).get("action_text") or "STOP")
        smoke_seed_calls = self._smoke_seed_visual_readback_memory(
            state,
            runtime_payload,
            action_text,
        )
        if smoke_seed_calls:
            tool_calls.extend(smoke_seed_calls)
        visual_readback = self._run_visual_readback(
            state=state,
            runtime_payload=runtime_payload,
            tool_calls=tool_calls,
            candidate_action=action_text,
            policy_context_available=policy_context_available,
            context_engine_context_available=context_engine_context_available,
            final_policy_payload_has_memory=final_policy_payload_has_memory,
        )
        if visual_readback:
            tool_calls.append(visual_readback["tool_call"])
            action_text = self._apply_visual_readback_controller(
                action_text,
                visual_readback["trace"],
            )
        metadata = self._metadata(
            decision,
            tool_calls,
            image_paths_used,
            state=state,
            runtime_context=runtime_payload,
            action_text=action_text,
            causal_recall=causal_recall,
            policy_payload_has_memory=final_policy_payload_has_memory,
        )
        if planner_error:
            metadata["planner_error"] = planner_error
        metadata["planner_fallback"] = planner_fallback
        if planned_action_text and not self.allow_planner_action_override:
            metadata["planner_action_guidance"] = planned_action_text
        if visual_readback:
            metadata["visual_readback"] = visual_readback["trace"]
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
        self._update_stop_fallback_state(action_text)
        return OpenClawRuntimeStepResult(
            ok=True,
            action_text=action_text,
            executor_command=self.executor.command_for_action(action_text),
            runtime_metadata=metadata,
        )

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
        for key in ("task_state", "recent_step_summary", "retrieved_memory_ids"):
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
        if state.step_id > 0 and state.step_id % context_engine.review_interval_steps == 0:
            record["review"] = context_engine.review_and_compact(
                current_step_id=state.step_id
            )
        metadata["context_engine"] = record

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
        policy_payload_has_memory: Optional[bool] = None,
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
                metadata["visual_analysis"]["memory_write_latency_ms"] = visual_analysis.get(
                    "latency_ms"
                )
        elif visual_analysis:
            metadata["visual_analysis"] = visual_analysis
        recall_usage = self._recall_usage(
            tool_calls,
            decision=decision,
            state=state,
            runtime_context=runtime_context or {},
            action_text=action_text,
            causal_recall=causal_recall or {},
            policy_payload_has_memory=policy_payload_has_memory,
        )
        if recall_usage:
            metadata["recall_usage"] = recall_usage
        if self.config.visual_readback_mode != "off":
            metadata["visual_readback_config"] = self._visual_readback_config_metadata()
        return metadata

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

    def _merge_planner_visual_observations(self, payload: Dict[str, Any], decision) -> None:
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
            observation
            for observation in observations
            if isinstance(observation, dict)
        ]

    def _auto_write_visual_memory(
        self,
        state: VLNState,
        payload: Dict[str, Any],
        decision,
    ) -> List[Dict[str, Any]]:
        if self.tool_adapter.get_tool_schema("MemoryWriteSkill") is None:
            return []
        arguments = self._auto_visual_memory_write_arguments(state, payload, decision)
        if not arguments:
            return []
        arguments.setdefault("recent_visual_memories", list(self.recent_visual_memories))
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

    def _auto_visual_memory_write_arguments(
        self,
        state: VLNState,
        payload: Dict[str, Any],
        decision,
    ) -> Dict[str, Any]:
        if self._planner_visual_analysis_ran(decision):
            observation = self._matching_visual_observation(
                payload,
                str(payload.get("current_image_path") or ""),
            )
            if not observation:
                return {}
            return self._memory_write_arguments(
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

        context_audit = self._gateway_visual_update_context_audit(decision)
        if not context_audit:
            return {}
        image_path = self._gateway_visual_update_image_path(payload, context_audit)
        if not image_path:
            return {}
        reason = str(getattr(decision, "reason", "") or "")
        retrieval_text = reason or state.instruction
        return self._memory_write_arguments(
            payload,
            {
                "image_path": image_path,
                "memory_source": "episode-local",
                "note": retrieval_text,
                "retrieval_text": retrieval_text,
                "source_image_role": "gateway_visual_update",
                "write_gate": {
                    "candidate_reason": "gateway_visual_memory_update",
                    "curator_decision": "write",
                    "curator_reason": retrieval_text
                    or "OpenClaw gateway visual memory update",
                    "confidence": None,
                },
                "metadata": {
                    "gateway_visual_memory_update_status": context_audit.get(
                        "visual_memory_update_status"
                    ),
                    "gateway_planner_step_mode": context_audit.get("planner_step_mode"),
                    "gateway_model_image_paths": list(
                        context_audit.get("model_image_paths") or []
                    ),
                },
            },
        )

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

    def _smoke_seed_visual_readback_memory(
        self,
        state: VLNState,
        payload: Dict[str, Any],
        candidate_action: str,
    ) -> List[Dict[str, Any]]:
        if not self.config.visual_readback_smoke_seed_memory:
            return []
        if self.config.visual_readback_mode != "image_read_controller":
            return []
        trigger_rule = self._visual_readback_trigger_rule(candidate_action, payload)
        if not trigger_rule:
            return []
        if self.tool_adapter.get_tool_schema("MemoryWriteSkill") is None:
            return []
        if self.tool_adapter.get_tool_schema("MemoryQuerySkill") is None:
            return []
        image_path = self._smoke_seed_image_path(payload)
        if not image_path:
            return []

        write_arguments = self._memory_write_arguments(
            payload,
            {
                "memory_source": self.config.memory_source,
                "memory_scope": "episode",
                "memory_namespace": self._episode_memory_namespace(state),
                "step_id": state.step_id,
                "image_path": image_path,
                "source_image_role": "visual_readback_smoke_seed",
                "note": "visual readback controlled smoke seed",
                "caption": "visual readback controlled smoke seed",
                "visual_observation": str(
                    payload.get("visual_readback_smoke_observation")
                    or state.instruction
                    or "visual readback controlled smoke seed"
                ),
                "retrieval_text": self._smoke_seed_retrieval_text(
                    state,
                    payload,
                    trigger_rule,
                ),
                "write_gate": {
                    "candidate_reason": "visual_readback_smoke_seed_memory",
                    "curator_decision": "write",
                    "curator_reason": "controlled smoke fixture for image readback",
                    "confidence": 1.0,
                },
                "metadata": {
                    "visual_readback_smoke_seed_memory": True,
                    "trigger_rule": trigger_rule,
                    "candidate_action": candidate_action,
                },
            },
        )
        write_result = self.tool_adapter.call_tool(
            "MemoryWriteSkill",
            write_arguments,
            state=state,
        )
        tool_calls = [write_result]
        if not self._memory_write_succeeded(write_result):
            return tool_calls
        self._remember_written_visual_memory(write_result)

        recall_arguments = self._memory_query_arguments(
            state,
            payload,
            {
                "text": self._smoke_seed_retrieval_text(
                    state,
                    payload,
                    trigger_rule,
                ),
                "step_id": state.step_id,
                "reason": "visual_readback_smoke_seed_memory",
                "n_results": self.config.visual_readback_top_k,
                "planner_reason": trigger_rule,
                "allowed_scopes": ["episode"],
                "memory_namespace": self._episode_memory_namespace(state),
            },
        )
        recall_result = self.tool_adapter.call_tool(
            "MemoryQuerySkill",
            recall_arguments,
            state=state,
        )
        tool_calls.append(recall_result)
        payload["visual_readback_trigger_source"] = "controlled_smoke_seed"
        payload["visual_readback_smoke_seed_memory"] = True
        return tool_calls

    def _smoke_seed_image_path(self, payload: Dict[str, Any]) -> str:
        current_image_path = str(payload.get("current_image_path") or "")
        recent_paths = payload.get("recent_keyframe_paths") or []
        if isinstance(recent_paths, list):
            for candidate in recent_paths:
                candidate_path = str(candidate or "")
                if candidate_path and candidate_path != current_image_path:
                    return candidate_path
        return current_image_path

    def _smoke_seed_retrieval_text(
        self,
        state: VLNState,
        payload: Dict[str, Any],
        trigger_rule: str,
    ) -> str:
        parts = [
            state.instruction,
            payload.get("memory_query"),
            payload.get("active_subgoal"),
            trigger_rule,
            "visual readback controlled smoke seed",
        ]
        return "\n".join(str(part) for part in parts if part)

    def _memory_write_succeeded(self, tool_result: Dict[str, Any]) -> bool:
        payload = tool_result.get("payload")
        return bool(
            tool_result.get("ok")
            and isinstance(payload, dict)
            and payload.get("written")
        )

    def _episode_memory_namespace(self, state: VLNState) -> str:
        return f"episode:{state.scene_id}:{state.episode_id}"

    def _planner_visual_analysis_ran(self, decision) -> bool:
        planner_runtime_metadata = getattr(decision, "runtime_metadata", {}) or {}
        if not isinstance(planner_runtime_metadata, dict):
            return False
        visual_analysis = planner_runtime_metadata.get("visual_analysis") or {}
        if not isinstance(visual_analysis, dict):
            return False
        return bool(visual_analysis.get("ran"))

    def _gateway_visual_update_context_audit(self, decision) -> Dict[str, Any]:
        planner_runtime_metadata = getattr(decision, "runtime_metadata", {}) or {}
        if not isinstance(planner_runtime_metadata, dict):
            return {}
        context_audit = planner_runtime_metadata.get("context_audit") or {}
        if not isinstance(context_audit, dict):
            return {}
        if context_audit.get("visual_memory_update_status") != "updated":
            return {}
        image_paths = [
            path
            for path in context_audit.get("model_image_paths") or []
            if isinstance(path, str) and path
        ]
        if not image_paths:
            return {}
        audit = dict(context_audit)
        audit["model_image_paths"] = image_paths
        return audit

    def _gateway_visual_update_image_path(
        self,
        payload: Dict[str, Any],
        context_audit: Dict[str, Any],
    ) -> str:
        image_paths = [
            path
            for path in context_audit.get("model_image_paths") or []
            if isinstance(path, str) and path
        ]
        if not image_paths:
            return ""
        current_image_path = payload.get("current_image_path")
        if isinstance(current_image_path, str) and current_image_path in image_paths:
            return current_image_path
        keyframe_candidate = payload.get("keyframe_candidate") or {}
        if isinstance(keyframe_candidate, dict):
            candidate_path = keyframe_candidate.get("image_path")
            if isinstance(candidate_path, str) and candidate_path in image_paths:
                return candidate_path
        return image_paths[0]

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
            enriched["write_gate"] = self._default_write_gate(payload, visual_observation)
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
                enriched["visual_observation"] = visual_observation["visual_observation"]
        enriched.setdefault("allowed_scopes", ["episode"])
        if not enriched.get("memory_namespace"):
            enriched["memory_namespace"] = f"episode:{state.scene_id}:{state.episode_id}"
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

    def _control_only_visual_readback_enabled(self) -> bool:
        return self.config.visual_readback_mode in {
            "image_read_controller",
            "current_only_controller",
        }

    def _clean_policy_payload(self, nav_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in nav_payload.items()
            if key not in {"active_subgoal", "memory_context_text", "memory_images"}
        }

    def _navigation_payload_has_memory(self, nav_payload: Dict[str, Any]) -> bool:
        return bool(
            nav_payload.get("active_subgoal")
            or nav_payload.get("memory_context_text")
            or nav_payload.get("memory_images")
        )

    def _run_visual_readback(
        self,
        state: VLNState,
        runtime_payload: Dict[str, Any],
        tool_calls: List[Dict[str, Any]],
        candidate_action: str,
        policy_context_available: bool,
        context_engine_context_available: bool,
        final_policy_payload_has_memory: bool,
    ) -> Dict[str, Any]:
        if not self._control_only_visual_readback_enabled():
            return {}
        if self.tool_adapter.get_tool_schema("VisualMemoryReadSkill") is None:
            return {}
        trigger_rule = self._visual_readback_trigger_rule(
            candidate_action,
            runtime_payload,
        )
        base_trace = {
            "mode": self.config.visual_readback_mode,
            "trigger_source": str(
                runtime_payload.get("visual_readback_trigger_source")
                or "online_controller"
            ),
            "trigger_rule": trigger_rule,
            "candidate_action": candidate_action,
            "policy_context_available": policy_context_available,
            "context_engine_context_available": context_engine_context_available,
            "actual_policy_payload_merge": final_policy_payload_has_memory,
            "used_by_policy": final_policy_payload_has_memory,
            "readback_state_used_by_policy": False,
            "final_policy_payload_has_memory": final_policy_payload_has_memory,
        }
        if not trigger_rule:
            base_trace.update({"read_status": "skipped", "skip_reason": "no_trigger"})
            return {"trace": base_trace, "tool_call": self._skipped_visual_readback_call(base_trace)}
        current_image_path = str(runtime_payload.get("current_image_path") or "")
        if not current_image_path:
            base_trace.update(
                {"read_status": "skipped", "skip_reason": "missing_current_image"}
            )
            return {"trace": base_trace, "tool_call": self._skipped_visual_readback_call(base_trace)}

        memory_hits = []
        if self.config.visual_readback_mode == "image_read_controller":
            memory_hits = self._latest_memory_hits(tool_calls)
        if self.config.visual_readback_mode == "image_read_controller" and not memory_hits:
            base_trace.update({"read_status": "skipped", "skip_reason": "no_memory_hit"})
            return {"trace": base_trace, "tool_call": self._skipped_visual_readback_call(base_trace)}

        tool_call = self.tool_adapter.call_tool(
            "VisualMemoryReadSkill",
            {
                "current_image_path": current_image_path,
                "memory_hits": memory_hits,
                "candidate_action": candidate_action,
                "trigger_rule": trigger_rule,
            },
            state=state,
        )
        trace = dict(base_trace)
        payload = tool_call.get("payload")
        if isinstance(payload, dict):
            trace.update(payload)
        return {"trace": trace, "tool_call": tool_call}

    def _apply_visual_readback_controller(
        self,
        candidate_action: str,
        trace: Dict[str, Any],
    ) -> str:
        action = self._normalize_action_text(candidate_action) or "STOP"
        trace.setdefault("controller_decision", "none")
        trace.setdefault("fallback_policy", self.config.visual_readback_stop_fallback_policy)
        trace.setdefault("fallback_source", "")
        trace.setdefault("fallback_denominator", "")
        trace.setdefault("stop_block_reason", "")
        trace.setdefault("final_action", action)
        trace.setdefault("controller_decision_changed_after_visual_read", False)
        trace.setdefault("executed_action_changed_after_visual_read", False)
        trace.setdefault("stop_blocked_after_visual_read", False)
        trace.setdefault("current_view_stop_verification", False)
        trace.setdefault("historical_memory_stop_causal", False)
        trace.setdefault("replan_request_logged_after_visual_read", False)
        trace["last_executed_non_stop_action"] = self.last_executed_non_stop_action
        trace["last_executed_non_stop_action_age"] = self.last_executed_non_stop_action_age

        labels = set(str(label) for label in trace.get("verifier_labels") or [])
        high_confidence = float(trace.get("verifier_confidence") or 0.0) >= float(
            self.config.visual_readback_high_confidence
        )
        if action == "STOP" and high_confidence and (
            "goal_not_visible" in labels or "insufficient_evidence" in labels
        ):
            return self._apply_stop_block(action, trace, labels)
        if action in {"TURN_LEFT", "TURN_RIGHT"} and high_confidence and "route_conflict" in labels:
            trace["controller_decision"] = "log_replan_request"
            trace["controller_decision_changed_after_visual_read"] = True
            trace["replan_request_logged_after_visual_read"] = True
        return action

    def _apply_stop_block(
        self,
        action: str,
        trace: Dict[str, Any],
        labels: set[str],
    ) -> str:
        stop_reason = "goal_not_visible" if "goal_not_visible" in labels else "insufficient_evidence"
        trace["stop_block_reason"] = stop_reason
        trace["stop_blocked_after_visual_read"] = True
        trace["current_view_stop_verification"] = True
        trace["controller_decision_changed_after_visual_read"] = True
        policy = self.config.visual_readback_stop_fallback_policy
        trace["fallback_policy"] = policy
        if policy == "previous_non_stop_else_move_forward":
            fallback_action = self.last_executed_non_stop_action or "MOVE_FORWARD"
            trace["controller_decision"] = "block_stop_executed"
            trace["fallback_source"] = (
                "previous_non_stop"
                if self.last_executed_non_stop_action
                else "default_move_forward"
            )
            trace["fallback_denominator"] = "executed_pilot"
            trace["final_action"] = fallback_action
            trace["executed_action_changed_after_visual_read"] = fallback_action != action
            return fallback_action
        trace["controller_decision"] = "block_stop_shadow"
        trace["fallback_source"] = "log_only"
        trace["fallback_denominator"] = "shadow_primary"
        trace["final_action"] = action
        trace["executed_action_changed_after_visual_read"] = False
        return action

    def _visual_readback_trigger_rule(
        self,
        candidate_action: str,
        runtime_payload: Dict[str, Any],
    ) -> str:
        explicit = str(runtime_payload.get("visual_readback_trigger_rule") or "")
        if explicit:
            return explicit
        action = self._normalize_action_text(candidate_action) or ""
        if action == "STOP":
            return "risky_stop"
        if action in {"TURN_LEFT", "TURN_RIGHT"}:
            return "decision_point"
        return ""

    def _latest_memory_hits(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        for call in reversed(tool_calls):
            if call.get("tool_name") != "MemoryQuerySkill":
                continue
            payload = call.get("payload")
            if not isinstance(payload, dict):
                continue
            hits = payload.get("memory_hits") or []
            return list(hits)
        return []

    def _skipped_visual_readback_call(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ok": True,
            "tool_name": "VisualMemoryReadSkill",
            "result_type": "visual_memory_read",
            "runtime_status": "skipped",
            "latency_ms": 0.0,
            "payload": dict(trace),
            "error": None,
        }

    def _visual_readback_config_metadata(self) -> Dict[str, Any]:
        return {
            "visual_readback_mode": self.config.visual_readback_mode,
            "visual_readback_top_k": self.config.visual_readback_top_k,
            "visual_readback_timeout_ms": self.config.visual_readback_timeout_ms,
            "visual_readback_low_confidence": self.config.visual_readback_low_confidence,
            "visual_readback_high_confidence": self.config.visual_readback_high_confidence,
            "visual_readback_shuffle_scope": self.config.visual_readback_shuffle_scope,
            "visual_readback_shuffle_seed": self.config.visual_readback_shuffle_seed,
            "visual_readback_control_only": self.config.visual_readback_control_only,
            "visual_readback_fixed_case_manifest_path": (
                self.config.visual_readback_fixed_case_manifest_path
            ),
            "visual_readback_stop_fallback_policy": (
                self.config.visual_readback_stop_fallback_policy
            ),
            "visual_readback_smoke_seed_memory": (
                self.config.visual_readback_smoke_seed_memory
            ),
        }

    def _reset_stop_state_if_episode_changed(self, state: VLNState) -> None:
        episode_key = (str(state.scene_id or ""), str(state.episode_id or ""))
        if episode_key == self._stop_state_episode_key:
            return
        self._stop_state_episode_key = episode_key
        self.last_executed_non_stop_action = ""
        self.last_executed_non_stop_action_age = 0

    def _update_stop_fallback_state(self, action_text: str) -> None:
        action = self._normalize_action_text(action_text)
        if action in {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"}:
            self.last_executed_non_stop_action = action
            self.last_executed_non_stop_action_age = 0
            return
        if self.last_executed_non_stop_action:
            self.last_executed_non_stop_action_age += 1

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
                    "written": bool(payload.get("written")) if isinstance(payload, dict) else False,
                    "skipped": bool(payload.get("skipped")) if isinstance(payload, dict) else False,
                    "skip_reason": payload.get("skip_reason", "") if isinstance(payload, dict) else "",
                    "image_path": record.get("image_path") or payload.get("image_path"),
                    "memory_scope": record.get("memory_scope") or payload.get("memory_scope"),
                    "memory_namespace": record.get("memory_namespace") or payload.get("memory_namespace"),
                    "memory_source": record.get("memory_source") or payload.get("memory_source"),
                    "write_gate": record.get("write_gate") or payload.get("write_gate", {}),
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
            -self.max_recent_visual_memories:
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
            if visual_payload.get("caption") or visual_payload.get("visual_observation"):
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
        policy_payload_has_memory: Optional[bool] = None,
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
            used_by_policy = (
                bool(policy_payload_has_memory)
                if policy_payload_has_memory is not None
                else bool(
                    policy_context.get("memory_context_text")
                    or policy_context.get("memory_images")
                )
            )
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
                    "used_by_policy": used_by_policy,
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
