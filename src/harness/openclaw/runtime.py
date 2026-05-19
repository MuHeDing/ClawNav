from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

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
    "memory_images",
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
    ) -> None:
        self.tool_adapter = OpenClawToolAdapter(tool_registry)
        self.planner = planner
        self.executor = executor
        self.fallback_planner = fallback_planner
        self.recent_visual_memories: List[Dict[str, Any]] = []
        self.max_recent_visual_memories = 10

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.tool_adapter.list_tools()

    def step(
        self,
        state: VLNState,
        payload: Dict[str, Any],
    ) -> OpenClawRuntimeStepResult:
        planner_error = ""
        planner_fallback = False
        image_paths_used = self._image_paths_used(payload)
        try:
            decision = self.planner.plan(state, runtime_context=payload)
        except Exception as exc:
            planner_error = str(exc)
            if self.fallback_planner is None:
                return OpenClawRuntimeStepResult(
                    ok=False,
                    action_text="STOP",
                    runtime_metadata={
                        "runtime_mode": "openclaw_bridge",
                        "planner_fallback": False,
                        "planner_error": planner_error,
                    },
                    error=planner_error,
                )
            decision = self.fallback_planner.plan(state, runtime_context=payload)
            planner_fallback = True
        tool_calls: List[Dict[str, Any]] = []
        causal_recall: Dict[str, Any] = {}

        if self._is_cli_agent_fallback_decision(decision):
            metadata = self._metadata(decision, tool_calls, image_paths_used)
            metadata["planner_agent_fallback"] = True
            metadata["planner_fallback"] = planner_fallback
            planner_error = str(decision.arguments.get("planner_error") or decision.reason)
            if planner_error:
                metadata["planner_error"] = planner_error
            return OpenClawRuntimeStepResult(
                ok=False,
                action_text="STOP",
                executor_command=self.executor.command_for_action("STOP"),
                runtime_metadata=metadata,
                error=planner_error or "openclaw_cli_agent_fallback",
            )

        nav_payload = self._navigation_payload(payload)

        if decision.intent in PRE_ACTION_INTENTS:
            arguments = dict(decision.arguments)
            self._merge_navigation_context(nav_payload, arguments)
            if decision.intent == "recall_memory":
                arguments = self._memory_query_arguments(state, payload, arguments)
            if decision.intent == "write_memory":
                arguments = self._memory_write_arguments(payload, arguments)
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
            self._merge_tool_navigation_context(nav_payload, tool_result)
            if decision.intent == "recall_memory":
                after_decision = self._after_recall_decision(state, payload, nav_payload)
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
                        self._merge_navigation_context(nav_payload, decision.arguments)

        self._merge_navigation_context(nav_payload, decision.arguments)
        if decision.intent == "replan" and not nav_payload.get("active_subgoal") and decision.reason:
            nav_payload["active_subgoal"] = decision.reason

        planned_action_text = self._planned_action_text(decision.arguments)
        if planned_action_text:
            metadata = self._metadata(
                decision,
                tool_calls,
                image_paths_used,
                state=state,
                runtime_context=payload,
                action_text=planned_action_text,
                causal_recall=causal_recall,
            )
            metadata["planner_action_override"] = planned_action_text
            metadata["policy_skipped"] = True
            if planner_error:
                metadata["planner_error"] = planner_error
            metadata["planner_fallback"] = planner_fallback
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
        metadata = self._metadata(
            decision,
            tool_calls,
            image_paths_used,
            state=state,
            runtime_context=payload,
            action_text=action_text,
            causal_recall=causal_recall,
        )
        if planner_error:
            metadata["planner_error"] = planner_error
        metadata["planner_fallback"] = planner_fallback
        if not nav_result.get("ok"):
            return OpenClawRuntimeStepResult(
                ok=False,
                action_text="STOP",
                runtime_metadata=metadata,
                error=nav_result.get("error") or "navigation_failed",
            )

        return OpenClawRuntimeStepResult(
            ok=True,
            action_text=action_text,
            executor_command=self.executor.command_for_action(action_text),
            runtime_metadata=metadata,
        )

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
        memory_writes = self._memory_writes(tool_calls)
        if memory_writes:
            metadata["memory_writes"] = memory_writes
        visual_analysis = self._visual_analysis(tool_calls)
        if visual_analysis:
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

    def _is_cli_agent_fallback_decision(self, decision) -> bool:
        reason = getattr(decision, "reason", "")
        return (
            isinstance(reason, str)
            and reason.startswith(OPENCLAW_CLI_AGENT_FALLBACK_REASON_PREFIX)
        )

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
    ) -> None:
        for key in ("active_subgoal", "memory_context_text"):
            value = context.get(key)
            if value:
                nav_payload[key] = value
        memory_images = context.get("memory_images")
        if memory_images:
            nav_payload["memory_images"] = memory_images

    def _merge_tool_navigation_context(
        self,
        nav_payload: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> None:
        payload = tool_result.get("payload")
        if not isinstance(payload, dict):
            return
        self._merge_navigation_context(nav_payload, payload)
        policy_context = payload.get("policy_context")
        if isinstance(policy_context, dict):
            self._merge_navigation_context(nav_payload, policy_context)

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
