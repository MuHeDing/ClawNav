import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from harness.openclaw.gateway_server import make_gateway_server
from harness.openclaw.visual_analyzer import OpenClawVisualAnalyzer


OpenClawRunner = Callable[[List[str], float], subprocess.CompletedProcess]


INTENT_TOOL_NAMES = {
    "act": "NavigationPolicySkill",
    "recall_memory": "MemoryQuerySkill",
    "write_memory": "MemoryWriteSkill",
    "verify_progress": "ProgressCriticSkill",
    "replan": "ReplannerSkill",
}

ACTION_TEXTS = {
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
}

ACTION_ALIASES = {
    "FORWARD": "MOVE_FORWARD",
    "MOVE": "MOVE_FORWARD",
    "LEFT": "TURN_LEFT",
    "RIGHT": "TURN_RIGHT",
}

PROMPT_STATE_KEYS = ("scene_id", "episode_id", "instruction", "step_id", "last_action")
PROMPT_RUNTIME_CONTEXT_KEYS = (
    "run_id",
    "policy_action",
    "current_image_path",
    "recent_keyframe_paths",
    "memory_context_text",
    "memory_images",
    "task_state",
    "recent_step_summary",
    "retrieved_memory_ids",
)
PROMPT_KEYFRAME_KEYS = ("step_id", "reason", "image_path")
PROMPT_TASK_STATE_KEYS = (
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
MAX_PROMPT_RECENT_KEYFRAME_PATHS = 2
MAX_PROMPT_MEMORY_CONTEXT_CHARS = 300
MAX_PROMPT_MEMORY_IMAGES = 2
MAX_PROMPT_RECENT_STEP_SUMMARY_CHARS = 500
MAX_PROMPT_RETRIEVED_MEMORY_IDS = 3
PLAN_MAX_TOTAL_TOKENS = 6000
QWEN_HARD_MAX_INPUT_TOKENS = 50000
OPENCLAW_SESSION_MODE = "fresh_per_step"


def run_openclaw_command(args: List[str], timeout_s: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=timeout_s,
    )


class OpenClawCliPlanPlanner:
    def __init__(
        self,
        recall_interval_steps: int = 5,
        run_openclaw: OpenClawRunner = run_openclaw_command,
        timeout_s: float = 5.0,
        gateway_url: str = "",
        planner_mode: str = "heuristic",
        agent_id: str = "main",
        agent_timeout_s: float = 60.0,
        openclaw_profile: str = "",
        agent_session_id: str = "",
        openclaw_visual_mode: str = "path",
        openclaw_visual_max_images: int = 2,
        openclaw_visual_interval_steps: int = 1,
        openclaw_visual_timeout_ms: int = 30000,
        openclaw_visual_model: str = "",
        visual_analyzer: Any = None,
        agent_session_timestamp: str = "",
        agent_max_input_tokens: int = 10000,
        openclaw_session_dir: str = "",
    ) -> None:
        self.recall_interval_steps = max(1, recall_interval_steps)
        self.run_openclaw = run_openclaw
        self.timeout_s = timeout_s
        self.gateway_url = gateway_url
        self.planner_mode = planner_mode
        self.agent_id = agent_id
        self.agent_timeout_s = agent_timeout_s
        self.openclaw_profile = openclaw_profile
        self.agent_session_id = agent_session_id or "clawnav"
        self.agent_session_timestamp = (
            self._safe_session_part(agent_session_timestamp)
            if agent_session_timestamp
            else datetime.now().strftime("%m%d%H%M")
        )
        self.agent_max_input_tokens = max(0, int(agent_max_input_tokens or 0))
        self.openclaw_session_dir = openclaw_session_dir
        self._agent_token_guard: Dict[str, Any] = {}
        self.openclaw_visual_mode = openclaw_visual_mode
        self.openclaw_visual_max_images = max(0, openclaw_visual_max_images)
        self.openclaw_visual_interval_steps = max(1, openclaw_visual_interval_steps)
        self.visual_analyzer = visual_analyzer
        if self.visual_analyzer is None and self.openclaw_visual_mode == "describe":
            self.visual_analyzer = OpenClawVisualAnalyzer(
                run_openclaw=run_openclaw,
                model=openclaw_visual_model,
                timeout_ms=openclaw_visual_timeout_ms,
            )

    def health_payload(self) -> Dict[str, Any]:
        health = self._gateway_health()
        return {
            "ok": True,
            "service": "openclaw_cli_plan_gateway",
            "planner_mode": self.planner_mode,
            "agent_id": self.agent_id,
            "agent_max_input_tokens": self.agent_max_input_tokens,
            "agent_token_guard": self._agent_token_guard,
            "openclaw_gateway": health,
        }

    def plan_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.planner_mode == "agent":
            try:
                return self._agent_plan(payload)
            except Exception as exc:
                fallback = self._heuristic_plan(payload)
                fallback["reason"] = f"openclaw_cli_agent_fallback:{fallback['reason']}"
                fallback["arguments"] = dict(fallback.get("arguments") or {})
                fallback["arguments"]["planner_error"] = str(exc)
                return fallback

        self._gateway_health()
        return self._heuristic_plan(payload)

    def _heuristic_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state") or {}
        instruction = str(state.get("instruction") or "")
        step_id = int(state.get("step_id") or 0)

        if step_id == 0:
            return self._memory_recall(
                instruction,
                step_id,
                "openclaw_cli_initial_recall",
            )
        if step_id % self.recall_interval_steps == 0:
            return self._memory_recall(
                instruction,
                step_id,
                "openclaw_cli_interval_recall",
            )
        return {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {},
            "reason": "openclaw_cli_default_act",
        }

    def _agent_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._agent_token_guard:
            return self._agent_token_limit_decision(self._agent_token_guard)
        prompt_payload = self._prompt_payload(payload)
        prompt = self._agent_prompt_from_prompt_payload(prompt_payload)
        session_id = self._agent_session_id_for_payload(prompt_payload)
        assembled_prompt_tokens = self._estimate_tokens(prompt)
        args = self._openclaw_args(
            "agent",
            "--agent",
            self.agent_id,
            "--json",
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--timeout",
            str(int(self.agent_timeout_s)),
        )
        result = self.run_openclaw(args, self.agent_timeout_s)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "openclaw agent failed").strip()
            raise RuntimeError(message)
        usage = self._agent_usage_metadata(result.stdout or "")
        if not usage:
            usage = self._agent_usage_from_session_file(session_id)
        context_audit = self._context_audit(
            prompt=prompt,
            usage=usage,
            session_id=session_id,
            assembled_prompt_tokens=assembled_prompt_tokens,
        )
        token_guard = self._agent_token_guard_metadata(usage)
        if token_guard:
            token_guard["context_audit"] = context_audit
            self._agent_token_guard = token_guard
            return self._agent_token_limit_decision(token_guard, context_audit=context_audit)
        agent_text = self._agent_visible_text(result.stdout or "")
        decision = self._extract_json_object(agent_text)
        normalized = self._normalize_decision(decision)
        self._enrich_write_memory_with_visual_observation(normalized, prompt_payload)
        visual_analysis = self._visual_analysis_metadata(prompt_payload)
        runtime_metadata = dict(normalized.get("runtime_metadata") or {})
        if visual_analysis:
            runtime_metadata["visual_analysis"] = visual_analysis
        runtime_metadata["context_audit"] = context_audit
        normalized["runtime_metadata"] = runtime_metadata
        return normalized

    def _agent_usage_metadata(self, stdout: str) -> Dict[str, Any]:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return {}
        usage = self._find_usage_dict(data)
        return dict(usage) if usage else {}

    def _find_usage_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            usage = value.get("usage")
            if isinstance(usage, dict) and (
                "input" in usage or "totalTokens" in usage or "output" in usage
            ):
                return usage
            for nested in value.values():
                found = self._find_usage_dict(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = self._find_usage_dict(item)
                if found:
                    return found
        return {}

    def _agent_usage_from_session_file(self, session_id: str) -> Dict[str, Any]:
        session_file = self._agent_session_file(session_id)
        if not session_file.exists():
            return {}
        last_usage: Dict[str, Any] = {}
        for line in session_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message") if isinstance(data, dict) else None
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            usage = message.get("usage")
            if isinstance(usage, dict):
                last_usage = usage
        return dict(last_usage)

    def _agent_session_file(self, session_id: str) -> Path:
        if self.openclaw_session_dir:
            return Path(self.openclaw_session_dir) / f"{session_id}.jsonl"
        return (
            Path.home()
            / ".openclaw"
            / "agents"
            / self.agent_id
            / "sessions"
            / f"{session_id}.jsonl"
        )

    def _agent_token_guard_metadata(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        if self.agent_max_input_tokens <= 0 or not usage:
            return {}
        input_tokens = int(usage.get("input") or 0)
        if input_tokens < self.agent_max_input_tokens:
            return {}
        return {
            "tripped": True,
            "input_tokens": input_tokens,
            "max_input_tokens": self.agent_max_input_tokens,
            "total_tokens": int(usage.get("totalTokens") or 0),
            "output_tokens": int(usage.get("output") or 0),
        }

    def _agent_token_limit_decision(
        self,
        token_guard: Dict[str, Any],
        context_audit: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        input_tokens = int(token_guard.get("input_tokens") or 0)
        max_input_tokens = int(token_guard.get("max_input_tokens") or self.agent_max_input_tokens)
        planner_error = (
            "openclaw_agent_input_token_limit:"
            f" input_tokens={input_tokens} max_input_tokens={max_input_tokens}"
        )
        return {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {
                "action_text": "STOP",
                "planner_error": planner_error,
                "input_tokens": input_tokens,
                "max_input_tokens": max_input_tokens,
            },
            "reason": "openclaw_agent_input_token_limit",
            "runtime_metadata": {
                "agent_token_guard": token_guard,
                "context_audit": context_audit or token_guard.get("context_audit") or {},
            },
        }

    def _agent_prompt(self, payload: Dict[str, Any]) -> str:
        prompt_payload = self._prompt_payload(payload)
        return self._agent_prompt_from_prompt_payload(prompt_payload)

    def _agent_prompt_from_prompt_payload(self, prompt_payload: Dict[str, Any]) -> str:
        compact_payload = json.dumps(prompt_payload, ensure_ascii=True, sort_keys=True)
        visual_context_lines = self._visual_context_lines(prompt_payload)
        return "\n".join(
            [
                "You are the OpenClaw LLM planner brain for ClawNav.",
                "Return only one JSON object. Do not wrap it in markdown.",
                "Allowed intents and tools:",
                "- act -> NavigationPolicySkill",
                "- recall_memory -> MemoryQuerySkill",
                "- write_memory -> MemoryWriteSkill",
                "- verify_progress -> ProgressCriticSkill",
                "- replan -> ReplannerSkill",
                "Schema:",
                '{"intent":"act|recall_memory|write_memory|verify_progress|replan",'
                '"tool_name":"...",'
                '"arguments":{},'
                '"reason":"short_reason"}',
                "For act/replan, when the next discrete action is known, put it in arguments.action_text.",
                "arguments.action_text must be one of STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT.",
                "For replan, also put the recovery instruction in arguments.active_subgoal.",
                "Do not bury mandatory action commands only in reason.",
                "For write_memory add visual fields and write_gate.",
                "write_gate has candidate_reason, curator_decision write|skip, curator_reason, confidence.",
                "Use episode memory for current; scene memory only approved prior. No future/oracle evidence.",
                "Use act unless memory recall/write, progress verification, or replanning is useful before the next navigation action.",
                *visual_context_lines,
                "Payload:",
                compact_payload,
            ]
        )

    def _agent_session_id_for_payload(self, prompt_payload: Dict[str, Any]) -> str:
        state = prompt_payload.get("state") or {}
        if not isinstance(state, dict):
            state = {}
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            runtime_context = {}
        parts = [
            self.agent_session_id,
            runtime_context.get("run_id") or "",
            state.get("scene_id") or "scene",
            state.get("episode_id") or "episode",
            f"step_{state.get('step_id') if state.get('step_id') is not None else 'unknown'}",
            self.agent_session_timestamp,
        ]
        safe_parts = [self._safe_session_part(part) for part in parts]
        safe_parts = [part for part in safe_parts if part]
        return "-".join(safe_parts)

    @staticmethod
    def _safe_session_part(part: Any) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(part)).strip("_")
        return safe[:80]

    def _context_audit(
        self,
        prompt: str,
        usage: Dict[str, Any],
        session_id: str,
        assembled_prompt_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        assembled = (
            int(assembled_prompt_tokens)
            if assembled_prompt_tokens is not None
            else self._estimate_tokens(prompt)
        )
        provider_input = self._usage_int(usage, "input")
        hidden_history = (
            provider_input - assembled
            if provider_input is not None
            else None
        )
        token_limit_exceeded = bool(
            assembled > QWEN_HARD_MAX_INPUT_TOKENS
            or (
                provider_input is not None
                and provider_input > QWEN_HARD_MAX_INPUT_TOKENS
            )
            or (
                provider_input is not None
                and self.agent_max_input_tokens > 0
                and provider_input >= self.agent_max_input_tokens
            )
        )
        return {
            "context_profile": "plan",
            "assembled_prompt_tokens": assembled,
            "provider_input_tokens": provider_input,
            "hidden_history_tokens_estimate": hidden_history,
            "max_total_tokens": PLAN_MAX_TOTAL_TOKENS,
            "agent_max_input_tokens": self.agent_max_input_tokens,
            "qwen_hard_max_input_tokens": QWEN_HARD_MAX_INPUT_TOKENS,
            "token_limit_exceeded": token_limit_exceeded,
            "history_tokens": 0,
            "openclaw_session_mode": OPENCLAW_SESSION_MODE,
            "openclaw_session_id": session_id,
        }

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _usage_int(self, usage: Dict[str, Any], key: str) -> Optional[int]:
        if not isinstance(usage, dict) or key not in usage:
            return None
        try:
            return int(usage.get(key))
        except (TypeError, ValueError):
            return None

    def _enrich_write_memory_with_visual_observation(
        self,
        decision: Dict[str, Any],
        prompt_payload: Dict[str, Any],
    ) -> None:
        if decision.get("intent") != "write_memory":
            return
        arguments = decision.get("arguments")
        if not isinstance(arguments, dict):
            return
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            return
        observations = runtime_context.get("visual_observations") or []
        if not isinstance(observations, list):
            return
        image_path = str(arguments.get("image_path") or runtime_context.get("current_image_path") or "")
        selected: Dict[str, Any] = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            if not selected:
                selected = observation
            if image_path and observation.get("image_path") == image_path:
                selected = observation
                break
        if not selected:
            return
        for key in (
            "image_path",
            "caption",
            "visual_observation",
            "objects",
            "landmarks",
            "spatial_cues",
            "navigation_relevance",
            "confidence",
        ):
            if key in selected and key not in arguments:
                arguments[key] = selected[key]

    def _prompt_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state") or {}
        prompt_state = self._copy_keys(state, PROMPT_STATE_KEYS)
        runtime_context = payload.get("runtime_context") or {}
        prompt_runtime_context = self._copy_keys(
            runtime_context,
            PROMPT_RUNTIME_CONTEXT_KEYS,
        )
        recent_paths = prompt_runtime_context.get("recent_keyframe_paths")
        if isinstance(recent_paths, list):
            prompt_runtime_context["recent_keyframe_paths"] = recent_paths[
                -MAX_PROMPT_RECENT_KEYFRAME_PATHS:
            ]
        memory_context_text = prompt_runtime_context.get("memory_context_text")
        if isinstance(memory_context_text, str):
            prompt_runtime_context["memory_context_text"] = memory_context_text[
                :MAX_PROMPT_MEMORY_CONTEXT_CHARS
            ]
        memory_images = prompt_runtime_context.get("memory_images")
        if isinstance(memory_images, list):
            prompt_runtime_context["memory_images"] = [
                path
                for path in memory_images[:MAX_PROMPT_MEMORY_IMAGES]
                if isinstance(path, str) and path
            ]
        task_state = prompt_runtime_context.get("task_state")
        if isinstance(task_state, dict):
            prompt_runtime_context["task_state"] = self._copy_keys(
                task_state,
                PROMPT_TASK_STATE_KEYS,
            )
        recent_step_summary = prompt_runtime_context.get("recent_step_summary")
        if isinstance(recent_step_summary, str):
            prompt_runtime_context["recent_step_summary"] = recent_step_summary[
                :MAX_PROMPT_RECENT_STEP_SUMMARY_CHARS
            ]
        retrieved_memory_ids = prompt_runtime_context.get("retrieved_memory_ids")
        if isinstance(retrieved_memory_ids, list):
            prompt_runtime_context["retrieved_memory_ids"] = [
                str(memory_id)
                for memory_id in retrieved_memory_ids[:MAX_PROMPT_RETRIEVED_MEMORY_IDS]
                if memory_id
            ]
        keyframe_candidate = runtime_context.get("keyframe_candidate")
        if isinstance(keyframe_candidate, dict):
            prompt_keyframe = self._copy_keys(keyframe_candidate, PROMPT_KEYFRAME_KEYS)
            if prompt_keyframe:
                prompt_runtime_context["keyframe_candidate"] = prompt_keyframe
        if (
            self.openclaw_visual_mode == "describe"
            and prompt_runtime_context
            and self._should_analyze_visual(prompt_state, prompt_runtime_context)
        ):
            visual_observations = self._visual_observations(prompt_runtime_context)
            if visual_observations:
                prompt_runtime_context["visual_observations"] = visual_observations
        prompt_payload: Dict[str, Any] = {"state": prompt_state}
        if prompt_runtime_context:
            prompt_payload["runtime_context"] = prompt_runtime_context
        return prompt_payload

    def _copy_keys(self, data: Any, keys) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        copied: Dict[str, Any] = {}
        for key in keys:
            value = data.get(key)
            if value is not None and value != "":
                copied[key] = value
        return copied

    def _should_analyze_visual(
        self,
        state: Dict[str, Any],
        runtime_context: Dict[str, Any],
    ) -> bool:
        if self.openclaw_visual_interval_steps <= 1:
            return True
        keyframe_candidate = runtime_context.get("keyframe_candidate")
        if isinstance(keyframe_candidate, dict) and keyframe_candidate.get("image_path"):
            return True
        step_id = state.get("step_id")
        if not isinstance(step_id, int):
            try:
                step_id = int(step_id)
            except (TypeError, ValueError):
                return True
        return step_id % self.openclaw_visual_interval_steps == 0

    def _visual_observations(self, runtime_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.visual_analyzer is None:
            return []
        image_paths = self._visual_analysis_paths(runtime_context)
        if not image_paths:
            return []
        return self.visual_analyzer.analyze(image_paths)

    def _visual_context_lines(self, payload: Dict[str, Any]) -> List[str]:
        runtime_context = payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            return []
        deduped = self._visual_image_paths(runtime_context)
        if not deduped:
            return []
        return [
            "Visual context image paths: " + json.dumps(deduped, ensure_ascii=True),
        ]

    def _visual_image_paths(self, runtime_context: Dict[str, Any]) -> List[str]:
        current_image_path = runtime_context.get("current_image_path")
        recent_keyframe_paths = runtime_context.get("recent_keyframe_paths") or []
        image_paths: List[str] = []
        if isinstance(current_image_path, str) and current_image_path:
            image_paths.append(current_image_path)
        if isinstance(recent_keyframe_paths, list):
            image_paths.extend(
                path for path in recent_keyframe_paths if isinstance(path, str) and path
            )
        deduped: List[str] = []
        for path in image_paths:
            if path not in deduped:
                deduped.append(path)
        return deduped

    def _visual_analysis_paths(self, runtime_context: Dict[str, Any]) -> List[str]:
        image_paths = self._visual_image_paths(runtime_context)
        if self.openclaw_visual_max_images <= 0:
            return []
        if len(image_paths) <= self.openclaw_visual_max_images:
            return image_paths
        current_path = runtime_context.get("current_image_path")
        selected: List[str] = []
        if isinstance(current_path, str) and current_path:
            selected.append(current_path)
        remaining_slots = self.openclaw_visual_max_images - len(selected)
        if remaining_slots > 0:
            recent_paths = [
                path
                for path in image_paths
                if path not in selected
            ]
            selected.extend(recent_paths[-remaining_slots:])
        return selected[: self.openclaw_visual_max_images]

    def _visual_analysis_metadata(self, prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
        runtime_context = prompt_payload.get("runtime_context") or {}
        if not isinstance(runtime_context, dict):
            return {}
        observations = runtime_context.get("visual_observations") or []
        if not isinstance(observations, list) or not observations:
            return {}
        image_paths: List[str] = []
        observation_summaries: List[Dict[str, Any]] = []
        vlm_latency_ms = 0.0
        cache_hits = 0
        failures = 0
        model = ""
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            summary = self._visual_observation_summary(observation)
            if summary:
                observation_summaries.append(summary)
            image_path = observation.get("image_path")
            if isinstance(image_path, str) and image_path:
                image_paths.append(image_path)
            metadata = observation.get("analysis_metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata.get("cache_hit"):
                cache_hits += 1
            else:
                vlm_latency_ms += float(metadata.get("vlm_latency_ms") or 0.0)
            if metadata.get("error") or observation.get("error"):
                failures += 1
            if not model and metadata.get("visual_model"):
                model = str(metadata.get("visual_model"))
        return {
            "ran": True,
            "num_images": len(image_paths),
            "image_paths": image_paths,
            "vlm_latency_ms": round(vlm_latency_ms, 3),
            "cache_hits": cache_hits,
            "failures": failures,
            "model": model,
            "observations": observation_summaries[: self.openclaw_visual_max_images],
        }

    def _visual_observation_summary(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key in (
            "image_path",
            "caption",
            "visual_observation",
            "objects",
            "landmarks",
            "place_category",
            "spatial_cues",
            "navigation_relevance",
            "confidence",
        ):
            value = observation.get(key)
            if value not in (None, "", []):
                summary[key] = value
        return summary

    def _agent_visible_text(self, stdout: str) -> str:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout
        if not isinstance(data, dict):
            return stdout
        for key in ("finalAssistantVisibleText", "finalAssistantRawText", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
        payloads = data.get("payloads")
        if isinstance(payloads, list):
            texts = [
                item.get("text", "")
                for item in payloads
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            joined = "\n".join(text for text in texts if text.strip())
            if joined.strip():
                return joined
        return stdout

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise RuntimeError("openclaw agent did not return a JSON object")

    def _normalize_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        intent = str(decision.get("intent") or "act")
        if intent not in INTENT_TOOL_NAMES:
            raise RuntimeError(f"openclaw agent returned unsupported intent: {intent}")
        arguments = decision.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise RuntimeError("openclaw agent returned non-object arguments")
        arguments = dict(arguments)
        action_text = self._normalize_action_text(arguments.get("action_text"))
        if action_text:
            arguments["action_text"] = action_text
        elif intent in {"act", "replan"}:
            inferred_action = self._infer_action_text(str(decision.get("reason") or ""))
            if inferred_action:
                arguments["action_text"] = inferred_action
        return {
            "intent": intent,
            "tool_name": str(decision.get("tool_name") or INTENT_TOOL_NAMES[intent]),
            "arguments": arguments,
            "reason": str(decision.get("reason") or "openclaw_agent"),
        }

    def _normalize_action_text(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        normalized = ACTION_ALIASES.get(normalized, normalized)
        return normalized if normalized in ACTION_TEXTS else ""

    def _infer_action_text(self, text: str) -> str:
        upper_text = text.upper()
        patterns = [
            r"\bMUST\s+(?:BE\s+)?(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\b",
            r"\b(?:ONLY|CORRECT)\s+(?:ACTION|COMMAND)\s+(?:IS\s+)?(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\b",
            r"\b(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\s+(?:IS\s+)?(?:THE\s+)?ONLY\b",
            r"\b(?:NEXT\s+ACTION|ACTION|COMMAND)\s*(?:IS|:)\s*(STOP|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, upper_text)
            if match:
                return self._normalize_action_text(match.group(1))
        return ""

    def _gateway_health(self) -> Dict[str, Any]:
        args = self._openclaw_args("gateway", "call", "health", "--json")
        if self.gateway_url:
            args.extend(["--url", self.gateway_url])
        args.extend(["--timeout", str(int(self.timeout_s * 1000))])
        result = self.run_openclaw(args, self.timeout_s)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "openclaw gateway health failed").strip()
            raise RuntimeError(message)
        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("openclaw gateway health returned invalid JSON") from exc
        if not isinstance(data, dict) or not data.get("ok"):
            raise RuntimeError("openclaw gateway health is not ok")
        return data

    def _openclaw_args(self, *args: str) -> List[str]:
        command = ["openclaw"]
        if self.openclaw_profile:
            command.extend(["--profile", self.openclaw_profile])
        command.extend(args)
        return command

    def _memory_recall(
        self,
        instruction: str,
        step_id: int,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "intent": "recall_memory",
            "tool_name": "MemoryQuerySkill",
            "arguments": {
                "text": instruction,
                "step_id": step_id,
                "reason": reason,
            },
            "reason": reason,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTTP /plan adapter backed by the OpenClaw CLI WebSocket gateway"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--recall_interval_steps", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--openclaw_gateway_url", default="")
    parser.add_argument("--planner_mode", choices=("heuristic", "agent"), default="heuristic")
    parser.add_argument("--agent_id", default="main")
    parser.add_argument("--agent_timeout", type=float, default=60.0)
    parser.add_argument("--agent_max_input_tokens", type=int, default=10000)
    parser.add_argument("--openclaw_profile", default="")
    parser.add_argument("--agent_session_id", default="")
    parser.add_argument("--openclaw_visual_mode", choices=("path", "describe"), default="path")
    parser.add_argument("--openclaw_visual_max_images", type=int, default=2)
    parser.add_argument("--openclaw_visual_interval_steps", type=int, default=1)
    parser.add_argument("--openclaw_visual_timeout_ms", type=int, default=30000)
    parser.add_argument("--openclaw_visual_model", default="")
    args = parser.parse_args()

    planner = OpenClawCliPlanPlanner(
        recall_interval_steps=args.recall_interval_steps,
        timeout_s=args.timeout,
        gateway_url=args.openclaw_gateway_url,
        planner_mode=args.planner_mode,
        agent_id=args.agent_id,
        agent_timeout_s=args.agent_timeout,
        agent_max_input_tokens=args.agent_max_input_tokens,
        openclaw_profile=args.openclaw_profile,
        agent_session_id=args.agent_session_id,
        openclaw_visual_mode=args.openclaw_visual_mode,
        openclaw_visual_max_images=args.openclaw_visual_max_images,
        openclaw_visual_interval_steps=args.openclaw_visual_interval_steps,
        openclaw_visual_timeout_ms=args.openclaw_visual_timeout_ms,
        openclaw_visual_model=args.openclaw_visual_model,
    )
    server = make_gateway_server(args.host, args.port, planner)
    print(
        f"OpenClaw CLI /plan adapter listening on http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
