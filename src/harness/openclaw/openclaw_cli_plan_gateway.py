import argparse
import json
import subprocess
from typing import Any, Callable, Dict, List, Optional

from harness.openclaw.gateway_server import make_gateway_server


OpenClawRunner = Callable[[List[str], float], subprocess.CompletedProcess]


INTENT_TOOL_NAMES = {
    "act": "NavigationPolicySkill",
    "recall_memory": "MemoryQuerySkill",
    "write_memory": "MemoryWriteSkill",
    "verify_progress": "ProgressCriticSkill",
    "replan": "ReplannerSkill",
}


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
    ) -> None:
        self.recall_interval_steps = max(1, recall_interval_steps)
        self.run_openclaw = run_openclaw
        self.timeout_s = timeout_s
        self.gateway_url = gateway_url
        self.planner_mode = planner_mode
        self.agent_id = agent_id
        self.agent_timeout_s = agent_timeout_s

    def health_payload(self) -> Dict[str, Any]:
        health = self._gateway_health()
        return {
            "ok": True,
            "service": "openclaw_cli_plan_gateway",
            "planner_mode": self.planner_mode,
            "agent_id": self.agent_id,
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
        prompt = self._agent_prompt(payload)
        args = [
            "openclaw",
            "agent",
            "--agent",
            self.agent_id,
            "--json",
            "--message",
            prompt,
            "--timeout",
            str(int(self.agent_timeout_s)),
        ]
        result = self.run_openclaw(args, self.agent_timeout_s)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "openclaw agent failed").strip()
            raise RuntimeError(message)
        agent_text = self._agent_visible_text(result.stdout or "")
        decision = self._extract_json_object(agent_text)
        return self._normalize_decision(decision)

    def _agent_prompt(self, payload: Dict[str, Any]) -> str:
        compact_payload = json.dumps(payload, ensure_ascii=True, sort_keys=True)
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
                "Use act unless memory recall/write, progress verification, or replanning is useful before the next navigation action.",
                "Payload:",
                compact_payload,
            ]
        )

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
        return {
            "intent": intent,
            "tool_name": str(decision.get("tool_name") or INTENT_TOOL_NAMES[intent]),
            "arguments": arguments,
            "reason": str(decision.get("reason") or "openclaw_agent"),
        }

    def _gateway_health(self) -> Dict[str, Any]:
        args = ["openclaw", "gateway", "call", "health", "--json"]
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
    args = parser.parse_args()

    planner = OpenClawCliPlanPlanner(
        recall_interval_steps=args.recall_interval_steps,
        timeout_s=args.timeout,
        gateway_url=args.openclaw_gateway_url,
        planner_mode=args.planner_mode,
        agent_id=args.agent_id,
        agent_timeout_s=args.agent_timeout,
    )
    server = make_gateway_server(args.host, args.port, planner)
    print(
        f"OpenClaw CLI /plan adapter listening on http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
