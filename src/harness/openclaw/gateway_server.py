import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict


class LocalOpenClawGatewayPlanner:
    def __init__(self, recall_interval_steps: int = 5) -> None:
        self.recall_interval_steps = max(1, recall_interval_steps)

    def plan_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state") or {}
        instruction = str(state.get("instruction") or "")
        step_id = int(state.get("step_id") or 0)

        if step_id == 0:
            return self._memory_recall(instruction, step_id, "gateway_initial_recall")
        if step_id % self.recall_interval_steps == 0:
            return self._memory_recall(instruction, step_id, "gateway_interval_recall")
        return {
            "intent": "act",
            "tool_name": "NavigationPolicySkill",
            "arguments": {},
            "reason": "gateway_default_act",
        }

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


def make_gateway_handler(planner: LocalOpenClawGatewayPlanner):
    class OpenClawGatewayHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                if hasattr(planner, "health_payload"):
                    self._send_json(planner.health_payload())
                else:
                    self._send_json({"ok": True, "service": "clawnav_openclaw_gateway"})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=503)

        def do_POST(self) -> None:
            if self.path != "/plan":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                payload = self._read_json()
                response = planner.plan_payload(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(response)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8") if body else "{}")
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                return

    return OpenClawGatewayHandler


def make_gateway_server(
    host: str,
    port: int,
    planner: LocalOpenClawGatewayPlanner,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_gateway_handler(planner))


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo-local OpenClaw-compatible gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--recall_interval_steps", type=int, default=5)
    args = parser.parse_args()

    planner = LocalOpenClawGatewayPlanner(
        recall_interval_steps=args.recall_interval_steps,
    )
    server = make_gateway_server(args.host, args.port, planner)
    print(f"OpenClaw gateway listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
