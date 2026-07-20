import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
from typing import Any, Dict


SEGMENT_REQUEST_MAX_BYTES = 64 * 1024
SEGMENT_INSTRUCTION_MAX_CHARS = 4096


class GatewayRequestError(ValueError):
    def __init__(self, category: str, status: int = 400) -> None:
        super().__init__(category)
        self.category = category
        self.status = status


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

    def health_payload(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "service": "clawnav_openclaw_gateway",
            "instruction_segmentation": True,
            "stage_schema_versions": ["instruction_stages_v1"],
            "action_schema_versions": ["legacy"],
            "active_stage_schema": "instruction_stages_v1",
            "active_action_schema": "legacy",
        }

    def segment_instruction_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if set(payload) != {"scene_id", "episode_id", "instruction"}:
            raise GatewayRequestError("invalid_request_fields")
        instruction = payload.get("instruction")
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or len(instruction.strip()) > SEGMENT_INSTRUCTION_MAX_CHARS
        ):
            raise GatewayRequestError("invalid_instruction")
        return {
            "stage_plan": {
                "schema_version": "instruction_stages_v1",
                "stages": [
                    {
                        "order": 0,
                        "route_clause": instruction.strip(),
                        "transition_type": "traverse",
                        "expected_landmarks": [],
                        "completion_cues": [],
                        "final_stage": True,
                    }
                ],
            },
            "runtime_metadata": {
                "segmentation_source": "local_fallback",
                "fallback_category": "none",
            },
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
            if self.path not in {"/plan", "/segment_instruction"}:
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                payload = self._read_json(
                    max_bytes=(
                        SEGMENT_REQUEST_MAX_BYTES
                        if self.path == "/segment_instruction"
                        else None
                    )
                )
                if self.path == "/segment_instruction":
                    if not hasattr(planner, "segment_instruction_payload"):
                        self._send_json({"error": "not found"}, status=404)
                        return
                    response = planner.segment_instruction_payload(payload)
                else:
                    response = planner.plan_payload(payload)
            except GatewayRequestError as exc:
                self._send_json(
                    {"error": "request rejected", "category": exc.category},
                    status=exc.status,
                )
                return
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(
                    {"error": "request rejected", "category": "invalid_json"},
                    status=400,
                )
                return
            except ValueError:
                self._send_json(
                    {"error": "request rejected", "category": "invalid_request"},
                    status=400,
                )
                return
            except Exception:
                self._send_json(
                    {"error": "planner failed", "category": "planner_failure"},
                    status=503,
                )
                return
            self._send_json(response)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self, max_bytes: int = None) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if max_bytes is not None and length > max_bytes:
                raise GatewayRequestError("request_too_large", status=413)
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
    if hasattr(planner, "segment_instruction_payload") and not _is_loopback_host(host):
        raise ValueError("instruction segmentation gateway must bind to loopback")
    return ThreadingHTTPServer((host, port), make_gateway_handler(planner))


def _is_loopback_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repo-local OpenClaw-compatible gateway"
    )
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
