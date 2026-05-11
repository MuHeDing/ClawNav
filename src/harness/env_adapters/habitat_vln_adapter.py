from typing import Any, Dict, Optional

from harness.env_adapters.base import BaseEmbodimentAdapter
from harness.types import VLNState


ACTIONS2IDX = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}

ORACLE_METRIC_KEYS = {
    "distance_to_goal",
    "success",
    "SPL",
    "spl",
    "softspl",
    "oracle_success",
    "oracle_path",
    "oracle_shortest_path",
    "oracle_shortest_path_action",
}


class HabitatVLNAdapter(BaseEmbodimentAdapter):
    def __init__(self, expose_pose_online: bool = False) -> None:
        self.expose_pose_online = expose_pose_online

    def build_state(
        self,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        metrics: Optional[Dict[str, Any]],
        step_id: int,
        last_action: Optional[str] = None,
    ) -> VLNState:
        metrics = metrics or {}
        pose = self._extract_agent_pose(env)
        diagnostics = self._diagnostics(metrics)
        online_pose = pose if self.expose_pose_online else None
        if pose is not None and not self.expose_pose_online:
            diagnostics["sim_position"] = pose.get("position")
            diagnostics["sim_rotation"] = pose.get("rotation")

        return VLNState(
            scene_id=str(getattr(episode, "scene_id", "")),
            episode_id=str(getattr(episode, "episode_id", "")),
            instruction=self._extract_instruction(episode),
            step_id=step_id,
            current_image=self._extract_rgb(observations),
            online_metrics=self._online_metrics(metrics),
            diagnostics=diagnostics,
            pose=online_pose,
            diagnostic_pose=pose if not self.expose_pose_online else None,
            last_action=last_action,
        )

    def action_space(self):
        return list(ACTIONS2IDX.keys())

    def action_to_executor_command(self, action_text: str) -> Dict[str, Any]:
        if action_text not in ACTIONS2IDX:
            raise ValueError(f"Unsupported Habitat action: {action_text}")
        return {
            "executor": "habitat_discrete",
            "action_text": action_text,
            "action_index": ACTIONS2IDX[action_text],
        }

    def _extract_rgb(self, observations: Dict[str, Any]) -> Any:
        return observations.get("rgb")

    def _extract_instruction(self, episode: Any) -> str:
        instruction = getattr(episode, "instruction", None)
        if isinstance(instruction, str):
            return instruction
        if instruction is not None:
            for attr in ("instruction_text", "text"):
                value = getattr(instruction, attr, None)
                if value:
                    return str(value)
        return str(getattr(episode, "instruction_text", ""))

    def _extract_agent_pose(self, env: Any) -> Optional[Dict[str, Any]]:
        sim = getattr(env, "sim", None)
        if sim is None or not hasattr(sim, "get_agent_state"):
            return None
        try:
            agent_state = sim.get_agent_state()
        except Exception:
            return None
        return {
            "position": self._to_plain(getattr(agent_state, "position", None)),
            "rotation": self._to_plain(getattr(agent_state, "rotation", None)),
        }

    def _online_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        online: Dict[str, Any] = {}
        collisions = metrics.get("collisions")
        if isinstance(collisions, dict) and "is_collision" in collisions:
            online["collision"] = bool(collisions["is_collision"])
        elif "collision" in metrics:
            online["collision"] = bool(metrics["collision"])
        return online

    def _diagnostics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        for key, value in metrics.items():
            if key in ORACLE_METRIC_KEYS:
                diagnostics[key] = value
        diagnostics["raw_metrics"] = dict(metrics)
        return diagnostics

    def _to_plain(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value
