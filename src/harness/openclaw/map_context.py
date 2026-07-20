from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


MAP_ASSIST_OFF = "off"
MAP_ASSIST_FLOORPLAN = "floorplan_map_assisted"
MAP_IMAGE_SOURCE = "habitat_pathfinder_navmesh"
POSE_SOURCE = "sim_agent_state_odometry"
INPUT_REGIME = "rgb_plus_privileged_floorplan_pose"
COLLISION_PROGRESS_THRESHOLD_M = 0.05


PolicyMapRenderer = Callable[..., Any]


def map_safety_contract() -> Dict[str, bool]:
    return {
        "contains_goal": False,
        "contains_reference_path": False,
        "contains_shortest_path": False,
        "contains_oracle_actions": False,
        "contains_goal_distance": False,
        "contains_success_labels": False,
        "contains_room_labels": False,
        "uses_decorated_top_down_map": False,
    }


class FloorplanMapContextProvider:
    """Builds goal/path-free floorplan context for Qwen-direct visual input."""

    def __init__(
        self,
        output_root: Union[Path, str],
        mode: str = MAP_ASSIST_OFF,
        frame_interval_steps: int = 5,
        collision_overlay_enabled: bool = False,
        map_resolution: int = 512,
        meters_per_pixel: Optional[float] = None,
        renderer: Optional[PolicyMapRenderer] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.mode = str(mode or MAP_ASSIST_OFF)
        self.frame_interval_steps = max(1, int(frame_interval_steps or 5))
        self.collision_overlay_enabled = bool(collision_overlay_enabled)
        self.map_resolution = int(map_resolution)
        self.meters_per_pixel = meters_per_pixel
        self._renderer = renderer
        self._episode_key: Tuple[str, str] = ("", "")
        self._visited_positions: List[Tuple[float, float, float]] = []
        self._collision_positions: List[Tuple[float, float, float]] = []
        self._last_position: Optional[Tuple[float, float, float]] = None
        self._last_map_artifact: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return self.mode == MAP_ASSIST_FLOORPLAN

    def reset_episode(self, scene_id: str = "", episode_id: str = "") -> None:
        self._episode_key = (str(scene_id or ""), str(episode_id or ""))
        self._visited_positions = []
        self._collision_positions = []
        self._last_position = None
        self._last_map_artifact = None

    def build_context(
        self,
        *,
        env: Any,
        state: Any,
        step_id: int,
        scene_id: str,
        episode_id: str,
        local_focus: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        scene_id = str(scene_id or "")
        episode_id = str(episode_id or "")
        if self._episode_key != (scene_id, episode_id):
            self.reset_episode(scene_id, episode_id)

        pose = self._pose_from_state(state)
        position = self._position_from_pose(pose)
        if position is not None:
            previous_position = self._last_position
            self._append_position(self._visited_positions, position)
            if (
                self.collision_overlay_enabled
                and self._is_no_progress_forward_collision(
                    state,
                    previous_position,
                    position,
                )
            ):
                self._append_position(self._collision_positions, position)
            self._last_position = position

        local_focus = bool(local_focus)
        interval_due = int(step_id) % self.frame_interval_steps == 0
        cached_scope = (
            str(self._last_map_artifact.get("map_view_scope") or "")
            if isinstance(self._last_map_artifact, dict)
            else ""
        )
        recovery_exit_refresh = not local_focus and cached_scope == "local_recovery"
        due = interval_due or local_focus or recovery_exit_refresh
        context: Dict[str, Any] = {
            "mode": MAP_ASSIST_FLOORPLAN,
            "input_regime": INPUT_REGIME,
            "map_frame_interval_steps": self.frame_interval_steps,
            "map_step_id": int(step_id),
            "map_frame_due": due,
            "map_interval_due": interval_due,
            "map_view_scope": "local_recovery" if local_focus else "global",
            "map_refresh_reason": (
                "turn_loop_recovery"
                if local_focus
                else (
                    "recovery_exit_global"
                    if recovery_exit_refresh
                    else ("interval" if interval_due else "cached_interval")
                )
            ),
            "map_available": False,
            "map_source": MAP_IMAGE_SOURCE,
            "pose_source": POSE_SOURCE,
            "map_safety": map_safety_contract(),
            "visited_trail_count": len(self._visited_positions),
            "collision_overlay_enabled": self.collision_overlay_enabled,
            "collision_point_count": len(self._collision_positions)
            if self.collision_overlay_enabled
            else 0,
        }
        if not due:
            self._attach_cached_map_artifact(context, int(step_id))
            return context

        try:
            rendered = self._render_policy_map(
                env=env,
                state=state,
                local_focus=local_focus,
            )
        except Exception as exc:
            context["map_generation_error"] = exc.__class__.__name__
            self._attach_cached_map_artifact(context, int(step_id))
            return context
        if rendered is None:
            context["map_generation_error"] = "map_unavailable"
            self._attach_cached_map_artifact(context, int(step_id))
            return context

        path = self._map_image_path(scene_id, episode_id, int(step_id))
        try:
            self._save_rendered_map(rendered, path)
        except Exception as exc:
            context["map_generation_error"] = exc.__class__.__name__
            self._attach_cached_map_artifact(context, int(step_id))
            return context

        context["map_available"] = True
        context["map_image_label"] = "map_view"
        context["internal_only"] = {
            "map_image_path": str(path),
            "map_image_hash": self._sha256_file(path),
        }
        self._last_map_artifact = {
            "map_step_id": int(step_id),
            "map_view_scope": context["map_view_scope"],
            "internal_only": dict(context["internal_only"]),
        }
        return context

    def _attach_cached_map_artifact(
        self,
        context: Dict[str, Any],
        step_id: int,
    ) -> None:
        cached = self._last_map_artifact
        if not isinstance(cached, dict):
            context["cached_map_available"] = False
            return
        cached_step = int(cached.get("map_step_id") or 0)
        context["cached_map_available"] = True
        context["cached_map_step_id"] = cached_step
        context["map_age_steps"] = int(step_id) - cached_step
        context["map_view_scope"] = str(cached.get("map_view_scope") or "global")
        internal = cached.get("internal_only")
        if isinstance(internal, dict):
            context["internal_only"] = dict(internal)

    def _render_policy_map(
        self,
        *,
        env: Any,
        state: Any,
        local_focus: bool = False,
    ) -> Any:
        if self._renderer is not None:
            return self._renderer(
                env=env,
                state=state,
                visited_positions=list(self._visited_positions),
                collision_positions=list(self._collision_positions),
            )
        return self._default_render_policy_map(
            env=env,
            state=state,
            local_focus=local_focus,
        )

    def _default_render_policy_map(
        self,
        *,
        env: Any,
        state: Any,
        local_focus: bool = False,
    ) -> Any:
        sim = getattr(env, "sim", None)
        if sim is None:
            return None
        try:
            from habitat.utils.visualizations import maps as habitat_maps
            from habitat_extensions import maps as map_utils
            import numpy as np
            from PIL import Image, ImageDraw
        except Exception:
            return None

        try:
            top_down = map_utils.get_top_down_map(
                sim,
                map_resolution=self.map_resolution,
                meters_per_pixel=self.meters_per_pixel,
            )
            image_array = map_utils.colorize_top_down_map(top_down).copy()
        except Exception:
            return None

        image = Image.fromarray(np.asarray(image_array, dtype=np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        trail_pixels = [
            self._position_to_pixel(position, image_array.shape[:2], sim, habitat_maps)
            for position in self._visited_positions
        ]
        trail_pixels = [pixel for pixel in trail_pixels if pixel is not None]
        if len(trail_pixels) >= 2:
            draw.line(trail_pixels, fill=(0, 170, 255), width=2)
        for pixel in trail_pixels[-12:]:
            self._draw_circle(draw, pixel, radius=2, fill=(0, 170, 255))

        current_position = self._position_from_pose(self._pose_from_state(state))
        current_pixel = (
            self._position_to_pixel(current_position, image_array.shape[:2], sim, habitat_maps)
            if current_position is not None
            else None
        )
        if current_pixel is not None:
            self._draw_circle(draw, current_pixel, radius=5, fill=(0, 60, 220))
            yaw = self._yaw_from_rotation(self._rotation_from_state(state))
            if yaw is not None:
                self._draw_heading(draw, current_pixel, yaw)

        if self.collision_overlay_enabled:
            for collision_position in self._collision_positions[-24:]:
                pixel = self._position_to_pixel(
                    collision_position,
                    image_array.shape[:2],
                    sim,
                    habitat_maps,
                )
                if pixel is not None:
                    self._draw_cross(draw, pixel, size=4, fill=(220, 90, 0))

        if local_focus and current_pixel is not None:
            return self._local_focus_image(image, current_pixel)
        return image

    def _local_focus_image(self, image: Any, center: Tuple[int, int]) -> Any:
        width, height = image.size
        side = min(width, height, max(128, (min(width, height) * 2) // 5))
        left = max(0, min(int(center[0]) - side // 2, width - side))
        top = max(0, min(int(center[1]) - side // 2, height - side))
        crop = image.crop((left, top, left + side, top + side))
        resampling = getattr(image, "Resampling", None)
        nearest = resampling.NEAREST if resampling is not None else 0
        return crop.resize((self.map_resolution, self.map_resolution), resample=nearest)

    def _map_image_path(self, scene_id: str, episode_id: str, step_id: int) -> Path:
        return (
            self.output_root
            / "openclaw_map_frames"
            / self._safe_path_part(scene_id or "scene")
            / self._safe_path_part(episode_id or "episode")
            / f"step_{step_id:06d}.png"
        )

    def _save_rendered_map(self, rendered: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(rendered, "save"):
            rendered.save(path)
            return
        if isinstance(rendered, bytes):
            path.write_bytes(rendered)
            return
        try:
            import numpy as np
            from PIL import Image
        except Exception:
            path.write_text(str(rendered), encoding="utf-8")
            return
        array = np.asarray(rendered)
        if array.ndim == 2:
            Image.fromarray(array.astype("uint8"), mode="L").save(path)
            return
        Image.fromarray(array.astype("uint8"), mode="RGB").save(path)

    def _pose_from_state(self, state: Any) -> Dict[str, Any]:
        for candidate in (
            getattr(state, "diagnostic_pose", None),
            getattr(state, "pose", None),
        ):
            if isinstance(candidate, dict):
                return dict(candidate)
        diagnostics = getattr(state, "diagnostics", None)
        if isinstance(diagnostics, dict):
            position = diagnostics.get("sim_position")
            rotation = diagnostics.get("sim_rotation")
            if position is not None or rotation is not None:
                return {"position": position, "rotation": rotation}
        return {}

    def _position_from_pose(self, pose: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        position = pose.get("position") if isinstance(pose, dict) else None
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)):
            return None
        if len(position) < 3:
            return None
        try:
            return (float(position[0]), float(position[1]), float(position[2]))
        except (TypeError, ValueError):
            return None

    def _rotation_from_state(self, state: Any) -> Any:
        pose = self._pose_from_state(state)
        return pose.get("rotation") if isinstance(pose, dict) else None

    def _position_to_pixel(
        self,
        position: Tuple[float, float, float],
        shape: Tuple[int, int],
        sim: Any,
        habitat_maps: Any,
    ) -> Optional[Tuple[int, int]]:
        try:
            row, col = habitat_maps.to_grid(position[2], position[0], shape, sim)
        except Exception:
            return None
        if row < 0 or col < 0 or row >= shape[0] or col >= shape[1]:
            return None
        return int(col), int(row)

    def _append_position(
        self,
        positions: List[Tuple[float, float, float]],
        position: Tuple[float, float, float],
    ) -> None:
        if positions:
            last = positions[-1]
            if math.dist((last[0], last[2]), (position[0], position[2])) < 0.02:
                return
        positions.append(position)

    def _state_collision(self, state: Any) -> bool:
        for container_name in ("online_metrics", "diagnostics"):
            container = getattr(state, container_name, None)
            if not isinstance(container, dict):
                continue
            if container.get("collision"):
                return True
            collisions = container.get("collisions")
            if isinstance(collisions, dict) and collisions.get("is_collision"):
                return True
        return False

    def _is_no_progress_forward_collision(
        self,
        state: Any,
        previous_position: Optional[Tuple[float, float, float]],
        position: Tuple[float, float, float],
    ) -> bool:
        if previous_position is None or not self._state_collision(state):
            return False
        last_action = str(getattr(state, "last_action", "") or "")
        normalized_action = last_action.strip().upper().replace("-", "_").replace(" ", "_")
        if normalized_action not in {"MOVE_FORWARD", "FORWARD"}:
            return False
        planar_delta = math.dist(
            (previous_position[0], previous_position[2]),
            (position[0], position[2]),
        )
        return planar_delta < COLLISION_PROGRESS_THRESHOLD_M

    def _yaw_from_rotation(self, rotation: Any) -> Optional[float]:
        if not isinstance(rotation, Sequence) or isinstance(rotation, (str, bytes)):
            return None
        if len(rotation) < 4:
            return None
        try:
            w, x, y, z = (float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
        except (TypeError, ValueError):
            return None
        return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))

    def _draw_heading(self, draw: Any, pixel: Tuple[int, int], yaw: float) -> None:
        length = 16
        x, y = pixel
        end = (
            int(round(x + math.sin(yaw) * length)),
            int(round(y - math.cos(yaw) * length)),
        )
        draw.line([pixel, end], fill=(0, 40, 180), width=3)

    def _draw_circle(self, draw: Any, pixel: Tuple[int, int], radius: int, fill: Tuple[int, int, int]) -> None:
        x, y = pixel
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

    def _draw_cross(self, draw: Any, pixel: Tuple[int, int], size: int, fill: Tuple[int, int, int]) -> None:
        x, y = pixel
        draw.line((x - size, y - size, x + size, y + size), fill=fill, width=2)
        draw.line((x - size, y + size, x + size, y - size), fill=fill, width=2)

    def _sha256_file(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _safe_path_part(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
