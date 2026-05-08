from collections import deque
from math import sqrt
from typing import Any, Deque, Dict, Iterable, List, Optional


ORACLE_METRIC_KEYS = {
    "distance_to_goal",
    "success",
    "spl",
    "SPL",
    "oracle_path",
    "oracle_shortest_path",
    "oracle_shortest_path_action",
    "oracle_action",
}


class WorkingMemory:
    def __init__(
        self,
        max_recent_frames: int = 8,
        max_recent_actions: int = 20,
        max_recent_poses: int = 20,
    ) -> None:
        self.frames: Deque[Any] = deque(maxlen=max_recent_frames)
        self.actions: Deque[str] = deque(maxlen=max_recent_actions)
        self.poses: Deque[List[float]] = deque(maxlen=max_recent_poses)
        self.online_metric_history: Deque[Dict[str, Any]] = deque(maxlen=max_recent_actions)
        self.diagnostic_metric_history: Deque[Dict[str, Any]] = deque(maxlen=max_recent_actions)

    def append_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    def append_action(self, action: str) -> None:
        self.actions.append(action)

    def append_pose(self, pose: Iterable[float]) -> None:
        self.poses.append([float(value) for value in pose])

    def append_online_metrics(self, metrics: Dict[str, Any]) -> None:
        safe_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in ORACLE_METRIC_KEYS
        }
        self.online_metric_history.append(safe_metrics)

    def append_diagnostics(self, metrics: Dict[str, Any]) -> None:
        self.diagnostic_metric_history.append(dict(metrics))

    def get_recent_frames(self, n: Optional[int] = None) -> List[Any]:
        frames = list(self.frames)
        if n is None:
            return frames
        return frames[-n:]

    def has_action_oscillation(self, window: int = 5) -> bool:
        recent = list(self.actions)[-window:]
        if len(recent) < window:
            return False
        if len(set(recent)) == 1 and recent[0] in {"TURN_LEFT", "TURN_RIGHT"}:
            return True
        turn_pair = {"TURN_LEFT", "TURN_RIGHT"}
        return all(action in turn_pair for action in recent) and len(set(recent)) == 2

    def has_low_displacement(self, threshold: float = 0.05, window: int = 3) -> bool:
        recent = list(self.poses)[-window:]
        if len(recent) < 2:
            return False
        start = recent[0]
        end = recent[-1]
        dims = min(len(start), len(end))
        displacement = sqrt(sum((end[idx] - start[idx]) ** 2 for idx in range(dims)))
        return displacement < threshold

    def has_repeated_observation(self, window: int = 3) -> bool:
        recent = self.get_recent_frames(window)
        if len(recent) < window:
            return False
        return all(frame == recent[0] for frame in recent)

    def should_promote_keyframe(self, step_id: int, interval: int = 10) -> bool:
        return step_id == 0 or (interval > 0 and step_id % interval == 0)

    def decision_metrics(self) -> Dict[str, Any]:
        if not self.online_metric_history:
            return {}
        merged: Dict[str, Any] = {}
        for metrics in self.online_metric_history:
            merged.update(
                {
                    key: value
                    for key, value in metrics.items()
                    if key not in ORACLE_METRIC_KEYS
                }
            )
        return merged

    def diagnostic_metrics(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for metrics in self.diagnostic_metric_history:
            merged.update(metrics)
        return merged
