from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


HARNESS_INTENTS = {
    "act",
    "recall_memory",
    "write_memory",
    "verify_progress",
    "replan",
    "stop",
}


@dataclass
class SkillResult:
    ok: bool
    result_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    error: Optional[str] = None

    @classmethod
    def ok_result(
        cls,
        result_type: str,
        payload: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
    ) -> "SkillResult":
        return cls(
            ok=True,
            result_type=result_type,
            payload=payload or {},
            confidence=confidence,
        )

    @classmethod
    def error_result(
        cls,
        error: str,
        result_type: str = "error",
        payload: Optional[Dict[str, Any]] = None,
    ) -> "SkillResult":
        return cls(
            ok=False,
            result_type=result_type,
            payload=payload or {},
            confidence=0.0,
            error=error,
        )


@dataclass
class HarnessDecision:
    intent: str
    skill_name: str
    reason: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.intent not in HARNESS_INTENTS:
            raise ValueError(f"Unsupported harness intent: {self.intent}")


@dataclass
class MemoryHit:
    memory_id: str
    memory_type: str
    name: str
    confidence: float
    target_pose: Optional[Dict[str, Any]] = None
    evidence_text: str = ""
    image_path: Optional[str] = None
    note: str = ""
    timestamp: Optional[float] = None
    memory_source: str = "episode-local"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgoalState:
    text: str
    status: str = "pending"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskState:
    episode_id: str = ""
    global_instruction: str = ""
    active_subgoal: Optional[SubgoalState] = None
    pending_subgoals: List[SubgoalState] = field(default_factory=list)
    completed_subgoals: List[SubgoalState] = field(default_factory=list)
    failed_subgoals: List[SubgoalState] = field(default_factory=list)
    failure_reason: Optional[str] = None
    recovery_attempts: List[str] = field(default_factory=list)
    success_criteria: str = ""


@dataclass
class MemoryRecallResult:
    hits: List[MemoryHit] = field(default_factory=list)
    query: str = ""
    backend: str = "fake"
    policy_context: Dict[str, Any] = field(default_factory=dict)
    control_context: Dict[str, Any] = field(default_factory=dict)
    executor_context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class VLNState:
    scene_id: str
    episode_id: str
    instruction: str
    step_id: int
    current_image: Any
    online_metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    pose: Optional[Any] = None
    diagnostic_pose: Optional[Any] = None
    last_action: Optional[str] = None
