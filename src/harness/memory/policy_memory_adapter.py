import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_POLICY_CONTEXT_CHARS = 240
MAX_EVIDENCE_TEXT_CHARS = 500

STOP_SEMANTIC_TERMS: Tuple[str, ...] = (
    "stop",
    "stopping",
    "wait",
    "arrived",
    "arrival",
    "reached",
    "successfully",
    "destination",
    "task complete",
    "goal reached",
    "final location",
    "turn left",
    "turn right",
    "go back",
    "move forward",
)

RAW_REASON_LABELS: Tuple[str, ...] = (
    "last qwen reason",
    "therefore",
    "i think",
    "we should",
)

ACTION_TEXTS = {"STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"}

ORACLE_KEYS = {
    "distance_to_goal",
    "success",
    "spl",
    "oracle_success",
    "os",
    "ndtw",
    "sdtw",
    "ne",
}

DIRECTION_WORDS: Tuple[str, ...] = (
    "left",
    "right",
    "straight",
    "forward",
    "upstairs",
    "downstairs",
)

LANDMARK_NOUNS: Tuple[str, ...] = (
    "arched doorway",
    "doorway",
    "door",
    "hallway",
    "corridor",
    "archway",
    "stairs",
    "staircase",
    "kitchen",
    "room",
    "bedroom",
    "bathroom",
    "lobby",
    "entrance",
    "elevator",
    "table",
    "chair",
    "couch",
    "sofa",
)


@dataclass(frozen=True)
class PolicyMemoryContext:
    memory_context_text: str = ""
    active_subgoal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.memory_context_text:
            data["memory_context_text"] = self.memory_context_text
        if self.active_subgoal:
            data["active_subgoal"] = self.active_subgoal
        return data


@dataclass(frozen=True)
class ControlMemoryContext:
    possible_goal_region: bool = False
    stop_confidence: str = ""
    requires_stop_verification: bool = False
    memory_freshness_steps: Optional[int] = None
    filtered_stop_terms: Tuple[str, ...] = field(default_factory=tuple)
    policy_cue_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "possible_goal_region": self.possible_goal_region,
            "requires_stop_verification": self.requires_stop_verification,
            "policy_cue_used": self.policy_cue_used,
        }
        if self.stop_confidence:
            data["stop_confidence"] = self.stop_confidence
        if self.memory_freshness_steps is not None:
            data["memory_freshness_steps"] = self.memory_freshness_steps
        if self.filtered_stop_terms:
            data["filtered_stop_terms"] = list(self.filtered_stop_terms)
        return data


@dataclass(frozen=True)
class EvidenceMemoryContext:
    raw_visual_summary: str = ""
    raw_suggested_subgoal: str = ""
    raw_qwen_reason: str = ""
    source_image_paths: Tuple[str, ...] = field(default_factory=tuple)
    runtime_context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.raw_visual_summary:
            data["raw_visual_summary"] = self.raw_visual_summary[:MAX_EVIDENCE_TEXT_CHARS]
        if self.raw_suggested_subgoal:
            data["raw_suggested_subgoal"] = self.raw_suggested_subgoal[:MAX_EVIDENCE_TEXT_CHARS]
        if self.raw_qwen_reason:
            data["raw_qwen_reason"] = self.raw_qwen_reason[:MAX_EVIDENCE_TEXT_CHARS]
        if self.source_image_paths:
            data["source_image_paths"] = list(self.source_image_paths[:5])
        if self.runtime_context:
            data["runtime_context"] = {
                str(key): value
                for key, value in self.runtime_context.items()
                if key not in ORACLE_KEYS and _json_scalar_or_small(value)
            }
        return data


@dataclass(frozen=True)
class MemoryGateResult:
    mode: str
    policy_context: PolicyMemoryContext = field(default_factory=PolicyMemoryContext)
    control_context: ControlMemoryContext = field(default_factory=ControlMemoryContext)
    evidence_context: EvidenceMemoryContext = field(default_factory=EvidenceMemoryContext)
    policy_context_used: bool = False
    raw_reason_included: bool = False
    stop_semantics_filtered: bool = False
    filtered_terms: Tuple[str, ...] = field(default_factory=tuple)

    def to_audit(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "policy_context_used": self.policy_context_used,
            "raw_reason_included": self.raw_reason_included,
            "stop_semantics_filtered": self.stop_semantics_filtered,
            "filtered_terms": list(self.filtered_terms),
            "policy_context": self.policy_context.to_dict(),
            "control_context": self.control_context.to_dict(),
            "evidence_context": self.evidence_context.to_dict(),
        }


def contains_stop_semantics(text: str) -> bool:
    return bool(_matched_stop_terms(text))


def filter_stop_semantics(text: str) -> tuple[str, list[str]]:
    kept: List[str] = []
    terms: List[str] = []
    for sentence in _sentences(text):
        matched = _matched_stop_terms(sentence)
        if matched:
            terms.extend(matched)
            continue
        kept.append(sentence.strip())
    return " ".join(item for item in kept if item).strip(), _dedupe(terms)


def distill_memory(
    raw_memory: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    mode: str,
    should_filter_stop_semantics: bool,
) -> MemoryGateResult:
    normalized_mode = _normalize_mode(mode)
    evidence = _evidence_context(raw_memory, runtime_context)
    if normalized_mode == "off":
        return _result(mode="off", evidence=evidence)
    if normalized_mode == "raw":
        return _raw_result(raw_memory, runtime_context, evidence)
    if normalized_mode == "no_reason":
        return _no_reason_result(
            raw_memory,
            runtime_context,
            evidence,
            should_filter_stop_semantics=should_filter_stop_semantics,
        )
    return _safe_cue_result(
        raw_memory,
        runtime_context,
        evidence,
        should_filter_stop_semantics=should_filter_stop_semantics,
    )


def to_navigation_arguments(result: MemoryGateResult) -> Dict[str, Any]:
    return result.policy_context.to_dict()


def _raw_result(
    raw_memory: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    evidence: EvidenceMemoryContext,
) -> MemoryGateResult:
    summary = _text(raw_memory.get("last_visual_summary"))
    subgoal = _text(raw_memory.get("last_suggested_subgoal"))
    reason = _text(raw_memory.get("last_qwen_reason"))
    lines: List[str] = []
    if summary:
        lines.append(f"Cached visual memory: {summary}")
    if subgoal:
        lines.append(f"Suggested subgoal: {subgoal}")
    if reason:
        lines.append(f"Last Qwen reason: {reason}")
    context_note = _text(runtime_context.get("memory_context_text"))
    if context_note:
        lines.append(f"Retrieved memory: {context_note}")
    policy_text = _bounded("\n".join(lines))
    return _result(
        mode="raw",
        policy=PolicyMemoryContext(
            memory_context_text=policy_text,
            active_subgoal=subgoal,
        ),
        evidence=evidence,
        raw_reason_included=bool(reason),
    )


def _no_reason_result(
    raw_memory: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    evidence: EvidenceMemoryContext,
    *,
    should_filter_stop_semantics: bool,
) -> MemoryGateResult:
    summary = _text(raw_memory.get("last_visual_summary"))
    subgoal = _text(raw_memory.get("last_suggested_subgoal"))
    lines: List[str] = []
    if summary:
        lines.append(f"Cached visual memory: {summary}")
    if subgoal:
        lines.append(f"Suggested subgoal: {subgoal}")
    context_note = _text(runtime_context.get("memory_context_text"))
    if context_note:
        lines.append(f"Retrieved memory: {context_note}")
    policy_text = "\n".join(lines)
    filtered_terms: List[str] = []
    if should_filter_stop_semantics:
        policy_text, filtered_terms = filter_stop_semantics(policy_text)
        if contains_stop_semantics(subgoal):
            subgoal = ""
    return _result(
        mode="no_reason",
        policy=PolicyMemoryContext(
            memory_context_text=_bounded(policy_text),
            active_subgoal=subgoal,
        ),
        evidence=evidence,
        filtered_terms=filtered_terms,
    )


def _safe_cue_result(
    raw_memory: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
    evidence: EvidenceMemoryContext,
    *,
    should_filter_stop_semantics: bool,
) -> MemoryGateResult:
    filtered_terms: List[str] = []
    clean_sentences: List[str] = []
    for source_text in (
        _text(raw_memory.get("last_visual_summary")),
        _text(raw_memory.get("last_suggested_subgoal")),
        _text(runtime_context.get("memory_context_text")),
    ):
        for sentence in _sentences(source_text):
            if _contains_raw_reason_label(sentence):
                continue
            matched = _matched_stop_terms(sentence)
            if matched and should_filter_stop_semantics:
                filtered_terms.extend(matched)
                continue
            clean_sentences.append(sentence)

    nav_cue = ""
    active_subgoal = ""
    direction_hint = ""
    landmarks: List[str] = []
    for sentence in clean_sentences:
        if not nav_cue:
            nav_cue = _navigation_cue(sentence)
            if nav_cue:
                active_subgoal = nav_cue.replace("Navigation cue: ", "", 1)
        if not direction_hint:
            direction_hint = _direction_hint(sentence)
        landmarks.extend(_landmarks(sentence))

    landmarks = _dedupe(landmarks)[:3]
    lines: List[str] = []
    if nav_cue:
        lines.append(nav_cue)
    if landmarks:
        lines.append("Visible landmarks: " + ", ".join(landmarks))
    if direction_hint:
        lines.append(f"Direction hint: {direction_hint}")

    policy_text = _bounded("\n".join(lines))
    if not policy_text:
        active_subgoal = ""
    if active_subgoal and not (_landmarks(active_subgoal) or _direction_hint(active_subgoal)):
        active_subgoal = ""
    return _result(
        mode="safe_cue",
        policy=PolicyMemoryContext(
            memory_context_text=policy_text,
            active_subgoal=active_subgoal,
        ),
        evidence=evidence,
        filtered_terms=filtered_terms,
    )


def _result(
    *,
    mode: str,
    policy: Optional[PolicyMemoryContext] = None,
    evidence: EvidenceMemoryContext,
    filtered_terms: Iterable[str] = (),
    raw_reason_included: bool = False,
) -> MemoryGateResult:
    policy = policy or PolicyMemoryContext()
    terms = tuple(_dedupe(filtered_terms))
    policy_used = bool(policy.memory_context_text or policy.active_subgoal)
    possible_goal = bool(terms)
    control = ControlMemoryContext(
        possible_goal_region=possible_goal,
        stop_confidence="medium" if possible_goal else "",
        requires_stop_verification=possible_goal or policy_used,
        filtered_stop_terms=terms,
        policy_cue_used=policy_used,
    )
    return MemoryGateResult(
        mode=mode,
        policy_context=policy,
        control_context=control,
        evidence_context=evidence,
        policy_context_used=policy_used,
        raw_reason_included=raw_reason_included,
        stop_semantics_filtered=bool(terms and mode != "raw"),
        filtered_terms=terms,
    )


def _evidence_context(
    raw_memory: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
) -> EvidenceMemoryContext:
    paths = raw_memory.get("source_image_paths") or []
    if not isinstance(paths, list):
        paths = []
    return EvidenceMemoryContext(
        raw_visual_summary=_text(raw_memory.get("last_visual_summary")),
        raw_suggested_subgoal=_text(raw_memory.get("last_suggested_subgoal")),
        raw_qwen_reason=_text(raw_memory.get("last_qwen_reason")),
        source_image_paths=tuple(str(path) for path in paths if path),
        runtime_context={
            str(key): value
            for key, value in runtime_context.items()
            if key not in ORACLE_KEYS and _json_scalar_or_small(value)
        },
    )


def _navigation_cue(sentence: str) -> str:
    text = _single_space(sentence)
    lower = text.lower()
    patterns = [
        r"\bcontinue\s+toward\s+(.+)$",
        r"\bproceed\s+(?:through|past|along)\s+(.+)$",
        r"\b(?:toward|towards)\s+(.+)$",
        r"\b(?:through|past|along)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        landmark = _clean_landmark(match.group(1))
        if landmark and (_landmarks(landmark) or any(word in landmark for word in LANDMARK_NOUNS)):
            if pattern.startswith(r"\bproceed"):
                verb = re.search(r"\b(proceed\s+(?:through|past|along))\s+", lower)
                prefix = verb.group(1) if verb else "proceed toward"
                return f"Navigation cue: {prefix} {landmark}"
            return f"Navigation cue: continue toward {landmark}"
    match = re.search(r"\bkeep\s+(.+?)\s+on\s+the\s+(left|right)\b", lower)
    if match:
        landmark = _clean_landmark(match.group(1))
        if landmark:
            return f"Navigation cue: keep {landmark} on the {match.group(2)}"
    return ""


def _direction_hint(sentence: str) -> str:
    lower = sentence.lower()
    for word in DIRECTION_WORDS:
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", lower):
            return word
    return ""


def _landmarks(sentence: str) -> List[str]:
    lower = sentence.lower()
    found: List[str] = []
    for noun in LANDMARK_NOUNS:
        if re.search(rf"(?<!\w){re.escape(noun)}(?!\w)", lower):
            found.append(noun)
    return _dedupe(found)


def _clean_landmark(text: str) -> str:
    cleaned = _single_space(text).lower()
    cleaned = re.split(r"\b(?:and|then|while|because|with)\b", cleaned, maxsplit=1)[0]
    cleaned = re.sub(r"[^a-z0-9 _-]", "", cleaned).strip()
    words = cleaned.split()
    if len(words) > 5:
        cleaned = " ".join(words[:5])
    return cleaned


def _joined_policy_sources(
    raw_memory: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
) -> str:
    return "\n".join(
        text
        for text in (
            _text(raw_memory.get("last_visual_summary")),
            _text(raw_memory.get("last_suggested_subgoal")),
            _text(raw_memory.get("last_qwen_reason")),
            _text(runtime_context.get("memory_context_text")),
        )
        if text
    )


def _matched_stop_terms(text: str) -> List[str]:
    lower = text.lower()
    terms: List[str] = []
    for term in STOP_SEMANTIC_TERMS:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        if re.search(pattern, lower):
            terms.append(term)
    return _dedupe(terms)


def _contains_raw_reason_label(text: str) -> bool:
    lower = text.lower()
    return any(label in lower for label in RAW_REASON_LABELS)


def _sentences(text: str) -> List[str]:
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _bounded(text: str, limit: int = DEFAULT_POLICY_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _normalize_mode(mode: str) -> str:
    candidate = str(mode or "raw").strip().lower()
    if candidate in {"raw", "no_reason", "safe_cue", "off"}:
        return candidate
    return "raw"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _single_space(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _json_scalar_or_small(value: Any) -> bool:
    if value is None or isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, str):
        return len(value) <= MAX_EVIDENCE_TEXT_CHARS
    if isinstance(value, list):
        return len(value) <= 5 and all(isinstance(item, (str, int, float, bool)) for item in value)
    return False
