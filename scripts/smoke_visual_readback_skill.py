#!/usr/bin/env python
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.config import HarnessConfig
from harness.env_adapters.habitat_vln_adapter import HabitatVLNAdapter
from harness.memory.memory_manager import MemoryManager
from harness.openclaw.executor import HabitatOpenClawExecutor
from harness.openclaw.planner import OpenClawPlanDecision
from harness.openclaw.runtime import OpenClawVLNRuntime
from harness.skill_registry import SkillRegistry
from harness.skills.memory_query import MemoryQuerySkill
from harness.skills.navigation_policy import NavigationPolicySkill
from harness.skills.visual_memory_read import (
    QwenDirectVisualReadbackAdapter,
    VisualMemoryReadSkill,
)
from harness.types import VLNState
from harness.visual_readback.memory_smoke import ImageBackedLocalMemoryClient


DEFAULT_MODEL = "qwen/qwen3.5-flash"
DEFAULT_MEMORY_ID = "memory-flash-default-1"
DEFAULT_SCENE_ID = "visual-readback-smoke-scene"
DEFAULT_EPISODE_ID = "visual-readback-smoke-episode"


class _FixedActionModel:
    def __init__(self, action_text: str = "TURN_LEFT") -> None:
        self.action_text = action_text

    def call_model(self, images: List[Any], task: str, step_id: int) -> List[str]:
        del images, task, step_id
        return [self.action_text]


class _RecallMemoryPlanner:
    def plan(self, state: VLNState, runtime_context: Dict[str, Any]) -> OpenClawPlanDecision:
        del runtime_context
        return OpenClawPlanDecision(
            intent="recall_memory",
            tool_name="MemoryQuerySkill",
            arguments={
                "text": state.instruction,
                "step_id": state.step_id,
                "reason": "decision_point",
                "n_results": 3,
                "allowed_scopes": ["episode"],
                "memory_namespace": _memory_namespace(state),
            },
            reason="decision_point",
            planner_backend="smoke",
        )


def run_visual_readback_skill_smoke(
    output_root: Path,
    *,
    current_image_path: str = "",
    memory_image_path: str = "",
    model: str = DEFAULT_MODEL,
    timeout_ms: int = 90000,
    adapter: Any = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    current_path, memory_path = _prepare_images(
        output_root,
        current_image_path=current_image_path,
        memory_image_path=memory_image_path,
    )
    skill = VisualMemoryReadSkill(
        adapter=adapter
        or QwenDirectVisualReadbackAdapter(model=model, timeout_ms=timeout_ms)
    )
    payload = _readback_payload(current_path, memory_path)
    state = _state()

    start = time.perf_counter()
    result = skill.run(state, payload)
    latency_ms = (time.perf_counter() - start) * 1000.0
    skill_payload = dict(result.payload)
    _write_json(output_root / "skill_payload.json", skill_payload)

    summary = _summary_from_payload(
        skill_payload,
        output_root=output_root,
        model=model,
        latency_ms=latency_ms,
        skill_ok=result.ok,
        adapter_class=skill.adapter.__class__.__name__,
    )
    _write_json(output_root / "summary.json", summary)
    return summary


def run_runtime_readback_smoke(
    output_root: Path,
    *,
    current_image_path: str = "",
    memory_image_path: str = "",
    model: str = DEFAULT_MODEL,
    timeout_ms: int = 90000,
    adapter: Any = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    current_path, memory_path = _prepare_images(
        output_root,
        current_image_path=current_image_path,
        memory_image_path=memory_image_path,
    )
    memory_client = _image_backed_client(memory_path)
    config = HarnessConfig(
        memory_backend="image_backed_local",
        visual_readback_mode="image_read_controller",
        visual_readback_stop_fallback_policy="log_only",
    )
    registry = SkillRegistry()
    registry.register(MemoryQuerySkill(MemoryManager(memory_client, config=config)))
    registry.register(NavigationPolicySkill(_FixedActionModel("TURN_LEFT")))
    registry.register(
        VisualMemoryReadSkill(
            adapter=adapter
            or QwenDirectVisualReadbackAdapter(model=model, timeout_ms=timeout_ms)
        )
    )
    runtime = OpenClawVLNRuntime(
        tool_registry=registry,
        planner=_RecallMemoryPlanner(),
        executor=HabitatOpenClawExecutor(HabitatVLNAdapter()),
        allow_planner_action_override=False,
        config=config,
    )
    result = runtime.step(
        _state(),
        payload={
            "current_image_path": str(current_path),
            "visual_readback_trigger_rule": "decision_point",
        },
    )
    metadata = dict(result.runtime_metadata)
    _write_json(output_root / "runtime_metadata.json", metadata)

    trace = metadata.get("visual_readback") if isinstance(metadata, dict) else {}
    trace = trace if isinstance(trace, dict) else {}
    summary = _runtime_summary(
        trace,
        metadata=metadata,
        output_root=output_root,
        model=model,
        runtime_ok=result.ok,
        action_text=result.action_text,
    )
    _write_json(output_root / "runtime_summary.json", summary)
    return summary


def default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"qwen35flash_default_visual_readback_smoke_{stamp}"


def _prepare_images(
    output_root: Path,
    *,
    current_image_path: str = "",
    memory_image_path: str = "",
) -> tuple[Path, Path]:
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    current_path = images_dir / "current.png"
    memory_path = images_dir / "memory.png"
    if current_image_path:
        _copy_image(Path(current_image_path), current_path)
    else:
        _write_fixture_image(current_path, kind="current")
    if memory_image_path:
        _copy_image(Path(memory_image_path), memory_path)
    else:
        _write_fixture_image(memory_path, kind="memory")
    return current_path, memory_path


def _copy_image(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"image path does not exist: {source}")
    if source.resolve() == target.resolve():
        return
    shutil.copyfile(source, target)


def _write_fixture_image(path: Path, *, kind: str) -> None:
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (320, 220), "white")
        draw = ImageDraw.Draw(image)
        if kind == "current":
            draw.rectangle((42, 40, 112, 180), fill=(220, 40, 40))
            draw.rectangle((184, 40, 254, 180), fill=(45, 90, 210))
            draw.text((54, 12), "CURRENT: red and blue pillars", fill=(0, 0, 0))
        else:
            draw.rectangle((48, 48, 270, 100), fill=(35, 160, 80))
            draw.polygon([(94, 180), (160, 112), (226, 180)], fill=(140, 82, 38))
            draw.text((62, 12), "MEMORY: green beam and brown triangle", fill=(0, 0, 0))
        image.save(path)
    except Exception:
        # 1x1 PNG fallback. The direct readback smoke still verifies image
        # attachment even when Pillow is unavailable.
        path.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
                "de0000000c4944415408d763f8ffff3f0005fe02fea73581e60000000049454e"
                "44ae426082"
            )
        )


def _image_backed_client(memory_path: Path) -> ImageBackedLocalMemoryClient:
    client = ImageBackedLocalMemoryClient()
    client.ingest_semantic(
        {
            "memory_id": DEFAULT_MEMORY_ID,
            "memory_type": "semantic_frame",
            "name": "visual readback smoke memory",
            "image_path": str(memory_path),
            "retrieval_text": "left opening landmark",
            "evidence_text": "stored memory frame with a green beam and brown triangle",
            "memory_scope": "episode",
            "memory_namespace": _memory_namespace(_state()),
            "memory_source": client.memory_source,
            "confidence": 0.77,
        }
    )
    return client


def _readback_payload(current_path: Path, memory_path: Path) -> Dict[str, Any]:
    return {
        "current_image_path": str(current_path),
        "memory_hits": [
            {
                "memory_id": DEFAULT_MEMORY_ID,
                "image_path": str(memory_path),
                "confidence": 0.77,
                "evidence_text": "stored memory frame with a green beam and brown triangle",
            }
        ],
        "candidate_action": "TURN_RIGHT",
        "trigger_rule": "decision_point",
    }


def _state() -> VLNState:
    return VLNState(
        scene_id=DEFAULT_SCENE_ID,
        episode_id=DEFAULT_EPISODE_ID,
        instruction="Use the remembered landmark to decide whether the current route conflicts.",
        step_id=4,
        current_image=None,
    )


def _memory_namespace(state: VLNState) -> str:
    return f"episode:{state.scene_id}:{state.episode_id}"


def _summary_from_payload(
    payload: Dict[str, Any],
    *,
    output_root: Path,
    model: str,
    latency_ms: float,
    skill_ok: bool,
    adapter_class: str,
) -> Dict[str, Any]:
    image_paths = list(payload.get("actually_read_image_paths") or [])
    read_status = str(payload.get("read_status") or "")
    error_type = str(payload.get("error_type") or "")
    return {
        "smoke_root": str(output_root),
        "model": model,
        "adapter_class": adapter_class,
        "skill_ok": skill_ok,
        "read_status": read_status,
        "parse_success_rate": 1.0 if read_status == "completed" else 0.0,
        "image_attach_rate": 1.0 if len(image_paths) >= 2 else 0.0,
        "timeout_rate": 1.0 if "timeout" in error_type.lower() else 0.0,
        "latency_ms": round(latency_ms, 3),
        "actually_read_image_paths": image_paths,
        "model_image_count": int(payload.get("model_image_count") or len(image_paths)),
        "matched_memory_ids": list(payload.get("matched_memory_ids") or []),
        "matched_memory_ids_from_attached_only": bool(
            payload.get("matched_memory_ids_from_attached_only")
        ),
        "verifier_labels": list(payload.get("verifier_labels") or []),
        "readback_confidence": float(payload.get("readback_confidence") or 0.0),
        "verifier_confidence": float(payload.get("verifier_confidence") or 0.0),
        "visual_grounding_status": str(payload.get("visual_grounding_status") or ""),
        "grounding_eval_protocol": payload.get("grounding_eval_protocol") or {},
        "visual_evidence_chars": len(str(payload.get("visual_evidence") or "")),
    }


def _runtime_summary(
    trace: Dict[str, Any],
    *,
    metadata: Dict[str, Any],
    output_root: Path,
    model: str,
    runtime_ok: bool,
    action_text: str,
) -> Dict[str, Any]:
    image_paths = list(trace.get("actually_read_image_paths") or [])
    tool_calls = metadata.get("tool_calls") if isinstance(metadata, dict) else []
    tool_calls = tool_calls if isinstance(tool_calls, list) else []
    return {
        "smoke_root": str(output_root),
        "model": model,
        "runtime_ok": runtime_ok,
        "action_text": action_text,
        "read_status": str(trace.get("read_status") or ""),
        "trigger_rule": str(trace.get("trigger_rule") or ""),
        "candidate_action": str(trace.get("candidate_action") or ""),
        "controller_decision": str(trace.get("controller_decision") or ""),
        "actually_read_image_paths": image_paths,
        "has_current_and_memory_images": len(image_paths) >= 2,
        "matched_memory_ids": list(trace.get("matched_memory_ids") or []),
        "memory_query_tool_call_count": sum(
            1 for call in tool_calls if call.get("tool_name") == "MemoryQuerySkill"
        ),
        "visual_read_tool_call_count": sum(
            1 for call in tool_calls if call.get("tool_name") == "VisualMemoryReadSkill"
        ),
        "policy_context_available": bool(trace.get("policy_context_available")),
        "actual_policy_payload_merge": bool(trace.get("actual_policy_payload_merge")),
        "used_by_policy": bool(trace.get("used_by_policy")),
        "readback_state_used_by_policy": bool(
            trace.get("readback_state_used_by_policy")
        ),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="")
    parser.add_argument("--current_image_path", default="")
    parser.add_argument("--memory_image_path", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout_ms", type=int, default=90000)
    parser.add_argument("--include_runtime", action="store_true")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root) if args.output_root else default_output_root()
    summary = run_visual_readback_skill_smoke(
        output_root=output_root,
        current_image_path=args.current_image_path,
        memory_image_path=args.memory_image_path,
        model=args.model,
        timeout_ms=args.timeout_ms,
    )
    result = {"summary": summary}
    if args.include_runtime:
        result["runtime_summary"] = run_runtime_readback_smoke(
            output_root=output_root,
            current_image_path=str(output_root / "images" / "current.png"),
            memory_image_path=str(output_root / "images" / "memory.png"),
            model=args.model,
            timeout_ms=args.timeout_ms,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
