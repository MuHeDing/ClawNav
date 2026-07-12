# OpenClaw Qwen Direct Policy Eval

This runbook is for `qwen_direct_policy`, where Qwen is the only action-producing
policy backend and OpenClaw owns memory, control gates, and traceability. It is
not a planner-action-override proof for the existing Janus-backed hybrid path.

## Launch

Start the local plan gateway in model mode:

```bash
POLICY_BACKEND=qwen_direct \
OPENCLAW_PLANNER_MODE=model \
OPENCLAW_MODEL_PROVIDER=qwen_api \
OPENCLAW_MODEL=qwen/qwen3.5-flash \
OPENCLAW_MODEL_FAST_MODE=qwen_text_only \
bash scripts/start_openclaw_cli_plan_gateway.sh
```

Run the one-episode smoke first:

```bash
POLICY_BACKEND=qwen_direct \
HARNESS_DEBUG_MAX_EPISODES=1 \
HARNESS_EPISODE_KEYS=2azQ1b91cZZ:11 \
OUTPUT_PATH=results/qwen_direct_policy_smoke \
bash scripts/evaluation_openclaw_gateway.sh
```

Do not start targeted 7/20 or shared-100 runs until the smoke trace proves the
direct-policy contract.

## Same-Episode Diagnostic

The first diagnostic episode is `2azQ1b91cZZ:11`. Run it before targeted 7/20
or shared-100. The diagnostic should force current visual evidence and attach
OpenClaw-retrieved history images before the current frame. The trace must show
the retrieved memory/keyframe image count, selected image count, provider image
count, retrieval source labels, and `current_image_last=true`.

Start the gateway for this diagnostic with fast-text disabled and enough image
budget for 8 retrieved memory/keyframe frames plus the current frame:

```bash
POLICY_BACKEND=qwen_direct \
OPENCLAW_PLANNER_MODE=model \
OPENCLAW_MODEL_PROVIDER=qwen_api \
OPENCLAW_MODEL=qwen/qwen3.5-flash \
OPENCLAW_MODEL_FAST_MODE=off \
OPENCLAW_MODEL_IMAGE_INTERVAL_STEPS=1 \
OPENCLAW_MODEL_MAX_IMAGES=9 \
bash scripts/start_openclaw_cli_plan_gateway.sh
```

Then run only the baseline failure episode:

```bash
POLICY_BACKEND=qwen_direct \
HARNESS_DEBUG_MAX_EPISODES=1 \
HARNESS_EPISODE_KEYS=2azQ1b91cZZ:11 \
OUTPUT_PATH=results/qwen_direct_policy_same_episode_visual_history \
bash scripts/evaluation_openclaw_gateway.sh
```

The diagnostic report must include action distribution, candidate action
distribution, final action source distribution, planner step modes, gate counts,
and Qwen failure count. Treat an all-forward trace as suspicious even when the
no-Janus proof passes; it means the direct policy still lacks usable turn/STOP
behavior or a gate intervention.

## Smoke Proof

Check `harness_traces/harness_trace_rank0.jsonl` and `large_eval_summary.json`
for these fields:

- `policy_backend=qwen_direct`
- `direct_policy=true`
- `janus_loaded=false`
- `navigation_policy_skill_called=false`
- `planned_tool=QwenDirectPolicy`
- `planner_authority=qwen`
- `qwen_candidate_requested=true`
- `qwen_model_called=true` or explicit `qwen_failure=true`
- `final_action_source` is `qwen`, `blocked_stop_gate`, `loop_gate`, `uncertainty_gate`, or `qwen_failure_stop`
- `qwen_direct_requery_triggered=true` appears when the STOP gate rejects a
  weak STOP and the runtime asks Qwen once more for a non-STOP corrective action
- if requery happens, inspect `qwen_direct_initial_candidate_action`,
  `blocked_action`, `stop_gate_decision`, and the final `candidate_action` to
  distinguish rejected STOP from the executed action

When `qwen_provider=qwen_api`, the trace must also show `qwen_api_called=true`
for model-call rows.

## Artifact Boundary

Default share-safe artifacts must keep provider payloads out of traces:

- `provider_payload_logged=false`
- raw provider prompts are not logged
- raw provider request bodies are not logged
- raw provider image payloads are not logged
- credentials are not logged
- direct Qwen prompt text omits `run_id` and raw image paths; image files are
  still attached through the provider call, and trace metadata may retain local
  paths for debugging

Local keyframe/current-frame image folders are debugging artifacts. They are not
part of a share-safe bundle unless exported through a separate redaction step.

## Fixed-Set Discipline

Freeze episode keys and baseline paths before running targeted or shared sets.
After a fixed set completes, compare against Janus:

```bash
PYTHONPATH=.:src python scripts/compare_qwen_direct_policy_results.py \
  --qwen-result results/qwen_direct_policy_shared100/result.json \
  --janus-result /path/to/janus_baseline/result.json \
  --qwen-trace results/qwen_direct_policy_shared100/harness_traces/harness_trace_rank0.jsonl \
  --output results/qwen_direct_policy_shared100/qwen_direct_vs_janus.json
```

The comparison must report aligned `(scene_id, episode_id)` counts plus
`both_success`, `qwen_only`, `janus_only`, and `both_fail`.
