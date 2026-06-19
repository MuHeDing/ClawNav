---
title: Event-Gated Keyframe Memory Design
date: 2026-06-18
status: design-draft
scope: ClawNav OpenClaw visual memory / visual readback
---

# Event-Gated Keyframe Memory Design

## Goal

Evaluate the current interval-only keyframe policy and, if Phase 0 shows missed route events, redundant attachments, or weak auditability, replace or augment it with an event-gated policy that saves visual memory at route-relevant moments:

```text
important action or state change
-> save a compact keyframe memory
-> retrieve a diverse small set for visual readback
-> keep trace evidence showing why the frame was saved and used
```

The goal is not to save fewer images for its own sake. The goal is to make `keyframes/` represent a useful route evidence chain: decision points, room transitions, uncertainty/conflict moments, and sparse coverage gaps.

## Current Behavior

The current path is implemented in `src/evaluation_harness.py`:

- `_runtime_payload(...)` calls `working_memory.should_promote_keyframe(step_id)`.
- When true, the current image is saved under `keyframes/<scene_id>/<episode_id>/step_*.png`.
- Otherwise, it is saved under `openclaw_current_frames/<scene_id>/<episode_id>/step_*.png`.
- The current documented default is step `0` and every `10` steps.

The current visual readback path then retrieves image-backed memory from the saved keyframes. In the inspected run:

- `memory_index.jsonl` had `110` rows.
- All memory image paths pointed to `keyframes/...`.
- `retrieved_image_paths` were all keyframes.
- `openclaw_current_frames/...` appeared only as the current view attached to readback.

This proves the historical-memory pool is already keyframe-based, but the keyframe-selection rule is still too mechanical.

## Problem

Interval keyframes create three issues:

1. They can miss important turns or STOP decisions if those events do not align with the interval.
2. They can save visually redundant frames during long straight motion.
3. They make readback retrieve nearby frames such as `step_000000`, `step_000010`, and `step_000020`, which can be less useful than a route-diverse set.

The slow visual-readback run also shows why keyframe quality matters. If readback is expensive, each attached memory image should carry evidence value.

## Phase 0: Baseline Premise Check

Before implementing event-gated saving, run the current `interval_10` behavior on the fixed qualitative manifest. This prevents the design from assuming that keyframe saving is the active bottleneck when a retrieval-only change may explain most of the readback waste.

Phase 0 setup artifact:

- Use a checked-in manifest path: `docs/plans/manifests/2026-06-18-event-gated-phase0-manifest.jsonl`.
- The implementation plan should add or validate it with:

```bash
python scripts/build_event_gated_phase0_manifest.py \
  --output docs/plans/manifests/2026-06-18-event-gated-phase0-manifest.jsonl
```

- Required manifest fields: `scene_id`, `episode_id`, `max_steps`, `seed` or deterministic episode ordering, predeclared route-event labels, accepted step tolerance, expected readback trigger key derivation, and source result/run provenance.
- At minimum, include one previously successful and one previously failed episode before running Phase 0. Additional episodes must be selected before seeing event-gated results.
- If the generator script is not available yet, Phase 0 implementation must add it before running comparisons; hand-edited manifests must still pass schema validation.

Phase 0 should report:

- missed `TURN_LEFT`, `TURN_RIGHT`, and `STOP` route-event opportunities where no interval keyframe exists within the accepted step tolerance
- adjacent-triplet rate for currently attached readback memories
- exact duplicate image-path rate for currently attached readback memories
- current readback memory-evidence-use counts using the identity-binding rules in this document
- readback trigger keys and unmatched trigger counts, so later arms can be compared on shared triggers

Decision checkpoint:

- If `interval_10_exact_dedupe` removes most redundant attachment failures and `interval_10` has low missed-event opportunity rate, keep event-gated saving limited to traceability work until a stronger need is shown.
- If `interval_10` misses decision-point opportunities, attaches redundant nearby memories on shared triggers, or lacks auditable memory-use evidence, proceed to the `event_gated_smoke_gate` checkpoint, then complete `event_gated_smoke_audit` before Phase A claims.

## Non-Goals

- Do not add a new Qwen call just to decide whether to save a keyframe.
- Do not use oracle metrics, distance-to-goal, success, SPL, future frames, or future path information for online keyframe decisions.
- Do not change JanusVLN model internals.
- Do not claim SR/SPL improvement from this design alone.
- Do not replace `openclaw_current_frames`; current-frame artifacts are still needed for readback and debugging.
- Do not implement cross-episode or long-term memory selection in this design. The first target remains episode-local.

## Proposed Approach

Use a hybrid event gate:

```text
save_keyframe =
  (direct_save_event or event_score >= threshold)
  and (cooldown allows it or cooldown_bypass_event)
  and (episode cap allows it or explicit debug override allows it)
```

The event score combines cheap online signals:

- candidate action type
- keyframe novelty versus recent saved keyframes
- route-stage or spatial-transition hints when available
- prior-step or v1/later post-readback controller uncertainty, with explicit phase metadata
- elapsed steps since last saved keyframe

The first implementation should use two smoke checkpoints. These are implementation checkpoints, not necessarily separate long-lived runtime modes:

`event_gated_smoke_gate` is the first runnable policy checkpoint:

- no new Qwen call
- no visual novelty computation
- no transition keyword detector
- direct-save events only: initial context, normalized `STOP`, `TURN_LEFT`, `TURN_RIGHT`, and configured coverage gap
- cooldown/cap enforcement
- core `keyframe_gate` trace metadata
- memory-write linkage sufficient to prove saved frames can be queried or deliberately marked as not queryable
- no requirement that `metadata.keyframe_gate` already round-trips through query results or `MemoryHit.metadata`

`event_gated_smoke_audit` then adds the audit layer needed before Phase A claims:

- readback query failure trace fields
- exact duplicate image-path cleanup for attached memories
- adjacent-triplet measurement
- evidence-use diagnostics
- summary counts for cap-hit, cap-blocked event classes, action normalization, retrieval text, write failure, and memory-id link status

The Phase A comparison arm called `event_gated_smoke` is considered valid only after both smoke checkpoints pass. This keeps the first runnable policy small while preserving a complete audit boundary before claims are made.

The full `event_gated_v1` can then add cheap visual novelty, transition hints, event-type diversity, and richer retrieval. This keeps the first smoke narrow enough to debug.

## Keyframe Event Classes

### 0. Initial Context Frames

Save the first available current image for each episode:

- `step_id == 0` is an `initial_context` direct-save event in both `event_gated_smoke` and `event_gated_v1`.
- It is subject to `keyframe_episode_cap`, but not normal cooldown.
- It initializes `steps_since_last_keyframe`.
- If the current image artifact is missing, do not synthesize a placeholder keyframe; trace `error_type=missing_current_image`.

### 1. Decision Point Frames

Save when the normalized clean policy candidate action is:

- `TURN_LEFT`
- `TURN_RIGHT`
- `STOP`

These are high-value frames because they correspond to discrete route-decision candidates. Later visual readback often needs these contexts to explain why a turn or stop was proposed, but coverage and retrieval metrics must distinguish a candidate decision from an executed decision.

Action contract:

| Field | Meaning | Used for gate? |
|---|---|---|
| `raw_action_text` | raw model/controller text before normalization | no |
| `navigation_policy_ok` | whether `NavigationPolicySkill` returned a successful candidate | yes |
| `candidate_action_status` | `ok`, `invalid_candidate_action`, or `navigation_failed` | yes |
| `normalized_candidate_action` | normalized policy candidate before visual-readback controller intervention | yes |
| `final_action_after_controller` | executed action after STOP blocking or other controller logic | trace only |
| `candidate_event_type` | pre-controller event identity, for example `candidate_decision_point` | trace and pre-readback save reason |
| `confirmed_event_type` | post-controller identity, for example `executed_decision_point` or `blocked_stop_shadow` | metrics and v1 retrieval preference |
| `controller_event_status` | `pending`, `executed_as_candidate`, `changed_by_controller`, or `blocked_stop_shadow` | metrics and v1 retrieval preference |

For `event_gated_smoke`, valid `STOP`, `TURN_LEFT`, and `TURN_RIGHT` candidates are direct-save events. They do not need a novelty co-signal to pass the threshold, but their initial save reason is `candidate_decision_point` until controller resolution confirms the executed event.

Action-status rule:

- Assign direct-save action scores only when `navigation_policy_ok=true` and `normalized_candidate_action` is one of `STOP`, `TURN_LEFT`, or `TURN_RIGHT`.
- For missing, empty, unrecognized, or failed policy results, set `normalized_candidate_action=""`, `eligible=false`, and `skip_reason` to `invalid_candidate_action` or `navigation_failed`.
- Do not promote a keyframe or run visual readback for that step based on a fallback `STOP` value from an error path.
- A candidate-only keyframe must not count as `keyframe_quality_event_coverage` for an executed route event unless `final_action_after_controller` matches the normalized candidate, or a post-readback/controller label explicitly confirms the frame as useful evidence.
- In v1 event-type-aware retrieval, STOP/turn slots should prefer `executed_decision_point`, then `blocked_stop_shadow` when the query is about STOP verification, then `candidate_decision_point` only when no confirmed alternative exists.

Recommended scoring metadata:

| Signal | Score |
|---|---:|
| `STOP` | `+1.0` |
| `TURN_LEFT` / `TURN_RIGHT` | `+1.0` |

### 2. Spatial Transition Frames

Save when non-oracle runtime evidence suggests a transition:

- current step has a `keyframe_candidate` from an existing visual update
- planner or readback text indicates doorway, hallway, room entry, foyer, dining room, den, living room, kitchen, stairs, corridor, outdoor threshold
- recent action pattern suggests a turn followed by forward motion through an opening
- current frame differs strongly from the last saved keyframe

This class is deferred until `event_gated_v1`. It should not block the first smoke.

When text-derived transition evidence is enabled, each text score must record:

- `source_field`
- `source_step_id`
- `source_age_steps`

Cached or readback-derived text may only contribute inside an explicit freshness window. Stale cached summaries must not describe the current image.

Recommended weight:

| Signal | Score |
|---|---:|
| probable room/doorway transition | `+0.9` |

### 3. Uncertainty or Conflict Frames

Save when an existing controller/readback result reports uncertainty or conflict:

- `route_conflict`
- `goal_not_visible`
- `insufficient_evidence`
- `log_replan_request`
- `block_stop_shadow`
- low verifier confidence
- planner fallback or model fallback

These frames are useful as diagnostic evidence. They also help later replay and fixed-trigger analysis.

These signals are causally late. They are produced after the candidate action and often after `VisualMemoryReadSkill`, so they must not be used to choose the memory images for the same readback call that produced them.

This class is deferred until `event_gated_v1` or later. `event_gated_smoke` may trace late uncertainty labels for diagnostics, but it must not promote keyframes from `post_readback_gate`.

Required phase contract:

- `pre_readback_gate`: may use normalized candidate action, coverage gap, current-frame artifact availability, prior-step state, and optional cheap novelty.
- `post_readback_gate`: in v1/later, may use `route_conflict`, `block_stop_shadow`, low confidence, or verifier labels only to promote the current frame for future retrieval.
- post-readback promotions set `available_for_readback_step = step_id + 1`.
- same-step replay is out of scope unless a later design explicitly adds fixed-trigger replay.

Recommended weights:

| Signal | Score |
|---|---:|
| route conflict / replan request | `+1.0` |
| risky STOP / STOP block | `+1.0` |
| insufficient evidence / low confidence | `+0.6` |
| planner fallback | `+0.8` |

### 4. Visual Novelty Frames

Save when the current image is substantially different from recent keyframes.

Initial low-dependency option:

- compute a perceptual hash or small resized grayscale descriptor
- compare against the last `K=3` saved keyframes
- treat the frame as novel if the minimum distance is above a threshold

Before enabling visual novelty in v1, the descriptor and threshold must be named in config or summary metadata and validated on the fixed qualitative manifest. Until then, novelty remains disabled.

Recommended weight:

| Signal | Score |
|---|---:|
| visually novel against recent keyframes | `+0.7` |
| highly redundant with recent keyframes | `redundancy_penalty = 0.8` |

### 5. Sparse Coverage Fallback

Save if the agent has moved for a long time without any saved keyframe.

Recommended initial rule:

- save if `steps_since_last_keyframe >= 40`
- use `50` for long-horizon evaluations if latency or storage is still too high

For `event_gated_smoke`, the configured coverage gap is a direct-save event. It fills route-evidence holes even when the additive score would otherwise stay below threshold.

Recommended scoring metadata:

| Signal | Score |
|---|---:|
| coverage gap >= 40 steps | `+1.0` |

The fallback should fill gaps, not recreate interval-based saving. The guard against interval-like behavior is the larger gap, cooldown, and episode cap, not a sub-threshold score.

The coverage score is non-stacking. A larger gap such as `>=60` may be recorded as diagnostic metadata, but it must not add a second `+1.0` on top of the configured coverage-gap score.

## Scoring Rule

Initial threshold:

```text
save if score >= 1.0
```

For `event_gated_smoke`, direct-save events are enough:

```text
direct_save_event =
  step_id == 0
  or (
    navigation_policy_ok
    and normalized_candidate_action in {STOP, TURN_LEFT, TURN_RIGHT}
  )
  or steps_since_last_keyframe >= keyframe_coverage_gap_steps
```

Initial score components:

```text
score =
  action_score
  + transition_score
  + uncertainty_score
  + visual_novelty_score
  + coverage_gap_score
  - redundancy_penalty
```

Example decisions:

| Situation | Expected outcome |
|---|---|
| `TURN_LEFT` from normalized candidate action | save |
| `MOVE_FORWARD` with similar view and only 8 steps since last keyframe | skip |
| `STOP` near destination | save |
| `MOVE_FORWARD` after 45 steps without keyframe | save fallback |
| `TURN_RIGHT` but image is nearly identical to a keyframe saved 2 steps ago | save in smoke, later v1 may suppress if redundancy policy is enabled and no direct-save override applies |
| `route_conflict` from readback | v1/later post-readback promote for future retrieval; not available to the same readback call |

## Cooldown and Caps

Use these limits and retrieval-window settings to prevent memory explosion and preserve recent-keyframe behavior:

| Limit | Initial value | Notes |
|---|---:|---|
| `keyframe_min_gap_steps` | `5` | Do not save nearby duplicates. |
| `keyframe_episode_cap` | `30` | Per episode cap for normal runs. |
| `keyframe_recent_window` | `8` | Keep current `recent_keyframe_paths` behavior unless retrieval changes require more. |

Cooldown bypass candidates by phase:

- smoke: `STOP`, `TURN_LEFT`, or `TURN_RIGHT` only when no keyframe has been saved for the current action segment
- v1/later: `route_conflict` in the post-readback gate
- v1/later: `block_stop_shadow` in the post-readback gate
- v1/later: planner/model fallback

Do not bypass the episode cap except for explicit debug mode. If the cap is reached, keep trace records explaining that the frame was eligible but skipped by cap.

Cap recovery policy:

- Smoke default is `fail_validation_and_rerun`: do not silently evict existing memories and do not claim the cap is validated if a late `STOP`, turn, or configured coverage-gap event is cap-blocked.
- If cap validation fails, either raise `keyframe_episode_cap` and rerun every compared arm with the new cap, or introduce a separate v1 replacement arm.
- V1 may add `priority_replacement`, but only as an explicit evaluation arm. Replacement may evict older `coverage_gap` or redundant candidate-only frames before confirmed decision/STOP/conflict frames.
- Trace `keyframe_cap_recovery_policy`, `keyframe_cap_validation_failed`, `keyframe_replacement_count`, and `evicted_keyframe_reason_counts` whenever cap pressure occurs.

Action-segment semantics:

- For direct-save action events, an action segment is the maximal consecutive run of the same normalized candidate action among `STOP`, `TURN_LEFT`, and `TURN_RIGHT`.
- The segment resets when the normalized candidate action changes, when the action becomes a non-direct-save action such as `MOVE_FORWARD`, or when an explicit controller state transition is recorded.
- Save at most one direct-save keyframe per action segment before normal cooldown/cap checks. This applies to repeated `STOP` steps as well as repeated turns.
- Trace `action_segment_id`, `segment_keyframe_saved`, and `last_saved_action_segment_id` so skipped repeated action frames are explainable.

## Proposed Trace Contract

Every step should record a `keyframe_gate` block, whether or not a keyframe is saved. This primary example is smoke-only; v1 may add visual-novelty and transition fields after those features are enabled.

```json
{
  "keyframe_gate": {
    "eligible": true,
    "saved": true,
    "save_reason": "candidate_decision_point",
    "policy_version": "event_gated_smoke",
    "score": 1.0,
    "threshold": 1.0,
    "cooldown_blocked": false,
    "cap_blocked": false,
    "action_segment_id": "TURN_RIGHT:3",
    "segment_keyframe_saved": false,
    "last_saved_action_segment_id": "",
    "steps_since_last_keyframe": 12,
    "gate_phase": "pre_readback",
    "signals": {
      "raw_action_text": "turn right",
      "navigation_policy_ok": true,
      "candidate_action_status": "ok",
      "normalized_candidate_action": "TURN_RIGHT",
      "final_action_after_controller": "",
      "candidate_event_type": "candidate_decision_point",
      "confirmed_event_type": "",
      "controller_event_status": "pending",
      "action_score": 1.0,
      "transition_score": 0.0,
      "uncertainty_score": 0.0,
      "visual_novelty_score": 0.0,
      "coverage_gap_score": 0.0,
      "redundancy_penalty": 0.0
    },
    "source_current_image_path": "results/.../openclaw_current_frames/scene/episode/step_000024.png",
    "promoted_keyframe_path": "results/.../keyframes/scene/episode/step_000024.png",
    "available_for_readback_step": 25,
    "promotion_status": "promoted",
    "write_status": "written",
    "memory_id": "episode:scene:episode:step_000024",
    "memory_id_link_status": "backend_returned",
    "skip_reason": "",
    "error_type": "",
    "image_path": "results/.../keyframes/scene/episode/step_000024.png"
  }
}
```

This makes the method auditable. A future result directory should let the user answer:

- why this frame was saved
- why another nearby frame was skipped
- which saved frame later became a retrieved memory image
- whether retrieval used diverse route evidence or repeated similar frames
- whether a gate failed because of missing image, cooldown, cap, copy/promotion error, or memory-write failure

Failure behavior:

| Condition | Save result | Navigation behavior | Trace status |
|---|---|---|---|
| missing current image artifact | no keyframe save | continue with text/controller path if possible | `promotion_status=skipped`, `error_type=missing_current_image` |
| copy/promotion failure | no keyframe save | continue; frame is not available for future retrieval | `promotion_status=failed`, `write_status=skipped`, `error_type=copy_failed` |
| `MemoryWriteSkill` failure | keyframe file may exist, but not queryable memory | continue; do not count as retrievable memory | `write_status=failed`, `error_type=memory_write_failed` |
| missing, empty, unrecognized, or failed candidate action | no decision-point keyframe save | continue through normal fallback outside the keyframe gate; do not run readback because of fallback `STOP` | `eligible=false`, `skip_reason=invalid_candidate_action` or `navigation_failed` |
| cooldown blocked | no keyframe save | continue normally | `eligible=true`, `saved=false`, `cooldown_blocked=true` |
| action-segment guard blocked | no keyframe save | continue normally | `eligible=true`, `saved=false`, `segment_keyframe_saved=true` |
| cap blocked | no keyframe save unless debug override is enabled | continue normally | `eligible=true`, `saved=false`, `cap_blocked=true` |
| v1 metadata round-trip failure | keyframe may remain usable as a raw image, but diversity metadata is invalid | continue in smoke-compatible retrieval, or fail v1 validation | `write_status=written`, `metadata_roundtrip_status=failed` |

Memory-write failure boundary:

- `MemoryWriteSkill` must catch exceptions from `client.ingest_semantic(record)` and validate failed or falsey backend responses.
- The skill returns a structured failed-write result with `ok=false`, `error_type`, and any backend error text safe to log.
- `OpenClawVLNRuntime` translates that result into `keyframe_gate.write_status="failed"` and `error_type="memory_write_failed"` without aborting the navigation step.
- Failed writes must not be added to `recent_keyframe_paths`, candidate-pool retrieval, or any retrievable-memory count. The keyframe file may remain on disk for debugging, but it is counted separately from queryable memory.

## Memory Write Contract

When a keyframe is saved and written to memory, preserve existing fields and add keyframe metadata:

```json
{
  "source_image_role": "event_gated_keyframe",
  "retrieval_text": "instruction=<episode instruction>; event=candidate_decision_point; action=TURN_RIGHT; step=24; namespace=<memory_namespace>",
  "metadata": {
    "keyframe_gate": {
      "reason": "candidate_decision_point",
      "score": 1.0,
      "event_types": ["candidate_decision_point"],
      "policy_version": "event_gated_smoke",
      "gate_phase": "pre_readback",
      "candidate_event_type": "candidate_decision_point",
      "confirmed_event_type": "",
      "controller_event_status": "pending",
      "source_current_image_path": "results/.../openclaw_current_frames/scene/episode/step_000024.png",
      "promoted_keyframe_path": "results/.../keyframes/scene/episode/step_000024.png",
      "available_for_readback_step": 25
    }
  }
}
```

Smoke implementation contract:

- `MemoryWriteSkill` stores keyframe metadata under `metadata.keyframe_gate`.
- the write payload and harness trace both preserve `metadata.keyframe_gate`.
- each smoke write includes queryable non-oracle `retrieval_text` or `action_context` populated from instruction text, `save_reason`, `normalized_candidate_action`, `event_types`, `step_id`, and `memory_namespace`.
- `retrieval_text` must not use a new Qwen caption or future/oracle labels. Empty or missing `retrieval_text` writes are counted as write-quality failures.
- `memory_index.jsonl` should record the metadata when feasible; if the index schema remains minimal, the companion trace must still allow lookup from `memory_id` to `keyframe_gate`.

Write identity contract:

- `MemoryWriteSkill` must capture backend-returned memory IDs from `ingest_semantic` when the backend returns them.
- The memory-write result and `keyframe_gate` trace should record `memory_id` and `memory_id_link_status`.
- If the backend cannot return a memory ID, fall back to a deterministic `memory_namespace:image_path` link and set `memory_id_link_status="path_only"`.
- A retrieved `MemoryHit.memory_id` should match the stored `memory_id`; if not, audit may fall back to the exact retrieved image path only when `memory_id_link_status="path_only"`.

Full-v1 implementation contract:

- the keyframe fields round-trip into `MemoryHit.metadata`, not only the raw write request.
- `ImageBackedLocalMemoryClient` returns `metadata.keyframe_gate`, `metadata.step_id`, `memory_scope`, and `memory_namespace` in each `MemoryHit`.
- diversity retrieval may depend on `metadata.keyframe_gate.event_types`, `metadata.keyframe_gate.score`, `metadata.keyframe_gate.policy_version`, and `metadata.step_id`.

Spatial HTTP compatibility:

- when the spatial HTTP backend is enabled for `event_gated_v1`, ingest and query responses must preserve `source_image_role`, `metadata.keyframe_gate`, `metadata.step_id`, `memory_scope`, and `memory_namespace`.
- `event_gated_v1` config should fail fast, or mark retrieval diversity disabled, if spatial query results cannot echo `metadata.keyframe_gate`.
- `event_gated_smoke` does not require spatial HTTP metadata round-trip. It only requires local trace/write-payload evidence that can explain why a frame was saved.

## Retrieval Changes

The keyframe-saving policy and readback retrieval should be aligned. If saving becomes event-gated but retrieval still returns three near-duplicate frames, the readback cost remains poorly spent.

Use diversity-aware top-K for visual readback:

```text
slot 1: most recent confirmed decision/STOP keyframe, then relevant candidate decision if no confirmed alternative exists
slot 2: most recent spatial-transition keyframe
slot 3: most instruction-relevant visually distinct keyframe
```

If fewer than three categories are available, fill with the best remaining non-duplicate keyframes.

For non-default `visual_readback_top_k`:

- when `top_k < 3`, apply the named slots in order and truncate.
- when `top_k > 3`, fill slots 4+ with the best remaining non-duplicate keyframes using the same tie-break order.

Initial constraints:

- default `visual_readback_top_k = 3`
- default `visual_readback_candidate_pool_k = 8`
- never attach duplicate image paths
- only select prior keyframes from the same episode for this design; cross-episode selection remains out of scope
- avoid selecting only adjacent interval-like frames unless no alternatives exist

The current `_auto_recall_memory(...)` hardcodes `n_results=3`. This design separates policy recall from visual-readback recall:

- `_auto_recall_memory(...)` remains the pre-planner/policy recall path unless a later refactor explicitly changes it.
- `_run_visual_readback(...)` or the visual-readback wrapper owns a readback-scoped `MemoryQuerySkill` call after a readback trigger is selected.
- the readback-scoped query uses `visual_readback_candidate_pool_k` for candidate recall and `visual_readback_top_k` for final image attachment.
- candidate-pool query results are not merged into the policy payload or `policy_context`; they are only used to choose memory images for the current readback call.

Candidate recall and final attachment:

```text
query candidate_pool_k memories
-> drop current-step and duplicate image paths
-> bucket by metadata.keyframe_gate.event_types
-> apply deterministic slot selection
-> attach final visual_readback_top_k memories
```

Deterministic selector requirements:

- `candidate_pool_requested_k` is the configured request size, normally `visual_readback_candidate_pool_k`.
- `candidate_pool_returned_count` is the number of memory hits returned before current-step, duplicate, or eligibility filtering.
- `eligible_prior_keyframe_count` is the number of prior keyframes still available after filtering.
- final attached count is `visual_readback_top_k`.
- event buckets are `executed_decision_point`, `blocked_stop_shadow`, `candidate_decision_point`, `spatial_transition`, `uncertainty_conflict`, `coverage_gap`, and `other`.
- adjacency means all selected keyframes are within one `keyframe_min_gap_steps` window or have consecutive saved-step order.
- tie-break order is higher retrieval score, then larger gate score, then newer step, then stable path sort.
- trace records `candidate_pool_requested_k`, `candidate_pool_returned_count`, `eligible_prior_keyframe_count`, `eligible_alternative_count`, selected bucket per slot, `dedupe_dropped_count`, and `adjacent_triplet_avoided`.

The v1 diversity gate computes adjacent-triplet rate only over readbacks with `eligible_alternative_count > 0`. Readbacks with no non-adjacent alternative should be reported separately, not counted as diversity failures.

First-slice retrieval should stay smaller: exact duplicate path removal plus measurement of adjacent-triplet rate is enough. Event-type-aware reranking should wait until keyframe metadata round-trips reliably.

Candidate-pool adequacy for v1:

- Before claiming diversity improvements, audit a fixed qualitative manifest with `candidate_pool_k=8` against a large-pool or full-episode candidate set.
- If `candidate_pool_k=8` hides non-adjacent alternatives that the large-pool audit finds, raise the pool size or mark diversity claims as inconclusive.

Readback query failure behavior:

| Condition | Readback behavior | Navigation behavior | Trace status |
|---|---|---|---|
| `MemoryQuerySkill` unavailable | do not call `VisualMemoryReadSkill` in `image_read_controller` | keep the candidate action; do not merge query `policy_context` | `read_status=skipped`, `skip_reason=memory_query_unavailable`, `candidate_pool_requested_k=<configured>`, `candidate_pool_returned_count=0`, `eligible_prior_keyframe_count=0` |
| memory query fails or times out | do not call `VisualMemoryReadSkill` in `image_read_controller` | keep the candidate action; do not merge query `policy_context` | `read_status=skipped`, `skip_reason=memory_query_failed`, `query_error_type=<error>`, `candidate_pool_returned_count=0`, `eligible_prior_keyframe_count=0` |
| zero eligible prior keyframes after filtering | do not call `VisualMemoryReadSkill` in `image_read_controller` | keep the candidate action; do not merge query `policy_context` | `read_status=skipped`, `skip_reason=no_memory_hit`, preserve `candidate_pool_returned_count`, set `eligible_prior_keyframe_count=0` |

## Online Data Flow

Target flow:

```text
JanusVLN step image
-> build runtime payload
-> HarnessModelProxy saves current image as an openclaw_current_frames artifact
-> payload includes current_image_path and deterministic keyframe_target_path
-> NavigationPolicySkill produces candidate action
-> pre_readback_gate runs with normalized candidate action, coverage, prior state, optional novelty
-> runtime promotes/copies current frame to keyframe_target_path if pre_readback_gate passes
-> runtime writes keyframe metadata to image-backed episode-local memory
-> VisualMemoryReadSkill triggers at the frozen STOP/turn rules for Phase A/B, or at explicit prior-conflict gates only in a separate trigger-policy experiment
-> readback retrieves prior event-gated keyframes; v1 may rerank them for diversity; current image is attached separately
-> post_readback_gate may promote/update current frame for future retrieval in v1/later if conflict/uncertainty appears
-> trace records current image + promoted keyframe + retrieved keyframes + verifier evidence
```

Promotion ownership contract:

- `HarnessModelProxy` owns initial current-frame artifact creation under `openclaw_current_frames/...`.
- `HarnessModelProxy` passes both `current_image_path` and `keyframe_target_path` into the runtime payload.
- `OpenClawVLNRuntime` owns the action-dependent gate and performs the copy/promotion before memory write.
- runtime returns `promoted_keyframe_path`, `keyframe_gate`, and write status in `runtime_metadata`.
- `HarnessModelProxy` consumes runtime metadata after `runtime.step(...)` and updates `recent_keyframe_paths` for the next step.
- same-step readback uses `current_image_path` directly; promoted keyframes enter the historical-memory retrieval pool from `step_id + 1`.

Eager-versus-lazy promotion decision:

- `event_gated_smoke_gate` may validate save/skip decisions from trace alone, but `event_gated_smoke_audit` uses eager keyframe promotion and memory writes for frames that pass the gate.
- The reason is compatibility with the current image-backed memory path: `MemoryQuerySkill` can retrieve memory records, but it does not query arbitrary `openclaw_current_frames` rows by trace labels.
- A lazy trace-indexed selector over `openclaw_current_frames` remains a possible future optimization or debugging tool, but it is not the primary smoke path unless a later implementation adds a query layer over trace labels.
- If the backend cannot write queryable memory, the run may still keep trace-only keyframe labels, but it must not count those frames as retrievable memory.

V1 post-readback merge/upsert contract:

- There must be at most one memory record per `(episode_id, step_id, promoted_keyframe_path)`.
- If `pre_readback_gate` already wrote the current frame and `post_readback_gate` later adds conflict or uncertainty labels for the same step, v1 should update that record rather than create a duplicate memory record.
- Post-readback labels append to `metadata.keyframe_gate.event_types` and update phase/status fields such as `post_readback_labels`, `post_readback_merge_status`, and `available_for_readback_step`.
- If the active backend cannot update existing memory records, post-readback labels should remain trace-only and `post_readback_merge_status="trace_only"`. Do not create a same-step duplicate memory record.

Gate phases:

| Phase | Runs | Allowed signals | Memory availability |
|---|---|---|---|
| `stage_a_current_artifact` | before policy | image exists, step id, output paths | current view only |
| `pre_readback_gate` | after candidate action, before readback | normalized action, coverage gap, prior-step state, optional cheap novelty | current view is attached directly; promoted keyframe is retrieved from `step_id + 1` |
| `post_readback_gate` | v1/later after visual readback/controller labels | route conflict, STOP block, low confidence, verifier labels | future retrieval only: `available_for_readback_step = step_id + 1` |

This avoids losing action-dependent decision frames while preventing readback-derived labels from leaking backward into the same readback's retrieval set.

## Configuration

Add config values with CLI flag > environment variable > default precedence.

Required smoke config surface:

| HarnessConfig field | CLI flag | Environment variable | Default | Purpose |
|---|---|---|---:|---|
| `keyframe_policy_mode` | `--keyframe-policy-mode` | `OPENCLAW_KEYFRAME_POLICY_MODE` | `interval` | Compatibility default. New values: `event_gated_smoke`, `event_gated_v1`. |
| `keyframe_min_gap_steps` | `--keyframe-min-gap-steps` | `OPENCLAW_KEYFRAME_MIN_GAP_STEPS` | `5` | Cooldown. |
| `keyframe_episode_cap` | `--keyframe-episode-cap` | `OPENCLAW_KEYFRAME_EPISODE_CAP` | `30` | Per-episode cap. |
| `keyframe_coverage_gap_steps` | `--keyframe-coverage-gap-steps` | `OPENCLAW_KEYFRAME_COVERAGE_GAP_STEPS` | `40` | Direct-save coverage fallback gap. |
| `keyframe_debug_save_all_eligible` | `--keyframe-debug-save-all-eligible` | `OPENCLAW_KEYFRAME_DEBUG_SAVE_ALL_ELIGIBLE` | `false` | Debug-only override for analysis. |
| `visual_readback_top_k` | `--visual-readback-top-k` | `OPENCLAW_VISUAL_READBACK_TOP_K` | `3` | Existing final readback attachment count. |

Deferred `event_gated_v1` config surface:

| HarnessConfig field | CLI flag | Environment variable | Default | Purpose |
|---|---|---|---:|---|
| `keyframe_gate_threshold` | `--keyframe-gate-threshold` | `OPENCLAW_KEYFRAME_GATE_THRESHOLD` | `1.0` | Additive score threshold for non-direct-save events. |
| `keyframe_recent_compare_k` | `--keyframe-recent-compare-k` | `OPENCLAW_KEYFRAME_RECENT_COMPARE_K` | `3` | Novelty comparison window for v1. |
| `keyframe_enable_visual_novelty` | `--keyframe-enable-visual-novelty` | `OPENCLAW_KEYFRAME_ENABLE_VISUAL_NOVELTY` | `false` | Enabled only for named v1 comparisons or ablations. |
| `keyframe_text_signal_max_age_steps` | `--keyframe-text-signal-max-age-steps` | `OPENCLAW_KEYFRAME_TEXT_SIGNAL_MAX_AGE_STEPS` | `3` | Freshness window for text-derived transition signals. |
| `visual_readback_candidate_pool_k` | `--visual-readback-candidate-pool-k` | `OPENCLAW_VISUAL_READBACK_CANDIDATE_POOL_K` | `8` | Candidate pool used before final diverse selection. |

Validation rules:

- `keyframe_policy_mode=interval` preserves current behavior.
- `event_gated_smoke` consumes only the required smoke config surface. It requires no visual novelty, no transition keyword scoring, no candidate-pool diversity, and no `MemoryHit.metadata` round-trip.
- V1-only config fields should be added only when the implementation consumes them.
- v1 evaluation arms introduce candidate-pool retrieval, visual novelty, and text transition scoring in separate validation steps before the final combined arm.
- `visual_readback_candidate_pool_k >= visual_readback_top_k` when v1 candidate-pool recall is enabled.
- launchers should pass through smoke-phase `OPENCLAW_KEYFRAME_*` variables before smoke validation. V1-only variables should not block `event_gated_smoke`.
- summaries must report effective flags for novelty, text transition scoring, candidate-pool recall, and event-type-aware retrieval.

Keep `interval` mode as the default until event-gated mode passes smoke and regression checks. This preserves existing benchmark comparability.

## Acceptance Metrics

A valid `event_gated_smoke` implementation should report these summary fields:

```text
keyframe_policy_mode
visual_readback_top_k
keyframe_saved_count
keyframe_saved_by_reason
keyframe_initial_context_saved_count
keyframe_eligible_direct_save_segment_count
keyframe_skipped_by_cooldown_count
keyframe_skipped_by_action_segment_count
keyframe_skipped_by_cap_count
keyframe_cap_hit_episode_count
keyframe_cap_hit_episode_rate
keyframe_cap_blocked_by_event_type
keyframe_latest_step_cap_blocked
keyframe_mean_gap_steps
keyframe_duplicate_rate
keyframe_promotion_failure_count
keyframe_write_failure_count
memory_id_link_status_counts
keyframe_empty_retrieval_text_count
raw_to_normalized_action_counts
action_normalization_unknown_count
keyframe_candidate_decision_count
keyframe_confirmed_decision_count
keyframe_candidate_only_decision_count
keyframe_blocked_stop_shadow_count
keyframe_cap_validation_failed
keyframe_replacement_count
readback_retrieved_keyframe_count
readback_adjacent_frame_triplet_rate
readback_exact_duplicate_dropped_count
readback_skipped_by_query_failure_count
readback_skipped_by_no_memory_hit_count
readback_memory_evidence_used_count
readback_current_frame_only_evidence_count
readback_ambiguous_evidence_reference_count
readback_no_evidence_reference_count
```

A valid `event_gated_v1` implementation should additionally report:

```text
visual_readback_candidate_pool_k
candidate_pool_requested_k
candidate_pool_returned_count
eligible_prior_keyframe_count
readback_distinct_event_type_count
readback_memory_evidence_used_rate
readback_evidence_source_invalid_count
keyframe_quality_precision
keyframe_quality_event_coverage
candidate_pool_large_pool_recall_check
```

`readback_memory_evidence_used_rate` is the primary v1 readback-value metric. A readback counts as memory-evidence-used only when at least one non-current retrieved keyframe is explicitly linked by `memory_id`, image path, step id, or readback slot label to `visual_evidence`, `audit_action_hint`, verifier labels, or controller reasoning. Current-frame-only evidence, ambiguous references, and readbacks without completed visual evidence should be counted separately:

```text
readback_memory_evidence_used_count
readback_current_frame_only_evidence_count
readback_ambiguous_evidence_reference_count
readback_no_evidence_reference_count
```

Smoke should report the evidence-use counts above as diagnostics, but it must not claim readback-value improvement from those counts alone. The v1 rate is trace/audit based; it does not claim navigation success.

Visual readback producer contract:

- `VisualMemoryReadSkill` output or its parser-facing wrapper should attach an evidence source to every `visual_evidence`, `audit_action_hint`, verifier label, and controller rationale item.
- Required source fields are `evidence_source_type` (`current_frame`, `retrieved_memory`, or `none`), plus at least one stable retrieved-memory identity when `evidence_source_type="retrieved_memory"`: `memory_id`, `retrieved_image_path`, `step_id`, or `readback_slot_id`.
- If an item references multiple memories, list all identities and count the readback as memory-evidence-used only when at least one identity resolves to a retrieved non-current keyframe.
- Missing, duplicate, or invalid IDs are counted under `readback_ambiguous_evidence_reference_count` or `readback_evidence_source_invalid_count`, not silently credited as memory use.

Gate-only checkpoint for `event_gated_smoke_gate`:

| Gate | Pass condition |
|---|---|
| Trace completeness | every step has `keyframe_gate` metadata |
| Initial context | `step_id == 0` produces an `initial_context` keyframe unless the image artifact is missing or cap blocks it |
| Action status | direct-save action scores require `navigation_policy_ok=true` and a recognized normalized action |
| Direct-save correctness | direct-save events produce eligible keyframe decisions and are saved unless cooldown, action-segment guard, or cap blocks them |
| Action normalization | raw-to-normalized counts and unknown action counts are reported; STOP/TURN variants have unit-test coverage |
| Cap validation | fixed manifest reports eligible direct-save segments, cap-hit episode rate, cap-blocked event classes, and latest-step cap blocks; no later STOP or coverage-gap event is cap-blocked before treating `30` as validated |
| Phase safety | post-readback labels are never available to same-step retrieval |
| Memory write correctness | saved queryable-memory image paths point to `keyframes/...`, not `openclaw_current_frames/...` |
| Storage sanity | saved keyframes per episode stay below cap except debug mode |
| Retrieval text | each written smoke memory has non-empty non-oracle `retrieval_text` or is counted as a write-quality failure |
| Write failure safety | memory-write exceptions or failed ingest responses set `write_status=failed` without aborting navigation or adding queryable memory |
| Candidate identity | candidate-only STOP/TURN saves are counted separately from confirmed executed route events |
| Latency | no new Qwen calls introduced by keyframe selection |
| Behavior safety | no online oracle fields used for keyframe decisions |
| Metadata write traceability | `metadata.keyframe_gate` appears in both the write payload and harness trace, and `memory_id` can be linked back to the saved-frame reason; `memory_index.jsonl` may stay minimal when the companion trace provides this link |

Audit checkpoint for `event_gated_smoke_audit`:

| Gate | Pass condition |
|---|---|
| Retrieval smoke | exact duplicate image paths are not attached to the same readback |
| Retrieval failure traceability | memory-query missing/failure/empty-pool paths produce deterministic `read_status` and `skip_reason` fields |
| Adjacent measurement | adjacent-triplet rate is reported with eligible-alternative denominators |
| Evidence diagnostics | memory-evidence/current-only/ambiguous/no-evidence counts are reported without claiming readback-value improvement |
| Summary completeness | cap-hit, cap-blocked event classes, action-normalization, retrieval text, write failure, and memory-id link counts are present |

Full-v1 gates:

| Gate | Pass condition |
|---|---|
| Metadata round-trip | `metadata.keyframe_gate` survives memory write, query, and `MemoryHit.metadata` |
| Diversity | less than 25% of readbacks attach three adjacent keyframes when alternatives exist |
| Candidate-pool adequacy | fixed-case audit shows `candidate_pool_k=8` does not hide available non-adjacent alternatives, or the configured pool is raised |
| Freshness | text-derived transition signals include `source_field`, `source_step_id`, and `source_age_steps` |
| Readback value | `readback_memory_evidence_used_rate` is reported with retrieved-keyframe identity binding and inspected against interval baselines |
| Quality audit | fixed-case keyframe labels report route-event precision and event coverage with predeclared denominators |

## Evaluation Plan

Freeze these conditions across all arms:

- episode manifest and ordering
- seed policy
- max-step budget
- visual readback mode
- readback trigger rules
- controller effect policy
- Qwen/model configuration

For Phase A/B memory-quality comparisons, explicit prior-conflict readback triggers are disabled unless the same prior-conflict trigger rule is enabled for every arm being compared. Trigger-policy changes belong to a separate stricter-trigger experiment, not to the keyframe-policy comparison.

Report unmatched trigger counts separately. When possible, compute readback-quality metrics on shared readback trigger keys so a different trigger frequency does not masquerade as better memory selection.

Phase 0: premise check:

1. `interval_10`: current behavior on the fixed qualitative manifest.

Do not start event-gated implementation until Phase 0 reports current missed turn/STOP opportunities, adjacent-triplet rate, exact duplicate rate, memory-evidence-use counts, and shared readback trigger keys.

Phase A: smoke evaluation:

1. `interval_10`: current behavior.
2. `interval_10_exact_dedupe`: current keyframe saving, but readback drops exact duplicate image paths.
3. `event_gated_smoke_gate`: implementation checkpoint only; initial context, direct-save action/coverage gate, action-segment guard, cooldown/cap, core trace/write metadata, no novelty and no transition keyword scoring.
4. `event_gated_smoke`: comparison arm after the audit checkpoint; gate behavior plus retrieval failure traces, exact readback image-path dedupe, adjacent-triplet measurement, evidence-use diagnostics, and summary counts.

Phase A checkpoint:

- Compare `interval_10_exact_dedupe` against `event_gated_smoke` on shared readback trigger keys.
- Separate keyframe-saving effects from exact-dedupe effects before claiming smoke value.
- If retrieval-only dedupe explains most of the gain and interval keyframes do not miss labeled decision opportunities, stop at retrieval cleanup or limit event-gated work to traceability.
- Pass requires redundancy to improve without losing route evidence: saved-keyframe duplicate rate or attached-memory adjacent-triplet rate must improve against the relevant interval baseline, while `keyframe_quality_event_coverage` and `retrieved_route_event_coverage` do not regress on predeclared shared route events.

Phase B: v1 evaluation:

1. `interval_10`: current behavior.
2. `interval_10_diverse_retrieval`: current keyframe saving, but retrieval applies exact dedupe and candidate-pool diversity logic.
3. `event_gated_smoke`: smoke behavior without novelty/text scoring.
4. `event_gated_v1_candidate_pool`: smoke behavior plus candidate-pool recall and event-type-aware retrieval.
5. `event_gated_v1_visual_novelty`: candidate-pool behavior plus cheap visual novelty.
6. `event_gated_v1_full`: visual-novelty behavior plus fresh text transition scoring.

Phase B pass gate:

- Event-gated v1 may claim better memory selection only if `readback_memory_evidence_used_rate` improves or adjacent-triplet rate drops on shared triggers without regression in `keyframe_quality_event_coverage` or `retrieved_route_event_coverage`.
- If route-event coverage regresses, restrict the claim to the specific improvement that survived inspection, such as traceability, exact-dedupe cleanup, or candidate-pool recall.
- If latency improves, attribute it only to fewer/cheaper readback calls when a separate trigger-policy arm changed call count. Do not attribute speedup to keyframe saving alone.

Report:

- keyframe count per episode
- readback completed count
- matched and unmatched readback trigger counts
- readback latency p50/p95
- average memory images attached per readback
- readback memory evidence used rate
- adjacent-triplet rate and candidate-pool size
- keyframe promotion/write failure counts
- route-evidence examples from `harness_trace_rank0.jsonl`
- SR/SPL/OS/NE only as secondary navigation metrics, not as proof of keyframe policy superiority

Use a fixed qualitative manifest instead of cherry-picking after results. Before running the comparison, choose the episodes and label route-event opportunities that the keyframe policy should cover. At minimum, choose one previously successful and one previously failed episode before running the comparison, then show:

```text
saved keyframe sequence
-> retrieval sequence at each readback trigger
-> visual_evidence / audit_action_hint
```

For the fixed cases, label route-event opportunities, saved frames, and retrieved frames by:

- route-event class: decision point, spatial transition, uncertainty/conflict, coverage gap, other
- whether the frame was useful for the readback trigger
- whether the readback explicitly used the memory image
- acceptable step tolerance for matching a saved keyframe to a route-event opportunity

Quality audit formulas:

```text
keyframe_quality_precision =
  useful_route_event_keyframes / labeled_saved_keyframes

keyframe_quality_event_coverage =
  predeclared_route_events_with_at_least_one_saved_keyframe_within_tolerance
  / predeclared_route_events
```

Retrieved-frame coverage should be reported separately from saved-frame coverage:

```text
retrieved_route_event_coverage =
  predeclared_route_events_with_at_least_one_retrieved_keyframe_within_tolerance
  / predeclared_route_events
```

Report these denominators before making claims that event-gated keyframes are more meaningful.

Speed claim boundary:

```text
visual_readback_call_count
visual_readback_completed_count
visual_readback_skipped_by_stricter_trigger_count
visual_readback_wall_time_p50
visual_readback_wall_time_p95
```

No faster-readback claim is allowed unless a separate stricter-trigger or cost-reduction arm reduces `visual_readback_call_count` or per-call wall time while preserving the memory-quality gates above.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Event gate misses useful straight-line progress evidence | keep 40-step coverage direct-save fallback |
| Too many turns create too many keyframes | cooldown and cap, plus one-save-per-action-segment logic |
| Repeated STOP exhausts episode cap | apply the same one-save-per-action-segment guard to `STOP` as to turns |
| Failed policy action is mistaken for real STOP | require `navigation_policy_ok=true` and recognized normalized action before action-based direct-save or readback |
| Candidate STOP/TURN is mistaken for executed route evidence | separate candidate and confirmed event identity; route-event coverage counts confirmed execution or explicitly labeled evidence only |
| Episode cap blocks late high-value evidence | smoke fails cap validation and reruns with a new cap, or v1 uses an explicit priority-replacement arm |
| Smoke memory is metadata-only and not semantically retrievable | require non-oracle `retrieval_text` or `action_context` on every smoke write |
| Memory write failure aborts navigation or pollutes retrieval | `MemoryWriteSkill` returns structured failed writes; runtime continues and excludes failed writes from queryable memory |
| Text keyword transition detector is brittle | use it as weak evidence, not a sole required condition |
| Cached text describes an old view | require source field, source step, and source age for all text-derived scores |
| Visual novelty adds CPU overhead | use small image descriptors first; no VLM call |
| Retrieval still returns redundant keyframes | add diversity-aware selection and trace adjacent-triplet rate |
| Retrieval-only dedupe explains most smoke gains | require `interval_10_exact_dedupe` in Phase A and compare on shared readback triggers |
| Diversity selector has too few candidates | separate candidate pool size from final attached top-K |
| Candidate pool hides better alternatives | audit `candidate_pool_k=8` against large-pool/full-episode candidates before claiming diversity improvements |
| Post-readback labels leak into same-step retrieval | enforce gate phases and `available_for_readback_step` |
| Post-readback labels duplicate a same-step memory | v1 uses one record per `(episode_id, step_id, promoted_keyframe_path)` or keeps labels trace-only |
| Saved memory cannot be linked back to gate reason | record backend memory IDs when available and fall back to deterministic path-only links when necessary |
| Evidence-use metric over-credits vague Qwen text | require evidence source IDs or slot references in visual-readback output; invalid references count separately |
| Evaluation improvement comes from different readback trigger counts | freeze readback trigger rules and report shared-trigger metrics plus unmatched trigger counts |
| Claims become overstated | restrict claims to memory-quality and readback-evidence improvements until intervention experiments exist |

## Implementation Boundary

This design is ready for an implementation plan, but it intentionally does not specify exact code edits line-by-line. The likely implementation areas are:

- `src/evaluation_harness.py`
  - runtime payload, keyframe artifact save/promotion, `keyframe_gate` trace source
- `src/harness/memory/working_memory.py` or equivalent
  - keyframe policy state, cooldown, cap, novelty cache
- `src/harness/openclaw/runtime.py`
  - action-dependent promotion, pre/post-readback gate phases, memory-write metadata, readback retrieval metadata
- `src/harness/skills/memory_write.py`
  - persist `metadata.keyframe_gate` without dropping event fields
- `src/harness/skills/memory_query.py` or memory manager layer
  - v1 candidate-pool query and final diverse selection
- `src/harness/visual_readback/memory_smoke.py` and spatial memory client path
  - v1 round-trip keyframe metadata into `MemoryHit.metadata`
- `scripts/summarize_openclaw_visual_readback.py`
  - keyframe policy and retrieval quality summary fields
- tests under `tests/`
  - no-oracle gate tests, action-status tests, cooldown/cap/action-segment tests, initial-context tests, failure-semantics tests, phase-order tests, trace contract tests, smoke metadata-write tests, memory-id link tests, readback-query failure tests, v1 metadata round-trip tests, retrieval diversity tests

## Recommended First Slice

Implement only the gate-only checkpoint first as `event_gated_smoke_gate`:

1. Add `keyframe_policy_mode=event_gated_smoke`.
2. Add direct-save events for `initial_context`, valid normalized `STOP`, `TURN_LEFT`, `TURN_RIGHT`, and coverage gap.
3. Add cooldown and cap.
4. Add action-status handling so failed, empty, or unrecognized policy actions cannot become fallback STOP keyframes.
5. Add candidate/confirmed event identity fields so candidate-only STOP/TURN saves do not count as executed route-event coverage.
6. Add `keyframe_gate` trace metadata, including phase, action contract, action segment, promotion/write status, memory-id link status, cap recovery policy, and failure reason fields.
7. Persist `metadata.keyframe_gate` and non-oracle `retrieval_text` or `action_context` in the memory-write payload and harness trace.
8. Implement the `MemoryWriteSkill` failure contract so failed writes do not abort navigation or enter queryable memory.
9. Verify `memory_id` can be linked back to `keyframe_gate`; `memory_index.jsonl` may stay minimal when the companion trace provides this link.
10. Keep visual novelty disabled for the first smoke.
11. Keep transition keyword scoring disabled for the first smoke.

Then complete `event_gated_smoke_audit` before using smoke in Phase A comparisons:

1. Add readback query failure trace fields for unavailable, failed, timed-out, and empty candidate pools.
2. Split candidate-pool trace counts into requested, returned, and eligible-prior-keyframe counts.
3. Keep retrieval final `top_k=3`, but only dedupe exact duplicate image paths and measure adjacent-triplet rate. Do not require category-aware reranking yet.
4. Report cap-hit, cap-blocked event-class, action-normalization, retrieval-text, write-failure, memory-id-link, and evidence-use diagnostic counts.
5. Add visual-readback evidence-source parsing so vague or invalid memory references are counted separately.
6. Apply the Phase A joint pass gate: less redundancy or adjacency without regression in saved/retrieved route-event coverage.

After this passes, add v1 in staged validation steps: candidate-pool diversity and event-type-aware reranking first, cheap visual novelty second, fresh text transition detection last.

## Expected Outcome

Expected `event_gated_smoke` artifact changes:

- `keyframes/` includes initial context, valid candidate decision-point keyframes, confirmed executed-event labels when available, and sparse coverage-gap keyframes with auditable save/skip reasons.
- `memory_index.jsonl` or the companion trace records why a frame was saved, links `memory_id` to `keyframe_gate`, and preserves non-oracle retrieval text.
- visual readback no longer attaches exact duplicate memory image paths in the same readback.
- trace files can explain keyframe saving, keyframe skipping, candidate-vs-confirmed event identity, cap validation, memory-id linking, write failures, readback query skips, and whether memory evidence was referenced.
- smoke reports adjacent-triplet and evidence-use diagnostics, but does not claim diverse retrieval or readback-value improvement by itself.

Expected `event_gated_v1` artifact changes:

- visual readback attaches more diverse memory images after candidate-pool and event-type-aware reranking pass validation.
- `keyframes/` has measured route-event precision and route-event coverage against the fixed qualitative manifest.
- post-readback conflict or uncertainty labels are merged into existing same-step memory records, or explicitly marked trace-only when the backend cannot update records.
- Qwen readback calls become easier to interpret because the attached memory images have traceable reasons and measured evidence-use.

The method remains compatible with the current control-only visual readback claim boundary: event-gated keyframes improve what memory is available to the controller/readback path, but they do not by themselves prove executed policy improvement.

This design validates evidence quality, not speed. It does not reduce `VisualMemoryReadSkill` call count or per-call latency by itself. Any faster-readback claim requires a separate stricter-trigger or cost-reduction arm that reports readback call count, skipped-by-trigger count, and wall-time metrics while preserving the memory-quality gates.
