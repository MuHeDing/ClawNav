# Qwen Dynamic Visual Context and Thinking Eval

This runbook evaluates dynamic visual evidence and native Qwen thinking as a
fixed 2x2 experiment. The four diagnostic arms use the same dated model,
synchronous transport, strict JSON schema, visual cadence, map cadence, controller
settings, and exact episode keys.

## Capability Probe

Run the explicit thinking probe before starting the matrix:

```bash
PYTHONPATH=src:. python scripts/check_openclaw_qwen_connection.py \
  --mode api \
  --model qwen/qwen3.5-flash-2026-02-23 \
  --thinking-mode on \
  --thinking-budget 1024 \
  --transport-mode sync
```

Do not run a thinking-on arm when the probe reports a stream-required transport
error. An unsupported-model fallback may be useful for compatibility, but it is
not a valid thinking-on experiment.

## Fixed Profiles

The profiles are:

| Profile | Dynamic visual context | Thinking |
|---|---:|---:|
| `fixed_off` | off | off |
| `dynamic_off` | on | off |
| `fixed_on` | off | on |
| `dynamic_on` | on | on |

Each profile pins:

- model `qwen/qwen3.5-flash-2026-02-23`
- transport `sync` and `temperature=0`
- output schema `route_v2`
- thinking budget `1024`
- visual update interval `1`
- floorplan cadence `5`
- motion feedback, forward-stall odometry, and collision overlay enabled
- diagnostic keys `2azQ1b91cZZ:10,11,12,16`

Dynamic profiles do not target, pad to, or truncate at a fixed image count.
The active navigation state and available semantic evidence roles determine the
attachments for each call; when `current` is available it is always attached
last. Fixed profiles retain the legacy image-limit behavior as their baseline.
Confirm this in trace with `model_image_count_policy=role_adaptive` and
`openclaw_model_max_images_applied=false` for dynamic arms.

Run each arm separately so a failed arm cannot contaminate the remaining
processes:

```bash
QWEN_ABLATION_PROFILE=fixed_off \
OPENCLAW_KILL_EXISTING_GATEWAY=1 \
OPENCLAW_GATEWAY_PORT=8013 \
OUTPUT_PATH=results/qwen_2x2_fixed_off \
bash scripts/run_qwen.sh
```

Repeat with `dynamic_off`, `fixed_on`, and `dynamic_on`, changing
`OUTPUT_PATH` to match the profile.

The four-episode set is a diagnostic smoke only. Freeze the larger evaluation
keys before looking at smoke outcomes; do not select the larger set from smoke
successes or failures. Pass the frozen comma-separated set through
`EPISODE_KEYS` when running the larger matrix; the four profile defaults remain
unchanged.

## Compare Arms

Create a manifest with paths relative to the manifest file:

```json
{
  "arms": {
    "fixed_off": {
      "result": "qwen_2x2_fixed_off/result.json",
      "trace": "qwen_2x2_fixed_off/harness_traces/harness_trace_rank0.jsonl"
    },
    "dynamic_off": {
      "result": "qwen_2x2_dynamic_off/result.json",
      "trace": "qwen_2x2_dynamic_off/harness_traces/harness_trace_rank0.jsonl"
    },
    "fixed_on": {
      "result": "qwen_2x2_fixed_on/result.json",
      "trace": "qwen_2x2_fixed_on/harness_traces/harness_trace_rank0.jsonl"
    },
    "dynamic_on": {
      "result": "qwen_2x2_dynamic_on/result.json",
      "trace": "qwen_2x2_dynamic_on/harness_traces/harness_trace_rank0.jsonl"
    }
  }
}
```

Run the comparison:

```bash
PYTHONPATH=src:. python scripts/compare_qwen_direct_policy_results.py \
  --ablation-manifest results/qwen_2x2_manifest.json \
  --output results/qwen_2x2_comparison.json
```

The two primary comparisons are fixed before results are inspected:

- Phase 3: `fixed_off` versus `dynamic_off`
- Phase 4: `dynamic_off` versus `dynamic_on`

`navigation_improvement` requires higher SR. When SR is equal, higher SPL is
reported as `efficiency_improvement_only`. Lower SR is `negative`.

## Invalid Comparisons

The comparator returns exit code 2 and `invalid_comparison` when any arm has
incomplete exact keys, duplicate keys, a mutable or mismatched model ID,
non-synchronous transport, non-`route_v2` output, a treatment-setting mismatch,
invalid JSON, incomplete latency audit, a thinking fallback, transport
incompatibility, or no call with proven thinking exercise in a thinking-on arm.

Reasoning text, raw provider responses, prompts, image payloads, and credentials
must never appear in the comparison artifact. Only bounded audit fields,
reasoning token counts, latency, image roles, actions, gates, and navigation
metrics are included.
