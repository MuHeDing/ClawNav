# Type-aware Mid-Layer Visual Token Pruning v1

## Goal

This v1 adds a minimal type-aware, instruction-guided visual token pruning path on top of the JanusVLN Qwen2.5-VL backbone.

The target is to compress working visual tokens inside the LLM, not to replace or restructure the external memory system.

## Scope

Included in v1:

- Keep fused 2D + 3D image tokens as the visual working set entering the LLM.
- Maintain `visual_token_mask` and `visual_token_types` for fused image-token positions.
- Run the first several decoder layers normally.
- Insert one pruning point in the decoder loop.
- Score visual tokens with `MLP([h_i_vis ; h_instr ; type_emb])`.
- Keep tokens with type-aware top-k ratios:
  - semantic-heavy: more aggressive
  - spatial-heavy: more conservative
  - mixed: intermediate
- Rebuild a shorter sequence so later decoder layers see fewer tokens.
- Add evaluation-time profiling logs for:
  - visual token count before/after pruning
  - per-layer latency
  - peak GPU memory

Not included in v1:

- full/pruned dual-branch consistency
- teacher branch
- SH-Fuser-style persistent dual streams
- memory-bank or recall-buffer pruning
- dataset pipeline changes
- cache-aware generation-time pruning with heterogeneous per-layer KV lengths

## Code Path

### Visual tokens entering the LLM

In `src/qwen_vl/model/modeling_qwen2_5_vl.py`:

- `self.visual(...)` produces 2D image tokens.
- `self.merger(...)` produces 3D spatial tokens from VGGT features.
- Fused visual tokens are built as:

```python
image_embeds = image_embeds + self.lam * image_embeds_3d
```

- These fused tokens are inserted into `inputs_embeds` via `masked_scatter`.

At this stage, v1 also builds:

- `visual_token_mask`
- `visual_token_types`

### Token type heuristic

v1 uses a simple and explainable heuristic for fused image tokens:

- `semantic-heavy` if the 2D branch dominates
- `spatial-heavy` if the scaled 3D branch dominates
- `mixed` otherwise

The current dominance test is based on branch feature norms with a small margin.

## Pruning Module

File:

- `src/qwen_vl/model/llm_visual_pruner.py`

Main pieces:

- `TypeAwareVisualTokenPruner`
- `type_aware_topk_keep_indices()`
- `rebuild_pruned_sequence()`
- `should_apply_visual_prune()`

### Scoring

For each visual token:

```python
score_i = MLP([h_i_vis ; h_instr ; type_emb])
```

Where:

- `h_i_vis`: current-layer visual token hidden state
- `h_instr`: text summary from non-visual tokens
- `type_emb`: embedding of token type

### Keep policy

For each type group independently:

- `k_sem = ceil(num_sem * sem_keep_ratio)`
- `k_spa = ceil(num_spa * spa_keep_ratio)`
- `k_mix = ceil(num_mix * mix_keep_ratio)`

Then selected tokens are merged back in original order with all text tokens preserved.

## Decoder Integration

Pruning is inserted inside `Qwen2_5_VLModel.forward()` in the decoder layer loop.

When `layer_idx == visual_prune_layer`:

1. collect current visual hidden states
2. score them with instruction guidance and type embeddings
3. perform type-aware top-k
4. rebuild the sequence
5. update:
   - `hidden_states`
   - `attention_mask`
   - `position_ids`
   - `visual_token_mask`
   - `visual_token_types`
6. continue later decoder layers on the shorter sequence

This is a real sequence-length reduction, not a mask-only ablation.

## Default Hyperparameters

Current v1 defaults:

- `use_llm_visual_prune = True`
- `visual_prune_layer = 4`
- `sem_keep_ratio = 0.4`
- `spa_keep_ratio = 0.7`
- `mix_keep_ratio = 0.55`

Rationale:

- prune after a few layers so early text-vision alignment can happen first
- prune semantic-heavy tokens harder because they are more redundant after early alignment
- keep spatial-heavy tokens more conservatively because navigation relies on geometry/layout cues
- keep mixed tokens in between as a hedge

## Training Path

Added training args in:

- `src/qwen_vl/train/argument.py`

Config propagation is handled in:

- `src/qwen_vl/train/train_qwen.py`

Train scripts updated:

- `scripts/train.sh`
- `scripts/train_extra.sh`

The v1 loss remains the original action loss.

## Evaluation Profiling

Files:

- `src/evaluation.py`
- `src/evaluation_rxr.py`

### Why profiling uses an auxiliary no-cache forward

The current generation stack uses `generate(use_cache=True)`.

True mid-layer pruning during cached generation would produce different KV lengths across early and late layers, while the standard HF cache flow largely assumes a consistent cache length when computing `cache_position` and related logic.

To avoid changing generation semantics in this v1, evaluation profiling does:

1. an auxiliary `use_cache=False` forward pass to collect pruning and layer-profile statistics
2. the original `generate(...)` call for the actual action output

This gives reliable pruning/timing/memory logs for analysis, while keeping generation behavior stable.

### Logged profiling fields

Per profiled call:

- prefill sequence length
- prefill visual token count
- prune applied or not
- prune layer
- per-layer:
  - `layer_idx`
  - `elapsed_ms`
  - `sequence_length`
  - `max_memory_allocated_mb`
- total forward time
- peak memory allocated
- generate time
- per-sample prune counts

Output file:

- `visual_prune_eval_profile_rank{rank}.jsonl`

## Verification

Current lightweight verification:

- unit tests for type-aware top-k
- unit tests for pruned sequence rebuilding
- unit tests for prefill/decode prune gating helper
- Python compilation checks on modified source files

## Known Limitations

- eval profiling currently measures an auxiliary no-cache forward, not the cached generation path
- generation-time cache-aware pruning is intentionally left out in this v1
- token typing currently covers fused image tokens, not a richer per-source token provenance graph
- video-token pruning is not implemented
- no consistency loss or teacher branch
