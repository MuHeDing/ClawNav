from __future__ import annotations

import time
from typing import Any, Dict, Optional, Type

import torch


def build_middle_visual_token_mask(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    kv_block_size: int,
    llm_start_blocks: int,
    llm_recent_blocks: int,
) -> torch.Tensor:
    """Return True only for image tokens in the LLM visual-token middle blocks."""

    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch, sequence_length)")

    kv_block_size = max(int(kv_block_size), 1)
    llm_start_blocks = max(int(llm_start_blocks), 0)
    llm_recent_blocks = max(int(llm_recent_blocks), 0)
    middle_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    image_mask = input_ids == int(image_token_id)

    for batch_idx in range(input_ids.shape[0]):
        positions = torch.nonzero(image_mask[batch_idx], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue

        total_visual_blocks = (positions.numel() + kv_block_size - 1) // kv_block_size
        middle_start_block = min(llm_start_blocks, total_visual_blocks)
        middle_end_block = max(middle_start_block, total_visual_blocks - llm_recent_blocks)
        start_offset = middle_start_block * kv_block_size
        end_offset = min(middle_end_block * kv_block_size, positions.numel())
        if start_offset < end_offset:
            middle_mask[batch_idx, positions[start_offset:end_offset]] = True

    return middle_mask


def can_use_adaptive_sparse_attention(
    *,
    query_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    min_seq_len: int,
    training: bool,
    require_cuda: bool = True,
) -> bool:
    if attention_mask is not None:
        return False
    if training:
        return False
    if query_states.dim() != 4:
        return False
    if query_states.shape[0] != 1:
        return False
    if query_states.shape[-2] < int(min_seq_len):
        return False
    if require_cuda and query_states.device.type != "cuda":
        return False
    return True


def _load_spargeattn_classes():
    try:
        from spas_sage_attn.adaptive_attention import AdaptiveSparseAttention
        from spas_sage_attn.mask_strategies import AdaptiveBlockMasker
    except Exception as exc:
        raise ImportError(
            "SpargeAttn adaptive attention is not importable. "
            "Fast_JanusVLN vendors spas_sage_attn under src/spas_sage_attn; "
            "check PYTHONPATH or pass --spargeattn_path to override it."
        ) from exc
    return AdaptiveSparseAttention, AdaptiveBlockMasker


def _default_block_sizes() -> tuple[int, int]:
    try:
        from SageAttention.sageattention.triton.attn_qk_int8_per_block import (
            BLOCK_M,
            BLOCK_N,
        )
    except Exception:
        return 128, 64
    return int(BLOCK_M), int(BLOCK_N)


def install_adaptive_sparse_attention_qwen(
    model: Any,
    *,
    verbose: bool = False,
    pvthreshd: float = 1e6,
    stateless: bool = False,
    mask_kwargs: Optional[Dict[str, Any]] = None,
    adaptive_attention_cls: Optional[Type[Any]] = None,
    masker_cls: Optional[Type[Any]] = None,
) -> Any:
    if adaptive_attention_cls is None or masker_cls is None:
        loaded_attention_cls, loaded_masker_cls = _load_spargeattn_classes()
        adaptive_attention_cls = adaptive_attention_cls or loaded_attention_cls
        masker_cls = masker_cls or loaded_masker_cls

    mask_kwargs = dict(mask_kwargs or {})
    default_q_block, default_kv_block = _default_block_sizes()
    q_block_size = mask_kwargs.pop("q_block_size", default_q_block)
    kv_block_size = mask_kwargs.pop("kv_block_size", default_kv_block)

    decoder = getattr(model, "model", model)
    layers = getattr(decoder, "layers", None)
    if not layers:
        raise ValueError("Qwen decoder layers were not found on model")

    ref_attn = layers[0].self_attn
    masker = masker_cls(
        num_layers=len(layers),
        num_heads=ref_attn.num_heads,
        head_dim=ref_attn.head_dim,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **mask_kwargs,
    )

    decoder.adaptive_masker = masker
    model.adaptive_masker = masker
    for layer_idx, layer in enumerate(layers):
        inner = adaptive_attention_cls(
            masker,
            layer_idx,
            pvthreshd=pvthreshd,
            smooth_k=True,
            stateless=stateless,
        )
        inner.log_sparsity = True
        inner.sparsity_records = []
        inner.prefill_sparsity_records = []
        inner.decode_sparsity_records = []
        inner.kernel_dense_attention_flops_records = []
        inner.kernel_sparse_attention_flops_records = []
        inner.kernel_saved_attention_flops_records = []
        inner.sparse_attention_latency_ms_records = []
        inner.sparse_attention_effective_tops_records = []
        inner.sparse_attention_actual_tops_records = []
        inner.full_attention_latency_ms_records = []
        inner.full_attention_effective_tops_records = []
        layer.self_attn.adaptive_sparse_attention = inner
        layer.self_attn.adaptive_sparse_verbose = bool(verbose)

    return masker


def reset_adaptive_sparse_attention_state(model: Any) -> None:
    masker = getattr(model, "adaptive_masker", None)
    if masker is None and hasattr(model, "model"):
        masker = getattr(model.model, "adaptive_masker", None)
    if masker is not None:
        masker.reset_all()


def _mean_or_none(records: Optional[list[float]]) -> Optional[float]:
    if not records:
        return None
    return float(sum(records) / len(records))


def _sum_records(records: Optional[list[float]]) -> float:
    if not records:
        return 0.0
    return float(sum(records))


def _ratio_or_none(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(float(numerator / denominator), 6)


def _tops_or_none(standard_attention_ops: float, latency_ms: float) -> Optional[float]:
    if latency_ms <= 0:
        return None
    latency_seconds = float(latency_ms) / 1000.0
    return round(float(standard_attention_ops) / latency_seconds / 1e12, 6)


def time_attention_call(fn: Any, device: torch.device) -> tuple[Any, Optional[float]]:
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        result = fn()
        end_event.record()
        torch.cuda.synchronize(device)
        return result, float(start_event.elapsed_time(end_event))

    start_time = time.perf_counter()
    result = fn()
    return result, float((time.perf_counter() - start_time) * 1000.0)


def _latest_kernel_attention_flops(inner: Any) -> Optional[tuple[float, float, float]]:
    dense_records = getattr(inner, "kernel_dense_attention_flops_records", None)
    sparse_records = getattr(inner, "kernel_sparse_attention_flops_records", None)
    saved_records = getattr(inner, "kernel_saved_attention_flops_records", None)
    if not dense_records or not sparse_records or not saved_records:
        return None
    return (
        float(dense_records[-1]),
        float(sparse_records[-1]),
        float(saved_records[-1]),
    )


def record_adaptive_sparse_flops(
    inner: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    config: Optional[Any] = None,
    sparse_latency_ms: Optional[float] = None,
    full_attention_latency_ms: Optional[float] = None,
) -> None:
    sparsity_records = getattr(inner, "sparsity_records", None)
    if not sparsity_records:
        return

    latest_sparsity = float(sparsity_records[-1])
    batch, heads, q_len, head_dim = query_states.shape
    kv_len = int(key_states.shape[-2])
    kernel_flops = _latest_kernel_attention_flops(inner)
    if kernel_flops is not None:
        dense_flops, sparse_flops, saved_flops = kernel_flops
    else:
        dense_flops = float(4.0 * batch * heads * int(q_len) * kv_len * int(head_dim))
        saved_flops = dense_flops * latest_sparsity
        sparse_flops = dense_flops - saved_flops

    if not hasattr(inner, "dense_attention_flops_records"):
        inner.dense_attention_flops_records = []
    if not hasattr(inner, "saved_attention_flops_records"):
        inner.saved_attention_flops_records = []
    if not hasattr(inner, "sparse_attention_flops_records"):
        inner.sparse_attention_flops_records = []

    inner.dense_attention_flops_records.append(dense_flops)
    inner.saved_attention_flops_records.append(saved_flops)
    inner.sparse_attention_flops_records.append(sparse_flops)

    # Latency/TOPS profiling is temporarily disabled. Keep this code block for
    # quick restoration when speed metrics are needed again.
    # if sparse_latency_ms is not None:
    #     if not hasattr(inner, "sparse_attention_latency_ms_records"):
    #         inner.sparse_attention_latency_ms_records = []
    #     if not hasattr(inner, "sparse_attention_effective_tops_records"):
    #         inner.sparse_attention_effective_tops_records = []
    #     if not hasattr(inner, "sparse_attention_actual_tops_records"):
    #         inner.sparse_attention_actual_tops_records = []
    #     sparse_latency_ms = float(sparse_latency_ms)
    #     inner.sparse_attention_latency_ms_records.append(sparse_latency_ms)
    #     sparse_effective_tops = _tops_or_none(dense_flops, sparse_latency_ms)
    #     if sparse_effective_tops is not None:
    #         inner.sparse_attention_effective_tops_records.append(sparse_effective_tops)
    #     sparse_actual_tops = _tops_or_none(sparse_flops, sparse_latency_ms)
    #     if sparse_actual_tops is not None:
    #         inner.sparse_attention_actual_tops_records.append(sparse_actual_tops)
    #
    # if full_attention_latency_ms is not None:
    #     if not hasattr(inner, "full_attention_latency_ms_records"):
    #         inner.full_attention_latency_ms_records = []
    #     if not hasattr(inner, "full_attention_effective_tops_records"):
    #         inner.full_attention_effective_tops_records = []
    #     full_attention_latency_ms = float(full_attention_latency_ms)
    #     inner.full_attention_latency_ms_records.append(full_attention_latency_ms)
    #     full_attention_effective_tops = _tops_or_none(dense_flops, full_attention_latency_ms)
    #     if full_attention_effective_tops is not None:
    #         inner.full_attention_effective_tops_records.append(full_attention_effective_tops)

    # LLM-layer and total-LLM FLOPs estimates are intentionally disabled.
    # Keep the attention-only FLOPs above; do not mix them with projection/MLP
    # estimates in the runtime sparse-attention report.


def summarize_adaptive_sparsity(model: Any) -> Dict[str, Any]:
    decoder = getattr(model, "model", model)
    layers = getattr(decoder, "layers", [])
    per_layer = []
    prefill_layer_means = []
    decode_layer_means = []
    total_dense_flops = 0.0
    total_sparse_flops = 0.0
    total_saved_flops = 0.0
    total_sparse_latency_ms = 0.0
    total_full_attention_latency_ms = None
    total_dense_llm_layer_flops = 0.0
    total_sparse_llm_layer_flops = 0.0
    total_saved_llm_layer_flops = 0.0
    for layer_idx, layer in enumerate(layers):
        inner = getattr(layer.self_attn, "adaptive_sparse_attention", None)
        records = getattr(inner, "sparsity_records", None)
        if not records:
            continue
        prefill_records = getattr(inner, "prefill_sparsity_records", None)
        decode_records = getattr(inner, "decode_sparsity_records", None)
        prefill_mean = _mean_or_none(prefill_records)
        decode_mean = _mean_or_none(decode_records)
        if prefill_mean is not None:
            prefill_layer_means.append(prefill_mean)
        if decode_mean is not None:
            decode_layer_means.append(decode_mean)

        dense_flops = _sum_records(getattr(inner, "dense_attention_flops_records", None))
        sparse_flops = _sum_records(getattr(inner, "sparse_attention_flops_records", None))
        saved_flops = _sum_records(getattr(inner, "saved_attention_flops_records", None))
        total_dense_flops += dense_flops
        total_sparse_flops += sparse_flops
        total_saved_flops += saved_flops
        sparse_latency_ms = _sum_records(getattr(inner, "sparse_attention_latency_ms_records", None))
        full_attention_latency_records = getattr(inner, "full_attention_latency_ms_records", None)
        full_attention_latency_ms = _sum_records(full_attention_latency_records) if full_attention_latency_records else None
        total_sparse_latency_ms += sparse_latency_ms
        if full_attention_latency_ms is not None:
            total_full_attention_latency_ms = (
                full_attention_latency_ms
                if total_full_attention_latency_ms is None
                else total_full_attention_latency_ms + full_attention_latency_ms
            )
        sparse_effective_tops = _tops_or_none(dense_flops, sparse_latency_ms) if sparse_latency_ms > 0 else None
        sparse_actual_tops = _tops_or_none(sparse_flops, sparse_latency_ms) if sparse_latency_ms > 0 else None
        full_attention_effective_tops = (
            _tops_or_none(dense_flops, full_attention_latency_ms)
            if full_attention_latency_ms is not None and full_attention_latency_ms > 0
            else None
        )
        dense_llm_layer_flops = 0.0
        sparse_llm_layer_flops = 0.0
        saved_llm_layer_flops = 0.0
        self_attention_reduction = _ratio_or_none(saved_flops, dense_flops)
        llm_layer_reduction = None

        per_layer.append(
            {
                "layer": layer_idx,
                "mean_sparsity": float(sum(records) / len(records)),
                "count": len(records),
                "mean_prefill_sparsity": prefill_mean,
                "prefill_count": len(prefill_records) if prefill_records else 0,
                "mean_decode_sparsity": decode_mean,
                "decode_count": len(decode_records) if decode_records else 0,
                "dense_attention_flops": dense_flops,
                "sparse_attention_flops": sparse_flops,
                "saved_attention_flops": saved_flops,
                "flops_reduction_ratio": self_attention_reduction,
                "self_attention_flops_reduction_ratio": self_attention_reduction,
                "sparse_attention_latency_ms": sparse_latency_ms,
                "sparse_attention_effective_tops": sparse_effective_tops,
                "sparse_attention_actual_tops": sparse_actual_tops,
                "full_attention_latency_ms": full_attention_latency_ms,
                "full_attention_effective_tops": full_attention_effective_tops,
                "dense_llm_layer_flops": dense_llm_layer_flops,
                "sparse_llm_layer_flops": sparse_llm_layer_flops,
                "saved_llm_layer_flops": saved_llm_layer_flops,
                "llm_layer_flops_reduction_ratio": llm_layer_reduction,
            }
        )

    mean_sparsity = None
    if per_layer:
        mean_sparsity = float(
            sum(record["mean_sparsity"] for record in per_layer) / len(per_layer)
        )
    self_attention_reduction = _ratio_or_none(total_saved_flops, total_dense_flops)
    sparse_effective_tops = (
        _tops_or_none(total_dense_flops, total_sparse_latency_ms)
        if total_sparse_latency_ms > 0
        else None
    )
    sparse_actual_tops = (
        _tops_or_none(total_sparse_flops, total_sparse_latency_ms)
        if total_sparse_latency_ms > 0
        else None
    )
    full_attention_effective_tops = (
        _tops_or_none(total_dense_flops, total_full_attention_latency_ms)
        if total_full_attention_latency_ms is not None and total_full_attention_latency_ms > 0
        else None
    )
    llm_layer_reduction = None
    instrumented_sparse_layers = len(per_layer)
    estimated_total_llm_num_layers = len(layers)
    estimated_total_llm_flops = 0.0
    estimated_total_llm_saved_flops = 0.0
    estimated_total_llm_sparse_flops = 0.0
    estimated_total_llm_reduction = None
    # Total-LLM FLOPs estimates are disabled with llm_layer_flops_reduction.
    return {
        "mean_sparsity": mean_sparsity,
        "mean_prefill_sparsity": _mean_or_none(prefill_layer_means),
        "mean_decode_sparsity": _mean_or_none(decode_layer_means),
        "dense_attention_flops": total_dense_flops,
        "sparse_attention_flops": total_sparse_flops,
        "saved_attention_flops": total_saved_flops,
        "flops_reduction_ratio": self_attention_reduction,
        "self_attention_flops_reduction_ratio": self_attention_reduction,
        "sparse_attention_latency_ms": total_sparse_latency_ms,
        "sparse_attention_effective_tops": sparse_effective_tops,
        "sparse_attention_actual_tops": sparse_actual_tops,
        "full_attention_latency_ms": total_full_attention_latency_ms,
        "full_attention_effective_tops": full_attention_effective_tops,
        "dense_llm_layer_flops": total_dense_llm_layer_flops,
        "sparse_llm_layer_flops": total_sparse_llm_layer_flops,
        "saved_llm_layer_flops": total_saved_llm_layer_flops,
        "llm_layer_flops_reduction_ratio": llm_layer_reduction,
        "estimated_total_llm_num_layers": estimated_total_llm_num_layers,
        "instrumented_sparse_layers": instrumented_sparse_layers,
        "estimated_total_llm_flops": estimated_total_llm_flops,
        "estimated_total_llm_saved_flops": estimated_total_llm_saved_flops,
        "estimated_total_llm_sparse_flops": estimated_total_llm_sparse_flops,
        "estimated_total_llm_flops_reduction_ratio": estimated_total_llm_reduction,
        "layers": per_layer,
    }


ADAPTIVE_SPARSITY_SUMMARY_KEYS = (
    "mean_sparsity",
    "mean_prefill_sparsity",
    "mean_decode_sparsity",
    "dense_attention_flops",
    "sparse_attention_flops",
    "saved_attention_flops",
    "flops_reduction_ratio",
    "self_attention_flops_reduction_ratio",
    "sparse_attention_latency_ms",
    "sparse_attention_effective_tops",
    "sparse_attention_actual_tops",
    "full_attention_latency_ms",
    "full_attention_effective_tops",
    "estimated_total_llm_num_layers",
    "instrumented_sparse_layers",
)

ADAPTIVE_SPARSITY_LAYER_KEYS = (
    "layer",
    "mean_sparsity",
    "count",
    "mean_prefill_sparsity",
    "prefill_count",
    "mean_decode_sparsity",
    "decode_count",
    "flops_reduction_ratio",
    "self_attention_flops_reduction_ratio",
)


def compact_adaptive_sparsity_summary(
    summary: Dict[str, Any],
    *,
    include_layers: bool = False,
) -> Dict[str, Any]:
    """Return a small, stable summary for persisted sparse-attention profiles."""

    compact = {
        key: summary.get(key)
        for key in ADAPTIVE_SPARSITY_SUMMARY_KEYS
        if key in summary
    }
    if include_layers:
        compact["layers"] = [
            {
                key: layer_summary.get(key)
                for key in ADAPTIVE_SPARSITY_LAYER_KEYS
                if key in layer_summary
            }
            for layer_summary in summary.get("layers", [])
        ]
    else:
        compact["layer_count"] = len(summary.get("layers", []))
    return compact
