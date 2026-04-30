"""High-level module wrapper for adaptive sparse attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .mask_strategies import AdaptiveBlockMasker
from .core import adaptive_block_sparse_attn_step, StageSparsity


class AdaptiveSparseAttention(nn.Module):
    """nn.Module drop-in that routes attention through the adaptive masker."""

    def __init__(
        self,
        masker: AdaptiveBlockMasker,
        layer_idx: int,
        *,
        pvthreshd: float = 50.0,
        smooth_k: bool = True,
        stateless: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(masker, AdaptiveBlockMasker):
            raise TypeError("masker must be an AdaptiveBlockMasker")
        self.masker = masker
        self.layer_idx = int(layer_idx)
        self.pvthreshd = float(pvthreshd)
        self.smooth_k = bool(smooth_k)
        self.stateless = bool(stateless)
        self.log_sparsity = False
        self.sparsity_records = []
        self.prefill_sparsity_records = []
        self.decode_sparsity_records = []
        self.kernel_dense_attention_flops_records = []
        self.kernel_sparse_attention_flops_records = []
        self.kernel_saved_attention_flops_records = []
        self.sparse_attention_latency_ms_records = []
        self.sparse_attention_effective_tops_records = []
        self.sparse_attention_actual_tops_records = []
        self.full_attention_latency_ms_records = []
        self.full_attention_effective_tops_records = []

    def reset_state(self) -> None:
        """Reset cached statistics for the bound layer."""

        self.masker.reset_layer(self.layer_idx)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        is_causal: bool = False,
        scale: Optional[float] = None,
        tensor_layout: str = "HND",
        tune_mode: bool = False,
        smooth_k: Optional[bool] = None,
        return_sparsity: bool = False,
        sparse_token_mask: Optional[torch.Tensor] = None,
    ):
        if mask is not None:
            raise NotImplementedError("Explicit attention masks are not supported")
        if tune_mode:
            raise NotImplementedError("Autotune is not implemented for adaptive masking")

        if self.stateless:
            self.reset_state()

        smooth_flag = self.smooth_k if smooth_k is None else smooth_k
        record_sparsity = self.log_sparsity or return_sparsity

        def log_stats(stats: StageSparsity) -> None:
            if not self.log_sparsity:
                return
            latest = None
            if stats.prefill is not None:
                latest = float(stats.prefill)
                self.prefill_sparsity_records.append(latest)
            if stats.decode is not None:
                latest = float(stats.decode)
                self.decode_sparsity_records.append(latest)
            if latest is not None:
                self.sparsity_records.append(latest)
            if stats.dense_attention_flops is not None:
                self.kernel_dense_attention_flops_records.append(float(stats.dense_attention_flops))
            if stats.sparse_attention_flops is not None:
                self.kernel_sparse_attention_flops_records.append(float(stats.sparse_attention_flops))
            if stats.saved_attention_flops is not None:
                self.kernel_saved_attention_flops_records.append(float(stats.saved_attention_flops))

        result = adaptive_block_sparse_attn_step(
            self.masker,
            self.layer_idx,
            q,
            k,
            v,
            is_causal=is_causal,
            scale=scale,
            smooth_k=smooth_flag,
            pvthreshd=self.pvthreshd,
            tensor_layout=tensor_layout,
            return_sparsity=record_sparsity,
            sparse_token_mask=sparse_token_mask,
        )

        if record_sparsity:
            attn_out, stats = result
            log_stats(stats)
            if return_sparsity:
                return attn_out, stats
            return attn_out

        return result
