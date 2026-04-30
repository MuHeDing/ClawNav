"""Adaptive block mask strategy for per-head sparse attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os
import torch


@dataclass
class LayerState:
    """Per-layer state shared across heads."""

    mu: torch.Tensor  # (H, B, D) raw block means of K
    vbar: torch.Tensor  # (H, B, D) block means of V
    gamma: torch.Tensor  # (H, B) per-block variance estimate
    token_count: torch.Tensor  # (H, B)
    g_prev: torch.Tensor  # (H, B)
    y_hat: torch.Tensor  # (H, D)
    y_hat_correction: torch.Tensor  # (H,)
    drop_mass: torch.Tensor  # (H,)
    lam: torch.Tensor  # (H,)
    tau: torch.Tensor  # (H,)
    tail_k_sum: torch.Tensor  # (H, D)
    tail_v_sum: torch.Tensor  # (H, D)
    tail_k_sq_sum: torch.Tensor  # (H, D)
    tail_token_count: torch.Tensor  # (H,)
    step_counter: int = 0
    processed_tokens: int = 0


class AdaptiveBlockMasker:
    """Generates head-wise block masks using adaptive statistics."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        q_block_size: int,
        kv_block_size: int,
        recent_window_len: int = 8,
        target_blocks: float = 9.0,
        target_drop_mass: float = 0.05,
        prefill_min_blocks: int = 1,
        tau_init: float = 1.0,
        lam_init: float = 0.05,
        mu_reg: float = 0.1,
        beta: float = 0.5,
        lr_lambda: float = 0.01,
        lr_tau: float = 0.02,
        epsilon: float = 1e-6,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.num_layers = num_layers
        if q_block_size <= 0 or kv_block_size <= 0:
            raise ValueError("block sizes must be positive")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_block_size = int(q_block_size)
        self.kv_block_size = int(kv_block_size)
        self.inv_sqrt_dim = 1.0 / float(self.head_dim) ** 0.5
        self.recent_window_len = recent_window_len
        self.target_blocks = target_blocks
        self.target_drop_mass = target_drop_mass
        self.prefill_min_blocks = max(int(prefill_min_blocks), 0)
        self.tau_init = max(float(tau_init), epsilon)
        self.lam_init = max(float(lam_init), epsilon)
        self.mu_reg = mu_reg
        self.beta = beta
        self.lr_lambda = lr_lambda
        self.lr_tau = lr_tau
        self.epsilon = epsilon

        self.layers: Dict[int, LayerState] = {
            layer: self._init_layer_state()
            for layer in range(num_layers)
        }

    def _layer_state(self, layer: int) -> LayerState:
        return self.layers[layer]

    def _init_layer_state(self) -> LayerState:
        zeros_vec = torch.zeros((self.num_heads, 0, self.head_dim), device=self.device, dtype=torch.float32)
        zeros = torch.zeros((self.num_heads, 0), device=self.device, dtype=torch.float32)
        return LayerState(
            mu=zeros_vec.clone(),
            vbar=zeros_vec.clone(),
            gamma=zeros.clone(),
            token_count=zeros.clone(),
            g_prev=zeros.clone(),
            y_hat=torch.zeros((self.num_heads, self.head_dim), device=self.device, dtype=torch.float32),
            y_hat_correction=torch.zeros((self.num_heads,), device=self.device, dtype=torch.float32),
            drop_mass=torch.zeros((self.num_heads,), device=self.device, dtype=torch.float32),
            lam=torch.full((self.num_heads,), float(self.lam_init), device=self.device, dtype=torch.float32),
            tau=torch.full((self.num_heads,), float(self.tau_init), device=self.device, dtype=torch.float32),
            tail_k_sum=torch.zeros((self.num_heads, self.head_dim), device=self.device, dtype=torch.float32),
            tail_v_sum=torch.zeros((self.num_heads, self.head_dim), device=self.device, dtype=torch.float32),
            tail_k_sq_sum=torch.zeros((self.num_heads, self.head_dim), device=self.device, dtype=torch.float32),
            tail_token_count=torch.zeros((self.num_heads,), device=self.device, dtype=torch.float32),
        )

    def reset_layer(self, layer: int) -> None:
        self.layers[layer] = self._init_layer_state()

    def reset_all(self) -> None:
        for layer in self.layers:
            self.layers[layer] = self._init_layer_state()

    # ---------------------------------------------------------------------
    # Incremental statistics maintenance
    # ---------------------------------------------------------------------
    def _accumulate_tail(
        self,
        state: LayerState,
        chunk_k: torch.Tensor,
        chunk_v: torch.Tensor,
    ) -> None:
        if chunk_k.numel() == 0:
            return

        state.tail_k_sum.add_(chunk_k.sum(dim=1))
        state.tail_v_sum.add_(chunk_v.sum(dim=1))
        state.tail_k_sq_sum.add_(chunk_k.square().sum(dim=1))
        state.tail_token_count.add_(float(chunk_k.size(1)))

        if state.tail_token_count[0].item() >= float(self.kv_block_size):
            self._finalize_tail_block(state)

    def _finalize_tail_block(self, state: LayerState) -> None:
        tail_tokens = int(state.tail_token_count[0].item())
        if tail_tokens < self.kv_block_size:
            return

        block_width = float(self.kv_block_size)

        mu_vec = state.tail_k_sum / block_width
        vbar_vec = state.tail_v_sum / block_width
        mean_sq = state.tail_k_sq_sum / block_width
        var_block = (mean_sq - mu_vec.pow(2)).clamp_min(0.0)
        gamma_block = var_block.mean(dim=-1, keepdim=True)

        mu_block = mu_vec.unsqueeze(1)
        vbar_block = vbar_vec.unsqueeze(1)

        token_block = torch.full(
            (self.num_heads, 1),
            block_width,
            device=self.device,
            dtype=torch.float32,
        )
        zeros_block = torch.zeros((self.num_heads, 1), device=self.device, dtype=torch.float32)

        state.mu = torch.cat([state.mu, mu_block], dim=1)
        state.vbar = torch.cat([state.vbar, vbar_block], dim=1)
        state.gamma = torch.cat([state.gamma, gamma_block], dim=1)
        state.token_count = torch.cat([state.token_count, token_block], dim=1)
        state.g_prev = torch.cat([state.g_prev, zeros_block], dim=1)

        state.tail_k_sum.zero_()
        state.tail_v_sum.zero_()
        state.tail_k_sq_sum.zero_()
        state.tail_token_count.zero_()

    def _apply_chunks(
        self,
        state: LayerState,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
    ) -> None:
        if k_seq.numel() == 0:
            return
        block_width = self.kv_block_size

        if state.tail_token_count[0].item() >= float(block_width):
            self._finalize_tail_block(state)

        tail_tokens = int(state.tail_token_count[0].item())
        if 0 < tail_tokens < block_width:
            tail_space = block_width - tail_tokens
            take = min(tail_space, k_seq.size(1))
            if take > 0:
                self._accumulate_tail(state, k_seq[:, :take, :], v_seq[:, :take, :])
                k_seq = k_seq[:, take:, :]
                v_seq = v_seq[:, take:, :]

        new_len = k_seq.size(1)
        if new_len >= block_width:
            num_full = new_len // block_width
            if num_full > 0:
                take = num_full * block_width
                k_full = k_seq[:, :take, :].reshape(self.num_heads, num_full, block_width, self.head_dim)
                v_full = v_seq[:, :take, :].reshape(self.num_heads, num_full, block_width, self.head_dim)

                k_sum = k_full.sum(dim=2)
                v_sum = v_full.sum(dim=2)
                sum_sq = k_full.square().sum(dim=2)

                mu_new = k_sum / float(block_width)
                vbar_new = v_sum / float(block_width)
                mean_sq_new = sum_sq / float(block_width)
                var_new = (mean_sq_new - mu_new.pow(2)).clamp_min(0.0)
                gamma_new = var_new.mean(dim=-1)

                token_new = torch.full(
                    (self.num_heads, num_full),
                    float(block_width),
                    device=self.device,
                    dtype=torch.float32,
                )
                zeros_new = torch.zeros((self.num_heads, num_full), device=self.device, dtype=torch.float32)

                state.mu = torch.cat([state.mu, mu_new], dim=1)
                state.vbar = torch.cat([state.vbar, vbar_new], dim=1)
                state.gamma = torch.cat([state.gamma, gamma_new], dim=1)
                state.token_count = torch.cat([state.token_count, token_new], dim=1)
                state.g_prev = torch.cat([state.g_prev, zeros_new], dim=1)

                k_seq = k_seq[:, take:, :]
                v_seq = v_seq[:, take:, :]

        if k_seq.size(1) > 0:
            self._accumulate_tail(state, k_seq, v_seq)

    def ingest_kv(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        tensor_layout: str = "HND",
    ) -> None:
        """Bulk-ingest an existing KV cache (prefill).

        Parameters
        ----------
        layer : int
            Layer index whose statistics should absorb this cache.
        k, v : torch.Tensor
            KV tensors shaped ``(batch, heads, seq, dim)`` ("HND") or
            ``(batch, seq, heads, dim)`` ("NHD"). Currently only ``batch=1``
            is supported. The routine iterates from ``state.processed_tokens``
            onward and skips work if the cache length has not increased.
        tensor_layout : str
            Layout flag describing ``k``/``v``. Either "HND" or "NHD".
        """
        assert tensor_layout in ["HND", "NHD"], "Unsupported tensor layout"
        if tensor_layout == "NHD":
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()

        if k.dim() != 4 or v.dim() != 4:
            raise ValueError("k and v must have shape (batch, heads, seq, dim)")
        if k.size(0) != 1 or v.size(0) != 1:
            raise NotImplementedError("Only batch size 1 is currently supported for adaptive masking")

        state = self._layer_state(layer)
        debug_masks = os.environ.get("ADAPTIVE_DEBUG", "0") == "1"
        total_tokens = k.size(2)
        if total_tokens <= state.processed_tokens:
            return

        k_seq = k[0].to(self.device, dtype=torch.float32)
        v_seq = v[0].to(self.device, dtype=torch.float32)

        start_idx = state.processed_tokens
        if start_idx >= total_tokens:
            return

        new_k = k_seq[:, start_idx:total_tokens, :]
        new_v = v_seq[:, start_idx:total_tokens, :]
        self._apply_chunks(state, new_k, new_v)

        state.processed_tokens = total_tokens

    def update_tail_statistics(
        self,
        layer: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        tensor_layout: str = "HND",
    ) -> None:
        """Incrementally update the tail block with newly produced tokens.

        Parameters
        ----------
        layer : int
            Target layer whose tail block should absorb the new tokens.
        k_new, v_new : torch.Tensor
            Incremental KV chunk. Accepted shapes:
              * ``(num_heads, head_dim)`` (single token)
              * ``(num_heads, tokens, head_dim)``
              * ``(1, num_heads, tokens, head_dim)``
              * ``(1, tokens, num_heads, head_dim)`` when ``tensor_layout`` is
                "NHD".
        tensor_layout : str
            Layout flag used when rank is 4. Either "HND" or "NHD".
        """
        state = self._layer_state(layer)
        assert tensor_layout in ["HND", "NHD"], "Unsupported tensor layout"

        if k_new.dim() == 4:
            if tensor_layout == "NHD":
                k_new = k_new.permute(0, 2, 1, 3)
                v_new = v_new.permute(0, 2, 1, 3)
            if k_new.size(0) != 1:
                raise NotImplementedError("Only batch size 1 is currently supported for adaptive masking")
            k_seq = k_new[0]
            v_seq = v_new[0]
        elif k_new.dim() == 3:
            k_seq = k_new
            v_seq = v_new
        elif k_new.dim() == 2:
            k_seq = k_new.unsqueeze(1)
            v_seq = v_new.unsqueeze(1)
        else:
            raise ValueError("k_new must have shape compatible with (num_heads, tokens, head_dim)")

        k_seq = k_seq.to(self.device, dtype=torch.float32)
        v_seq = v_seq.to(self.device, dtype=torch.float32)
        self._apply_chunks(state, k_seq, v_seq)
        state.processed_tokens += k_seq.size(1)

    # ---------------------------------------------------------------------
    # Mask computation & feedback
    # ---------------------------------------------------------------------
    def compute_mask(
        self,
        layer: int,
        q: torch.Tensor,
        tensor_layout: str = "HND",
        prefill: bool = False,
    ) -> torch.Tensor:
        """Produce the boolean block mask for a given layer/head set.

        Parameters
        ----------
        layer : int
            Target layer whose mask should be generated.
        q : torch.Tensor
            Query summary. Accepted shapes include ``(num_heads, head_dim)``,
            ``(1, num_heads, head_dim)``, ``(num_heads, tokens, head_dim)`` and
            four-dimensional ``(batch, num_heads, tokens, head_dim)`` or
            ``(batch, tokens, num_heads, head_dim)`` (the latter when
            ``tensor_layout`` is ``"NHD"``). Higher-rank tensors are reduced
            along the sequence dimension via mean.
        tensor_layout : str
            Layout hint used when ``q`` is four-dimensional. Either "HND" or
            "NHD".
        prefill : bool
            When ``True`` the mask falls back to the mean-similarity-only rule
            without using ``phi`` or updating lambda/tau feedback terms.

        Returns
        -------
        torch.Tensor
            Boolean mask. Shape is ``(num_heads, num_k_blocks)`` for a single
            query summary, or ``(num_heads, num_q_blocks, num_k_blocks)`` when
            ``prefill`` is ``True`` and multiple query blocks are supplied.
        """
        state = self._layer_state(layer)
        debug_masks = os.environ.get("ADAPTIVE_DEBUG", "0") == "1"

        multi_query = False
        if prefill and q.dim() == 3 and q.size(0) == self.num_heads:
            q_vec = q.to(self.device, dtype=torch.float32)
            multi_query = True
        elif q.dim() == 4:
            if tensor_layout == "NHD":
                q = q.permute(0, 2, 1, 3)
            q_vec = q.mean(dim=2).squeeze(0)
        elif q.dim() == 3:
            if q.size(0) == 1:
                q_vec = q.squeeze(0)
            else:
                q_vec = q.to(self.device, dtype=torch.float32)
                multi_query = True
        elif q.dim() == 2:
            q_vec = q
        else:
            raise ValueError("q must have shape compatible with (num_heads, head_dim)")

        if multi_query and q_vec.dim() != 3:
            raise ValueError("When providing multiple query blocks, q must have shape (num_heads, num_blocks, head_dim)")

        q_vec = q_vec.to(self.device, dtype=torch.float32)

        num_blocks = state.mu.size(1)
        tail_tokens = int(state.tail_token_count[0].item())
        tail_present = tail_tokens > 0
        total_blocks = num_blocks + (1 if tail_present else 0)

        if total_blocks == 0:
            state.drop_mass.zero_()
            if multi_query:
                return torch.ones((self.num_heads, q_vec.size(1), 1), device=self.device, dtype=torch.bool)
            return torch.ones((self.num_heads, 1), device=self.device, dtype=torch.bool)

        recent_total = min(self.recent_window_len, total_blocks)
        recent_tail = 1 if tail_present and recent_total > 0 else 0
        recent_completed = min(num_blocks, max(recent_total - recent_tail, 0))
        nr_end = num_blocks - recent_completed

        if multi_query:
            num_q = q_vec.size(1)
            mask = torch.zeros((self.num_heads, num_q, total_blocks), device=self.device, dtype=torch.bool)
            if tail_present:
                mask[:, :, -1].fill_(True)
            if recent_completed > 0:
                start_idx = num_blocks - recent_completed
                mask[:, :, start_idx:num_blocks].fill_(True)
                state.g_prev[:, start_idx:].fill_(1.0)
            if nr_end <= 0:
                state.drop_mass.zero_()
                return mask

            q_vec = q_vec.to(torch.float32).unsqueeze(1)
            q_rms = torch.sqrt((q_vec * q_vec).mean(dim=-1, keepdim=True) + self.epsilon)
            q_normed = (q_vec / q_rms).squeeze(1)

            mu_blocks = state.mu[:, :nr_end, :].to(torch.float32)
            mu_rms = torch.sqrt((mu_blocks * mu_blocks).mean(dim=-1, keepdim=True) + self.epsilon)
            mu_normed = mu_blocks / mu_rms

            gamma_nr = state.gamma[:, :nr_end].to(torch.float32)

            tau = state.tau.clamp_min(self.epsilon).to(torch.float32)

            scale = self.inv_sqrt_dim
            q_norm2 = (q_normed * q_normed).sum(dim=-1) * (scale * scale)
            logits = torch.einsum('hqd,hkd->hqk', q_normed, mu_normed) * scale
            logits = logits / tau.view(-1, 1, 1)
            tau_sq = (tau * tau).view(-1, 1, 1)
            logits = logits + gamma_nr.unsqueeze(1) * q_norm2.unsqueeze(-1) / (2.0 * tau_sq)

            m_tilde = torch.softmax(logits, dim=-1)

            drop_mass = max(0.0, min(1.0, float(self.target_drop_mass)))
            coverage = max(0.0, 1.0 - drop_mass)
            sorted_vals, sorted_idx = torch.sort(m_tilde, dim=-1, descending=True)
            cdf = sorted_vals.cumsum(dim=-1)
            coverage_tensor = torch.full(
                (self.num_heads, num_q, 1),
                coverage,
                device=self.device,
                dtype=torch.float32,
            )
            keep_counts = torch.searchsorted(cdf, coverage_tensor, right=True).squeeze(-1)
            keep_counts = keep_counts.to(torch.int64)

            min_keep = min(self.prefill_min_blocks, nr_end)
            min_keep = max(min_keep, 1)
            max_keep = nr_end
            keep_counts = keep_counts.clamp(min=min_keep, max=max_keep)

            mask_sorted = torch.arange(nr_end, device=self.device).view(1, 1, -1)
            mask_sorted = mask_sorted < keep_counts.unsqueeze(-1)

            mask_nr = torch.zeros_like(m_tilde, dtype=torch.bool)
            mask_nr.scatter_(dim=-1, index=sorted_idx, src=mask_sorted)

            if debug_masks:
                kept_counts = mask_nr.sum(dim=-1).float()
                kept_mass = (m_tilde * mask_nr.float()).sum(dim=-1)
                logits_min = logits.amin(dim=-1)
                logits_max = logits.amax(dim=-1)
                gamma_min = gamma_nr.amin(dim=-1, keepdim=False)
                gamma_max = gamma_nr.amax(dim=-1, keepdim=False)
                qnorm_avg = q_norm2.mean(dim=-1)
                print(
                    f"[AdaptiveMask][layer={layer}] prefill multi keep avg={kept_counts.mean().item():.2f}"
                    f" min={kept_counts.min().item():.0f} max={kept_counts.max().item():.0f}"
                    f" mass_avg={kept_mass.mean().item():.3f}"
                )
                print(
                    f"  logits[min={logits_min.mean().item():.3f}, max={logits_max.mean().item():.3f}]"
                    f" gamma[min={gamma_min.mean().item():.5f}, max={gamma_max.mean().item():.5f}]"
                    f" q_norm2_avg={qnorm_avg.mean().item():.3f}"
                )

            mask[:, :, :nr_end] = mask_nr
            state.drop_mass.zero_()

            if total_blocks > 0 and mask.size(1) > 0:
                mask[:, :, 0].fill_(True)

            if num_blocks > 0 and mask.size(1) > 0:
                state.g_prev[:, :num_blocks] = mask[:, -1, :num_blocks].to(state.g_prev.dtype)
                state.g_prev[:, 0].fill_(1.0)
            elif num_blocks > 0:
                state.g_prev[:, 0].fill_(1.0)
            return mask

        mask = torch.zeros((self.num_heads, total_blocks), device=self.device, dtype=torch.bool)
        if tail_present:
            mask[:, -1].fill_(True)
        if recent_completed > 0:
            start_idx = num_blocks - recent_completed
            mask[:, start_idx:num_blocks].fill_(True)
            state.g_prev[:, start_idx:].fill_(1.0)
        if nr_end <= 0:
            if num_blocks > 0:
                state.g_prev[:, 0].fill_(1.0)
                mask[:, 0].fill_(True)
            state.drop_mass.zero_()
            return mask

        mu_raw_nr = state.mu[:, :nr_end, :].to(torch.float32)
        sq_sum_nr = state.sq_sum[:, :nr_end, :].to(torch.float32)
        tok_nr = state.token_count[:, :nr_end].to(torch.float32).clamp_min(1.0)
        vbar_nr = state.vbar[:, :nr_end, :].to(torch.float32)

        if prefill:
            q_vec = q_vec.to(torch.float32).unsqueeze(1)
            q_rms = torch.sqrt((q_vec * q_vec).mean(dim=-1, keepdim=True) + self.epsilon)
            q_normed = (q_vec / q_rms).squeeze(1)

            mu_blocks = mu_raw_nr
            mu_rms = torch.sqrt((mu_blocks * mu_blocks).mean(dim=-1, keepdim=True) + self.epsilon)
            mu_normed = mu_blocks / mu_rms

            mean_sq = sq_sum_nr / tok_nr.unsqueeze(-1)
            mean_sq_norm = mean_sq / (mu_rms * mu_rms)
            var_norm = (mean_sq_norm - mu_normed.pow(2)).clamp_min(0.0)
            gamma_nr = var_norm.sum(dim=-1) / float(self.head_dim)

            scale = self.inv_sqrt_dim
            q_norm2 = (q_normed * q_normed).sum(dim=-1) * (scale * scale)

            tau = state.tau.clamp_min(self.epsilon).unsqueeze(-1)

            logits = torch.matmul(mu_normed, q_normed.unsqueeze(-1)).squeeze(-1) * scale
            logits = logits / tau
            tau_sq = tau * tau
            logits = logits + gamma_nr * q_norm2.unsqueeze(-1) / (2.0 * tau_sq)

            m_tilde = torch.softmax(logits, dim=-1)

            drop_mass = max(0.0, min(1.0, float(self.target_drop_mass)))
            coverage = max(0.0, 1.0 - drop_mass)
            sorted_vals, sorted_idx = torch.sort(m_tilde, dim=-1, descending=True)
            cdf = sorted_vals.cumsum(dim=-1)
            coverage_tensor = torch.full(
                (self.num_heads, 1),
                coverage,
                device=self.device,
                dtype=torch.float32,
            )
            keep_counts = torch.searchsorted(cdf, coverage_tensor, right=True).squeeze(-1)
            keep_counts = keep_counts.to(torch.int64)

            min_keep = min(self.prefill_min_blocks, nr_end)
            min_keep = max(min_keep, 1)
            max_keep = nr_end
            keep_counts = keep_counts.clamp(min=min_keep, max=max_keep)

            mask_sorted = torch.arange(nr_end, device=self.device).view(1, -1)
            mask_sorted = mask_sorted < keep_counts.unsqueeze(-1)

            g_nr = torch.zeros_like(m_tilde, dtype=torch.bool)
            g_nr.scatter_(dim=-1, index=sorted_idx, src=mask_sorted)

            if debug_masks:
                kept_counts = g_nr.sum(dim=-1).float()
                kept_mass = (m_tilde * g_nr.float()).sum(dim=-1)
                logits_min = logits.amin(dim=-1)
                logits_max = logits.amax(dim=-1)
                gamma_min = gamma_nr.amin(dim=-1)
                gamma_max = gamma_nr.amax(dim=-1)
                qnorm_avg = q_norm2
                print(
                    f"[AdaptiveMask][layer={layer}] prefill keep avg={kept_counts.mean().item():.2f}"
                    f" min={kept_counts.min().item():.0f} max={kept_counts.max().item():.0f}"
                    f" mass_avg={kept_mass.mean().item():.3f}"
                )
                print(
                    f"  logits[min={logits_min.mean().item():.3f}, max={logits_max.mean().item():.3f}]"
                    f" gamma[min={gamma_min.mean().item():.5f}, max={gamma_max.mean().item():.5f}]"
                    f" q_norm2_avg={qnorm_avg.mean().item():.3f}"
                )
            state.drop_mass.zero_()
            state.g_prev[:, :nr_end] = g_nr.to(state.g_prev.dtype)
            mask[:, :nr_end] = g_nr
        else:
            state.g_prev[:, :nr_end].fill_(1.0)
            mask[:, :nr_end].fill_(True)
            state.drop_mass.zero_()

        if num_blocks > 0:
            state.g_prev[:, 0].fill_(1.0)
            mask[:, 0].fill_(True)
        return mask

    def register_output(
        self,
        layer: int,
        y: torch.Tensor,
        tensor_layout: str = "HND",
    ) -> None:
        """Feed back the attention output for EMA and lambda updates.

        Parameters
        ----------
        layer : int
            Layer whose statistics should receive the feedback.
        y : torch.Tensor
            Attention output aligned with ``q``. Supports shapes
            ``(num_heads, head_dim)``, ``(1, num_heads, head_dim)``,
            ``(num_heads, tokens, head_dim)`` 以及四维 ``(batch, num_heads,
            tokens, head_dim)`` 或 ``(batch, tokens, num_heads, head_dim)``，
            四维输入会在序列维上取平均。
        tensor_layout : str
            Layout flag used when ``y`` 是四维张量. Either "HND" or "NHD".
        """
        state = self._layer_state(layer)
        debug_masks = os.environ.get("ADAPTIVE_DEBUG", "0") == "1"
        if y.dim() == 4:
            if tensor_layout == "NHD":
                y = y.permute(0, 2, 1, 3)
            y_vec = y.mean(dim=2).squeeze(0)
        elif y.dim() == 3:
            y_vec = y.squeeze(0)
        else:
            y_vec = y

        beta = self.beta
        state.y_hat.mul_(1.0 - beta).add_(y_vec, alpha=beta)
        state.y_hat_correction.mul_(1.0 - beta).add_(beta)

        num_blocks = state.g_prev.size(1)
        tail_present = state.tail_token_count[0].item() > 0
        total_blocks = num_blocks + (1 if tail_present else 0)
        recent_total = min(self.recent_window_len, total_blocks)
        recent_tail = 1 if tail_present and recent_total > 0 else 0
        recent_completed = min(num_blocks, max(recent_total - recent_tail, 0))
        nr_end = num_blocks - recent_completed
        if nr_end > 0:
            sel_nr = state.g_prev[:, :nr_end].sum(dim=-1)
            drop_mass = state.drop_mass
            state.lam.add_(-self.lr_lambda * (drop_mass - self.target_drop_mass))
            state.tau.add_(self.lr_tau * (drop_mass - self.target_drop_mass)).clamp_(min=self.epsilon)
            if debug_masks:
                print(
                    f"[AdaptiveMask][layer={layer}] feedback sel_nr_avg={sel_nr.mean().item():.2f}"
                    f" lam_avg={state.lam.mean().item():.4f} tau_avg={state.tau.mean().item():.3f}"
                    f" drop_mass_avg={drop_mass.mean().item():.4f}"
                )
        else:
            state.drop_mass.zero_()
            if debug_masks:
                print(
                    f"[AdaptiveMask][layer={layer}] feedback (no NR blocks)"
                    f" lam_avg={state.lam.mean().item():.4f} tau_avg={state.tau.mean().item():.3f}"
                )

        state.step_counter += 1

    # Convenience API ------------------------------------------------------
    def step(
        self,
        layer: int,
        q: torch.Tensor,
        k_new: Optional[torch.Tensor] = None,
        v_new: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k_new is not None and v_new is not None:
            self.update_tail_statistics(layer, k_new, v_new)
        return self.compute_mask(layer, q)
