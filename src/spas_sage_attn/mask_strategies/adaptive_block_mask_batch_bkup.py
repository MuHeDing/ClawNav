"""Adaptive block mask strategy for per-head sparse attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os
import torch
import torch.nn.functional as F


@dataclass
class LayerState:
    """Per-layer state shared across heads and batches."""

    batch_size: int
    mu: torch.Tensor  # (B, H, K, D)
    vbar: torch.Tensor  # (B, H, K, D)
    gamma: torch.Tensor  # (B, H, K)
    token_count: torch.Tensor  # (B, H, K)
    g_prev: torch.Tensor  # (B, H, K)
    y_hat: torch.Tensor  # (B, H, D)
    y_hat_correction: torch.Tensor  # (B, H)
    drop_mass: torch.Tensor  # (B, H)
    lam: torch.Tensor  # (B, H)
    tau: torch.Tensor  # (B, H)
    step_counter: int = 0


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
        q_clip_ratio: float = 2.0,
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
        self.q_clip_ratio = max(float(q_clip_ratio), 1.0)
        self.epsilon = epsilon

        self.layers: Dict[int, LayerState] = {
            layer: self._init_layer_state()
            for layer in range(num_layers)
        }

    def _layer_state(self, layer: int) -> LayerState:
        return self.layers[layer]

    def _init_layer_state(self, batch_size: int = 1) -> LayerState:
        zeros_vec = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device=self.device, dtype=torch.float32)
        zeros = torch.zeros((batch_size, self.num_heads, 0), device=self.device, dtype=torch.float32)
        zeros_h = torch.zeros((batch_size, self.num_heads), device=self.device, dtype=torch.float32)
        return LayerState(
            batch_size=batch_size,
            mu=zeros_vec.clone(),
            vbar=zeros_vec.clone(),
            gamma=zeros.clone(),
            token_count=zeros.clone(),
            g_prev=zeros.clone(),
            y_hat=torch.zeros((batch_size, self.num_heads, self.head_dim), device=self.device, dtype=torch.float32),
            y_hat_correction=torch.zeros((batch_size, self.num_heads), device=self.device, dtype=torch.float32),
            drop_mass=zeros_h.clone(),
            lam=torch.full((batch_size, self.num_heads), float(self.lam_init), device=self.device, dtype=torch.float32),
            tau=torch.full((batch_size, self.num_heads), float(self.tau_init), device=self.device, dtype=torch.float32),
        )

    def reset_layer(self, layer: int, batch_size: Optional[int] = None) -> None:
        if batch_size is None:
            batch_size = self.layers[layer].batch_size
        self.layers[layer] = self._init_layer_state(batch_size)

    def reset_all(self, batch_size: Optional[int] = None) -> None:
        for layer in self.layers:
            current_bs = batch_size if batch_size is not None else self.layers[layer].batch_size
            self.layers[layer] = self._init_layer_state(current_bs)

    def ingest_kv(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        tensor_layout: str = "HND",
    ) -> None:
        """Absorb the current full KV cache for the given layer.

        The diffusion pipeline supplies the complete cache every iteration,
        so we rebuild the per-block statistics from scratch instead of keeping
        incremental tail buffers.
        """

        assert tensor_layout in ["HND", "NHD"], "Unsupported tensor layout"
        if tensor_layout == "NHD":
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()

        if k.dim() != 4 or v.dim() != 4:
            raise ValueError("k and v must have shape (batch, heads, seq, dim)")

        batch_size = k.size(0)
        if v.size(0) != batch_size:
            raise ValueError("k and v must share the same batch dimension")

        state = self._layer_state(layer)
        old_g_prev = state.g_prev
        old_drop_mass = state.drop_mass
        old_step_counter = state.step_counter

        if state.batch_size != batch_size:
            state = self._init_layer_state(batch_size)
            self.layers[layer] = state
            old_g_prev = state.g_prev
            old_drop_mass = state.drop_mass
            old_step_counter = state.step_counter

        k_seq = k.to(self.device, dtype=torch.float32)
        v_seq = v.to(self.device, dtype=torch.float32)
        seq_len = k_seq.size(2)
        if seq_len == 0:
            self.layers[layer] = self._init_layer_state(batch_size)
            return

        block = self.kv_block_size
        num_blocks = (seq_len + block - 1) // block
        pad = num_blocks * block - seq_len

        if pad > 0:
            k_padded = F.pad(k_seq, (0, 0, 0, pad))
            v_padded = F.pad(v_seq, (0, 0, 0, pad))
        else:
            k_padded = k_seq
            v_padded = v_seq

        k_blocks = k_padded.view(batch_size, self.num_heads, num_blocks, block, self.head_dim)
        v_blocks = v_padded.view(batch_size, self.num_heads, num_blocks, block, self.head_dim)

        valid_mask = torch.ones(
            (batch_size, self.num_heads, num_blocks, block, 1),
            device=self.device,
            dtype=torch.float32,
        )
        if pad > 0:
            valid_mask[:, :, -1, -pad:, :] = 0.0

        counts = valid_mask.sum(dim=3).clamp_min(1.0).squeeze(-1)

        mu_blocks = (k_blocks * valid_mask).sum(dim=3) / counts.unsqueeze(-1)
        vbar_blocks = (v_blocks * valid_mask).sum(dim=3) / counts.unsqueeze(-1)

        k_sq_sum = (k_blocks.square() * valid_mask).sum(dim=3)
        mean_sq = mu_blocks.square()
        var_blocks = (k_sq_sum / counts.unsqueeze(-1)) - mean_sq
        var_blocks = var_blocks.clamp_min(0.0)
        gamma_blocks = var_blocks.mean(dim=-1)

        state.batch_size = batch_size
        state.mu = mu_blocks
        state.vbar = vbar_blocks
        state.gamma = gamma_blocks
        state.token_count = counts

        new_shape = (batch_size, self.num_heads, num_blocks)
        if num_blocks == 0:
            state.g_prev = torch.zeros(new_shape, device=self.device, dtype=torch.float32)
        else:
            if old_g_prev.numel() == 0 or old_g_prev.size(2) == 0:
                g_prev_new = torch.zeros(new_shape, device=self.device, dtype=torch.float32)
            else:
                g_prev_new = torch.zeros(new_shape, device=self.device, dtype=old_g_prev.dtype)
                copy_blocks = min(old_g_prev.size(2), num_blocks)
                if copy_blocks > 0:
                    g_prev_new[:, :, :copy_blocks] = old_g_prev[:, :, :copy_blocks]
            state.g_prev = g_prev_new

        state.drop_mass = old_drop_mass
        state.step_counter = old_step_counter

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
        if q.dim() == 4:
            if tensor_layout == "NHD":
                q = q.permute(0, 2, 1, 3)
            q_vec = q.to(self.device, dtype=torch.float32)
            multi_query = True
        elif q.dim() == 3:
            if tensor_layout == "NHD":
                raise NotImplementedError("3D NHD layout is not supported")
            q_vec = q.to(self.device, dtype=torch.float32)
            multi_query = False
        else:
            raise ValueError("q must have rank 3 or 4")

        batch_size = q_vec.size(0) if q_vec.dim() == 4 else q_vec.size(0)
        if batch_size != state.batch_size:
            raise ValueError(
                f"Batch size mismatch: got {batch_size}, expected {state.batch_size}."
            )

        num_blocks = state.mu.size(2)
        total_blocks = num_blocks

        if total_blocks == 0:
            state.drop_mass.zero_()
            if multi_query:
                return torch.ones(
                    (batch_size, self.num_heads, q_vec.size(2), 1),
                    device=self.device,
                    dtype=torch.bool,
                )
            return torch.ones(
                (batch_size, self.num_heads, 1),
                device=self.device,
                dtype=torch.bool,
            )

        recent_completed = min(num_blocks, self.recent_window_len)
        nr_end = num_blocks - recent_completed

        if multi_query:
            num_q = q_vec.size(2)
            mask = torch.zeros(
                (batch_size, self.num_heads, num_q, total_blocks),
                device=self.device,
                dtype=torch.bool,
            )
            if recent_completed > 0:
                start_idx = total_blocks - recent_completed
                mask[:, :, :, start_idx:total_blocks].fill_(True)
                state.g_prev[:, :, start_idx:total_blocks].fill_(1.0)
            if nr_end <= 0:
                state.drop_mass.zero_()
                mask[:, :, :, 0].fill_(True)
                return mask

            mu_nr = state.mu[:, :, :nr_end, :]
            gamma_nr = state.gamma[:, :, :nr_end]
            tau = state.tau.clamp_min(self.epsilon)

            mu_scale = mu_nr.norm(dim=-1).mean(dim=2, keepdim=True).clamp_min(self.epsilon)
            mu_normed = mu_nr / mu_scale.unsqueeze(-1)
            mu_scale_sq = (mu_scale.squeeze(-1) ** 2).unsqueeze(-1)
            gamma_scaled = gamma_nr / mu_scale_sq

            q_scale = q_vec.norm(dim=-1).mean(dim=2, keepdim=True).clamp_min(self.epsilon)
            q_normed = q_vec / q_scale.unsqueeze(-1)

            scale = self.inv_sqrt_dim
            q_norm2 = (q_normed * q_normed).sum(dim=-1) * (scale * scale)

            tau_exp = tau.unsqueeze(-1).unsqueeze(-1)
            tau_sq = tau_exp * tau_exp

            logits = torch.einsum('bhqd,bhkd->bhqk', q_normed, mu_normed) * scale
            logits = logits / tau_exp
            logits = logits + gamma_scaled.unsqueeze(2) * q_norm2.unsqueeze(-1) / (2.0 * tau_sq)

            m_tilde = torch.softmax(logits, dim=-1)

            drop_mass = max(0.0, min(1.0, float(self.target_drop_mass)))
            coverage = max(0.0, 1.0 - drop_mass)
            sorted_vals, sorted_idx = torch.sort(m_tilde, dim=-1, descending=True)
            cdf = sorted_vals.cumsum(dim=-1)
            coverage_tensor = torch.full(
                (batch_size, self.num_heads, num_q, 1),
                coverage,
                device=self.device,
                dtype=torch.float32,
            )
            keep_counts = torch.searchsorted(cdf, coverage_tensor, right=True).squeeze(-1)
            keep_counts = keep_counts.to(torch.int64)

            min_keep = max(min(self.prefill_min_blocks, nr_end), 1)
            max_keep = nr_end
            keep_counts = keep_counts.clamp(min=min_keep, max=max_keep)

            arange_idx = torch.arange(nr_end, device=self.device).view(1, 1, 1, -1)
            mask_sorted = arange_idx < keep_counts.unsqueeze(-1)

            mask_nr = torch.zeros_like(m_tilde, dtype=torch.bool)
            mask_nr.scatter_(dim=-1, index=sorted_idx, src=mask_sorted)

            mask[:, :, :, :nr_end] = mask_nr
            mask[:, :, :, 0].fill_(True)
            state.drop_mass.zero_()
            state.g_prev[:, :, :nr_end] = mask_nr[:, :, -1, :nr_end].to(state.g_prev.dtype)
            state.g_prev[:, :, 0].fill_(1.0)
            if debug_masks:
                kept_counts = mask_nr.sum(dim=-1).float().mean(dim=(0, 1))
                kept_mass = (m_tilde * mask_nr.float()).sum(dim=-1).mean(dim=(0, 1))
                active_ratio = mask.float().mean().item()
                sparsity = 1.0 - active_ratio
                print(
                    f"[AdaptiveMask][layer={layer}] prefill multi keep avg={kept_counts.mean().item():.2f}"
                    f" mass_avg={kept_mass.mean().item():.3f}"
                    f" active_ratio={active_ratio:.4f} sparsity={sparsity:.4f}"
                )
            return mask

        # Single query summary path (e.g., decode)
        if q_vec.dim() != 3:
            raise ValueError("Expected q with shape (batch, heads, dim) for single-query path")

        mask = torch.zeros((batch_size, self.num_heads, total_blocks), device=self.device, dtype=torch.bool)
        if recent_completed > 0:
            start_idx = total_blocks - recent_completed
            mask[:, :, start_idx:total_blocks].fill_(True)
            state.g_prev[:, :, start_idx:total_blocks].fill_(1.0)
        if nr_end <= 0:
            if num_blocks > 0:
                state.g_prev[:, :, 0].fill_(1.0)
                mask[:, :, 0].fill_(True)
            state.drop_mass.zero_()
            return mask

        mu_nr = state.mu[:, :, :nr_end, :]
        gamma_nr = state.gamma[:, :, :nr_end]
        g_prev_nr = state.g_prev[:, :, :nr_end]
        vbar_nr = state.vbar[:, :, :nr_end, :]

        mu_scale = mu_nr.norm(dim=-1).mean(dim=2, keepdim=True).clamp_min(self.epsilon)
        mu_normed = mu_nr / mu_scale.unsqueeze(-1)
        mu_scale_sq = (mu_scale.squeeze(-1) ** 2).unsqueeze(-1)
        gamma_scaled = gamma_nr / mu_scale_sq

        q_scale = q_vec.norm(dim=-1, keepdim=True).clamp_min(self.epsilon)
        q_normed = q_vec / q_scale

        scale = self.inv_sqrt_dim
        q_norm2 = (q_normed * q_normed).sum(dim=-1) * (scale * scale)

        tau = state.tau.clamp_min(self.epsilon)
        tau_unsqueezed = tau.unsqueeze(-1)
        tau_sq = tau_unsqueezed * tau_unsqueezed

        logits = torch.einsum('bhkd,bhd->bhk', mu_normed, q_normed) * scale
        logits = logits / tau_unsqueezed
        logits = logits + gamma_scaled * q_norm2.unsqueeze(-1) / (2.0 * tau_sq)

        m_tilde = torch.softmax(logits, dim=-1)

        if prefill:
            drop_mass = max(0.0, min(1.0, float(self.target_drop_mass)))
            coverage = max(0.0, 1.0 - drop_mass)
            sorted_vals, sorted_idx = torch.sort(m_tilde, dim=-1, descending=True)
            cdf = sorted_vals.cumsum(dim=-1)
            coverage_tensor = torch.full(
                (batch_size, self.num_heads, 1), coverage, device=self.device, dtype=torch.float32
            )
            keep_counts = torch.searchsorted(cdf, coverage_tensor, right=True).squeeze(-1)
            keep_counts = keep_counts.to(torch.int64)

            min_keep = max(min(self.prefill_min_blocks, nr_end), 1)
            max_keep = nr_end
            keep_counts = keep_counts.clamp(min=min_keep, max=max_keep)

            arange_idx = torch.arange(nr_end, device=self.device).view(1, 1, -1)
            mask_sorted = arange_idx < keep_counts.unsqueeze(-1)

            g_nr = torch.zeros_like(m_tilde, dtype=torch.bool)
            g_nr.scatter_(dim=-1, index=sorted_idx, src=mask_sorted)

            state.drop_mass.zero_()
            state.g_prev[:, :, :nr_end] = g_nr.to(state.g_prev.dtype)
            mask[:, :, :nr_end] = g_nr
            if debug_masks:
                kept_counts = g_nr.sum(dim=-1).float().mean()
                kept_mass = (m_tilde * g_nr.float()).sum(dim=-1).mean()
                active_ratio = mask.float().mean().item()
                sparsity = 1.0 - active_ratio
                print(
                    f"[AdaptiveMask][layer={layer}] prefill keep avg={kept_counts.item():.2f}"
                    f" mass_avg={kept_mass.item():.3f}"
                    f" active_ratio={active_ratio:.4f} sparsity={sparsity:.4f}"
                )
        else:
            corr = state.y_hat_correction.clamp_min(self.epsilon).unsqueeze(-1)
            y_hat = state.y_hat / corr
            phi_raw = torch.sum((vbar_nr - y_hat.unsqueeze(2)) ** 2, dim=-1)

            mean_phi = phi_raw.mean(dim=-1, keepdim=True)
            if nr_end > 1:
                var_phi = phi_raw.var(dim=-1, unbiased=True, keepdim=True)
            else:
                var_phi = torch.full_like(mean_phi, self.epsilon)
            var_phi = var_phi.clamp_min(self.epsilon)
            phi = (phi_raw - mean_phi) / var_phi.sqrt()

            mu_term = self.mu_reg * (1.0 - 2.0 * g_prev_nr)
            if state.step_counter == 0:
                mu_term = torch.zeros_like(mu_term)
            lam_term = state.lam.unsqueeze(-1)

            scores = m_tilde * phi - lam_term - mu_term
            g_nr_bool = scores > 0
            g_nr = g_nr_bool.to(state.g_prev.dtype)

            dropped_mass = (m_tilde * (~g_nr_bool).float()).sum(dim=-1)
            state.drop_mass.copy_(dropped_mass)


            state.g_prev[:, :, :nr_end] = g_nr
            mask[:, :, :nr_end] = g_nr_bool

        state.g_prev[:, :, 0].fill_(1.0)
        mask[:, :, 0].fill_(True)
        if debug_masks:
            kept_counts = mask[:, :, :nr_end].sum(dim=-1).float().mean()
            active_ratio = mask.float().mean().item()
            sparsity = 1.0 - active_ratio
            print(
                f"[AdaptiveMask][layer={layer}] single-query keep avg={kept_counts.item():.2f}"
                f" active_ratio={active_ratio:.4f} sparsity={sparsity:.4f}"
            )
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
            y_vec = y.mean(dim=2)
        elif y.dim() == 3:
            if y.size(0) == state.batch_size and y.size(1) == self.num_heads:
                y_vec = y
            elif y.size(0) == self.num_heads and state.batch_size == 1:
                y_vec = y.unsqueeze(0)
            else:
                raise ValueError("Unsupported shape for y in register_output")
        else:
            raise ValueError("register_output expects a 3D or 4D tensor")

        beta = self.beta
        state.y_hat.mul_(1.0 - beta).add_(y_vec, alpha=beta)
        state.y_hat_correction.mul_(1.0 - beta).add_(beta)

        num_blocks = state.g_prev.size(2)
        recent_completed = min(num_blocks, self.recent_window_len)
        nr_end = num_blocks - recent_completed
        if nr_end > 0:
            sel_nr = state.g_prev[:, :, :nr_end].sum(dim=-1)
            state.lam.add_(self.lr_lambda * (sel_nr - self.target_blocks))

            drop_mass = state.drop_mass
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
        if k_new is not None or v_new is not None:
            raise NotImplementedError("Incremental KV updates are disabled in diffusion mode")
        return self.compute_mask(layer, q)
