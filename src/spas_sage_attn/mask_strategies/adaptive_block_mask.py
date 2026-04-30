"""Adaptive block mask strategy for per-head sparse attention."""

from __future__ import annotations

import math
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
    g_prev_mask: torch.Tensor  # (B, H, Q, K)
    y_hat: torch.Tensor  # (B, H, Q, D)
    drop_mass: torch.Tensor  # (B, H)
    lam: torch.Tensor  # (B, H)
    tau: torch.Tensor  # (B, H)
    step_counter: int = 0
    recent_start: int = 0
    total_blocks: int = 0


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
        protected_initial_blocks: int = 4,  # JANUSVLN_PHASE5_FIX: Protect initial blocks
        tau_init: float = 1.0,
        lam_init: float = 0.05,
        mu_reg: float = 0.1,
        beta: float = 0.5,
        lr_lambda: float = 0.01,
        lr_tau: float = 0.02,
        alpha_min: float = 0.2,
        alpha_max: float = 5.0,
        rho_lambda: float = 0.2,
        warmup_steps: int = 3,
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
        self.protected_initial_blocks = max(int(protected_initial_blocks), 1)  # JANUSVLN_PHASE5_FIX
        self.tau_init = max(float(tau_init), epsilon)
        self.lam_init = max(float(lam_init), epsilon)
        self.mu_reg = mu_reg
        self.beta = beta
        self.lr_lambda = lr_lambda
        self.lr_tau = lr_tau
        self.q_clip_ratio = max(float(q_clip_ratio), 1.0)
        self.epsilon = epsilon
        self.alpha_min = max(float(alpha_min), epsilon)
        self.alpha_max = max(float(alpha_max), self.alpha_min)
        self.rho_lambda = float(rho_lambda)
        self.desired_coverage = 1.0 - float(self.target_drop_mass)
        self.warmup_steps = max(int(warmup_steps), 1)

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
            g_prev_mask=torch.ones((batch_size, self.num_heads, 0, 0), device=self.device, dtype=torch.float32),
            y_hat=zeros_vec.clone(),
            drop_mass=zeros_h.clone(),
            lam=torch.full((batch_size, self.num_heads), float(self.lam_init), device=self.device, dtype=torch.float32),
            tau=torch.full((batch_size, self.num_heads), float(self.tau_init), device=self.device, dtype=torch.float32),
            recent_start=0,
            total_blocks=0,
        )

    def reset_layer(self, layer: int, batch_size: Optional[int] = None) -> None:
        if batch_size is None:
            batch_size = self.layers[layer].batch_size
        self.layers[layer] = self._init_layer_state(batch_size)

    def reset_all(self, batch_size: Optional[int] = None) -> None:
        for layer in self.layers:
            current_bs = batch_size if batch_size is not None else self.layers[layer].batch_size
            self.layers[layer] = self._init_layer_state(current_bs)

    def step_index(self, layer: int) -> int:
        """Return the completed-iteration count for ``layer``."""

        return int(self._layer_state(layer).step_counter)

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
        old_drop_mass = state.drop_mass
        old_step_counter = state.step_counter

        if state.batch_size != batch_size:
            state = self._init_layer_state(batch_size)
            self.layers[layer] = state
            old_drop_mass = state.drop_mass
            old_step_counter = state.step_counter

        k_seq = k.to(self.device, dtype=torch.float32)
        v_seq = v.to(self.device, dtype=torch.float32)
        seq_len = k_seq.size(2)
        if seq_len == 0:
            self.layers[layer] = self._init_layer_state(batch_size)
            return

        block = self.kv_block_size
        num_full = seq_len // block
        tail = seq_len % block
        total_blocks = num_full + (1 if tail > 0 else 0)

        if total_blocks == 0:
            self.layers[layer] = self._init_layer_state(batch_size)
            return

        recent_start = max(total_blocks - self.recent_window_len, 0)
        # JANUSVLN_PHASE5_FIX: Candidate blocks start after protected initial blocks
        candidate_start = min(self.protected_initial_blocks, total_blocks)
        candidate_end = min(recent_start, num_full)
        if candidate_end < candidate_start:
            candidate_end = candidate_start
        candidate_count = max(candidate_end - candidate_start, 0)

        if candidate_count > 0:
            start_idx = candidate_start * block
            end_idx = candidate_end * block
            k_sel = k_seq[:, :, start_idx:end_idx, :].view(batch_size, self.num_heads, candidate_count, block, self.head_dim)
            v_sel = v_seq[:, :, start_idx:end_idx, :].view(batch_size, self.num_heads, candidate_count, block, self.head_dim)

            mu_blocks = k_sel.mean(dim=3)
            vbar_blocks = v_sel.mean(dim=3)
            if k_sel.size(3) > 1:
                var_blocks = k_sel.var(dim=3, correction=1)
            else:
                var_blocks = torch.zeros_like(mu_blocks)
            var_blocks = var_blocks.clamp_min(0.0)
            gamma_blocks = var_blocks.mean(dim=-1)
            counts = torch.full((batch_size, self.num_heads, candidate_count), float(block), device=self.device, dtype=torch.float32)
        else:
            mu_blocks = torch.zeros((batch_size, self.num_heads, 0, self.head_dim), device=self.device, dtype=torch.float32)
            vbar_blocks = torch.zeros_like(mu_blocks)
            gamma_blocks = torch.zeros((batch_size, self.num_heads, 0), device=self.device, dtype=torch.float32)
            counts = torch.zeros((batch_size, self.num_heads, 0), device=self.device, dtype=torch.float32)

        state.batch_size = batch_size
        state.mu = mu_blocks
        state.vbar = vbar_blocks
        state.gamma = gamma_blocks
        state.token_count = counts

        if candidate_count <= 0:
            state.g_prev_mask = torch.ones((batch_size, self.num_heads, 0, 0), device=self.device, dtype=torch.float32)
        else:
            if state.g_prev_mask.numel() == 0 or state.g_prev_mask.size(-1) != candidate_count:
                state.g_prev_mask = torch.ones((batch_size, self.num_heads, 0, candidate_count), device=self.device, dtype=torch.float32)
            elif state.g_prev_mask.size(0) != batch_size:
                q_dim = state.g_prev_mask.size(2)
                state.g_prev_mask = torch.ones((batch_size, self.num_heads, q_dim, candidate_count), device=self.device, dtype=torch.float32)

        state.drop_mass = old_drop_mass
        state.step_counter = old_step_counter
        state.recent_start = recent_start
        state.total_blocks = total_blocks

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
            Deprecated flag kept for caller compatibility. The masker now
            自动依据 ``step_counter`` 与反馈统计决定是否进入自适应路径，调用方无需再依赖
            该标志控制 decode 流程。

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
            q_vec = q.to(self.device, dtype=torch.float32).unsqueeze(2)
            multi_query = False
        else:
            raise ValueError("q must have rank 3 or 4")

        batch_size = q_vec.size(0)
        if batch_size != state.batch_size:
            raise ValueError(
                f"Batch size mismatch: got {batch_size}, expected {state.batch_size}."
            )

        total_blocks = state.total_blocks
        if total_blocks <= 0:
            state.drop_mass.zero_()
            mask_shape = (batch_size, self.num_heads, q_vec.size(2) if q_vec.dim() == 4 else 1, 1)
            mask_full = torch.ones(mask_shape, device=self.device, dtype=torch.bool)
            if multi_query:
                return mask_full
            return mask_full[:, :, 0, :]

        recent_start = min(max(state.recent_start, 0), total_blocks)
        candidate_count = state.mu.size(2)
        # JANUSVLN_PHASE5_FIX: Candidate blocks start after protected initial blocks
        candidate_start = min(self.protected_initial_blocks, total_blocks)
        candidate_end = candidate_start + candidate_count

        num_q = q_vec.size(2)
        mask = torch.zeros((batch_size, self.num_heads, num_q, total_blocks), device=self.device, dtype=torch.bool)

        # JANUSVLN_PHASE5_FIX: Protect initial blocks for VLN task
        # System prompt + navigation instruction + image features typically need 3-4 blocks (192-256 tokens)
        # This prevents critical instruction tokens from being masked
        protected_end = min(self.protected_initial_blocks, total_blocks)
        mask[:, :, :, 0:protected_end].fill_(True)

        # Protect recent window (current observation)
        if recent_start < total_blocks:
            mask[:, :, :, recent_start:total_blocks].fill_(True)

        if candidate_count <= 0:
            state.drop_mass.zero_()
            if multi_query:
                return mask
            return mask[:, :, 0, :]

        mu_c = state.mu[:, :, :candidate_count, :]
        vbar_c = state.vbar[:, :, :candidate_count, :]
        gamma_c = state.gamma[:, :, :candidate_count]
        tau = state.tau.clamp_min(self.epsilon)

        mu_scale = mu_c.norm(dim=-1).mean(dim=2, keepdim=True).clamp_min(self.epsilon)
        mu_normed = mu_c / mu_scale.unsqueeze(-1)
        mu_scale_sq = (mu_scale.squeeze(-1) ** 2).unsqueeze(-1)
        gamma_scaled = gamma_c / mu_scale_sq

        q_scale = q_vec.norm(dim=-1).mean(dim=2, keepdim=True).clamp_min(self.epsilon)
        q_normed = q_vec / q_scale.unsqueeze(-1)

        scale = self.inv_sqrt_dim
        q_norm2 = (q_normed * q_normed).sum(dim=-1) * scale

        tau_exp = tau.unsqueeze(-1).unsqueeze(-1)
        tau_sq = tau_exp * tau_exp

        logits = torch.einsum('bhqd,bhkd->bhqk', q_normed, mu_normed) * scale
        logits = logits / tau_exp
        logits = logits + gamma_scaled.unsqueeze(2) * q_norm2.unsqueeze(-1) / (2.0 * tau_sq)

        mass_probs = torch.softmax(logits, dim=-1)

        prev_mask = state.g_prev_mask
        if prev_mask.numel() == 0 or prev_mask.size(-1) != candidate_count:
            g_prev_blocks = torch.ones((batch_size, self.num_heads, num_q, candidate_count), device=self.device, dtype=torch.float32)
        else:
            g_prev_blocks = prev_mask

        has_feedback = (state.step_counter >= self.warmup_steps) and (
            state.g_prev_mask.size(-1) == candidate_count and candidate_count > 0
        )

        should_log = debug_masks and (
            state.step_counter < 3 or state.step_counter % 10 == 0
        )

        if not has_feedback:
            coverage_tensor = torch.full((batch_size, self.num_heads, num_q, 1), self.desired_coverage, device=self.device, dtype=torch.float32)
            sorted_vals, sorted_idx = torch.sort(mass_probs, dim=-1, descending=True)
            cdf = sorted_vals.cumsum(dim=-1)
            keep_counts = torch.searchsorted(cdf, coverage_tensor, right=True).to(torch.int64)
            min_keep = max(min(self.prefill_min_blocks, candidate_count), 1)
            keep_counts = keep_counts.clamp(min=min_keep, max=candidate_count)

            arange_idx = torch.arange(candidate_count, device=self.device).view(1, 1, 1, -1)
            keep_counts_flat = keep_counts.squeeze(-1)
            mask_sorted = arange_idx < keep_counts_flat.unsqueeze(-1)
            mask_candidates = torch.zeros_like(mass_probs, dtype=torch.float32)
            mask_candidates.scatter_(dim=-1, index=sorted_idx, src=mask_sorted.to(torch.float32))
            mask_candidates_bool = mask_candidates > 0.5

            mask[:, :, :, candidate_start:candidate_end] = mask_candidates_bool

            selected = mask_candidates
            coverage_q = (mass_probs * selected).sum(dim=-1)
            coverage_mean = coverage_q.mean(dim=2)
            state.drop_mass.copy_(1.0 - coverage_mean)
            if self.lr_tau > 0.0:
                state.tau.add_(-self.lr_tau * (coverage_mean - self.desired_coverage)).clamp_(min=self.epsilon)
            state.g_prev_mask = mask_candidates_bool.to(torch.float32)

            if should_log:
                kept_avg = selected.sum(dim=-1).float().mean().item() / max(mass_probs.size(-1), 1)
                keep_counts_mean = keep_counts_flat.float().mean().item()
                keep_counts_std = (
                    keep_counts_flat.float().std(unbiased=False).item()
                    if keep_counts_flat.numel() > 1
                    else 0.0
                )
                active_ratio = mask.float().mean().item()
                sparsity = 1.0 - active_ratio
                print(
                    f"[AdaptiveMask][layer={layer}] step={state.step_counter} mode=warmup"
                    f" cand={candidate_count} keep={keep_counts_mean:.1f}±{keep_counts_std:.1f}"
                    f" ratio={kept_avg:.3f} cov={coverage_mean.mean().item():.3f}"
                    f" tau={state.tau.mean().item():.3f} drop={state.drop_mass.mean().item():.3f}"
                    f" sparsity={sparsity:.4f}"
                )

            if multi_query:
                return mask
            return mask[:, :, 0, :]

        mass_score = logits - torch.logsumexp(logits, dim=-1, keepdim=True) + math.log(max(candidate_count, 1))

        if candidate_count == 0:
            phi_raw = torch.zeros((batch_size, self.num_heads, num_q, 0), device=self.device, dtype=torch.float32)
        else:
            y_blocks = state.y_hat
            if y_blocks.size(2) == 0:
                phi_raw = torch.zeros((batch_size, self.num_heads, num_q, candidate_count), device=self.device, dtype=torch.float32)
            else:
                if y_blocks.size(2) > num_q:
                    y_blocks = y_blocks[:, :, :num_q, :]
                elif y_blocks.size(2) < num_q:
                    pad = num_q - y_blocks.size(2)
                    pad_tensor = torch.zeros((batch_size, self.num_heads, pad, self.head_dim), device=self.device, dtype=torch.float32)
                    y_blocks = torch.cat([y_blocks, pad_tensor], dim=2)

                y_blocks = y_blocks[:, :, :num_q, :].to(torch.float32)
                v_blocks = vbar_c.to(torch.float32)

                y_flat = y_blocks.reshape(-1, y_blocks.size(2), self.head_dim)
                v_flat = v_blocks.reshape(-1, candidate_count, self.head_dim)
                phi_flat = torch.cdist(
                    y_flat,
                    v_flat,
                    p=2.0,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                phi_raw = phi_flat.pow_(2.0).view(batch_size, self.num_heads, num_q, candidate_count)

        std_mass = torch.std(mass_score, dim=-1, keepdim=True, unbiased=False).clamp_min(self.epsilon)
        std_phi = torch.std(phi_raw, dim=-1, keepdim=True, unbiased=False).clamp_min(self.epsilon)
        alpha_t = (std_mass / std_phi).clamp(self.alpha_min, self.alpha_max)

        phi_scaled = alpha_t * phi_raw
        # log_mean_exp = torch.logsumexp(phi_scaled, dim=-1, keepdim=True) - math.log(max(candidate_count, 1))
        phi_comp = phi_scaled #- log_mean_exp

        g_prev = g_prev_blocks
        mu_term = self.mu_reg * (1.0 - 2.0 * g_prev)
        base = 3 * mass_score + phi_comp - mu_term

        q_value = 1.0 - (float(self.target_blocks) / max(float(candidate_count), self.epsilon))
        q_value = min(max(q_value, 0.0), 1.0)
        lambda_per_q = torch.quantile(base, q_value, dim=-1)
        lambda_star = lambda_per_q.mean(dim=-1)
        state.lam = (1.0 - self.rho_lambda) * state.lam + self.rho_lambda * lambda_star.to(state.lam.dtype)

        lambda_shift = state.lam.unsqueeze(-1).unsqueeze(-1)
        scores = base - lambda_shift
        mask_candidates = scores > 0

        mask[:, :, :, candidate_start:candidate_end] = mask_candidates

        selected = mask_candidates.float()
        coverage_q = (mass_probs * selected).sum(dim=-1)
        coverage_mean = coverage_q.mean(dim=2)
        state.drop_mass.copy_(1.0 - coverage_mean)
        if self.lr_tau > 0.0:
            state.tau.add_(-self.lr_tau * (coverage_mean - self.desired_coverage)).clamp_(min=self.epsilon)
        state.g_prev_mask = mask_candidates.to(torch.float32)

        if should_log:
            kept_avg = selected.sum(dim=-1).float().mean().item() / max(mass_probs.size(-1), 1)
            keep_counts = selected.sum(dim=-1)
            keep_counts_mean = keep_counts.mean().item()
            keep_counts_std = (
                keep_counts.std(unbiased=False).item()
                if keep_counts.numel() > 1
                else 0.0
            )
            active_ratio = mask.float().mean().item()
            sparsity = 1.0 - active_ratio
            alpha_mean = alpha_t.mean().item()
            mass_std_mean = std_mass.mean().item()
            mass_mean = mass_score.mean().item()
            mass_min = mass_score.min().item()
            mass_max = mass_score.max().item()
            phi_mean = phi_comp.mean().item()
            phi_std = phi_comp.std(unbiased=False).item()
            mu_mean = mu_term.mean().item()
            mu_std = mu_term.std(unbiased=False).item()
            base_mean = base.mean().item()
            base_min = base.min().item()
            base_max = base.max().item()
            lambda_mean = state.lam.mean().item()
            tau_mean = state.tau.mean().item()
            drop_mean = state.drop_mass.mean().item()
            print(
                f"[AdaptiveMask][layer={layer}] step={state.step_counter} mode=adaptive"
                f" cand={candidate_count} keep={keep_counts_mean:.1f}±{keep_counts_std:.1f}"
                f" ratio={kept_avg:.3f} cov={coverage_mean.mean().item():.3f}"
                f" drop={drop_mean:.3f} lambda={lambda_mean:.3f} tau={tau_mean:.3f}"
                f" alpha={alpha_mean:.3f} mass_std={mass_std_mean:.3f}"
                f" mass=({mass_mean:.3f},{mass_min:.3f},{mass_max:.3f})"
                f" phi=({phi_mean:.3f},{phi_std:.3f}) mu=({mu_mean:.3f},{mu_std:.3f})"
                f" base=({base_mean:.3f},{base_min:.3f},{base_max:.3f})"
                f" sparsity={sparsity:.4f}"
            )

        if multi_query:
            return mask
        return mask[:, :, 0, :]

        # # Single query summary path (e.g., decode)
        # if q_vec.dim() != 3:
        #     raise ValueError("Expected q with shape (batch, heads, dim) for single-query path")

        # mask = torch.zeros((batch_size, self.num_heads, total_blocks), device=self.device, dtype=torch.bool)
        # if recent_completed > 0:
        #     start_idx = total_blocks - recent_completed
        #     mask[:, :, start_idx:total_blocks].fill_(True)
        #     state.g_prev[:, :, start_idx:total_blocks].fill_(1.0)
        # if nr_end <= 0:
        #     if num_blocks > 0:
        #         state.g_prev[:, :, 0].fill_(1.0)
        #         mask[:, :, 0].fill_(True)
        #     state.drop_mass.zero_()
        #     return mask

        # mu_nr = state.mu[:, :, :nr_end, :]
        # gamma_nr = state.gamma[:, :, :nr_end]
        # g_prev_nr = state.g_prev[:, :, :nr_end]
        # vbar_nr = state.vbar[:, :, :nr_end, :]

        # mu_scale = mu_nr.norm(dim=-1).mean(dim=2, keepdim=True).clamp_min(self.epsilon)
        # mu_normed = mu_nr / mu_scale.unsqueeze(-1)
        # mu_scale_sq = (mu_scale.squeeze(-1) ** 2).unsqueeze(-1)
        # gamma_scaled = gamma_nr / mu_scale_sq

        # q_scale = q_vec.norm(dim=-1, keepdim=True).clamp_min(self.epsilon)
        # q_normed = q_vec / q_scale

        # scale = self.inv_sqrt_dim
        # q_norm2 = (q_normed * q_normed).sum(dim=-1) * (scale * scale)

        # tau = state.tau.clamp_min(self.epsilon)
        # tau_unsqueezed = tau.unsqueeze(-1)
        # tau_sq = tau_unsqueezed * tau_unsqueezed

        # logits = torch.einsum('bhkd,bhd->bhk', mu_normed, q_normed) * scale
        # logits = logits / tau_unsqueezed
        # logits = logits + gamma_scaled * q_norm2.unsqueeze(-1) / (2.0 * tau_sq)

        # m_tilde = torch.softmax(logits, dim=-1)

        # if prefill:
        #     drop_mass = max(0.0, min(1.0, float(self.target_drop_mass)))
        #     coverage = max(0.0, 1.0 - drop_mass)
        #     sorted_vals, sorted_idx = torch.sort(m_tilde, dim=-1, descending=True)
        #     cdf = sorted_vals.cumsum(dim=-1)
        #     coverage_tensor = torch.full(
        #         (batch_size, self.num_heads, 1), coverage, device=self.device, dtype=torch.float32
        #     )
        #     keep_counts = torch.searchsorted(cdf, coverage_tensor, right=True).squeeze(-1)
        #     keep_counts = keep_counts.to(torch.int64)

        #     min_keep = max(min(self.prefill_min_blocks, nr_end), 1)
        #     max_keep = nr_end
        #     keep_counts = keep_counts.clamp(min=min_keep, max=max_keep)

        #     arange_idx = torch.arange(nr_end, device=self.device).view(1, 1, -1)
        #     mask_sorted = arange_idx < keep_counts.unsqueeze(-1)

        #     g_nr = torch.zeros_like(m_tilde, dtype=torch.bool)
        #     g_nr.scatter_(dim=-1, index=sorted_idx, src=mask_sorted)

        #     state.drop_mass.zero_()
        #     state.g_prev[:, :, :nr_end] = g_nr.to(state.g_prev.dtype)
        #     mask[:, :, :nr_end] = g_nr
        #     if debug_masks:
        #         kept_counts = g_nr.sum(dim=-1).float().mean()
        #         kept_mass = (m_tilde * g_nr.float()).sum(dim=-1).mean()
        #         active_ratio = mask.float().mean().item()
        #         sparsity = 1.0 - active_ratio
        #         print(
        #             f"[AdaptiveMask][layer={layer}] prefill keep avg={kept_counts.item():.2f}"
        #             f" mass_avg={kept_mass.item():.3f}"
        #             f" active_ratio={active_ratio:.4f} sparsity={sparsity:.4f}"
        #         )
        # else:
        #     corr = state.y_hat_correction.clamp_min(self.epsilon).unsqueeze(-1)
        #     y_hat = state.y_hat / corr
        #     phi_raw = torch.sum((vbar_nr - y_hat.unsqueeze(2)) ** 2, dim=-1)

        #     mean_phi = phi_raw.mean(dim=-1, keepdim=True)
        #     if nr_end > 1:
        #         var_phi = phi_raw.var(dim=-1, unbiased=True, keepdim=True)
        #     else:
        #         var_phi = torch.full_like(mean_phi, self.epsilon)
        #     var_phi = var_phi.clamp_min(self.epsilon)
        #     phi = (phi_raw - mean_phi) / var_phi.sqrt()

        #     mu_term = self.mu_reg * (1.0 - 2.0 * g_prev_nr)
        #     if state.step_counter == 0:
        #         mu_term = torch.zeros_like(mu_term)
        #     lam_term = state.lam.unsqueeze(-1)

        #     scores = m_tilde * phi - lam_term - mu_term
        #     g_nr_bool = scores > 0
        #     g_nr = g_nr_bool.to(state.g_prev.dtype)

        #     dropped_mass = (m_tilde * (~g_nr_bool).float()).sum(dim=-1)
        #     state.drop_mass.copy_(dropped_mass)


        #     state.g_prev[:, :, :nr_end] = g_nr
        #     mask[:, :, :nr_end] = g_nr_bool

        # state.g_prev[:, :, 0].fill_(1.0)
        # mask[:, :, 0].fill_(True)
        # if debug_masks:
        #     kept_counts = mask[:, :, :nr_end].sum(dim=-1).float().mean()
        #     active_ratio = mask.float().mean().item()
        #     sparsity = 1.0 - active_ratio
        #     print(
        #         f"[AdaptiveMask][layer={layer}] single-query keep avg={kept_counts.item():.2f}"
        #         f" active_ratio={active_ratio:.4f} sparsity={sparsity:.4f}"
        #     )
        # return mask

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
            y_tensor = y.to(self.device, dtype=torch.float32)
        elif y.dim() == 3:
            if y.size(0) == state.batch_size and y.size(1) == self.num_heads:
                y_tensor = y.to(self.device, dtype=torch.float32).unsqueeze(2)
            elif y.size(0) == self.num_heads and state.batch_size == 1:
                y_tensor = y.to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            else:
                raise ValueError("Unsupported shape for y in register_output")
        else:
            raise ValueError("register_output expects a 3D or 4D tensor")

        block = self.q_block_size
        seq_len = y_tensor.size(2)
        num_blocks = (seq_len + block - 1) // block

        if num_blocks == 0:
            state.y_hat = torch.zeros((state.batch_size, self.num_heads, 0, self.head_dim), device=self.device, dtype=torch.float32)
        else:
            total_width = num_blocks * block
            pad_len = total_width - seq_len
            if pad_len > 0:
                y_padded = F.pad(y_tensor, (0, 0, 0, pad_len))
            else:
                y_padded = y_tensor

            y_blocks = y_padded.view(state.batch_size, self.num_heads, num_blocks, block, self.head_dim).to(torch.float32)
            valid = torch.ones((state.batch_size, 1, num_blocks, block, 1), device=y_blocks.device, dtype=torch.float32)
            if pad_len > 0:
                valid[..., -1, -pad_len:, :] = 0.0

            y_sum = (y_blocks * valid).sum(dim=3)
            counts = valid.sum(dim=3).clamp_min(1.0)
            y_block_mean = (y_sum / counts).to(state.y_hat.dtype)

            if state.y_hat.size(2) != num_blocks:
                state.y_hat = torch.zeros((state.batch_size, self.num_heads, num_blocks, self.head_dim), device=self.device, dtype=torch.float32)

            if state.step_counter == 0 or not state.y_hat.abs().any():
                state.y_hat.copy_(y_block_mean)
            else:
                beta = self.beta
                state.y_hat.mul_(1.0 - beta).add_(y_block_mean, alpha=beta)

        candidate_count = state.mu.size(2)

        should_log = debug_masks and (state.step_counter < 3 or state.step_counter % 10 == 0)

        if candidate_count > 0:
            drop_mass = state.drop_mass
            state.tau.add_(self.lr_tau * (drop_mass - self.target_drop_mass)).clamp_(min=self.epsilon)
            if should_log:
                lam_mean = state.lam.mean().item() if state.lam.numel() > 0 else float('nan')
                y_hat_norm = state.y_hat.norm(dim=-1).mean().item() if state.y_hat.numel() > 0 else 0.0
                print(
                    f"[AdaptiveMask][layer={layer}] feedback step={state.step_counter}"
                    f" tau_avg={state.tau.mean().item():.3f} drop_mass_avg={drop_mass.mean().item():.4f}"
                    f" lambda_avg={lam_mean:.3f} y_hat_norm={y_hat_norm:.3f}"
                )
        else:
            state.drop_mass.zero_()
            if should_log:
                print(
                    f"[AdaptiveMask][layer={layer}] feedback (no candidate blocks)"
                    f" tau_avg={state.tau.mean().item():.3f}"
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
