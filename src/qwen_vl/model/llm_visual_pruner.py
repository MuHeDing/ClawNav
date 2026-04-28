import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


TYPE_SEMANTIC_HEAVY = 0
TYPE_SPATIAL_HEAVY = 1
TYPE_MIXED = 2
NON_VISUAL_TYPE = -1


def should_apply_visual_prune(
    use_cache: bool,
    past_seen_tokens: int,
    visual_token_mask: Optional[torch.Tensor],
) -> bool:
    if visual_token_mask is None or not bool(visual_token_mask.any().item()):
        return False
    if not use_cache:
        return True
    return past_seen_tokens == 0


def _ceil_keep_count(num_tokens: int, keep_ratio: float) -> int:
    if num_tokens <= 0:
        return 0
    if keep_ratio <= 0:
        return 0
    return min(num_tokens, int(math.ceil(num_tokens * keep_ratio)))


def type_aware_topk_keep_indices(
    scores: torch.Tensor,
    token_types: torch.Tensor,
    sem_keep_ratio: float,
    spa_keep_ratio: float,
    mix_keep_ratio: float,
) -> torch.Tensor:
    keep_indices = []
    type_specs = (
        (TYPE_SEMANTIC_HEAVY, sem_keep_ratio),
        (TYPE_SPATIAL_HEAVY, spa_keep_ratio),
        (TYPE_MIXED, mix_keep_ratio),
    )

    for token_type, keep_ratio in type_specs:
        type_positions = torch.nonzero(token_types == token_type, as_tuple=False).flatten()
        if type_positions.numel() == 0:
            continue

        keep_count = _ceil_keep_count(type_positions.numel(), keep_ratio)
        if keep_count == 0:
            continue

        type_scores = scores.index_select(0, type_positions)
        topk_relative = torch.topk(type_scores, k=keep_count, largest=True, sorted=False).indices
        selected = type_positions.index_select(0, topk_relative)
        keep_indices.append(selected)

    if not keep_indices:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    keep_indices = torch.cat(keep_indices, dim=0)
    keep_indices, _ = torch.sort(keep_indices)
    return keep_indices


def rebuild_pruned_sequence(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    visual_token_mask: torch.Tensor,
    visual_token_types: torch.LongTensor,
    keep_indices: Sequence[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.LongTensor, List[torch.LongTensor]]:
    batch_size, _, hidden_dim = hidden_states.shape
    kept_token_indices: List[torch.LongTensor] = []
    kept_lengths: List[int] = []

    for batch_idx in range(batch_size):
        active_positions = torch.nonzero(attention_mask[batch_idx].to(torch.bool), as_tuple=False).flatten()
        active_visual_mask = visual_token_mask[batch_idx].index_select(0, active_positions)
        text_positions = active_positions[~active_visual_mask]

        sample_keep = keep_indices[batch_idx].to(device=hidden_states.device, dtype=torch.long)
        if sample_keep.numel() > 0:
            sample_keep = torch.unique(sample_keep, sorted=True)

        merged_positions = torch.cat([text_positions, sample_keep], dim=0)
        if merged_positions.numel() == 0:
            merged_positions = active_positions[:1]
        merged_positions, _ = torch.sort(merged_positions)

        kept_token_indices.append(merged_positions)
        kept_lengths.append(merged_positions.numel())

    max_kept_length = max(kept_lengths)
    pruned_hidden_states = hidden_states.new_zeros(batch_size, max_kept_length, hidden_dim)
    pruned_attention_mask = attention_mask.new_zeros(batch_size, max_kept_length)
    pruned_position_ids = position_ids.new_zeros(position_ids.shape[0], batch_size, max_kept_length)
    pruned_visual_mask = visual_token_mask.new_zeros(batch_size, max_kept_length)
    pruned_visual_types = visual_token_types.new_full((batch_size, max_kept_length), NON_VISUAL_TYPE)

    for batch_idx, merged_positions in enumerate(kept_token_indices):
        kept_length = merged_positions.numel()
        if kept_length == 0:
            continue

        pruned_hidden_states[batch_idx, :kept_length] = hidden_states[batch_idx].index_select(0, merged_positions)
        pruned_attention_mask[batch_idx, :kept_length] = 1
        pruned_position_ids[:, batch_idx, :kept_length] = position_ids[:, batch_idx].index_select(1, merged_positions)
        pruned_visual_mask[batch_idx, :kept_length] = visual_token_mask[batch_idx].index_select(0, merged_positions)
        pruned_visual_types[batch_idx, :kept_length] = visual_token_types[batch_idx].index_select(0, merged_positions)

    return (
        pruned_hidden_states,
        pruned_attention_mask,
        pruned_position_ids,
        pruned_visual_mask,
        pruned_visual_types,
        kept_token_indices,
    )


@dataclass
class VisualPruneStats:
    visual_before: int
    visual_after: int
    sem_before: int
    spa_before: int
    mix_before: int
    sem_kept: int
    spa_kept: int
    mix_kept: int
    sequence_before: int
    sequence_after: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "visual_before": self.visual_before,
            "visual_after": self.visual_after,
            "sem_before": self.sem_before,
            "spa_before": self.spa_before,
            "mix_before": self.mix_before,
            "sem_kept": self.sem_kept,
            "spa_kept": self.spa_kept,
            "mix_kept": self.mix_kept,
            "sequence_before": self.sequence_before,
            "sequence_after": self.sequence_after,
        }


class TypeAwareVisualTokenPruner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        sem_keep_ratio: float = 0.4,
        spa_keep_ratio: float = 0.7,
        mix_keep_ratio: float = 0.55,
        type_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.sem_keep_ratio = sem_keep_ratio
        self.spa_keep_ratio = spa_keep_ratio
        self.mix_keep_ratio = mix_keep_ratio
        self.type_embed_dim = type_embed_dim or max(hidden_size // 4, 32)

        self.type_embedding = nn.Embedding(3, self.type_embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 2 + self.type_embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def build_instruction_summary(
        self,
        hidden_states: torch.Tensor,
        visual_token_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = attention_mask.to(torch.bool)
        text_mask = valid_mask & (~visual_token_mask)
        fallback_mask = valid_mask

        summaries = []
        for batch_idx in range(hidden_states.shape[0]):
            active_text = hidden_states[batch_idx][text_mask[batch_idx]]
            if active_text.numel() == 0:
                active_text = hidden_states[batch_idx][fallback_mask[batch_idx]]
            if active_text.numel() == 0:
                active_text = hidden_states[batch_idx, :1]
            summaries.append(active_text.mean(dim=0))
        return torch.stack(summaries, dim=0)

    def score_visual_tokens(
        self,
        visual_hidden_states: torch.Tensor,
        instruction_hidden_state: torch.Tensor,
        visual_token_types: torch.LongTensor,
    ) -> torch.Tensor:
        instruction_hidden_state = instruction_hidden_state.unsqueeze(0).expand(visual_hidden_states.shape[0], -1)
        type_emb = self.type_embedding(visual_token_types)
        scorer_inputs = torch.cat([visual_hidden_states, instruction_hidden_state, type_emb], dim=-1)
        return self.scorer(scorer_inputs).squeeze(-1)

    def prune_one_sample(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_token_mask: torch.Tensor,
        visual_token_types: torch.LongTensor,
        instruction_hidden_state: torch.Tensor,
    ) -> Tuple[torch.LongTensor, VisualPruneStats]:
        active_positions = torch.nonzero(attention_mask.to(torch.bool), as_tuple=False).flatten()
        active_visual_positions = active_positions[visual_token_mask.index_select(0, active_positions)]
        active_visual_types = visual_token_types.index_select(0, active_visual_positions)

        if active_visual_positions.numel() == 0:
            empty_stats = VisualPruneStats(
                visual_before=0,
                visual_after=0,
                sem_before=0,
                spa_before=0,
                mix_before=0,
                sem_kept=0,
                spa_kept=0,
                mix_kept=0,
                sequence_before=int(active_positions.numel()),
                sequence_after=int(active_positions.numel()),
            )
            return active_visual_positions, empty_stats

        visual_hidden_states = hidden_states.index_select(0, active_visual_positions)
        scores = self.score_visual_tokens(visual_hidden_states, instruction_hidden_state, active_visual_types)
        kept_relative_indices = type_aware_topk_keep_indices(
            scores=scores,
            token_types=active_visual_types,
            sem_keep_ratio=self.sem_keep_ratio,
            spa_keep_ratio=self.spa_keep_ratio,
            mix_keep_ratio=self.mix_keep_ratio,
        )
        kept_absolute_indices = active_visual_positions.index_select(0, kept_relative_indices)

        stats = VisualPruneStats(
            visual_before=int(active_visual_positions.numel()),
            visual_after=int(kept_absolute_indices.numel()),
            sem_before=int((active_visual_types == TYPE_SEMANTIC_HEAVY).sum().item()),
            spa_before=int((active_visual_types == TYPE_SPATIAL_HEAVY).sum().item()),
            mix_before=int((active_visual_types == TYPE_MIXED).sum().item()),
            sem_kept=int((active_visual_types.index_select(0, kept_relative_indices) == TYPE_SEMANTIC_HEAVY).sum().item()) if kept_relative_indices.numel() > 0 else 0,
            spa_kept=int((active_visual_types.index_select(0, kept_relative_indices) == TYPE_SPATIAL_HEAVY).sum().item()) if kept_relative_indices.numel() > 0 else 0,
            mix_kept=int((active_visual_types.index_select(0, kept_relative_indices) == TYPE_MIXED).sum().item()) if kept_relative_indices.numel() > 0 else 0,
            sequence_before=int(active_positions.numel()),
            sequence_after=int(active_positions.numel() - active_visual_positions.numel() + kept_absolute_indices.numel()),
        )
        return kept_absolute_indices, stats

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        visual_token_mask: torch.Tensor,
        visual_token_types: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.LongTensor, List[VisualPruneStats], List[torch.LongTensor]]:
        instruction_summary = self.build_instruction_summary(hidden_states, visual_token_mask, attention_mask)

        keep_indices: List[torch.LongTensor] = []
        batch_stats: List[VisualPruneStats] = []
        for batch_idx in range(hidden_states.shape[0]):
            sample_keep_indices, sample_stats = self.prune_one_sample(
                hidden_states=hidden_states[batch_idx],
                attention_mask=attention_mask[batch_idx],
                visual_token_mask=visual_token_mask[batch_idx],
                visual_token_types=visual_token_types[batch_idx],
                instruction_hidden_state=instruction_summary[batch_idx],
            )
            keep_indices.append(sample_keep_indices)
            batch_stats.append(sample_stats)

        (
            pruned_hidden_states,
            pruned_attention_mask,
            pruned_position_ids,
            pruned_visual_mask,
            pruned_visual_types,
            kept_token_indices,
        ) = rebuild_pruned_sequence(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_token_mask=visual_token_mask,
            visual_token_types=visual_token_types,
            keep_indices=keep_indices,
        )

        return (
            pruned_hidden_states,
            pruned_attention_mask,
            pruned_position_ids,
            pruned_visual_mask,
            pruned_visual_types,
            batch_stats,
            kept_token_indices,
        )
