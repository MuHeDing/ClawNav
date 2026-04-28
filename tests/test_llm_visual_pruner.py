import torch

from qwen_vl.model.llm_visual_pruner import (
    TYPE_MIXED,
    TYPE_SEMANTIC_HEAVY,
    TYPE_SPATIAL_HEAVY,
    rebuild_pruned_sequence,
    should_apply_visual_prune,
    type_aware_topk_keep_indices,
)


def test_type_aware_topk_keep_indices_groups_by_type_and_preserves_order():
    scores = torch.tensor([0.2, 0.9, 0.4, 0.8, 0.6], dtype=torch.float32)
    token_types = torch.tensor(
        [
            TYPE_SEMANTIC_HEAVY,
            TYPE_SEMANTIC_HEAVY,
            TYPE_SPATIAL_HEAVY,
            TYPE_SPATIAL_HEAVY,
            TYPE_MIXED,
        ],
        dtype=torch.long,
    )

    keep = type_aware_topk_keep_indices(
        scores=scores,
        token_types=token_types,
        sem_keep_ratio=0.5,
        spa_keep_ratio=0.5,
        mix_keep_ratio=1.0,
    )

    assert keep.tolist() == [1, 3, 4]


def test_rebuild_pruned_sequence_keeps_all_text_tokens_and_pads_batch():
    hidden_states = torch.arange(2 * 6 * 3, dtype=torch.float32).view(2, 6, 3)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    position_ids = torch.tensor(
        [
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 0, 0],
            ],
            [
                [10, 11, 12, 13, 14, 15],
                [10, 11, 12, 13, 0, 0],
            ],
            [
                [20, 21, 22, 23, 24, 25],
                [20, 21, 22, 23, 0, 0],
            ],
        ],
        dtype=torch.long,
    )
    visual_token_mask = torch.tensor(
        [
            [False, True, True, False, True, False],
            [False, True, False, True, False, False],
        ]
    )
    visual_token_types = torch.tensor(
        [
            [-1, TYPE_SEMANTIC_HEAVY, TYPE_SPATIAL_HEAVY, -1, TYPE_MIXED, -1],
            [-1, TYPE_SEMANTIC_HEAVY, -1, TYPE_MIXED, -1, -1],
        ],
        dtype=torch.long,
    )
    keep_indices = [
        torch.tensor([2, 4], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
    ]

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

    assert pruned_hidden_states.shape == (2, 5, 3)
    assert pruned_attention_mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert pruned_visual_mask.tolist() == [
        [False, True, False, True, False],
        [False, True, False, False, False],
    ]
    assert pruned_visual_types.tolist() == [
        [-1, TYPE_SPATIAL_HEAVY, -1, TYPE_MIXED, -1],
        [-1, TYPE_SEMANTIC_HEAVY, -1, -1, -1],
    ]
    assert kept_token_indices[0].tolist() == [0, 2, 3, 4, 5]
    assert kept_token_indices[1].tolist() == [0, 1, 2]
    assert pruned_position_ids[:, 0, :5].tolist() == [
        [0, 2, 3, 4, 5],
        [10, 12, 13, 14, 15],
        [20, 22, 23, 24, 25],
    ]


def test_should_apply_visual_prune_allows_prefill_but_blocks_decode_step():
    visual_mask = torch.tensor([[False, True, True]])

    assert should_apply_visual_prune(
        use_cache=False,
        past_seen_tokens=0,
        visual_token_mask=visual_mask,
    )
    assert should_apply_visual_prune(
        use_cache=True,
        past_seen_tokens=0,
        visual_token_mask=visual_mask,
    )
    assert not should_apply_visual_prune(
        use_cache=True,
        past_seen_tokens=8,
        visual_token_mask=visual_mask,
    )
    assert not should_apply_visual_prune(
        use_cache=False,
        past_seen_tokens=0,
        visual_token_mask=torch.zeros_like(visual_mask),
    )
