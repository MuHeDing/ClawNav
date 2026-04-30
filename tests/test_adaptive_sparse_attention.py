from pathlib import Path
import sys

import torch
import pytest

from qwen_vl.model.adaptive_sparse_attention import (
    build_middle_visual_token_mask,
    can_use_adaptive_sparse_attention,
    install_adaptive_sparse_attention_qwen,
    _load_spargeattn_classes,
    record_adaptive_sparse_flops,
    reset_adaptive_sparse_attention_state,
    summarize_adaptive_sparsity,
)
from evaluation_debug_utils import (
    format_ratio,
    resolve_qualitative_output_path,
    should_save_step_artifacts,
)
from spas_sage_attn.core import estimate_block_sparse_attention_flops


class FakeMasker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reset_all_called = 0

    def reset_all(self):
        self.reset_all_called += 1


class FakeAdaptiveAttention:
    def __init__(self, masker, layer_idx, pvthreshd, smooth_k, stateless):
        self.masker = masker
        self.layer_idx = layer_idx
        self.pvthreshd = pvthreshd
        self.smooth_k = smooth_k
        self.stateless = stateless
        self.log_sparsity = False
        self.sparsity_records = []
        self.prefill_sparsity_records = []
        self.decode_sparsity_records = []


class FakeSelfAttention:
    def __init__(self):
        self.num_heads = 8
        self.head_dim = 64


class FakeLayer:
    def __init__(self):
        self.self_attn = FakeSelfAttention()


class FakeDecoder:
    def __init__(self, num_layers=3):
        self.layers = [FakeLayer() for _ in range(num_layers)]


class FakeModel:
    def __init__(self):
        self.model = FakeDecoder()


class FakeConfig:
    hidden_size = 16
    intermediate_size = 32
    num_attention_heads = 2
    num_key_value_heads = 1


class CallableInner:
    def __init__(self):
        self.calls = []

    def __call__(self, query_states, key_states, value_states, **kwargs):
        self.calls.append((query_states, key_states, value_states, kwargs))
        return query_states


def test_can_use_adaptive_sparse_attention_requires_safe_prefill_shape():
    query_states = torch.empty(1, 8, 128, 64)

    assert can_use_adaptive_sparse_attention(
        query_states=query_states,
        attention_mask=None,
        min_seq_len=128,
        training=False,
        require_cuda=False,
    )
    assert not can_use_adaptive_sparse_attention(
        query_states=torch.empty(2, 8, 128, 64),
        attention_mask=None,
        min_seq_len=128,
        training=False,
        require_cuda=False,
    )
    assert not can_use_adaptive_sparse_attention(
        query_states=torch.empty(1, 8, 64, 64),
        attention_mask=None,
        min_seq_len=128,
        training=False,
        require_cuda=False,
    )
    assert not can_use_adaptive_sparse_attention(
        query_states=query_states,
        attention_mask=torch.ones(1, 128),
        min_seq_len=128,
        training=False,
        require_cuda=False,
    )
    assert not can_use_adaptive_sparse_attention(
        query_states=query_states,
        attention_mask=None,
        min_seq_len=128,
        training=True,
        require_cuda=False,
    )


def test_install_adaptive_sparse_attention_qwen_attaches_layer_wrappers():
    model = FakeModel()

    masker = install_adaptive_sparse_attention_qwen(
        model,
        pvthreshd=1e6,
        stateless=False,
        mask_kwargs={"target_blocks": 79, "target_drop_mass": 0.68},
        adaptive_attention_cls=FakeAdaptiveAttention,
        masker_cls=FakeMasker,
    )

    assert model.adaptive_masker is masker
    assert model.model.adaptive_masker is masker
    assert masker.kwargs["num_layers"] == 3
    assert masker.kwargs["num_heads"] == 8
    assert masker.kwargs["head_dim"] == 64
    assert masker.kwargs["target_blocks"] == 79
    for layer_idx, layer in enumerate(model.model.layers):
        inner = layer.self_attn.adaptive_sparse_attention
        assert isinstance(inner, FakeAdaptiveAttention)
        assert inner.layer_idx == layer_idx
        assert inner.pvthreshd == 1e6
        assert inner.log_sparsity


def test_load_spargeattn_classes_uses_vendored_package_by_default():
    adaptive_attention_cls, masker_cls = _load_spargeattn_classes()

    package_path = Path(__file__).resolve().parents[1] / "src" / "spas_sage_attn"
    assert adaptive_attention_cls.__module__ == "spas_sage_attn.adaptive_attention"
    assert masker_cls.__module__ == "spas_sage_attn.mask_strategies.adaptive_block_mask"
    module_path = Path(sys.modules[adaptive_attention_cls.__module__].__file__).resolve()
    assert package_path in module_path.parents


def test_reset_and_summarize_adaptive_sparse_attention_state():
    model = FakeModel()
    install_adaptive_sparse_attention_qwen(
        model,
        adaptive_attention_cls=FakeAdaptiveAttention,
        masker_cls=FakeMasker,
    )
    model.model.layers[0].self_attn.adaptive_sparse_attention.sparsity_records = [0.4, 0.6]
    model.model.layers[0].self_attn.adaptive_sparse_attention.prefill_sparsity_records = [0.4]
    model.model.layers[0].self_attn.adaptive_sparse_attention.decode_sparsity_records = [0.6]
    model.model.layers[0].self_attn.adaptive_sparse_attention.dense_attention_flops_records = [100.0, 200.0]
    model.model.layers[0].self_attn.adaptive_sparse_attention.saved_attention_flops_records = [40.0, 120.0]
    model.model.layers[0].self_attn.adaptive_sparse_attention.sparse_attention_flops_records = [60.0, 80.0]
    model.model.layers[2].self_attn.adaptive_sparse_attention.sparsity_records = [0.2]
    model.model.layers[2].self_attn.adaptive_sparse_attention.dense_attention_flops_records = [100.0]
    model.model.layers[2].self_attn.adaptive_sparse_attention.saved_attention_flops_records = [20.0]
    model.model.layers[2].self_attn.adaptive_sparse_attention.sparse_attention_flops_records = [80.0]

    summary = summarize_adaptive_sparsity(model)
    reset_adaptive_sparse_attention_state(model)

    assert summary["mean_sparsity"] == 0.35
    assert summary["mean_prefill_sparsity"] == 0.4
    assert summary["mean_decode_sparsity"] == 0.6
    assert summary["dense_attention_flops"] == 400.0
    assert summary["sparse_attention_flops"] == 220.0
    assert summary["saved_attention_flops"] == 180.0
    assert summary["flops_reduction_ratio"] == 0.45
    assert summary["self_attention_flops_reduction_ratio"] == 0.45
    assert summary["sparse_attention_latency_ms"] == 0.0
    assert summary["sparse_attention_effective_tops"] is None
    assert summary["sparse_attention_actual_tops"] is None
    assert summary["full_attention_latency_ms"] is None
    assert summary["full_attention_effective_tops"] is None
    assert summary["dense_llm_layer_flops"] == 0.0
    assert summary["sparse_llm_layer_flops"] == 0.0
    assert summary["saved_llm_layer_flops"] == 0.0
    assert summary["llm_layer_flops_reduction_ratio"] is None
    assert summary["estimated_total_llm_flops"] == 0.0
    assert summary["estimated_total_llm_saved_flops"] == 0.0
    assert summary["estimated_total_llm_sparse_flops"] == 0.0
    assert summary["estimated_total_llm_flops_reduction_ratio"] is None
    assert summary["layers"] == [
        {
            "layer": 0,
            "mean_sparsity": 0.5,
            "count": 2,
            "mean_prefill_sparsity": 0.4,
            "prefill_count": 1,
            "mean_decode_sparsity": 0.6,
            "decode_count": 1,
            "dense_attention_flops": 300.0,
            "sparse_attention_flops": 140.0,
            "saved_attention_flops": 160.0,
            "flops_reduction_ratio": 0.533333,
            "self_attention_flops_reduction_ratio": 0.533333,
            "sparse_attention_latency_ms": 0.0,
            "sparse_attention_effective_tops": None,
            "sparse_attention_actual_tops": None,
            "full_attention_latency_ms": None,
            "full_attention_effective_tops": None,
            "dense_llm_layer_flops": 0.0,
            "sparse_llm_layer_flops": 0.0,
            "saved_llm_layer_flops": 0.0,
            "llm_layer_flops_reduction_ratio": None,
        },
        {
            "layer": 2,
            "mean_sparsity": 0.2,
            "count": 1,
            "mean_prefill_sparsity": None,
            "prefill_count": 0,
            "mean_decode_sparsity": None,
            "decode_count": 0,
            "dense_attention_flops": 100.0,
            "sparse_attention_flops": 80.0,
            "saved_attention_flops": 20.0,
            "flops_reduction_ratio": 0.2,
            "self_attention_flops_reduction_ratio": 0.2,
            "sparse_attention_latency_ms": 0.0,
            "sparse_attention_effective_tops": None,
            "sparse_attention_actual_tops": None,
            "full_attention_latency_ms": None,
            "full_attention_effective_tops": None,
            "dense_llm_layer_flops": 0.0,
            "sparse_llm_layer_flops": 0.0,
            "saved_llm_layer_flops": 0.0,
            "llm_layer_flops_reduction_ratio": None,
        },
    ]
    assert model.adaptive_masker.reset_all_called == 1


def test_summarize_adaptive_sparse_attention_ignores_disabled_llm_layer_savings():
    model = FakeModel()
    model.model = FakeDecoder(num_layers=4)
    install_adaptive_sparse_attention_qwen(
        model,
        adaptive_attention_cls=FakeAdaptiveAttention,
        masker_cls=FakeMasker,
    )
    for layer_idx in [2, 3]:
        inner = model.model.layers[layer_idx].self_attn.adaptive_sparse_attention
        inner.sparsity_records = [0.1]
        inner.dense_attention_flops_records = [100.0]
        inner.saved_attention_flops_records = [10.0]
        inner.sparse_attention_flops_records = [90.0]
        inner.dense_llm_layer_flops_records = [1000.0]
        inner.saved_llm_layer_flops_records = [10.0]
        inner.sparse_llm_layer_flops_records = [990.0]

    summary = summarize_adaptive_sparsity(model)

    assert summary["dense_llm_layer_flops"] == 0.0
    assert summary["saved_llm_layer_flops"] == 0.0
    assert summary["sparse_llm_layer_flops"] == 0.0
    assert summary["llm_layer_flops_reduction_ratio"] is None
    assert summary["estimated_total_llm_num_layers"] == 4
    assert summary["instrumented_sparse_layers"] == 2
    assert summary["estimated_total_llm_flops"] == 0.0
    assert summary["estimated_total_llm_saved_flops"] == 0.0
    assert summary["estimated_total_llm_sparse_flops"] == 0.0
    assert summary["estimated_total_llm_flops_reduction_ratio"] is None


def test_record_adaptive_sparse_flops_records_self_attention_only():
    inner = FakeAdaptiveAttention(None, 0, 1e6, True, False)
    inner.sparsity_records = [0.25]
    query_states = torch.empty(1, 2, 4, 8)
    key_states = torch.empty(1, 1, 6, 8)

    record_adaptive_sparse_flops(inner, query_states, key_states, config=FakeConfig())

    dense_attention_flops = 4.0 * 1 * 2 * 4 * 6 * 8
    saved_flops = dense_attention_flops * 0.25

    assert inner.dense_attention_flops_records == [dense_attention_flops]
    assert inner.saved_attention_flops_records == [saved_flops]
    assert inner.sparse_attention_flops_records == [dense_attention_flops - saved_flops]
    assert not hasattr(inner, "dense_llm_layer_flops_records")
    assert not hasattr(inner, "saved_llm_layer_flops_records")
    assert not hasattr(inner, "sparse_llm_layer_flops_records")


def test_estimate_block_sparse_attention_flops_counts_partial_blocks():
    mask_blocks = torch.tensor([[[[True, False], [False, True]]]])

    stats = estimate_block_sparse_attention_flops(
        mask_blocks,
        q_len=3,
        kv_len=5,
        q_block_size=2,
        kv_block_size=4,
        head_dim=8,
    )

    assert stats["dense_attention_flops"] == 480.0
    assert stats["sparse_attention_flops"] == 288.0
    assert stats["saved_attention_flops"] == 192.0
    assert stats["flops_reduction_ratio"] == 0.4


def test_estimate_block_sparse_attention_flops_sums_batch_and_heads_once():
    mask_blocks = torch.ones(2, 3, 1, 1, dtype=torch.bool)

    stats = estimate_block_sparse_attention_flops(
        mask_blocks,
        q_len=2,
        kv_len=3,
        q_block_size=4,
        kv_block_size=4,
        head_dim=8,
    )

    expected_flops = 4.0 * 2 * 3 * 2 * 3 * 8
    assert stats["dense_attention_flops"] == expected_flops
    assert stats["sparse_attention_flops"] == expected_flops
    assert stats["saved_attention_flops"] == 0.0


def test_record_adaptive_sparse_flops_prefers_block_sparse_kernel_flops():
    inner = FakeAdaptiveAttention(None, 0, 1e6, True, False)
    inner.sparsity_records = [0.5]
    inner.kernel_dense_attention_flops_records = [480.0]
    inner.kernel_sparse_attention_flops_records = [288.0]
    inner.kernel_saved_attention_flops_records = [192.0]
    query_states = torch.empty(1, 2, 4, 8)
    key_states = torch.empty(1, 1, 6, 8)

    record_adaptive_sparse_flops(inner, query_states, key_states, config=FakeConfig())

    assert inner.dense_attention_flops_records == [480.0]
    assert inner.sparse_attention_flops_records == [288.0]
    assert inner.saved_attention_flops_records == [192.0]
    assert not hasattr(inner, "saved_llm_layer_flops_records")


def test_record_adaptive_sparse_flops_records_effective_and_actual_tops():
    inner = FakeAdaptiveAttention(None, 0, 1e6, True, False)
    inner.sparsity_records = [0.5]
    inner.kernel_dense_attention_flops_records = [2_000_000_000_000.0]
    inner.kernel_sparse_attention_flops_records = [1_000_000_000_000.0]
    inner.kernel_saved_attention_flops_records = [1_000_000_000_000.0]

    record_adaptive_sparse_flops(
        inner,
        torch.empty(1, 2, 4, 8),
        torch.empty(1, 1, 6, 8),
        sparse_latency_ms=2.0,
        full_attention_latency_ms=4.0,
    )

    assert inner.sparse_attention_latency_ms_records == [2.0]
    assert inner.sparse_attention_effective_tops_records == [1000.0]
    assert inner.sparse_attention_actual_tops_records == [500.0]
    assert inner.full_attention_latency_ms_records == [4.0]
    assert inner.full_attention_effective_tops_records == [500.0]


def test_summarize_adaptive_sparse_attention_reports_aggregate_tops():
    model = FakeModel()
    install_adaptive_sparse_attention_qwen(
        model,
        adaptive_attention_cls=FakeAdaptiveAttention,
        masker_cls=FakeMasker,
    )
    inner = model.model.layers[0].self_attn.adaptive_sparse_attention
    inner.sparsity_records = [0.5, 0.5]
    inner.dense_attention_flops_records = [2_000_000_000_000.0, 1_000_000_000_000.0]
    inner.saved_attention_flops_records = [1_000_000_000_000.0, 500_000_000_000.0]
    inner.sparse_attention_flops_records = [1_000_000_000_000.0, 500_000_000_000.0]
    inner.sparse_attention_latency_ms_records = [2.0, 4.0]
    inner.full_attention_latency_ms_records = [4.0, 6.0]

    summary = summarize_adaptive_sparsity(model)

    assert summary["sparse_attention_latency_ms"] == 6.0
    assert summary["sparse_attention_effective_tops"] == 500.0
    assert summary["sparse_attention_actual_tops"] == 250.0
    assert summary["full_attention_latency_ms"] == 10.0
    assert summary["full_attention_effective_tops"] == 300.0
    assert summary["layers"][0]["sparse_attention_effective_tops"] == 500.0
    assert summary["layers"][0]["sparse_attention_actual_tops"] == 250.0
    assert summary["layers"][0]["full_attention_effective_tops"] == 300.0


def test_record_adaptive_sparse_flops_uses_latest_sparsity_record():
    inner = FakeAdaptiveAttention(None, 0, 1e6, True, False)
    inner.sparsity_records = [0.25]
    query_states = torch.empty(1, 2, 4, 8)
    key_states = torch.empty(1, 2, 6, 8)

    record_adaptive_sparse_flops(inner, query_states, key_states)

    dense_flops = 4.0 * 1 * 2 * 4 * 6 * 8
    assert inner.dense_attention_flops_records == [dense_flops]
    assert inner.saved_attention_flops_records == [dense_flops * 0.25]
    assert inner.sparse_attention_flops_records == [dense_flops * 0.75]


def test_build_middle_visual_token_mask_keeps_text_start_and_recent_dense():
    image_token_id = 99
    input_ids = torch.tensor(
        [
            [
                1,
                image_token_id,
                image_token_id,
                2,
                image_token_id,
                image_token_id,
                3,
                image_token_id,
                image_token_id,
                4,
                image_token_id,
                image_token_id,
                5,
            ]
        ]
    )

    mask = build_middle_visual_token_mask(
        input_ids,
        image_token_id=image_token_id,
        kv_block_size=2,
        llm_start_blocks=1,
        llm_recent_blocks=1,
    )

    assert mask.tolist() == [[False, False, False, False, True, True, False, True, True, False, False, False, False]]


def test_build_middle_visual_token_mask_uses_visual_token_blocks_not_image_runs():
    image_token_id = 99
    input_ids = torch.tensor(
        [
            [
                1,
                image_token_id,
                image_token_id,
                image_token_id,
                2,
                image_token_id,
                image_token_id,
                image_token_id,
                3,
                image_token_id,
                image_token_id,
                image_token_id,
                4,
            ]
        ]
    )

    mask = build_middle_visual_token_mask(
        input_ids,
        image_token_id=image_token_id,
        kv_block_size=3,
        llm_start_blocks=1,
        llm_recent_blocks=1,
    )

    assert mask.tolist() == [[False, False, False, False, False, True, True, True, False, False, False, False, False]]


def test_evaluation_debug_step_artifacts_are_explicitly_enabled(tmp_path):
    assert not should_save_step_artifacts(
        save_step_artifacts=False,
        should_save_video=False,
        save_step_artifacts_with_video_only=False,
    )
    assert not should_save_step_artifacts(
        save_step_artifacts=True,
        should_save_video=False,
        save_step_artifacts_with_video_only=True,
    )
    assert should_save_step_artifacts(
        save_step_artifacts=True,
        should_save_video=False,
        save_step_artifacts_with_video_only=False,
    )
    assert should_save_step_artifacts(
        save_step_artifacts=True,
        should_save_video=True,
        save_step_artifacts_with_video_only=True,
    )

    assert resolve_qualitative_output_path(
        output_path=tmp_path,
        rank=1,
        disabled=False,
    ) == tmp_path / "qualitative_trajectories_rank1.json"
    assert resolve_qualitative_output_path(
        output_path=tmp_path,
        rank=1,
        disabled=True,
    ) is None


def test_format_ratio_keeps_two_decimal_places():
    assert format_ratio(0.042317) == "0.04"
    assert format_ratio(0.076121) == "0.08"
    assert format_ratio(None) == "None"


def test_qwen_attention_sparse_helper_calls_inner_when_gate_allows(monkeypatch):
    qwen_modeling = pytest.importorskip(
        "qwen_vl.model.modeling_qwen2_5_vl",
        reason="Qwen modeling import depends on the local transformers environment",
    )
    Qwen2_5_VLAttention = qwen_modeling.Qwen2_5_VLAttention
    attn = Qwen2_5_VLAttention.__new__(Qwen2_5_VLAttention)
    attn.config = type("Config", (), {"adaptive_sparse_min_seq_len": 4})()
    attn.training = False
    attn.is_causal = True
    attn.layer_idx = 2
    inner = CallableInner()
    attn.adaptive_sparse_attention = inner

    monkeypatch.setattr(
        qwen_modeling,
        "can_use_adaptive_sparse_attention",
        lambda **kwargs: True,
    )
    query_states = torch.ones(1, 2, 4, 8)
    key_states = torch.ones(1, 2, 4, 8)
    value_states = torch.ones(1, 2, 4, 8)

    output = attn._adaptive_sparse_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask=None,
    )

    assert output is query_states
    assert len(inner.calls) == 1
    assert inner.calls[0][3]["is_causal"] is True
    assert inner.calls[0][3]["tensor_layout"] == "HND"


def test_qwen_attention_sparse_helper_returns_none_when_gate_rejects(monkeypatch):
    qwen_modeling = pytest.importorskip(
        "qwen_vl.model.modeling_qwen2_5_vl",
        reason="Qwen modeling import depends on the local transformers environment",
    )
    Qwen2_5_VLAttention = qwen_modeling.Qwen2_5_VLAttention
    attn = Qwen2_5_VLAttention.__new__(Qwen2_5_VLAttention)
    attn.config = type("Config", (), {"adaptive_sparse_min_seq_len": 4})()
    attn.training = False
    attn.is_causal = True
    attn.layer_idx = 2
    inner = CallableInner()
    attn.adaptive_sparse_attention = inner

    monkeypatch.setattr(
        qwen_modeling,
        "can_use_adaptive_sparse_attention",
        lambda **kwargs: False,
    )

    output = attn._adaptive_sparse_attention_forward(
        torch.ones(1, 2, 4, 8),
        torch.ones(1, 2, 4, 8),
        torch.ones(1, 2, 4, 8),
        attention_mask=None,
    )

    assert output is None
    assert inner.calls == []


def test_qwen_attention_sparse_helper_respects_start_layer(monkeypatch):
    qwen_modeling = pytest.importorskip(
        "qwen_vl.model.modeling_qwen2_5_vl",
        reason="Qwen modeling import depends on the local transformers environment",
    )
    Qwen2_5_VLAttention = qwen_modeling.Qwen2_5_VLAttention
    attn = Qwen2_5_VLAttention.__new__(Qwen2_5_VLAttention)
    attn.config = type(
        "Config",
        (),
        {"adaptive_sparse_min_seq_len": 4, "adaptive_sparse_llm_start_layer": 14},
    )()
    attn.training = False
    attn.is_causal = True
    attn.layer_idx = 13
    inner = CallableInner()
    attn.adaptive_sparse_attention = inner

    monkeypatch.setattr(
        qwen_modeling,
        "can_use_adaptive_sparse_attention",
        lambda **kwargs: True,
    )

    output = attn._adaptive_sparse_attention_forward(
        torch.ones(1, 2, 4, 8),
        torch.ones(1, 2, 4, 8),
        torch.ones(1, 2, 4, 8),
        attention_mask=None,
    )

    assert output is None
    assert inner.calls == []
