"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from .utils import (
    hyperparameter_check,
    get_block_map_meansim,
    get_block_map_meansim_fuse_quant,
    get_vanilla_qk_quant,
    block_map_lut_triton,
)
from .mask_strategies import AdaptiveBlockMasker
from einops import rearrange

import spas_sage_attn._qattn as qattn
import spas_sage_attn._fused as fused


@dataclass
class StageSparsity:
    """Holds sparsity statistics for prefill and decode phases."""

    prefill: Optional[float] = None
    decode: Optional[float] = None
    dense_attention_flops: Optional[float] = None
    sparse_attention_flops: Optional[float] = None
    saved_attention_flops: Optional[float] = None


def _causal_pair_count(
    q_start: int,
    q_end: int,
    kv_start: int,
    kv_end: int,
) -> int:
    """Count token pairs in a block rectangle that satisfy kv_position <= q_position."""

    if q_end <= q_start or kv_end <= kv_start:
        return 0
    if kv_end <= q_start:
        return (q_end - q_start) * (kv_end - kv_start)
    if kv_start >= q_end:
        return 0

    ramp_start = max(q_start, kv_start)
    ramp_end = min(q_end - 1, kv_end - 1)
    ramp_pairs = 0
    if ramp_start <= ramp_end:
        ramp_len = ramp_end - ramp_start + 1
        first = ramp_start - kv_start + 1
        last = ramp_end - kv_start + 1
        ramp_pairs = ramp_len * (first + last) // 2

    full_start = max(q_start, kv_end)
    full_rows = max(q_end - full_start, 0)
    full_pairs = full_rows * (kv_end - kv_start)
    return ramp_pairs + full_pairs


def estimate_block_sparse_attention_flops(
    mask_blocks: torch.Tensor,
    *,
    q_len: int,
    kv_len: int,
    q_block_size: int,
    kv_block_size: int,
    head_dim: int,
    is_causal: bool = False,
) -> dict[str, float]:
    """Estimate QK plus AV FLOPs from the final active block mask."""

    if mask_blocks.dim() != 4:
        raise ValueError("mask_blocks must have shape (batch, heads, q_blocks, kv_blocks)")

    batch, heads, num_q_blocks, num_kv_blocks = mask_blocks.shape
    q_len = int(q_len)
    kv_len = int(kv_len)
    q_block_size = int(q_block_size)
    kv_block_size = int(kv_block_size)
    head_dim = int(head_dim)
    if q_len <= 0 or kv_len <= 0 or head_dim <= 0 or q_block_size <= 0 or kv_block_size <= 0:
        return {
            "dense_attention_flops": 0.0,
            "sparse_attention_flops": 0.0,
            "saved_attention_flops": 0.0,
            "flops_reduction_ratio": 0.0,
        }

    use_causal_counts = bool(is_causal and q_len == kv_len)
    if use_causal_counts:
        dense_pairs = q_len * (q_len + 1) // 2
    else:
        dense_pairs = q_len * kv_len
    dense_flops = float(4.0 * int(batch) * int(heads) * dense_pairs * head_dim)

    pair_counts = mask_blocks.new_zeros((num_q_blocks, num_kv_blocks), dtype=torch.float32)
    for q_block_idx in range(num_q_blocks):
        q_start = q_block_idx * q_block_size
        q_end = min(q_start + q_block_size, q_len)
        if q_start >= q_end:
            continue
        for kv_block_idx in range(num_kv_blocks):
            kv_start = kv_block_idx * kv_block_size
            kv_end = min(kv_start + kv_block_size, kv_len)
            if kv_start >= kv_end:
                continue
            if use_causal_counts:
                pairs = _causal_pair_count(q_start, q_end, kv_start, kv_end)
            else:
                pairs = (q_end - q_start) * (kv_end - kv_start)
            pair_counts[q_block_idx, kv_block_idx] = float(pairs)

    sparse_pairs = (
        mask_blocks.to(torch.float32) * pair_counts.view(1, 1, num_q_blocks, num_kv_blocks)
    ).sum()
    sparse_flops = float((4.0 * sparse_pairs * head_dim).item())
    sparse_flops = min(sparse_flops, dense_flops)
    saved_flops = max(dense_flops - sparse_flops, 0.0)
    reduction = saved_flops / dense_flops if dense_flops > 0 else 0.0
    return {
        "dense_attention_flops": dense_flops,
        "sparse_attention_flops": sparse_flops,
        "saved_attention_flops": saved_flops,
        "flops_reduction_ratio": float(reduction),
    }


def _run_block_sparse_kernel(
    q_chunk: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    km: Optional[torch.Tensor],
    mask_blocks: torch.Tensor,
    pvthreshd_vec: torch.Tensor,
    scale: float,
    use_sm90: bool,
    is_causal: bool,
):
    """Helper to quantize inputs and dispatch the sparse kernel once."""

    lut, valid_block_num = block_map_lut_triton(mask_blocks)
    if use_sm90:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q_chunk, k, km, 64, 128)
    else:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q_chunk, k, km)

    out = torch.empty_like(q_chunk)
    _is_causal = 1 if is_causal else 0
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
        q_int8,
        k_int8,
        v,
        out,
        lut,
        valid_block_num,
        pvthreshd_vec,
        q_scale,
        k_scale,
        1,
        _is_causal,
        1,
        scale,
        0,
    )

    return out, valid_block_num

def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs

@torch.compiler.disable
def spas_sage_attn_meansim_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink)  # 

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    _is_causal = 1 if is_causal else 0
    o = torch.empty_like(q)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o
    
@torch.compiler.disable
def block_sparse_sage2_attn_cuda(q, k, v, mask_id=None, dropout_p=0.0, scale=None, smooth_k=True, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)
    
    arch = get_cuda_arch_versions()[q.device.index]
    
    if arch == "sm90":
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 64, 128)
    else:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km)
    lut, valid_block_num = block_map_lut_triton(block_map=mask_id)
    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    ## quant v
    b, h_kv, kv_len, head_dim = v.shape
    padded_len = (kv_len + 63) // 64 * 64
    v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
    fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
    v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
    fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 448.0, 1)

    o = torch.empty_like(q)
    
    if arch == "sm90":
        qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
    else:
        qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
    
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o
    
@torch.compiler.disable
def adaptive_block_sparse_attn_step(
    masker: AdaptiveBlockMasker,
    layer: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    smooth_k: bool = True,
    pvthreshd: float = 50.0,
    tensor_layout: str = "HND",
    return_sparsity: bool = False,
    sparse_token_mask: Optional[torch.Tensor] = None,
):
    """Run sparse attention using the adaptive mask strategy.

    The routine automatically detects the prefill phase (when no incremental
    ``k_new``/``v_new`` chunk is supplied). During prefill it evaluates masks
    using the mean-similarity terms only—equivalent to setting ``phi=1`` and
    freezing the lambda/tau feedback—so the entire sequence can be processed in
    a single block-sparse kernel call. Decode steps fall back to the fully
    adaptive block-by-block updates.

    Args:
        masker: AdaptiveBlockMasker instance maintaining per-layer state.
        layer: Layer index whose statistics should be updated.
        q: Query tensor shaped ``(B, H, L, D)`` ("HND") or ``(B, L, H, D)``
            ("NHD"). ``L`` may span multiple blocks of ``masker.q_block_size``.
        k, v: Full key/value tensors matching ``tensor_layout``. Used both for
            statistics (ingest) and the actual attention call.
        k_new, v_new: Optional incremental KV chunk to update the tail block.
            Accepts the same layouts as ``k``/``v`` as well as squeezed
            variants ``(H, T, D)`` or ``(H, D)``.
        is_causal: Whether to enforce causal masking inside the sparse kernel.
        scale: Optional softmax scale override. If ``None`` the kernel default
            (``1/sqrt(head_dim)``) is used.
        smooth_k: Whether to subtract the mean from K blocks when updating the
            statistics and calling the kernel.
        pvthreshd: Per-head PV threshold passed to the block-sparse kernel.
            Set to a very large value to effectively disable PV pruning.
        tensor_layout: Either ``"HND"`` or ``"NHD"``.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, float]:
            Attention output with the same layout as ``q``. When
            ``return_sparsity`` is True, a tuple ``(output, sparsity)`` is
            returned where ``sparsity`` is the fraction of masked-off blocks.
    """

    if not isinstance(masker, AdaptiveBlockMasker):
        raise TypeError("masker must be an instance of AdaptiveBlockMasker")

    assert tensor_layout in ["HND", "NHD"], "tensor_layout must be 'HND' or 'NHD'"

    original_layout = tensor_layout
    if tensor_layout == "NHD":
        q = rearrange(q, 'B L H D -> B H L D').contiguous()
        k = rearrange(k, 'B L H D -> B H L D').contiguous()
        v = rearrange(v, 'B L H D -> B H L D').contiguous()
        tensor_layout = "HND"

    if q.dim() != 4:
        raise ValueError("q must have four dimensions (batch, heads, seq, dim)")

    debug_masks = os.environ.get("ADAPTIVE_DEBUG", "0") == "1"
    if debug_masks:

        def _shape_str(tensor: Optional[torch.Tensor]) -> str:
            return "None" if tensor is None else "x".join(str(dim) for dim in tensor.shape)

        print(
            f"[AdaptiveCore][layer={layer}] mode=full-pass "
            f"q_shape={_shape_str(q)} k_shape={_shape_str(k)} v_shape={_shape_str(v)} "
            f"layout={original_layout}->{tensor_layout}"
        )

    if k_new is not None or v_new is not None:
        raise NotImplementedError("Incremental KV updates are disabled in diffusion mode")

    masker.ingest_kv(layer, k, v, tensor_layout=tensor_layout)

    dtype = q.dtype
    if dtype in (torch.float32, torch.float16):
        q = q.contiguous().to(torch.float16)
        k = k.contiguous().to(torch.float16)
        v = v.contiguous().to(torch.float16)
    else:
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.float16)

    torch.cuda.set_device(v.device)

    headdim = q.size(-1)
    assert headdim in [64, 128], "head_dim must be either 64 or 128"

    if smooth_k:
        km = k.mean(dim=2, keepdim=True)
    else:
        km = None

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    pvthreshd_vec = hyperparameter_check(pvthreshd, q.size(1), q.device)
    q_block = masker.q_block_size
    kv_block = masker.kv_block_size
    seq_len = q.size(2)

    arch = get_cuda_arch_versions()[q.device.index]
    use_sm90 = arch == "sm90"

    num_q_blocks = (seq_len + q_block - 1) // q_block
    if num_q_blocks == 0:
        if return_sparsity:
            return q.clone(), StageSparsity(prefill=None, decode=None)
        return q.clone()

    total_width = num_q_blocks * q_block
    pad_len = total_width - seq_len
    if pad_len > 0:
        q_padded = F.pad(q, (0, 0, 0, pad_len))
    else:
        q_padded = q

    batch_size = q.size(0)
    q_blocks = q_padded.view(batch_size, q.size(1), num_q_blocks, q_block, headdim)
    valid_mask = q.new_ones((batch_size, 1, num_q_blocks, q_block, 1), dtype=q.dtype)
    if pad_len > 0:
        valid_mask[..., -1, -pad_len:, :] = 0.0

    q_sum = (q_blocks.to(torch.float32) * valid_mask.to(torch.float32)).sum(dim=3)
    counts = valid_mask.sum(dim=3).clamp_min(1.0)
    q_summary = (q_sum / counts).to(masker.device, dtype=torch.float32)

    mask_prefill = masker.compute_mask(layer, q_summary, tensor_layout="HND")
    if sparse_token_mask is not None:
        token_mask = sparse_token_mask.to(mask_prefill.device, dtype=torch.bool)
        if (
            token_mask.dim() != 2
            or token_mask.shape[0] != batch_size
            or token_mask.shape[1] != seq_len
            or token_mask.shape[1] != k.size(2)
        ):
            token_mask = None
    else:
        token_mask = None

    if token_mask is not None:
        valid_q = torch.ones_like(token_mask, dtype=torch.bool)
        if pad_len > 0:
            token_mask_q = F.pad(token_mask, (0, pad_len), value=False)
            valid_q = F.pad(valid_q, (0, pad_len), value=False)
        else:
            token_mask_q = token_mask
        q_allowed = token_mask_q.view(batch_size, num_q_blocks, q_block)
        q_valid = valid_q.view(batch_size, num_q_blocks, q_block)
        q_valid_count = q_valid.sum(dim=-1)
        sparse_q_blocks = ((q_allowed & q_valid).sum(dim=-1) == q_valid_count) & (q_valid_count > 0)

        kv_len = k.size(2)
        kv_blocks = mask_prefill.shape[-1]
        kv_total_width = kv_blocks * kv_block
        kv_pad_len = kv_total_width - kv_len
        kv_token_mask = token_mask[:, :kv_len]
        valid_kv = torch.ones_like(kv_token_mask, dtype=torch.bool)
        if kv_pad_len > 0:
            kv_token_mask = F.pad(kv_token_mask, (0, kv_pad_len), value=False)
            valid_kv = F.pad(valid_kv, (0, kv_pad_len), value=False)
        kv_allowed = kv_token_mask.view(batch_size, kv_blocks, kv_block)
        kv_valid = valid_kv.view(batch_size, kv_blocks, kv_block)
        kv_valid_count = kv_valid.sum(dim=-1)
        sparse_kv_blocks = ((kv_allowed & kv_valid).sum(dim=-1) == kv_valid_count) & (kv_valid_count > 0)

        dense_q = torch.ones_like(mask_prefill, dtype=torch.bool)
        mask_prefill = torch.where(
            sparse_q_blocks[:, None, :, None],
            mask_prefill,
            dense_q,
        )
        mask_prefill = mask_prefill | (~sparse_kv_blocks[:, None, None, :])

    mask_blocks = mask_prefill.to(q.device, dtype=torch.bool).contiguous()

    mask_float = mask_prefill.to(torch.float32)
    if mask_float.numel() > 0:
        active_ratio = float(mask_float.mean().item())
        sparsity_value = 1.0 - active_ratio
    else:
        sparsity_value = 0.0

    flops_stats = estimate_block_sparse_attention_flops(
        mask_blocks,
        q_len=seq_len,
        kv_len=k.size(2),
        q_block_size=q_block,
        kv_block_size=kv_block,
        head_dim=headdim,
        is_causal=is_causal,
    )
    current_step = masker.step_index(layer)
    stage_stats = StageSparsity(
        prefill=sparsity_value if current_step == 0 else None,
        decode=sparsity_value if current_step > 0 else None,
        dense_attention_flops=flops_stats["dense_attention_flops"],
        sparse_attention_flops=flops_stats["sparse_attention_flops"],
        saved_attention_flops=flops_stats["saved_attention_flops"],
    )

    o, _ = _run_block_sparse_kernel(
        q,
        k,
        v,
        km,
        mask_blocks,
        pvthreshd_vec,
        scale,
        use_sm90,
        is_causal,
    )

    masker.register_output(layer, o, tensor_layout="HND")

    if original_layout == "NHD":
        o = rearrange(o, 'B H L D -> B L H D')

    if return_sparsity:
        return o, stage_stats
    return o

@torch.compiler.disable
def spas_sage2_attn_meansim_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink)  # 

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    ## quant v
    b, h_kv, kv_len, head_dim = v.shape
    padded_len = (kv_len + 63) // 64 * 64
    v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
    fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
    v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
    #fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 448.0, 1)
    fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

    _is_causal = 1 if is_causal else 0
    o = torch.empty_like(q)
       
    arch = get_cuda_arch_versions()[q.device.index]
    if arch == "sm90":
        qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
    else:
        #qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        qattn.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
    
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o
