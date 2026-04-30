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
    # The f16 sparse kernel below validates q_scale/k_scale against its own
    # CTA tile sizes. Unlike the fp8 path, it has no sm90-specific dispatch
    # here, so keep the default quantization blocks on every architecture.
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
    if q.size(0) != 1:
        raise NotImplementedError("Adaptive masking currently supports batch size 1")

    state = masker._layer_state(layer)
    prefill_mode = (
        k_new is None
        and v_new is None
        and state.processed_tokens == 0
    )

    debug_masks = os.environ.get("ADAPTIVE_DEBUG", "0") == "1"
    if debug_masks:
        phase = "prefill" if prefill_mode else "decode"

        def _shape_str(tensor: Optional[torch.Tensor]) -> str:
            return "None" if tensor is None else "x".join(str(dim) for dim in tensor.shape)

        print(
            f"[AdaptiveCore][layer={layer}] phase={phase} "
            f"q_shape={_shape_str(q)} k_shape={_shape_str(k)} v_shape={_shape_str(v)} "
            f"k_new_shape={_shape_str(k_new)} v_new_shape={_shape_str(v_new)} "
            f"processed_tokens={state.processed_tokens} layout={original_layout}->{tensor_layout}"
        )

    if k_new is not None and v_new is not None:
        masker.update_tail_statistics(layer, k_new, v_new, tensor_layout=tensor_layout)
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
    seq_len = q.size(2)

    arch = get_cuda_arch_versions()[q.device.index]
    use_sm90 = arch == "sm90"

    if prefill_mode:
        num_q_blocks = (seq_len + q_block - 1) // q_block
        if num_q_blocks == 0:
            if return_sparsity:
                return q.clone(), 0.0
            return q.clone()

        total_width = num_q_blocks * q_block
        pad_len = total_width - seq_len
        if pad_len > 0:
            q_padded = F.pad(q, (0, 0, 0, pad_len))
        else:
            q_padded = q

        q_blocks = q_padded.view(1, q.size(1), num_q_blocks, q_block, headdim)
        valid_mask = q.new_ones((1, 1, num_q_blocks, q_block, 1), dtype=q.dtype)
        if pad_len > 0:
            valid_mask[..., -1, -pad_len:, :] = 0.0

        q_sum = (q_blocks.to(torch.float32) * valid_mask.to(torch.float32)).sum(dim=3)
        counts = valid_mask.sum(dim=3).clamp_min(1.0)
        q_summary = (q_sum / counts).squeeze(0).to(masker.device, dtype=torch.float32)

        mask_prefill = masker.compute_mask(layer, q_summary, tensor_layout="HND", prefill=True)
        mask_blocks = mask_prefill.to(q.device, dtype=torch.bool).unsqueeze(0).contiguous()

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

        if original_layout == "NHD":
            o = rearrange(o, 'B H L D -> B L H D')

        prefill_sparsity = 0.0
        total_entries = mask_prefill.numel()
        if total_entries > 0:
            active_blocks = mask_prefill.to(torch.float32).sum()
            prefill_sparsity = 1.0 - (active_blocks / float(total_entries))
            prefill_sparsity = float(prefill_sparsity.item())

        if return_sparsity:
            return o, StageSparsity(prefill=prefill_sparsity, decode=None)
        return o

    outputs = []
    active_blocks = 0.0
    total_blocks = 0.0
    state = masker._layer_state(layer)

    for start in range(0, seq_len, q_block):
        end = min(start + q_block, seq_len)
        q_chunk = q[:, :, start:end, :]
        if q_chunk.numel() == 0:
            continue

        actual_len = q_chunk.size(2)
        if actual_len == 1:
            q_summary = q_chunk[0, :, 0, :].to(masker.device, dtype=torch.float32)
        else:
            q_summary = q_chunk[0, :, :actual_len, :].mean(dim=1).to(masker.device, dtype=torch.float32)

        mask = masker.compute_mask(layer, q_summary, tensor_layout="HND").to(q.device, torch.bool)
        mask.fill_(True)
        if state.g_prev.size(1) >= mask.size(1):
            state.g_prev[:, :mask.size(1)].fill_(1.0)
        state.drop_mass.zero_()

        mask_blocks = mask.unsqueeze(0).unsqueeze(2).contiguous()

        o_chunk, _ = _run_block_sparse_kernel(
            q_chunk,
            k,
            v,
            km,
            mask_blocks,
            pvthreshd_vec,
            scale,
            use_sm90,
            is_causal,
        )

        active_blocks += float(mask.to(torch.float32).sum().item())
        total_blocks += mask.numel()

        chunk_out = o_chunk[:, :, :actual_len, :]
        outputs.append(chunk_out)
        masker.register_output(layer, chunk_out, tensor_layout="HND")

    if not outputs:
        o = torch.empty_like(q)
    else:
        o = torch.cat(outputs, dim=2)

    if original_layout == "NHD":
        o = rearrange(o, 'B H L D -> B L H D')

    if return_sparsity:
        decode_sparsity = 1.0 - (active_blocks / max(total_blocks, 1.0))
        decode_sparsity = float(decode_sparsity)
        return o, StageSparsity(prefill=None, decode=decode_sparsity)
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
