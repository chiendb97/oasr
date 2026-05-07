"""Parity tests for ``oasr.fmha`` against an SDPA reference.

These tests run against whichever backend ``OASR_ATTN_BACKEND`` selects:
* ``sdpa`` -- exercises the fallback path; verifies the public functional
  API + wrapper integration produce numerically identical results to the
  legacy SDPA call.
* ``cute`` -- exercises the SM120 CuteDSL kernel.

Run with::

    pytest tests/test_fmha.py -v                          # default backend
    OASR_ATTN_BACKEND=cute pytest tests/test_fmha.py -v  # force cute backend
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference: a clean SDPA path that mirrors oasr.fmha_forward's contract.
# Used to compare both backends against a single source of truth.
# ---------------------------------------------------------------------------

def _ref_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    attn_bias: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, H, T_q, D = q.shape
    H_kv = k.size(1)
    T_k = k.size(2)
    if H % H_kv != 0:
        raise ValueError("H % H_kv != 0")
    if H_kv != H:
        n_repeat = H // H_kv
        k = k.repeat_interleave(n_repeat, dim=1)
        v = v.repeat_interleave(n_repeat, dim=1)

    masks = []
    if attn_bias is not None:
        masks.append(attn_bias.to(q.dtype))
    if cache_seqlens is not None:
        arange = torch.arange(T_k, device=cache_seqlens.device)
        keep = arange.unsqueeze(0) < cache_seqlens.unsqueeze(1)
        pad = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
        pad = pad.unsqueeze(1).unsqueeze(1)  # (B,1,1,T_k)
        masks.append(pad)

    full_mask = None
    if masks:
        full_mask = masks[0]
        for m in masks[1:]:
            full_mask = full_mask + m

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=full_mask, scale=softmax_scale,
    )


# ---------------------------------------------------------------------------
# Test parameter grid
# ---------------------------------------------------------------------------

_SHAPES = [
    # (B, H, H_kv, T_q, T_k, D)
    (1, 4, 4, 8, 16, 64),     # smallest streaming chunk
    (4, 4, 4, 8, 64, 64),     # bigger batch
    (1, 4, 4, 16, 32, 64),    # T_q not 8
    (2, 8, 8, 8, 128, 64),    # bigger H
    (2, 8, 1, 8, 64, 64),     # MQA
    (2, 8, 2, 8, 64, 64),     # GQA
    (1, 4, 4, 64, 256, 64),   # offline-ish shape
]

_DTYPES = [torch.float16]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    _DTYPES.append(torch.bfloat16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def fmha():
    """Import oasr.fmha_forward and force a fresh backend probe."""
    # Force a re-read of the env var since other tests may have set/unset it.
    from oasr.jit.attention import set_backend_mode
    mode = os.environ.get("OASR_ATTN_BACKEND", "auto").lower()
    set_backend_mode(mode)
    from oasr import fmha
    return fmha


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_fmha_offline(fmha, cuda, dtype, shape):
    """Offline mode: no bias, no length mask."""
    B, H, H_kv, T_q, T_k, D = shape
    torch.manual_seed(0)
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    out = fmha(q, k, v, softmax_scale=scale)
    ref = _ref_fmha(q, k, v, scale)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_fmha_with_bias(fmha, cuda, dtype, shape):
    """Offline + additive bias (rel-pos style)."""
    B, H, H_kv, T_q, T_k, D = shape
    torch.manual_seed(1)
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    bias = torch.randn(B, H, T_q, T_k, device=cuda, dtype=dtype) * 0.1
    scale = 1.0 / math.sqrt(D)

    out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
    ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_fmha_with_length_mask(fmha, cuda, dtype, shape):
    """Per-stream length mask via cache_seqlens (heterogeneous)."""
    B, H, H_kv, T_q, T_k, D = shape
    torch.manual_seed(2)
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    # Half the streams get a short context (~1/4 of T_k), the rest full T_k.
    base = max(1, T_k // 4)
    seqlens = torch.tensor(
        [base if i < B // 2 else T_k for i in range(B)],
        dtype=torch.int32, device=cuda,
    )
    scale = 1.0 / math.sqrt(D)

    out = fmha(q, k, v, softmax_scale=scale, cache_seqlens=seqlens)
    ref = _ref_fmha(q, k, v, scale, cache_seqlens=seqlens)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_fmha_bias_and_mask(fmha, cuda, dtype, shape):
    """Combined bias + length mask (matches RelPosMHA paged-streaming usage)."""
    B, H, H_kv, T_q, T_k, D = shape
    torch.manual_seed(3)
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    bias = torch.randn(B, H, T_q, T_k, device=cuda, dtype=dtype) * 0.1
    seqlens = torch.tensor(
        [max(1, T_k - i) for i in range(B)],
        dtype=torch.int32, device=cuda,
    )
    scale = 1.0 / math.sqrt(D)

    out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias, cache_seqlens=seqlens)
    ref = _ref_fmha(q, k, v, scale, attn_bias=bias, cache_seqlens=seqlens)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_fmha_paged_not_implemented(fmha, cuda):
    """Paged mode (block_table) is intentionally NotImplementedError in this revision."""
    q = torch.randn(1, 4, 8, 64, device=cuda, dtype=torch.float16)
    pool_k = torch.randn(16, 16, 4, 64, device=cuda, dtype=torch.float16)  # paged layout
    pool_v = torch.randn(16, 16, 4, 64, device=cuda, dtype=torch.float16)
    bt = torch.zeros(1, 4, dtype=torch.int32, device=cuda)
    seqlens = torch.tensor([8], dtype=torch.int32, device=cuda)
    with pytest.raises(NotImplementedError, match="paged"):
        fmha(q, pool_k, pool_v,
             softmax_scale=0.125, cache_seqlens=seqlens, block_table=bt)


def test_fmha_fp32_falls_back_to_sdpa(fmha, cuda):
    """fp32 isn't supported by the cute kernel but works via SDPA fallback."""
    q32 = torch.randn(1, 4, 8, 64, device=cuda, dtype=torch.float32)
    k32 = torch.randn(1, 4, 16, 64, device=cuda, dtype=torch.float32)
    v32 = torch.randn(1, 4, 16, 64, device=cuda, dtype=torch.float32)
    out = fmha(q32, k32, v32, softmax_scale=0.125)
    ref = _ref_fmha(q32, k32, v32, 0.125)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_fmha_gqa_validation(fmha, cuda):
    """H must be divisible by H_kv."""
    q = torch.randn(1, 8, 8, 64, device=cuda, dtype=torch.float16)
    k = torch.randn(1, 3, 16, 64, device=cuda, dtype=torch.float16)  # 8 % 3 != 0
    v = torch.randn(1, 3, 16, 64, device=cuda, dtype=torch.float16)
    with pytest.raises(ValueError, match="divisible"):
        fmha(q, k, v, softmax_scale=0.125)
