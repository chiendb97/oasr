# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the variable-length (sequence-packed) attention API.

``fmha_varlen`` packs several segments into one ``(total, H, D)`` tensor and
restricts each segment to itself via ``cu_seqlens``.  The reference is the
existing dense ``fmha`` run per-segment (B=1) — including the packed
block-diagonal rel-pos bias keyed by ``bias_offsets``.

Exercises odd segment lengths (33, 249 — real audio frame counts), GQA/MQA,
single-segment == dense, and many tiny segments.
"""

from __future__ import annotations

import math
from typing import List, Optional

import pytest
import torch

from oasr.attention import fmha, fmha_varlen


def _tol(dtype):
    return dict(rtol=2e-2, atol=2e-2)


def _make_packed(seg_lens, H, H_kv, D, dtype, device):
    """Random packed q/k/v + cu_seqlens for the given segment lengths."""
    total = sum(seg_lens)
    q = torch.randn(total, H, D, device=device, dtype=dtype)
    k = torch.randn(total, H_kv, D, device=device, dtype=dtype)
    v = torch.randn(total, H_kv, D, device=device, dtype=dtype)
    cu = torch.zeros(len(seg_lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seg_lens, dtype=torch.int32, device=device).cumsum(0)
    return q, k, v, cu


def _build_packed_bias(q, cu, H, D, dtype, device, scale):
    """A packed block-diagonal additive bias (random) + bias_offsets."""
    seg_lens = (cu[1:] - cu[:-1]).tolist()
    sizes = [H * t * t for t in seg_lens]
    offsets = torch.zeros(len(seg_lens) + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.tensor(sizes, dtype=torch.int64, device=device).cumsum(0)
    bias = torch.randn(int(offsets[-1]), device=device, dtype=dtype) * 0.1
    return bias, offsets


def _ref_per_segment(q, k, v, cu, scale, bias, offsets, H):
    """Reference: dense fmha per segment, assembled back into packed output."""
    cu_l = cu.tolist()
    bo = offsets.tolist() if offsets is not None else None
    out = torch.empty_like(q)
    for s in range(len(cu_l) - 1):
        a, b = cu_l[s], cu_l[s + 1]
        qs = q[a:b].transpose(0, 1).unsqueeze(0)
        ks = k[a:b].transpose(0, 1).unsqueeze(0)
        vs = v[a:b].transpose(0, 1).unsqueeze(0)
        bias_s = None
        if bias is not None:
            # clone() → a fresh 16-byte-aligned allocation (the dense cute
            # kernel rejects misaligned bias slices of the packed buffer).
            bias_s = bias[bo[s]:bo[s + 1]].view(1, H, b - a, b - a).clone()
        out_s = fmha(qs, ks, vs, softmax_scale=scale, attn_bias=bias_s)
        out[a:b] = out_s.squeeze(0).transpose(0, 1)
    return out


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seg_lens", [[33], [64, 64], [33, 249, 17], [8] * 12])
@pytest.mark.parametrize("with_bias", [False, True])
def test_varlen_matches_per_segment_dense(dtype, seg_lens, with_bias, device):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unsupported")
    torch.manual_seed(0)
    H, H_kv, D = 4, 4, 64
    scale = 1.0 / math.sqrt(D)
    q, k, v, cu = _make_packed(seg_lens, H, H_kv, D, dtype, device)
    bias, offsets = (None, None)
    if with_bias:
        bias, offsets = _build_packed_bias(q, cu, H, D, dtype, device, scale)

    out = fmha_varlen(
        q, k, v, softmax_scale=scale,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max(seg_lens), max_seqlen_k=max(seg_lens),
        attn_bias=bias, bias_offsets=offsets,
    )
    ref = _ref_per_segment(q, k, v, cu, scale, bias, offsets, H)
    assert out.shape == q.shape
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("gqa", [(8, 2), (8, 1)])
def test_varlen_gqa(dtype, gqa, device):
    torch.manual_seed(1)
    H, H_kv = gqa
    D = 64
    scale = 1.0 / math.sqrt(D)
    seg_lens = [40, 55, 33]
    q, k, v, cu = _make_packed(seg_lens, H, H_kv, D, dtype, device)
    out = fmha_varlen(
        q, k, v, softmax_scale=scale,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max(seg_lens), max_seqlen_k=max(seg_lens),
    )
    ref = _ref_per_segment(q, k, v, cu, scale, None, None, H)
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.cuda
def test_varlen_single_segment_equals_dense(device):
    dtype = torch.float16
    torch.manual_seed(2)
    H, D, T = 4, 64, 50
    scale = 1.0 / math.sqrt(D)
    q, k, v, cu = _make_packed([T], H, H, D, dtype, device)
    out = fmha_varlen(
        q, k, v, softmax_scale=scale,
        cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=T, max_seqlen_k=T,
    )
    dense = fmha(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), softmax_scale=scale,
    ).squeeze(0).transpose(0, 1)
    torch.testing.assert_close(out, dense, **_tol(dtype))
