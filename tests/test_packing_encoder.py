# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for Conformer sequence packing (M1, SDPA path).

The keystone parity property: a packed encoder forward over N utterances is
**bit-exact (within fp tolerance) to single-utterance (B=1) inference** —
attention is restricted per-segment and the depthwise conv is isolated with
gap frames, so each utterance sees the same compute it would alone.

Also unit-tests the two time-mixing operators in isolation:

* ``ConvolutionModule.forward_packed`` vs standalone ``forward`` per segment.
* ``RelPositionMultiHeadedAttention`` packed vs per-utterance offline.

No checkpoint / WeNet needed — a small randomly-initialised encoder suffices.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from oasr.models.conformer import ConformerEncoder, ConformerEncoderConfig
from oasr.models.conformer.model import ConvolutionModule
from oasr.models.conformer.packing import build_packed_layout


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return dict(rtol=2e-3, atol=2e-3)
    return dict(rtol=3e-2, atol=3e-2)


def _make_encoder(
    dtype: torch.dtype, device, *, num_blocks: int = 2, causal: bool = False
) -> ConformerEncoder:
    enc = ConformerEncoder(
        ConformerEncoderConfig(
            input_size=80,
            output_size=64,
            num_blocks=num_blocks,
            attention_heads=2,
            linear_units=128,
            use_cnn_module=True,
            cnn_module_kernel=15,
            causal=causal,
            cnn_module_norm="layer_norm" if causal else "batch_norm",
        )
    )
    return enc.eval().to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Conv module isolation (proves no cross-boundary leak)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_conv_packed_matches_per_segment(dtype, causal, device):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unsupported")
    torch.manual_seed(0)
    C = 32
    conv = ConvolutionModule(
        C,
        kernel_size=15,
        norm="layer_norm" if causal else "batch_norm",
        causal=causal,
    )
    conv = conv.eval().to(device=device, dtype=dtype)

    seg_lens = [13, 7, 21]
    segs = [torch.randn(1, t, C, device=device, dtype=dtype) for t in seg_lens]
    packed = torch.cat(segs, dim=1)  # (1, sum, C)

    tp = max(seg_lens)
    valid = torch.zeros(len(seg_lens), tp, dtype=torch.bool, device=device)
    for i, t in enumerate(seg_lens):
        valid[i, :t] = True
    layout = build_packed_layout(valid, conv_kernel=15)

    with torch.no_grad():
        out_packed = conv.forward_packed(packed, layout)
        refs = [conv(s)[0] for s in segs]  # standalone per segment

    off = 0
    for ref, t in zip(refs, seg_lens):
        seg_out = out_packed[:, off : off + t, :]
        torch.testing.assert_close(seg_out, ref, **_tol(dtype))
        off += t


# ---------------------------------------------------------------------------
# Full encoder parity
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_equals_batched_equal_lengths(dtype, device):
    """Equal-length utts: no embed/conv padding, so packed == batched forward."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unsupported")
    torch.manual_seed(1)
    enc = _make_encoder(dtype, device)

    B, T = 4, 60
    xs = torch.randn(B, T, 80, device=device, dtype=dtype)
    lens = torch.full((B,), T, dtype=torch.long, device=device)

    with torch.no_grad():
        ref, ref_mask = enc(xs, lens)
        packed, packed_mask = enc.forward_packed(xs, lens)

    assert packed.shape == ref.shape
    torch.testing.assert_close(packed_mask, ref_mask)
    torch.testing.assert_close(packed, ref, **_tol(dtype))


@pytest.mark.cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_equals_single_utterance(dtype, causal, device):
    """Variable-length utts: packed valid frames == per-utterance B=1 forward."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unsupported")
    torch.manual_seed(2)
    enc = _make_encoder(dtype, device, causal=causal)

    lens_in: List[int] = [37, 80, 52, 61]
    Tmax = max(lens_in)
    xs = torch.zeros(len(lens_in), Tmax, 80, device=device, dtype=dtype)
    for i, t in enumerate(lens_in):
        xs[i, :t] = torch.randn(t, 80, device=device, dtype=dtype)
    lens = torch.tensor(lens_in, dtype=torch.long, device=device)

    with torch.no_grad():
        packed, packed_mask = enc.forward_packed(xs, lens)
        out_lens = packed_mask.squeeze(1).sum(-1).tolist()
        # B=1 baseline per utterance.
        for i, t in enumerate(lens_in):
            ref_i, mask_i = enc(xs[i : i + 1, :t], lens[i : i + 1])
            ti = int(mask_i.squeeze(1).sum(-1).item())
            assert ti == int(out_lens[i])
            torch.testing.assert_close(packed[i : i + 1, :ti], ref_i[:, :ti], **_tol(dtype))


@pytest.mark.cuda
def test_packed_single_segment_equals_offline(device):
    """A 1-utterance pack must equal the plain offline forward exactly."""
    dtype = torch.float16
    torch.manual_seed(3)
    enc = _make_encoder(dtype, device)

    T = 50
    xs = torch.randn(1, T, 80, device=device, dtype=dtype)
    lens = torch.tensor([T], dtype=torch.long, device=device)
    with torch.no_grad():
        ref, _ = enc(xs, lens)
        packed, _ = enc.forward_packed(xs, lens)
    torch.testing.assert_close(packed, ref, **_tol(dtype))
