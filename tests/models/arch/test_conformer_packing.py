# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer sequence packing (``oasr/models/conformer/packing.py``).

Two halves of one feature, in one file and in the models family -- they used
to be two files in the *engine* family, which is neither module they touch:

* **the layout builder.** The host version cannot be CUDA-graph captured: it
  performs a D2H (``seg_lengths.tolist()``) and three data-dependent-shape
  operations (boolean mask indexing, ``repeat_interleave`` with a tensor, and
  a per-segment Python loop). The device builder removes all four, and must
  produce the same layout.
* **the numerics.** A packed encoder forward over N utterances is bit-exact
  (within fp tolerance) to single-utterance inference: attention is restricted
  per segment and the depthwise conv is isolated with gap frames, so each
  utterance sees the compute it would have seen alone. The two time-mixing
  operators are also checked in isolation.

No checkpoint and no ``wenet``: a small randomly-initialised encoder suffices,
which is why this is not folded into ``test_conformer.py`` -- that module is
the upstream-parity test and skips entirely without ``wenet`` installed.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from oasr.models.conformer import ConformerEncoder, ConformerEncoderConfig
from oasr.models.conformer.model import ConvolutionModule
from oasr.models.conformer.packing import build_packed_layout, build_packed_layout_device

# ---------------------------------------------------------------------------
# The device-side layout builder
# ---------------------------------------------------------------------------


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FIELDS = (
    "cu_seqlens",
    "pack_src_idx",
    "conv_gather_idx",
    "conv_batched_idx",
    "seg_valid_mask",
    "bias_offsets",
    "bias_gather_idx",
)

SHAPES = [
    (4, 128, 15, 8),
    (8, 200, 15, 4),
    (3, 64, 31, 8),
    (16, 175, 15, 8),
    (1, 96, 15, 2),
]


def _mask(lengths: torch.Tensor, padded_t: int) -> torch.Tensor:
    ar = torch.arange(padded_t, device=lengths.device, dtype=torch.int64)
    return ar.unsqueeze(0) < lengths.unsqueeze(1).to(torch.int64)


@cuda_only
class TestMatchesHostBuilder:
    @pytest.mark.parametrize("S,Tp,K,H", SHAPES)
    def test_every_field_is_identical(self, S, Tp, K, H):
        dev = torch.device("cuda")
        torch.manual_seed(S * 1000 + Tp)
        lengths = torch.randint(Tp // 2, Tp + 1, (S,), device=dev, dtype=torch.int32)
        ref = build_packed_layout(_mask(lengths, Tp), K, num_heads=H)
        got = build_packed_layout_device(lengths, Tp, K, H)

        for name in FIELDS:
            a, b = getattr(ref, name), getattr(got, name)
            assert (a is None) == (b is None), name
            if a is not None:
                assert torch.equal(a, b), f"{name} differs"
        for name in ("num_segs", "total_tokens", "max_seg_len", "gapped_len", "src_rows"):
            assert getattr(ref, name) == getattr(got, name), name

    def test_conv_only_layout_has_no_bias(self):
        dev = torch.device("cuda")
        lengths = torch.tensor([40, 30, 20], device=dev, dtype=torch.int32)
        ref = build_packed_layout(_mask(lengths, 40), 15, num_heads=None)
        got = build_packed_layout_device(lengths, 40, 15, None)
        assert ref.bias_offsets is None and got.bias_offsets is None
        assert torch.equal(ref.conv_gather_idx, got.conv_gather_idx)


@cuda_only
class TestCapacities:
    """The capacities are what replace the D2H, so they must actually bound."""

    def test_padding_capacity_extends_without_moving_the_real_prefix(self):
        dev = torch.device("cuda")
        lengths = torch.tensor([50, 40, 30], device=dev, dtype=torch.int32)
        exact = build_packed_layout_device(lengths, 50, 15, 8)
        padded = build_packed_layout_device(
            lengths,
            50,
            15,
            8,
            total_capacity=exact.total_tokens + 37,
            max_seg_capacity=exact.max_seg_len,
            bias_capacity=exact.bias_gather_idx.numel() + 99,
        )
        assert padded.total_tokens == exact.total_tokens + 37
        n = exact.total_tokens
        assert torch.equal(padded.pack_src_idx[:n], exact.pack_src_idx)
        assert torch.equal(padded.conv_gather_idx[:n], exact.conv_gather_idx)
        m = exact.bias_gather_idx.numel()
        assert torch.equal(padded.bias_gather_idx[:m], exact.bias_gather_idx)
        assert torch.equal(padded.bias_offsets, exact.bias_offsets)

    def test_zero_length_segments_pad_the_segment_axis_for_free(self):
        """Trailing zero-length segments are how S is padded to a bucket."""
        dev = torch.device("cuda")
        real = torch.tensor([50, 40], device=dev, dtype=torch.int32)
        padded = torch.tensor([50, 40, 0, 0], device=dev, dtype=torch.int32)
        a = build_packed_layout_device(real, 50, 15, 8)
        b = build_packed_layout_device(padded, 50, 15, 8)
        assert a.total_tokens == b.total_tokens
        assert torch.equal(a.pack_src_idx, b.pack_src_idx)
        assert torch.equal(a.conv_gather_idx, b.conv_gather_idx)
        # An empty segment contributes no bias block.
        assert int(b.bias_offsets[-1]) == int(a.bias_offsets[-1])


@cuda_only
class TestCapturable:
    def test_device_builder_captures_and_host_builder_does_not(self):
        """The whole point: this is the op set that can go inside a graph."""
        import tvm_ffi

        dev = torch.device("cuda")
        lengths = torch.tensor([60, 48, 32, 16], device=dev, dtype=torch.int32)
        Tp = 60
        exact = build_packed_layout_device(lengths, Tp, 15, 8)
        caps = {
            "total_capacity": exact.total_tokens,
            "max_seg_capacity": exact.max_seg_len,
            "bias_capacity": int(exact.bias_gather_idx.numel()),
        }

        def device_build():
            return build_packed_layout_device(lengths, Tp, 15, 8, **caps)

        device_build()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with tvm_ffi.use_torch_stream(torch.cuda.graph(graph)):
            captured = device_build()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured.pack_src_idx, exact.pack_src_idx)
        assert torch.equal(captured.bias_gather_idx, exact.bias_gather_idx)

        # ...and the host builder still cannot be captured, which is why the
        # device one exists.  Guard against "fixed" meaning "silently equivalent".
        mask = _mask(lengths, Tp)
        build_packed_layout(mask, 15, num_heads=8)
        torch.cuda.synchronize()
        g2 = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="pinned|capture"):
            with tvm_ffi.use_torch_stream(torch.cuda.graph(g2)):
                build_packed_layout(mask, 15, num_heads=8)
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Packed == single-utterance, and the two operators underneath
# ---------------------------------------------------------------------------


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"rtol": 2e-3, "atol": 2e-3}
    return {"rtol": 3e-2, "atol": 3e-2}


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
