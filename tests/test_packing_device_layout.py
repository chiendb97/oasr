# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``build_packed_layout_device`` vs the host builder it replaces.

The host builder cannot be CUDA-graph captured — it performs a D2H
(``seg_lengths.tolist()``) and three data-dependent-shape operations (boolean
mask indexing, ``repeat_interleave`` with a tensor, and a per-segment Python
loop).  The device builder removes all four.  These tests pin that it produces
the *same layout*, and that it captures.
"""

from __future__ import annotations

import pytest
import torch

from oasr.models.conformer.packing import build_packed_layout, build_packed_layout_device

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
