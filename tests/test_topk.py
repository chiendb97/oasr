#!/usr/bin/env python3
"""Unit tests for oasr.topk() functional API."""

import pytest
import torch

import oasr


class TestTopK:
    """Tests for oasr.topk() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,channels",
        [
            (1, 64, 128),
            (2, 128, 256),
            (4, 64, 512),
        ],
    )
    @pytest.mark.parametrize("k", [1, 5, 10])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_topk_correctness(self, batch_size, seq_len, channels, k, dtype):
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)

        values, indices = oasr.topk(x, k)

        # Indices must be valid column indices
        assert (indices >= 0).all()
        assert (indices < channels).all()

        # Gathered values must match returned values exactly
        gathered = x.gather(-1, indices.long())
        assert (gathered == values).all(), "gathered values differ from returned values"

        # Values must match torch.topk (same sorted order)
        ref_values, _ = torch.topk(x, k, dim=-1)
        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(values, ref_values, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_topk_output_dtypes(self, dtype):
        x = torch.randn(4, 64, 256, device="cuda", dtype=dtype)
        values, indices = oasr.topk(x, 5)

        assert values.dtype == dtype
        assert indices.dtype == torch.int32
        assert values.shape == (4, 64, 5)
        assert indices.shape == (4, 64, 5)

    def test_topk_k_equals_1(self):
        """k=1 should match argmax."""
        x = torch.randn(8, 32, 128, device="cuda", dtype=torch.float32)
        values, indices = oasr.topk(x, 1)

        ref_values, ref_indices = torch.max(x, dim=-1, keepdim=True)
        torch.testing.assert_close(values, ref_values, rtol=1e-5, atol=1e-5)
        assert (indices == ref_indices.int()).all()

    def test_topk_sorted_descending(self):
        """Values must be in descending order."""
        x = torch.randn(4, 64, 512, device="cuda", dtype=torch.float32)
        values, _ = oasr.topk(x, 20)

        diffs = values[..., 1:] - values[..., :-1]
        assert (diffs <= 0).all(), "values must be in descending order"

    def test_topk_destination_passing(self):
        """Pre-allocated outputs should be written to and returned."""
        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16)
        out_v = torch.empty(2, 64, 5, device="cuda", dtype=torch.float16)
        out_i = torch.empty(2, 64, 5, device="cuda", dtype=torch.int32)

        values, indices = oasr.topk(x, 5, out_values=out_v, out_indices=out_i)

        assert values.data_ptr() == out_v.data_ptr()
        assert indices.data_ptr() == out_i.data_ptr()

    def test_topk_2d_input(self):
        """Works on 2D inputs."""
        x = torch.randn(256, 4096, device="cuda", dtype=torch.float32)
        values, indices = oasr.topk(x, 5)

        assert values.shape == (256, 5)
        assert indices.shape == (256, 5)

        ref_values, _ = torch.topk(x, 5, dim=-1)
        torch.testing.assert_close(values, ref_values, rtol=1e-4, atol=1e-4)

    def test_topk_unique_indices(self):
        """Each row must have unique indices (no element selected twice)."""
        x = torch.randn(4, 64, 512, device="cuda", dtype=torch.float32)
        _, indices = oasr.topk(x, 10)

        flat = indices.view(-1, 10)
        for i in range(flat.shape[0]):
            row = flat[i].tolist()
            assert len(set(row)) == 10, f"row {i} has duplicate indices: {row}"

    def test_topk_cpu_error(self):
        """CPU tensors must raise an error."""
        x = torch.randn(2, 64, 128, dtype=torch.float32)
        with pytest.raises(Exception):
            oasr.topk(x, 5)

    def test_topk_layer(self):
        """TopK nn.Module wrapper should return correct shapes."""
        from oasr.layers import TopK

        layer = TopK(k=5)
        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16)
        values, indices = layer(x)

        assert values.shape == (2, 64, 5)
        assert indices.shape == (2, 64, 5)
