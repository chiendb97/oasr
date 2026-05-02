#!/usr/bin/env python3
"""Unit tests for oasr.softmax() functional API."""

import pytest
import torch
import torch.nn.functional as F

import oasr


class TestSoftmax:
    """Tests for oasr.softmax() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,channels",
        [
            (1, 64, 128),
            (2, 128, 256),
            (4, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_softmax_correctness(self, batch_size, seq_len, channels, dtype):
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)

        output = oasr.softmax(x)
        expected = F.softmax(x.float(), dim=-1).to(dtype)

        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_softmax_sums_to_one(self, dtype):
        x = torch.randn(4, 128, 256, device="cuda", dtype=dtype)
        output = oasr.softmax(x)

        row_sums = output.float().sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_softmax_destination_passing(self, dtype):
        x = torch.randn(2, 64, 128, device="cuda", dtype=dtype)
        out = torch.empty_like(x)

        result = oasr.softmax(x, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = F.softmax(x.float(), dim=-1).to(dtype)
        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)

    def test_softmax_numerical_stability(self):
        """Large inputs should not produce NaN or Inf."""
        x = torch.full((2, 64, 128), 1e4, device="cuda", dtype=torch.float32)
        output = oasr.softmax(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        # uniform distribution expected
        torch.testing.assert_close(
            output, torch.full_like(output, 1.0 / 128), rtol=1e-4, atol=1e-4
        )

    def test_softmax_2d_input(self):
        """Softmax should work on 2D inputs."""
        x = torch.randn(32, 256, device="cuda", dtype=torch.float32)
        output = oasr.softmax(x)
        expected = F.softmax(x, dim=-1)
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_softmax_cpu_error(self):
        """CPU tensors should raise an error."""
        x = torch.randn(2, 64, 128, dtype=torch.float32)
        with pytest.raises(Exception):
            oasr.softmax(x)
