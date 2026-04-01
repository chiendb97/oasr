#!/usr/bin/env python3
"""
Unit tests for CMVN (Cepstral Mean and Variance Normalization) kernel.
"""

import pytest
import torch

import oasr


class TestCMVN:
    """Tests for oasr.cmvn() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,num_cols",
        [
            (1, 64, 80),
            (2, 128, 256),
            (4, 256, 512),
            (8, 100, 40),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_cmvn_correctness(self, batch_size, seq_len, num_cols, dtype):
        """Test CMVN correctness against PyTorch reference."""
        x = torch.randn(batch_size, seq_len, num_cols, device="cuda", dtype=dtype)
        mean = torch.randn(num_cols, device="cuda", dtype=dtype)
        istd = torch.randn(num_cols, device="cuda", dtype=dtype).abs() + 0.1

        output = oasr.cmvn(x, mean, istd)

        expected = (x - mean) * istd
        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cmvn_destination_passing(self, dtype):
        """Test CMVN with pre-allocated output."""
        batch_size, seq_len, num_cols = 2, 128, 256
        x = torch.randn(batch_size, seq_len, num_cols, device="cuda", dtype=dtype)
        mean = torch.randn(num_cols, device="cuda", dtype=dtype)
        istd = torch.randn(num_cols, device="cuda", dtype=dtype).abs() + 0.1
        out = torch.empty_like(x)

        result = oasr.cmvn(x, mean, istd, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = (x - mean) * istd
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_cmvn_2d_input(self):
        """Test CMVN with 2D input [m, n]."""
        m, n = 100, 80
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        mean = torch.randn(n, device="cuda", dtype=torch.float32)
        istd = torch.randn(n, device="cuda", dtype=torch.float32).abs() + 0.1

        output = oasr.cmvn(x, mean, istd)

        expected = (x - mean) * istd
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_cmvn_typical_asr_shapes(self):
        """Test CMVN with typical ASR feature dimensions (80-dim mel, 40-dim fbank)."""
        for num_cols in [40, 80]:
            x = torch.randn(4, 200, num_cols, device="cuda", dtype=torch.float32)
            mean = torch.randn(num_cols, device="cuda", dtype=torch.float32)
            istd = torch.randn(num_cols, device="cuda", dtype=torch.float32).abs() + 0.1

            output = oasr.cmvn(x, mean, istd)
            expected = (x - mean) * istd
            torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
