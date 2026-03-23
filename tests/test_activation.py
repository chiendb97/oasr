#!/usr/bin/env python3
"""
Unit tests for functional activation API (TVM-FFI JIT path).
"""

import pytest
import torch
import torch.nn.functional as F

import oasr


class TestGLU:
    """Tests for oasr.glu() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,channels",
        [
            (2, 128, 256),
            (4, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_glu(self, batch_size, seq_len, channels, dtype):
        x = torch.randn(batch_size, seq_len, 2 * channels, device="cuda", dtype=dtype)

        output = oasr.glu(x)

        expected = F.glu(x, dim=-1).to(dtype)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_glu_destination_passing(self, dtype):
        """Test GLU with pre-allocated output tensor."""
        batch_size, seq_len, channels = 2, 128, 256
        x = torch.randn(batch_size, seq_len, 2 * channels, device="cuda", dtype=dtype)
        out = torch.empty(batch_size, seq_len, channels, device="cuda", dtype=dtype)

        result = oasr.glu(x, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = F.glu(x, dim=-1).to(dtype)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


class TestSwish:
    """Tests for oasr.swish() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,channels",
        [
            (2, 128, 256),
            (4, 256, 512),
        ],
    )
    def test_swish(self, batch_size, seq_len, channels):
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=torch.float32)

        output = oasr.swish(x)

        expected = F.silu(x)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_swish_destination_passing(self):
        """Test Swish with pre-allocated output tensor."""
        batch_size, seq_len, channels = 2, 128, 256
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        result = oasr.swish(x, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = F.silu(x)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
