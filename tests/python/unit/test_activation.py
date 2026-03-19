#!/usr/bin/env python3
"""
Unit tests for convolution.
Uses torch.testing.assert_close for correctness verification.
"""

import oasr
import pytest
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, 'python')


class TestGLU:
    """Tests for GLU activation kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels", [
        (2, 128, 256),
        (4, 256, 512),
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_glu(self, batch_size, seq_len, channels, dtype):
        """Test GLU against F.glu."""
        x = torch.randn(batch_size, seq_len, 2 * channels,
                        device='cuda', dtype=dtype)

        output = oasr.kernels.activation.glu(
            x
        )
        oasr.synchronize()

        expected = F.glu(x, dim=-1).to(dtype)

        rtol = 1e-2 if dtype == torch.float16 else 1e-2
        atol = 1e-2 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestSwish:
    """Tests for Swish (SiLU) activation kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels", [
        (2, 128, 256),
        (4, 256, 512),
    ])
    def test_swish(self, batch_size, seq_len, channels):
        """Test Swish against F.silu."""
        dtype = torch.float32

        x = torch.randn(batch_size, seq_len, channels,
                        device='cuda', dtype=dtype)

        output = oasr.kernels.activation.swish(
            x
        )
        oasr.synchronize()

        expected = F.silu(x)

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
