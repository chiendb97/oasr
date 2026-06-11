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


def _ref_swoosh_l(x: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035


def _ref_swoosh_r(x: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


class TestSwoosh:
    """Tests for oasr.swoosh_l() / oasr.swoosh_r() functional API."""

    @pytest.mark.parametrize(
        "fn,ref", [(oasr.swoosh_l, _ref_swoosh_l), (oasr.swoosh_r, _ref_swoosh_r)]
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 128, 256),  # vec4-aligned last dim
            (4, 250, 384),
            (2, 8, 50, 19),  # 4D (conv-output-like)
            (2, 128, 255),  # non-vec-aligned last dim -> scalar path
        ],
    )
    def test_swoosh_fp32(self, fn, ref, shape):
        x = torch.randn(*shape, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(fn(x), ref(x), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "fn,ref", [(oasr.swoosh_l, _ref_swoosh_l), (oasr.swoosh_r, _ref_swoosh_r)]
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_swoosh_half(self, fn, ref, dtype):
        x = torch.randn(4, 200, 512, device="cuda", dtype=dtype)
        torch.testing.assert_close(fn(x), ref(x), rtol=2e-2, atol=2e-2)

    def test_swoosh_large_magnitude(self):
        """Numerical stability over a wide input range (softplus must not overflow)."""
        x = torch.linspace(-60.0, 60.0, 4096, device="cuda", dtype=torch.float32).reshape(1, 64, 64)
        torch.testing.assert_close(oasr.swoosh_l(x), _ref_swoosh_l(x), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(oasr.swoosh_r(x), _ref_swoosh_r(x), rtol=1e-5, atol=1e-5)

    def test_swoosh_noncontiguous(self):
        """A non-contiguous (transposed) input is handled by an internal .contiguous()."""
        x = torch.randn(2, 256, 128, device="cuda", dtype=torch.float32).transpose(1, 2)
        assert not x.is_contiguous()
        torch.testing.assert_close(oasr.swoosh_l(x), _ref_swoosh_l(x), rtol=1e-5, atol=1e-5)

    def test_swoosh_destination_passing(self):
        x = torch.randn(2, 128, 256, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        result = oasr.swoosh_r(x, out=out)
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(out, _ref_swoosh_r(x), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
