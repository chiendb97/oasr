#!/usr/bin/env python3
"""
Unit tests for functional normalization API (TVM-FFI JIT path).
"""

import pytest
import torch

import oasr


class TestLayerNorm:
    """Tests for oasr.layer_norm() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size",
        [
            (1, 64, 128),
            (2, 128, 256),
            (4, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_layer_norm(self, batch_size, seq_len, hidden_size, dtype):
        eps = 1e-5
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        bias = torch.randn(hidden_size, device="cuda", dtype=dtype)

        output = oasr.layer_norm(x, weight, bias, eps)

        ln = torch.nn.LayerNorm(hidden_size, eps=eps, device="cuda", dtype=dtype)
        ln.weight.data = weight.clone()
        ln.bias.data = bias.clone()
        expected = ln(x)

        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    def test_layer_norm_no_bias(self):
        """Test LayerNorm without bias."""
        batch_size, seq_len, hidden_size = 2, 128, 256
        eps = 1e-5
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)
        weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)

        output = oasr.layer_norm(x, weight, bias=None, eps=eps)

        ln = torch.nn.LayerNorm(hidden_size, eps=eps, device="cuda", dtype=torch.float32)
        ln.weight.data = weight.clone()
        ln.bias.data.zero_()
        expected = ln(x) - ln.bias.data  # Subtract bias since ref adds it
        # Simpler: just use F.layer_norm
        expected = torch.nn.functional.layer_norm(x, (hidden_size,), weight, None, eps)

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_layer_norm_destination_passing(self):
        """Test LayerNorm with pre-allocated output."""
        batch_size, seq_len, hidden_size = 2, 128, 256
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)
        weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        bias = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)

        result = oasr.layer_norm(x, weight, bias, 1e-5, out=out)

        assert result.data_ptr() == out.data_ptr()


class TestRMSNorm:
    """Tests for oasr.rms_norm() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size",
        [
            (1, 64, 128),
            (2, 128, 256),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_rms_norm(self, batch_size, seq_len, hidden_size, dtype):
        eps = 1e-6
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

        output = oasr.rms_norm(x, weight, eps=eps)

        # RMSNorm reference: y = x * weight / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        expected = (x.float() / rms * weight.float()).to(dtype)

        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestBatchNorm1D:
    """Tests for oasr.batch_norm_1d() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,channels",
        [
            (2, 128, 64),
            (4, 256, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_batch_norm_1d(self, batch_size, seq_len, channels, dtype):
        eps = 1e-5
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
        weight = torch.ones(channels, device="cuda", dtype=dtype)
        bias = torch.zeros(channels, device="cuda", dtype=dtype)
        running_mean = torch.zeros(channels, device="cuda", dtype=dtype)
        running_var = torch.ones(channels, device="cuda", dtype=dtype)

        output = oasr.batch_norm_1d(x, weight, bias, running_mean, running_var, eps)

        # Reference: (x - mean) / sqrt(var + eps) * weight + bias
        expected = ((x.float() - running_mean.float()) / torch.sqrt(
            running_var.float() + eps
        ) * weight.float() + bias.float()).to(dtype)

        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestGroupNorm:
    """Tests for oasr.group_norm() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,channels,num_groups",
        [
            (2, 128, 64, 4),
            (4, 256, 128, 8),
        ],
    )
    def test_group_norm(self, batch_size, seq_len, channels, num_groups):
        eps = 1e-5
        dtype = torch.float32
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
        weight = torch.ones(channels, device="cuda", dtype=dtype)
        bias = torch.zeros(channels, device="cuda", dtype=dtype)

        output = oasr.group_norm(x, weight, bias, num_groups, eps)

        assert output.shape == x.shape


class TestAddLayerNorm:
    """Tests for oasr.add_layer_norm() functional API."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_add_layer_norm(self, dtype):
        batch_size, seq_len, hidden_size = 2, 128, 256
        eps = 1e-5
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        bias = torch.randn(hidden_size, device="cuda", dtype=dtype)

        output = oasr.add_layer_norm(x, residual, weight, bias, eps)

        # Reference: LayerNorm(x + residual)
        combined = x + residual
        expected = torch.nn.functional.layer_norm(combined, (hidden_size,), weight, bias, eps)

        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
