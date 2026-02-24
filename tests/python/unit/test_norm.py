#!/usr/bin/env python3
"""
Unit tests for normalization kernels.
Uses torch.testing.assert_close for correctness verification.
"""

import pytest
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, 'python')


@pytest.fixture
def oasr():
    """Import oasr module."""
    import oasr
    return oasr


class TestLayerNorm:
    """Tests for LayerNorm kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,hidden_size", [
        (1, 64, 128),
        (2, 128, 256),
        (4, 256, 512),
        (8, 512, 768),
        (2, 128, 1024),
    ])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_layer_norm(self, oasr, batch_size, seq_len, hidden_size, dtype):
        """Test LayerNorm against PyTorch reference."""
        eps = 1e-5
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
        beta = torch.randn(hidden_size, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        # Get DataType enum
        dtype_map = {
            torch.float32: oasr.DataType.FP32,
            torch.float16: oasr.DataType.FP16,
            torch.bfloat16: oasr.DataType.BF16,
        }
        
        oasr.kernels.norm.layer_norm(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, seq_len, hidden_size,
            eps, dtype_map[dtype]
        )
        oasr.synchronize()
        
        # PyTorch reference
        layer_norm = torch.nn.LayerNorm(hidden_size, eps=eps, device='cuda', dtype=dtype)
        layer_norm.weight.data = gamma.clone()
        layer_norm.bias.data = beta.clone()
        expected = layer_norm(x)
        
        # Use torch.testing.assert_close for verification
        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    def test_layer_norm_no_beta(self, oasr):
        """Test LayerNorm without beta (bias)."""
        batch_size, seq_len, hidden_size = 2, 128, 256
        eps = 1e-5
        dtype = torch.float32
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        oasr.kernels.norm.layer_norm(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), 0,  # nullptr for beta
            batch_size, seq_len, hidden_size,
            eps, oasr.DataType.FP32
        )
        oasr.synchronize()
        
        # Manual reference without beta
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        expected = (x - mean) / torch.sqrt(var + eps) * gamma
        
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestRMSNorm:
    """Tests for RMSNorm kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,hidden_size", [
        (1, 64, 128),
        (2, 128, 256),
        (4, 256, 512),
    ])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_rms_norm(self, oasr, batch_size, seq_len, hidden_size, dtype):
        """Test RMSNorm against reference implementation."""
        eps = 1e-5
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
        beta = torch.randn(hidden_size, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        dtype_map = {
            torch.float32: oasr.DataType.FP32,
            torch.float16: oasr.DataType.FP16,
        }
        
        oasr.kernels.norm.rms_norm(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, seq_len, hidden_size,
            eps, dtype_map[dtype]
        )
        oasr.synchronize()
        
        # Reference: RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected = x / rms * gamma + beta
        
        rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestAddLayerNorm:
    """Tests for fused Add + LayerNorm kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,hidden_size", [
        (2, 128, 256),
        (4, 256, 512),
    ])
    def test_add_layer_norm(self, oasr, batch_size, seq_len, hidden_size):
        """Test fused Add + LayerNorm."""
        eps = 1e-5
        dtype = torch.float32
        
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        residual = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
        beta = torch.randn(hidden_size, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        oasr.kernels.norm.add_layer_norm(
            x.data_ptr(), residual.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, seq_len, hidden_size,
            eps, oasr.DataType.FP32
        )
        oasr.synchronize()
        
        # Reference
        layer_norm = torch.nn.LayerNorm(hidden_size, eps=eps, device='cuda', dtype=dtype)
        layer_norm.weight.data = gamma.clone()
        layer_norm.bias.data = beta.clone()
        expected = layer_norm(x + residual)
        
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestBatchNorm1D:
    """Tests for BatchNorm1D kernel (inference mode)."""

    @pytest.mark.parametrize("batch_size,seq_len,channels", [
        (2, 128, 256),
        (4, 64, 512),
    ])
    def test_batch_norm_1d(self, oasr, batch_size, seq_len, channels):
        """Test BatchNorm1D in inference mode."""
        eps = 1e-5
        dtype = torch.float32
        
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        gamma = torch.randn(channels, device='cuda', dtype=dtype)
        beta = torch.randn(channels, device='cuda', dtype=dtype)
        running_mean = torch.randn(channels, device='cuda', dtype=dtype)
        running_var = torch.abs(torch.randn(channels, device='cuda', dtype=dtype)) + 0.1
        output = torch.empty_like(x)
        
        oasr.kernels.norm.batch_norm_1d(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            running_mean.data_ptr(), running_var.data_ptr(),
            batch_size, seq_len, channels,
            eps, oasr.DataType.FP32
        )
        oasr.synchronize()
        
        # Reference: (x - mean) / sqrt(var + eps) * gamma + beta
        expected = (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
        
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestGroupNorm:
    """Tests for GroupNorm kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels,num_groups", [
        (2, 64, 256, 32),
        (4, 128, 512, 64),
        (2, 64, 128, 8),
        (2, 64, 4096, 2),
    ])
    def test_group_norm(self, oasr, batch_size, seq_len, channels, num_groups):
        """Test GroupNorm kernel."""
        eps = 1e-5
        dtype = torch.float32
        
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        gamma = torch.randn(channels, device='cuda', dtype=dtype)
        beta = torch.randn(channels, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        oasr.kernels.norm.group_norm(
            x.data_ptr(), output.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            batch_size, seq_len, channels, num_groups,
            eps, oasr.DataType.FP32
        )
        oasr.synchronize()
        
        # Reference (manual GroupNorm for [N, T, C] format)
        channels_per_group = channels // num_groups
        x_reshaped = x.view(batch_size, seq_len, num_groups, channels_per_group)
        mean = x_reshaped.mean(dim=-1, keepdim=True)
        var = x_reshaped.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_reshaped - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view(batch_size, seq_len, channels)
        expected = x_norm * gamma + beta
        
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
