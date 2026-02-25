#!/usr/bin/env python3
"""
Unit tests for convolution.
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
    try:
        from oasr import ConvType, ActivationType
    except ImportError:
        from oasr._C import ConvType, ActivationType
        oasr.ConvType = ConvType
        oasr.ActivationType = ActivationType
    return oasr


class TestDepthwiseConv1D:
    """Tests for DepthwiseConv1D kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels,kernel_size", [
        (1, 64, 128, 3),
        (2, 128, 256, 7),
        (4, 256, 512, 31),
    ])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_depthwise_conv1d(self, oasr, batch_size, seq_len, channels, kernel_size, dtype):
        """Test DepthwiseConv1D against PyTorch reference."""
        padding = (kernel_size - 1) // 2
        
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
        bias = torch.randn(channels, device='cuda', dtype=dtype)
        
        dtype_map = {
            torch.float32: oasr.DataType.FP32,
            torch.float16: oasr.DataType.FP16,
        }
        
        output = oasr.kernels.conv.depthwise_conv1d(
            x, weight, bias,
            batch_size, seq_len, channels, kernel_size, padding,
            dtype_map[dtype]
        )
        oasr.synchronize()
        
        # PyTorch reference
        x_nchw = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        weight_pt = weight.permute(1, 0).view(channels, 1, kernel_size) # [channels, 1, kernel_size]
        ref_nchw = F.conv1d(x_nchw, weight_pt, bias=bias, stride=1, padding=padding, groups=channels)

        expected = ref_nchw.permute(0, 2, 1)  # [batch, seq_len, channels]
        
        rtol = 1e-4 if dtype == torch.float32 else 5e-2
        atol = 1e-4 if dtype == torch.float32 else 5e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("batch_size,seq_len,channels,kernel_size", [
        (1, 64, 128, 3),
        (2, 128, 256, 7),
        (4, 256, 512, 31),
    ])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_depthwise_conv1d_causal(self, oasr, batch_size, seq_len, channels, kernel_size, dtype):
        """Test causal DepthwiseConv1D."""
        padding = 0
        lorder = kernel_size - 1

        x = torch.randn(batch_size, seq_len + lorder, channels, device='cuda', dtype=dtype)
        weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
        bias = torch.randn(channels, device='cuda', dtype=dtype)

        dtype_map = {
            torch.float32: oasr.DataType.FP32,
            torch.float16: oasr.DataType.FP16,
        }
        
        output = oasr.kernels.conv.depthwise_conv1d(
            x, weight, bias,
            batch_size, seq_len + lorder, channels, kernel_size, padding,
            dtype_map[dtype]
        )
        oasr.synchronize()
        
        # Causal = left padding only
        x_nchw = x.permute(0, 2, 1)
        weight_pt = weight.permute(1, 0).view(channels, 1, kernel_size)
        ref_nchw = F.conv1d(x_nchw, weight_pt, bias=bias, stride=1, padding=padding, groups=channels)
        expected = ref_nchw.permute(0, 2, 1)

        rtol = 1e-4 if dtype == torch.float32 else 5e-2
        atol = 1e-4 if dtype == torch.float32 else 5e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestPointwiseConv1D:
    """Tests for PointwiseConv1D (linear projection) kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,in_ch,out_ch", [
        (2, 128, 256, 512),
        (4, 256, 512, 256),
        (2, 64, 768, 768),
    ])
    def test_pointwise_conv1d(self, oasr, batch_size, seq_len, in_ch, out_ch):
        """Test PointwiseConv1D against F.linear."""
        dtype = torch.float32
        
        x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
        weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
        bias = torch.randn(out_ch, device='cuda', dtype=dtype)
        
        output = oasr.kernels.conv.pointwise_conv1d(
            x, weight, bias,
            batch_size, seq_len, in_ch, out_ch,
            oasr.ActivationType.SWISH, False,
            oasr.DataType.FP32
        )
        oasr.synchronize()
        
        expected = F.linear(x, weight, bias)
        
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestGLU:
    """Tests for GLU activation kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels", [
        (2, 128, 256),
        (4, 256, 512),
    ])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_glu(self, oasr, batch_size, seq_len, channels, dtype):
        """Test GLU against F.glu."""
        x = torch.randn(batch_size, seq_len, 2 * channels, device='cuda', dtype=dtype)
        
        dtype_map = {
            torch.float32: oasr.DataType.FP32,
            torch.float16: oasr.DataType.FP16,
        }
        
        output = oasr.kernels.conv.glu(
            x,
            batch_size, seq_len, channels,
            dtype_map[dtype]
        )
        oasr.synchronize()
        
        expected = F.glu(x, dim=-1)
        
        rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestSwish:
    """Tests for Swish (SiLU) activation kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels", [
        (2, 128, 256),
        (4, 256, 512),
    ])
    def test_swish(self, oasr, batch_size, seq_len, channels):
        """Test Swish against F.silu."""
        dtype = torch.float32
        
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        
        output = oasr.kernels.conv.swish(
            x,
            batch_size, seq_len, channels,
            oasr.DataType.FP32
        )
        oasr.synchronize()
        
        expected = F.silu(x)
        
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


class TestBatchNormSwish:
    """Tests for fused BatchNorm + Swish kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels", [
        (2, 128, 256),
        (4, 64, 512),
    ])
    def test_batch_norm_swish(self, oasr, batch_size, seq_len, channels):
        """Test fused BatchNorm + Swish."""
        eps = 1e-5
        dtype = torch.float32
        
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        gamma = torch.randn(channels, device='cuda', dtype=dtype)
        beta = torch.randn(channels, device='cuda', dtype=dtype)
        running_mean = torch.randn(channels, device='cuda', dtype=dtype)
        running_var = torch.abs(torch.randn(channels, device='cuda', dtype=dtype)) + 0.1
        
        output = oasr.kernels.conv.batch_norm_swish(
            x,
            gamma, beta,
            running_mean, running_var,
            batch_size, seq_len, channels,
            eps, oasr.DataType.FP32
        )
        oasr.synchronize()
        
        # Reference: BatchNorm then Swish
        bn_out = (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
        expected = F.silu(bn_out)
        
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


class TestConv1D:
    """Tests for general Conv1D kernel."""

    @pytest.mark.parametrize("in_ch,out_ch,kernel_size,stride,padding", [
        (128, 256, 3, 1, 1),
        (256, 256, 5, 1, 2),
        (512, 256, 3, 2, 1),
    ])
    def test_conv1d_standard(self, oasr, in_ch, out_ch, kernel_size, stride, padding):
        """Test standard Conv1D."""
        batch_size, seq_len = 2, 64
        dtype = torch.float32
        
        x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
        weight = torch.randn(out_ch, in_ch, kernel_size, device='cuda', dtype=dtype)
        bias = torch.randn(out_ch, device='cuda', dtype=dtype)
        
        output = oasr.kernels.conv.conv1d(
            x, weight, bias,
            batch_size, seq_len, in_ch, out_ch,
            kernel_size, stride, padding, 1, 1,  # dilation=1, groups=1
            oasr.ConvType.STANDARD, oasr.DataType.FP32,
            True, False,  # channels_last, is_causal
            oasr.ActivationType.SWISH, False  # no activation
        )
        oasr.synchronize()
        
        # PyTorch reference
        x_nchw = x.permute(0, 2, 1)
        ref_nchw = F.conv1d(x_nchw, weight, bias, stride=stride, padding=padding)
        expected = ref_nchw.permute(0, 2, 1)
        
        torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


class TestConformerConvPattern:
    """Integration test for Conformer-style convolution pattern."""

    def test_conformer_conv_pattern(self, oasr):
        """Test full Conformer conv block pattern."""
        batch_size, seq_len, d_model = 2, 128, 256
        kernel_size = 31
        dtype = torch.float32
        
        # Inputs
        x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype)
        
        # Weights
        pw1_weight = torch.randn(2 * d_model, d_model, device='cuda', dtype=dtype)
        pw1_bias = torch.randn(2 * d_model, device='cuda', dtype=dtype)
        dw_weight = torch.randn(d_model, kernel_size, device='cuda', dtype=dtype)
        dw_bias = torch.randn(d_model, device='cuda', dtype=dtype)
        pw2_weight = torch.randn(d_model, d_model, device='cuda', dtype=dtype)
        pw2_bias = torch.randn(d_model, device='cuda', dtype=dtype)
        
        # OASR pipeline
        pw1_out = oasr.kernels.conv.pointwise_conv1d(
            x, pw1_weight, pw1_bias,
            batch_size, seq_len, d_model, 2 * d_model,
            oasr.ActivationType.SWISH, False, oasr.DataType.FP32
        )
        
        glu_out = oasr.kernels.conv.glu(
            pw1_out,
            batch_size, seq_len, d_model, oasr.DataType.FP32
        )
        
        dw_out = oasr.kernels.conv.depthwise_conv1d(
            glu_out, dw_weight, dw_bias,
            batch_size, seq_len, d_model, kernel_size, kernel_size // 2,
            oasr.DataType.FP32
        )
        
        swish_out = oasr.kernels.conv.swish(
            dw_out,
            batch_size, seq_len, d_model, oasr.DataType.FP32
        )
        
        output = oasr.kernels.conv.pointwise_conv1d(
            swish_out, pw2_weight, pw2_bias,
            batch_size, seq_len, d_model, d_model,
            oasr.ActivationType.SWISH, False, oasr.DataType.FP32
        )
        oasr.synchronize()
        
        # PyTorch reference
        ref_pw1 = F.linear(x, pw1_weight, pw1_bias)
        ref_glu = F.glu(ref_pw1, dim=-1)
        ref_glu_nchw = ref_glu.permute(0, 2, 1)
        dw_weight_pt = dw_weight.view(d_model, 1, kernel_size)
        ref_dw_nchw = F.conv1d(ref_glu_nchw, dw_weight_pt, dw_bias, padding=kernel_size // 2, groups=d_model)
        ref_dw = ref_dw_nchw.permute(0, 2, 1)
        ref_swish = F.silu(ref_dw)
        expected = F.linear(ref_swish, pw2_weight, pw2_bias)
        
        torch.testing.assert_close(output, expected, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
