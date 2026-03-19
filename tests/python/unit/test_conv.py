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


class TestDepthwiseConv1D:
    """Tests for DepthwiseConv1D kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels,kernel_size", [
        (1, 64, 128, 3),
        (2, 128, 256, 7),
        (4, 256, 512, 31),
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_depthwise_conv1d(self, batch_size, seq_len, channels, kernel_size, dtype):
        """Test DepthwiseConv1D against PyTorch reference."""
        padding = (kernel_size - 1) // 2

        x = torch.randn(batch_size, seq_len, channels,
                        device='cuda', dtype=dtype)
        weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
        bias = torch.randn(channels, device='cuda', dtype=dtype)

        output = oasr.kernels.conv.depthwise_conv1d(
            x, weight, bias, padding
        )
        oasr.synchronize()

        # PyTorch reference
        x_nchw = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        weight_pt = weight.permute(1, 0).view(
            channels, 1, kernel_size)  # [channels, 1, kernel_size]
        ref_nchw = F.conv1d(x_nchw, weight_pt, bias=bias,
                            stride=1, padding=padding, groups=channels)

        expected = ref_nchw.permute(0, 2, 1)  # [batch, seq_len, channels]

        rtol = 1e-2 if dtype == torch.float16 else 1e-2
        atol = 1e-2 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("batch_size,seq_len,channels,kernel_size", [
        (1, 64, 128, 3),
        (2, 128, 256, 7),
        (4, 256, 512, 31),
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_depthwise_conv1d_causal(self, batch_size, seq_len, channels, kernel_size, dtype):
        """Test causal DepthwiseConv1D."""
        padding = 0
        lorder = kernel_size - 1

        x = torch.randn(batch_size, seq_len + lorder,
                        channels, device='cuda', dtype=dtype)
        weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
        bias = torch.randn(channels, device='cuda', dtype=dtype)

        output = oasr.kernels.conv.depthwise_conv1d(
            x, weight, bias, padding
        )
        oasr.synchronize()

        # Causal = left padding only
        x_nchw = x.permute(0, 2, 1)
        weight_pt = weight.permute(1, 0).view(channels, 1, kernel_size)
        ref_nchw = F.conv1d(x_nchw, weight_pt, bias=bias,
                            stride=1, padding=padding, groups=channels)
        expected = ref_nchw.permute(0, 2, 1)

        rtol = 1e-2 if dtype == torch.float16 else 1e-2
        atol = 1e-2 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestDepthwiseConv1DSilu:
    """Tests for DepthwiseConv1D kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,channels,kernel_size", [
        (1, 64, 128, 3),
        (2, 128, 256, 7),
        (4, 256, 512, 31),
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_depthwise_conv1d_silu(self, batch_size, seq_len, channels, kernel_size, dtype):
        """Test DepthwiseConv1D against PyTorch reference."""
        padding = (kernel_size - 1) // 2

        x = torch.randn(batch_size, seq_len, channels,
                        device='cuda', dtype=dtype)
        weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
        bias = torch.randn(channels, device='cuda', dtype=dtype)

        output = oasr.kernels.conv.depthwise_conv1d_silu(
            x, weight, bias, padding
        )
        oasr.synchronize()

        # PyTorch reference
        x_nchw = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        weight_pt = weight.permute(1, 0).view(
            channels, 1, kernel_size)  # [channels, 1, kernel_size]
        ref_nchw = F.conv1d(x_nchw, weight_pt, bias=bias,
                            stride=1, padding=padding, groups=channels)

        # [batch, seq_len, channels]
        expected = F.silu(ref_nchw.permute(0, 2, 1))

        rtol = 1e-2 if dtype == torch.float16 else 1e-2
        atol = 1e-2 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("batch_size,seq_len,channels,kernel_size", [
        (1, 64, 128, 3),
        (2, 128, 256, 7),
        (4, 256, 512, 31),
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_depthwise_conv1d_causal_silu(self, batch_size, seq_len, channels, kernel_size, dtype):
        """Test causal DepthwiseConv1D."""
        padding = 0
        lorder = kernel_size - 1

        x = torch.randn(batch_size, seq_len + lorder,
                        channels, device='cuda', dtype=dtype)
        weight = torch.randn(kernel_size, channels, device='cuda', dtype=dtype)
        bias = torch.randn(channels, device='cuda', dtype=dtype)

        output = oasr.kernels.conv.depthwise_conv1d_silu(
            x, weight, bias, padding
        )
        oasr.synchronize()

        # Causal = left padding only
        x_nchw = x.permute(0, 2, 1)
        weight_pt = weight.permute(1, 0).view(channels, 1, kernel_size)
        ref_nchw = F.conv1d(x_nchw, weight_pt, bias=bias,
                            stride=1, padding=padding, groups=channels)
        expected = F.silu(ref_nchw.permute(0, 2, 1))

        rtol = 1e-2 if dtype == torch.float16 else 1e-2
        atol = 1e-2 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestPointwiseConv1D:
    """Tests for PointwiseConv1D (linear projection) kernel."""

    @pytest.mark.parametrize("batch_size,seq_len,in_ch,out_ch", [
        (2, 128, 256, 512),
        (4, 256, 512, 256),
        (2, 64, 768, 768),
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_pointwise_conv1d(self, batch_size, seq_len, in_ch, out_ch, dtype):
        """Test PointwiseConv1D against F.linear."""
        x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
        weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
        bias = torch.randn(out_ch, device='cuda', dtype=dtype)

        output = oasr.kernels.conv.pointwise_conv1d(
            x, weight, bias
        )
        oasr.synchronize()

        expected = F.linear(x, weight, bias).to(dtype)

        rtol = 1e-2 if dtype == torch.float16 else 1e-2
        atol = 1e-2 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("batch_size,seq_len,in_ch,out_ch", [
        (2, 128, 256, 512),
        (4, 256, 512, 256),
        (2, 64, 768, 768),
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_pointwise_conv1d_activation(self, batch_size, seq_len, in_ch, out_ch, dtype):
        """Test PointwiseConv1DActivation against F.linear."""
        x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
        weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
        bias = torch.randn(out_ch, device='cuda', dtype=dtype)

        output = oasr.kernels.conv.pointwise_conv1d_activation(
            x, weight, bias,
            oasr.ActivationType.SWISH,
        )
        oasr.synchronize()

        expected = F.silu(F.linear(x, weight, bias)).to(dtype)

        rtol = 1e-2 if dtype == torch.float16 else 1e-2
        atol = 1e-2 if dtype == torch.float16 else 1e-2
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


class TestConv2d:
    """Tests for the conv2d kernel (oasr.kernels.conv.conv2d).

    The CUTLASS backend operates on NHWC tensors and requires
    ``in_channels % 8 == 0`` and ``out_channels % 8 == 0``.

    The PyTorch reference uses F.conv2d on NCHW input/filter and then
    transposes the result back to NHWC for comparison.
    """

    @pytest.mark.parametrize("N,H,W,IC,K,R,S,pad,stride,dilation", [
        (2, 16, 16, 32,  64, 3, 3, 1, 1, 1),  # 3x3, same padding
        (1, 28, 28, 64, 128, 1, 1, 0, 1, 1),  # 1x1 pointwise
        (2, 14, 14, 128, 64, 3, 3, 1, 2, 1),  # stride-2 downsampling
        (1,  8,  8,  32, 32, 3, 3, 2, 1, 2),  # dilation-2
        (2, 200, 80,  1, 64, 3, 3, 1, 2, 1),  # IC=1 direct kernel, stride-2
        # IC=1 direct kernel, same padding
        (4, 100, 40,  1, 32, 3, 3, 1, 1, 1),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv2d(self, N, H, W, IC, K, R, S, pad, stride, dilation, dtype):
        """Test conv2d against F.conv2d reference."""
        # NHWC input and KRSC filter
        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype) * 0.1
        w = torch.randn(K, R, S, IC, device="cuda", dtype=dtype) * 0.1
        bias = torch.randn(K, device="cuda", dtype=dtype) * 0.1

        output = oasr.kernels.conv.conv2d(
            x, w, bias, pad, pad, stride, stride, dilation, dilation)
        oasr.synchronize()

        # Reference: F.conv2d expects NCHW input and KCRS filter
        x_nchw = x.float().permute(0, 3, 1, 2).contiguous()
        # [K, R, S, IC] -> [K, IC, R, S]
        w_kcrs = w.float().permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(x_nchw, w_kcrs, bias.float(),
                            padding=pad, stride=stride, dilation=dilation)
        # [N, K, P, Q] -> [N, P, Q, K]
        expected = ref_nchw.permute(0, 2, 3, 1).to(dtype)

        assert output.shape == expected.shape
        rtol, atol = (1e-2, 1e-2) if dtype == torch.float16 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("N,H,W,IC,K,R,S,pad,stride,dilation", [
        (2, 16, 16, 32, 64, 3, 3, 1, 1, 1),
        (1, 28, 28, 64, 128, 1, 1, 0, 1, 1),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv2d_no_bias(self, N, H, W, IC, K, R, S, pad, stride, dilation, dtype):
        """Test conv2d without bias."""
        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype) * 0.1
        w = torch.randn(K, R, S, IC, device="cuda", dtype=dtype) * 0.1

        output = oasr.kernels.conv.conv2d(
            x, w, None, pad, pad, stride, stride, dilation, dilation)
        oasr.synchronize()

        x_nchw = x.float().permute(0, 3, 1, 2).contiguous()
        w_kcrs = w.float().permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(x_nchw, w_kcrs, bias=None,
                            padding=pad, stride=stride, dilation=dilation)
        expected = ref_nchw.permute(0, 2, 3, 1).to(dtype)

        assert output.shape == expected.shape
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


class TestConv2dActivation:
    """Tests for the conv2d_activation kernel (fused epilogue)."""

    @pytest.mark.parametrize("N,H,W,IC,K,R,S,pad", [
        (2, 16, 16, 32,  64, 3, 3, 1),
        (1, 28, 28, 64, 128, 1, 1, 0),
        # IC=1 direct kernel with fused activation
        (2, 200, 80,  1, 64, 3, 3, 1),
    ])
    @pytest.mark.parametrize("activation,ref_fn", [
        (lambda: oasr.ActivationType.RELU, F.relu),
        (lambda: oasr.ActivationType.GELU, F.gelu),
        (lambda: oasr.ActivationType.SWISH, F.silu),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv2d_activation(self, N, H, W, IC, K, R, S, pad, activation, ref_fn, dtype):
        """Test conv2d_activation against F.conv2d + activation reference."""
        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype) * 0.1
        w = torch.randn(K, R, S, IC, device="cuda", dtype=dtype) * 0.1
        bias = torch.randn(K, device="cuda", dtype=dtype) * 0.1

        output = oasr.kernels.conv.conv2d_activation(
            x, w, bias, activation(), pad, pad, 1, 1, 1, 1
        )
        oasr.synchronize()

        x_nchw = x.float().permute(0, 3, 1, 2).contiguous()
        w_kcrs = w.float().permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(x_nchw, w_kcrs, bias.float(), padding=pad)
        expected = ref_fn(ref_nchw.permute(0, 2, 3, 1)).to(dtype)

        assert output.shape == expected.shape
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


class TestConv2dLayer:
    """Tests for the Conv2d and Conv2dActivation nn.Module wrappers."""

    @pytest.mark.parametrize("N,H,W,IC,K,kernel_size,padding,stride", [
        (2, 16, 16, 32,  64, 3, 1, 1),
        (1, 28, 28, 64, 128, 1, 0, 1),
        (2, 14, 14, 64,  32, 3, 1, 2),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv2d_layer(self, N, H, W, IC, K, kernel_size, padding, stride, dtype):
        """Test Conv2d layer forward pass against F.conv2d reference."""
        layer = oasr.Conv2d(
            in_channels=IC, out_channels=K,
            kernel_size=kernel_size, padding=padding, stride=stride,
            bias=True, dtype=dtype, device="cuda",
        )

        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype) * 0.1
        output = layer(x)
        oasr.synchronize()

        # Reference: convert weights back to KCRS for F.conv2d
        w_kcrs = layer.weight.float().permute(
            0, 3, 1, 2).contiguous()  # [K,R,S,IC] -> [K,IC,R,S]
        ref_nchw = F.conv2d(
            x.float().permute(0, 3, 1, 2).contiguous(),
            w_kcrs, layer.bias.float(),
            padding=padding, stride=stride,
        )
        expected = ref_nchw.permute(0, 2, 3, 1).to(dtype)

        assert output.shape == expected.shape
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("activation_type,ref_fn", [
        ("relu",  F.relu),
        ("gelu",  F.gelu),
        ("swish", F.silu),
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv2d_activation_layer(self, activation_type, ref_fn, dtype):
        """Test Conv2dActivation layer forward pass against F.conv2d + activation."""
        N, H, W, IC, K = 2, 16, 16, 32, 64
        layer = oasr.Conv2dActivation(
            in_channels=IC, out_channels=K,
            kernel_size=3, padding=1,
            activation_type=activation_type,
            bias=True, dtype=dtype, device="cuda",
        )

        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype) * 0.1
        output = layer(x)
        oasr.synchronize()

        w_kcrs = layer.weight.float().permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(
            x.float().permute(0, 3, 1, 2).contiguous(),
            w_kcrs, layer.bias.float(), padding=1,
        )
        expected = ref_fn(ref_nchw.permute(0, 2, 3, 1)).to(dtype)

        assert output.shape == expected.shape
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_load_from_pytorch_conv2d(self):
        """Test loading weights from a standard PyTorch nn.Conv2d state dict."""
        dtype = torch.float16
        IC, K, R = 32, 64, 3
        pt_conv = torch.nn.Conv2d(
            IC, K, R, padding=1, bias=True).cuda().to(dtype)
        oasr_conv = oasr.Conv2d(IC, K, R, padding=1,
                                bias=True, dtype=dtype, device="cuda")

        oasr_conv.load_state_dict(pt_conv.state_dict())

        # Verify the weight was transposed correctly: [K, IC, R, S] -> [K, R, S, IC]
        assert oasr_conv.weight.shape == (K, R, R, IC)
        expected_w = pt_conv.weight.permute(0, 2, 3, 1).contiguous()
        torch.testing.assert_close(oasr_conv.weight.data, expected_w)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
