#!/usr/bin/env python3
"""
Unit tests for functional convolution API (TVM-FFI JIT path).
"""

import pytest
import torch
import torch.nn.functional as F

import oasr


class TestDepthwiseConv1D:
    """Tests for oasr.depthwise_conv1d() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,channels,kernel_size",
        [
            (1, 64, 128, 3),
            (2, 128, 256, 7),
            (4, 256, 512, 31),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_depthwise_conv1d(self, batch_size, seq_len, channels, kernel_size, dtype):
        padding = (kernel_size - 1) // 2

        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
        weight = torch.randn(kernel_size, channels, device="cuda", dtype=dtype)
        bias = torch.randn(channels, device="cuda", dtype=dtype)

        output = oasr.depthwise_conv1d(x, weight, bias, padding)

        # PyTorch reference
        x_nchw = x.permute(0, 2, 1)
        weight_pt = weight.permute(1, 0).view(channels, 1, kernel_size)
        ref_nchw = F.conv1d(x_nchw, weight_pt, bias=bias, stride=1, padding=padding, groups=channels)
        expected = ref_nchw.permute(0, 2, 1)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_depthwise_conv1d_destination_passing(self, dtype):
        """Test depthwise conv1d with pre-allocated output tensor."""
        batch_size, seq_len, channels, kernel_size = 2, 128, 256, 7
        padding = (kernel_size - 1) // 2

        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
        weight = torch.randn(kernel_size, channels, device="cuda", dtype=dtype)
        bias = torch.randn(channels, device="cuda", dtype=dtype)
        out = torch.empty(batch_size, seq_len, channels, device="cuda", dtype=dtype)

        result = oasr.depthwise_conv1d(x, weight, bias, padding, out=out)

        assert result.data_ptr() == out.data_ptr()


class TestPointwiseConv1D:
    """Tests for oasr.pointwise_conv1d() functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,in_channels,out_channels",
        [
            (2, 128, 256, 512),
            (4, 256, 512, 256),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_pointwise_conv1d(self, batch_size, seq_len, in_channels, out_channels, dtype):
        x = torch.randn(batch_size, seq_len, in_channels, device="cuda", dtype=dtype)
        weight = torch.randn(out_channels, in_channels, device="cuda", dtype=dtype)

        output = oasr.pointwise_conv1d(x, weight)

        # Reference: reshape to [B*T, C_in] @ [C_out, C_in]^T = [B*T, C_out]
        x_flat = x.reshape(-1, in_channels)
        expected = torch.matmul(x_flat, weight.T).reshape(batch_size, seq_len, out_channels)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


class TestConv2D:
    """Tests for oasr.conv2d() functional API."""

    @pytest.mark.parametrize(
        "N,H,W,IC,K,R,S",
        [
            (1, 32, 32, 16, 32, 3, 3),
            (2, 16, 16, 32, 64, 3, 3),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv2d(self, N, H, W, IC, K, R, S, dtype):
        pad_h, pad_w = 1, 1
        stride_h, stride_w = 1, 1
        dilation_h, dilation_w = 1, 1

        # NHWC layout
        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype)
        # KRSC layout
        filt = torch.randn(K, R, S, IC, device="cuda", dtype=dtype)

        output = oasr.conv2d(
            x, filt, bias=None,
            pad_h=pad_h, pad_w=pad_w,
            stride_h=stride_h, stride_w=stride_w,
            dilation_h=dilation_h, dilation_w=dilation_w,
        )

        # Check output shape
        P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
        Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1
        assert output.shape == (N, P, Q, K)


class TestConv2DCudnn:
    """Tests for oasr.conv2d() with cuDNN backend (small IC, e.g. conformer subsampling)."""

    @pytest.mark.parametrize(
        "N,H,W,IC,K,R,S,pad_h,pad_w,stride_h,stride_w",
        [
            # IC=1: conformer subsampling first layer
            (1, 80, 100, 1, 32, 3, 3, 1, 1, 2, 2),
            (2, 80, 200, 1, 64, 3, 3, 1, 1, 2, 2),
            # IC=3: e.g. RGB-like input
            (1, 32, 32, 3, 32, 3, 3, 1, 1, 1, 1),
            # IC=4
            (2, 16, 16, 4, 64, 3, 3, 1, 1, 2, 2),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cudnn_conv2d(self, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w, dtype):
        dilation_h, dilation_w = 1, 1

        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype)
        filt = torch.randn(K, R, S, IC, device="cuda", dtype=dtype)

        output = oasr.conv2d(
            x, filt, bias=None,
            pad_h=pad_h, pad_w=pad_w,
            stride_h=stride_h, stride_w=stride_w,
            dilation_h=dilation_h, dilation_w=dilation_w,
        )

        # PyTorch reference: convert NHWC → NCHW for F.conv2d
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        # KRSC → KCRS (PyTorch standard)
        filt_nchw = filt.permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(
            x_nchw.float(), filt_nchw.float(),
            padding=(pad_h, pad_w), stride=(stride_h, stride_w),
            dilation=(dilation_h, dilation_w),
        )
        expected = ref_nchw.permute(0, 2, 3, 1).to(dtype)

        P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
        Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1
        assert output.shape == (N, P, Q, K)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cudnn_conv2d_with_bias(self, dtype):
        N, H, W, IC, K, R, S = 2, 80, 100, 1, 32, 3, 3
        pad_h, pad_w, stride_h, stride_w = 1, 1, 2, 2

        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype)
        filt = torch.randn(K, R, S, IC, device="cuda", dtype=dtype)
        bias = torch.randn(K, device="cuda", dtype=dtype)

        output = oasr.conv2d(
            x, filt, bias=bias,
            pad_h=pad_h, pad_w=pad_w,
            stride_h=stride_h, stride_w=stride_w,
        )

        # PyTorch reference
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        filt_nchw = filt.permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(
            x_nchw.float(), filt_nchw.float(), bias=bias.float(),
            padding=(pad_h, pad_w), stride=(stride_h, stride_w),
        )
        expected = ref_nchw.permute(0, 2, 3, 1).to(dtype)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "activation_type,activation_fn",
        [
            (0, lambda x: torch.relu(x)),                       # RELU
            (1, lambda x: torch.nn.functional.gelu(x)),         # GELU
            (2, lambda x: torch.nn.functional.silu(x)),         # SWISH
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cudnn_conv2d_activation(self, activation_type, activation_fn, dtype):
        N, H, W, IC, K, R, S = 1, 80, 100, 1, 32, 3, 3
        pad_h, pad_w, stride_h, stride_w = 1, 1, 2, 2

        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype)
        filt = torch.randn(K, R, S, IC, device="cuda", dtype=dtype)
        bias = torch.randn(K, device="cuda", dtype=dtype)

        output = oasr.conv2d_activation(
            x, filt, bias=bias,
            activation_type=activation_type,
            pad_h=pad_h, pad_w=pad_w,
            stride_h=stride_h, stride_w=stride_w,
        )

        # PyTorch reference
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        filt_nchw = filt.permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(
            x_nchw.float(), filt_nchw.float(), bias=bias.float(),
            padding=(pad_h, pad_w), stride=(stride_h, stride_w),
        )
        expected = activation_fn(ref_nchw).permute(0, 2, 3, 1).to(dtype)

        rtol, atol = (2e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
