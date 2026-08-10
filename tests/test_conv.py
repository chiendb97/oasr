#!/usr/bin/env python3
"""
Unit tests for functional convolution API (TVM-FFI JIT path).
"""

import pytest
import torch
import torch.nn.functional as F

import oasr
from oasr.jit.conv import (
    CONV2D_DEFAULT,
    get_unique_conv2d_compile_configs,
    select_default_conv1d_activation_config,
    select_default_conv1d_config,
)

# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")


class TestDenseConv1D:
    """Tests for the packed-BTC dense Conv1D functional API."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,in_channels,out_channels,kernel_size,padding,stride,dilation",
        [
            (1, 3000, 80, 384, 3, 1, 1, 1),
            (1, 3000, 384, 384, 3, 1, 2, 1),
            (2, 65, 64, 128, 5, 4, 2, 2),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv1d(
        self,
        batch_size,
        seq_len,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        dilation,
        dtype,
    ):
        x = torch.randn(batch_size, seq_len, in_channels, device="cuda", dtype=dtype)
        weight = torch.randn(out_channels, kernel_size, in_channels, device="cuda", dtype=dtype)
        bias = torch.randn(out_channels, device="cuda", dtype=dtype)

        output = oasr.conv1d(x, weight, bias, padding, stride, dilation)
        expected = F.conv1d(
            x.transpose(1, 2),
            weight.permute(0, 2, 1),
            bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
        ).transpose(1, 2)

        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
        assert output.is_contiguous()

    @pytest.mark.parametrize("activation,fn", [(0, F.relu), (2, F.silu)])
    def test_conv1d_activation(self, activation, fn):
        x = torch.randn(2, 127, 64, device="cuda", dtype=torch.float16)
        weight = torch.randn(128, 3, 64, device="cuda", dtype=torch.float16)
        bias = torch.randn(128, device="cuda", dtype=torch.float16)

        output = oasr.conv1d_activation(x, weight, bias, activation, padding=1)
        expected = fn(
            F.conv1d(x.transpose(1, 2), weight.permute(0, 2, 1), bias, padding=1)
        ).transpose(1, 2)
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_paraformer_conv1d_relu_production_tactic(self, dtype):
        x = torch.randn(1, 502, 512, device="cuda", dtype=dtype)
        weight = torch.randn(512, 3, 512, device="cuda", dtype=dtype)
        bias = torch.randn(512, device="cuda", dtype=dtype)

        output = oasr.conv1d_activation(x, weight, bias, 0)
        expected = F.relu(F.conv1d(x.transpose(1, 2), weight.permute(0, 2, 1), bias)).transpose(
            1, 2
        )

        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    def test_conv1d_destination_passing(self):
        x = torch.randn(2, 31, 64, device="cuda", dtype=torch.float16)
        weight = torch.randn(128, 3, 64, device="cuda", dtype=torch.float16)
        out = torch.empty(2, 16, 128, device="cuda", dtype=torch.float16)

        result = oasr.conv1d(x, weight, padding=1, stride=2, out=out)

        assert result.data_ptr() == out.data_ptr()

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 3000, 80, 384, 3, 1, 1, 1),
            (1, 3000, 384, 384, 3, 1, 2, 1),
            (1, 3000, 128, 1280, 3, 1, 1, 1),
            (1, 3000, 1280, 1280, 3, 1, 2, 1),
            (1, 502, 512, 512, 3, 0, 1, 1),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sm120_production_tactic_is_compiled(self, shape, dtype):
        cfg = select_default_conv1d_config(*shape, dtype, 120)
        assert cfg.compile_name != CONV2D_DEFAULT.compile_name
        assert cfg.compile_name in get_unique_conv2d_compile_configs(120)

    def test_unmeasured_shape_uses_default(self):
        shape = (2, 3000, 80, 384, 3, 1, 1, 1)
        assert select_default_conv1d_config(*shape, torch.float16, 120) is CONV2D_DEFAULT
        assert select_default_conv1d_config(*shape, torch.bfloat16, 120) is CONV2D_DEFAULT
        assert select_default_conv1d_config(*shape, torch.float16, 80) is CONV2D_DEFAULT

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sm120_paraformer_activation_tactic_is_compiled(self, dtype):
        shape = (1, 502, 512, 512, 3, 0, 1, 1)
        cfg = select_default_conv1d_activation_config(*shape, dtype, 120)
        assert cfg.compile_name != CONV2D_DEFAULT.compile_name
        assert cfg.compile_name in get_unique_conv2d_compile_configs(120)


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
        ref_nchw = F.conv1d(
            x_nchw, weight_pt, bias=bias, stride=1, padding=padding, groups=channels
        )
        expected = ref_nchw.permute(0, 2, 1)

        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

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

    @pytest.mark.parametrize("padding", [(8, 2), (10, 0), (0, 10)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_asymmetric_padding(self, padding, dtype):
        batch_size, seq_len, channels, kernel_size = 2, 37, 128, 11
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
        weight = torch.randn(kernel_size, channels, device="cuda", dtype=dtype)
        bias = torch.randn(channels, device="cuda", dtype=dtype)

        output = oasr.depthwise_conv1d(x, weight, bias, padding)
        expected = F.conv1d(
            F.pad(x.transpose(1, 2), padding),
            weight.T.unsqueeze(1),
            bias,
            groups=channels,
        ).transpose(1, 2)

        assert output.shape == x.shape
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float16])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_fsmn_mask_and_input(self, mask_dtype, dtype):
        batch_size, seq_len, channels, kernel_size = 2, 43, 128, 11
        padding = (7, 3)
        x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
        weight = torch.randn(kernel_size, channels, device="cuda", dtype=dtype)
        # Include internal holes, not only right padding: this catches applying
        # the mask only after convolution instead of on every input tap.
        mask_bool = torch.rand(batch_size, seq_len, 1, device="cuda") > 0.25
        mask = mask_bool if mask_dtype is torch.bool else mask_bool.to(dtype)

        output = oasr.depthwise_conv1d(
            x,
            weight,
            padding=padding,
            mask=mask,
            add_input=True,
        )
        masked = x * mask
        conv = F.conv1d(
            F.pad(masked.transpose(1, 2), padding),
            weight.T.unsqueeze(1),
            groups=channels,
        ).transpose(1, 2)
        expected = (conv + masked) * mask

        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_asymmetric_silu(self, dtype):
        seq_len, channels, kernel_size = 31, 128, 7
        padding = (5, 1)
        x = torch.randn(1, seq_len, channels, device="cuda", dtype=dtype)
        weight = torch.randn(kernel_size, channels, device="cuda", dtype=dtype)
        bias = torch.randn(channels, device="cuda", dtype=dtype)

        output = oasr.depthwise_conv1d_silu(x, weight, bias, padding)
        expected = F.silu(
            F.conv1d(
                F.pad(x.transpose(1, 2), padding),
                weight.T.unsqueeze(1),
                bias,
                groups=channels,
            )
        ).transpose(1, 2)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


class TestPointwiseConv1dLayer:
    """Tests for oasr.layers.PointwiseConv1d (delegates to oasr.gemm)."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,in_channels,out_channels",
        [
            (2, 128, 256, 512),
            (4, 256, 512, 256),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_pointwise_conv1d_layer(self, batch_size, seq_len, in_channels, out_channels, dtype):
        layer = oasr.PointwiseConv1d(in_channels, out_channels, bias=True).cuda().to(dtype)
        x = torch.randn(batch_size, seq_len, in_channels, device="cuda", dtype=dtype)

        output = layer(x)

        assert output.shape == (batch_size, seq_len, out_channels)
        # Reference via plain matmul with the same weights
        weight = layer.weight.squeeze(-1)  # [out_channels, in_channels]
        x_flat = x.reshape(-1, in_channels)
        expected = torch.matmul(x_flat, weight.T)
        if layer.bias is not None:
            expected = expected + layer.bias
        expected = expected.reshape(batch_size, seq_len, out_channels)
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
            x,
            filt,
            bias=None,
            pad_h=pad_h,
            pad_w=pad_w,
            stride_h=stride_h,
            stride_w=stride_w,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
        )

        # Check output shape
        P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
        Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1
        assert output.shape == (N, P, Q, K)


class TestGroupedConv2D:
    """Grouped/depthwise direct NHWC kernel and 1x1 GEMM specialization."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "N,H,W,IC,K,R,S,pad,stride,groups",
        [
            (2, 31, 17, 16, 16, 3, 3, 1, 1, 16),
            (1, 25, 13, 128, 128, 7, 7, 3, 1, 128),
            (2, 19, 11, 16, 32, 3, 3, 1, 2, 4),
            (1, 13, 9, 8, 8, 5, 5, 2, 1, 2),
        ],
    )
    def test_grouped_conv2d(self, dtype, N, H, W, IC, K, R, S, pad, stride, groups):
        x = torch.randn(N, H, W, IC, device="cuda", dtype=dtype)
        weight = torch.randn(K, R, S, IC // groups, device="cuda", dtype=dtype)
        bias = torch.randn(K, device="cuda", dtype=dtype)

        got = oasr.conv2d(x, weight, bias, pad, pad, stride, stride, groups=groups)
        ref = F.conv2d(
            x.permute(0, 3, 1, 2),
            weight.permute(0, 3, 1, 2),
            bias,
            padding=pad,
            stride=stride,
            groups=groups,
        ).permute(0, 2, 3, 1)

        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
        assert got.is_contiguous()

    @pytest.mark.parametrize(
        "activation,fn",
        [(0, F.relu), (1, lambda x: F.gelu(x, approximate="tanh")), (2, F.silu)],
    )
    def test_grouped_conv2d_activation(self, activation, fn):
        x = torch.randn(2, 23, 13, 16, device="cuda", dtype=torch.float16)
        weight = torch.randn(16, 3, 3, 1, device="cuda", dtype=torch.float16)
        bias = torch.randn(16, device="cuda", dtype=torch.float16)

        got = oasr.conv2d_activation(x, weight, bias, activation, 1, 1, groups=16)
        ref = fn(
            F.conv2d(
                x.permute(0, 3, 1, 2),
                weight.permute(0, 3, 1, 2),
                bias,
                padding=1,
                groups=16,
            )
        ).permute(0, 2, 3, 1)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_pointwise_conv2d_uses_destination(self, dtype):
        x = torch.randn(2, 17, 9, 64, device="cuda", dtype=dtype)
        weight = torch.randn(128, 1, 1, 64, device="cuda", dtype=dtype)
        bias = torch.randn(128, device="cuda", dtype=dtype)
        out = torch.empty(2, 17, 9, 128, device="cuda", dtype=dtype)

        got = oasr.conv2d(x, weight, bias, out=out)
        ref = F.conv2d(x.permute(0, 3, 1, 2), weight.permute(0, 3, 1, 2), bias).permute(0, 2, 3, 1)
        assert got.data_ptr() == out.data_ptr()
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    def test_invalid_groups_rejected_before_jit(self):
        x = torch.randn(1, 5, 5, 8, device="cuda", dtype=torch.float16)
        weight = torch.randn(8, 3, 3, 2, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="must divide"):
            oasr.conv2d(x, weight, groups=3)


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
            x,
            filt,
            bias=None,
            pad_h=pad_h,
            pad_w=pad_w,
            stride_h=stride_h,
            stride_w=stride_w,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
        )

        # PyTorch reference: convert NHWC → NCHW for F.conv2d
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        # KRSC → KCRS (PyTorch standard)
        filt_nchw = filt.permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(
            x_nchw.float(),
            filt_nchw.float(),
            padding=(pad_h, pad_w),
            stride=(stride_h, stride_w),
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
            x,
            filt,
            bias=bias,
            pad_h=pad_h,
            pad_w=pad_w,
            stride_h=stride_h,
            stride_w=stride_w,
        )

        # PyTorch reference
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        filt_nchw = filt.permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(
            x_nchw.float(),
            filt_nchw.float(),
            bias=bias.float(),
            padding=(pad_h, pad_w),
            stride=(stride_h, stride_w),
        )
        expected = ref_nchw.permute(0, 2, 3, 1).to(dtype)

        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "activation_type,activation_fn",
        [
            (0, lambda x: torch.relu(x)),  # RELU
            (1, lambda x: torch.nn.functional.gelu(x)),  # GELU
            (2, lambda x: torch.nn.functional.silu(x)),  # SWISH
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
            x,
            filt,
            bias=bias,
            activation_type=activation_type,
            pad_h=pad_h,
            pad_w=pad_w,
            stride_h=stride_h,
            stride_w=stride_w,
        )

        # PyTorch reference
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        filt_nchw = filt.permute(0, 3, 1, 2).contiguous()
        ref_nchw = F.conv2d(
            x_nchw.float(),
            filt_nchw.float(),
            bias=bias.float(),
            padding=(pad_h, pad_w),
            stride=(stride_h, stride_w),
        )
        expected = activation_fn(ref_nchw).permute(0, 2, 3, 1).to(dtype)

        rtol, atol = (2e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
