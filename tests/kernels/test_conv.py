#!/usr/bin/env python3
"""
Unit tests for functional convolution API (TVM-FFI JIT path).
"""

import functools

import pytest
import torch
import torch.nn.functional as F

import oasr
from oasr.functionals.conv import _get_conv2d_module
from oasr.jit.conv import (
    _CONV1D_ACTIVATION_HEURISTIC_RULES,
    _CONV1D_HEURISTIC_RULES,
    CONV2D_DEFAULT,
    CutlassConv2dConfig,
    conv2d_activation_func_name,
    conv2d_func_name,
    get_unique_conv2d_compile_configs,
    heuristic_inactive,
    reset_heuristic_stats,
    select_default_conv1d_activation_config,
    select_default_conv1d_config,
    select_default_conv2d_config,
)
from oasr.jit.core import _TARGET_SMS, _get_target_sm

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

    def test_the_tables_are_registered_per_sm_family(self):
        """The Conv1D half of the same defect as the GEMM heuristic: the selector
        opened with ``if sm != 120``, so which architectures are measured was a
        branch with one possible answer.  It is these dicts' keys now, and a key
        outside the compiled families would be data no call can reach."""
        for registry in (_CONV1D_HEURISTIC_RULES, _CONV1D_ACTIVATION_HEURISTIC_RULES):
            assert registry, "no Conv1D rule table is registered at all"
            assert set(registry) <= set(_TARGET_SMS), (
                f"tables registered for {sorted(set(registry) - set(_TARGET_SMS))}, "
                f"which _SM_FAMILY never resolves to"
            )

    def test_every_table_only_names_tiles_its_own_arch_compiles(self):
        """A rule naming a config outside its arch's emitted set raises
        ``AttributeError: Module has no function …`` on the first call that hits
        it — on that architecture only.  Both sides are pure functions of ``sm``,
        so every registered table is checked from any box."""
        for registry in (_CONV1D_HEURISTIC_RULES, _CONV1D_ACTIVATION_HEURISTIC_RULES):
            for sm, by_dtype in registry.items():
                compiled = get_unique_conv2d_compile_configs(sm)
                for dtype, by_shape in by_dtype.items():
                    for shape, cfg in by_shape.items():
                        where = f"sm{sm} {dtype} {shape}"
                        assert cfg.kSmVersion == sm, (
                            f"{where}: registered under sm{sm} but names a "
                            f"kSmVersion={cfg.kSmVersion} config"
                        )
                        assert (
                            cfg.compile_name in compiled
                        ), f"{where}: {cfg.compile_name} is not in sm{sm}'s emitted set"

    def test_an_untuned_arch_is_counted_not_silent(self):
        """Conv1D has no per-shape miss table, so an architecture with no rules at
        all was invisible in every counter — ``format_gap_report`` said "every
        call reached an OASR kernel" while every conv1d ran the fallback tile."""
        sm = next((s for s in _TARGET_SMS if s not in _CONV1D_HEURISTIC_RULES), None)
        if sm is None:
            pytest.skip("every compiled SM family has a tuned Conv1D table")
        shape = (1, 3000, 80, 384, 3, 1, 1, 1)
        reset_heuristic_stats()
        assert select_default_conv1d_config(*shape, torch.float16, sm) is CONV2D_DEFAULT
        assert heuristic_inactive() == {sm: 1}
        reset_heuristic_stats()
        select_default_conv1d_config(*shape, torch.float16, 120)
        assert not heuristic_inactive()

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

        P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
        Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1
        assert output.shape == (N, P, Q, K)

    def test_untuned_tile_is_the_narrowest_covering_n_extent(self):
        """``select_default_conv2d_config`` may only narrow the default's N tile.

        Same M and K extents and stage count, so the grid and the accumulation
        order are the default's; and the narrowest compiled N extent that still
        covers the output channels, so no CTA computes columns it discards.
        """
        sm = _get_target_sm()
        default = CONV2D_DEFAULT
        if not isinstance(default, CutlassConv2dConfig):
            pytest.skip("a CUTLASS 3.x architecture keeps its default tile")
        mainloop = (default.block_m, default.block_k, default.warp_k, default.kStages)
        widths = sorted(
            {
                cfg.block_n
                for cfg in get_unique_conv2d_compile_configs(sm).values()
                if (cfg.block_m, cfg.block_k, cfg.warp_k, cfg.kStages) == mainloop
            }
        )
        for k in (1, 8, 16, 31, 32, 33, 48, 64, 65, 96, 128, 129, 512):
            cfg = select_default_conv2d_config(k, sm)
            assert cfg.compile_name in get_unique_conv2d_compile_configs(sm)
            assert (cfg.block_m, cfg.block_k, cfg.warp_k, cfg.kStages) == mainloop
            covering = [n for n in widths if k <= n < default.block_n]
            if covering:
                assert cfg.block_n == covering[0], f"K={k}: block_n {cfg.block_n}"
            else:
                assert cfg is default, f"K={k}: {cfg.compile_name}"

    @pytest.mark.parametrize("K", [8, 32, 64])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_untuned_tile_is_bit_identical_to_the_default(self, K, dtype):
        """Narrowing the N tile is a pure speed choice (Zipformer's ``8 -> 32``
        encoder_embed conv: 3.0x), so the default tile is an exact oracle."""
        if select_default_conv2d_config(K, _get_target_sm()) is CONV2D_DEFAULT:
            pytest.skip(f"no narrower compiled tile covers K={K} on this architecture")
        torch.manual_seed(0)
        x = torch.randn(3, 41, 80, 8, device="cuda", dtype=dtype)
        w = torch.randn(K, 3, 3, 8, device="cuda", dtype=dtype) * 0.1
        b = torch.randn(K, device="cuda", dtype=dtype)
        module = _get_conv2d_module()

        got = oasr.conv2d(x, w, b, stride_h=2, stride_w=2)
        want = torch.full_like(got, float("nan"))
        getattr(module, conv2d_func_name(CONV2D_DEFAULT))(want, x, w, b, 0, 0, 2, 2, 1, 1)
        assert torch.equal(got, want)

        got = oasr.conv2d_activation(x, w, b, activation_type=0, stride_h=2, stride_w=2)
        want = torch.full_like(got, float("nan"))
        getattr(module, conv2d_activation_func_name(CONV2D_DEFAULT))(
            want, x, w, b, 0, 0, 0, 2, 2, 1, 1
        )
        assert torch.equal(got, want)


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

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "N,H,W,C,pad_h,pad_w",
        [
            (2, 14, 19, 128, 3, 3),  # Zipformer's ConvNeXt, offline
            (2, 14, 19, 128, 0, 3),  # ... streaming (left context is cached, not padded)
            (1, 11, 40, 64, 3, 3),  # more than one 32-column tile
            (3, 9, 5, 40, 3, 3),  # a channel group running past the last channel
        ],
    )
    def test_tiled_depthwise_7x7(self, dtype, N, H, W, C, pad_h, pad_w):
        """The shared-memory tiled path (stride-1, undilated 7x7 depthwise)."""
        x = torch.randn(N, H, W, C, device="cuda", dtype=dtype)
        weight = torch.randn(C, 7, 7, 1, device="cuda", dtype=dtype)
        bias = torch.randn(C, device="cuda", dtype=dtype)
        got = oasr.conv2d(x, weight, bias, pad_h, pad_w, 1, 1, groups=C)
        ref = F.conv2d(
            x.permute(0, 3, 1, 2).float(),
            weight.permute(0, 3, 1, 2).float(),
            bias.float(),
            padding=(pad_h, pad_w),
            groups=C,
        ).permute(0, 2, 3, 1)
        torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=5e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_tiled_depthwise_staging_paths_agree(self, dtype):
        """A misaligned input takes the element-wise staging; the result must be
        bit-identical to the 16-byte staging of the same values."""
        x = torch.randn(2, 12, 19, 128, device="cuda", dtype=dtype)
        weight = torch.randn(128, 7, 7, 1, device="cuda", dtype=dtype)
        bias = torch.randn(128, device="cuda", dtype=dtype)
        backing = torch.empty(x.numel() + 1, device="cuda", dtype=dtype)
        shifted = backing[1:].view_as(x)  # 2-byte offset: not 16-byte aligned
        shifted.copy_(x)
        aligned = oasr.conv2d(x, weight, bias, 3, 3, 1, 1, groups=128)
        unaligned = oasr.conv2d(shifted, weight, bias, 3, 3, 1, 1, groups=128)
        assert torch.equal(aligned, unaligned)

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


class TestRowSlices:
    """Where a batch is split, decided from the addressable extent, not guessed."""

    def test_a_batch_that_fits_is_one_launch(self):
        from oasr.functionals.conv import _row_slices

        x = torch.empty(3, 512, 520, 512, device="meta", dtype=torch.bfloat16)
        assert _row_slices("conv2d", x, x[:, ::2, ::2, :32]) is None

    def test_the_split_bounds_every_tensor_at_2gib(self):
        from oasr.functionals.conv import _MAX_LAUNCH_TENSOR_BYTES, _row_slices

        x = torch.empty(9, 512, 520, 512, device="meta", dtype=torch.bfloat16)
        out = torch.empty(9, 255, 259, 32, device="meta", dtype=torch.bfloat16)
        slices = _row_slices("conv2d", x, out)
        per_row = x.stride(0) * x.element_size()
        assert [s.start for s in slices] == list(range(0, 9, slices[0].stop))
        assert slices[-1].stop == 9
        rows = slices[0].stop
        assert rows * per_row <= _MAX_LAUNCH_TENSOR_BYTES < (rows + 1) * per_row

    def test_the_widest_tensor_decides(self):
        """An output wider than the input (a channel-expanding conv) sets the split."""
        from oasr.functionals.conv import _MAX_LAUNCH_TENSOR_BYTES, _row_slices

        x = torch.empty(64, 30000, 80, 1, device="meta", dtype=torch.float16)
        out = torch.empty(64, 29998, 80, 8, device="meta", dtype=torch.float16)
        rows = _row_slices("conv2d", x, out)[0].stop
        assert rows * out.stride(0) * out.element_size() <= _MAX_LAUNCH_TENSOR_BYTES

    def test_a_row_too_large_to_address_is_refused(self):
        """No split along the batch can fix it, and launching it corrupts memory."""
        from oasr.functionals.conv import _row_slices

        x = torch.empty(2, 400000, 1024, 4, device="meta", dtype=torch.float16)
        with pytest.raises(ValueError, match="one batch row spans 3.05 GiB"):
            _row_slices("conv2d", x, x)


@pytest.mark.cuda
@pytest.mark.slow
class TestConvLargeTensor:
    """Every convolution entry point on a batch one launch cannot address.

    CUTLASS's implicit-GEMM convolutions (Conv2D, and Conv1D as a height-one
    Conv2D) address their tensors with 32-bit **byte** offsets, and past 2 GiB
    refuse the problem.  At ``max_batch_size >= ~220`` that once came back as an
    **engine-wide empty transcript**: the Conformer subsampling's second conv
    crossed the limit, the launcher raised, and the micro-batch retry that would
    have named it died on already-released waveforms.  The in-tree grouped and
    depthwise kernels used 32-bit *element* offsets and did worse past 2^31
    elements: an illegal address that poisons the CUDA context (depthwise
    ``conv2d``), or a silently wrong tail of the batch (``depthwise_conv1d``).

    Only ``conv2d_activation`` split the batch.  Every entry point now does, by
    the same rule, and the in-tree kernels address with 64-bit offsets.  The
    oracle is the same op on one row at a time rather than a torch reference:
    what is checked is that tiling changes nothing, and a bit-exact comparison
    says that where a tolerance would not.
    """

    @staticmethod
    def _rand(*shape, dtype=torch.bfloat16, scale=0.1):
        return torch.randn(*shape, device="cuda", dtype=dtype) * scale

    @staticmethod
    def _per_row(op, x, want, *args, **kwargs):
        for i in range(x.shape[0]):
            op(x[i : i + 1], *args, out=want[i : i + 1], **kwargs)
        return want

    @pytest.mark.parametrize("fused", [True, False], ids=["conv2d_activation", "conv2d"])
    def test_conv2d_over_2gib_matches_per_row(self, fused):
        # 8 * 512 * 520 * 512 * 2 B = 2.03 GiB of input — just past the limit,
        # and the smallest shape that crosses it without a large output.
        N, H, W, IC, K = 8, 512, 520, 512, 32
        assert N * H * W * IC * 2 > 2**31, "shape must exceed the int32 byte limit"
        torch.manual_seed(0)
        x = self._rand(N, H, W, IC)
        filt = self._rand(K, 3, 3, IC, scale=0.02)
        bias = self._rand(K)
        if fused:
            op = functools.partial(oasr.conv2d_activation, activation_type=0)
        else:
            op = oasr.conv2d
        got = op(x, filt, bias, stride_h=2, stride_w=2)
        want = self._per_row(op, x, torch.empty_like(got), filt, bias, stride_h=2, stride_w=2)
        assert torch.equal(got, want)

    def test_depthwise_conv2d_past_2_31_elements_matches_per_row(self):
        """Nemotron's subsampling shape: 10 rows of 300 s.  ``N * P`` exceeds the
        packed kernel's grid, so this was the general grouped kernel — whose
        ``int`` offsets wrapped into an illegal address."""
        N, H, W, C = 10, 15000, 64, 256
        assert N * H * W * C > 2**31, "shape must exceed int32 element offsets"
        torch.manual_seed(0)
        x = self._rand(N, H, W, C, dtype=torch.float16)
        filt = self._rand(C, 3, 3, 1, dtype=torch.float16)
        bias = self._rand(C, dtype=torch.float16)
        got = oasr.conv2d(x, filt, bias, stride_h=2, stride_w=2, groups=C)
        want = self._per_row(
            oasr.conv2d, x, torch.empty_like(got), filt, bias, stride_h=2, stride_w=2, groups=C
        )
        assert torch.equal(got, want)

    @pytest.mark.parametrize("fused", [False, True], ids=["conv1d", "conv1d_activation"])
    def test_conv1d_over_2gib_matches_per_row(self, fused):
        B, T, C = 300, 3000, 1280  # Whisper's second frontend conv; 2.15 GiB
        assert B * T * C * 2 > 2**31
        torch.manual_seed(0)
        x = self._rand(B, T, C)
        filt = self._rand(C, 3, C, scale=0.02)
        bias = self._rand(C)
        op = functools.partial(oasr.conv1d_activation, activation_type=1) if fused else oasr.conv1d
        got = op(x, filt, bias, padding=1)
        want = self._per_row(op, x, torch.empty_like(got), filt, bias, padding=1)
        assert torch.equal(got, want)

    @pytest.mark.parametrize("variant", ["plain", "fsmn", "silu"])
    def test_depthwise_conv1d_past_2_31_elements_matches_per_row(self, variant):
        """A Conformer-width conv module over 56 rows of 300 s.  On ``main`` the
        rows past 2^31 elements came back silently wrong (max error 5.7)."""
        B, T, C, K = 56, 30000, 1280, 15
        assert B * T * C > 2**31, "shape must exceed int32 element offsets"
        torch.manual_seed(0)
        x = self._rand(B, T, C, dtype=torch.float16)
        weight = self._rand(K, C, dtype=torch.float16)
        bias = self._rand(C, dtype=torch.float16)
        if variant == "silu":
            op, extra = oasr.depthwise_conv1d_silu, {}
        else:
            op = oasr.depthwise_conv1d
            extra = {}
            if variant == "fsmn":  # Paraformer's masked residual contract
                mask = torch.ones(B, T, 1, device="cuda", dtype=torch.bool)
                mask[:, -1000:] = False
                extra = {"mask": mask, "add_input": True}
        got = op(x, weight, bias, K // 2, **extra)
        want = torch.empty_like(got)
        for i in range(B):
            row = {k: (v[i : i + 1] if k == "mask" else v) for k, v in extra.items()}
            op(x[i : i + 1], weight, bias, K // 2, out=want[i : i + 1], **row)
        assert torch.equal(got, want)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestEveryCompiledConv2dVariantIsCorrect:
    """The conv2d half of the sweep in ``test_gemm.py``.

    Implicit GEMM instantiates the same ``DefaultThreadMapTensorOp`` epilogue
    from the same shared tile list (``oasr.jit.gemm.TileShapeConfigs``), so a
    tile CUTLASS cannot address produces a wrong conv2d kernel as readily as a
    wrong GEMM one — ``b128x16x64_w32x16x64`` measured 1.2x relative error here.
    Conv2D never had a rule naming it, which is exactly why the sweep is over
    the compiled set rather than over the rules.
    """

    @pytest.mark.parametrize(
        "name,cfg",
        sorted(get_unique_conv2d_compile_configs(_get_target_sm()).items()),
        ids=sorted(get_unique_conv2d_compile_configs(_get_target_sm())),
    )
    def test_matches_an_fp32_reference(self, name, cfg):
        # IC >= _CUDNN_IC_THRESHOLD and a 3x3 filter, so neither the cuDNN
        # branch nor the 1x1 GEMM specialisation takes the call away from the
        # CUTLASS variant under test.
        N, H, W, IC, OC, R, S = 1, 64, 32, 64, 72, 3, 3
        torch.manual_seed(0)
        x = torch.randn(N, H, W, IC, device="cuda", dtype=torch.float16) * 0.5
        w = torch.randn(OC, R, S, IC, device="cuda", dtype=torch.float16) * 0.05
        b = torch.randn(OC, device="cuda", dtype=torch.float16) * 0.05
        # NaN rather than empty: a partially written tile otherwise returns
        # whatever the allocator last held, which can pass a loose tolerance.
        out = torch.full((N, H, W, OC), float("nan"), device="cuda", dtype=torch.float16)

        fn = getattr(_get_conv2d_module(), conv2d_func_name(cfg))
        fn(out, x, w, b, 1, 1, 1, 1, 1, 1)

        ref = F.conv2d(
            x.permute(0, 3, 1, 2).float(),
            w.permute(0, 3, 1, 2).float(),
            b.float(),
            padding=1,
        ).permute(0, 2, 3, 1)
        assert torch.isfinite(out).all(), f"{name}: left NaN behind"
        rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
        assert rel < 2e-2, f"{name}: relative error {rel:.4g}"
