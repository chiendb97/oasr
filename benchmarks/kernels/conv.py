# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Convolution -- ``oasr/functionals/conv.py``.

Ten subroutines.  Eight came from the routine module; ``dense_conv1d`` and
``fsmn_chain`` came from the two standalone scripts that were the only way to
reach them, so neither was in ``--list`` and neither could write a CSV row.

``pointwise_conv1d`` is ``oasr.gemm`` -- a 1x1 convolution is a GEMM, and the
kernel underneath is the same one ``--family gemm`` measures.  It stays here
because this is where a reader looks for it, but its 139-line private autotune
path is gone: autotuning is a context manager around any sweep, in
``benchmarks/core/driver.py``.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import (
    conv1d_flops,
    conv2d_flops,
    conv2d_output_hw,
    dtype_size,
    gemm_flops,
)

SUBROUTINES = [
    "depthwise_conv1d",
    "depthwise_conv1d_causal",
    "dense_conv1d",
    "pointwise_conv1d",
    "pointwise_conv1d_activation",
    "conv2d",
    "conv2d_activation",
    "grouped_conv2d",
    "pointwise_conv2d",
    "fsmn_chain",
]

_CONV2D_FAMILY = ("conv2d", "conv2d_activation", "grouped_conv2d", "pointwise_conv2d")

_DW = [
    {"batch": 32, "seq": 250, "channels": 256, "kernel_size": 15},
    {"batch": 64, "seq": 250, "channels": 256, "kernel_size": 15},
    {"batch": 64, "seq": 250, "channels": 512, "kernel_size": 31},
    {"batch": 64, "seq": 500, "channels": 256, "kernel_size": 15},
    {"batch": 64, "seq": 500, "channels": 512, "kernel_size": 31},
    {"batch": 32, "seq": 125, "channels": 256, "kernel_size": 15},
    {"batch": 32, "seq": 125, "channels": 256, "kernel_size": 31},
    {"batch": 16, "seq": 125, "channels": 256, "kernel_size": 15},
    {"batch": 64, "seq": 125, "channels": 512, "kernel_size": 31},
]
_PW1D = [
    {"batch": 32, "seq": 250, "channels": 256, "out_channels": 512},
    {"batch": 64, "seq": 250, "channels": 256, "out_channels": 512},
    {"batch": 64, "seq": 250, "channels": 512, "out_channels": 1024},
    {"batch": 64, "seq": 250, "channels": 256, "out_channels": 2048},
    {"batch": 64, "seq": 250, "channels": 512, "out_channels": 2048},
    {"batch": 64, "seq": 500, "channels": 256, "out_channels": 512},
    {"batch": 64, "seq": 500, "channels": 512, "out_channels": 1024},
]
_C2D = [
    {"N": 16, "H": 200, "W": 80, "IC": 1, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 16, "H": 100, "W": 40, "IC": 64, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 32, "H": 200, "W": 80, "IC": 1, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 32, "H": 100, "W": 40, "IC": 64, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 8, "H": 300, "W": 80, "IC": 1, "K": 256, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 8, "H": 150, "W": 40, "IC": 256, "K": 256, "R": 3, "S": 3, "pad": 0, "stride": 2},
    {"N": 16, "H": 100, "W": 40, "IC": 128, "K": 128, "R": 3, "S": 3, "pad": 1, "stride": 1},
    {"N": 16, "H": 100, "W": 40, "IC": 256, "K": 256, "R": 3, "S": 3, "pad": 1, "stride": 1},
]

DEFAULT_CONFIGS: Dict[str, list] = {
    # Same shapes for both: the causal variant is the same kernel reached with
    # zero padding and a pre-padded input, so comparing them at different
    # shapes would compare the shapes instead.
    "depthwise_conv1d": _DW,
    "depthwise_conv1d_causal": _DW,
    "pointwise_conv1d": _PW1D,
    "pointwise_conv1d_activation": _PW1D,
    "conv2d": _C2D,
    "conv2d_activation": _C2D[:4] + _C2D[5:7],
    "grouped_conv2d": [
        {
            "N": n,
            "H": 50,
            "W": 19,
            "IC": 128,
            "K": 128,
            "R": 7,
            "S": 7,
            "pad": 3,
            "stride": 1,
            "groups": 128,
        }
        for n in (1, 8)
    ]
    + [
        {
            "N": n,
            "H": 100,
            "W": 40,
            "IC": 256,
            "K": 256,
            "R": 3,
            "S": 3,
            "pad": 0,
            "stride": 2,
            "groups": 256,
        }
        for n in (1, 8)
    ],
    "pointwise_conv2d": [
        {"N": n, "H": 50, "W": 19, "IC": ic, "K": k, "R": 1, "S": 1, "pad": 0, "stride": 1}
        for n in (1, 8)
        for ic, k in ((128, 384), (384, 128))
    ],
    # Whisper and Qwen2-Audio front-ends, plus the Paraformer CIF predictor
    # (which pads one frame each side before the call, hence padding 0).
    "dense_conv1d": [
        {
            "batch": 1,
            "seq": 3000,
            "channels": 80,
            "out_channels": 384,
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
        },
        {
            "batch": 1,
            "seq": 3000,
            "channels": 384,
            "out_channels": 384,
            "kernel_size": 3,
            "padding": 1,
            "stride": 2,
        },
        {
            "batch": 1,
            "seq": 3000,
            "channels": 128,
            "out_channels": 1280,
            "kernel_size": 3,
            "padding": 1,
            "stride": 1,
        },
        {
            "batch": 1,
            "seq": 3000,
            "channels": 1280,
            "out_channels": 1280,
            "kernel_size": 3,
            "padding": 1,
            "stride": 2,
        },
        {
            "batch": 1,
            "seq": 502,
            "channels": 512,
            "out_channels": 512,
            "kernel_size": 3,
            "padding": 0,
            "stride": 1,
        },
    ],
    # Paraformer SANM: LFR-short and LFR-long, plus shifted windows whose
    # asymmetric padding is the case the fused mask path exists for.
    "fsmn_chain": [
        {
            "batch": b,
            "seq": s,
            "channels": 512,
            "kernel_size": 11,
            "padding_left": 5,
            "padding_right": 5,
        }
        for b, s in ((1, 80), (1, 500), (8, 80), (8, 500))
    ]
    + [
        {
            "batch": b,
            "seq": 500,
            "channels": 512,
            "kernel_size": 11,
            "padding_left": 7,
            "padding_right": 3,
        }
        for b in (1, 8)
    ],
}

#: The dense and FSMN sweeps compare arms that differ by a layout or a fusion,
#: which is exactly where a warm allocator can manufacture a gap.
INTERLEAVE_SUBROUTINES = frozenset({"dense_conv1d", "fsmn_chain"})


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length (1D)")
    parser.add_argument("--channels", type=int, default=None, help="Input channels")
    parser.add_argument("--out-channels", type=int, default=None, help="Output channels (1D)")
    parser.add_argument("--kernel-size", type=int, default=None, help="Kernel size (1D)")
    parser.add_argument("--padding", type=int, default=None, help="Padding (dense_conv1d)")
    parser.add_argument("--height", type=int, default=None, help="Input height (2D)")
    parser.add_argument("--width", type=int, default=None, help="Input width (2D)")
    parser.add_argument("--out-filters", type=int, default=None, help="Output filters (2D)")
    parser.add_argument("--groups", type=int, default=1, help="Channel groups (2D)")
    parser.add_argument("--filter-h", type=int, default=3, help="Filter height (2D)")
    parser.add_argument("--filter-w", type=int, default=3, help="Filter width (2D)")
    parser.add_argument("--pad", type=int, default=0, help="Padding (2D)")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--activation", default=None, help="Activation (swish, relu, gelu)")
    parser.add_argument(
        "--mask-dtype",
        choices=["bool", "dtype"],
        default="bool",
        help="Mask element type for fsmn_chain",
    )


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    if subroutine in _CONV2D_FAMILY:
        dims = (args.batch, args.height, args.width, args.channels, args.out_filters)
        if all(v is not None for v in dims):
            return [
                {
                    "N": args.batch,
                    "H": args.height,
                    "W": args.width,
                    "IC": args.channels,
                    "K": args.out_filters,
                    "R": args.filter_h,
                    "S": args.filter_w,
                    "pad": args.pad,
                    "stride": args.stride,
                    "groups": args.groups,
                }
            ]
        return DEFAULT_CONFIGS[subroutine]

    if subroutine == "dense_conv1d":
        dims = (args.batch, args.seq, args.channels, args.out_channels, args.kernel_size)
        if all(v is not None for v in dims):
            return [
                {
                    "batch": args.batch,
                    "seq": args.seq,
                    "channels": args.channels,
                    "out_channels": args.out_channels,
                    "kernel_size": args.kernel_size,
                    "padding": args.padding if args.padding is not None else 0,
                    "stride": args.stride,
                }
            ]
        return DEFAULT_CONFIGS[subroutine]

    if subroutine == "fsmn_chain":
        dims = (args.batch, args.seq, args.channels, args.kernel_size)
        if all(v is not None for v in dims):
            half = args.kernel_size // 2
            return [
                {
                    "batch": args.batch,
                    "seq": args.seq,
                    "channels": args.channels,
                    "kernel_size": args.kernel_size,
                    "padding_left": half,
                    "padding_right": half,
                }
            ]
        return DEFAULT_CONFIGS[subroutine]

    if subroutine in ("pointwise_conv1d", "pointwise_conv1d_activation"):
        dims = (args.batch, args.seq, args.channels, args.out_channels)
        if all(v is not None for v in dims):
            return [
                {
                    "batch": args.batch,
                    "seq": args.seq,
                    "channels": args.channels,
                    "out_channels": args.out_channels,
                }
            ]
        return DEFAULT_CONFIGS[subroutine]

    dims = (args.batch, args.seq, args.channels, args.kernel_size)
    if all(v is not None for v in dims):
        return [
            {
                "batch": args.batch,
                "seq": args.seq,
                "channels": args.channels,
                "kernel_size": args.kernel_size,
            }
        ]
    return DEFAULT_CONFIGS[subroutine]


def _activation_id(name):
    if name is None:
        return oasr.ACTIVATION_SWISH
    return oasr.get_activation_type_id(name)


def _torch_activation(act_id):
    if act_id == oasr.ACTIVATION_RELU:
        return F.relu
    if act_id == oasr.ACTIVATION_GELU:
        # OASR's fused GELU epilogue is the tanh approximation; the exact erf
        # form is ACTIVATION_GELU_ERF.  One oracle for both would hide a real
        # accuracy difference.
        return lambda t: F.gelu(t, approximate="tanh")
    if act_id == oasr.ACTIVATION_GELU_ERF:
        return F.gelu
    return F.silu


def _depthwise_fns(cfg, dtype, causal):
    b, s, c, k = cfg["batch"], cfg["seq"], cfg["channels"], cfg["kernel_size"]
    padding = 0 if causal else k // 2
    seq_in = s + k - 1 if causal else s
    x = torch.randn(b, seq_in, c, device="cuda", dtype=dtype)
    weight = torch.randn(k, c, device="cuda", dtype=dtype)
    bias = torch.randn(c, device="cuda", dtype=dtype)
    x_nchw = x.permute(0, 2, 1).contiguous()
    # (kernel_size, channels) -> (channels, 1, kernel_size).  A bare `view`
    # reinterprets the bytes instead of transposing, and the reference then
    # convolves with a scrambled filter.  tests/kernels/test_conv.py states it.
    weight_pt = weight.permute(1, 0).reshape(c, 1, k)
    return {
        "cuda": lambda: oasr.depthwise_conv1d(x, weight, bias, padding),
        # Transposed back to (B, T, C): oasr.depthwise_conv1d is channels-last,
        # so a raw (B, C, T) reference makes --refcheck crash on the shape
        # instead of reporting a mismatch.
        "torch": lambda: F.conv1d(x_nchw, weight_pt, bias, padding=padding, groups=c).transpose(
            1, 2
        ),
    }


def _dense_conv1d_fns(cfg, dtype):
    """The dense BTC Conv1D lane: CUTLASS, the production dispatcher, and cuDNN.

    ``production`` is what the model actually calls, so the gap between it and
    ``cutlass_default`` is the dispatch rule rather than the kernel.
    """
    from oasr.functionals.conv import (
        _default_conv1d_fn,
        _dispatch_conv1d,
        _get_cudnn_conv2d_module,
    )

    b, s = cfg["batch"], cfg["seq"]
    ic, oc, k = cfg["channels"], cfg["out_channels"], cfg["kernel_size"]
    pad, stride = cfg["padding"], cfg["stride"]
    out_seq = (s + 2 * pad - k) // stride + 1
    x = torch.randn(b, s, ic, device="cuda", dtype=dtype)
    weight = torch.randn(oc, k, ic, device="cuda", dtype=dtype)
    bias = torch.randn(oc, device="cuda", dtype=dtype)
    outs = {
        name: torch.empty(b, out_seq, oc, device="cuda", dtype=dtype)
        for name in ("cutlass_default", "production", "cudnn")
    }
    cutlass = _default_conv1d_fn()
    cudnn = _get_cudnn_conv2d_module()

    return {
        "cutlass_default": lambda: cutlass(
            outs["cutlass_default"], x, weight, bias, pad, stride, 1
        ),
        "production": lambda: _dispatch_conv1d(outs["production"], x, weight, bias, pad, stride, 1),
        "cudnn": lambda: cudnn.cudnn_conv1d(outs["cudnn"], x, weight, bias, pad, stride, 1),
        "torch": lambda: F.conv1d(
            x.transpose(1, 2), weight.permute(0, 2, 1), bias, padding=pad, stride=stride
        ).transpose(1, 2),
    }


def _fsmn_fns(cfg, dtype, mask_dtype):
    """Paraformer's SANM memory block, fused against the sequence it replaced.

    ``previous_oasr`` is the cast + mul + pad + kernel + add + mul chain that
    was in Paraformer's modules.py, so the fused arm is measured against the
    thing it removed rather than against PyTorch alone.
    """
    b, s, c, k = cfg["batch"], cfg["seq"], cfg["channels"], cfg["kernel_size"]
    pad = (cfg["padding_left"], cfg["padding_right"])
    x = torch.randn(b, s, c, device="cuda", dtype=dtype)
    weight = torch.randn(k, c, device="cuda", dtype=dtype)
    weight_pt = weight.T.unsqueeze(1).contiguous()
    lengths = torch.randint(max(1, s // 2), s + 1, (b,), device="cuda", dtype=torch.int64)
    bool_mask = (torch.arange(s, device="cuda").unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
    mask = bool_mask if mask_dtype == "bool" else bool_mask.to(dtype)
    dense_mask = bool_mask.to(dtype)
    kernel_out = torch.empty_like(x)

    def previous():
        masked = x * dense_mask
        conv = oasr.depthwise_conv1d(F.pad(masked, (0, 0, pad[0], pad[1])), weight)
        return (conv + masked) * dense_mask

    def torch_e2e():
        masked = x * dense_mask
        conv = F.conv1d(F.pad(masked.transpose(1, 2), pad), weight_pt, groups=c).transpose(1, 2)
        return (conv + masked) * dense_mask

    return {
        # Destination-passing, so the allocation is out of the measurement.
        "cuda_kernel": lambda: oasr.depthwise_conv1d(
            x, weight, padding=pad, out=kernel_out, mask=mask, add_input=True
        ),
        "cuda": lambda: oasr.depthwise_conv1d(x, weight, padding=pad, mask=mask, add_input=True),
        "oasr_previous": previous,
        "torch": torch_e2e,
    }


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if subroutine == "depthwise_conv1d":
        return _depthwise_fns(cfg, dtype, causal=False)
    if subroutine == "depthwise_conv1d_causal":
        return _depthwise_fns(cfg, dtype, causal=True)
    if subroutine == "dense_conv1d":
        return _dense_conv1d_fns(cfg, dtype)
    if subroutine == "fsmn_chain":
        return _fsmn_fns(cfg, dtype, getattr(args, "mask_dtype", "bool"))

    if subroutine in ("pointwise_conv1d", "pointwise_conv1d_activation"):
        b, s = cfg["batch"], cfg["seq"]
        ic, oc = cfg["channels"], cfg["out_channels"]
        x = torch.randn(b, s, ic, device="cuda", dtype=dtype)
        weight = torch.randn(oc, ic, device="cuda", dtype=dtype)
        bias = torch.randn(oc, device="cuda", dtype=dtype)
        if subroutine == "pointwise_conv1d":
            return {
                "cutlass": lambda: oasr.gemm(x, weight, bias),
                "torch": lambda: F.linear(x, weight, bias),
            }
        act = _activation_id(getattr(args, "activation", None))
        torch_act = _torch_activation(act)
        return {
            "cutlass": lambda: oasr.gemm_activation(x, weight, bias, act),
            "torch": lambda: torch_act(F.linear(x, weight, bias)),
        }

    # 2D family
    n, h, w = cfg["N"], cfg["H"], cfg["W"]
    ic, k, r, s_ = cfg["IC"], cfg["K"], cfg["R"], cfg["S"]
    pad, stride = cfg["pad"], cfg["stride"]
    groups = cfg.get("groups", 1)
    x_nhwc = torch.randn(n, h, w, ic, device="cuda", dtype=dtype)
    w_krsc = torch.randn(k, r, s_, ic // groups, device="cuda", dtype=dtype)
    bias = torch.randn(k, device="cuda", dtype=dtype)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_kcrs = w_krsc.permute(0, 3, 1, 2).contiguous()

    if subroutine == "conv2d_activation":
        act = _activation_id(getattr(args, "activation", None))
        torch_act = _torch_activation(act)
        return {
            "cutlass": lambda: oasr.conv2d_activation(
                x_nhwc, w_krsc, bias, act, pad, pad, stride, stride
            ),
            "torch": lambda: torch_act(
                F.conv2d(x_nchw, w_kcrs, bias, stride=stride, padding=pad)
            ).permute(0, 2, 3, 1),
        }
    return {
        # Permuted back to NHWC: oasr.conv2d is channels-last, so a raw NCHW
        # reference makes --refcheck compare mismatched layouts.
        "cutlass": lambda: oasr.conv2d(
            x_nhwc, w_krsc, bias, pad, pad, stride, stride, groups=groups
        ),
        "torch": lambda: F.conv2d(
            x_nchw, w_kcrs, bias, padding=pad, stride=stride, groups=groups
        ).permute(0, 2, 3, 1),
    }


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    elem = dtype_size(dtype)

    if subroutine in ("depthwise_conv1d", "depthwise_conv1d_causal", "fsmn_chain"):
        b, s, c, k = cfg["batch"], cfg["seq"], cfg["channels"], cfg["kernel_size"]
        # Both flops and bytes: the arithmetic is real but tiny, and reporting
        # both is what shows *why* this is memory-bound.  The previous harness
        # wrote tflops=0.0 here, which reads as "measured zero".
        work = Work(
            shape=f"[{b}, {s}, {c}] k={k}",
            params=params_of(cfg),
            flops=conv1d_flops(b, s, c, c, k, groups=c),
            bytes=2 * b * s * c * elem + k * c * elem,
        )
        if subroutine == "fsmn_chain":
            work.shape += f" pad={cfg['padding_left']},{cfg['padding_right']}"
        return work

    if subroutine == "dense_conv1d":
        b, s = cfg["batch"], cfg["seq"]
        ic, oc, k = cfg["channels"], cfg["out_channels"], cfg["kernel_size"]
        out_seq = (s + 2 * cfg["padding"] - k) // cfg["stride"] + 1
        return Work(
            shape=f"[{b}, {s}, {ic}] -> {oc} k={k} p={cfg['padding']} s={cfg['stride']}",
            params=params_of(cfg),
            flops=conv1d_flops(b, out_seq, oc, ic, k),
            bytes=(b * s * ic + b * out_seq * oc + oc * k * ic) * elem,
        )

    if subroutine in ("pointwise_conv1d", "pointwise_conv1d_activation"):
        b, s = cfg["batch"], cfg["seq"]
        ic, oc = cfg["channels"], cfg["out_channels"]
        return Work(
            shape=f"[{b}, {s}, {ic}] -> {oc}",
            params=params_of(cfg),
            flops=gemm_flops(b * s, oc, ic),
            bytes=(b * s * ic + b * s * oc + oc * ic) * elem,
        )

    n, h, w = cfg["N"], cfg["H"], cfg["W"]
    ic, k, r, s_ = cfg["IC"], cfg["K"], cfg["R"], cfg["S"]
    pad, stride = cfg["pad"], cfg["stride"]
    groups = cfg.get("groups", 1)
    oh, ow = conv2d_output_hw(h, w, r, s_, pad, stride)
    suffix = f" g={groups}" if groups != 1 else ""
    return Work(
        shape=f"[{n},{h},{w},{ic}->{k}] {r}x{s_} p={pad} s={stride}{suffix}",
        params=params_of(cfg),
        flops=conv2d_flops(n, h, w, ic, k, r, s_, pad, stride, groups),
        bytes=(n * h * w * ic + n * oh * ow * k + k * r * s_ * (ic // groups)) * elem,
    )
