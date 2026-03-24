"""Convolution family benchmark routines.

Covers: depthwise_conv1d, depthwise_conv1d_causal, pointwise_conv1d,
        pointwise_conv1d_activation, conv2d, conv2d_activation.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import (
    BenchResult,
    OutputWriter,
    bench_fn,
    check_close,
    compute_bandwidth_tb_s,
    compute_gemm_tflops,
    dtype_size,
    make_bench_parser,
    run_main,
    run_profile,
)

SUBROUTINES = [
    "depthwise_conv1d",
    "depthwise_conv1d_causal",
    "pointwise_conv1d",
    "pointwise_conv1d_activation",
    "conv2d",
    "conv2d_activation",
]

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "depthwise_conv1d": [
        {"batch": 32, "seq": 250, "channels": 256, "kernel_size": 15},
        {"batch": 64, "seq": 250, "channels": 256, "kernel_size": 15},
        {"batch": 64, "seq": 250, "channels": 512, "kernel_size": 31},
        {"batch": 64, "seq": 500, "channels": 256, "kernel_size": 15},
        {"batch": 64, "seq": 500, "channels": 512, "kernel_size": 31},
        {"batch": 32, "seq": 125, "channels": 256, "kernel_size": 15},
        {"batch": 32, "seq": 125, "channels": 256, "kernel_size": 31},
        {"batch": 16, "seq": 125, "channels": 256, "kernel_size": 15},
        {"batch": 64, "seq": 125, "channels": 512, "kernel_size": 31},
    ],
    "depthwise_conv1d_causal": [
        {"batch": 32, "seq": 250, "channels": 256, "kernel_size": 15},
        {"batch": 64, "seq": 250, "channels": 256, "kernel_size": 15},
        {"batch": 64, "seq": 250, "channels": 512, "kernel_size": 31},
        {"batch": 64, "seq": 500, "channels": 256, "kernel_size": 15},
        {"batch": 64, "seq": 500, "channels": 512, "kernel_size": 31},
        {"batch": 32, "seq": 125, "channels": 256, "kernel_size": 15},
        {"batch": 32, "seq": 125, "channels": 256, "kernel_size": 31},
        {"batch": 16, "seq": 125, "channels": 256, "kernel_size": 15},
        {"batch": 64, "seq": 125, "channels": 512, "kernel_size": 31},
    ],
    "pointwise_conv1d": [
        {"batch": 32, "seq": 250, "channels": 256, "out_channels": 512},
        {"batch": 64, "seq": 250, "channels": 256, "out_channels": 512},
        {"batch": 64, "seq": 250, "channels": 512, "out_channels": 1024},
        {"batch": 64, "seq": 250, "channels": 256, "out_channels": 2048},
        {"batch": 64, "seq": 250, "channels": 512, "out_channels": 2048},
        {"batch": 64, "seq": 500, "channels": 256, "out_channels": 512},
        {"batch": 64, "seq": 500, "channels": 512, "out_channels": 1024},
    ],
    "pointwise_conv1d_activation": [
        {"batch": 32, "seq": 250, "channels": 256, "out_channels": 512},
        {"batch": 64, "seq": 250, "channels": 256, "out_channels": 512},
        {"batch": 64, "seq": 250, "channels": 512, "out_channels": 1024},
        {"batch": 64, "seq": 250, "channels": 256, "out_channels": 2048},
        {"batch": 64, "seq": 250, "channels": 512, "out_channels": 2048},
        {"batch": 64, "seq": 500, "channels": 256, "out_channels": 512},
        {"batch": 64, "seq": 500, "channels": 512, "out_channels": 1024},
    ],
    "conv2d": [
        {"N": 16, "H": 200, "W": 80, "IC": 1, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 16, "H": 100, "W": 40, "IC": 64, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 32, "H": 200, "W": 80, "IC": 1, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 32, "H": 100, "W": 40, "IC": 64, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 8, "H": 300, "W": 80, "IC": 1, "K": 256, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 8, "H": 150, "W": 40, "IC": 256, "K": 256, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 16, "H": 100, "W": 40, "IC": 128, "K": 128, "R": 3, "S": 3, "pad": 1, "stride": 1},
        {"N": 16, "H": 100, "W": 40, "IC": 256, "K": 256, "R": 3, "S": 3, "pad": 1, "stride": 1},
    ],
    "conv2d_activation": [
        {"N": 16, "H": 200, "W": 80, "IC": 1, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 16, "H": 100, "W": 40, "IC": 64, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 32, "H": 200, "W": 80, "IC": 1, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 32, "H": 100, "W": 40, "IC": 64, "K": 64, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 8, "H": 150, "W": 40, "IC": 256, "K": 256, "R": 3, "S": 3, "pad": 0, "stride": 2},
        {"N": 16, "H": 100, "W": 40, "IC": 256, "K": 256, "R": 3, "S": 3, "pad": 1, "stride": 1},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "depthwise_conv1d": (64, 250, 512, 31),
    "depthwise_conv1d_causal": (64, 250, 512, 31),
    "pointwise_conv1d": (64, 250, 512, 1024),
    "conv2d": (16, 100, 40, 256, 256, 3, 3, 1, 1),
    "conv2d_activation": (16, 100, 40, 256, 256, 3, 3, 1, 1),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------


def setup_depthwise_conv1d(batch_size, seq_len, channels, kernel_size, dtype=torch.float16):
    padding = kernel_size // 2
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
    weight = torch.randn(kernel_size, channels, device="cuda", dtype=dtype)
    bias = torch.randn(channels, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.depthwise_conv1d(x, weight, bias, padding)

    x_nchw = x.permute(0, 2, 1).contiguous()
    weight_pt = weight.view(channels, 1, kernel_size)

    def pytorch_fn():
        return F.conv1d(x_nchw, weight_pt, bias, padding=padding, groups=channels)

    return oasr_fn, pytorch_fn


def setup_depthwise_conv1d_causal(batch_size, seq_len, channels, kernel_size, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len + kernel_size - 1, channels, device="cuda", dtype=dtype)
    weight = torch.randn(kernel_size, channels, device="cuda", dtype=dtype)
    bias = torch.randn(channels, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.depthwise_conv1d(x, weight, bias, 0)

    x_nchw = x.permute(0, 2, 1).contiguous()
    weight_pt = weight.view(channels, 1, kernel_size)

    def pytorch_fn():
        return F.conv1d(x_nchw, weight_pt, bias, padding=0, groups=channels)

    return oasr_fn, pytorch_fn


def setup_pointwise_conv1d(batch_size, seq_len, in_ch, out_ch, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len, in_ch, device="cuda", dtype=dtype)
    weight = torch.randn(out_ch, in_ch, device="cuda", dtype=dtype)
    bias = torch.randn(out_ch, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.pointwise_conv1d(x, weight, bias)

    def pytorch_fn():
        return F.linear(x, weight, bias)

    return oasr_fn, pytorch_fn


def setup_pointwise_conv1d_activation(
    batch_size, seq_len, in_ch, out_ch, activation=None, dtype=torch.float16
):
    if activation is None:
        activation = oasr.ACTIVATION_SWISH
    x = torch.randn(batch_size, seq_len, in_ch, device="cuda", dtype=dtype)
    weight = torch.randn(out_ch, in_ch, device="cuda", dtype=dtype)
    bias = torch.randn(out_ch, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.pointwise_conv1d_activation(x, weight, bias, activation)

    def pytorch_fn():
        out = F.linear(x, weight, bias)
        if activation == oasr.ACTIVATION_SWISH:
            return F.silu(out)
        elif activation == oasr.ACTIVATION_RELU:
            return F.relu(out)
        elif activation == oasr.ACTIVATION_GELU:
            return F.gelu(out)
        return out

    return oasr_fn, pytorch_fn


def setup_conv2d(batch_size, H, W, IC, K, R, S, pad=0, stride=1, dtype=torch.float16):
    x_nhwc = torch.randn(batch_size, H, W, IC, device="cuda", dtype=dtype)
    w_nhwc = torch.randn(K, R, S, IC, device="cuda", dtype=dtype)
    bias = torch.randn(K, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.conv2d(x_nhwc, w_nhwc, bias, pad, pad, stride, stride)

    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()

    def pytorch_fn():
        return F.conv2d(x_nchw, w_nchw, bias, stride=stride, padding=pad)

    return oasr_fn, pytorch_fn


def setup_conv2d_activation(
    batch_size, H, W, IC, K, R, S, pad=0, stride=1, activation=None, dtype=torch.float16
):
    if activation is None:
        activation = oasr.ACTIVATION_SWISH
    x_nhwc = torch.randn(batch_size, H, W, IC, device="cuda", dtype=dtype)
    w_nhwc = torch.randn(K, R, S, IC, device="cuda", dtype=dtype)
    bias = torch.randn(K, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.conv2d_activation(x_nhwc, w_nhwc, bias, activation, pad, pad, stride, stride)

    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    w_nchw = w_nhwc.permute(0, 3, 1, 2).contiguous()

    if activation == oasr.ACTIVATION_RELU:
        act_fn = F.relu
    elif activation == oasr.ACTIVATION_GELU:
        act_fn = F.gelu
    else:
        act_fn = F.silu

    def pytorch_fn():
        return act_fn(F.conv2d(x_nchw, w_nchw, bias, stride=stride, padding=pad))

    return oasr_fn, pytorch_fn


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length (1D conv)")
    parser.add_argument("--channels", type=int, default=None, help="Input channels")
    parser.add_argument("--out-channels", type=int, default=None, help="Output channels (pointwise)")
    parser.add_argument("--kernel-size", type=int, default=None, help="Kernel size (1D conv)")
    parser.add_argument("--height", type=int, default=None, help="Input height (2D conv)")
    parser.add_argument("--width", type=int, default=None, help="Input width (2D conv)")
    parser.add_argument("--out-filters", type=int, default=None, help="Output filters (2D conv)")
    parser.add_argument("--filter-h", type=int, default=3, help="Filter height (2D conv)")
    parser.add_argument("--filter-w", type=int, default=3, help="Filter width (2D conv)")
    parser.add_argument("--pad", type=int, default=0, help="Padding")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--activation", type=str, default=None, help="Activation (swish, relu, gelu)")


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "depthwise_conv1d")
    dtype_str = getattr(args, "dtype", "float16")
    from benchmarks.routines.bench_utils import parse_dtype

    dtype = parse_dtype(dtype_str)
    backends = getattr(args, "backends", ["oasr", "pytorch"])
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    configs = _resolve_configs(args, subroutine)

    for cfg in configs:
        oasr_fn, pytorch_fn = _setup_for_config(subroutine, cfg, dtype)
        fn_map = {"oasr": oasr_fn, "pytorch": pytorch_fn}

        shape_str = _shape_str(subroutine, cfg)

        if do_check and "oasr" in backends and "pytorch" in backends:
            oasr_out = oasr_fn()
            pytorch_out = pytorch_fn()
            passed, max_diff = check_close(oasr_out, pytorch_out)
            if not passed:
                print(f"  [ERROR] Output mismatch for {shape_str} (max_diff={max_diff:.6f})")
                if not allow_mismatch:
                    continue

        for backend in backends:
            if backend not in fn_map:
                print(f"  [WARNING] Unknown backend '{backend}', skipping")
                continue
            median_ms, std_ms = bench_fn(
                fn_map[backend],
                dry_run_iters=dry_run_iters,
                num_iters=num_iters,
                use_cuda_events=use_cuda_events,
            )
            tflops = _compute_tflops(subroutine, cfg, median_ms)
            output.write_result(BenchResult(
                routine="conv",
                subroutine=subroutine,
                backend=backend,
                shape=shape_str,
                dtype=dtype_str,
                median_ms=median_ms,
                std_ms=std_ms,
                tflops=tflops,
            ))


def _resolve_configs(args, subroutine):
    if subroutine in ("conv2d", "conv2d_activation"):
        return _resolve_conv2d_configs(args, subroutine)
    return _resolve_conv1d_configs(args, subroutine)


def _resolve_conv1d_configs(args, subroutine):
    batch = getattr(args, "batch", None)
    seq = getattr(args, "seq", None)
    channels = getattr(args, "channels", None)
    kernel_size = getattr(args, "kernel_size", None)
    out_channels = getattr(args, "out_channels", None)

    if subroutine in ("pointwise_conv1d", "pointwise_conv1d_activation"):
        if all(v is not None for v in [batch, seq, channels, out_channels]):
            return [{"batch": batch, "seq": seq, "channels": channels, "out_channels": out_channels}]
    elif all(v is not None for v in [batch, seq, channels, kernel_size]):
        return [{"batch": batch, "seq": seq, "channels": channels, "kernel_size": kernel_size}]

    return DEFAULT_CONFIGS.get(subroutine, [])


def _resolve_conv2d_configs(args, subroutine):
    N = getattr(args, "batch", None)
    H = getattr(args, "height", None)
    W = getattr(args, "width", None)
    IC = getattr(args, "channels", None)
    K = getattr(args, "out_filters", None)
    R = getattr(args, "filter_h", 3)
    S = getattr(args, "filter_w", 3)
    pad = getattr(args, "pad", 0)
    stride = getattr(args, "stride", 1)

    if all(v is not None for v in [N, H, W, IC, K]):
        return [{"N": N, "H": H, "W": W, "IC": IC, "K": K, "R": R, "S": S, "pad": pad, "stride": stride}]

    return DEFAULT_CONFIGS.get(subroutine, [])


def _setup_for_config(subroutine, cfg, dtype):
    if subroutine == "depthwise_conv1d":
        return setup_depthwise_conv1d(
            cfg["batch"], cfg["seq"], cfg["channels"], cfg["kernel_size"], dtype
        )
    elif subroutine == "depthwise_conv1d_causal":
        return setup_depthwise_conv1d_causal(
            cfg["batch"], cfg["seq"], cfg["channels"], cfg["kernel_size"], dtype
        )
    elif subroutine == "pointwise_conv1d":
        return setup_pointwise_conv1d(
            cfg["batch"], cfg["seq"], cfg["channels"], cfg["out_channels"], dtype
        )
    elif subroutine == "pointwise_conv1d_activation":
        return setup_pointwise_conv1d_activation(
            cfg["batch"], cfg["seq"], cfg["channels"], cfg["out_channels"], dtype=dtype
        )
    elif subroutine == "conv2d":
        return setup_conv2d(
            cfg["N"], cfg["H"], cfg["W"], cfg["IC"], cfg["K"],
            cfg["R"], cfg["S"], cfg["pad"], cfg["stride"], dtype,
        )
    elif subroutine == "conv2d_activation":
        return setup_conv2d_activation(
            cfg["N"], cfg["H"], cfg["W"], cfg["IC"], cfg["K"],
            cfg["R"], cfg["S"], cfg["pad"], cfg["stride"], dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown conv subroutine: {subroutine}")


def _compute_tflops(subroutine, cfg, time_ms):
    if subroutine in ("conv2d", "conv2d_activation"):
        N, H, W = cfg["N"], cfg["H"], cfg["W"]
        IC, K, R, S = cfg["IC"], cfg["K"], cfg["R"], cfg["S"]
        stride = cfg.get("stride", 1)
        pad = cfg.get("pad", 0)
        OH = (H + 2 * pad - R) // stride + 1
        OW = (W + 2 * pad - S) // stride + 1
        flops = 2 * N * OH * OW * K * IC * R * S
        return flops / (time_ms * 1e-3) / 1e12 if time_ms > 0 else 0.0
    elif subroutine in ("pointwise_conv1d", "pointwise_conv1d_activation"):
        b, s = cfg["batch"], cfg["seq"]
        ic, oc = cfg["channels"], cfg["out_channels"]
        return compute_gemm_tflops(b * s, oc, ic, time_ms)
    else:
        # Depthwise: memory-bound, no meaningful TFLOPS
        return 0.0


def _shape_str(subroutine, cfg):
    if subroutine in ("conv2d", "conv2d_activation"):
        c = cfg
        return f"[{c['N']},{c['H']},{c['W']},{c['IC']}->{c['K']}] {c['R']}x{c['S']} p={c['pad']} s={c['stride']}"
    elif subroutine in ("pointwise_conv1d", "pointwise_conv1d_activation"):
        return f"[{cfg['batch']}, {cfg['seq']}, {cfg['channels']}] -> {cfg['out_channels']}"
    else:
        return f"[{cfg['batch']}, {cfg['seq']}, {cfg['channels']}] k={cfg['kernel_size']}"


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------


def run_standalone(variant: str = "depthwise_conv1d") -> None:
    """Standalone entry point for backwards-compat bench_*.py wrappers."""
    _VARIANT_MAP = {
        "depthwise_conv1d": {
            "title": "DepthwiseConv1D Kernels",
            "subroutines": ["depthwise_conv1d", "depthwise_conv1d_causal"],
        },
        "pointwise_conv1d": {
            "title": "PointwiseConv1D Kernels",
            "subroutines": ["pointwise_conv1d", "pointwise_conv1d_activation"],
        },
        "conv2d": {
            "title": "Conv2D NHWC Kernels",
            "subroutines": ["conv2d", "conv2d_activation"],
        },
    }

    info = _VARIANT_MAP.get(variant, _VARIANT_MAP["depthwise_conv1d"])
    title = info["title"]
    subs = info["subroutines"]

    pcfg = {k: PROFILE_CONFIGS[k] for k in subs if k in PROFILE_CONFIGS}
    setup_funcs = {}
    for sub in subs:
        if sub in PROFILE_CONFIGS:
            setup_funcs[sub] = _make_profile_setup(sub)

    def benchmark():
        output = OutputWriter()
        for sub in subs:
            configs = DEFAULT_CONFIGS.get(sub, [])
            output.write_header(f"{sub} Benchmark")
            for cfg in configs:
                oasr_fn, pytorch_fn = _setup_for_config(sub, cfg, torch.float16)
                shape_str = _shape_str(sub, cfg)
                for backend, fn in [("oasr", oasr_fn), ("pytorch", pytorch_fn)]:
                    median_ms, std_ms = bench_fn(fn)
                    tflops = _compute_tflops(sub, cfg, median_ms)
                    output.write_result(BenchResult(
                        routine="conv", subroutine=sub, backend=backend,
                        shape=shape_str, dtype="float16",
                        median_ms=median_ms, std_ms=std_ms,
                        tflops=tflops,
                    ))
        output.finalize()

    run_main(title, pcfg, setup_funcs, benchmark)


def _make_profile_setup(subroutine):
    cfg_tuple = PROFILE_CONFIGS[subroutine]

    def _setup():
        if subroutine == "depthwise_conv1d":
            return setup_depthwise_conv1d(*cfg_tuple)
        elif subroutine == "depthwise_conv1d_causal":
            return setup_depthwise_conv1d_causal(*cfg_tuple)
        elif subroutine == "pointwise_conv1d":
            return setup_pointwise_conv1d(*cfg_tuple)
        elif subroutine == "conv2d":
            return setup_conv2d(*cfg_tuple)
        elif subroutine == "conv2d_activation":
            return setup_conv2d_activation(*cfg_tuple)
        else:
            return setup_depthwise_conv1d(*cfg_tuple)

    return _setup
