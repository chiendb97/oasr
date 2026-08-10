#!/usr/bin/env python3
"""Dense BTC Conv1D: CUTLASS vs direct-NHWC cuDNN vs torch end to end.

The arms are rotated between measurement rounds so allocator/cache warmth does
not systematically favor the second implementation.  Reported sigma is over
round medians, not over asynchronous launch issue time.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import bench_fn
from oasr.conv import (
    _default_conv1d_activation_fn,
    _default_conv1d_fn,
    _dispatch_conv1d,
    _dispatch_conv1d_activation,
    _get_cudnn_conv2d_module,
)

SHAPES = (
    # Whisper front-end.
    (1, 3000, 80, 384, 3, 1, 1),
    (1, 3000, 384, 384, 3, 1, 2),
    # Qwen2-Audio front-end.
    (1, 3000, 128, 1280, 3, 1, 1),
    (1, 3000, 1280, 1280, 3, 1, 2),
    # Paraformer CIF predictor.
    # The predictor explicitly pads one frame on each side before this call.
    (1, 502, 512, 512, 3, 0, 1),
)


def _measure_interleaved(arms, rounds: int, iterations: int):
    samples = {name: [] for name in arms}
    names = list(arms)
    for round_idx in range(rounds):
        order = names[round_idx % len(names) :] + names[: round_idx % len(names)]
        for name in order:
            median_ms, _ = bench_fn(arms[name], dry_run_iters=2, num_iters=iterations)
            samples[name].append(median_ms)
    return {
        name: (statistics.median(values), statistics.stdev(values) if len(values) > 1 else 0.0)
        for name, values in samples.items()
    }


def _one_shape(shape, dtype, activation, rounds, iterations, tune):
    batch, seq, in_channels, out_channels, kernel, padding, stride = shape
    out_seq = (seq + 2 * padding - kernel) // stride + 1
    x = torch.randn(batch, seq, in_channels, device="cuda", dtype=dtype)
    weight = torch.randn(out_channels, kernel, in_channels, device="cuda", dtype=dtype)
    bias = torch.randn(out_channels, device="cuda", dtype=dtype)
    output_names = ["cutlass_default", "production", "cudnn"] + (["tuned"] if tune else [])
    outputs = {
        name: torch.empty(batch, out_seq, out_channels, device="cuda", dtype=dtype)
        for name in output_names
    }

    cudnn = _get_cudnn_conv2d_module()
    if activation is None:
        cutlass = _default_conv1d_fn()

        def cutlass_fn():
            cutlass(outputs["cutlass_default"], x, weight, bias, padding, stride, 1)

        def production_fn():
            _dispatch_conv1d(outputs["production"], x, weight, bias, padding, stride, 1)

        def cudnn_fn():
            cudnn.cudnn_conv1d(outputs["cudnn"], x, weight, bias, padding, stride, 1)

        def torch_fn():
            return F.conv1d(
                x.transpose(1, 2),
                weight.permute(0, 2, 1),
                bias,
                padding=padding,
                stride=stride,
            ).transpose(1, 2)

        def tuned_fn():
            return oasr.conv1d(
                x,
                weight,
                bias,
                padding=padding,
                stride=stride,
                out=outputs["tuned"],
            )

    else:
        cutlass = _default_conv1d_activation_fn()
        activation_id = {"relu": 0, "gelu_tanh": 1, "silu": 2}[activation]
        torch_activation = {
            "relu": F.relu,
            "gelu_tanh": lambda value: F.gelu(value, approximate="tanh"),
            "silu": F.silu,
        }[activation]

        def cutlass_fn():
            cutlass(
                outputs["cutlass_default"],
                x,
                weight,
                bias,
                activation_id,
                padding,
                stride,
                1,
            )

        def production_fn():
            _dispatch_conv1d_activation(
                outputs["production"],
                x,
                weight,
                bias,
                activation_id,
                padding,
                stride,
                1,
            )

        def cudnn_fn():
            cudnn.cudnn_conv1d_activation(
                outputs["cudnn"], x, weight, bias, activation_id, padding, stride, 1
            )

        def torch_fn():
            value = F.conv1d(
                x.transpose(1, 2),
                weight.permute(0, 2, 1),
                bias,
                padding=padding,
                stride=stride,
            )
            return torch_activation(value).transpose(1, 2)

        def tuned_fn():
            return oasr.conv1d_activation(
                x,
                weight,
                bias,
                activation_id,
                padding=padding,
                stride=stride,
                out=outputs["tuned"],
            )

    cutlass_fn()
    cudnn_fn()
    ref = torch_fn()
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs["cutlass_default"], ref, rtol=2e-2, atol=2e-2)
    production_fn()
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs["production"], ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(outputs["cudnn"], ref, rtol=2e-2, atol=2e-2)
    selected = None
    if tune:
        tuned_fn()  # Profile once; this leaves the last candidate's output behind.
        tuned_fn()  # The cache hit executes the selected tactic for the parity check.
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs["tuned"], ref, rtol=2e-2, atol=2e-2)
        from oasr.tune import get_selected_config
        from oasr.tune.autotuner import OpKey

        op = "conv1d" if activation is None else "conv1d_activation"
        selected = get_selected_config(
            OpKey("conv", op),
            (
                batch,
                seq,
                in_channels,
                out_channels,
                kernel,
                padding,
                stride,
                1,
            ),
            str(dtype).removeprefix("torch."),
            torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1],
        )

    arms = {
        "cutlass_default": cutlass_fn,
        "production": production_fn,
        "cudnn": cudnn_fn,
        "torch_e2e": torch_fn,
    }
    if tune:
        arms["tuned"] = tuned_fn
    timings = _measure_interleaved(arms, rounds, iterations)
    flops = 2 * batch * out_seq * out_channels * in_channels * kernel
    label = f"B{batch} T{seq} {in_channels}->{out_channels} k{kernel} p{padding} s{stride}"
    print(f"{label:<40}", end="")
    order = (
        ("cutlass_default", "production", "tuned", "cudnn", "torch_e2e") if tune else tuple(arms)
    )
    for name in order:
        median_ms, std_ms = timings[name]
        tflops = flops / (median_ms * 1e-3) / 1e12
        print(f" {name}={median_ms:.4f}±{std_ms:.4f} ms ({tflops:.1f}T)", end="")
    print(f" torch/production={timings['torch_e2e'][0] / timings['production'][0]:.3f}x")
    if selected is not None:
        print(f"  selected={selected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--activation", choices=("relu", "gelu_tanh", "silu"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--tune", action="store_true", help="Profile CUTLASS/cuDNN tactics")
    parser.add_argument("--shape-index", type=int, help="Run only this zero-based SHAPES entry")
    parser.add_argument("--batch", type=int, help="Override the batch dimension of every shape")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = getattr(torch, args.dtype)
    print(f"dense Conv1D {args.dtype}, activation={args.activation or 'identity'}")
    tuning = oasr.autotune(warmup=25, rep=100) if args.tune else contextlib.nullcontext()
    shapes = SHAPES if args.shape_index is None else (SHAPES[args.shape_index],)
    if args.batch is not None:
        if args.batch <= 0:
            parser.error("--batch must be positive")
        shapes = tuple((args.batch, *shape[1:]) for shape in shapes)
    with tuning:
        for shape in shapes:
            _one_shape(shape, dtype, args.activation, args.rounds, args.iterations, tune=args.tune)


if __name__ == "__main__":
    main()
