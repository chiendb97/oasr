#!/usr/bin/env python3
"""Depthwise BTC Conv1D and Paraformer FSMN end-to-end benchmark.

Measurement arms rotate between rounds so allocator/cache warmth cannot always
favor the same implementation.  Sigma is over round medians.  The FSMN arms
compare the new single-kernel expression against the exact pre-KG2 OASR chain
and the equivalent PyTorch/cuDNN chain, including their intermediate tensors.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import bench_fn

SHAPES = (
    # Paraformer encoder: LFR-short and LFR-long utterances.
    (1, 80, 512, 11, 5, 5),
    (1, 500, 512, 11, 5, 5),
    (8, 80, 512, 11, 5, 5),
    (8, 500, 512, 11, 5, 5),
    # Shifted SANM windows exercise asymmetric padding.
    (1, 500, 512, 11, 7, 3),
    (8, 500, 512, 11, 7, 3),
)

DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}

_COL_SHAPE = 34
_COL_TIME = 20
_COL_SPEEDUP = 10

_HEADER = (
    f"{'shape':>{_COL_SHAPE}}"
    f"  {'Fused E2E':>{_COL_TIME}}"
    f"  {'Kernel':>{_COL_TIME}}"
    f"  {'Previous OASR':>{_COL_TIME}}"
    f"  {'PyTorch':>{_COL_TIME}}"
    f"  {'Old/New':>{_COL_SPEEDUP}}"
    f"  {'Torch/New':>{_COL_SPEEDUP}}"
)
_SEP = "-" * len(_HEADER)
_TITLE = "OASR Depthwise Conv1D Benchmark"
_TITLE_SEP = "=" * len(_HEADER)


def _fmt_time(timing: tuple[float, float]) -> str:
    median_ms, std_ms = timing
    return f"{median_ms:.4f}±{std_ms:.4f}ms"


def _fmt_speedup(value: float) -> str:
    return f"{value:.2f}x"


def _shape_str(shape) -> str:
    batch, seq_len, channels, kernel_size, padding_left, padding_right = shape
    return f"[{batch},{seq_len},{channels}] k={kernel_size} p=({padding_left},{padding_right})"


def _row(shape, timings) -> str:
    fused_ms = timings["fused_e2e"][0]
    return (
        f"{_shape_str(shape):>{_COL_SHAPE}}"
        f"  {_fmt_time(timings['fused_e2e']):>{_COL_TIME}}"
        f"  {_fmt_time(timings['kernel']):>{_COL_TIME}}"
        f"  {_fmt_time(timings['previous_oasr']):>{_COL_TIME}}"
        f"  {_fmt_time(timings['torch_e2e']):>{_COL_TIME}}"
        f"  {_fmt_speedup(timings['previous_oasr'][0] / fused_ms):>{_COL_SPEEDUP}}"
        f"  {_fmt_speedup(timings['torch_e2e'][0] / fused_ms):>{_COL_SPEEDUP}}"
    )


def _measure_interleaved(arms, rounds: int, iterations: int):
    samples = {name: [] for name in arms}
    names = list(arms)
    for round_idx in range(rounds):
        offset = round_idx % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            median_ms, _ = bench_fn(arms[name], dry_run_iters=2, num_iters=iterations)
            samples[name].append(median_ms)
    return {
        name: (statistics.median(values), statistics.stdev(values) if len(values) > 1 else 0.0)
        for name, values in samples.items()
    }


def _run_shape(shape, dtype, rounds: int, iterations: int, mask_dtype: str):
    batch, seq_len, channels, kernel_size, padding_left, padding_right = shape
    padding = (padding_left, padding_right)
    x = torch.randn(batch, seq_len, channels, device="cuda", dtype=dtype)
    weight = torch.randn(kernel_size, channels, device="cuda", dtype=dtype)
    weight_torch = weight.T.unsqueeze(1).contiguous()
    lengths = torch.randint(
        max(1, seq_len // 2), seq_len + 1, (batch,), device="cuda", dtype=torch.int64
    )
    bool_mask = (
        torch.arange(seq_len, device="cuda").unsqueeze(0) < lengths.unsqueeze(1)
    ).unsqueeze(-1)
    input_mask = bool_mask if mask_dtype == "bool" else bool_mask.to(dtype)
    kernel_output = torch.empty_like(x)

    def kernel_only():
        return oasr.depthwise_conv1d(
            x,
            weight,
            padding=padding,
            out=kernel_output,
            mask=input_mask,
            add_input=True,
        )

    def fused_e2e():
        return oasr.depthwise_conv1d(
            x,
            weight,
            padding=padding,
            mask=input_mask,
            add_input=True,
        )

    def previous_oasr_e2e():
        # This is the operation sequence removed from Paraformer modules.py:
        # cast + mul + pad + depthwise kernel + add + mul.
        mask = bool_mask.to(dtype) if mask_dtype == "bool" else input_mask
        masked = x * mask
        padded = F.pad(masked, (0, 0, padding_left, padding_right))
        conv = oasr.depthwise_conv1d(padded, weight)
        return (conv + masked) * mask

    def torch_e2e():
        mask = bool_mask.to(dtype) if mask_dtype == "bool" else input_mask
        masked = x * mask
        conv = F.conv1d(
            F.pad(masked.transpose(1, 2), padding),
            weight_torch,
            groups=channels,
        ).transpose(1, 2)
        return (conv + masked) * mask

    fused = fused_e2e()
    previous = previous_oasr_e2e()
    torch_ref = torch_e2e()
    torch.cuda.synchronize()
    torch.testing.assert_close(fused, torch_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(previous, torch_ref, rtol=2e-2, atol=2e-2)

    arms = {
        "kernel": kernel_only,
        "fused_e2e": fused_e2e,
        "previous_oasr": previous_oasr_e2e,
        "torch_e2e": torch_e2e,
    }
    return _measure_interleaved(arms, rounds, iterations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "all"),
        default="all",
        help="Dtype section(s) to include in the report",
    )
    parser.add_argument("--mask-dtype", choices=("bool", "activation"), default="bool")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--shape-index", type=int, help="Run one zero-based SHAPES entry")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rounds <= 0 or args.iterations <= 0:
        parser.error("--rounds and --iterations must be positive")

    shapes = SHAPES if args.shape_index is None else (SHAPES[args.shape_index],)
    device = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    dtypes = tuple(DTYPE_NAMES) if args.dtype == "all" else (getattr(torch, args.dtype),)

    print(_TITLE)
    print(_TITLE_SEP)
    print(
        f"device={device}, sm={capability[0]}{capability[1]}, mask={args.mask_dtype}, "
        f"rounds={args.rounds}, iterations={args.iterations}"
    )
    for dtype in dtypes:
        print(f"\n--- {DTYPE_NAMES[dtype]} ---")
        print(_HEADER)
        print(_SEP)
        for shape in shapes:
            timings = _run_shape(shape, dtype, args.rounds, args.iterations, args.mask_dtype)
            print(_row(shape, timings))

    print()


if __name__ == "__main__":
    main()
