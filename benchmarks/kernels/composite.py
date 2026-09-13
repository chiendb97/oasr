# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Whole sub-modules built from several kernels.

``conv_block`` is the Conformer ConvolutionModule: pointwise GEMM -> GLU ->
depthwise conv -> Swish -> pointwise GEMM.

It declares ``flops``, which the previous version did not.  The point of a
composite benchmark is to be compared against the sum of its parts, and without
a work model it could only be compared in milliseconds -- against parts measured
at different shapes.  ``bytes`` stays absent on purpose: the activation traffic
of a five-kernel chain depends on how much of it fuses, so one number would mean
something different for each arm.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import conv1d_flops, gemm_flops

SUBROUTINES = ["conv_block"]

DEFAULT_CONFIGS: Dict[str, list] = {
    "conv_block": [
        {"batch": 32, "seq": 250, "d_model": 256, "kernel_size": 15},
        {"batch": 64, "seq": 250, "d_model": 256, "kernel_size": 15},
        {"batch": 64, "seq": 250, "d_model": 512, "kernel_size": 31},
        {"batch": 64, "seq": 500, "d_model": 256, "kernel_size": 15},
        {"batch": 64, "seq": 500, "d_model": 512, "kernel_size": 31},
    ],
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument("--kernel-size", type=int, default=None, help="Depthwise kernel size")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    dims = (args.batch, args.seq, args.d_model, args.kernel_size)
    if all(v is not None for v in dims):
        return [{"batch": dims[0], "seq": dims[1], "d_model": dims[2], "kernel_size": dims[3]}]
    return DEFAULT_CONFIGS[subroutine]


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    b, s, d, k = cfg["batch"], cfg["seq"], cfg["d_model"], cfg["kernel_size"]
    x = torch.randn(b, s, d, device="cuda", dtype=dtype)
    pw1_w = torch.randn(2 * d, d, device="cuda", dtype=dtype)
    pw1_b = torch.randn(2 * d, device="cuda", dtype=dtype)
    dw_w = torch.randn(k, d, device="cuda", dtype=dtype)
    dw_b = torch.randn(d, device="cuda", dtype=dtype)
    pw2_w = torch.randn(d, d, device="cuda", dtype=dtype)
    pw2_b = torch.randn(d, device="cuda", dtype=dtype)

    # (kernel_size, d_model) -> (d_model, 1, kernel_size).  A bare `view` here
    # reinterprets the bytes instead of transposing, and the reference then
    # convolves with a scrambled filter.
    dw_w_pt = dw_w.permute(1, 0).reshape(d, 1, k)

    def oasr_fn():
        out = oasr.gemm(x, pw1_w, pw1_b)
        out = oasr.glu(out)
        out = oasr.depthwise_conv1d(out, dw_w, dw_b, k // 2)
        out = oasr.swish(out)
        return oasr.gemm(out, pw2_w, pw2_b)

    def torch_fn():
        out = F.glu(F.linear(x, pw1_w, pw1_b), dim=-1)
        out = F.conv1d(out.permute(0, 2, 1), dw_w_pt, dw_b, padding=k // 2, groups=d)
        out = F.silu(out.permute(0, 2, 1))
        return F.linear(out, pw2_w, pw2_b)

    return {"cuda": oasr_fn, "torch": torch_fn}


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    b, s, d, k = cfg["batch"], cfg["seq"], cfg["d_model"], cfg["kernel_size"]
    rows = b * s
    flops = (
        gemm_flops(rows, 2 * d, d)  # pointwise expand
        + conv1d_flops(b, s, d, d, k, groups=d)  # depthwise
        + gemm_flops(rows, d, d)  # pointwise project
    )
    return Work(shape=f"[{b}, {s}, {d}] k={k}", params=params_of(cfg), flops=flops)
