# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Top-k selection -- ``oasr/functionals/topk.py``."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size

SUBROUTINES = ["topk"]

DEFAULT_CONFIGS: Dict[str, list] = {
    "topk": [
        {"batch": 64, "seq": 250, "channels": 256, "k": 5},
        {"batch": 64, "seq": 250, "channels": 512, "k": 5},
        {"batch": 64, "seq": 250, "channels": 1024, "k": 5},
        {"batch": 64, "seq": 250, "channels": 2048, "k": 5},
        {"batch": 64, "seq": 250, "channels": 4096, "k": 5},
        {"batch": 64, "seq": 250, "channels": 4096, "k": 50},
        {"batch": 64, "seq": 250, "channels": 8192, "k": 5},
    ],
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--channels", type=int, default=None, help="Channel dimension")
    parser.add_argument("--k", type=int, default=None, help="Number of top elements")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    dims = (args.batch, args.seq, args.channels, args.k)
    if all(v is not None for v in dims):
        b, s, c, k = dims
        return [{"batch": b, "seq": s, "channels": c, "k": k}]
    return DEFAULT_CONFIGS[subroutine]


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    x = torch.randn(cfg["batch"], cfg["seq"], cfg["channels"], device="cuda", dtype=dtype)
    k = cfg["k"]
    return {
        "cuda": lambda: oasr.topk(x, k),
        "torch": lambda: torch.topk(x, k, dim=-1),
    }


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    b, s, c, k = cfg["batch"], cfg["seq"], cfg["channels"], cfg["k"]
    elem = dtype_size(dtype)
    # Read the input, write k values and k int32 indices.  Memory-bound: no
    # FLOP model, so `flops` stays None and the tflops cell stays empty.
    nbytes = b * s * c * elem + b * s * k * elem + b * s * k * 4
    return Work(shape=f"[{b}, {s}, {c}] k={k}", params=params_of(cfg), bytes=nbytes)
