# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Real FFT -- ``oasr/functionals/fft.py``.

Feature-extraction shapes: ``batch * num_frames`` flattened transforms, with
``n_fft`` a power of two in ``[8, 2048]``.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size

SUBROUTINES = ["rfft", "rfft_power"]

#: The kernel is single-precision only; the driver reports the override rather
#: than silently labelling an fp32 measurement float16.
FORCE_DTYPE = "float32"

_SHAPES = [
    {"batch": 32, "num_frames": 250, "n_fft": 256},
    {"batch": 32, "num_frames": 250, "n_fft": 512},
    {"batch": 64, "num_frames": 500, "n_fft": 512},
    {"batch": 64, "num_frames": 500, "n_fft": 1024},
    {"batch": 64, "num_frames": 500, "n_fft": 2048},
]
DEFAULT_CONFIGS: Dict[str, list] = {"rfft": _SHAPES, "rfft_power": _SHAPES}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--num-frames", type=int, default=None, help="Frames per utterance")
    parser.add_argument("--n-fft", type=int, default=None, help="FFT length (power of two)")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    dims = (args.batch, args.num_frames, args.n_fft)
    if all(v is not None for v in dims):
        return [{"batch": dims[0], "num_frames": dims[1], "n_fft": dims[2]}]
    return DEFAULT_CONFIGS[subroutine]


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    x = torch.randn(cfg["batch"], cfg["num_frames"], cfg["n_fft"], device="cuda", dtype=dtype)
    if subroutine == "rfft_power":
        return {
            "cuda": lambda: oasr.rfft_power(x),
            "torch": lambda: torch.fft.rfft(x).abs().pow(2),
        }
    return {"cuda": lambda: oasr.rfft(x), "torch": lambda: torch.fft.rfft(x)}


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    b, f, n = cfg["batch"], cfg["num_frames"], cfg["n_fft"]
    elem = dtype_size(dtype)
    # rfft writes interleaved complex (two reals per bin); rfft_power writes one.
    out_per_frame = (n // 2 + 1) * (2 if subroutine == "rfft" else 1)
    return Work(
        shape=f"[{b}, {f}, n_fft={n}]",
        params=params_of(cfg),
        bytes=b * f * n * elem + b * f * out_per_frame * elem,
    )
