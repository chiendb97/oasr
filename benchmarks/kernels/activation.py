# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Pointwise activations -- ``oasr/functionals/activation.py``.

All memory-bound: each declares bytes and no FLOP model, so the ``tflops`` cell
stays empty rather than reporting a misleading zero.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size

SUBROUTINES = ["gelu", "glu", "relu", "sigmoid", "swish", "tanh"]

DEFAULT_CONFIGS: Dict[str, list] = {
    "gelu": [
        {"batch": 8, "seq": 1500, "channels": 384},
        {"batch": 16, "seq": 1500, "channels": 384},
        {"batch": 8, "seq": 1500, "channels": 1280},
        {"batch": 16, "seq": 1500, "channels": 1280},
    ],
    "glu": [
        {"batch": 32, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 512},
        {"batch": 64, "seq": 500, "channels": 256},
        {"batch": 64, "seq": 500, "channels": 512},
    ],
    "relu": [
        {"batch": 1, "seq": 1, "channels": 512},
        {"batch": 8, "seq": 1, "channels": 512},
        {"batch": 32, "seq": 1, "channels": 640},
        {"batch": 8, "seq": 250, "channels": 256},
    ],
    "sigmoid": [
        {"batch": 1, "seq": 400, "channels": 8},
        {"batch": 8, "seq": 400, "channels": 8},
        {"batch": 32, "seq": 400, "channels": 8},
        {"batch": 8, "seq": 250, "channels": 256},
    ],
    "swish": [
        {"batch": 32, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 512},
        {"batch": 64, "seq": 500, "channels": 512},
    ],
    "tanh": [
        {"batch": 1, "seq": 1, "channels": 512},
        {"batch": 8, "seq": 1, "channels": 512},
        {"batch": 32, "seq": 1, "channels": 512},
        {"batch": 16, "seq": 1500, "channels": 384, "row_stride_factor": 3},
        {"batch": 8, "seq": 250, "channels": 256},
    ],
}

#: ``(oasr fn, torch fn)`` per subroutine.  GLU is the one that reads twice what
#: it writes -- it halves the channel axis.
_OPS = {
    "gelu": (lambda x: oasr.gelu(x), lambda x: F.gelu(x)),
    "glu": (lambda x: oasr.glu(x), lambda x: F.glu(x, dim=-1)),
    "relu": (lambda x: oasr.relu(x), lambda x: torch.relu(x)),
    "sigmoid": (lambda x: oasr.sigmoid(x), lambda x: torch.sigmoid(x)),
    "swish": (lambda x: oasr.swish(x), lambda x: F.silu(x)),
    "tanh": (lambda x: oasr.tanh(x), lambda x: torch.tanh(x)),
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--channels", type=int, default=None, help="Channel dimension")
    parser.add_argument(
        "--row-stride-factor",
        type=int,
        default=1,
        help="Tanh input backing-row width as a multiple of channels (Zipformer: 3)",
    )


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    if all(v is not None for v in (args.batch, args.seq, args.channels)):
        stride = args.row_stride_factor
        if stride != 1 and subroutine != "tanh":
            raise SystemExit("[ERROR] --row-stride-factor is only supported by tanh")
        return [
            {
                "batch": args.batch,
                "seq": args.seq,
                "channels": args.channels,
                "row_stride_factor": stride,
            }
        ]
    return DEFAULT_CONFIGS[subroutine]


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
    stride = cfg.get("row_stride_factor", 1)
    if subroutine == "glu":
        x = torch.randn(b, s, 2 * c, device="cuda", dtype=dtype)
    elif stride != 1:
        # A non-contiguous view: the backing row is wider than the slice, which
        # is how Zipformer hands tanh its input.
        if stride < 1:
            raise ValueError(f"row_stride_factor must be positive, got {stride}")
        x = torch.randn(b, s, stride * c, device="cuda", dtype=dtype)[..., :c]
    else:
        x = torch.randn(b, s, c, device="cuda", dtype=dtype)

    oasr_op, torch_op = _OPS[subroutine]
    return {"cuda": lambda: oasr_op(x), "torch": lambda: torch_op(x)}


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
    elem = dtype_size(dtype)
    reads = 2 if subroutine == "glu" else 1
    shape = f"[{b}, {s}, {2 * c}]" if subroutine == "glu" else f"[{b}, {s}, {c}]"
    if cfg.get("row_stride_factor", 1) != 1:
        shape += f" row_stride={cfg['row_stride_factor']}x"
    return Work(
        shape=shape,
        params=params_of(cfg),
        bytes=b * s * c * elem * (reads + 1),
    )
