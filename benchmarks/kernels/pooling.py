# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""1D pooling -- ``oasr/functionals/pooling.py``.

Interleaved by default: the arms differ by a layout transpose, which is exactly
the kind of gap a warm allocator can manufacture in a single-order A/B.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size
from oasr.functionals.pooling import _pool1d_output_length

SUBROUTINES = ["avg_pool1d", "max_pool1d"]
INTERLEAVE = True

DEFAULT_CONFIGS: Dict[str, list] = {
    "avg_pool1d": [
        {
            "batch": batch,
            "seq": 1500,
            "channels": 1280,
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "ceil_mode": False,
            "count_include_pad": True,
        }
        for batch in (1, 2, 4, 8)
    ],
    # The dilation the ASR-derived speech detectors run: a one-channel per-frame
    # trace, stride 1, half-width padding.  One channel is the launcher's narrow
    # path -- the case the CTA-per-row mapping used to serve at 1/32 occupancy.
    "max_pool1d": [
        {
            "batch": batch,
            "seq": 1500,
            "channels": 1,
            "kernel_size": 5,
            "stride": 1,
            "padding": 2,
            "ceil_mode": False,
            "count_include_pad": False,
        }
        for batch in (1, 8, 32)
    ]
    + [
        {
            "batch": 4,
            "seq": 500,
            "channels": 512,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": False,
        }
    ],
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Input time extent")
    parser.add_argument("--channels", type=int, default=None, help="Channel dimension")
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--ceil-mode", action="store_true")
    parser.add_argument("--exclude-pad", action="store_true")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    if all(v is not None for v in (args.batch, args.seq, args.channels)):
        return [
            {
                "batch": args.batch,
                "seq": args.seq,
                "channels": args.channels,
                "kernel_size": args.kernel_size,
                "stride": args.stride,
                "padding": args.padding,
                "ceil_mode": args.ceil_mode,
                "count_include_pad": not args.exclude_pad,
            }
        ]
    return DEFAULT_CONFIGS[subroutine]


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    k, s, p = cfg["kernel_size"], cfg["stride"], cfg["padding"]
    ceil, incl = cfg["ceil_mode"], cfg["count_include_pad"]
    x_btc = torch.randn(cfg["batch"], cfg["seq"], cfg["channels"], device="cuda", dtype=dtype)
    # Pre-transposed: the torch arm is the raw pooling baseline, not a layout cost.
    x_bct = x_btc.transpose(1, 2).contiguous()

    if subroutine == "max_pool1d":
        # count_include_pad has no meaning without a divisor; accepting it keeps
        # one testlist able to drive both subroutines.
        return {
            "cuda": lambda: oasr.max_pool1d(x_btc, k, s, p, ceil),
            "torch": lambda: F.max_pool1d(x_bct, k, s, p, 1, ceil).transpose(1, 2),
        }

    return {
        "cuda": lambda: oasr.avg_pool1d(x_btc, k, s, p, ceil, incl),
        "torch": lambda: F.avg_pool1d(x_bct, k, s, p, ceil, incl).transpose(1, 2),
        # The pre-KG4 Speech-LLM expression, verbatim: BTC -> BCT -> BTC with a
        # materialising contiguous().  Kept as an arm because the kernel exists
        # to replace it, so the comparison that justifies it should be runnable.
        "torch_previous": lambda: F.avg_pool1d(x_btc.transpose(1, 2), k, s, p, ceil, incl)
        .transpose(1, 2)
        .contiguous(),
    }


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    out_len = _pool1d_output_length(
        cfg["seq"], cfg["kernel_size"], cfg["stride"], cfg["padding"], cfg["ceil_mode"]
    )
    elements = cfg["batch"] * cfg["channels"] * (cfg["seq"] + out_len)
    return Work(
        shape=(
            f"[{cfg['batch']},{cfg['seq']},{cfg['channels']}] "
            f"k={cfg['kernel_size']} s={cfg['stride']} p={cfg['padding']}"
        ),
        params=params_of(cfg),
        bytes=elements * dtype_size(dtype),
    )
