# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Softmax -- ``oasr/functionals/softmax.py``."""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size

SUBROUTINES = ["softmax", "masked_softmax"]

DEFAULT_CONFIGS: Dict[str, list] = {
    "softmax": [
        {"batch": 32, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 512},
        {"batch": 64, "seq": 500, "channels": 256},
        {"batch": 64, "seq": 500, "channels": 512},
        {"batch": 64, "seq": 250, "channels": 1024},
    ],
    # KG6: Zipformer's shared attention weights.  The score tensor is
    # (head, batch, time, time), so the row length *is* the sequence length and
    # the row count scales with it -- a T^2 problem, which is why the unfused
    # arm pays six passes over it.  Lengths are what the stack's downsampling
    # ladder produces for ~5-10 s of audio.
    "masked_softmax": [
        {"heads": 8, "batch": 1, "seq": 500},
        {"heads": 8, "batch": 8, "seq": 500},
        {"heads": 8, "batch": 32, "seq": 500},
        {"heads": 8, "batch": 8, "seq": 250},
        {"heads": 8, "batch": 32, "seq": 125},
        {"heads": 4, "batch": 32, "seq": 63},
    ],
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--channels", type=int, default=None, help="Softmax dimension")
    parser.add_argument("--heads", type=int, default=None, help="Heads (masked_softmax)")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    if subroutine == "masked_softmax":
        if all(v is not None for v in (args.heads, args.batch, args.seq)):
            return [{"heads": args.heads, "batch": args.batch, "seq": args.seq}]
        return DEFAULT_CONFIGS["masked_softmax"]
    if all(v is not None for v in (args.batch, args.seq, args.channels)):
        return [{"batch": args.batch, "seq": args.seq, "channels": args.channels}]
    return DEFAULT_CONFIGS["softmax"]


def _masked_fns(heads: int, batch: int, seq: int, dtype: torch.dtype) -> Dict[str, Callable]:
    """Zipformer's ``scores + rel_pos_bias -> mask -> softmax``, three ways.

    ``cuda_unfused`` is the arm that matters: the same OASR softmax reached
    through separate add / masked_fill / contiguous ops, i.e. what the model ran
    *before* this kernel existed.  Without it a fused-vs-torch number cannot say
    how much of the win is the fusion and how much is the kernel underneath.
    """
    scores = torch.randn(heads, batch, seq, seq, device="cuda", dtype=dtype)
    # The relative-position product, consumed as the shifted window the encoder
    # builds: (head, batch, time, 2T-1) read as (head, batch, T, T).
    pos = torch.randn(heads, batch, seq, 2 * seq - 1, device="cuda", dtype=dtype)
    bias = pos.as_strided(
        (heads, batch, seq, seq),
        (pos.stride(0), pos.stride(1), pos.stride(2) - pos.stride(3), pos.stride(3)),
        storage_offset=pos.stride(3) * (seq - 1),
    )
    key_padding = torch.rand(batch, seq, device="cuda") < 0.15
    key_padding[:, 0] = False
    mask = key_padding.unsqueeze(1)

    def unfused(softmax_fn):
        biased = (scores + bias).masked_fill(mask, -1000)
        return softmax_fn(biased.contiguous())

    return {
        "cuda": lambda: oasr.masked_softmax(scores, bias=bias, mask2=mask, mask_value=-1000.0),
        "cuda_unfused": lambda: unfused(oasr.softmax),
        "torch": lambda: unfused(lambda t: F.softmax(t, dim=-1)),
    }


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if subroutine == "masked_softmax":
        return _masked_fns(cfg["heads"], cfg["batch"], cfg["seq"], dtype)
    x = torch.randn(cfg["batch"], cfg["seq"], cfg["channels"], device="cuda", dtype=dtype)
    return {"cuda": lambda: oasr.softmax(x), "torch": lambda: F.softmax(x, dim=-1)}


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    elem = dtype_size(dtype)
    if subroutine == "masked_softmax":
        h, b, s = cfg["heads"], cfg["batch"], cfg["seq"]
        # The *fused* traffic -- read scores, read the bias window, write output
        # -- reported for every arm, so the column is effective bandwidth for
        # the same logical work rather than a count of how many passes an arm took.
        return Work(
            shape=f"[{h}, {b}, {s}, {s}]",
            params=params_of(cfg),
            bytes=3 * h * b * s * s * elem,
        )
    b, s, c = cfg["batch"], cfg["seq"], cfg["channels"]
    return Work(shape=f"[{b}, {s}, {c}]", params=params_of(cfg), bytes=2 * b * s * c * elem)
