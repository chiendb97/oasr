# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The int8 convolution -- ``oasr/functionals/scaled_conv.py``.

``out = act(scale[k] * conv(q_x, q_w) + bias[k])`` over an int8 NHWC input
quantized with one static per-tensor scale and an int8 KRSC filter quantized per
output channel: the forward of a quantized ``Conv2d`` (``conv1d`` is the
height-one problem).  Subroutines:

``scaled_conv2d``
    The convolution alone, operands quantized ahead of time.  What the kernel costs.
``quant_scaled_conv2d``
    ``quantize_static`` of the fp16/bf16 input, then the convolution -- what a
    quantized ``Conv2d`` pays in place of one half-precision convolution.
``scaled_conv1d``
    The ``(B, T, C)`` convolution over time of a quantized ``Conv1d``, quantize
    included (Whisper's front-end).

Arms: ``cutlass`` (the int8 kernel), ``fp16`` (``oasr.conv2d_activation`` /
``oasr.conv1d`` on the unquantized problem: the speed baseline, informational in
refcheck -- its difference is the quantization error), and ``reference`` (the
exact fp64 twin, the refcheck oracle; never timed unless named).
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Callable, Dict, List, Optional

import torch

import oasr
from benchmarks.core.driver import Work, params_of
from oasr.functionals.quant import quantize_static, reference_quantize_static
from oasr.functionals.scaled_conv import (
    reference_scaled_conv1d,
    reference_scaled_conv2d,
    scaled_conv1d,
    scaled_conv2d,
)
from oasr.functionals.scaled_mm import activation_id

SUBROUTINES = ["scaled_conv2d", "quant_scaled_conv2d", "scaled_conv1d"]

INTERLEAVE = True
REF_BACKEND = "fp16"
ORACLE_BACKEND = "reference"
NON_GATING_BACKENDS = frozenset({"fp16"})

#: One output ulp (2**-10 fp16, 2**-7 bf16) with an absolute floor of the same size.
TOLERANCES = {
    **{(sub, torch.float16): (1e-3, 1e-3) for sub in SUBROUTINES},
    **{(sub, torch.bfloat16): (8e-3, 8e-3) for sub in SUBROUTINES},
}

_SERVED = (torch.float16, torch.bfloat16)
ACTIVATIONS = ("none", "relu", "swish", "gelu", "gelu_tanh")


def _cfg(n, h, w, c, k, r, s, stride=1, pad=0, activation="none") -> Dict[str, Any]:
    return {
        "N": n, "H": h, "W": w, "C": c, "K": k, "R": r, "S": s,
        "stride": stride, "pad": pad, "activation": activation,
    }  # fmt: skip


#: The U2++ Conformer's dense subsampling conv (256 -> 256, 3x3 / 2 over the
#: one-channel stem's output) at batch 1, 8 and 64 of 6.5 s utterances.
_CONFORMER = [_cfg(n, 324, 39, 256, 256, 3, 3, 2, 0, "relu") for n in (1, 8, 64)]

#: Whisper's front-end: conv1 (mel -> d, k3) and conv2 (d -> d, k3 / 2), a 30 s window.
_WHISPER = [
    _cfg(b, 1, 3000, c, k, 1, 3, s, 1, "none")
    for b in (1, 16)
    for c, k, s in ((80, 384, 1), (384, 384, 2), (128, 1280, 1), (1280, 1280, 2))
]

DEFAULT_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "scaled_conv2d": _CONFORMER,
    "quant_scaled_conv2d": _CONFORMER,
    "scaled_conv1d": _WHISPER,
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    for name, help_ in (
        ("N", "Batch"), ("H", "Input height (1 for conv1d)"), ("W", "Input width / time"),
        ("C", "Input channels (C % 16 == 0)"), ("K", "Output channels (K % 8 == 0)"),
        ("R", "Filter height"), ("S", "Filter width"),
    ):  # fmt: skip
        parser.add_argument(f"--{name}", type=int, default=None, help=help_)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--pad", type=int, default=None)
    parser.add_argument("--activation", choices=ACTIVATIONS, default=None)


def resolve_configs(args: argparse.Namespace, subroutine: str) -> List[Dict[str, Any]]:
    dims = [getattr(args, n, None) for n in ("N", "H", "W", "C", "K", "R", "S")]
    if all(d is not None for d in dims):
        return [
            _cfg(
                *dims, args.stride or 1, args.pad or 0, getattr(args, "activation", None) or "none"
            )
        ]
    if any(d is not None for d in dims):
        raise SystemExit("--N --H --W --C --K --R --S go together for scaled_conv")
    return DEFAULT_CONFIGS[subroutine]


def _act(cfg: Dict[str, Any]) -> Optional[str]:
    return None if cfg["activation"] == "none" else cfg["activation"]


def build_fns(
    subroutine: str, cfg: Dict[str, Any], dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if dtype not in _SERVED:
        raise ValueError(f"scaled_conv writes fp16/bf16, not {dtype}")
    n, h, w, c, k, r, s = (cfg[d] for d in ("N", "H", "W", "C", "K", "R", "S"))
    stride, pad, act = cfg["stride"], cfg["pad"], _act(cfg)
    activation_id(act)
    x = torch.randn(n, h, w, c, device="cuda", dtype=dtype)
    wt = (torch.randn(k, r, s, c, device="cuda") / math.sqrt(r * s * c)).to(dtype)
    bias = (torch.randn(k, device="cuda") * 0.1).to(dtype)
    # Per-output-channel weights; one static input scale at the input's absmax.
    wmax = wt.float().reshape(k, -1).abs().amax(dim=1).clamp_min(1e-8)
    w_q = torch.round(wt.float() / (wmax / 127).reshape(k, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
    in_scale = (x.float().abs().amax().clamp_min(1e-8) / 127).reshape(1)
    scale = (in_scale * (wmax / 127)).contiguous()
    mul = torch.full((c,), 1.0, device="cuda") / in_scale
    x_q = quantize_static(x, mul)
    q_buf = torch.empty_like(x_q)

    if subroutine == "scaled_conv1d":
        x1, qb1 = x.reshape(n, w, c), q_buf.reshape(n, w, c)
        w1, wq1 = wt.reshape(k, s, c), w_q.reshape(k, s, c)

        def cutlass() -> torch.Tensor:
            q = quantize_static(x1, mul, out=qb1)
            return scaled_conv1d(q, wq1, scale, bias, act, pad, stride, out_dtype=dtype)

        def fp16() -> torch.Tensor:
            out = oasr.conv1d(x1, w1, bias, padding=pad, stride=stride)
            return out if act is None else _torch_act(act)(out)

        def reference() -> torch.Tensor:
            q = reference_quantize_static(x1, mul)
            return reference_scaled_conv1d(q, wq1, scale, bias, act, pad, stride, out_dtype=dtype)

        return {"cutlass": cutlass, "fp16": fp16, "reference": reference}

    def fp16() -> torch.Tensor:
        if act in (None, "gelu"):
            out = oasr.conv2d(x, wt, bias, pad, pad, stride, stride)
            return out if act is None else _torch_act(act)(out)
        return oasr.conv2d_activation(
            x, wt, bias, oasr.get_activation_type_id(_FP16_ACT[act]), pad, pad, stride, stride
        )

    if subroutine == "scaled_conv2d":
        return {
            "cutlass": lambda: scaled_conv2d(
                x_q, w_q, scale, bias, act, pad, stride, out_dtype=dtype
            ),
            "fp16": fp16,
            "reference": lambda: reference_scaled_conv2d(
                x_q, w_q, scale, bias, act, pad, stride, out_dtype=dtype
            ),
        }

    def cutlass() -> torch.Tensor:
        q = quantize_static(x, mul, out=q_buf)
        return scaled_conv2d(q, w_q, scale, bias, act, pad, stride, out_dtype=dtype)

    def reference() -> torch.Tensor:
        q = reference_quantize_static(x, mul)
        return reference_scaled_conv2d(q, w_q, scale, bias, act, pad, stride, out_dtype=dtype)

    return {"cutlass": cutlass, "fp16": fp16, "reference": reference}


#: The fp16 convolution's activation ids by the quantized layer's spelling.
_FP16_ACT = {"relu": "relu", "swish": "swish", "gelu_tanh": "gelu"}


def _torch_act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    from oasr.functionals.scaled_mm import _TORCH_ACTIVATION

    return _TORCH_ACTIVATION[name]


def describe(subroutine: str, cfg: Dict[str, Any], dtype: torch.dtype) -> Work:
    n, h, w, c, k, r, s = (cfg[d] for d in ("N", "H", "W", "C", "K", "R", "S"))
    stride, pad = cfg["stride"], cfg["pad"]
    if subroutine == "scaled_conv1d":
        p, q = 1, (w + 2 * pad - s) // stride + 1
    else:
        p, q = (h + 2 * pad - r) // stride + 1, (w + 2 * pad - s) // stride + 1
    flops = 2 * n * p * q * k * r * s * c
    elem = torch.finfo(dtype).bits // 8
    # The int8 input and filter in, the half-precision output out.
    nbytes = n * h * w * c + k * r * s * c + n * p * q * k * elem
    if subroutine != "scaled_conv2d":
        nbytes += n * h * w * c * elem  # the quantize reads the fp input
    return Work(
        shape=f"[{n}, {h}, {w}, {c}] * [{k}, {r}, {s}, {c}] / {stride}",
        params=params_of(cfg),
        flops=flops,
        bytes=nbytes,
    )
