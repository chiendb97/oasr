# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Normalisation -- ``oasr/functionals/norm.py``.

All twelve subroutines survive the consolidation.  ``batch_norm_swish`` looks
like ``batch_norm_activation --activation swish`` and was slated to be merged
into it, but they call two different kernels -- a specialised fusion and a
generic one taking an activation id -- and the comparison between them is the
reason the specialised one exists.  Same for ``add_*`` against
``add_*_residual``: different kernels, one of which also writes the summed
residual back.  What did collapse is the *builders*: one closure factory per
family, parameterised, instead of near-duplicate copies.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size

SUBROUTINES = [
    "layer_norm",
    "add_layer_norm",
    "add_layer_norm_residual",
    "layer_norm_activation",
    "rms_norm",
    "add_rms_norm",
    "add_rms_norm_residual",
    "batch_norm",
    "batch_norm_swish",
    "batch_norm_activation",
    "group_norm",
    "cmvn",
]

_BN = [
    {"batch": 32, "seq": 250, "hidden": 256},
    {"batch": 64, "seq": 250, "hidden": 256},
    {"batch": 64, "seq": 250, "hidden": 512},
    {"batch": 64, "seq": 500, "hidden": 512},
]
_ACT = [
    {"batch": 64, "seq": 250, "hidden": 256},
    {"batch": 64, "seq": 250, "hidden": 512},
    {"batch": 64, "seq": 500, "hidden": 512},
]
#: Qwen2-Audio-7B decode / prefill, plus a tiny test geometry.
_RMS_ADD = [
    {"batch": 1, "seq": 1, "hidden": 3584},
    {"batch": 4, "seq": 128, "hidden": 3584},
    {"batch": 8, "seq": 256, "hidden": 512},
]
#: Whisper / Qwen2-Audio tower, Paraformer, Nemotron.
_LN_ADD = [
    {"batch": 16, "seq": 1500, "hidden": 384},
    {"batch": 32, "seq": 250, "hidden": 512},
    {"batch": 8, "seq": 200, "hidden": 1024},
]

DEFAULT_CONFIGS: Dict[str, list] = {
    "layer_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 256},
        {"batch": 64, "seq": 500, "hidden": 512},
        {"batch": 32, "seq": 500, "hidden": 512},
    ],
    "add_layer_norm": _LN_ADD,
    "add_layer_norm_residual": _LN_ADD,
    "layer_norm_activation": _ACT,
    "rms_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "add_rms_norm": _RMS_ADD,
    "add_rms_norm_residual": _RMS_ADD,
    "batch_norm": _BN,
    "batch_norm_swish": _BN,
    "batch_norm_activation": _ACT,
    "group_norm": [
        {"batch": 32, "seq": 250, "hidden": 256, "num_groups": 32},
        {"batch": 64, "seq": 250, "hidden": 256, "num_groups": 32},
        {"batch": 64, "seq": 250, "hidden": 512, "num_groups": 64},
        {"batch": 64, "seq": 500, "hidden": 512, "num_groups": 64},
    ],
    "cmvn": [
        {"batch": 32, "seq": 250, "hidden": 80},
        {"batch": 64, "seq": 250, "hidden": 80},
        {"batch": 64, "seq": 500, "hidden": 80},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
}

_EPS = 1e-5
_RMS_EPS = 1e-6


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden / channel dimension")
    parser.add_argument("--num-groups", type=int, default=32, help="Groups (group_norm)")
    parser.add_argument(
        "--activation",
        default="swish",
        help="Activation for the fused variants (relu, gelu, swish)",
    )


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    if all(v is not None for v in (args.batch, args.seq, args.hidden)):
        cfg = {"batch": args.batch, "seq": args.seq, "hidden": args.hidden}
        if subroutine == "group_norm":
            cfg["num_groups"] = args.num_groups
        return [cfg]
    return DEFAULT_CONFIGS[subroutine]


def _layer_norm_fns(b, s, h, dtype, subroutine, activation):
    x = torch.randn(b, s, h, device="cuda", dtype=dtype)
    gamma = torch.randn(h, device="cuda", dtype=dtype)
    beta = torch.randn(h, device="cuda", dtype=dtype)

    if subroutine in ("add_layer_norm", "add_layer_norm_residual"):
        residual = torch.randn_like(x)
        alpha = 0.5 if subroutine.endswith("_residual") else 1.0
        out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        if subroutine.endswith("_residual"):

            def oasr_fn():
                normalized, _ = oasr.add_layer_norm_residual(
                    x, residual, gamma, beta, _EPS, alpha, out=out, residual_out=residual_out
                )
                return normalized

        else:

            def oasr_fn():
                return oasr.add_layer_norm(x, residual, gamma, beta, _EPS)

        def torch_fn():
            # fp32 reference: the kernel accumulates the sum in fp32 too, so a
            # half-precision reference would measure the reference's rounding.
            summed = residual.float() + alpha * x.float()
            return F.layer_norm(summed, (h,), gamma.float(), beta.float(), _EPS).to(dtype)

        return {"cuda": oasr_fn, "torch": torch_fn}

    ln = torch.nn.LayerNorm(h, eps=_EPS, device="cuda", dtype=dtype)
    ln.weight.data = gamma.clone()
    ln.bias.data = beta.clone()
    if subroutine == "layer_norm_activation":
        from oasr.utils.mappings import get_activation

        act_id = oasr.get_activation_type_id(activation)
        torch_act = get_activation(activation).cuda()
        return {
            "cuda": lambda: oasr.layer_norm_activation(x, gamma, beta, _EPS, act_id),
            "torch": lambda: torch_act(ln(x)),
        }
    return {"cuda": lambda: oasr.layer_norm(x, gamma, beta, _EPS), "torch": lambda: ln(x)}


def _rms_norm_fns(b, s, h, dtype, subroutine):
    x = torch.randn(b, s, h, device="cuda", dtype=dtype)
    gamma = torch.randn(h, device="cuda", dtype=dtype)

    if subroutine == "rms_norm":

        def torch_fn():
            xf = x.float()
            scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + _EPS)
            return (xf * scale * gamma.float()).to(dtype)

        return {"cuda": lambda: oasr.rms_norm(x, gamma, None, _EPS), "torch": torch_fn}

    residual = torch.randn_like(x)
    alpha = 1.0
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    if subroutine.endswith("_residual"):

        def oasr_fn():
            normalized, _ = oasr.add_rms_norm_residual(
                x, residual, gamma, None, _RMS_EPS, alpha, out=out, residual_out=residual_out
            )
            return normalized

    else:

        def oasr_fn():
            return oasr.add_rms_norm(x, residual, gamma, None, _RMS_EPS, alpha, out=out)

    def torch_fn():
        summed = residual.float() + alpha * x.float()
        scale = torch.rsqrt(summed.pow(2).mean(dim=-1, keepdim=True) + _RMS_EPS)
        return (summed * scale * gamma.float()).to(dtype)

    return {"cuda": oasr_fn, "torch": torch_fn}


def _batch_norm_fns(b, s, c, dtype, subroutine, activation):
    x = torch.randn(b, s, c, device="cuda", dtype=dtype)
    gamma = torch.randn(c, device="cuda", dtype=dtype)
    beta = torch.randn(c, device="cuda", dtype=dtype)
    mean = torch.randn(c, device="cuda", dtype=dtype)
    var = torch.randn(c, device="cuda", dtype=dtype).abs() + 0.1

    def normalized():
        return (x - mean) / torch.sqrt(var + _EPS) * gamma + beta

    if subroutine == "batch_norm_swish":
        return {
            "cuda": lambda: oasr.batch_norm_swish(x, gamma, beta, mean, var, _EPS),
            "torch": lambda: F.silu(normalized()),
        }
    if subroutine == "batch_norm_activation":
        from oasr.utils.mappings import get_activation

        act_id = oasr.get_activation_type_id(activation)
        torch_act = get_activation(activation).cuda()
        return {
            "cuda": lambda: oasr.batch_norm_activation(x, gamma, beta, mean, var, _EPS, act_id),
            "torch": lambda: torch_act(normalized()),
        }
    return {
        "cuda": lambda: oasr.batch_norm_1d(x, gamma, beta, mean, var, _EPS),
        "torch": normalized,
    }


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
    activation = getattr(args, "activation", "swish")

    if subroutine.endswith("rms_norm") or subroutine.endswith("rms_norm_residual"):
        return _rms_norm_fns(b, s, h, dtype, subroutine)
    if subroutine.startswith("batch_norm"):
        return _batch_norm_fns(b, s, h, dtype, subroutine, activation)
    if subroutine == "group_norm":
        groups = cfg.get("num_groups", 32)
        per_group = h // groups
        x = torch.randn(b, s, h, device="cuda", dtype=dtype)
        gamma = torch.randn(h, device="cuda", dtype=dtype)
        beta = torch.randn(h, device="cuda", dtype=dtype)

        def torch_fn():
            xr = x.view(b, s, groups, per_group)
            mean = xr.mean(dim=-1, keepdim=True)
            var = xr.var(dim=-1, keepdim=True, unbiased=False)
            return ((xr - mean) / torch.sqrt(var + _EPS)).view(b, s, h) * gamma + beta

        return {
            "cuda": lambda: oasr.group_norm(x, gamma, beta, groups, _EPS),
            "torch": torch_fn,
        }
    if subroutine == "cmvn":
        x = torch.randn(b, s, h, device="cuda", dtype=dtype)
        mean = torch.randn(h, device="cuda", dtype=dtype)
        istd = torch.randn(h, device="cuda", dtype=dtype).abs() + 0.1
        return {
            "cuda": lambda: oasr.cmvn(x, mean, istd),
            "torch": lambda: (x - mean) * istd,
        }
    return _layer_norm_fns(b, s, h, dtype, subroutine, activation)


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
    elem = dtype_size(dtype)
    reads = 2 if subroutine.startswith("add_") else 1
    writes = 2 if subroutine.endswith("_residual") else 1
    shape = f"[{b}, {s}, {h}]"
    if subroutine == "group_norm":
        shape += f" g={cfg.get('num_groups', 32)}"
    return Work(
        shape=shape,
        params=params_of(cfg),
        # activations in and out, plus gamma and beta
        bytes=b * s * h * elem * (reads + writes) + h * elem * 2,
    )
