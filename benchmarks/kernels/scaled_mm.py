# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The 8-bit scaled GEMM -- ``oasr/functionals/scaled_mm.py``.

``out = act(a_scale[m] * b_scale[n] * (a @ bᵀ)[m, n] + bias[n])`` with 8-bit
per-token activations and 8-bit per-channel weights.  Five subroutines, the
int8 ones first:

``scaled_mm``
    The GEMM alone, operands quantized ahead of time.  What the kernel costs.
``quant_scaled_mm``
    ``quantize_per_token`` of an fp16/bf16 activation, then the GEMM -- the
    end-to-end cost a quantized layer pays in place of one half-precision GEMM.
``ffn_chain``
    A whole feed-forward block, ``w_2(swish(w_1(x)))`` with ``N`` = the model
    width and ``K`` = the FFN width: the declared GEMM chain.  ``cutlass`` is the
    fused chain (``scaled_mm_quant`` quantizes ``w_1``'s output at ``w_2``'s
    static scale; no fp intermediate, no second quantize), ``dynamic`` the
    unfused per-token path it replaces, ``fp16`` the two half-precision GEMMs.
``scaled_mm_fp8`` / ``quant_scaled_mm_fp8``
    The same two measurements in fp8 e4m3 (sm89+): ``cutlass`` is the Sm89 fp8
    lane, ``torch`` the ``torch._scaled_mm`` lane (``TorchScaledMMLane``), the
    oracle the fp8 reference.  An fp8 GEMM sums in fp32, so its tolerance adds
    that accumulation's rounding to the output ulp.

Arms:

``cutlass``
    ``oasr.scaled_mm`` on the device's default lane, dequant fused in the
    epilogue: the native CUTLASS 3.x kernel on Hopper / Blackwell data-center
    parts, the CUTLASS 2.x ``mma.sync`` one elsewhere.
``mma_sync``
    The same call on the CUTLASS 2.x ``mma.sync`` lane, present only where the
    default lane is native: what the native kernel buys, interleaved in one run
    (``OASR_SCALED_MM_BACKEND=mma_sync`` is the end-to-end form).
``torch``
    ``TorchIntMMLane`` -- ``torch._int_mm`` plus a torch epilogue, the A/B lane
    ``OASR_SCALED_MM_BACKEND=torch`` serves.  In ``quant_scaled_mm`` it is fed
    by the same CUDA quantizer, as it is when served.
``fp16``
    ``oasr.gemm`` / ``oasr.gemm_activation`` on the unquantized ``[M, N, K]``
    problem with the same bias and activation, in the run's dtype (so bf16
    under ``--dtype bfloat16``).  It is the speed baseline -- ``speedup_vs_ref``
    is how much the int8 path buys over the GEMM it replaces -- but it computes
    something else, so its refcheck is informational (``NON_GATING_BACKENDS``):
    the difference it reports is the quantization error, not a defect.
``reference``
    The refcheck oracle (``ORACLE_BACKEND``): ``reference_scaled_mm``, an exact
    fp64 accumulation with an fp32 epilogue.  Never timed unless named.  The
    kernel's epilogue fuses ``a_scale * t + bias`` into one FMA, so the int8
    arms agree with it to one output ulp, which is the tolerance -- per dtype,
    because a bf16 ulp is eight fp16 ones.

Outputs and quantization buffers are preallocated where the API allows, so the
timing excludes the allocator.  ``TorchIntMMLane.run`` has no ``out=`` and pays
for its own; that is the lane as served.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Callable, Dict, List, Optional

import torch

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import gemm_flops
from oasr.functionals.quant import reference_quantize_per_token
from oasr.functionals.scaled_mm import activation_id, reference_scaled_mm

SUBROUTINES = ["scaled_mm", "quant_scaled_mm", "ffn_chain", "scaled_mm_fp8", "quant_scaled_mm_fp8"]

#: The fp8 e4m3 subroutines.
_FP8_SUBROUTINES = ("scaled_mm_fp8", "quant_scaled_mm_fp8")

#: The comparison is int8 against half precision on the same allocator state.
INTERLEAVE = True

#: Speed baseline: the half-precision GEMM a quantized layer replaces.
REF_BACKEND = "fp16"

#: Correctness oracle: the exact torch twin.
ORACLE_BACKEND = "reference"

#: A different computation; its refcheck reports the quantization error
#: (``dynamic``: per-token scales where the fused chain has a static one).
NON_GATING_BACKENDS = frozenset({"fp16", "dynamic"})

#: One output ulp: 2**-10 relative for fp16, 2**-7 for bf16, with an absolute
#: floor of the same size for values near zero (the tanh-GELU epilogue differs
#: by more than an ulp *relative* there, never by more than this absolutely).
TOLERANCES = {
    **{(sub, torch.float16): (1e-3, 1e-3) for sub in SUBROUTINES},
    **{(sub, torch.bfloat16): (8e-3, 8e-3) for sub in SUBROUTINES},
    # The chain's intermediate is int8: where the reference's first GEMM lands one
    # output ulp from the kernel's (the epilogue's FMA) at a rounding tie, one
    # int8 code of ``w_1``'s output flips, and ``w_2`` carries that step -- one
    # static input scale times one weight, ~3e-3 on these operands.  The kernel
    # itself is bit-identical to the unfused CUDA pair
    # (``tests/quantization/test_quant_fusion.py``); this bounds the twin.
    ("ffn_chain", torch.float16): (1e-2, 1e-2),
    ("ffn_chain", torch.bfloat16): (1e-2, 1e-2),
    # fp8 sums in fp32 against the reference's exact sum: an ulp plus that rounding.
    **{(sub, torch.float16): (4e-3, 4e-3) for sub in _FP8_SUBROUTINES},
    **{(sub, torch.bfloat16): (1.6e-2, 1.6e-2) for sub in _FP8_SUBROUTINES},
}

#: The output dtypes the kernel writes.
_SERVED = (torch.float16, torch.bfloat16)

ACTIVATIONS = ("none", "relu", "swish", "gelu", "gelu_tanh")


def _mma_sync_lane(fmt: str) -> Optional[str]:
    """The ``mma.sync`` lane id when this device's default *fmt* lane is another one, else ``None``."""
    from oasr.jit.scaled_mm import LANE_ID, default_lane

    return LANE_ID[fmt] if default_lane(fmt=fmt) != LANE_ID[fmt] else None


def _with_mma_sync(
    fns: Dict[str, Callable[[], Any]], fmt: str, arm: Callable[[str], Callable[[], Any]]
) -> Dict[str, Callable[[], Any]]:
    """*fns* plus the ``mma_sync`` arm (``arm(lane)``) where the default lane is native."""
    lane = _mma_sync_lane(fmt)
    if lane is not None:
        fns["mma_sync"] = arm(lane)
    return fns


def _cfg(M: int, N: int, K: int, activation: str = "none", bias: bool = True) -> Dict[str, Any]:
    return {"M": M, "N": N, "K": K, "activation": activation, "bias": bias}


#: Conformer-base (d_model 256, FFN 2048): the attention input projection, the
#: square output projection, the FFN pair -- ``w_1`` is a fused-swish
#: ``LinearActivation`` -- at a streaming step, a micro-batch and an offline batch.
_CONFORMER: List[Dict[str, Any]] = [
    _cfg(m, n, k, act)
    for m in (16, 2048, 16384)
    for n, k, act in (
        (768, 256, "none"),
        (256, 256, "none"),
        (2048, 256, "swish"),
        (256, 2048, "none"),
    )
]

DEFAULT_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "scaled_mm": _CONFORMER,
    "quant_scaled_mm": _CONFORMER,
    "scaled_mm_fp8": _CONFORMER,
    "quant_scaled_mm_fp8": _CONFORMER,
    # Conformer-base FFN: width 256, FFN 2048, swish.
    "ffn_chain": [_cfg(m, 256, 2048, "swish") for m in (16, 256, 2048, 16384)],
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--M", type=int, default=None, help="Rows (tokens)")
    parser.add_argument("--N", type=int, default=None, help="Output channels (N % 8 == 0)")
    parser.add_argument("--K", type=int, default=None, help="Contraction (K % 16 == 0)")
    parser.add_argument(
        "--activation",
        choices=ACTIVATIONS,
        default=None,
        help="Fused epilogue activation, as oasr.layers spells it (gelu = erf). Default: none",
    )
    parser.add_argument("--bias", action="store_true", help="Add a per-channel bias")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> List[Dict[str, Any]]:
    dims = [getattr(args, name, None) for name in ("M", "N", "K")]
    activation = getattr(args, "activation", None)
    bias = bool(getattr(args, "bias", False))
    if all(d is not None for d in dims):
        M, N, K = dims
        return [_cfg(M, N, K, activation or "none", bias)]
    if any(d is not None for d in dims):
        # A partial shape used to run the whole default sweep under its name.
        raise SystemExit("--M, --N and --K go together for scaled_mm")
    configs = DEFAULT_CONFIGS[subroutine]
    if activation is not None:
        configs = [{**cfg, "activation": activation} for cfg in configs]
    return configs


def _act(cfg: Dict[str, Any]) -> Optional[str]:
    return None if cfg["activation"] == "none" else cfg["activation"]


def build_fns(
    subroutine: str, cfg: Dict[str, Any], dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    from oasr.quantization.kernels import TorchIntMMLane

    if dtype not in _SERVED:
        raise ValueError(f"scaled_mm writes fp16/bf16, not {dtype}")
    M, N, K = cfg["M"], cfg["N"], cfg["K"]
    act = _act(cfg)
    activation_id(act)  # an unknown name fails here, not inside a timed loop

    # Unit-variance activations against 1/sqrt(K) weights keep the output O(1),
    # where the ulp tolerance means what it says.
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    w = (torch.randn(N, K, device="cuda", dtype=torch.float32) / math.sqrt(K)).to(dtype)
    bias = (torch.randn(N, device="cuda") * 0.1).to(dtype) if cfg["bias"] else None
    # Per-output-channel weights are per-row quantization of the [N, K] matrix.
    w_q, w_scale = reference_quantize_per_token(w)
    lane = TorchIntMMLane()
    out = torch.empty(M, N, device="cuda", dtype=dtype)
    out_fp = torch.empty(M, N, device="cuda", dtype=dtype)

    if act is None:

        def fp16() -> torch.Tensor:
            return oasr.gemm(x, w, bias, out=out_fp)

    else:
        act_type = activation_id(act)

        def fp16() -> torch.Tensor:
            return oasr.gemm_activation(x, w, bias, act_type, out=out_fp)

    if subroutine == "ffn_chain":
        return _ffn_chain_fns(cfg, dtype)
    if subroutine in _FP8_SUBROUTINES:
        return _fp8_fns(subroutine, x, w, bias, act, dtype, out, fp16)

    if subroutine == "scaled_mm":
        a_q, a_scale = oasr.quantize_per_token(x)
        return _with_mma_sync(
            {
                "cutlass": lambda: oasr.scaled_mm(
                    a_q, w_q, a_scale, w_scale, bias, act, out_dtype=dtype, out=out
                ),
                "torch": lambda: lane.run(a_q, w_q, a_scale, w_scale, bias, act, dtype),
                "fp16": fp16,
                "reference": lambda: reference_scaled_mm(
                    a_q, w_q, a_scale, w_scale, bias, act, out_dtype=dtype
                ),
            },
            "int8",
            lambda mma: lambda: oasr.scaled_mm(
                a_q, w_q, a_scale, w_scale, bias, act, out_dtype=dtype, out=out, lane=mma
            ),
        )

    q_buf = torch.empty(M, K, device="cuda", dtype=torch.int8)
    s_buf = torch.empty(M, device="cuda", dtype=torch.float32)

    def cutlass() -> torch.Tensor:
        q, s = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
        return oasr.scaled_mm(q, w_q, s, w_scale, bias, act, out_dtype=dtype, out=out)

    def torch_lane() -> torch.Tensor:
        q, s = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
        return lane.run(q, w_q, s, w_scale, bias, act, dtype)

    def reference() -> torch.Tensor:
        q, s = reference_quantize_per_token(x)
        return reference_scaled_mm(q, w_q, s, w_scale, bias, act, out_dtype=dtype)

    def mma_sync(mma: str) -> Callable[[], Any]:
        def run() -> torch.Tensor:
            q, s = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
            return oasr.scaled_mm(q, w_q, s, w_scale, bias, act, out_dtype=dtype, out=out, lane=mma)

        return run

    return _with_mma_sync(
        {"cutlass": cutlass, "torch": torch_lane, "fp16": fp16, "reference": reference},
        "int8",
        mma_sync,
    )


def _fp8_fns(
    subroutine: str,
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    act: Optional[str],
    dtype: torch.dtype,
    out: torch.Tensor,
    fp16: Callable[[], Any],
) -> Dict[str, Callable[[], Any]]:
    from oasr.quantization.kernels import TorchScaledMMLane

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 89:
        raise ValueError(f"{subroutine}: fp8 tensor cores need sm89+")
    fp8 = torch.float8_e4m3fn
    w_q, w_scale = reference_quantize_per_token(w, dtype=fp8)
    lane = TorchScaledMMLane()
    if subroutine == "scaled_mm_fp8":
        a_q, a_scale = oasr.quantize_per_token(x, dtype=fp8)
        return _with_mma_sync(
            {
                "cutlass": lambda: oasr.scaled_mm(
                    a_q, w_q, a_scale, w_scale, bias, act, out_dtype=dtype, out=out
                ),
                "torch": lambda: lane.run(a_q, w_q, a_scale, w_scale, bias, act, dtype),
                "fp16": fp16,
                "reference": lambda: reference_scaled_mm(
                    a_q, w_q, a_scale, w_scale, bias, act, out_dtype=dtype
                ),
            },
            "fp8",
            lambda mma: lambda: oasr.scaled_mm(
                a_q, w_q, a_scale, w_scale, bias, act, out_dtype=dtype, out=out, lane=mma
            ),
        )

    M, K = x.shape
    q_buf = torch.empty(M, K, device="cuda", dtype=fp8)
    s_buf = torch.empty(M, device="cuda", dtype=torch.float32)

    def cutlass() -> torch.Tensor:
        q, s = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
        return oasr.scaled_mm(q, w_q, s, w_scale, bias, act, out_dtype=dtype, out=out)

    def torch_lane() -> torch.Tensor:
        q, s = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
        return lane.run(q, w_q, s, w_scale, bias, act, dtype)

    def reference() -> torch.Tensor:
        q, s = reference_quantize_per_token(x, dtype=fp8)
        return reference_scaled_mm(q, w_q, s, w_scale, bias, act, out_dtype=dtype)

    def mma_sync(mma: str) -> Callable[[], Any]:
        def run() -> torch.Tensor:
            q, s = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
            return oasr.scaled_mm(q, w_q, s, w_scale, bias, act, out_dtype=dtype, out=out, lane=mma)

        return run

    return _with_mma_sync(
        {"cutlass": cutlass, "torch": torch_lane, "fp16": fp16, "reference": reference},
        "fp8",
        mma_sync,
    )


def _ffn_chain_fns(cfg: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Callable[[], Any]]:
    from oasr.functionals.quant import static_multiplier
    from oasr.functionals.scaled_mm import reference_scaled_mm_quant

    M, d, ffn = cfg["M"], cfg["N"], cfg["K"]
    act = _act(cfg) or "swish"
    act_id = activation_id(act)
    x = torch.randn(M, d, device="cuda", dtype=dtype)
    w1 = (torch.randn(ffn, d, device="cuda") / math.sqrt(d)).to(dtype)
    w2 = (torch.randn(d, ffn, device="cuda") / math.sqrt(ffn)).to(dtype)
    b1 = (torch.randn(ffn, device="cuda") * 0.1).to(dtype)
    b2 = (torch.randn(d, device="cuda") * 0.1).to(dtype)
    w1_q, w1_s = reference_quantize_per_token(w1)
    w2_q, w2_s = reference_quantize_per_token(w2)
    # w_2's static input scale: the intermediate's absmax on this input (what
    # ``static_inputs`` would calibrate, without its clip search).
    q0, s0 = reference_quantize_per_token(x)
    y = reference_scaled_mm(q0, w1_q, s0, w1_s, b1, act, out_dtype=dtype)
    in_scale = (y.float().abs().amax().clamp_min(1e-8) / 127).reshape(1)
    mul = static_multiplier(in_scale, None, ffn)

    q_buf = torch.empty(M, d, device="cuda", dtype=torch.int8)
    s_buf = torch.empty(M, device="cuda", dtype=torch.float32)
    y_q = torch.empty(M, ffn, device="cuda", dtype=torch.int8)
    y_buf = torch.empty(M, ffn, device="cuda", dtype=dtype)
    y_qd = torch.empty(M, ffn, device="cuda", dtype=torch.int8)
    y_s = torch.empty(M, device="cuda", dtype=torch.float32)
    out = torch.empty(M, d, device="cuda", dtype=dtype)

    def fused() -> torch.Tensor:
        q, sc = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
        oasr.scaled_mm_quant(q, w1_q, sc, w1_s, b1, act, mul, dtype, out=y_q)
        return oasr.scaled_mm(y_q, w2_q, in_scale, w2_s, b2, None, out_dtype=dtype, out=out)

    def dynamic() -> torch.Tensor:
        q, sc = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
        oasr.scaled_mm(q, w1_q, sc, w1_s, b1, act, out_dtype=dtype, out=y_buf)
        qy, sy = oasr.quantize_per_token(y_buf, out=y_qd, scale=y_s)
        return oasr.scaled_mm(qy, w2_q, sy, w2_s, b2, None, out_dtype=dtype, out=out)

    def fp16() -> torch.Tensor:
        h = oasr.gemm_activation(x, w1, b1, act_id, out=y_buf)
        return oasr.gemm(h, w2, b2, out=out)

    def reference() -> torch.Tensor:
        yq = reference_scaled_mm_quant(q0, w1_q, s0, w1_s, b1, act, mul, dtype)
        return reference_scaled_mm(yq, w2_q, in_scale, w2_s, b2, None, out_dtype=dtype)

    def mma_sync(mma: str) -> Callable[[], Any]:
        def run() -> torch.Tensor:
            q, sc = oasr.quantize_per_token(x, out=q_buf, scale=s_buf)
            oasr.scaled_mm_quant(q, w1_q, sc, w1_s, b1, act, mul, dtype, out=y_q, lane=mma)
            return oasr.scaled_mm(
                y_q, w2_q, in_scale, w2_s, b2, None, out_dtype=dtype, out=out, lane=mma
            )

        return run

    return _with_mma_sync(
        {"cutlass": fused, "dynamic": dynamic, "fp16": fp16, "reference": reference},
        "int8",
        mma_sync,
    )


def describe(subroutine: str, cfg: Dict[str, Any], dtype: torch.dtype) -> Work:
    M, N, K = cfg["M"], cfg["N"], cfg["K"]
    if subroutine == "ffn_chain":
        return Work(
            shape=f"FFN M={M} d={N} ffn={K} {cfg['activation']}",
            params=params_of(cfg),
            flops=gemm_flops(M, K, N) + gemm_flops(M, N, K),
        )
    epilogue = f" {cfg['activation']}" if cfg["activation"] != "none" else ""
    if cfg["bias"]:
        epilogue += " +bias"
    return Work(
        shape=f"({M}, {N}, {K}){epilogue}",
        params=params_of(cfg),
        flops=gemm_flops(M, N, K),
    )
