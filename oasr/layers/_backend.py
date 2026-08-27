# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Which backend an :mod:`oasr.layers` module computes on.

OASR targets **GPU inference**.  ``oasr`` is the backend; PyTorch is an
*optional backend you select*, not a safety net the framework slides into.
The difference is not cosmetic: a layer that quietly routes an unsupported
shape to ``F.linear`` makes a missing kernel invisible, and an invisible gap
never gets closed.  Everything here exists to keep gaps visible.

Two tiers, and they are not the same thing:

**Out of scope.**  CPU tensors and fp32.  The framework does not target them —
they exist so the upstream parity oracles and the CPU test suite can run at
all — so the torch backend serves them.  Reported once per process, never an
error, and not a gap: there is nothing to close.

**In scope, unimplemented.**  A CUDA fp16/bf16 shape an OASR kernel refuses.
That is a **kernel gap**.  It must be declared in :data:`KERNEL_GAPS`, naming
what is missing and which layer has to fix it, or the module *raises* rather
than degrading.  Declared gaps are counted and printable
(:func:`format_gap_report`), and ``tests/test_layer_waist.py`` asserts the
declared set only shrinks.

Separately from both, a few calls go to torch because the kernel is measurably
**slower**, not because it cannot run — see :data:`GEMM_MIN_ROWS` and the
attention table in ``oasr/layers/attention/core.py``.  Those are performance
policy, counted under their own heading so they are never mistaken for gaps.
Each one is also a standing argument for kernel work.

``OASR_LAYERS_BACKEND`` selects the backend for the process:

``oasr`` (default)
    OASR kernels.  Out-of-scope inputs use torch; an undeclared in-scope
    refusal raises.
``torch``
    The optional backend, selected deliberately: never calls a kernel.  Used
    by the CPU parity oracles and as the A/B for "is this the kernels' fault".
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch

logger = logging.getLogger(__name__)

#: dtypes the CUTLASS GEMM kernels accept.
GEMM_DTYPES = (torch.float16, torch.bfloat16)
#: dtypes the handwritten norm kernels accept.
NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
#: dtypes the framework actually serves.  fp32 is out of scope (oracles only).
SERVED_DTYPES = (torch.float16, torch.bfloat16)
#: CUTLASS 2.x alignment-8 iterators: both GEMM free dimensions must divide by 8.
GEMM_ALIGNMENT = 8

#: GEMM row floor for the default tiled kernel. Exact shape rules may override
#: this performance policy; dispatch must remain independent of capture state.
GEMM_MIN_ROWS = 128

#: Work floor for fused causal-plus-window attention. Below it, fixed launch
#: overhead outweighs avoiding the library path's materialized mask.
FMHA_CAUSAL_WINDOW_MIN_MACS = 1 << 30


@dataclass(frozen=True)
class KernelGap:
    """A shape OASR has no kernel for, on hardware and in a dtype it serves."""

    #: Stable slug, used as the counter key and in the conformance test.
    id: str
    #: What the kernels cannot do.
    what: str
    #: Where it has to be fixed, and how.  Not "why we gave up".
    fix: str


#: Every in-scope shape currently served by torch because no kernel exists.
#:
#: This list is the project's honest kernel-coverage debt.  Adding an entry is
#: a deliberate act that shows up in review; removing one is the goal.  A
#: refusal with no entry here raises instead of silently degrading.
KERNEL_GAPS: Dict[str, KernelGap] = {
    g.id: g
    for g in (
        KernelGap(
            id="fmha-head-dim",
            what=(
                "a head_dim so large that even a single-stage cp.async ring "
                "overflows shared memory (>256 on a 99 KB arch)"
            ),
            fix=(
                "kernel: a smaller n_block for very wide heads, which would trade "
                "occupancy for the tile. head_dim 128 used to land here and no "
                "longer does — the ring depth is now sized to the arch's smem "
                "budget instead of hardcoded (FmhaSm80.select_num_stages), which "
                "is what stranded Paraformer's d_k=128 SANM attention on SDPA. No "
                "in-tree model reaches the remaining limit"
            ),
        ),
        KernelGap(
            id="fmha-paged-config",
            what=(
                "a paged attention config the arch class refuses: a head_dim off "
                "the paged loader's 32-element MMA k-stride (it skips per-element "
                "head-dim predication), or a page size no K tile is a multiple of"
            ),
            fix=(
                "kernel: predicate the paged loader's head-dim reads, which is "
                "what the dense path already does, and allow a page size that "
                "does not divide n_block by walking a partial page. Reached only "
                "by tiny test configs today — every shipped decoder's head_dim is "
                "64 or 128 and the pool's page size is a power of two — so the "
                "fallback here is a gather plus SDPA, correct but a full copy of "
                "the pages the paged path exists to avoid"
            ),
        ),
    )
}

_VALID_MODES = ("oasr", "torch")

_MODE: Optional[str] = None
#: gap id -> times taken this process.
_GAP_HITS: Counter = Counter()
#: policy reason -> times taken this process.
_POLICY_HITS: Counter = Counter()
#: out-of-scope reason -> times taken (reported once each).
_OUT_OF_SCOPE: Counter = Counter()


def layers_backend() -> str:
    """Resolved ``OASR_LAYERS_BACKEND`` (cached after the first read)."""
    global _MODE
    if _MODE is None:
        mode = os.environ.get("OASR_LAYERS_BACKEND", "oasr").strip().lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"OASR_LAYERS_BACKEND={mode!r} is not one of {_VALID_MODES}. "
                "(There is no 'auto': torch is a backend you select, not a fallback.)"
            )
        _MODE = mode
    return _MODE


def set_layers_backend(mode: str) -> None:
    """Select the backend for this process (tests, benchmarks, A/B switches)."""
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    global _MODE
    _MODE = mode


@contextmanager
def layers_backend_override(mode: str) -> Iterator[None]:
    """Scoped :func:`set_layers_backend`."""
    previous = layers_backend()
    set_layers_backend(mode)
    try:
        yield
    finally:
        set_layers_backend(previous)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def gap_hits() -> Dict[str, int]:
    """Declared kernel gaps taken this process, by id."""
    return dict(_GAP_HITS)


def policy_hits() -> Dict[str, int]:
    """Torch calls made on *performance* grounds, by reason."""
    return dict(_POLICY_HITS)


def reset_backend_stats() -> None:
    """Clear the counters (per-test isolation, per-benchmark accounting).

    Includes the GEMM rule-miss table, so ``reset`` → run → report stays one call
    now that :func:`format_gap_report` reports both.
    """
    _GAP_HITS.clear()
    _POLICY_HITS.clear()
    _OUT_OF_SCOPE.clear()
    try:
        from oasr.jit.gemm import reset_rule_misses
    except Exception:  # noqa: BLE001
        return
    reset_rule_misses()


def format_gap_report() -> str:
    """Human-readable summary of what did not reach a kernel, and why.

    Deliberately separates the three categories: a gap is debt, a policy hit is
    a decision, an out-of-scope hit is neither.
    """
    lines = [f"oasr.layers backend: {layers_backend()}"]
    if _GAP_HITS:
        lines.append("  kernel gaps taken (missing kernels — debt):")
        for gid, n in sorted(_GAP_HITS.items()):
            lines.append(f"    {gid:<20} x{n:<8} {KERNEL_GAPS[gid].what}")
            lines.append(f"    {'':<20}  {'':<8} fix at → {KERNEL_GAPS[gid].fix}")
    if _POLICY_HITS:
        lines.append("  torch chosen on performance grounds (measured, not missing):")
        for reason, n in sorted(_POLICY_HITS.items()):
            lines.append(f"    {reason:<20} x{n}")
    if _OUT_OF_SCOPE:
        lines.append("  out of scope for a GPU inference framework:")
        for reason, n in sorted(_OUT_OF_SCOPE.items()):
            lines.append(f"    {reason:<20} x{n}")
    if not (_GAP_HITS or _POLICY_HITS or _OUT_OF_SCOPE):
        lines.append("  every call reached an OASR kernel")
    # A GEMM that reached a kernel can still have reached an *untuned* one, which
    # is a third thing: not debt, not a decision, just nobody having measured this
    # model's widths.  Imported lazily — ``oasr.jit.gemm`` is deliberately kept off
    # the layers import path (see the note at the top of ``oasr/functionals/gemm.py``).
    try:
        from oasr.jit.gemm import rule_misses
    except Exception:  # noqa: BLE001 — diagnostics must never break the caller
        return "\n".join(lines)
    misses = rule_misses()
    if misses:
        lines.append("  GEMM shapes with no tuned rule (ran on the fallback tile):")
        for (op, N, K), (calls, m_lo, m_hi) in sorted(misses.items(), key=lambda kv: -kv[1][0]):
            span = f"{m_lo}" if m_lo == m_hi else f"{m_lo}..{m_hi}"
            lines.append(f"    {op:<18} N={N:<6} K={K:<6} x{calls:<7} M={span}")
        lines.append("    tune with → scripts/tune_asr_gemm.py (see oasr/jit/gemm.py)")
    return "\n".join(lines)


def out_of_scope(reason: str) -> bool:
    """CPU / fp32: served by torch because the framework does not target them."""
    first = _OUT_OF_SCOPE[reason] == 0
    _OUT_OF_SCOPE[reason] += 1
    if first:
        logger.info(
            "oasr.layers: %s — using the torch backend (OASR targets GPU fp16/bf16 "
            "inference; this path exists for the CPU parity oracles).",
            reason,
        )
    return False


def take_gap(gap_id: str, detail: str) -> bool:
    """Record a declared kernel gap, or raise if it is not declared.

    Returning ``False`` (use torch) is only legitimate for a gap somebody has
    written down. An undeclared refusal is a bug or an unfinished kernel, and
    it should stop the run rather than quietly cost throughput forever.
    """
    gap = KERNEL_GAPS.get(gap_id)
    if gap is None:
        raise RuntimeError(
            f"no OASR kernel for this call ({detail}) and no declared gap "
            f"{gap_id!r}. Fix it at the kernel or model layer, or — if it "
            f"genuinely cannot be fixed yet — add a KernelGap to "
            f"oasr/layers/_backend.py saying where it must be fixed."
        )
    first = _GAP_HITS[gap_id] == 0
    _GAP_HITS[gap_id] += 1
    if first:
        logger.warning(
            "oasr.layers: kernel gap %r taken (%s) — %s. Fix at → %s",
            gap_id,
            detail,
            gap.what,
            gap.fix,
        )
    return False


def take_policy(reason: str) -> bool:
    _POLICY_HITS[reason] += 1
    return False


# ---------------------------------------------------------------------------
# Per-family backend decisions
# ---------------------------------------------------------------------------


def use_gemm_kernel(x: torch.Tensor, in_features: int, out_features: int) -> bool:
    """Should this projection go through ``oasr.gemm``?"""
    if layers_backend() == "torch":
        return False
    if not x.is_cuda:
        return out_of_scope("CPU tensor")
    if x.dtype not in SERVED_DTYPES:
        return out_of_scope(f"dtype {x.dtype}")

    if in_features % GEMM_ALIGNMENT or out_features % GEMM_ALIGNMENT:
        # Not declared as a gap: every in-tree case is fixable at the model
        # layer by padding the projection, which the WeNet CTC head has done
        # since day one.  So this raises, and the fix is to pad.
        return take_gap(
            "gemm-unaligned",
            f"Linear({in_features} -> {out_features}) is not "
            f"{GEMM_ALIGNMENT}-aligned on both axes; pad the projection "
            f"(see oasr/models/conformer/convert.py for the CTC-head precedent)",
        )

    if x.numel() // in_features < GEMM_MIN_ROWS:
        return take_policy("gemm-below-row-floor")
    return True


def is_row_dense(x: torch.Tensor) -> bool:
    """Do ``x``'s trailing-dim rows tile its memory exactly, in some order?

    The real precondition of a row-wise kernel walking ``base + row * row_len``
    — see ``IsRowDense`` in ``csrc/tvm_ffi_utils.h``, which this mirrors.  It
    accepts a *permuted* dense view (Zipformer's ``(T, B, C)`` transpose) and
    rejects a padded one (``x[..., :32]`` of a wider buffer), where plain
    ``is_contiguous()`` refuses both.
    """
    if x.dim() < 1 or x.stride(-1) != 1:
        return False
    span = 1
    for extent, stride in zip(x.shape, x.stride()):
        if extent > 1 and stride <= 0:
            return False
        span += (extent - 1) * stride
    return span == x.numel()


def use_norm_kernel(x: torch.Tensor) -> bool:
    """Should this normalization go through the OASR norm kernels?"""
    if layers_backend() == "torch":
        return False
    if not x.is_cuda:
        return out_of_scope("CPU tensor")
    if x.dtype not in NORM_DTYPES:
        return out_of_scope(f"dtype {x.dtype}")
    if not is_row_dense(x):
        # A padded row stride is not a kernel gap: the rows do not tile memory,
        # so there is nothing for a row-wise kernel to walk.  ``torch.empty_like``
        # does not preserve such strides either, so the output rows would not
        # line up with the input's even if it did.
        return take_policy("norm-rows-not-dense")
    return True


def use_conv_kernel(x: torch.Tensor) -> bool:
    """Should this convolution / activation go through an OASR kernel?

    Deliberately **only** the out-of-scope gate — CPU tensors and dtypes the
    handwritten kernels do not dispatch (``DISPATCH_DLPACK_HALF_DTYPE`` in
    ``csrc/conv.cu`` / ``csrc/conv2d.cu`` / ``csrc/activation.cu`` covers fp16
    and bf16 only).  No alignment rule and no row floor: those belong to the
    launchers, which already raise with a message naming the fix, and adding
    them here would silently reroute calls that work today.

    This exists because ``oasr.layers.conv`` used to have no torch path at all:
    every class called its kernel unconditionally, so a conv-front-end
    architecture could not be built *and run* on CPU, which is what every fp32
    parity oracle in this repo needs.  The waist's contract is that each layer
    owns both paths (see the module docstring); conv was the exception.
    """
    if layers_backend() == "torch":
        return False
    if not x.is_cuda:
        return out_of_scope("CPU tensor")
    if x.dtype not in SERVED_DTYPES:
        return out_of_scope(f"dtype {x.dtype}")
    return True


def use_pooling_kernel(x: torch.Tensor) -> bool:
    """Should pooling run through the OASR kernel?

    Pooling has the same serving scope as convolution: CUDA FP16/BF16.  The
    direct kernel covers every argument combination accepted by the waist, so
    an in-scope refusal is an error rather than a torch fallback.
    """
    if layers_backend() == "torch":
        return False
    if not x.is_cuda:
        return out_of_scope("CPU tensor")
    if x.dtype not in SERVED_DTYPES:
        return out_of_scope(f"dtype {x.dtype}")
    return True


def use_recurrent_kernel(x: torch.Tensor) -> bool:
    """Should an LSTM/RNN run through the fused recurrent kernels?

    The recurrent family covers every positive unidirectional shape in CUDA
    FP16/BF16.  CPU/fp32 remains the formula-level parity path; an in-scope
    refusal is therefore an error, not a silent framework fallback.
    """
    if layers_backend() == "torch":
        return False
    if not x.is_cuda:
        return out_of_scope("CPU tensor")
    if x.dtype not in SERVED_DTYPES:
        return out_of_scope(f"dtype {x.dtype}")
    return True


def use_gated_mlp_kernel(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    activation: str,
    has_bias: bool = False,
) -> bool:
    """Should a gated MLP fuse gate + up + activation + multiply into one kernel?

    Declining is **never** a kernel gap.  The alternative is the two-GEMM path,
    which is itself fully kernel-backed (``oasr.gemm_activation`` for the gate,
    ``oasr.gemm`` for the up, and one elementwise multiply); what the fusion
    removes is four passes over an ``(M, N)`` intermediate and two launches, not
    a hole in coverage.  So an out-of-band or unsupported shape is a *policy*
    hit under its own reason, and the measurements behind the band live in
    :mod:`oasr.jit.mlp`.
    """
    if layers_backend() == "torch":
        return False
    if not x.is_cuda:
        return out_of_scope("CPU tensor")
    if x.dtype not in SERVED_DTYPES:
        return out_of_scope(f"dtype {x.dtype}")
    from oasr.jit.mlp import get_gated_mlp_mode

    if get_gated_mlp_mode() == "off":
        # The rollback switch, not a judgement about this shape.  Counting it
        # would make an A/B run look like a table of declines.
        return False
    from oasr.functionals.mlp import gated_mlp_available

    if gated_mlp_available(x, w_gate, activation=activation, has_bias=has_bias):
        return True
    return take_policy("gated-mlp-unfused")


def use_fmha_kernel() -> bool:
    """Is the OASR attention kernel selectable at all?"""
    return layers_backend() != "torch"


__all__ = [
    "GEMM_ALIGNMENT",
    "GEMM_DTYPES",
    "GEMM_MIN_ROWS",
    "KERNEL_GAPS",
    "KernelGap",
    "NORM_DTYPES",
    "SERVED_DTYPES",
    "format_gap_report",
    "gap_hits",
    "is_row_dense",
    "layers_backend",
    "layers_backend_override",
    "policy_hits",
    "reset_backend_stats",
    "set_layers_backend",
    "out_of_scope",
    "take_gap",
    "take_policy",
    "use_conv_kernel",
    "use_fmha_kernel",
    "use_gated_mlp_kernel",
    "use_gemm_kernel",
    "use_norm_kernel",
    "use_pooling_kernel",
    "use_recurrent_kernel",
]
