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

#: Row floor below which a GEMM should not go to CUTLASS.
#:
#: CUTLASS tiles the M axis, and the default config's tile is 128 rows wide, so
#: a GEMM with ``M < 128`` leaves most of every tile empty.  cuBLAS switches to
#: a GEMV-shaped kernel instead and wins — by a little when the problem is
#: small, by a lot when it is not.  Measured on an RTX 5090, fp16, kernel vs
#: ``F.linear``:
#:
#: ===================  ========  =======  =========
#: shape (M, K, N)      CUTLASS   cuBLAS   ratio
#: ===================  ========  =======  =========
#: (8, 384, 384)        16.2 µs   14.8 µs  1.10×
#: (64, 384, 384)       16.2 µs   14.8 µs  1.10×
#: (4, 3584, 3584)      81.2 µs   15.2 µs  **5.34×**
#: (3000, 384, 384)     16.8 µs   15.6 µs  1.07×
#: ===================  ========  =======  =========
#:
#: That last skinny-but-large row is why this is a **row** count and not the
#: work product it started as: ``4 × 3584 × 3584`` is 51 M MACs, comfortably
#: over any sane work floor, yet it is the worst shape of the set — and it is
#: exactly a Qwen2-Audio-7B decode step.  A MACs floor cannot see the problem
#: because the problem is the *shape*, not the size.
#:
#: This is a **policy**, not a kernel gap.  The real fix is tuned rules for
#: these shapes: ``select_default_config`` has no entry for ``(3584, 3584)`` and
#: falls back to the default tile, and the tuner has only ever been run over
#: Conformer geometries.
#:
#: It must **not** be conditioned on ``is_current_stream_capturing()``, tempting
#: as that is (under capture the dispatch cost is paid once and replayed free).
#: A graph's contract is to reproduce the eager result, and a capture-dependent
#: branch breaks it: capture picked CUTLASS while eager picked cuBLAS for the
#: same shape, and the one-ulp fp16 difference reached the transducer decoder
#: as *different tokens*.  Any refinement has to stay a pure function of the
#: call.
GEMM_MIN_ROWS = 128

#: Work floor below which a **causal + windowed** attention stays on SDPA.
#:
#: Causal alone belongs to SDPA (it has a flash path and needs no mask tensor).
#: Causal *combined* with a key window is the opposite case: SDPA refuses
#: ``is_causal`` alongside ``attn_mask``, so the caller must materialize a
#: ``(B, 1, T_q, T_k)`` tensor and thereby forfeits flash, while the fused kernel
#: takes the same window as two length vectors and skips whole K blocks below the
#: diagonal.  That is worth 1.8-3.3x **on the attention op** — but only once the
#: fused path's fixed ~68 µs floor (the ``_ensure_canonical`` copies of q/k/v plus
#: the wrapper) is amortized.  Measured on an RTX 5090, bf16, with the strides the
#: real call site produces (q a ``split_heads`` view, k/v slices of a
#: capacity-preallocated KV buffer):
#:
#: =========================  =======  ========  =======
#: shape                      MACs     SDPA      ratio
#: =========================  =======  ========  =======
#: B1 H4 P128 D64             0.004 G  23.1 µs   0.34×
#: B4 H8 P256 D64             0.134 G  32.9 µs   0.48×
#: B4 H12 P384 D64            0.453 G  48.1 µs   0.67×
#: B2 H28 P384 D128           1.057 G  85.1 µs   1.02×
#: B4 H16 P512 D64            1.074 G  82.2 µs   0.99×
#: B4 H28 P512 D128           3.758 G  255.8 µs  1.99×
#: B4 H28 P1600 D128          45.9 G   2106 µs   3.29×
#: =========================  =======  ========  =======
#:
#: Unlike :data:`GEMM_MIN_ROWS`, a *work* measure is the right one here, and the
#: two rows either side of the threshold are why: 1.057 G at D=128 and 1.074 G at
#: D=64 land on the same ratio despite different B, H, P and D.  A fixed floor
#: being amortized predicts exactly that coincidence; a shape rule would not.
#:
#: Read the ratio as an *op-level* one.  A Qwen2-7B prefill layer is dominated by
#: its GEMMs (d=3584 qkv/o/mlp), so the same change is 1.03-1.05x over the whole
#: 32-layer prefill and 1.013x over an engine ``transcribe_offline`` with a short
#: generation — real, small, and transcript-identical.  Both of those were
#: measured with the arms **interleaved**: a single-order A/B first read 0.876x,
#: which was the second arm benefiting from a warm allocator rather than the
#: fused path losing.  The op-level number is still the right one to set this
#: threshold from, because it is what the threshold decides.
#:
#: Scoped to the causal+window combination, which is what was swept.  The
#: window-only routing is measured separately (see ``attention/core.py``), where
#: a short query extent — not total work — is what loses.
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
    """Clear the counters (per-test isolation, per-benchmark accounting)."""
    _GAP_HITS.clear()
    _POLICY_HITS.clear()
    _OUT_OF_SCOPE.clear()


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
    "use_fmha_kernel",
    "use_gemm_kernel",
    "use_norm_kernel",
]
