# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Where a layer picks between an OASR kernel and the torch reference.

``oasr/layers/`` is the narrow waist every model implementation goes through
(see ``docs/architecture.md``).  A layer is **one** module class, not two: it
owns a kernel path *and* a reference path and chooses between them here.  That
is what lets a model written against the waist run unchanged on CPU/fp32 — the
parity oracles and the whole CPU test suite — while picking up the CUDA kernels
on the serving path with no model-side edit.

The kernels have hard preconditions the reference does not:

==========  ================================================================
GEMM        CUDA, fp16/bf16, and ``in_features`` **and** ``out_features``
            8-aligned.  CUTLASS 2.x alignment-8 iterators reject anything
            else outright — ``oasr.gemm`` on ``N=8404`` (Paraformer's vocab)
            raises rather than degrading, so the check has to happen here.
            Plus a **work floor**: see :data:`GEMM_MIN_MACS`.
Norm        CUDA, fp32/fp16/bf16, **contiguous**.  The kernels address rows as
            ``input + row * hidden_size``, so a transposed activation (the
            layout a conv encoder's ``transpose(1, 2)`` leaves behind) has to
            take the torch path.  Any hidden size is fine: the launchers drop
            to the scalar kernel when it is not a multiple of the vector width.
FMHA        CUDA, fp16/bf16 for the CuteDSL kernel; ``oasr.fmha`` itself
            degrades to SDPA otherwise, so only the ``torch`` override
            below needs handling here.
==========  ================================================================

``OASR_LAYERS_BACKEND`` overrides the choice process-wide:

``auto`` (default)
    Kernel where it can run, torch otherwise.
``torch``
    Never call a kernel — the debugging fallback the layer waist is supposed
    to have.  A numerical difference that survives ``OASR_LAYERS_BACKEND=torch``
    is not the kernels' fault.
``oasr``
    Kernel or raise, for GEMM and norm.  Nothing degrades silently, which is
    how you *prove* a model reaches the kernels instead of assuming it —
    ``tests/test_layer_waist.py`` runs every registered architecture under it.
    Attention is deliberately exempt: it has its own ``OASR_ATTN_BACKEND``
    switch, and several legitimate mask shapes (left padding, causal) are
    SDPA-only by construction rather than by omission.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

#: dtypes the CUTLASS GEMM kernels accept.
GEMM_DTYPES = (torch.float16, torch.bfloat16)
#: dtypes the handwritten norm kernels accept.
NORM_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
#: CUTLASS 2.x alignment-8 iterators: both GEMM free dimensions must divide by 8.
GEMM_ALIGNMENT = 8

#: Work floor (multiply-accumulates) below which ``oasr.gemm`` loses to
#: ``F.linear`` on *launch overhead* rather than on kernel quality.
#:
#: Reaching the CUTLASS kernel costs a fixed ~20 µs of Python per call — the
#: shape-aware heuristic lookup, a workspace allocation, two reshapes, the
#: ``@oasr_api`` wrapper.  Measured on an RTX 5090, ``Linear(384, 384)`` in
#: fp16: 22 µs through the kernel vs 15 µs through ``F.linear``, and the two
#: only converge around 4e9 MACs.  A batched encoder forward amortizes it
#: (Conformer and Zipformer measure within ±3% either way end to end); an
#: **eager autoregressive decode step** does not — whisper-tiny's 30-step loop
#: measured 75.0 ms on the kernel against 47.1 ms on torch, 1.59×, which is
#: the whole of a 1.4× end-to-end regression.
#:
#: The floor must **not** be conditioned on ``is_current_stream_capturing()``,
#: tempting as that is — under capture the dispatch cost is paid once and
#: replayed for free, so the kernel looks unconditionally right.  But a graph's
#: contract is to reproduce the eager result, and a capture-dependent branch
#: silently breaks it: the captured path picked CUTLASS while the eager path
#: picked cuBLAS for the same shape, and the one-ulp fp16 difference reached
#: the decoder as *different tokens*
#: (``test_streaming_graph_capture_is_token_identical_to_eager``).  Any future
#: refinement of this rule has to stay a pure function of the call.
#:
#: Placement has ~1.5 orders of magnitude of room, so it is not a knife edge:
#: whisper-tiny's decode step is 2.4e6 MACs at ``B=16`` and Qwen2-Audio-7B's is
#: 5.1e7 at ``B=4`` — and where the 7B does fall below (``B=1``), torch is the
#: right answer for the same reason.  The real fix is cheaper dispatch or
#: CUDA-graph capture of the AR step; this is the honest interim.
GEMM_MIN_MACS = 1 << 24

_VALID_MODES = ("auto", "torch", "oasr")

_MODE: Optional[str] = None


def layers_backend() -> str:
    """Resolved ``OASR_LAYERS_BACKEND`` mode (cached after the first read)."""
    global _MODE
    if _MODE is None:
        mode = os.environ.get("OASR_LAYERS_BACKEND", "auto").strip().lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"OASR_LAYERS_BACKEND={mode!r} is not one of {_VALID_MODES}",
            )
        _MODE = mode
    return _MODE


def set_layers_backend(mode: str) -> None:
    """Override the mode for this process (tests, benchmarks, A/B switches)."""
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


def _refuse(what: str, reason: str) -> bool:
    """Honour strict mode: either explain the fallback or take it silently."""
    if layers_backend() == "oasr":
        raise RuntimeError(
            f"OASR_LAYERS_BACKEND=oasr but the {what} kernel cannot run: {reason}. "
            "Set OASR_LAYERS_BACKEND=auto to fall back to torch."
        )
    return False


def use_gemm_kernel(x: torch.Tensor, in_features: int, out_features: int) -> bool:
    """Can (and should) this projection go through ``oasr.gemm``?"""
    mode = layers_backend()
    if mode == "torch":
        return False
    if not x.is_cuda:
        return _refuse("GEMM", f"input is on {x.device}, kernels are CUDA-only")
    if x.dtype not in GEMM_DTYPES:
        return _refuse("GEMM", f"dtype {x.dtype} is not one of {GEMM_DTYPES}")
    if in_features % GEMM_ALIGNMENT or out_features % GEMM_ALIGNMENT:
        return _refuse(
            "GEMM",
            f"({in_features} -> {out_features}) is not {GEMM_ALIGNMENT}-aligned on both axes",
        )
    if mode == "oasr":
        # Strict mode answers "can the kernel run here", not "should it": the
        # work floor below is a performance policy, and applying it would make
        # the reach check under-report.
        return True
    if x.numel() * out_features < GEMM_MIN_MACS:
        return False
    return True


def use_norm_kernel(x: torch.Tensor) -> bool:
    """Can (and should) this normalization go through the OASR norm kernels?"""
    mode = layers_backend()
    if mode == "torch":
        return False
    if not x.is_cuda:
        return _refuse("norm", f"input is on {x.device}, kernels are CUDA-only")
    if x.dtype not in NORM_DTYPES:
        return _refuse("norm", f"dtype {x.dtype} is not one of {NORM_DTYPES}")
    if not x.is_contiguous():
        return _refuse("norm", f"input is not contiguous (strides {tuple(x.stride())})")
    return True


def use_fmha_kernel() -> bool:
    """Is ``oasr.fmha`` allowed at all?  (It picks cute vs SDPA internally.)"""
    return layers_backend() != "torch"


__all__ = [
    "GEMM_ALIGNMENT",
    "GEMM_DTYPES",
    "NORM_DTYPES",
    "layers_backend",
    "layers_backend_override",
    "set_layers_backend",
    "use_fmha_kernel",
    "use_gemm_kernel",
    "use_norm_kernel",
]
