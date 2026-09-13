# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Work models and derived metrics -- one implementation of each.

The division (``flops / seconds``) lives in :mod:`benchmarks.core.schema`, which
derives ``tflops`` and ``bandwidth_tb_s`` from whatever a family declares.  What
lives here is the part that genuinely differs per op: *how much work is it*, and
*how many bytes does it touch*.

The previous suite had six copies of the GEMM FLOP expression, six of the
bandwidth division, and four percentile rules that disagreed.  Two of those
copies had already drifted apart: ``bench_conv2d.py`` omitted the ``groups``
divisor that ``routines/conv.py`` applied, so the two reported FLOPs differing
by a factor of ``groups`` for the same grouped convolution.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile, ``pct`` in ``[0, 100]``.

    Ranked over ``len - 1`` so that ``pct=0`` is the minimum and ``pct=100`` the
    maximum.  Two of the four rules this replaces used ``int(q * len)``, which
    is biased one sample high at the top of the range and needs a clamp to avoid
    indexing off the end.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def rtfx(audio_s: float, wall_s: float) -> float:
    """Real-time factor as ``audio / wall``: higher is faster.

    The one definition.  A column named ``rtf`` used to mean ``wall / audio`` in
    the engine harness and ``audio / wall`` in the service harness -- its own
    reciprocal, depending on which file wrote it.
    """
    return audio_s / wall_s if wall_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def dtype_size(dtype: torch.dtype) -> int:
    """Bytes per element of *dtype*."""
    return torch.tensor([], dtype=dtype).element_size()


def elementwise_bytes(numel: int, dtype: torch.dtype, reads: int = 1, writes: int = 1) -> int:
    """Bytes moved by a pointwise op reading *reads* and writing *writes* tensors."""
    return numel * dtype_size(dtype) * (reads + writes)


# ---------------------------------------------------------------------------
# Work models
# ---------------------------------------------------------------------------


def gemm_flops(m: int, n: int, k: int) -> float:
    """``2 M N K`` -- one multiply and one add per inner-product term."""
    return 2.0 * m * n * k


def bmm_flops(b: int, m: int, n: int, k: int) -> float:
    return 2.0 * b * m * n * k


def group_gemm_flops(problem_sizes: Sequence[Tuple[int, int, int]]) -> float:
    """Sum over the group's problems, which need not share a shape."""
    return sum(gemm_flops(m, n, k) for m, n, k in problem_sizes)


def conv1d_flops(
    batch: int, out_len: int, out_ch: int, in_ch: int, kernel: int, groups: int = 1
) -> float:
    return 2.0 * batch * out_len * out_ch * (in_ch // groups) * kernel


def conv2d_output_hw(h: int, w: int, r: int, s: int, pad: int, stride: int) -> Tuple[int, int]:
    return (h + 2 * pad - r) // stride + 1, (w + 2 * pad - s) // stride + 1


def conv2d_flops(
    batch: int,
    h: int,
    w: int,
    in_ch: int,
    out_ch: int,
    r: int,
    s: int,
    pad: int = 0,
    stride: int = 1,
    groups: int = 1,
) -> float:
    """Implicit-GEMM FLOPs, with the ``groups`` divisor applied.

    Omitting that divisor -- as one of the two previous copies did -- overstates
    a grouped convolution by exactly ``groups``.
    """
    oh, ow = conv2d_output_hw(h, w, r, s, pad, stride)
    return 2.0 * batch * oh * ow * out_ch * (in_ch // groups) * r * s


def fmha_flops(
    batch: int, heads: int, t_q: int, t_k: int, head_dim: int, causal: bool = False
) -> float:
    """``4 B H Tq Tk D`` -- two GEMMs (QK and SV), each ``2 M N K``.

    Halved when causal: a causal kernel evaluates roughly the lower triangle, so
    charging it the full rectangle overstates its achieved TFLOPS by ~2x.
    """
    flops = 4.0 * batch * heads * t_q * t_k * head_dim
    return flops * 0.5 if causal else flops


def recurrent_flops(
    batch: int, seq: int, input_size: int, hidden: int, layers: int, gates: int
) -> float:
    """Per-layer input and recurrent projections, summed over layers."""
    total = 0.0
    for layer in range(layers):
        layer_input = input_size if layer == 0 else hidden
        total += 2.0 * gates * batch * seq * hidden * (layer_input + hidden)
    return total


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

REFCHECK_PASS = "pass"
REFCHECK_FAIL = "fail"
REFCHECK_SKIP = "skip"


def check_close(
    actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2
) -> Tuple[bool, float]:
    """Compare two tensors in fp32.  Returns ``(passed, max_abs_diff)``."""
    a = actual.float()
    b = expected.float()
    diff = (a - b).abs()
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol)), float(diff.max().item())
