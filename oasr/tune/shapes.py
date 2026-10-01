# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Dynamic-dimension buckets: the one function tuning and lookup share.

Production systems tune dynamic dimensions on buckets and must use the *same*
map at tuning time and at lookup time -- TensorRT-LLM rounds M to a power of
two in both places; FlashInfer's bug #5449 is what happens when the two
disagree.  OASR's ladder is finer where ASR lives:

* step 8 up to 128 -- streaming cohorts (``M = cohort x chunk_frames``) and AR
  decode rows live here, where one tile height is the difference between one
  wave and two;
* ratio sqrt(2) up to 1024;
* ratio 2 above that, to 2**20.

Buckets round **up**: a bucket's value is its upper edge, and a result tuned
for it is only applied to sizes at or below the size it was measured at.
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Sequence, Tuple

__all__ = ["BUCKET_EDGES", "bucket", "ladder", "DYNAMIC_DIMS", "bucket_shape_sig"]


def _edges() -> List[int]:
    edges = list(range(8, 129, 8))
    for v in (181, 256, 362, 512, 724, 1024):
        edges.append(v)
    v = 2048
    while v <= 1 << 20:
        edges.append(v)
        v *= 2
    return edges


#: Ascending bucket upper edges.
BUCKET_EDGES: Tuple[int, ...] = tuple(_edges())


def bucket(m: int) -> int:
    """The bucket *m* belongs to: the smallest edge ``>= m``.

    Above the last edge the size is its own bucket, rounded up to a power of
    two, so no size is ever mapped *down*.
    """
    m = max(1, int(m))
    i = bisect.bisect_left(BUCKET_EDGES, m)
    if i < len(BUCKET_EDGES):
        return BUCKET_EDGES[i]
    return 1 << (m - 1).bit_length()


def ladder(m_max: int, m_min: int = 1) -> List[int]:
    """Every bucket edge in ``[bucket(m_min), bucket(m_max)]``: fill points for a sweep."""
    lo, hi = bucket(m_min), bucket(m_max)
    out = [e for e in BUCKET_EDGES if lo <= e <= hi]
    if hi not in out:
        out.append(hi)
    return out


#: ``(family, op) -> indices of the autotuner shape_sig that are dynamic``.
#: Everything not listed is static and matched exactly.
DYNAMIC_DIMS: Dict[Tuple[str, str], Tuple[int, ...]] = {
    ("gemm", "gemm"): (0,),  # (M, N, K)
    ("gemm", "gemm_activation"): (0,),
    ("gemm", "gemm_log_softmax"): (0,),
    ("gemm", "bmm"): (0, 1),  # (batch, M, N, K)
    ("gemm", "group_gemm"): (0,),  # (L, groups, N, K)
    ("conv", "conv1d"): (0, 1),  # (B, T, Cin, Cout, k, pad, stride, dil)
    ("conv", "conv1d_activation"): (0, 1),
    ("conv", "conv2d"): (0, 1),  # (N, H, W, C, K, R, S, ...) -- H is time
    ("conv", "conv2d_activation"): (0, 1),
    ("recurrent", "lstm"): (0, 1),  # (T, B, H, has_bias)
    ("recurrent", "rnn_tanh"): (0, 1),
    ("recurrent", "rnn_relu"): (0, 1),
}


def bucket_shape_sig(family: str, op: str, shape_sig: Sequence[int]) -> Tuple[int, ...]:
    """*shape_sig* with every dynamic dimension replaced by its bucket."""
    dyn = DYNAMIC_DIMS.get((family, op), ())
    return tuple(bucket(v) if i in dyn else int(v) for i, v in enumerate(shape_sig))
