# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Selection telemetry: what production resolved, and what it missed.

The runtime half of the tuning loop.  Every shape the resolvers could not serve
from a tuning entry is already counted (``oasr.jit.gemm.rule_misses``, the
``model``/``default`` tier counts in ``oasr.tune.database``); this module turns
those counters into

* :func:`export_misses` -- a census fragment (``oasr tune census --capture``-
  compatible points) of the shapes a real workload issued with no entry, which
  is the next census: the tuner measures what production actually missed;
* :func:`summary` -- the numbers ``format_gap_report`` and
  ``ASREngine.tuning_report`` print.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["export_misses", "summary"]


def export_misses() -> Dict[str, Any]:
    """This process's untuned GEMM shapes as census points.

    A miss records its call count and M range; the points are the range's two
    ends (a miss at one M is one point).  Every point is ``eager_fraction=1``
    because nothing is known about capture here -- the next build can refine it
    with a real census.
    """
    from oasr.jit import gemm as jg
    from oasr.tune import database

    points: List[Dict[str, Any]] = []
    for (op, N, K), (calls, m_lo, m_hi) in sorted(jg.rule_misses().items()):
        for m in sorted({m_lo, m_hi}):
            points.append(
                {
                    "op": op,
                    "N": N,
                    "K": K,
                    "dtype": "bfloat16",
                    "batch": 1,
                    "M": int(m),
                    "calls": float(calls),
                    "weight": float(calls),
                    "must": False,
                    "eager_fraction": 1.0,
                    "keys": ["telemetry"],
                }
            )
    return {
        "version": 1,
        "working_set_bytes": 0,
        "provenance": {
            "source": "telemetry",
            "tiers": {"|".join(k): v for k, v in database.tier_counts().items()},
        },
        "points": points,
    }


def summary() -> Dict[str, Any]:
    """Tier counts and miss counts, as plain data."""
    from oasr.jit import gemm as jg
    from oasr.tune import database

    tiers: Dict[str, Dict[str, int]] = {}
    for (family, op, tier), n in database.tier_counts().items():
        tiers.setdefault(f"{family}.{op}", {})[tier] = n
    return {
        "tiers": tiers,
        "misses": len(jg.rule_misses()),
        "inactive": dict(jg.heuristic_inactive()),
    }
