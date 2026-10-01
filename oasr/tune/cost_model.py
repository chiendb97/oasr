# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""An analytic, calibrated cost model for the CUTLASS 2.x GEMM lane.

The decomposition the dynamic-shape GEMM literature converged on (DietCode's
occupancy term, MikPoly's wave x per-kernel pipeline curve, TRT-LLM's
wave-quantisation score), in the form that fits OASR's configs::

    T(c; M, N, K) = W(c; M, N) * r(c; M, N) * (a_c + b_c * k_iters(c; K))
                                                # waves x resident CTAs x per-CTA-alone time
                  + R_c(M, N, split)                               # split-K reduction
                  + launches(c) * L_graph                          # in-graph node cost
    T            >= bytes(M, N, K, c) / DRAM bandwidth              # memory floor

    W = ceil(ceil(M/bm) * ceil(N/bn) * split / (SMs * occupancy))
    occupancy = min(smem_budget // smem(c), threads_per_SM // threads(c), 24)

``(a_c, b_c)`` start from an uncalibrated prior derived from tile arithmetic and
the device's measured throughput, and are fitted per config by least squares
on the measurement log every ``oasr tune build`` appends to.  Three uses:

1. **pruning** -- benchmark only ``top_k(rank) U forced``;
2. **runtime fallback** -- the argmin over the *compiled* set for a shape no
   tuning file covers, resolved once per shape and memoised;
3. **boundary prior** -- where two configs' curves cross.

Known weakness, stated rather than hidden: analytic models are least accurate
below one wave, where launch and issue dominate -- OASR's streaming regime --
which is why the tuner never lets the model decide a must-tune point alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["GemmCostModel", "Features", "features_of", "spearman", "recall_at_k"]


@dataclass(frozen=True)
class Features:
    """What the model needs to know about one config (from its tuning-DB params)."""

    bm: int
    bn: int
    bk: int
    stages: int
    threads: int
    split: int
    parallel_split_k: bool
    stream_k: bool

    @property
    def smem(self) -> int:
        return self.stages * (self.bm + self.bn) * self.bk * 2

    @property
    def launches(self) -> int:
        return 2 if (self.parallel_split_k or self.stream_k) else 1


def features_of(params: Dict[str, Any]) -> Optional[Features]:
    """Features of a CUTLASS 2.x config's params, or ``None`` for other kinds/lanes."""
    if params.get("kind") != "cutlass" or params.get("lane") != "sm80_mma":
        return None
    tile = [int(v) for v in params["tile"]]
    warp = [int(v) for v in params["warp"]]
    warps = (tile[0] // warp[0]) * (tile[1] // warp[1]) * max(1, tile[2] // warp[2])
    return Features(
        bm=tile[0],
        bn=tile[1],
        bk=tile[2],
        stages=int(params["stages"]),
        threads=32 * warps,
        split=max(1, int(params.get("split_k", 1))),
        parallel_split_k=bool(params.get("parallel_split_k", False)),
        stream_k=bool(params.get("stream_k", False)),
    )


def _ceil(a: float, b: float) -> int:
    return int(-(-a // b))


#: Operand bytes one SM can pull per second through L2/L1 (~64 B/clk at ~2.4 GHz):
#: the ceiling on a lone CTA's feed, whatever DRAM delivers to the whole device.
_SM_BYTES_PER_S = 150e9


@dataclass
class GemmCostModel:
    """``predict_ms(params, M, N, K)`` for one device, with per-config coefficients."""

    num_sms: int
    smem_budget: int
    max_threads_per_sm: int
    dram_gbps: float
    tensor_tflops: float
    launch_us_graph: float = 1.0
    #: ``config_id -> (a_us, b_us)`` fitted from measurements.
    coeffs: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    #: Rows each config's coefficients were fitted from.
    fitted_rows: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def for_arch(cls, arch) -> "GemmCostModel":
        return cls(
            num_sms=arch.num_sms,
            smem_budget=arch.smem_budget(),
            max_threads_per_sm=arch.max_threads_per_sm,
            dram_gbps=float(arch.dram_gbps or 1000.0),
            tensor_tflops=float(arch.tensor_tflops or 100.0),
            launch_us_graph=float(arch.launch_us_graph or 1.0),
        )

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def occupancy(self, f: Features) -> int:
        by_smem = self.smem_budget // f.smem if f.smem > 0 else 24
        by_threads = self.max_threads_per_sm // max(f.threads, 32)
        return max(1, min(by_smem, by_threads, 24))

    def waves(self, f: Features, M: int, N: int) -> int:
        ctas = _ceil(M, f.bm) * _ceil(N, f.bn) * f.split
        return _ceil(ctas, self.num_sms * self.occupancy(f))

    def k_iters(self, f: Features, K: int) -> int:
        return _ceil(K, f.split * f.bk)

    def residency(self, f: Features, M: int, N: int) -> int:
        """CTAs co-resident on one SM in the busiest wave.

        Full waves hold ``occupancy`` CTAs per SM; a single partial wave holds
        ``ceil(ctas / SMs)``.  A small-M problem is the second kind -- 32 CTAs on
        170 SMs is one per SM -- and charging it full occupancy's sharing
        overpredicted every thin tile 3-4x (48-70 us predicted, 15.6 measured).
        """
        ctas = _ceil(M, f.bm) * _ceil(N, f.bn) * f.split
        return max(1, min(self.occupancy(f), _ceil(ctas, self.num_sms)))

    def prior(self, f: Features) -> Tuple[float, float]:
        """Uncalibrated per-CTA-*alone* ``(a_us, b_us)`` from tile arithmetic.

        A K iteration of one CTA is ``2*bm*bn*bk`` flops and ``(bm+bn)*bk*2``
        bytes; alone on an SM it gets the SM's share of tensor throughput and
        at most ``_SM_BYTES_PER_US`` of operand bandwidth.  ``a`` is pipeline
        fill (``stages`` iterations), the epilogue's output write, and a fixed
        microsecond.  Co-resident CTAs multiply the time (:meth:`residency`).
        """
        per_sm_flops = self.tensor_tflops * 1e12 / self.num_sms
        per_sm_bytes = min(self.dram_gbps * 1e9 / self.num_sms * 4.0, _SM_BYTES_PER_S)
        compute = 2.0 * f.bm * f.bn * f.bk / per_sm_flops * 1e6
        memory = (f.bm + f.bn) * f.bk * 2.0 / per_sm_bytes * 1e6
        b = max(compute, memory)
        a = f.stages * b + f.bm * f.bn * 2.0 / per_sm_bytes * 1e6 + 1.0
        return a, b

    def predict_ms(self, cid: str, params: Dict[str, Any], M: int, N: int, K: int) -> float:
        f = features_of(params)
        if f is None:
            return math.inf
        if f.split > 1 and f.split > _ceil(K, f.bk):
            # More K partitions than K tiles: the surplus partitions do no
            # mainloop work and still take their turn in the serialised (or
            # reduced) epilogue -- measured 18 us against 2.2 us for the plain
            # tile at K=256, split 16.  Never a candidate.
            return math.inf
        a, b = self.coeffs.get(cid) or self.prior(f)
        W = self.waves(f, M, N) * self.residency(f, M, N)
        t_us = W * (a + b * self.k_iters(f, K)) + f.launches * self.launch_us_graph
        if f.parallel_split_k:
            t_us += (M * N * f.split * 4 + M * N * 2) / (self.dram_gbps * 1e3)
        floor_us = ((M * K + N * K + M * N) * 2) / (self.dram_gbps * 1e3)
        return max(t_us, floor_us) / 1e3

    # ------------------------------------------------------------------
    # Use
    # ------------------------------------------------------------------

    def rank(
        self, configs: Dict[str, Dict[str, Any]], M: int, N: int, K: int
    ) -> List[Tuple[str, float]]:
        scored = [(cid, self.predict_ms(cid, p, M, N, K)) for cid, p in configs.items()]
        return sorted((s for s in scored if math.isfinite(s[1])), key=lambda s: s[1])

    def top_k(
        self, configs: Dict[str, Dict[str, Any]], M: int, N: int, K: int, k: int
    ) -> List[str]:
        return [cid for cid, _ in self.rank(configs, M, N, K)[:k]]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, rows: Iterable[Dict[str, Any]], min_rows: int = 3) -> int:
        """Fit ``(a, b)`` per config from measurement rows; returns configs fitted.

        A row is ``{"config": id, "params": {...}, "M", "N", "K", "median_ms"}``.
        The model is linear in ``(a, b)`` given the shape --
        ``t - launches * L = W * a + W * k * b`` -- so each config is a
        least-squares fit, clipped to non-negative coefficients.  Rows are
        weighted by ``1 / t^2``, i.e. the fit minimises *relative* error: an
        absolute fit is dominated by the many-wave large-M rows and loses the
        small-M regime -- exactly where ranking is hardest and OASR's streaming
        shapes live.
        """
        by_cfg: Dict[str, List[Tuple[float, float, float]]] = {}
        for r in rows:
            f = features_of(r["params"])
            if f is None:
                continue
            M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
            W = self.waves(f, M, N) * self.residency(f, M, N)
            k = self.k_iters(f, K)
            t_us = float(r["median_ms"]) * 1e3 - f.launches * self.launch_us_graph
            if f.parallel_split_k:
                t_us -= (M * N * f.split * 4 + M * N * 2) / (self.dram_gbps * 1e3)
            if f.split > 1 and f.split > _ceil(K, f.bk):
                continue  # infeasible by construction (see predict_ms)
            by_cfg.setdefault(str(r["config"]), []).append((float(W), float(W * k), t_us))
        fitted = 0
        for cid, pts in by_cfg.items():
            if len(pts) < min_rows:
                continue
            ws = [1.0 / max(t, 1e-3) ** 2 for _, _, t in pts]
            sxx = sum(w * x * x for w, (x, _, _) in zip(ws, pts))
            sxy = sum(w * x * y for w, (x, y, _) in zip(ws, pts))
            syy = sum(w * y * y for w, (_, y, _) in zip(ws, pts))
            sxt = sum(w * x * t for w, (x, _, t) in zip(ws, pts))
            syt = sum(w * y * t for w, (_, y, t) in zip(ws, pts))
            det = sxx * syy - sxy * sxy
            if det <= 1e-12:
                continue
            a = (sxt * syy - syt * sxy) / det
            b = (syt * sxx - sxt * sxy) / det
            if b < 0:
                b = 0.0
                a = sxt / sxx if sxx else 0.0
            if a < 0:
                a = 0.0
                b = syt / syy if syy else 0.0
            self.coeffs[cid] = (max(a, 0.0), max(b, 0.0))
            self.fitted_rows[cid] = len(pts)
            fitted += 1
        return fitted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        return {
            "num_sms": self.num_sms,
            "smem_budget": self.smem_budget,
            "max_threads_per_sm": self.max_threads_per_sm,
            "dram_gbps": self.dram_gbps,
            "tensor_tflops": self.tensor_tflops,
            "launch_us_graph": self.launch_us_graph,
            "coeffs": {k: list(v) for k, v in self.coeffs.items()},
            "fitted_rows": dict(self.fitted_rows),
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "GemmCostModel":
        m = cls(
            **{
                k: d[k]
                for k in (
                    "num_sms",
                    "smem_budget",
                    "max_threads_per_sm",
                    "dram_gbps",
                    "tensor_tflops",
                    "launch_us_graph",
                )
            }
        )
        m.coeffs = {k: (float(v[0]), float(v[1])) for k, v in d.get("coeffs", {}).items()}
        m.fitted_rows = {k: int(v) for k, v in d.get("fitted_rows", {}).items()}
        return m


# =============================================================================
# Ranking quality
# =============================================================================


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation of two equal-length sequences."""
    n = len(a)
    if n < 2:
        return 1.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else 1.0


def recall_at_k(predicted: Sequence[str], measured_best: str, k: int) -> bool:
    """Whether the measured best config is among the model's top *k*."""
    return measured_best in list(predicted)[:k]
