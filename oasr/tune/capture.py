# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Capture the real GEMM shapes a workload issues through the OASR functional API.

The analytic shape derivation in ``scripts/analyze_asr_cutlass_configs.py`` is
unreliable — it mis-models operator fusion, assumes attention projections hit the
OASR GEMM path (they are ``torch.nn.Linear``), and fabricates BMM problems that
actually go through ``oasr.fmha``.  The only trustworthy source of the shapes that
reach the OASR-tunable CUTLASS path is the workload itself.

This module wraps the functional entry points (:func:`oasr.gemm`,
:func:`oasr.gemm_activation`, :func:`oasr.bmm`, :func:`oasr.group_gemm`) with thin
recorders that log ``(op, M, N, K, dtype)`` plus a call count and a FLOP weight,
then forward to the original.  Overhead is one dict update per call — negligible
next to a GEMM launch.

Two ways to drive it:

* **Programmatic** — wrap a run in the context manager::

      from oasr.tune.capture import capture_gemm_shapes
      with capture_gemm_shapes() as rec:
          engine.transcribe(...)
      rec.to_json("shapes.json")

* **Env autostart** — set ``OASR_CAPTURE_GEMM=/path/shapes.json`` before launching
  any script that imports ``oasr`` (e.g. ``benchmarks/bench_engine.py``); capture
  starts at import and dumps at process exit.  Run streaming with
  ``--cuda-graphs off`` so every chunk re-enters the Python wrapper and call counts
  reflect true frequency — captured CUDA graphs replay without re-entering it
  (shapes are still observed once at capture time, only the weight under-counts).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("oasr.tune")

# Functional entry points patched during capture (attribute names on the ``oasr``
# package, which is how the layers call them: ``oasr.gemm(...)`` etc.).
_PATCH_NAMES = ("gemm", "gemm_activation", "bmm", "group_gemm")


def _dtype_str(dtype) -> str:
    return str(dtype).replace("torch.", "")


@dataclass
class _ShapeStat:
    """Accumulated stats for one ``(op, N, K, dtype[, batch])`` key."""

    op: str
    N: int
    K: int
    dtype: str
    batch: int = 1
    call_count: int = 0
    total_flops: float = 0.0
    m_counts: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "N": self.N,
            "K": self.K,
            "dtype": self.dtype,
            "batch": self.batch,
            "call_count": self.call_count,
            "total_flops": self.total_flops,
            "m_counts": {str(m): c for m, c in sorted(self.m_counts.items())},
        }


class GemmShapeRecorder:
    """Thread-safe accumulator of observed GEMM shapes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[Tuple[str, int, int, str, int], _ShapeStat] = {}

    def record(self, op: str, M: int, N: int, K: int, dtype: str, batch: int = 1) -> None:
        if M <= 0 or N <= 0 or K <= 0:
            return
        key = (op, N, K, dtype, batch)
        flops = 2.0 * M * N * K * batch
        with self._lock:
            st = self._stats.get(key)
            if st is None:
                st = _ShapeStat(op=op, N=N, K=K, dtype=dtype, batch=batch)
                self._stats[key] = st
            st.call_count += 1
            st.total_flops += flops
            st.m_counts[int(M)] += 1

    def aggregate(self) -> List[_ShapeStat]:
        with self._lock:
            return sorted(
                self._stats.values(), key=lambda s: s.total_flops, reverse=True
            )

    def to_json(self, path: str) -> None:
        data = {"version": 1, "stats": [s.to_dict() for s in self.aggregate()]}
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        logger.info(
            "[capture] wrote %d shape groups to %s", len(self._stats), path
        )

    @staticmethod
    def load_json(path: str) -> List[_ShapeStat]:
        with open(path) as f:
            data = json.load(f)
        out: List[_ShapeStat] = []
        for d in data["stats"]:
            st = _ShapeStat(
                op=d["op"], N=d["N"], K=d["K"], dtype=d["dtype"], batch=d.get("batch", 1)
            )
            st.call_count = d["call_count"]
            st.total_flops = d["total_flops"]
            st.m_counts = Counter({int(m): c for m, c in d["m_counts"].items()})
            out.append(st)
        return out

    def __len__(self) -> int:
        return len(self._stats)


def _shapes_of(op: str, args, kwargs) -> Optional[Tuple[int, int, int, int]]:
    """Extract ``(M, N, K, batch)`` from a functional call's args.

    Layout matches :mod:`oasr.gemm`: ``A`` is operand 0, ``B`` (the ``[N, K]``
    weight, or ``[batch, N, K]`` for bmm/group_gemm) is operand 1.
    """
    A = args[0] if args else kwargs.get("A")
    B = args[1] if len(args) > 1 else kwargs.get("B")
    if A is None or B is None:
        return None
    K = int(A.shape[-1])
    if op == "bmm":
        # A: [batch, M, K], B: [batch, N, K]
        batch = int(A.shape[0])
        M = int(A.shape[1])
        N = int(B.shape[1])
        return M, N, K, batch
    if op == "group_gemm":
        # A: [L, K], B: [Bcount, N, K]
        M = int(A.numel() // K)
        N = int(B.shape[1])
        return M, N, K, 1
    # gemm / gemm_activation — A: [*, K], B: [N, K]
    M = int(A.numel() // K)
    N = int(B.shape[0])
    return M, N, K, 1


@contextlib.contextmanager
def capture_gemm_shapes(recorder: Optional[GemmShapeRecorder] = None):
    """Monkeypatch the OASR functional GEMM entries to record shapes for the block."""
    import oasr

    rec = recorder if recorder is not None else GemmShapeRecorder()
    originals = {}
    try:
        for name in _PATCH_NAMES:
            orig = getattr(oasr, name, None)
            if orig is None:
                continue
            originals[name] = orig

            def make_wrapper(op_name, fn):
                def wrapper(*args, **kwargs):
                    shp = _shapes_of(op_name, args, kwargs)
                    if shp is not None:
                        A = args[0] if args else kwargs.get("A")
                        M, N, K, batch = shp
                        rec.record(op_name, M, N, K, _dtype_str(A.dtype), batch)
                    return fn(*args, **kwargs)

                return wrapper

            setattr(oasr, name, make_wrapper(name, orig))
        yield rec
    finally:
        for name, orig in originals.items():
            setattr(oasr, name, orig)


def maybe_autostart() -> None:
    """If ``OASR_CAPTURE_GEMM`` is set, start capture and dump on process exit.

    Cheap no-op when the env var is absent; called once at the end of
    ``oasr/__init__.py``.
    """
    path = os.environ.get("OASR_CAPTURE_GEMM")
    if not path:
        return
    import atexit

    rec = GemmShapeRecorder()
    cm = capture_gemm_shapes(rec)
    cm.__enter__()
    logger.warning("[capture] OASR_CAPTURE_GEMM active -> %s", path)

    def _dump():
        try:
            cm.__exit__(None, None, None)
        finally:
            rec.to_json(path)

    atexit.register(_dump)
