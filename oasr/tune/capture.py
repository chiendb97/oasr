# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Capture GEMM shapes and frequency from a real workload.

Functional entry points are wrapped to record shape, dtype, calls, and FLOP
weight. Graph replay does not re-enter wrappers, so disable graphs when call
frequency matters. ``OASR_CAPTURE_GEMM`` enables process-wide capture.
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
# package, which is how most layers call them: ``oasr.gemm(...)`` etc.).
# ``gemm_log_softmax`` is the CTC head (``oasr.layers.ctc.CtcProjection``).
_PATCH_NAMES = ("gemm", "gemm_activation", "bmm", "group_gemm", "gemm_log_softmax")

# ...but not every caller goes through the package attribute.  Four in-tree call
# sites do ``from oasr.functionals.gemm import ...`` inside a function body, so
# rebinding ``oasr.gemm`` alone left them invisible — and an invisible shape gets
# no tuned rule.  That is how the recurrent input projection (K=640, N=4H) came
# to sit on ``GEMM_DEFAULT``, a fixed 128x128x64 tile that is 2-5x slower than
# cuBLAS at the M this path runs.  Patching the defining module as well covers a
# function-body import, because that import resolves at call time.
#
# ``_dispatch_gemm`` / ``_dispatch_gemm_activation`` are the shape-aware
# dispatchers that ``oasr.functionals.conv`` calls directly for its im2col path;
# they take ``(out, A, B, C, [activation,] N, K, M)``, so their shapes are read
# from the explicit arguments rather than from operand geometry.
_MODULE_PATCH_NAMES = _PATCH_NAMES
_DISPATCH_PATCH_SPECS = {
    # name: (index of N, index of K, index of M) in the positional signature
    "_dispatch_gemm": (4, 5, 6),
    "_dispatch_gemm_activation": (5, 6, 7),
}


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
            return sorted(self._stats.values(), key=lambda s: s.total_flops, reverse=True)

    def to_json(self, path: str) -> None:
        data = {"version": 1, "stats": [s.to_dict() for s in self.aggregate()]}
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        logger.info("[capture] wrote %d shape groups to %s", len(self._stats), path)

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

    Layout matches :mod:`oasr.functionals.gemm`: ``A`` is operand 0, ``B`` (the ``[N, K]``
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


#: Re-entrancy guard.  ``gemm()`` reaches ``_dispatch_gemm`` through its own
#: module globals, so with both patched one call would be recorded twice.  Only
#: the outermost wrapper records; the depth is per-thread because the engine
#: runs the front-end dispatcher on its own.
_depth = threading.local()


@contextlib.contextmanager
def _outermost():
    d = getattr(_depth, "n", 0)
    _depth.n = d + 1
    try:
        yield d == 0
    finally:
        _depth.n = d


def _make_wrapper(rec, op_name, fn):
    def wrapper(*args, **kwargs):
        with _outermost() as record:
            if record:
                shp = _shapes_of(op_name, args, kwargs)
                if shp is not None:
                    A = args[0] if args else kwargs.get("A")
                    M, N, K, batch = shp
                    rec.record(op_name, M, N, K, _dtype_str(A.dtype), batch)
            return fn(*args, **kwargs)

    return wrapper


def _make_dispatch_wrapper(rec, op_name, fn, idx):
    """Wrap a ``_dispatch_*`` entry, whose N/K/M arrive as explicit arguments."""
    n_i, k_i, m_i = idx

    def wrapper(*args, **kwargs):
        with _outermost() as record:
            if record:
                try:
                    rec.record(
                        op_name,
                        int(args[m_i]),
                        int(args[n_i]),
                        int(args[k_i]),
                        _dtype_str(args[1].dtype),
                        1,
                    )
                except (IndexError, TypeError, AttributeError):
                    pass
            return fn(*args, **kwargs)

    return wrapper


@contextlib.contextmanager
def capture_gemm_shapes(recorder: Optional[GemmShapeRecorder] = None):
    """Monkeypatch the OASR functional GEMM entries to record shapes for the block."""
    import oasr
    import oasr.functionals.gemm as _fg

    rec = recorder if recorder is not None else GemmShapeRecorder()
    originals: List[Tuple[object, str, object]] = []
    try:
        for name in _PATCH_NAMES:
            orig = getattr(oasr, name, None)
            if orig is None:
                continue
            originals.append((oasr, name, orig))
            setattr(oasr, name, _make_wrapper(rec, name, orig))
        # The defining module, for callers that import the name directly.  Same
        # recorder, so a call that goes through both is recorded once: the
        # package wrapper calls the *original* it captured, not the patched
        # module attribute.
        for name in _MODULE_PATCH_NAMES:
            orig = getattr(_fg, name, None)
            if orig is None:
                continue
            originals.append((_fg, name, orig))
            setattr(_fg, name, _make_wrapper(rec, name, orig))
        for name, idx in _DISPATCH_PATCH_SPECS.items():
            orig = getattr(_fg, name, None)
            if orig is None:
                continue
            op_name = name[len("_dispatch_") :]
            originals.append((_fg, name, orig))
            setattr(_fg, name, _make_dispatch_wrapper(rec, op_name, orig, idx))
        yield rec
    finally:
        for owner, name, orig in reversed(originals):
            setattr(owner, name, orig)


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
