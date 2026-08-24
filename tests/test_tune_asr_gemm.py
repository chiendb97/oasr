# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Rule-emission logic in ``scripts/tune_asr_gemm.py``.

The measurement protocol is the part of this script that was wrong: it timed a
back-to-back eager launch loop, and every GEMM arm at ASR sizes is faster than
the ~9.6 us it costs to *issue* a call, so the loop read 10-20 us for all of them.
It reported 2.00x where a graph-captured measurement says 4.6x on the LSTM gate
projection, and picked a tile 22% slower than the best.

``_bench`` now returns two numbers and ``emit_rules`` gates on both.  Both halves
are covered here without a GPU: emission is a pure function of the timings.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "tune_asr_gemm", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tune_asr_gemm.py"
)
assert _SPEC and _SPEC.loader
tune = importlib.util.module_from_spec(_SPEC)
sys.modules["tune_asr_gemm"] = tune
_SPEC.loader.exec_module(tune)


class _Tactic:
    """Minimal stand-in for a registry tactic."""

    def __init__(self, backend="cutlass", **cfg):
        self.backend = backend
        base = {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "warp_m": 32,
            "warp_n": 32,
            "warp_k": 64,
            "kStages": 3,
            "split_k": 1,
            "stream_k": 0,
            "parallel_split_k": 0,
        }
        base.update(cfg)
        self.config = tuple(base.items())


def _rep(op="gemm", M=64, N=256, K=256, m_max=64):
    return tune.RepShape(op, M, N, K, "bfloat16", 1, m_max, 1.0)


def _results(winner_loop, winner_solo, default_loop, default_solo, **cfg):
    """A two-arm result list: a candidate and the fallback."""
    win = tune.TacticResult(_Tactic(**cfg), winner_loop, False, winner_solo)
    dflt = tune.TacticResult(_Tactic(block_m=128, block_n=128), default_loop, True, default_solo)
    return sorted([win, dflt], key=lambda r: r.median_ms)


class TestSelfOverlapGate:
    """A configuration that only wins back-to-back won by overlapping with itself.

    Independent launches queued together can run concurrently, which flatters a
    low-occupancy tile — a thin tile at small M is 20 CTAs on 170 SMs.  Inside a
    real layer the GEMM has dependent work behind it and never gets that.
    Self-overlap is not a speedup a model can spend.
    """

    def _emit(self, results, rep):
        key = (rep.op, rep.M, rep.N, rep.K, rep.dtype, rep.batch)
        return tune.emit_rules({key: results}, [rep], 120, min_speedup=1.05)

    def test_a_win_on_both_measurements_is_emitted(self):
        rep = _rep()
        out = self._emit(_results(1.0, 1.0, 2.0, 2.0, block_m=32), rep)
        assert "CutlassGemmConfig(block_m=32" in out
        assert "2.00x vs default" in out, "the emitted comment must carry the measurement"

    def test_a_win_only_back_to_back_is_refused(self, capsys):
        rep = _rep()
        # 2x on the loop, but the single replay says the default is faster.
        out = self._emit(_results(1.0, 1.2, 2.0, 1.0, block_m=32), rep)
        assert (
            "CutlassGemmConfig(block_m=32" not in out
        ), "a self-overlap-only win must not become a rule"
        assert "self-overlap only" in capsys.readouterr().out

    def test_a_loss_on_the_loop_is_refused_as_before(self, capsys):
        rep = _rep()
        out = self._emit(_results(1.99, 1.99, 2.0, 2.0, block_m=32), rep)
        assert "CutlassGemmConfig(block_m=32" not in out
        assert "keeping the fallback" in capsys.readouterr().out

    def test_solo_defaults_to_the_loop_number(self):
        """An old-style result with no solo timing must behave as it used to."""
        t = tune.TacticResult(_Tactic(), 1.0, False)
        assert t.solo_ms == t.median_ms


class TestBenchProtocol:
    """What ``_bench`` is contracted to do, without running it."""

    def test_captures_are_pooled_and_share_one_side_stream(self):
        """Both were unbounded allocations: a graph pool per capture, and a
        workspace-cache key per stream.  A sweep hit 30 GiB and died."""
        assert tune._GRAPH_ITERS > 1
        assert tune._GRAPH_REPS >= 3
        src = pathlib.Path(
            pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tune_asr_gemm.py"
        ).read_text()
        assert "pool=_graph_pool()" in src, "captures must share one graph pool"
        assert "side = _side_stream()" in src, "warm-ups must share one side stream"
        # The call, not the prose: ``_bench``'s docstring names what it replaced.
        assert (
            "from triton.testing import do_bench" not in src
        ), "the eager back-to-back loop must be gone"
        assert "do_bench(" not in src

    def test_returns_two_timings(self):
        import inspect

        doc = inspect.getdoc(tune._bench) or ""
        assert "loop_ms" in doc and "solo_ms" in doc
        sig = inspect.signature(tune._bench)
        assert "Tuple" in str(sig.return_annotation) or sig.return_annotation is not inspect._empty


class TestMLadder:
    """Bucket edges — unchanged, but they decide which M a rule covers."""

    @pytest.mark.parametrize("m,edge", [(1, 16), (16, 16), (17, 32), (500, 512), (10**9, 131072)])
    def test_ladder_edge(self, m, edge):
        assert tune._ladder_edge(m) == edge
