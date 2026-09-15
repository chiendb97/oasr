#!/usr/bin/env python3
"""``include/oasr/common/reduction.h`` — the warp and block reductions.

A shared header with no kernel of its own, so nothing in the suite reached it.
That is how ``warpReduceMax`` came to shuffle the wrong operand: its butterfly
exchanged the *original* ``val`` each round instead of the running maximum, so a
lane ended with the max over the six lanes it partnered with rather than over all
32.  Only ``softmaxKernel`` calls it, through ``blockReduceMax``, and nothing
launches ``softmaxKernel`` — ``softmax::Softmax`` dispatches the online kernel —
so the wrong answer was never computed.

These tests exist because "never computed" is not a property anyone maintains.
The inputs are chosen so the bug is *visible*: a uniform input, or one whose
maximum sits in lane 0, returns the right answer either way.
"""

import functools

import pytest
import torch
from helpers import REPO_ROOT

pytestmark = pytest.mark.cuda


@functools.lru_cache(maxsize=1)
def _probe():
    """Compile the probe TU with the same flags and includes as a real kernel."""
    from oasr.jit.core import gen_jit_spec

    src = REPO_ROOT / "tests/kernels/fixtures/reduction_probe.cu"
    assert src.exists(), src
    return gen_jit_spec("reduction_probe", [src]).build_and_load()


def _warp_max(values):
    out = torch.empty(len(values), device="cuda", dtype=torch.float32)
    inp = torch.tensor(values, device="cuda", dtype=torch.float32)
    _probe().warp_reduce_max(out, inp)
    torch.cuda.synchronize()
    return out


class TestWarpReduceMax:
    def test_every_lane_ends_with_the_whole_warp_maximum(self):
        """The property a butterfly is *for*: all 32 lanes agree, on the max of all 32.

        With the maximum in lane 17, the broken butterfly leaves the lanes that
        never partnered with 17 holding something smaller — so this checks the
        whole vector, not ``out[0]``.
        """
        values = [float(i) for i in range(32)]
        values[17] = 100.0
        out = _warp_max(values)
        assert out.tolist() == [100.0] * 32, out.tolist()

    @pytest.mark.parametrize("winner", list(range(32)))
    def test_the_maximum_is_found_from_any_lane(self, winner):
        """One case per lane, because the broken version is right for exactly the
        lanes its partner set happens to cover — a single spot check passes."""
        values = [0.0] * 32
        values[winner] = 1.0
        out = _warp_max(values)
        assert out.tolist() == [1.0] * 32, f"winner in lane {winner}: {out.tolist()}"

    def test_a_partial_warp_ignores_the_inactive_lanes(self):
        """Lanes past ``n`` feed in a sentinel, so a short row must not pick it up
        and must not let it win."""
        out = _warp_max([-5.0, -9.0, -1.0])
        assert out.tolist() == [-1.0, -1.0, -1.0]

    def test_negative_values_do_not_reduce_to_zero(self):
        out = _warp_max([-3.0] * 32)
        assert out.tolist() == [-3.0] * 32


class TestBlockReduceMax:
    @pytest.mark.parametrize("threads,n", [(32, 32), (64, 64), (256, 200), (1024, 1024)])
    def test_matches_torch_amax(self, threads, n):
        """Across the warp boundary: ``blockReduceMax`` reduces within warps, then
        across them through shared memory, so a per-warp bug shows up only once
        there is more than one warp."""
        torch.manual_seed(n)
        inp = torch.randn(n, device="cuda", dtype=torch.float32)
        out = torch.empty(1, device="cuda", dtype=torch.float32)
        _probe().block_reduce_max(out, inp, threads)
        torch.cuda.synchronize()
        assert out.item() == pytest.approx(inp.max().item()), f"threads={threads} n={n}"

    def test_the_maximum_in_the_last_warp_still_wins(self):
        """The case a single-warp test cannot see."""
        n = 256
        inp = torch.full((n,), -1.0, device="cuda", dtype=torch.float32)
        inp[n - 1] = 42.0
        out = torch.empty(1, device="cuda", dtype=torch.float32)
        _probe().block_reduce_max(out, inp, n)
        torch.cuda.synchronize()
        assert out.item() == 42.0
