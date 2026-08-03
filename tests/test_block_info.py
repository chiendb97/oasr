"""Trace tests for ``oasr.kernels.cute.block_info``.

This helper is the one FMHA module no kernel imports -- ``FmhaSm80.kernel``
computes the same K-tile bounds inline -- and being unreachable is exactly why
it rotted: it called ``cute.arch.min_s32`` / ``cute.arch.max_s32``, which exist
in no CuTeDSL release this project supports (4.5 / 4.6), so *any* causal or
windowed trace through it would have raised ``AttributeError``.  Nothing caught
it because nothing ran it.  Tracing it here is the cheap guard: a CuTeDSL
upgrade that moves an integer min/max API now fails a test rather than waiting
for a first caller.

The bounds are checked against the tile arithmetic the module's own docstring
promises, so it stays a usable reference instead of merely compiling.

Lives in its own file rather than in ``tests/test_fmha.py`` because that module
enables ``from __future__ import annotations``: under PEP 563 the CuTeDSL
decorators resolve a ``@cute.jit`` signature from *strings* in the defining
module's namespace, and ``cute`` is imported lazily there, so every annotation
lookup raises ``NameError``.  Same reason the kernels themselves carry a
"do not enable" note about it.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
cuda_driver = pytest.importorskip("cuda.bindings.driver")

from cutlass.cute.runtime import from_dlpack  # noqa: E402

from oasr.kernels.cute.block_info import make_block_info  # noqa: E402

M_BLOCK = 64
N_BLOCK = 64


class BlockInfoProbe:
    """Minimal launcher: one thread computes the bounds and stores them."""

    def __init__(self, *, m_block_size, n_block_size, causal, window_left, window_right):
        self._m = m_block_size
        self._n = n_block_size
        self._causal = causal
        self._wl = window_left
        self._wr = window_right

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        m_block: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        stream: cuda_driver.CUstream,
    ):
        self.kernel(mOut, m_block, seqlen_q, seqlen_k).launch(
            grid=[1, 1, 1], block=[1, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        m_block: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
    ):
        info = make_block_info(
            m_block,
            seqlen_q,
            seqlen_k,
            self._m,
            self._n,
            self._causal,
            self._wl,
            self._wr,
        )
        mOut[0] = info.n_block_min
        mOut[1] = info.n_block_max


def reference_bounds(m_block, seqlen_q, seqlen_k, m, n, causal, wl, wr):
    """The contract from ``block_info``'s docstring, in plain Python."""
    if causal or wr >= 0:
        n_max_row = min(seqlen_k, (m_block + 1) * m + (wr if wr >= 0 else 0))
    else:
        n_max_row = seqlen_k
    n_min_row = max(0, m_block * m - wl) if wl >= 0 else 0
    n_block_max = -(-n_max_row // n)  # ceil_div
    n_block_min = n_min_row // n
    if m_block * m >= seqlen_q:  # Q tile past the end -> empty K range
        n_block_max = n_block_min
    return n_block_min, n_block_max


# (m_block, seqlen_q, seqlen_k, causal, window_left, window_right)
CASES = [
    (0, 128, 1024, False, -1, -1),  # unbounded: the whole K range
    (1, 128, 1024, True, -1, -1),  # causal: the diagonal caps the top
    (3, 512, 1024, True, -1, -1),  # causal, deeper row block
    (4, 512, 1024, False, 128, -1),  # left window raises the floor
    (4, 512, 1024, True, 128, 0),  # causal + window: both ends at once
    (2, 512, 300, False, -1, -1),  # seqlen_k not a whole number of tiles
    (7, 128, 1024, False, -1, -1),  # Q tile past seqlen_q -> empty
]


@pytest.mark.parametrize("m_block,seqlen_q,seqlen_k,causal,wl,wr", CASES)
def test_bounds_match_reference(m_block, seqlen_q, seqlen_k, causal, wl, wr):
    probe = BlockInfoProbe(
        m_block_size=M_BLOCK,
        n_block_size=N_BLOCK,
        causal=causal,
        window_left=wl,
        window_right=wr,
    )
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    def args():
        return (
            from_dlpack(out, assumed_align=4).mark_layout_dynamic(leading_dim=0),
            cutlass.Int32(m_block),
            cutlass.Int32(seqlen_q),
            cutlass.Int32(seqlen_k),
            stream,
        )

    cute.compile(probe, *args())(*args())
    torch.cuda.synchronize()

    assert tuple(out.tolist()) == reference_bounds(
        m_block, seqlen_q, seqlen_k, M_BLOCK, N_BLOCK, causal, wl, wr
    )


def test_narrowing_actually_narrows():
    """Negative control for the cases above: they would all pass just as well
    against a helper that returned ``[0, ceil(seqlen_k / N))`` every time."""
    unbounded = reference_bounds(1, 512, 1024, M_BLOCK, N_BLOCK, False, -1, -1)
    causal = reference_bounds(1, 512, 1024, M_BLOCK, N_BLOCK, True, -1, -1)
    assert causal[1] < unbounded[1], "causal must lower n_block_max"
    windowed = reference_bounds(4, 512, 1024, M_BLOCK, N_BLOCK, False, 128, -1)
    assert windowed[0] > unbounded[0], "a left window must raise n_block_min"
