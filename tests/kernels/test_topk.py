#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.topk`` (``oasr/functionals/topk.py``) against ``torch.topk``.

The kernel selects per row, so batch and sequence are grid parallelism, not
code paths: the axes that reach a different branch are ``channels`` (which
row width, and so which selection strategy) and ``k``.  The grid is chosen on
that basis rather than as a product of everything available.

Parity with ``torch.topk`` is the oracle for the values, and on distinct values
it already implies descending order, unique indices and ``k=1 == argmax``.  It
does not pin the indices among ties: a masked row is mostly ``-inf``, and a
selection that returned a pad's index (-1, or one past the row) with the right
value would pass it -- in fp16 a ``-FLT_MAX`` pad even casts to ``-inf``.  So
:class:`TestRowsWithFewerFiniteValuesThanK` checks the indices themselves.
"""

import pytest
import torch
from helpers import tol

import oasr

# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.cuda


class TestTopK:
    @pytest.mark.parametrize("shape", [(2, 128, 256), (4, 64, 512), (256, 4096)])
    @pytest.mark.parametrize("k", [1, 10])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_matches_torch(self, shape, k, dtype):
        """Values, order and index validity, against the reference selection.

        The 2-D shape is in the grid rather than in a test of its own: rank is
        a reshape in the wrapper, and the kernel sees the same rows either way.
        """
        x = torch.randn(*shape, device="cuda", dtype=dtype)

        values, indices = oasr.topk(x, k)

        assert indices.dtype == torch.int32
        assert (indices >= 0).all() and (indices < shape[-1]).all()
        # The returned values must be the ones the indices point at, not a
        # separately-computed answer that happens to be close.
        assert (x.gather(-1, indices.long()) == values).all()

        ref_values, _ = torch.topk(x, k, dim=-1)
        torch.testing.assert_close(values, ref_values, **tol(dtype))

    def test_destination_passing(self):
        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16)
        out_v = torch.empty(2, 64, 5, device="cuda", dtype=torch.float16)
        out_i = torch.empty(2, 64, 5, device="cuda", dtype=torch.int32)

        values, indices = oasr.topk(x, 5, out_values=out_v, out_indices=out_i)

        assert values.data_ptr() == out_v.data_ptr()
        assert indices.data_ptr() == out_i.data_ptr()
        torch.testing.assert_close(values, torch.topk(x, 5, dim=-1).values)

    def test_cpu_error(self):
        """CPU is out of scope for this kernel, and says so rather than falling back."""
        x = torch.randn(2, 64, 128, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="CUDA tensor"):
            oasr.topk(x, 5)

    def test_layer_matches_the_functional(self):
        from oasr.layers import TopK

        x = torch.randn(2, 64, 256, device="cuda", dtype=torch.float16)
        values, indices = TopK(k=5)(x)

        ref_v, ref_i = oasr.topk(x, 5)
        assert torch.equal(values, ref_v) and torch.equal(indices, ref_i)


#: (num_cols, k) -> the selection ``topk::TopK`` runs for it.
_PATHS = {
    (20, 5): "iterative-narrow-row",
    (16384, 4): "iterative-too-wide-for-bitonic",
    (1000, 200): "iterative-k-past-bitonic",
    (64, 4): "bitonic",
    (5000, 40): "bitonic-wide",
    (5000, 200): "radix-sort",
}


class TestRowsWithFewerFiniteValuesThanK:
    """A row whose finite values run out before ``k`` does.

    Every selection pads a short row out to its block, and the iterative one
    retires each element it picks; all of them used ``-FLT_MAX`` for that, which
    outranks a ``-inf`` entry, so a pad was selected once the finite values ran
    out: index -1 (bitonic), an index past the row (radix sort) or index 0 again
    (iterative).  ``-inf`` rows are what a masked logit row looks like.
    """

    @pytest.mark.parametrize("cols,k", list(_PATHS), ids=list(_PATHS.values()))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_the_rest_is_selected_from_the_minus_inf_entries(self, cols, k, dtype):
        x = torch.full((3, cols), float("-inf"), device="cuda", dtype=dtype)
        x[0, 7], x[0, 0] = 2.0, 1.0  # two finite values
        x[1, cols - 1] = 0.5  # one, at the far end of the row
        # row 2 is all -inf

        values, indices = oasr.topk(x, k)

        idx = indices.long()
        assert ((idx >= 0) & (idx < cols)).all(), "an index outside the row"
        assert all(len(set(row)) == k for row in idx.tolist()), "an index selected twice"
        assert torch.equal(x.gather(-1, idx), values)
        assert torch.equal(values, torch.topk(x, k, dim=-1).values)

    @pytest.mark.parametrize("cols,k", list(_PATHS), ids=list(_PATHS.values()))
    def test_nan_ranks_above_every_value_as_in_torch(self, cols, k):
        """Every selection compared floats, which never picks a NaN."""
        x = torch.randn(2, cols, device="cuda")
        x[0, 3] = x[0, cols - 1] = float("nan")
        x[1, :] = float("nan")

        values, indices = oasr.topk(x, k)

        want = torch.topk(x, k, dim=-1).values
        torch.testing.assert_close(values, want, rtol=0, atol=0, equal_nan=True)
        gathered = x.gather(-1, indices.long())
        torch.testing.assert_close(gathered, values, rtol=0, atol=0, equal_nan=True)
        assert all(len(set(row)) == k for row in indices.tolist())


class TestLauncherContract:
    """The launcher checked the input's last stride and nothing about the outputs."""

    def test_outputs_of_the_wrong_shape_or_dtype_are_refused(self):
        """Both used to be written through: past the end of a short buffer, or
        as half-precision bits into a float one."""
        x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
        short = torch.empty(4, 2, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="values must be shaped"):
            oasr.topk(x, 8, out_values=short)
        wide = torch.empty(4, 8, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="values must have the input's dtype"):
            oasr.topk(x, 8, out_values=wide)
        long_rows = torch.empty(4, 9, device="cuda", dtype=torch.int32)
        with pytest.raises(RuntimeError, match="indices must be shaped"):
            oasr.topk(x, 8, out_indices=long_rows)

    def test_a_padded_row_stride_is_read_at_its_own_rows(self):
        x = torch.randn(4, 512, device="cuda", dtype=torch.float16)[:, :256]
        assert x.stride(-1) == 1 and not x.is_contiguous()
        values, _ = oasr.topk(x, 3)
        assert torch.equal(values, torch.topk(x, 3, dim=-1).values)

    def test_no_rows(self):
        """A zero-block grid is a launch error, not a no-op."""
        values, indices = oasr.topk(torch.empty(0, 64, device="cuda", dtype=torch.float16), 4)
        assert values.shape == indices.shape == (0, 4)
