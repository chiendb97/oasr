#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr.topk`` (``oasr/functionals/topk.py``) against ``torch.topk``.

The kernel selects per row, so batch and sequence are grid parallelism, not
code paths: the axes that reach a different branch are ``channels`` (which
row width, and so which selection strategy) and ``k``.  The grid is chosen on
that basis rather than as a product of everything available.

Parity with ``torch.topk`` is the whole oracle.  It already implies descending
order, unique indices and ``k=1 == argmax``, so those are not asserted again --
a test that restates a consequence of the oracle cannot fail unless the oracle
already did.
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
