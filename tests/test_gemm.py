#!/usr/bin/env python3
"""
Unit tests for functional GEMM API (TVM-FFI JIT path).
"""

import itertools

import pytest
import torch

import oasr

# Every test in this module allocates directly on ``device="cuda"`` and calls a
# JIT-compiled kernel, so the whole file is CUDA-only.  Declaring that here is
# what lets the CPU CI job run `pytest tests/` and get a green, meaningful run
# instead of a wall of `RuntimeError: No CUDA GPUs are available`.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="OASR kernels require CUDA")


class TestGemm:
    """Tests for oasr.gemm() functional API."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (64, 128, 256),
            (32, 32, 32),
            (256, 32, 128),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm(self, M, N, K, dtype):
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)

        D = oasr.gemm(A, B)

        expected = torch.matmul(A, B.T)
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gemm_destination_passing(self, dtype):
        """Test GEMM with pre-allocated output."""
        M, N, K = 64, 128, 256
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(N, K, device="cuda", dtype=dtype)
        out = torch.empty(M, N, device="cuda", dtype=dtype)

        result = oasr.gemm(A, B, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = torch.matmul(A, B.T)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


class TestGroupGemm:
    """Tests for oasr.group_gemm() functional API.

    The grouped path had no functional coverage at all, which is how its SM90
    kernel shipped never having compiled: ``offset`` is the cumulative *end* row
    of each group, so group ``i`` is ``A[offset[i-1]:offset[i]] @ B[i].T``.
    """

    @staticmethod
    def _reference(A, B, rows):
        start = 0
        parts = []
        for group, count in enumerate(rows):
            parts.append(A[start : start + count] @ B[group].transpose(0, 1))
            start += count
        return torch.cat(parts, dim=0)

    @pytest.mark.parametrize("rows", [[32, 16, 48], [64, 64], [8]])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_group_gemm(self, rows, dtype):
        torch.manual_seed(0)
        N, K = 128, 64
        A = torch.randn(sum(rows), K, device="cuda", dtype=dtype)
        B = torch.randn(len(rows), N, K, device="cuda", dtype=dtype)
        offset = torch.tensor(list(itertools.accumulate(rows)), device="cuda", dtype=torch.int32)

        D = oasr.group_gemm(A, B, offset)

        torch.testing.assert_close(D, self._reference(A, B, rows), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_group_gemm_destination_passing(self, dtype):
        torch.manual_seed(1)
        rows, N, K = [32, 32], 128, 64
        A = torch.randn(sum(rows), K, device="cuda", dtype=dtype)
        B = torch.randn(len(rows), N, K, device="cuda", dtype=dtype)
        offset = torch.tensor(list(itertools.accumulate(rows)), device="cuda", dtype=torch.int32)
        out = torch.empty(sum(rows), N, device="cuda", dtype=dtype)

        result = oasr.group_gemm(A, B, offset, out=out)

        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(result, self._reference(A, B, rows), rtol=1e-2, atol=1e-2)


class TestBmm:
    """Tests for oasr.bmm() functional API."""

    @pytest.mark.parametrize(
        "batch_size,M,N,K",
        [
            (4, 64, 128, 256),
            (3, 32, 32, 32),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bmm(self, batch_size, M, N, K, dtype):
        A = torch.randn(batch_size, M, K, device="cuda", dtype=dtype)
        B = torch.randn(batch_size, N, K, device="cuda", dtype=dtype)

        D = oasr.bmm(A, B)

        expected = torch.bmm(A, B.permute(0, 2, 1))
        torch.testing.assert_close(D, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bmm_destination_passing(self, dtype):
        """Test BMM with pre-allocated output."""
        batch_size, M, N, K = 4, 64, 128, 256
        A = torch.randn(batch_size, M, K, device="cuda", dtype=dtype)
        B = torch.randn(batch_size, N, K, device="cuda", dtype=dtype)
        out = torch.empty(batch_size, M, N, device="cuda", dtype=dtype)

        result = oasr.bmm(A, B, out=out)

        assert result.data_ptr() == out.data_ptr()
        expected = torch.bmm(A, B.permute(0, 2, 1))
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestBmmGeneralLane:
    """The shapes the tuned alignment-8 tile variants refuse (KG5).

    Every case here is a Zipformer attention product with its real strides: the
    operands are 4-D permuted views of a ``(time, batch, head, dim)`` activation,
    one of them broadcasts over the request batch, and N or K is small, odd, or
    both.  These used to be ``torch.matmul`` in the model file, which is why the
    ratchet in ``test_layer_waist.py`` exists alongside them.
    """

    @staticmethod
    def _check(got, A, B, expected_shape):
        assert tuple(got.shape) == expected_shape, (tuple(got.shape), expected_shape)
        # fp32 reference: at K = 496 an fp16 reference accumulates its own error.
        expected = torch.matmul(A.float(), B.float().transpose(-1, -2))
        torch.testing.assert_close(got.float(), expected, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "T,heads,batch",
        [(496, 4, 1), (124, 4, 3), (62, 8, 1)],
        ids=["stack0-b1", "stack2-b3", "stack3-b1"],
    )
    def test_score_product_permuted_4d(self, T, heads, batch, dtype):
        """``q @ k``: K = 32 aligned, N = T arbitrary, both operands permuted."""
        x = torch.randn(T, batch, heads, 96, device="cuda", dtype=dtype)
        q = x[..., 0:32].permute(2, 1, 0, 3)
        k = x[..., 32:64].permute(2, 1, 0, 3)
        assert not q.is_contiguous() and not k.is_contiguous()
        self._check(oasr.bmm(q, k), q, k, (heads, batch, T, T))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("T,heads,batch", [(496, 4, 1), (124, 4, 3)])
    def test_relative_position_product_broadcast_batch(self, T, heads, batch, dtype):
        """``p @ pos_emb``: K = 4, N = 2T-1 (always odd), B shared over the batch.

        The broadcast axis is the one no single batch stride can express, so this
        is also the case that exercises the launcher's per-outer-index loop.
        """
        p = torch.randn(T, batch, heads, 4, device="cuda", dtype=dtype).permute(2, 1, 0, 3)
        pos = torch.randn(1, 2 * T - 1, heads, 4, device="cuda", dtype=dtype).permute(2, 0, 1, 3)
        self._check(oasr.bmm(p, pos), p, pos, (heads, batch, T, 2 * T - 1))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("T,heads,batch", [(496, 4, 1), (62, 8, 3)])
    def test_value_product_row_major_b(self, T, heads, batch, dtype):
        """``probs @ v``: N = 12, K = T, and B contiguous along N, not K.

        This is the layout that would otherwise cost a ``.contiguous()`` — one
        extra copy *and* one extra launch — at every value product.
        """
        w = torch.randn(heads, batch, T, T, device="cuda", dtype=dtype)
        v = torch.randn(T, batch, heads, 12, device="cuda", dtype=dtype).permute(2, 1, 3, 0)
        assert v.stride(-1) != 1 and v.stride(-2) == 1
        self._check(oasr.bmm(w, v), w, v, (heads, batch, T, 12))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_value_product_single_head(self, dtype):
        """NonlinAttention shares one head's weights across a wide value dim."""
        T, hidden, batch = 248, 192, 2
        w = torch.randn(1, batch, T, T, device="cuda", dtype=dtype)
        v = torch.randn(T, batch, 1, hidden, device="cuda", dtype=dtype).permute(2, 1, 3, 0)
        self._check(oasr.bmm(w, v), w, v, (1, batch, T, hidden))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M,N,K", [(7, 11, 5), (33, 1, 16), (33, 17, 1), (9, 17, 4)])
    def test_unaligned_3d(self, M, N, K, dtype):
        """The historical ``[batch, N, K]`` contract, now with arbitrary N and K.

        ``K = 5`` is odd, so it has no tensor-op instantiation at all and proves
        the SIMT lane is reachable rather than a dead branch.
        """
        A = torch.randn(2, M, K, device="cuda", dtype=dtype)
        B = torch.randn(2, N, K, device="cuda", dtype=dtype)
        self._check(oasr.bmm(A, B), A, B, (2, M, N))

    @pytest.mark.parametrize("a_batch,b_batch", [(4, 1), (1, 4), (1, 1)])
    def test_broadcast_within_the_3d_fast_path(self, a_batch, b_batch):
        """3-D broadcasting has its own branch, because 3-D has its own fast path.

        The pre-KG5 shape (two 3-D operands) bypasses the general broadcast
        machinery to keep the entry point cheap — measured, that routing was
        worth 2.0 us a call — so its broadcast rule is separate code and needs
        its own coverage.
        """
        A = torch.randn(a_batch, 9, 16, device="cuda", dtype=torch.float16)
        B = torch.randn(b_batch, 5, 16, device="cuda", dtype=torch.float16)
        self._check(oasr.bmm(A, B), A, B, (max(a_batch, b_batch), 9, 5))

    def test_non_broadcastable_3d_batch_is_refused(self):
        A = torch.randn(4, 9, 16, device="cuda", dtype=torch.float16)
        B = torch.randn(3, 5, 16, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="not broadcastable"):
            oasr.bmm(A, B)

    def test_broadcast_a_3d_operand_against_a_4d_one(self):
        A = torch.randn(5, 13, 24, device="cuda", dtype=torch.float16)
        B = torch.randn(2, 5, 9, 24, device="cuda", dtype=torch.float16)
        self._check(oasr.bmm(A, B), A, B, (2, 5, 13, 9))

    def test_aligned_contiguous_keeps_the_tuned_lane(self, monkeypatch):
        """KG5 must not move the established alignment-8 lane onto the general one.

        The tuned lane carries the shape heuristic and its measured tile choices;
        a contiguous 3-D alignment-8 call reaching the general dispatcher would
        be a silent performance regression, not a wrong answer, so nothing else
        would catch it.
        """
        import oasr.functionals.gemm as gemm_module

        monkeypatch.setattr(
            gemm_module,
            "_general_bmm_fn",
            lambda: pytest.fail("aligned contiguous BMM reached the general lane"),
        )
        A = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16)
        B = torch.randn(4, 24, 64, device="cuda", dtype=torch.float16)
        self._check(oasr.bmm(A, B), A, B, (4, 32, 24))

    def test_b_contiguous_along_neither_axis_is_refused(self):
        """A layout with no CUTLASS expression must say so, not be reinterpreted."""
        base = torch.randn(2, 8, 16, 4, device="cuda", dtype=torch.float16)
        A = torch.randn(2, 6, 16, device="cuda", dtype=torch.float16)
        B = base[..., 0]  # (2, 8, 16) -- strides (512, 64, 4): neither axis is 1
        assert B.stride(-1) != 1 and B.stride(-2) != 1
        with pytest.raises(Exception, match="contiguous along one of its two trailing axes"):
            oasr.bmm(A, B)

    def test_survives_cuda_graph_capture_and_replay(self):
        """The streaming encoder is graph-captured, so this lane has to be.

        Two ways it could fail and neither shows up in an eager test: an
        allocation inside the launcher (``GemmBatched`` needs no workspace, which
        is why there is none), and the per-outer-index host loop, whose trip
        count must come from shapes rather than from anything that changes
        between capture and replay.  The broadcast case is the one that loops.
        """
        T, heads, batch = 124, 4, 2
        p = torch.randn(T, batch, heads, 4, device="cuda", dtype=torch.float16).permute(2, 1, 0, 3)
        pos = torch.randn(1, 2 * T - 1, heads, 4, device="cuda", dtype=torch.float16)
        pos_v = pos.permute(2, 0, 1, 3)
        out = torch.empty(heads, batch, T, 2 * T - 1, device="cuda", dtype=torch.float16)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                oasr.bmm(p, pos_v, out=out)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            oasr.bmm(p, pos_v, out=out)

        # New operand values through the captured buffers: a replay must
        # recompute, not reproduce the captured result.
        p.copy_(torch.randn_like(p))
        pos.copy_(torch.randn_like(pos))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.matmul(p.float(), pos_v.float().transpose(-1, -2))
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)

    def test_out_shape_mismatch_is_refused(self):
        A = torch.randn(2, 7, 5, device="cuda", dtype=torch.float16)
        B = torch.randn(2, 11, 5, device="cuda", dtype=torch.float16)
        out = torch.empty(2, 7, 12, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="expected"):
            oasr.bmm(A, B, out=out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestGemmInputLayout:
    """The launchers flatten N-D inputs themselves and require contiguity.

    Both properties are load-bearing.  Flattening in C++ (``FLATTENED_ROWS``)
    is what removed the two per-call ``reshape(-1, K)`` calls that made
    ``oasr.gemm`` lose to ``F.linear`` on small shapes.  The contiguity check is
    a correctness fix that came with it.
    """

    @staticmethod
    def _weight(N, K):
        return torch.randn(N, K, device="cuda", dtype=torch.float16)

    @pytest.mark.parametrize("shape", [(64,), (8, 64), (4, 6, 64), (2, 3, 5, 64)])
    def test_leading_dims_are_flattened(self, shape):
        """1-D through 4-D activations all work; the launcher computes M."""
        A = torch.randn(*shape, device="cuda", dtype=torch.float16)
        B = self._weight(32, 64)
        got = oasr.gemm(A, B)
        assert tuple(got.shape) == tuple(shape[:-1]) + (32,)
        torch.testing.assert_close(got, torch.nn.functional.linear(A, B), rtol=2e-2, atol=2e-2)

    def test_row_strided_2d_input_is_correct(self):
        """Regression: ``x[:, -1]`` of a ``(B, T, D)`` tensor.

        ``reshape(-1, K)`` is a *no-op* on an already-2-D tensor, so it never
        delivered the row-major layout the kernel assumes — the launcher only
        checked that the tensor was on CUDA, and the kernel indexes rows as
        ``A + row * K``.  This shape used to come back with a max error of ~30
        against ``F.linear``, silently.  The N-D path was accidentally safe
        because flattening a strided N-D tensor cannot be a view.
        """
        x = torch.randn(4, 6, 64, device="cuda", dtype=torch.float16)
        A = x[:, -1]
        assert not A.is_contiguous(), "the test input must actually be strided"
        B = self._weight(64, 64)
        torch.testing.assert_close(
            oasr.gemm(A, B), torch.nn.functional.linear(A, B), rtol=2e-2, atol=2e-2
        )

    def test_noncontiguous_out_is_rejected(self):
        """A strided ``out`` would be written at the wrong rows; say so."""
        A = torch.randn(8, 64, device="cuda", dtype=torch.float16)
        B = self._weight(64, 64)
        bad = torch.empty(8, 2, 64, device="cuda", dtype=torch.float16)[:, 0]
        with pytest.raises(Exception, match="[Cc]ontiguous"):
            oasr.gemm(A, B, None, out=bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestGemmAlignmentContract:
    """Projection GEMMs reject unaligned shapes; BMM has a declared general lane.

    The projections used to disagree with each other.  ``gemm`` let CUTLASS fail
    and surfaced "GEMM kernel failed", which tells the caller nothing actionable;
    ``gemm_log_softmax`` silently rerouted the same input to cuBLAS, which tells
    them nothing at all and quietly leaves the model off the kernel path forever.
    They keep one precondition and one actionable answer, because every unaligned
    projection in this repo is an output head and padding it at the model layer
    is the established fix (``align_out_features`` / ``pad_output_projection``).

    BMM is the one member that cannot take that answer: Zipformer's attention
    needs arbitrary N and K on a *contraction* it does not own, so KG5 gave it a
    general lane instead of an error message.
    """

    MSG = "8-aligned"

    @staticmethod
    def _ab(M, N, K, dtype=torch.float16):
        return (
            torch.randn(M, K, device="cuda", dtype=dtype),
            torch.randn(N, K, device="cuda", dtype=dtype),
        )

    @pytest.mark.parametrize("N,K", [(500, 64), (64, 60), (30, 24)])
    def test_gemm_rejects_unaligned(self, N, K):
        A, B = self._ab(16, N, K)
        with pytest.raises(Exception, match=self.MSG):
            oasr.gemm(A, B)

    @pytest.mark.parametrize("N,K", [(500, 64), (64, 60)])
    def test_gemm_activation_rejects_unaligned(self, N, K):
        A, B = self._ab(16, N, K)
        with pytest.raises(Exception, match=self.MSG):
            oasr.gemm_activation(A, B, None, oasr.ACTIVATION_RELU)

    @pytest.mark.parametrize("N,K", [(500, 64), (64, 60)])
    def test_gemm_log_softmax_rejects_unaligned(self, N, K):
        """Regression: this one used to succeed via a silent cuBLAS reroute."""
        A, B = self._ab(16, N, K)
        with pytest.raises(Exception, match=self.MSG):
            oasr.gemm_log_softmax(A, B)

    def test_bmm_accepts_unaligned(self):
        """The one deliberate exception to the family's alignment contract."""
        A = torch.randn(2, 16, 60, device="cuda", dtype=torch.float16)
        B = torch.randn(2, 64, 60, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(
            oasr.bmm(A, B).float(),
            torch.matmul(A.float(), B.float().transpose(-1, -2)),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_the_message_names_the_fix(self):
        """An error a caller cannot act on is barely better than a silent one."""
        A, B = self._ab(16, 500, 64)
        with pytest.raises(Exception) as exc:
            oasr.gemm(A, B)
        text = str(exc.value)
        assert "N=500" in text and "K=64" in text, text
        assert "align_out_features" in text, text

    @pytest.mark.parametrize("N,K", [(504, 64), (64, 64), (8, 8)])
    def test_aligned_shapes_still_work(self, N, K):
        A, B = self._ab(16, N, K)
        torch.testing.assert_close(
            oasr.gemm(A, B), torch.nn.functional.linear(A, B), rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
