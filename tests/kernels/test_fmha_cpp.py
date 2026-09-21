# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The C++ CUTLASS/CuTe FMHA backend's own surface.

``test_fmha.py`` is backend-agnostic parity: it runs the same assertions
against whichever lane ``OASR_ATTN_BACKEND`` picks, which is what proves the
two agree.  What it structurally cannot cover is the set of properties that
exist *because* this lane is C++ -- the ones the CuTeDSL lane either does not
have or reaches by a different route:

* the **tile resolver** lives in a ``constexpr`` C++ function and is mirrored in
  Python for routing.  A mirror nobody checks is a mirror that drifts, so the
  agreement is asserted here, over four architectures, from one box.
* **arbitrary runtime strides**: the whole point of the backend is that
  ``_ensure_canonical``'s copy is gone, so the claim "no copy" needs an
  allocation count, not just a numeric compare.
* **sliding windows**, which this lane has and the CuTeDSL one has no argument
  for.
* **varlen through the same instantiation**: packed inputs are a dense call
  with a zero batch stride, so the assertion worth making is that a packed run
  equals a per-segment dense run exactly.
* **CUDA-graph replay and determinism**, which no FMHA test pinned at all
  before this file (AGENTS.md rule 11 had no attention pin).

Nothing here is a second copy of a parity test.  Where a claim is about
*numbers* it is stated as an ``atol=0`` identity between two ways of asking the
kernel for the same restriction -- a window as a parameter against the same
window as an ``-inf`` bias, say.  That is a sharper instrument than a tolerance
against SDPA, because both sides run the same kernel and any disagreement is
the boundary arithmetic and nothing else.
"""

from __future__ import annotations

import functools

import pytest
import torch
from helpers import (
    FMHA_TOL,
    assert_dest_passing,
    assert_graph_replay,
    device_sm,
    ref_fmha,
    window_bias,
)

pytestmark = pytest.mark.cuda

_SUPPORTED = (80, 86, 89, 120)


def _requires_cxx():
    if device_sm() not in _SUPPORTED:
        pytest.skip(f"the C++ FMHA lane is not compiled for sm_{device_sm()}")


@pytest.fixture(scope="module")
def fmha_cxx():
    """``oasr.fmha`` bound to the C++ backend.

    A *bound callable* rather than a global mode flip: ``set_backend_mode``
    clears three compile caches, so driving an A/B through the environment
    variable re-invokes ``cutlass.cute.compile()`` on every flip.
    """
    _requires_cxx()
    from oasr.functionals.attention import fmha

    return functools.partial(fmha, backend="cxx")


@pytest.fixture(scope="module")
def fmha_varlen_cxx():
    _requires_cxx()
    from oasr.functionals.attention import fmha_varlen

    return functools.partial(fmha_varlen, backend="cxx")


def _qkv(B, H, T_q, T_k, D, *, H_kv=None, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    H_kv = H if H_kv is None else H_kv
    dev = "cuda"
    return (
        torch.randn(B, H, T_q, D, dtype=dtype, device=dev),
        torch.randn(B, H_kv, T_k, D, dtype=dtype, device=dev),
        torch.randn(B, H_kv, T_k, D, dtype=dtype, device=dev),
    )


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _config_probe():
    """``csrc/fmha_jit_binding.cu`` on its own -- the real file, not a copy.

    Compiling the shipped binding standalone (it pulls in only
    ``cutlass_fmha_configs.h``) is the point: a purpose-built fixture TU would
    be a *second* transcription of the resolver, which is the drift this test
    exists to catch.
    """
    from oasr.jit import env
    from oasr.jit.core import gen_jit_spec

    return gen_jit_spec(
        "fmha_config_probe", [env.OASR_CSRC_DIR / "fmha_jit_binding.cu"]
    ).build_and_load()


def _cxx_resolve(sm: int, elem_bits: int, head_dim: int):
    out = torch.zeros(7, dtype=torch.int32)
    _config_probe().fmha_resolved_config(out, sm, elem_bits, head_dim)
    return out.tolist()


class TestTileSelectionAgrees:
    """``fmhaResolveTile`` and :func:`oasr.jit.fmha.resolve_config`, field for field.

    Asked over an ``sm`` *argument*, so one built module answers for every
    supported architecture.  That is deliberate and it is the property a live
    device query would destroy: on this box a query would report sm_120's
    99 KB for all four rows, and it is exactly the sm_86/sm_89 budget that was
    once wrong (`.artifacts/arch_portability_audit.md` § A3).
    """

    @pytest.mark.parametrize("sm", [80, 86, 89, 120])
    @pytest.mark.parametrize("elem_bits", [16])
    @pytest.mark.parametrize(
        "head_dim", [8, 16, 24, 32, 40, 64, 72, 96, 128, 160, 192, 256, 320, 384, 512]
    )
    def test_python_mirror_matches_cxx(self, sm, elem_bits, head_dim):
        from oasr.jit.fmha import resolve_config

        got = _cxx_resolve(sm, elem_bits, head_dim)
        want = resolve_config(sm, elem_bits, head_dim)
        fields = [
            int(want.valid),
            want.block_m,
            want.block_n,
            want.num_warps,
            want.num_stages,
            int(want.q_in_regs),
            want.smem_bytes,
        ]
        assert got == fields, (
            f"resolver drift at sm_{sm} d{head_dim}: C++ {got} vs Python {fields}. "
            f"These are one decision written twice; the Python copy only exists "
            f"so the layer waist can route before anything is built."
        )

    @pytest.mark.parametrize("sm", [80, 86, 89, 120])
    def test_smem_budget_matches(self, sm):
        from oasr.jit.fmha import smem_budget

        assert int(_config_probe().fmha_smem_budget(sm)) == smem_budget(sm)

    def test_the_chosen_tile_fits_the_budget(self):
        """Whatever it picks must fit -- on every arch, not just this one."""
        from oasr.jit.fmha import smem_budget

        for sm in _SUPPORTED:
            for head_dim in (32, 64, 128, 192, 256):
                valid, _bm, _bn, _w, stages, _q, smem = _cxx_resolve(sm, 16, head_dim)
                if valid:
                    assert 0 < smem <= smem_budget(sm), (sm, head_dim, smem)
                    assert stages >= 1

    def test_a_head_dim_that_cannot_fit_is_refused_not_clamped(self):
        """The resolver says no rather than returning a tile that overflows."""
        valid, *_rest = _cxx_resolve(120, 16, 4096)
        assert valid == 0, (
            "head_dim 4096 cannot fit any architecture's shared memory; the "
            "resolver must decline so the Python side routes to SDPA instead "
            "of instantiating a kernel the driver will refuse to configure"
        )


class TestSmemBudgetRefusal:
    def test_python_and_cxx_refuse_the_same_configs(self):
        from oasr.jit.fmha import config_supported, resolve_config

        for sm in _SUPPORTED:
            for head_dim in (32, 64, 128, 256, 512, 1024, 4096):
                cxx_valid = bool(_cxx_resolve(sm, 16, head_dim)[0])
                assert resolve_config(sm, 16, head_dim).valid == cxx_valid
                if not cxx_valid:
                    assert not config_supported(
                        sm=sm, dtype_str="float16", head_dim=head_dim
                    ), f"sm_{sm} d{head_dim}: a config with no tile must not be offered"

    def test_an_unserviceable_shape_raises_rather_than_launching(self, fmha_cxx):
        # head_dim is a property of the tensors, so reach for one no arch can
        # hold.  The failure must arrive as an exception, not as a kernel that
        # the driver silently declines to configure.
        big = 4096
        q = torch.randn(1, 1, 8, big, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 1, 8, big, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 1, 8, big, dtype=torch.float16, device="cuda")
        # `RuntimeError` specifically: the refusal must come from the resolver
        # declining to render a cell, not from nvcc, the driver, or an
        # AttributeError on a symbol that was never exported.
        with pytest.raises(RuntimeError, match="shared memory"):
            fmha_cxx(q, k, v, softmax_scale=0.1)


# ---------------------------------------------------------------------------
# Arbitrary strides -- lever L1
# ---------------------------------------------------------------------------


class TestArbitraryStrides:
    """Every layout a real call site hands over, taken as-is.

    The CuTeDSL lane needs strictly canonical row-major strides, so
    ``_ensure_canonical`` copies q, k and v on the way in -- measured at
    1.24-2.15x of the whole call.  This backend's ``StrideQKV`` carries runtime
    strides on every mode but the innermost, so the copy is gone.  "Gone" is an
    allocation claim, so it is asserted as one.
    """

    def _layouts(self, B, H, T, D, dtype=torch.float16):
        """(name, q, k, v) over the shapes the engine actually produces."""
        dev = "cuda"
        torch.manual_seed(7)
        out = []

        # 1. plain contiguous (B, H, T, D)
        q = torch.randn(B, H, T, D, dtype=dtype, device=dev)
        k = torch.randn(B, H, T, D, dtype=dtype, device=dev)
        v = torch.randn(B, H, T, D, dtype=dtype, device=dev)
        out.append(("contiguous", q, k, v))

        # 2. head-split view of (B, T, H*D) -- what every attention layer has
        #    after its qkv projection, and what `_ensure_canonical` copies.
        packed = torch.randn(B, T, 3, H, D, dtype=dtype, device=dev)
        out.append(
            (
                "head_split",
                packed[:, :, 0].permute(0, 2, 1, 3),
                packed[:, :, 1].permute(0, 2, 1, 3),
                packed[:, :, 2].permute(0, 2, 1, 3),
            )
        )

        # 3. a capacity buffer sliced on time -- a stride *gap*, not a
        #    non-unit last stride.  This is the KV cache shape.
        cap = torch.randn(B, H, T * 2, D, dtype=dtype, device=dev)
        out.append(("capacity_slice", q, cap[:, :, :T], cap[:, :, T:]))

        # 4. a batch slice of a larger pool
        pool = torch.randn(B * 2, H, T, D, dtype=dtype, device=dev)
        out.append(("batch_slice", pool[:B], pool[B:], pool[:B]))

        # 5. head slice: a subset of heads out of a wider tensor
        wide = torch.randn(B, H * 2, T, D, dtype=dtype, device=dev)
        out.append(("head_slice", wide[:, :H], wide[:, H:], wide[:, :H]))

        # 6. transposed batch/head
        bh = torch.randn(H, B, T, D, dtype=dtype, device=dev).transpose(0, 1)
        out.append(("bh_swapped", bh, bh, bh))
        return out

    @pytest.mark.parametrize("causal", [False, True])
    def test_strided_equals_the_contiguous_copy_bit_for_bit(self, fmha_cxx, causal):
        B, H, T, D = 2, 4, 48, 64
        for name, q, k, v in self._layouts(B, H, T, D):
            got = fmha_cxx(q, k, v, softmax_scale=0.125, causal=causal)
            want = fmha_cxx(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                softmax_scale=0.125,
                causal=causal,
            )
            torch.testing.assert_close(
                got,
                want,
                atol=0,
                rtol=0,
                msg=lambda m, n=name: (f"layout {n!r} differs from its own contiguous copy: {m}"),
            )

    @staticmethod
    def _allocations(fn) -> int:
        """How many times the caching allocator handed out a block.

        A *count*, not ``memory_allocated()``.  The obvious instrument reads
        back to its starting value the moment the temporary is freed, so it
        reports zero for a path that copies q, k and v on every call -- which
        is to say it cannot fail, which is to say it tests nothing.  The
        monotonic counter in ``memory_stats`` sees the transient.
        """
        fn()  # warm: first call builds/loads and allocates for that
        torch.cuda.synchronize()
        key = "allocation.all.allocated"
        before = torch.cuda.memory_stats()[key]
        fn()
        torch.cuda.synchronize()
        return torch.cuda.memory_stats()[key] - before

    def test_a_head_split_view_is_not_copied(self, fmha_cxx):
        """The lever, stated as the allocation it removes."""
        B, H, T, D = 2, 4, 64, 64
        packed = torch.randn(B, T, 3, H, D, dtype=torch.float16, device="cuda")
        q, k, v = (packed[:, :, i].permute(0, 2, 1, 3) for i in range(3))
        out = torch.empty(B, H, T, D, dtype=torch.float16, device="cuda")

        n = self._allocations(lambda: fmha_cxx(q, k, v, softmax_scale=0.125, out=out))
        assert n == 0, (
            f"the C++ lane made {n} allocation(s) during a head-split call -- "
            f"that is exactly the canonical-stride copy this backend exists to "
            f"remove"
        )

    def test_the_cutedsl_lane_does_copy(self, fmha_cxx):
        """The other half of the claim, so the instrument itself is checked.

        Without this the previous test passes on a broken measurement, which is
        how it was written the first time.
        """
        pytest.importorskip("cutlass")
        from oasr.functionals.attention import fmha

        B, H, T, D = 2, 4, 64, 64
        packed = torch.randn(B, T, 3, H, D, dtype=torch.float16, device="cuda")
        q, k, v = (packed[:, :, i].permute(0, 2, 1, 3) for i in range(3))
        out = torch.empty(B, H, T, D, dtype=torch.float16, device="cuda")
        n = self._allocations(lambda: fmha(q, k, v, softmax_scale=0.125, out=out, backend="cute"))
        assert n > 0, (
            "the CuTeDSL lane is expected to copy q/k/v to canonical strides; "
            "if it no longer does, the L1 lever no longer exists and this "
            "file's claim needs re-measuring rather than deleting"
        )

    def test_a_non_unit_last_stride_is_copied_not_misread(self, fmha_cxx):
        """The one layout the kernel cannot express, and must not pretend to.

        ``StrideQKV``'s innermost mode is a compile-time 1.  A tensor whose last
        stride is not 1 would be read at wrong addresses with no fault at all,
        so the Python side copies it.  Silent wrongness is the failure mode
        being bought off here, which is why this asserts the *numbers*.
        """
        B, H, T, D = 1, 2, 32, 64
        wide = torch.randn(B, H, T, D, 2, dtype=torch.float16, device="cuda")
        q = wide[..., 0]  # stride(-1) == 2
        assert q.stride(-1) == 2
        k, v = wide[..., 1], wide[..., 0]
        got = fmha_cxx(q, k, v, softmax_scale=0.125)
        want = ref_fmha(q.contiguous(), k.contiguous(), v.contiguous(), 0.125)
        torch.testing.assert_close(got, want, **FMHA_TOL)


class TestOutIsHonouredWhenStrided:
    def test_a_strided_out_receives_the_result(self, fmha_cxx):
        B, H, T, D = 2, 4, 32, 64
        q, k, v = _qkv(B, H, T, T, D)
        # An `out` that is a view into a wider buffer: the launcher may stage
        # through a temporary, but the caller's tensor must end up holding the
        # answer -- returning a *different* tensor would leave `out` silently
        # uninitialised, which the CuTeDSL lane did until it was fixed.
        holder = torch.full((B, H, T, D, 2), float("nan"), dtype=torch.float16, device="cuda")
        out = holder[..., 0]
        assert out.stride(-1) == 2
        ret = fmha_cxx(q, k, v, softmax_scale=0.125, out=out)
        assert ret.data_ptr() == out.data_ptr()
        want = fmha_cxx(q, k, v, softmax_scale=0.125)
        torch.testing.assert_close(out, want, atol=0, rtol=0)
        assert torch.isfinite(holder[..., 1]).logical_not().all(), (
            "the interleaved neighbour was overwritten -- the store predicate "
            "is writing past its row"
        )


class TestDestinationPassing:
    def test_out_first_and_out_is_the_output(self, fmha_cxx):
        q, k, v = _qkv(2, 4, 32, 64, 64)
        want = fmha_cxx(q, k, v, softmax_scale=0.125)
        out = torch.empty_like(want)
        assert_dest_passing(fmha_cxx, q, k, v, softmax_scale=0.125, out=out, expected=want)


# ---------------------------------------------------------------------------
# Masking: the same restriction, asked for two ways
# ---------------------------------------------------------------------------


class TestMaskedInteriorSplit:
    """The four-way K-loop split must not change the answer.

    The mainloop walks a Q tile's K range in four passes -- the first tile, the
    causal/local right edge, an interior with *no* predicate at all, and a left
    edge carrying the window and ``seqstart_k``.  Getting a split point wrong
    produces a plausible result: the row still sums to 1, nothing is NaN, it
    just attended to a slightly wrong set of keys.

    So the assertion is an identity rather than a tolerance.  Asking for causal
    as a *parameter* routes through the split; asking for the same triangle as
    an ``-inf`` bias routes through the unmasked interior with a bias add.  The
    two must agree exactly, at every ``T_k`` from one K tile to five.
    """

    @pytest.mark.parametrize("T_k", [1, 15, 16, 32, 63, 64, 65, 128, 129, 192, 256, 320])
    def test_causal_as_parameter_equals_causal_as_bias(self, fmha_cxx, T_k):
        B, H, D = 2, 4, 64
        T_q = T_k
        q, k, v = _qkv(B, H, T_q, T_k, D, seed=T_k)
        bias = window_bias(T_q, T_k, causal=True, dtype=q.dtype, device=q.device).expand(
            B, H, T_q, T_k
        )
        by_param = fmha_cxx(q, k, v, softmax_scale=0.125, causal=True)
        by_bias = fmha_cxx(q, k, v, softmax_scale=0.125, attn_bias=bias)
        torch.testing.assert_close(by_param, by_bias, atol=0, rtol=0)

    @pytest.mark.parametrize("T_q,T_k", [(8, 256), (16, 129), (64, 64), (128, 320), (1, 512)])
    def test_lengths_and_starts_compose_with_the_split(self, fmha_cxx, T_q, T_k):
        B, H, D = 2, 4, 64
        q, k, v = _qkv(B, H, T_q, T_k, D, seed=T_q + T_k)
        lens = torch.tensor([T_k, max(1, T_k // 3)], dtype=torch.int32, device="cuda")
        starts = torch.tensor([0, min(70, T_k // 4)], dtype=torch.int32, device="cuda")
        got = fmha_cxx(q, k, v, softmax_scale=0.125, cache_seqlens=lens, cache_seqstarts=starts)
        want = ref_fmha(q, k, v, 0.125, cache_seqlens=lens, cache_seqstarts=starts)
        torch.testing.assert_close(got, want, **FMHA_TOL)


class TestSlidingWindow:
    """A per-row window: this lane has one, the CuTeDSL lane has no argument for it.

    There was no kernel-level window test of any kind before this class.
    """

    @pytest.mark.parametrize(
        "window_left,window_right",
        [(-1, 0), (0, 0), (1, 0), (8, 0), (16, 16), (0, 16), (-1, 16), (8, -1), (31, 1)],
    )
    @pytest.mark.parametrize("T", [17, 64, 96, 200])
    def test_window_as_parameter_equals_window_as_bias(
        self, fmha_cxx, window_left, window_right, T
    ):
        B, H, D = 2, 4, 64
        q, k, v = _qkv(B, H, T, T, D, seed=T + window_left)
        bias = window_bias(
            T,
            T,
            window_left=window_left,
            window_right=window_right,
            dtype=q.dtype,
            device=q.device,
        ).expand(B, H, T, T)
        by_param = fmha_cxx(
            q,
            k,
            v,
            softmax_scale=0.125,
            window_left=window_left,
            window_right=window_right,
        )
        by_bias = fmha_cxx(q, k, v, softmax_scale=0.125, attn_bias=bias)
        torch.testing.assert_close(
            by_param,
            by_bias,
            atol=0,
            rtol=0,
            msg=f"window ({window_left}, {window_right}) at T={T}",
        )

    def test_an_unbounded_right_window_does_not_mask_the_diagonal(self, fmha_cxx):
        """``window_right == -1`` means unbounded, not "one column short".

        The natural reading of the bound arithmetic makes ``-1`` shrink the
        right limit by a column, which masks the diagonal itself -- and that is
        invisible from any aggregate: the row still sums to 1 and stays finite.
        Pinned against the no-window answer, which is what unbounded-both-ways
        must reduce to.
        """
        B, H, T, D = 1, 2, 96, 64
        q, k, v = _qkv(B, H, T, T, D, seed=3)
        windowed = fmha_cxx(q, k, v, softmax_scale=0.125, window_left=T, window_right=-1)
        plain = fmha_cxx(q, k, v, softmax_scale=0.125)
        torch.testing.assert_close(windowed, plain, atol=0, rtol=0)

    def test_window_composes_with_cache_seqlens(self, fmha_cxx):
        B, H, T, D = 2, 4, 96, 64
        q, k, v = _qkv(B, H, T, T, D, seed=11)
        lens = torch.tensor([T, 40], dtype=torch.int32, device="cuda")
        got = fmha_cxx(
            q, k, v, softmax_scale=0.125, cache_seqlens=lens, window_left=24, window_right=0
        )
        want = ref_fmha(q, k, v, 0.125, cache_seqlens=lens, window_left=24, window_right=0)
        torch.testing.assert_close(got, want, **FMHA_TOL)

    def test_the_cute_backend_refuses_a_window_rather_than_ignoring_it(self):
        """Naming a backend means requiring it (rule 3, restated for attention)."""
        pytest.importorskip("cutlass")
        from oasr.functionals.attention import fmha

        q, k, v = _qkv(1, 2, 32, 32, 64)
        with pytest.raises(NotImplementedError, match="sliding window"):
            fmha(q, k, v, softmax_scale=0.125, backend="cute", window_left=8, window_right=0)


# ---------------------------------------------------------------------------
# Varlen through the same instantiation
# ---------------------------------------------------------------------------


def _pack(lens_q, lens_k, H, H_kv, D, dtype=torch.float16, seed=0, bias=False):
    torch.manual_seed(seed)
    dev = "cuda"
    tq, tk = sum(lens_q), sum(lens_k)
    q = torch.randn(tq, H, D, dtype=dtype, device=dev)
    k = torch.randn(tk, H_kv, D, dtype=dtype, device=dev)
    v = torch.randn(tk, H_kv, D, dtype=dtype, device=dev)
    cu_q = torch.tensor(
        [0] + torch.tensor(lens_q).cumsum(0).tolist(), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + torch.tensor(lens_k).cumsum(0).tolist(), dtype=torch.int32, device=dev
    )
    if not bias:
        return q, k, v, cu_q, cu_k, None, None
    sizes = [H * a * b for a, b in zip(lens_q, lens_k)]
    offs = torch.tensor([0] + torch.tensor(sizes).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    flat = torch.randn(int(offs[-1]), dtype=dtype, device=dev)
    return q, k, v, cu_q, cu_k, flat, offs


class TestVarlenIsTheDenseKernel:
    """Packed inputs run the *same* compiled variant as dense ones.

    That is the design claim -- a packed tensor is a dense one with batch
    stride 0 and a per-segment row offset -- so the test is an equality against
    a per-segment dense call, not a tolerance against SDPA.
    """

    @pytest.mark.parametrize(
        "lens", [[8, 13, 64], [64, 64], [1, 1, 1], [37, 90], [128], [3, 5, 7, 11], [200, 7]]
    )
    @pytest.mark.parametrize("causal", [False, True])
    def test_packed_equals_per_segment_dense(self, fmha_cxx, fmha_varlen_cxx, lens, causal):
        H, H_kv, D = 4, 2, 64
        q, k, v, cu_q, cu_k, _b, _o = _pack(lens, lens, H, H_kv, D, seed=len(lens))
        got = fmha_varlen_cxx(
            q,
            k,
            v,
            softmax_scale=0.125,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(lens),
            max_seqlen_k=max(lens),
            causal=causal,
        )
        for s, (a, b) in enumerate(zip(cu_q[:-1].tolist(), cu_q[1:].tolist())):
            qs = q[a:b].transpose(0, 1).unsqueeze(0)
            ks = k[a:b].transpose(0, 1).unsqueeze(0)
            vs = v[a:b].transpose(0, 1).unsqueeze(0)
            want = fmha_cxx(qs, ks, vs, softmax_scale=0.125, causal=causal)
            torch.testing.assert_close(
                got[a:b].transpose(0, 1).unsqueeze(0),
                want,
                atol=0,
                rtol=0,
                msg=f"segment {s} of {lens} differs from the same rows run dense",
            )

    @pytest.mark.parametrize("lens", [[8, 13, 64], [16, 16, 16], [33, 17], [5, 64, 5]])
    def test_a_packed_bias_uses_per_segment_strides(self, fmha_cxx, fmha_varlen_cxx, lens):
        """Segment ``s``'s bias block is ``(H, T_q_s, T_k_s)`` -- its *own* size.

        The row stride of a packed block-diagonal bias is that segment's
        ``seqlen_k``, and the head stride ``seqlen_q * seqlen_k``.  Neither can
        come from a host-side constant, so the kernel derives both.  A
        "use the max" launcher is the plausible wrong answer, and it is correct
        on every segment that happens to *be* the max -- which is why this is
        parametrised over uneven lengths and why ``[16, 16, 16]`` is included
        as the case that would pass either way.
        """
        H, H_kv, D = 4, 2, 64
        q, k, v, cu_q, cu_k, flat, offs = _pack(lens, lens, H, H_kv, D, seed=sum(lens), bias=True)
        got = fmha_varlen_cxx(
            q,
            k,
            v,
            softmax_scale=0.125,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(lens),
            max_seqlen_k=max(lens),
            attn_bias=flat,
            bias_offsets=offs,
        )
        for s, (a, b) in enumerate(zip(cu_q[:-1].tolist(), cu_q[1:].tolist())):
            n = b - a
            bias_s = flat[offs[s] : offs[s + 1]].view(1, H, n, n)
            qs = q[a:b].transpose(0, 1).unsqueeze(0)
            ks = k[a:b].transpose(0, 1).unsqueeze(0)
            vs = v[a:b].transpose(0, 1).unsqueeze(0)
            want = fmha_cxx(qs, ks, vs, softmax_scale=0.125, attn_bias=bias_s.clone())
            torch.testing.assert_close(
                got[a:b].transpose(0, 1).unsqueeze(0),
                want,
                atol=0,
                rtol=0,
                msg=f"segment {s} of {lens}: packed bias read with the wrong stride",
            )

    def test_varlen_names_a_backend_and_gets_it(self):
        """It used to fall back to SDPA in silence where ``fmha`` raises."""
        from oasr.functionals.attention import fmha_varlen

        q, k, v, cu_q, cu_k, _b, _o = _pack([8, 8], [8, 8], 4, 4, 64)
        with pytest.raises(NotImplementedError):
            fmha_varlen(
                q.float(),
                k.float(),
                v.float(),
                softmax_scale=0.125,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=8,
                max_seqlen_k=8,
                backend="cxx",
            )


# ---------------------------------------------------------------------------
# Graph capture and determinism -- rule 11 had no attention pin
# ---------------------------------------------------------------------------


class TestGraphReplay:
    """A replay must recompute, and must agree with eager exactly.

    Rule 11 is about dispatch never branching on capture state.  The way that
    rule fails in practice is a kernel that picks a different tile, a different
    split count or a different code path under capture, and the resulting
    one-ulp difference has changed decoded tokens before.  ``atol=0`` is
    therefore the only tolerance that tests the rule.
    """

    @pytest.mark.parametrize("causal", [False, True])
    def test_replay_recomputes_and_matches_eager(self, fmha_cxx, causal):
        B, H, T, D = 2, 4, 64, 64
        q, k, v = _qkv(B, H, T, T, D, seed=5)
        out = torch.empty(B, H, T, D, dtype=torch.float16, device="cuda")

        def launch():
            fmha_cxx(q, k, v, softmax_scale=0.125, causal=causal, out=out, validate=False)

        def mutate():
            torch.manual_seed(99)
            q.copy_(torch.randn_like(q))
            k.copy_(torch.randn_like(k))

        def expected():
            return fmha_cxx(q, k, v, softmax_scale=0.125, causal=causal)

        assert_graph_replay(launch, out=out, mutate=mutate, expected=expected, atol=0, rtol=0)

    def test_a_paged_replay_recomputes(self, fmha_cxx):
        B, H, D = 2, 4, 64
        page, npages = 16, 8
        torch.manual_seed(13)
        k = torch.randn(npages, page, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(npages, page, H, D, dtype=torch.float16, device="cuda")
        table = torch.arange(npages, dtype=torch.int32, device="cuda").view(B, -1)
        q = torch.randn(B, H, 8, D, dtype=torch.float16, device="cuda")
        lens = torch.tensor([page * (npages // B)] * B, dtype=torch.int32, device="cuda")
        out = torch.empty(B, H, 8, D, dtype=torch.float16, device="cuda")

        kw = {
            "softmax_scale": 0.125,
            "cache_seqlens": lens,
            "block_table": table,
            "validate": False,
        }

        assert_graph_replay(
            lambda: fmha_cxx(q, k, v, out=out, **kw),
            out=out,
            mutate=lambda: k.copy_(torch.randn_like(k)),
            expected=lambda: fmha_cxx(q, k, v, **kw),
            atol=0,
            rtol=0,
        )


class TestDeterminism:
    """Same inputs, same bits -- every time, and regardless of what ran before.

    An online-softmax kernel is only deterministic if its reduction order is a
    function of the shape alone.  A tile or split count chosen from anything
    else -- occupancy, a cached workspace, capture state -- shows up here and
    nowhere else.
    """

    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("T_k", [64, 200, 512])
    def test_repeated_calls_are_bit_identical(self, fmha_cxx, causal, T_k):
        q, k, v = _qkv(2, 4, 64, T_k, 64, seed=T_k)
        first = fmha_cxx(q, k, v, softmax_scale=0.125, causal=causal)
        for _ in range(4):
            again = fmha_cxx(q, k, v, softmax_scale=0.125, causal=causal)
            torch.testing.assert_close(again, first, atol=0, rtol=0)

    def test_a_batch_row_does_not_depend_on_its_neighbours(self, fmha_cxx):
        """Row ``b`` must compute the same alone as it does in a cohort.

        Streaming has no cohort-width invariance in general, but *attention*
        does: each (batch, head) tile is independent.  A shared reduction or a
        scheduler that balanced work across rows would break it, and the
        symptom downstream is a transcript that changes with batch composition.
        """
        B, H, T, D = 4, 4, 48, 64
        q, k, v = _qkv(B, H, T, T, D, seed=21)
        lens = torch.tensor([T, T // 2, T // 4, 1], dtype=torch.int32, device="cuda")
        cohort = fmha_cxx(q, k, v, softmax_scale=0.125, cache_seqlens=lens, causal=True)
        for b in range(B):
            solo = fmha_cxx(
                q[b : b + 1],
                k[b : b + 1],
                v[b : b + 1],
                softmax_scale=0.125,
                cache_seqlens=lens[b : b + 1],
                causal=True,
            )
            torch.testing.assert_close(cohort[b : b + 1], solo, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Split-KV (flash decoding)
# ---------------------------------------------------------------------------


class TestSplitCountIsPure:
    """`num_splits` is a function of shape and SM count, and of nothing else.

    AGENTS.md rule 11 says dispatch must never branch on CUDA-graph capture
    state.  Split-KV is the sharpest case in this family: splitting changes the
    order the fp32 partials are summed, so a capture-dependent count makes a
    replayed graph produce *different numbers* than eager -- and a one-ulp
    difference in attention has moved a decoded token in this repo before.
    """

    @pytest.mark.parametrize("num_sms", [1, 16, 68, 108, 132, 170])
    @pytest.mark.parametrize("cta_count", [1, 2, 4, 8, 32, 128, 170, 256, 1024])
    @pytest.mark.parametrize("n_blocks", [1, 2, 3, 8, 16, 64, 128, 1024])
    def test_python_mirror_matches_cxx(self, cta_count, n_blocks, num_sms):
        from oasr.jit.fmha import num_splits

        got = int(_config_probe().fmha_num_splits(cta_count, n_blocks, num_sms))
        assert got == num_splits(
            cta_count, n_blocks, num_sms
        ), f"split-count drift at cta={cta_count} n={n_blocks} sms={num_sms}"

    def test_a_full_grid_is_never_split(self):
        from oasr.jit.fmha import num_splits

        for sms in (68, 132, 170):
            assert num_splits(sms, 1024, sms) == 1
            assert num_splits(sms * 4, 1024, sms) == 1

    def test_a_split_always_owns_real_work(self):
        """No split may be created that would walk fewer than two K tiles."""
        from oasr.jit.fmha import MIN_BLOCKS_PER_SPLIT, num_splits

        for n in range(1, 400):
            s = num_splits(cta_count=1, n_blocks=n, num_sms=170)
            assert s >= 1
            if s > 1:
                assert n // s >= MIN_BLOCKS_PER_SPLIT - 1, (n, s)

    def test_a_short_k_extent_is_never_split(self):
        """The floor below which the combine pass cannot be repaid.

        A *measured* constant, not a guess -- the table is in
        ``include/oasr/attention/cutlass_fmha_configs.h``.  Eagerly, splitting
        a 16-tile range runs at 0.64x because the two workspace allocations and
        the second launch cost more than the occupancy buys back.
        """
        from oasr.jit.fmha import MIN_BLOCKS_FOR_SPLIT, num_splits

        for n in range(1, MIN_BLOCKS_FOR_SPLIT):
            assert num_splits(cta_count=1, n_blocks=n, num_sms=170) == 1, n
        assert num_splits(cta_count=1, n_blocks=MIN_BLOCKS_FOR_SPLIT, num_sms=170) > 1

    def test_the_ranges_tile_the_whole_k_extent_exactly_once(self):
        """Every K tile belongs to exactly one split -- no gap, no overlap.

        A gap silently drops keys (the row renormalises, so the output stays
        finite and plausible); an overlap double-counts them.  Neither shows up
        as anything but a slightly wrong number.
        """
        from oasr.jit.fmha import split_range

        for lo, hi in [(0, 1), (0, 17), (3, 20), (5, 5), (0, 128), (7, 8)]:
            for splits in (1, 2, 3, 5, 8, 16):
                covered = []
                for s in range(splits):
                    a, b = split_range(lo, hi, s, splits)
                    assert lo <= a <= b <= hi, (lo, hi, s, splits, a, b)
                    covered.extend(range(a, b))
                assert sorted(covered) == list(
                    range(lo, hi)
                ), f"[{lo},{hi}) into {splits} splits covered {sorted(covered)}"

    def test_the_routing_decision_does_not_move_under_graph_capture(self, fmha_cxx):
        """The rule, tested as the rule rather than as its consequence."""
        from oasr.functionals.attention import _split_count

        kw = {"B": 1, "H": 8, "T_q": 1, "T_k": 8192, "D": 64, "causal": False, "local": False}
        eager = _split_count(**kw)
        assert eager > 1, "pick a shape that actually splits, or this proves nothing"

        q, k, v = _qkv(1, 8, 1, 8192, 64)
        out = torch.empty(1, 8, 1, 64, dtype=torch.float16, device="cuda")
        g = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fmha_cxx(q, k, v, softmax_scale=0.125, out=out, validate=False)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        captured = []
        with torch.cuda.graph(g):
            captured.append(_split_count(**kw))
            fmha_cxx(q, k, v, softmax_scale=0.125, out=out, validate=False)
        assert captured == [eager], (
            f"the split count differed under capture ({captured[0]}) from eager "
            f"({eager}); rule 11"
        )


class TestSplitKV:
    """A split run must equal an unsplit one, not merely be close to SDPA."""

    @pytest.mark.parametrize(
        "B,H,H_kv,T_q,T_k",
        [
            (1, 8, 8, 1, 2048),
            (1, 8, 8, 1, 8192),
            (4, 8, 2, 1, 4096),
            (1, 32, 4, 1, 4096),
            (2, 4, 4, 8, 2048),
            (1, 8, 8, 64, 4096),  # a full M tile as well as an empty one
            (1, 8, 8, 1, 2100),  # T_k not a multiple of the K tile
            (3, 5, 5, 1, 2777),  # nor of the split count
        ],
    )
    def test_split_matches_sdpa(self, fmha_cxx, B, H, H_kv, T_q, T_k):
        from oasr.functionals.attention import _split_count

        assert (
            _split_count(B=B, H=H, T_q=T_q, T_k=T_k, D=64, causal=False, local=False) > 1
        ), "this shape must actually take the split path"
        q, k, v = _qkv(B, H, T_q, T_k, 64, H_kv=H_kv, seed=T_k)
        got = fmha_cxx(q, k, v, softmax_scale=0.125)
        want = ref_fmha(q, k, v, 0.125)
        torch.testing.assert_close(got, want, **FMHA_TOL)

    def test_split_composes_with_lengths_and_starts(self, fmha_cxx):
        """A split whose whole chunk is past `seqlen_k` must weigh exactly zero.

        It still runs -- the grid is a function of shape, not of the per-stream
        lengths, which is what makes a split launch capturable -- and writes a
        `-inf` LSE.  If that were a NaN instead, one dead split would poison
        every row of its stream.
        """
        B, H, T_k = 2, 8, 4096
        q, k, v = _qkv(B, H, 1, T_k, 64, seed=9)
        # Stream 1 uses a sixteenth of the cache, so most splits are empty.
        lens = torch.tensor([T_k, 200], dtype=torch.int32, device="cuda")
        starts = torch.tensor([0, 64], dtype=torch.int32, device="cuda")
        got = fmha_cxx(q, k, v, softmax_scale=0.125, cache_seqlens=lens, cache_seqstarts=starts)
        want = ref_fmha(q, k, v, 0.125, cache_seqlens=lens, cache_seqstarts=starts)
        assert torch.isfinite(got).all(), "an empty split leaked a NaN into its stream"
        torch.testing.assert_close(got, want, **FMHA_TOL)

    def test_a_fully_masked_row_is_zero_under_split(self, fmha_cxx):
        """Every split empty means no live key at all -- the answer is zero.

        The combine divides by the sum of the split weights, and when every
        split is `-inf` that sum is zero.  Dividing anyway gives NaN, which is
        not inert: the next layer's masked key still contributes `0 * NaN`.
        """
        B, H, T_k = 2, 8, 4096
        q, k, v = _qkv(B, H, 1, T_k, 64, seed=17)
        lens = torch.tensor([T_k, 0], dtype=torch.int32, device="cuda")
        got = fmha_cxx(q, k, v, softmax_scale=0.125, cache_seqlens=lens)
        assert torch.isfinite(got).all()
        assert torch.all(got[1] == 0), "a stream with no keys must come back exactly zero"

    def test_paged_split_matches_the_gathered_reference(self, fmha_cxx):
        B, H, D = 2, 8, 64
        page, per_seq = 32, 128
        npages = B * per_seq
        torch.manual_seed(23)
        k = torch.randn(npages, page, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(npages, page, H, D, dtype=torch.float16, device="cuda")
        table = torch.randperm(npages, device="cuda").to(torch.int32).view(B, per_seq)
        q = torch.randn(B, H, 1, D, dtype=torch.float16, device="cuda")
        T_k = page * per_seq
        lens = torch.tensor([T_k, T_k - 5], dtype=torch.int32, device="cuda")

        from oasr.functionals.attention import _split_count, gather_paged_kv

        assert _split_count(B=B, H=H, T_q=1, T_k=T_k, D=D, causal=False, local=False) > 1
        got = fmha_cxx(q, k, v, softmax_scale=0.125, cache_seqlens=lens, block_table=table)
        k_d, v_d = gather_paged_kv(k, v, table)
        want = ref_fmha(q, k_d, v_d, 0.125, cache_seqlens=lens)
        torch.testing.assert_close(got, want, **FMHA_TOL)

    def test_split_is_deterministic_and_graph_safe(self, fmha_cxx):
        B, H, T_k = 1, 8, 4096
        q, k, v = _qkv(B, H, 1, T_k, 64, seed=31)
        out = torch.empty(B, H, 1, 64, dtype=torch.float16, device="cuda")

        first = fmha_cxx(q, k, v, softmax_scale=0.125).clone()
        for _ in range(4):
            torch.testing.assert_close(
                fmha_cxx(q, k, v, softmax_scale=0.125), first, atol=0, rtol=0
            )

        assert_graph_replay(
            lambda: fmha_cxx(q, k, v, softmax_scale=0.125, out=out, validate=False),
            out=out,
            mutate=lambda: k.copy_(torch.randn_like(k)),
            expected=lambda: fmha_cxx(q, k, v, softmax_scale=0.125),
            atol=0,
            rtol=0,
        )


class TestAutoDegradesButANamedBackendDoesNot:
    """`backend=None` may re-route; `backend="cute"` may not.

    Two questions that look like one.  `select_backend()` answers "which lane
    does this process prefer", which cannot see the shape; whether *this* call
    is servable is a second question, and the sliding window is where they come
    apart -- the C++ lane has one and the CuTeDSL lane has no argument for it.

    Under `auto` a windowed call must reach a lane that can serve it. Under
    `backend="cute"` it must raise, because naming a backend is a requirement.
    Getting the first wrong makes a perfectly serviceable call fail on a
    machine that has the kernel two lanes over; getting the second wrong makes
    an A/B silently measure the same lane twice.
    """

    def test_auto_routes_a_windowed_dense_call_to_a_lane_that_can_serve_it(self):
        from oasr.functionals.attention import fmha

        q, k, v = _qkv(1, 4, 64, 64, 64, seed=41)
        got = fmha(q, k, v, softmax_scale=0.125, window_left=16, window_right=0)
        want = fmha(q, k, v, softmax_scale=0.125, window_left=16, window_right=0, backend="sdpa")
        torch.testing.assert_close(got, want, **FMHA_TOL)

    @pytest.mark.parametrize("kind", ["causal", "window"])
    def test_auto_routes_a_masked_varlen_call_too(self, kind):
        """The dense arbiter answers "cute" here, and would be wrong.

        The CuTeDSL kernel *does* do causal -- just not in its separate varlen
        kernel, which has no mask argument at all. Asking the dense question
        about a packed call is the bug this pins.
        """
        from oasr.functionals.attention import fmha_varlen

        q, k, v, cu_q, cu_k, _b, _o = _pack([8, 13, 64], [8, 13, 64], 4, 4, 64, seed=43)
        mask = {"causal": True} if kind == "causal" else {"window_left": 16, "window_right": 0}
        kw = {
            "softmax_scale": 0.125,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": 64,
            "max_seqlen_k": 64,
            **mask,
        }
        got = fmha_varlen(q, k, v, **kw)
        want = fmha_varlen(q, k, v, backend="sdpa", **kw)
        torch.testing.assert_close(got, want, **FMHA_TOL)

    def test_naming_cute_still_raises_for_both(self):
        pytest.importorskip("cutlass")
        from oasr.functionals.attention import fmha, fmha_varlen

        q, k, v = _qkv(1, 4, 64, 64, 64, seed=45)
        with pytest.raises(NotImplementedError):
            fmha(
                q,
                k,
                v,
                softmax_scale=0.125,
                window_left=16,
                window_right=0,
                backend="cute",
            )
        pq, pk, pv, cu_q, cu_k, _b, _o = _pack([8, 8], [8, 8], 4, 4, 64, seed=47)
        with pytest.raises(NotImplementedError):
            fmha_varlen(
                pq,
                pk,
                pv,
                softmax_scale=0.125,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=8,
                max_seqlen_k=8,
                causal=True,
                backend="cute",
            )
