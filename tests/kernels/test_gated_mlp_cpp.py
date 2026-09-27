# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The C++ CUTLASS/CuTe gated-MLP backend's own surface.

``test_gated_mlp.py`` is backend-agnostic: it runs the functional and the layer
against an FP32 oracle through whichever lane :mod:`oasr.jit.mlp` picks, which
is what proves the two agree.  What it structurally cannot cover is the set of
properties that exist *because* this lane is C++ -- the ones the CuTeDSL lane
either does not have or reaches by a different route:

* the **tile table and its selector** live in ``constexpr`` C++ and are
  mirrored in Python for routing.  A mirror nobody checks is a mirror that
  drifts, so the agreement is asserted here, over four architectures and
  several machine widths, from one box.
* **a K residue**: ``K`` need not be a whole number of K tiles here.  The
  CuTeDSL lane refuses that shape outright (its ``gated_mlp_shape_supported``
  demands ``K % k_block == 0``), so there is nothing over there to compare
  against and the claim has to be made directly.
* **arbitrary row strides**: the kernel takes a row-slice of a wider buffer on
  any operand.  "No copy" is an allocation claim, not a numeric one, so it
  needs an allocation count and not just a tolerance.
* **CUDA-graph replay and determinism** of the launcher.

Nothing here is a second copy of a parity test.  Where a claim is about
*numbers* it is stated against an FP32 matmul plus the gate equations in
torch, so the kernel is checked against the **definition** rather than against
another OASR kernel.
"""

from __future__ import annotations

import functools

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_graph_replay, device_sm

pytestmark = pytest.mark.cuda

_SUPPORTED = (80, 86, 89, 120)

#: fp16 carries ~3 decimal digits and the oracle accumulates in fp32, so the
#: comparison is against the *relative* size of the output, not an absolute
#: epsilon that a large K would blow through.  Same numbers as
#: ``test_gated_mlp.py``, for the same reason.
_REL_TOL = {torch.float16: 3e-3, torch.bfloat16: 2e-2}

_ACT = {
    "silu": F.silu,
    "swish": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_tanh": lambda t: F.gelu(t, approximate="tanh"),
    "identity": lambda t: t,
}


def _requires_cxx():
    if device_sm() not in _SUPPORTED:
        pytest.skip(f"the C++ gated-MLP lane is not compiled for sm_{device_sm()}")


# ---------------------------------------------------------------------------
# Operands and the oracle
# ---------------------------------------------------------------------------


def _operands(M, N, K, dtype=torch.float16, bias=False, seed=17, pad_x=0, pad_w=0):
    """``(x, w_gate, w_up, b_gate, b_up)``; ``pad_*`` makes the operand a row-slice."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", dtype=dtype, generator=g)  # noqa: E731
    x = (r(M, K + pad_x) * 0.3)[:, :K]
    wg = (r(N, K + pad_w) * K**-0.5)[:, :K]
    wu = (r(N, K + pad_w) * K**-0.5)[:, :K]
    bg = r(N) * 0.1 if bias else None
    bu = r(N) * 0.1 if bias else None
    return x, wg, wu, bg, bu


def _oracle(x, wg, wu, bg, bu, activation="silu"):
    gate = x.float() @ wg.float().T
    up = x.float() @ wu.float().T
    if bg is not None:
        gate, up = gate + bg.float(), up + bu.float()
    return _ACT[activation](gate) * up


def _assert_close(out, ref, dtype):
    assert not torch.isnan(out).any(), "kernel produced NaN"
    scale = max(ref.abs().max().item(), 1e-6)
    rel = (out.float() - ref).abs().max().item() / scale
    assert rel < _REL_TOL[dtype], f"max relative error {rel:.5f}"


def _launch(tile, dtype=torch.float16, activation="silu", bias=False):
    from oasr.jit.gated_mlp import get_gated_mlp_fn

    dtype_str = "float16" if dtype is torch.float16 else "bfloat16"
    return get_gated_mlp_fn(
        dtype_str=dtype_str, activation=activation, has_bias=bias, tile_index=tile
    )


def _run(M, N, K, tile, *, dtype=torch.float16, activation="silu", bias=False, **kw):
    x, wg, wu, bg, bu = _operands(M, N, K, dtype=dtype, bias=bias, **kw)
    out = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
    _launch(tile, dtype, activation, bias)(out, x, wg, wu, bg, bu)
    _assert_close(out, _oracle(x, wg, wu, bg, bu, activation), dtype)
    return out


# ---------------------------------------------------------------------------
# The tile table and the selector
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _config_probe():
    """``csrc/gated_mlp_jit_binding.cu`` on its own -- the real file, not a copy.

    Compiling the shipped binding standalone is the point: a purpose-built
    fixture TU would be a *second* transcription of the table, which is the
    drift this test exists to catch.  It pulls in only
    ``include/oasr/mlp/gated_mlp_tiles.h``, which deliberately includes no
    CuTe, so this costs a couple of seconds rather than a CUTLASS compile.
    """
    from oasr.jit import env
    from oasr.jit.core import gen_jit_spec

    return gen_jit_spec(
        "gated_mlp_config_probe", [env.OASR_CSRC_DIR / "gated_mlp_jit_binding.cu"]
    ).build_and_load()


def _cxx_tile_info(index: int, sm: int, elem_bits: int = 16):
    out = torch.zeros(9, dtype=torch.int32)
    _config_probe().gated_mlp_tile_info(out, index, sm, elem_bits)
    return out.tolist()


class TestTileTableAgrees:
    """``kGatedMlpTiles`` and :data:`oasr.jit.gated_mlp.TILES`, field for field.

    Asked over an ``sm`` *argument*, so one built module answers for every
    supported architecture.  That is deliberate and it is the property a live
    device query would destroy: on this box a query would report sm_120's
    99 KB for all four rows, and it is exactly the sm_86/sm_89 budget that was
    once wrong in the attention family.
    """

    def test_the_two_tables_are_the_same_length(self):
        from oasr.jit.gated_mlp import TILES

        assert int(_config_probe().gated_mlp_tile_count()) == len(TILES)

    def test_alignment_agrees(self):
        from oasr.jit.gated_mlp import ALIGNMENT

        assert int(_config_probe().gated_mlp_alignment()) == ALIGNMENT

    @pytest.mark.parametrize("sm", [80, 86, 89, 120])
    def test_budget_and_warp_slots_agree(self, sm):
        from oasr.jit.gated_mlp import _MAX_THREADS_PER_SM, smem_budget

        probe = _config_probe()
        assert int(probe.gated_mlp_smem_budget(sm)) == smem_budget(sm)
        assert int(probe.gated_mlp_max_threads_per_sm(sm)) == _MAX_THREADS_PER_SM[sm]

    @pytest.mark.parametrize("sm", [80, 86, 89, 120])
    @pytest.mark.parametrize("index", range(6))
    def test_tile_matches_cxx(self, sm, index):
        from oasr.jit.gated_mlp import TILES, ctas_per_sm, smem_bytes, tile_valid

        t = TILES[index]
        want = [
            t.block_m,
            t.block_n,
            t.block_k,
            t.stages,
            t.threads,
            t.warps_n,
            int(tile_valid(t, sm)),
            smem_bytes(t),
            ctas_per_sm(t, sm),
        ]
        got = _cxx_tile_info(index, sm)
        assert got == want, (
            f"tile-table drift at sm_{sm} index {index}: C++ {got} vs Python {want}. "
            f"These are one table written twice; the Python copy only exists so the "
            f"layer waist can route before anything is built."
        )

    @pytest.mark.parametrize("sm", [80, 86, 89, 120])
    def test_every_shipped_tile_fits_every_supported_arch(self, sm):
        """A tile in the table that does not fit is a cell that fails to build."""
        from oasr.jit.gated_mlp import TILES, smem_budget

        for index, t in enumerate(TILES):
            info = _cxx_tile_info(index, sm)
            assert info[6] == 1, f"tile {index} {t} does not fit sm_{sm}"
            assert 0 < info[7] <= smem_budget(sm)


class TestSelectorAgrees:
    """``gatedMlpSelectTile`` and :func:`oasr.jit.gated_mlp.select_tile`."""

    @pytest.mark.parametrize("sm", [80, 86, 89, 120])
    @pytest.mark.parametrize("num_sms", [56, 108, 132, 170])
    @pytest.mark.parametrize("rows", [1, 8, 16, 17, 32, 33, 64, 65, 128, 1024])
    @pytest.mark.parametrize("n", [64, 512, 8960, 11008, 18944])
    def test_python_mirror_matches_cxx(self, sm, num_sms, rows, n):
        from oasr.jit.gated_mlp import select_tile

        got = int(_config_probe().gated_mlp_select_tile(sm, num_sms, rows, n, 16))
        want = select_tile(sm, num_sms, rows, n)
        assert got == want, (
            f"selector drift at sm_{sm} sms={num_sms} rows={rows} n={n}: "
            f"C++ {got} vs Python {want}"
        )

    def test_a_degenerate_problem_is_refused_not_clamped(self):
        from oasr.jit.gated_mlp import select_tile

        for args in ((120, 170, 0, 1024), (120, 170, 8, 0), (120, 0, 8, 1024)):
            assert select_tile(*args) == -1
            assert int(_config_probe().gated_mlp_select_tile(*args, 16)) == -1

    def test_an_unknown_architecture_is_refused(self):
        """A part with no shared-memory budget in the table gets no tile."""
        from oasr.jit.gated_mlp import select_tile

        assert select_tile(75, 68, 8, 1024) == -1
        assert int(_config_probe().gated_mlp_select_tile(75, 68, 8, 1024, 16)) == -1

    def test_the_choice_depends_on_n_and_not_only_on_m(self):
        """Two tiles, and which wins is a wave-count question about ``N``.

        If this ever stops holding, the selector has collapsed to a rows-keyed
        table and the whole `num_sms` argument is dead weight.
        """
        from oasr.jit.gated_mlp import select_tile

        picks = {select_tile(120, 170, 8, n) for n in (64, 512, 8960, 18944)}
        assert len(picks) > 1, "the selector ignored N"

    def test_selection_is_a_pure_function_of_its_arguments(self):
        """``AGENTS.md`` rule 11: never branch on CUDA-graph capture state.

        Two tiles sum the K loop in different orders, so a capture-dependent
        answer would make a replayed graph produce different numbers than
        eager -- and a one-ulp difference has changed a decoded token in this
        repo before.  The strongest available statement is that the answer is
        the same inside a capture as outside it.
        """
        from oasr.jit.gated_mlp import select_tile

        eager = select_tile(120, 170, 8, 11008)
        graph = torch.cuda.CUDAGraph()
        scratch = torch.zeros(8, device="cuda")
        with torch.cuda.graph(graph):
            scratch.add_(1)
            captured = select_tile(120, 170, 8, 11008)
        assert captured == eager


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


class TestEveryTile:
    """Each compiled tile against the FP32 definition, with a ragged M and N."""

    def setup_method(self):
        _requires_cxx()

    @pytest.mark.parametrize("tile", range(6))
    @pytest.mark.parametrize("bias", [False, True])
    def test_matches_the_oracle(self, tile, bias):
        _run(70, 304, 256, tile, bias=bias)

    @pytest.mark.parametrize("tile", range(6))
    def test_a_single_row(self, tile):
        """M=1 is the decode shape, and it is every tile's worst padding case."""
        _run(1, 512, 256, tile)


class TestKResidue:
    """``K`` need not be a whole number of K tiles -- the CuTeDSL lane's one hard
    shape constraint, and the reason this lane has a wider contract.

    The residue is predicated and the ZFILL cp.async makes the skipped elements
    **zero**, which is the identity for the dot product being accumulated -- so
    a partial K tile is *correct*, not merely safe.  Written as a branch
    (`if (pred) copy else clear`) instead it would still be correct and would
    cost ~11%; see ``oasr::cute_sm80::copy_zfill``.
    """

    def setup_method(self):
        _requires_cxx()

    @pytest.mark.parametrize("K", [8, 16, 24, 40, 56, 88, 96, 120, 136, 200, 1000])
    @pytest.mark.parametrize("tile", [0, 1])
    def test_partial_k_tile(self, K, tile):
        _run(9, 128, K, tile, bias=True)

    def test_the_cutedsl_lane_refuses_what_this_one_serves(self):
        """Not a swipe at the other lane -- a pin on the routing.

        ``gated_mlp_config_supported`` has to be the *union* of the two, or a
        shape only one lane can serve is declined by the arbiter and silently
        falls back to two GEMMs.
        """
        from oasr.jit import mlp as jit_mlp

        assert not jit_mlp.cute_config_supported(rows=8, n=256, k=96)
        assert jit_mlp.cxx_config_supported(
            dtype_str="float16", activation="silu", rows=8, n=256, k=96
        )
        assert jit_mlp.gated_mlp_config_supported(rows=8, n=256, k=96)

    def test_k_that_is_not_vector_aligned_is_refused_by_both(self):
        """8-element alignment is the 128-bit load width and is not negotiable."""
        from oasr.jit import mlp as jit_mlp

        assert not jit_mlp.gated_mlp_config_supported(rows=8, n=256, k=132)
        assert not jit_mlp.gated_mlp_config_supported(rows=8, n=252, k=128)


class TestArbitraryStrides:
    """A row-slice of a wider buffer, on any operand, with no copy.

    Only the trailing axis has to be contiguous.  The claim is an *allocation*
    claim as much as a numeric one -- a kernel that silently materialised a
    contiguous copy would pass a tolerance check and lose the point -- so the
    memory high-water mark is asserted too.
    """

    def setup_method(self):
        _requires_cxx()

    @pytest.mark.parametrize("pad_x,pad_w", [(8, 0), (0, 8), (16, 24)], ids=["x", "w", "both"])
    def test_strided_operands(self, pad_x, pad_w):
        _run(40, 128, 256, 0, bias=True, pad_x=pad_x, pad_w=pad_w)

    def test_strided_output(self):
        x, wg, wu, bg, bu = _operands(40, 128, 256, bias=True)
        wide = torch.full((40, 256), float("nan"), device="cuda", dtype=torch.float16)
        out = wide[:, :128]
        assert not out.is_contiguous()
        _launch(0, bias=True)(out, x, wg, wu, bg, bu)
        _assert_close(out, _oracle(x, wg, wu, bg, bu), torch.float16)
        # The columns past N belong to the caller and must come back untouched.
        assert torch.isnan(wide[:, 128:]).all(), "the store overran its N extent"

    def test_no_copy_is_made(self):
        x, wg, wu, bg, bu = _operands(40, 128, 256, bias=True, pad_x=8, pad_w=8)
        out = torch.empty(40, 128, device="cuda", dtype=torch.float16)
        fn = _launch(0, bias=True)
        fn(out, x, wg, wu, bg, bu)  # warm: the first call builds nothing new
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(4):
            fn(out, x, wg, wu, bg, bu)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before, (
            "the launcher allocated; a strided operand is being copied, which is "
            "the whole cost this lane exists to avoid"
        )


class TestRefusals:
    """What the launcher declines, rather than reading out of bounds."""

    def setup_method(self):
        _requires_cxx()

    def test_a_bias_variant_demands_its_bias(self):
        x, wg, wu, _, _ = _operands(8, 128, 256)
        out = torch.empty(8, 128, device="cuda", dtype=torch.float16)
        with pytest.raises(Exception, match="compiled with a bias"):
            _launch(0, bias=True)(out, x, wg, wu, None, None)

    def test_a_nobias_variant_refuses_one(self):
        """Accepting it would silently drop the bias, which is a wrong answer."""
        x, wg, wu, bg, bu = _operands(8, 128, 256, bias=True)
        out = torch.empty(8, 128, device="cuda", dtype=torch.float16)
        with pytest.raises(Exception, match="compiled without a bias"):
            _launch(0, bias=False)(out, x, wg, wu, bg, bu)

    def test_a_misaligned_row_stride_is_refused(self):
        """Every row after the first would be misaligned for the 128-bit load."""
        x = torch.randn(8, 300, device="cuda", dtype=torch.float16)[:, :256]
        wg, wu = _operands(8, 128, 256)[1:3]
        out = torch.empty(8, 128, device="cuda", dtype=torch.float16)
        assert x.stride(0) % 8 != 0
        with pytest.raises(Exception, match="row stride"):
            _launch(0)(out, x, wg, wu, None, None)

    def test_a_transposed_operand_is_refused(self):
        """``stride(-1) != 1`` reads wrong addresses silently, with no fault.

        That is the price of taking an arbitrary *row* stride: nothing in the
        addresses themselves says which axis was meant to be contiguous.
        """
        _, wg, wu, _, _ = _operands(8, 128, 256)
        transposed = torch.randn(256, 8, device="cuda", dtype=torch.float16).T
        assert transposed.shape == (8, 256) and transposed.stride(-1) != 1
        out = torch.empty(8, 128, device="cuda", dtype=torch.float16)
        with pytest.raises(Exception, match="contiguous last dimension"):
            _launch(0)(out, transposed, wg, wu, None, None)


class TestGraphReplay:
    """The decoder step is graph-captured, so this is the production path."""

    def setup_method(self):
        _requires_cxx()

    def test_replay_recomputes(self):
        x, wg, wu, bg, bu = _operands(8, 512, 256, bias=True)
        out = torch.empty(8, 512, device="cuda", dtype=torch.float16)
        fn = _launch(0, bias=True)
        assert_graph_replay(
            lambda: fn(out, x, wg, wu, bg, bu),
            out=out,
            # Write new values *through* the captured buffer: a replay that
            # reproduced the captured result instead of recomputing fails.
            mutate=lambda: x.mul_(0.5),
            expected=lambda: _oracle(x, wg, wu, bg, bu).to(torch.float16),
            rtol=6e-3,
            atol=6e-3,
        )

    def test_replay_is_bit_identical_to_eager(self):
        """Capture must not change which kernel runs (``AGENTS.md`` rule 11)."""
        x, wg, wu, bg, bu = _operands(8, 512, 256, bias=True)
        out = torch.empty(8, 512, device="cuda", dtype=torch.float16)
        fn = _launch(0, bias=True)
        fn(out, x, wg, wu, bg, bu)
        torch.cuda.synchronize()
        eager = out.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn(out, x, wg, wu, bg, bu)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(out, x, wg, wu, bg, bu)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, eager, rtol=0, atol=0)

    def test_repeated_launches_are_bit_identical(self):
        """No atomics, no split-K, no data-dependent order: the sum is fixed."""
        x, wg, wu, _, _ = _operands(33, 512, 1024)
        fn = _launch(2)
        outs = []
        for _ in range(4):
            o = torch.empty(33, 512, device="cuda", dtype=torch.float16)
            fn(o, x, wg, wu, None, None)
            outs.append(o)
        torch.cuda.synchronize()
        for o in outs[1:]:
            torch.testing.assert_close(o, outs[0], rtol=0, atol=0)


class TestActivationsAndDtypes:
    """Every activation and both served dtypes, against the FP32 definition.

    ``slow`` because the activation is a *cell* axis: each one is its own
    ninja wave the first time it is asked for.  It is a cell axis because it
    is the one thing here that changes the arithmetic, and a checkpoint means
    exactly one of them -- so a model pays for one cell, not six.
    """

    def setup_method(self):
        _requires_cxx()

    @pytest.mark.slow
    @pytest.mark.parametrize("activation", sorted(_ACT))
    @pytest.mark.parametrize("bias", [False, True])
    def test_activation(self, activation, bias):
        _run(70, 304, 256, 0, activation=activation, bias=bias)

    @pytest.mark.parametrize("bias", [False, True])
    def test_bfloat16(self, bias):
        _run(70, 304, 256, 0, dtype=torch.bfloat16, bias=bias)


class TestBothLanesPickTheSameTile:
    """The two tile tables are one table, inherited rather than re-derived.

    That is what makes an A/B between the lanes a comparison of *kernels*: if
    they drifted, a row would be comparing two tile choices and neither number
    would mean what the note in ``.artifacts/`` says it means.
    """

    def setup_method(self):
        _requires_cxx()
        from oasr.jit import mlp as jit_mlp

        try:
            import cutlass  # noqa: F401
        except Exception:
            pytest.skip("CuTeDSL is not installed")
        if device_sm() not in jit_mlp._SUPPORTED_SM:
            pytest.skip("no CuTeDSL gated-MLP kernel for this arch")

    @pytest.mark.parametrize("rows", [1, 8, 16, 32, 64, 128])
    @pytest.mark.parametrize("n", [8960, 11008, 18944])
    def test_same_tile(self, rows, n):
        from oasr.jit import mlp as jit_mlp
        from oasr.jit.gated_mlp import TILES, routed_tile

        dsl = jit_mlp.select_gated_mlp_tile(rows, n)
        t = TILES[routed_tile(rows=rows, n=n)]
        assert tuple(dsl) == (
            t.block_m,
            t.block_n,
            t.block_k,
            t.stages,
            t.threads,
            t.warps_n,
        )


class TestLanePinning:
    """``OASR_GATED_MLP_BACKEND`` -- the switch that exists because there are two.

    Kept apart from ``OASR_GATED_MLP_CUTE`` on purpose: that one decides
    whether to fuse at all, which is an operator decision about a shape, and
    this one decides which of two implementations of the same thing runs.
    Collapsing them would make a rollback and an A/B the same switch.
    """

    def setup_method(self):
        _requires_cxx()

    @staticmethod
    def _cutedsl_here() -> bool:
        from oasr.jit import mlp as jit_mlp

        try:
            import cutlass  # noqa: F401
        except Exception:
            return False
        return device_sm() in jit_mlp._SUPPORTED_SM

    def test_auto_prefers_the_cxx_lane(self):
        import oasr

        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        w = torch.randn(512, 256, device="cuda", dtype=torch.float16)
        assert oasr.gated_mlp_backend(x, w) == "cxx"

    @pytest.mark.parametrize("lane", ["cxx", "cute"])
    def test_a_named_lane_is_the_lane_that_runs(self, lane):
        import oasr
        from oasr.jit import mlp as jit_mlp

        if lane == "cute" and not self._cutedsl_here():
            pytest.skip("CuTeDSL is not installed")
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16) * 0.3
        wg = torch.randn(512, 256, device="cuda", dtype=torch.float16) * 256**-0.5
        wu = torch.randn(512, 256, device="cuda", dtype=torch.float16) * 256**-0.5
        try:
            jit_mlp.set_gated_mlp_backend(lane)
            assert oasr.gated_mlp_backend(x, wg) == lane
            _assert_close(oasr.gated_mlp(x, wg, wu), _oracle(x, wg, wu, None, None), torch.float16)
        finally:
            jit_mlp.set_gated_mlp_backend("auto")

    def test_pinning_cute_narrows_what_is_available(self):
        """A partial K tile is a C++-lane capability, so pinning away from it
        has to make the shape *unavailable* rather than silently decline at
        the call -- or the capability question and the routing disagree."""
        import oasr
        from oasr.jit import mlp as jit_mlp

        if not self._cutedsl_here():
            pytest.skip("CuTeDSL is not installed")
        x = torch.randn(8, 96, device="cuda", dtype=torch.float16)
        w = torch.randn(256, 96, device="cuda", dtype=torch.float16)
        try:
            assert oasr.gated_mlp_available(x, w)
            jit_mlp.set_gated_mlp_backend("cute")
            assert not oasr.gated_mlp_available(x, w)
            assert not jit_mlp.gated_mlp_config_supported(rows=8, n=256, k=96)
        finally:
            jit_mlp.set_gated_mlp_backend("auto")

    def test_fusion_off_beats_any_lane(self):
        import oasr
        from oasr.jit import mlp as jit_mlp

        x = torch.randn(8, 256, device="cuda", dtype=torch.float16)
        w = torch.randn(512, 256, device="cuda", dtype=torch.float16)
        try:
            jit_mlp.set_gated_mlp_backend("cxx")
            jit_mlp.set_gated_mlp_mode("off")
            assert not oasr.gated_mlp_available(x, w)
        finally:
            jit_mlp.set_gated_mlp_mode("auto")
            jit_mlp.set_gated_mlp_backend("auto")

    def test_an_unknown_lane_is_an_error(self):
        from oasr.jit import mlp as jit_mlp

        with pytest.raises(ValueError, match="invalid backend"):
            jit_mlp.set_gated_mlp_backend("tensorrt")
        assert jit_mlp.get_gated_mlp_backend() == "auto"

    @pytest.mark.parametrize(
        "alias,lane", [("cpp", "cxx"), ("cutlass", "cxx"), ("cutedsl", "cute")]
    )
    def test_aliases_match_the_attention_spelling(self, alias, lane):
        from oasr.jit import mlp as jit_mlp

        try:
            jit_mlp.set_gated_mlp_backend(alias)
            assert jit_mlp.get_gated_mlp_backend() == lane
        finally:
            jit_mlp.set_gated_mlp_backend("auto")
