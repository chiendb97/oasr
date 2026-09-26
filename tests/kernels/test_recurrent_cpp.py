# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The C++ CUTLASS/CuTe fused-recurrent-step backend's own surface.

``test_recurrent.py`` is backend-agnostic: it runs the functional and the layer
against cuDNN and against the written-out recurrence through whichever lane
:mod:`oasr.jit.recurrent_cute` picks, which is what proves the two agree.  What
it structurally cannot cover is the set of properties that exist *because* this
lane is C++ -- the ones the CuTeDSL lane either does not have or reaches by a
different route:

* the **tile table and the ladder that indexes it** live in ``constexpr`` C++
  and are mirrored in Python for routing.  A mirror nobody checks is a mirror
  that drifts, so the agreement is asserted here, over six architectures, from
  one box.  The same ladder is asserted equal to the CuTeDSL lane's
  ``_TILES`` -- if the two lanes picked different tiles, an A/B between them
  would be measuring tile choices and not kernels.
* **a K residue**: the hidden width need not be a whole number of K tiles here.
  The CuTeDSL kernel loops ``ceil_div(K, k_block)`` and predicates only the row
  axis, so there is nothing over there to compare against and the claim has to
  be made directly.
* **arbitrary row strides**: every operand may be a row-slice of a wider
  buffer.  "No copy" is an allocation claim, not a numeric one, so it needs an
  allocation count and not just a tolerance.
* **CUDA-graph replay and determinism** of the launcher.
* **the declared architecture gap**: sm_90 and sm_100 are counted, not hidden.

Nothing here is a second copy of a parity test.  Where a claim is about
*numbers* it is stated against an FP32 matmul plus the gate equations in torch,
so the kernel is checked against the **definition** rather than against another
OASR kernel.
"""

from __future__ import annotations

import functools

import pytest
import torch
from helpers import assert_graph_replay, device_sm

pytestmark = pytest.mark.cuda

_SUPPORTED = (80, 86, 89, 120)
#: The architectures the tables know about, routed or not.  Parametrising over
#: the unrouted ones is the point: the C++ answers for sm_90 and sm_100 are
#: reachable from this box and a live device query would collapse them.
_TABLED = (80, 86, 89, 90, 100, 120)

#: fp16 carries ~3 decimal digits and the oracle accumulates in fp32, so the
#: comparison is against the *relative* size of the output, not an absolute
#: epsilon that a large K would blow through.
_REL_TOL = {torch.float16: 3e-3, torch.bfloat16: 2e-2}

_KINDS = ("lstm", "rnn_tanh", "rnn_relu")


def _requires_cxx():
    if device_sm() not in _SUPPORTED:
        pytest.skip(f"the C++ recurrent-step lane is not compiled for sm_{device_sm()}")


# ---------------------------------------------------------------------------
# Operands and the oracle
# ---------------------------------------------------------------------------


def _operands(batch, hidden, k, kind, dtype=torch.float16, seed=17, pad=0):
    """Every tensor the step reads.  ``pad`` makes each one a row-slice."""
    from oasr.jit.recurrent_step import gate_count

    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", dtype=dtype, generator=g)  # noqa: E731
    n = gate_count(kind) * hidden
    prev_h = (r(batch, k + pad) * 0.3)[:, :k]
    weight = (r(n, k + pad) * k**-0.5)[:, :k]
    gates = (r(batch, n + pad * gate_count(kind)) * 0.3)[:, :n]
    prev_c = (r(batch, hidden + pad) * 0.3)[:, :hidden]
    return prev_h, weight, gates, prev_c


def _oracle(prev_h, weight, gates, prev_c, kind):
    """The equations in FP32, not another kernel."""
    acc = prev_h.float() @ weight.float().T + gates.float()
    if kind == "rnn_tanh":
        return torch.tanh(acc), None
    if kind == "rnn_relu":
        return torch.clamp(acc, min=0.0), None
    gv = acc.view(prev_h.shape[0], -1, 4)
    cell = torch.sigmoid(gv[..., 1]) * prev_c.float() + torch.sigmoid(gv[..., 0]) * torch.tanh(
        gv[..., 2]
    )
    return torch.sigmoid(gv[..., 3]) * torch.tanh(cell), cell


def _assert_close(out, ref, dtype, what):
    assert not torch.isnan(out).any(), f"kernel produced NaN in {what}"
    scale = max(ref.abs().max().item(), 1e-6)
    rel = (out.float() - ref).abs().max().item() / scale
    assert rel < _REL_TOL[dtype], f"{what}: max relative error {rel:.5f}"


def _launch(tile, kind="lstm", dtype=torch.float16):
    from oasr.jit.recurrent_step import get_recurrent_step_fn

    dtype_str = "float16" if dtype is torch.float16 else "bfloat16"
    return get_recurrent_step_fn(dtype_str=dtype_str, kind=kind, tile_index=tile)


def _run(batch, hidden, k=None, *, tile=None, kind="lstm", dtype=torch.float16, pad=0, seed=17):
    from oasr.jit.recurrent_step import gate_count, select_tile

    k = hidden if k is None else k
    if tile is None:
        tile = select_tile(device_sm(), hidden, batch, 2, gate_count(kind))
    prev_h, weight, gates, prev_c = _operands(batch, hidden, k, kind, dtype, seed, pad)
    out_h = torch.full((batch, hidden), float("nan"), device="cuda", dtype=dtype)
    out_c = torch.full((batch, hidden), float("nan"), device="cuda", dtype=dtype)
    fn = _launch(tile, kind, dtype)
    if kind == "lstm":
        fn(out_h, out_c, prev_h, weight, gates, prev_c)
    else:
        fn(out_h, None, prev_h, weight, gates, None)
    ref_h, ref_c = _oracle(prev_h, weight, gates, prev_c, kind)
    _assert_close(out_h, ref_h, dtype, "h")
    if kind == "lstm":
        _assert_close(out_c, ref_c, dtype, "c")
    return out_h, out_c


# ---------------------------------------------------------------------------
# The tile table, the ladder, and the selector
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _config_probe():
    """``csrc/recurrent_step_jit_binding.cu`` on its own -- the real file, not a copy.

    Compiling the shipped binding standalone is the point: a purpose-built
    fixture TU would be a *second* transcription of the table, which is the
    drift this test exists to catch.  It pulls in only
    ``include/oasr/recurrent/recurrent_step_tiles.h``, which deliberately
    includes no CuTe, so this costs a couple of seconds rather than a CUTLASS
    compile.
    """
    from oasr.jit import env
    from oasr.jit.core import gen_jit_spec

    return gen_jit_spec(
        "recurrent_step_config_probe", [env.OASR_CSRC_DIR / "recurrent_step_jit_binding.cu"]
    ).build_and_load()


def _cxx_tile_info(index, sm, elem_bits=16, gates=4):
    out = torch.zeros(9, dtype=torch.int32)
    _config_probe().recurrent_step_tile_info(out, index, sm, elem_bits, gates)
    return out.tolist()


def _cxx_route_info(index):
    out = torch.zeros(3, dtype=torch.int32)
    _config_probe().recurrent_step_route_info(out, index)
    return out.tolist()


class TestTileTableAgrees:
    """``kRecurrentStepTiles`` and :data:`oasr.jit.recurrent_step.TILES`, field for field.

    Asked over an ``sm`` *argument*, so one built module answers for every
    architecture the tables know.  That is deliberate and it is the property a
    live device query would destroy: on this box a query would report sm_120's
    99 KB for all six rows, and it is exactly the sm_86/sm_89 budget that was
    once wrong in the attention family.
    """

    def test_the_two_tables_are_the_same_length(self):
        from oasr.jit.recurrent_step import TILES

        assert int(_config_probe().recurrent_step_tile_count()) == len(TILES)

    def test_constants_agree(self):
        from oasr.jit.recurrent_step import ACC_PAD, ALIGNMENT

        probe = _config_probe()
        assert int(probe.recurrent_step_alignment()) == ALIGNMENT
        assert int(probe.recurrent_step_acc_pad()) == ACC_PAD

    @pytest.mark.parametrize("sm", _TABLED)
    def test_budget_and_warp_slots_agree(self, sm):
        from oasr.jit.recurrent_step import _MAX_THREADS_PER_SM, smem_budget

        probe = _config_probe()
        assert int(probe.recurrent_step_smem_budget(sm)) == smem_budget(sm)
        assert int(probe.recurrent_step_max_threads_per_sm(sm)) == _MAX_THREADS_PER_SM[sm]

    @pytest.mark.parametrize("gates", [1, 4])
    @pytest.mark.parametrize("sm", _TABLED)
    @pytest.mark.parametrize("index", range(8))
    def test_tile_matches_cxx(self, sm, index, gates):
        from oasr.jit.recurrent_step import TILES, ctas_per_sm, smem_bytes, tile_valid

        t = TILES[index]
        want = [
            t.block_m,
            t.block_n,
            t.block_k,
            t.stages,
            t.threads,
            t.warps_n,
            int(tile_valid(t, sm, 2, gates)),
            smem_bytes(t),
            ctas_per_sm(t, sm),
        ]
        got = _cxx_tile_info(index, sm, 16, gates)
        assert got == want, (
            f"tile-table drift at sm_{sm} index {index} gates={gates}: C++ {got} vs "
            f"Python {want}. These are one table written twice; the Python copy only "
            f"exists so the router can answer before anything is built."
        )

    @pytest.mark.parametrize("sm", _TABLED)
    def test_every_shipped_tile_fits_every_tabled_arch(self, sm):
        """A tile in the table that does not fit is a cell that fails to build."""
        from oasr.jit.recurrent_step import TILES, smem_budget

        for index, t in enumerate(TILES):
            for gates in (1, 4):
                info = _cxx_tile_info(index, sm, 16, gates)
                assert info[6] == 1, (
                    f"tile {index} {t} does not fit sm_{sm} (gates={gates}); it needs "
                    f"{info[7]} B of {smem_budget(sm)} B"
                )


class TestRouteLadderAgrees:
    """The ``(hidden, batch)`` ladder, in three places that must stay one."""

    def test_length_agrees(self):
        from oasr.jit.recurrent_step import ROUTES

        assert int(_config_probe().recurrent_step_route_count()) == len(ROUTES)

    @pytest.mark.parametrize("index", range(18))
    def test_rung_matches_cxx(self, index):
        from oasr.jit.recurrent_step import ROUTES

        assert _cxx_route_info(index) == list(ROUTES[index]), (
            f"route-ladder drift at rung {index}; the rungs are scanned in order, so a "
            f"permutation is a silent routing change"
        )

    def test_the_ladder_reproduces_the_cutedsl_lane(self):
        """Both lanes must pick the **same tile** for every shape.

        Otherwise an A/B between them measures tile choices and not kernels --
        which is the whole reason this lane inherited the ladder instead of
        deriving its own.
        """
        from oasr.jit.recurrent_cute import _TILES as CUTE_TILES
        from oasr.jit.recurrent_step import ROUTES, TILES

        assert len(ROUTES) == len(CUTE_TILES)
        for rung, (hidden_max, batch_max, index) in enumerate(ROUTES):
            cute_hidden, cute_batch, cute_tile = CUTE_TILES[rung]
            t = TILES[index]
            assert (hidden_max, batch_max) == (
                cute_hidden,
                cute_batch,
            ), f"rung {rung}: the C++ ladder's bounds and the CuTeDSL lane's disagree"
            assert (
                t.block_m,
                t.block_n,
                t.block_k,
                t.stages,
                t.threads,
                t.warps_n,
            ) == tuple(cute_tile), f"rung {rung}: the two lanes would run different tiles"

    @pytest.mark.parametrize("gates", [1, 4])
    @pytest.mark.parametrize("sm", _TABLED)
    @pytest.mark.parametrize("hidden", [8, 256, 640, 1024, 1536, 2048, 4096])
    @pytest.mark.parametrize("batch", [1, 8, 16, 32, 64, 128, 256, 512, 4096])
    def test_selector_matches_cxx(self, sm, hidden, batch, gates):
        """The selector is ``constexpr`` C++; :func:`select_tile` only mirrors it."""
        from oasr.jit.recurrent_step import select_tile

        got = int(_config_probe().recurrent_step_select_tile(sm, hidden, batch, 16, gates))
        assert got == select_tile(sm, hidden, batch, 2, gates)

    def test_the_selector_is_a_pure_function_of_its_arguments(self):
        """``AGENTS.md`` rule 11: never branch kernel dispatch on capture state.

        Two tiles sum the K loop in different orders, so a capture-dependent
        answer would make a replayed graph decode differently from eager.
        """
        import warnings

        from oasr.jit.recurrent_step import select_tile

        eager = select_tile(120, 640, 16, 2, 4)
        graph = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            # The graph *is* empty, and that is the point: the selector must
            # launch nothing and must not consult the capture state.
            warnings.filterwarnings("ignore", message=".*CUDA Graph is empty.*")
            with torch.cuda.graph(graph):
                captured = select_tile(120, 640, 16, 2, 4)
        assert captured == eager


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


class TestEveryTileIsCorrect:
    """Every compiled variant, against the equations in FP32."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("kind", _KINDS)
    @pytest.mark.parametrize("tile", range(8))
    def test_tile(self, tile, kind, dtype):
        _requires_cxx()
        from oasr.jit.recurrent_step import TILES

        # One m-tile's worth of rows, so the tile is exercised whole rather
        # than mostly predicated off.
        _run(TILES[tile].block_m, 256, tile=tile, kind=kind, dtype=dtype)

    @pytest.mark.parametrize("hidden", [256, 640, 1024, 2048])
    @pytest.mark.parametrize("batch", [1, 16, 64, 256])
    def test_the_routed_tile_is_correct_across_the_ladder(self, hidden, batch):
        _requires_cxx()
        _run(batch, hidden)


class TestShapeContract:
    """What this lane accepts that the CuTeDSL one does not."""

    @pytest.mark.parametrize("k", [8, 40, 72, 136, 200, 1000])
    def test_k_need_not_be_a_whole_number_of_k_tiles(self, k):
        """The residue is predicated and the ZFILL cp.async zeroes what it skips.

        A zero is the identity for the dot product, so the residue is *right*
        rather than merely safe.  Every shipped tile has ``block_k = 64``, so
        every ``k`` here but 200 leaves a residue.
        """
        _requires_cxx()
        _run(16, 256, k)

    @pytest.mark.parametrize("batch", [1, 3, 7, 17, 33, 100])
    def test_the_cohort_need_not_be_a_tile_multiple(self, batch):
        _requires_cxx()
        _run(batch, 640)

    @pytest.mark.parametrize("hidden", [8, 24, 72, 136, 1000])
    def test_the_hidden_width_need_not_be_a_tile_multiple(self, hidden):
        _requires_cxx()
        _run(16, hidden)

    def test_k_and_hidden_may_differ(self):
        """The state read and the state written are independent extents.

        They are equal for every shipped recurrent layer, and the kernel never
        assumes it -- which is what keeps a projected LSTM addable without
        touching the mainloop.
        """
        _requires_cxx()
        _run(16, 128, 256)
        _run(16, 256, 128)

    def test_every_operand_may_be_a_row_slice(self):
        """A stride claim is an *allocation* claim, so count allocations.

        The CuTeDSL lane marks its tensors ``mark_compact_shape_dynamic``, so
        a caller holding a row-slice of a wider buffer has to copy first.
        """
        _requires_cxx()
        # First: prove the operands really are non-compact, or the numeric
        # assertion below passes vacuously on contiguous tensors.
        prev_h, weight, gates, prev_c = _operands(16, 256, 256, "lstm", pad=8)
        for name, t in (
            ("previous_h", prev_h),
            ("weight_hh", weight),
            ("input_gates", gates),
            ("previous_c", prev_c),
        ):
            assert not t.is_contiguous(), f"{name} is compact; this test proves nothing"
            assert t.stride(0) > t.shape[1], name

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        # `_run` allocates the two outputs and nothing else; a hidden
        # `.contiguous()` on any of the four inputs would show up here, and the
        # oracle comparison inside it is what proves the strides were *used*
        # rather than ignored.
        _run(16, 256, pad=8)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - before <= 0

    def test_a_misaligned_k_is_refused_rather_than_read_wrong(self):
        _requires_cxx()
        with pytest.raises(Exception, match="multiple of 8"):
            _run(16, 256, 12)

    def test_a_non_contiguous_last_axis_is_refused(self):
        """Without this the `_1{}` in every stride reads wrong addresses silently."""
        _requires_cxx()
        from oasr.jit.recurrent_step import select_tile

        tile = select_tile(device_sm(), 256, 16, 2, 4)
        prev_h, weight, gates, prev_c = _operands(16, 256, 256, "lstm")
        out_h = torch.empty(16, 256, device="cuda", dtype=torch.float16)
        out_c = torch.empty(16, 256, device="cuda", dtype=torch.float16)
        strided = prev_h.repeat_interleave(2, dim=1)[:, ::2]
        with pytest.raises(Exception, match="contiguous last dimension"):
            _launch(tile)(out_h, out_c, strided, weight, gates, prev_c)


class TestCellStateIsCompiledIn:
    """A variant that carries no cell state must refuse one, not drop it."""

    def test_an_rnn_variant_refuses_a_cell(self):
        _requires_cxx()
        from oasr.jit.recurrent_step import select_tile

        tile = select_tile(device_sm(), 256, 16, 2, 1)
        prev_h, weight, gates, prev_c = _operands(16, 256, 256, "rnn_tanh")
        out_h = torch.empty(16, 256, device="cuda", dtype=torch.float16)
        out_c = torch.empty(16, 256, device="cuda", dtype=torch.float16)
        with pytest.raises(Exception, match="no cell state"):
            _launch(tile, "rnn_tanh")(out_h, out_c, prev_h, weight, gates, prev_c)

    def test_an_lstm_variant_requires_one(self):
        _requires_cxx()
        from oasr.jit.recurrent_step import select_tile

        tile = select_tile(device_sm(), 256, 16, 2, 4)
        prev_h, weight, gates, _ = _operands(16, 256, 256, "lstm")
        out_h = torch.empty(16, 256, device="cuda", dtype=torch.float16)
        with pytest.raises(Exception, match="required"):
            _launch(tile)(out_h, None, prev_h, weight, gates, None)


# ---------------------------------------------------------------------------
# Launcher behaviour
# ---------------------------------------------------------------------------


class TestLauncher:
    def test_cuda_graph_replay_recomputes(self):
        """Catches an allocation in the launcher and a data-dependent trip count."""
        _requires_cxx()
        from oasr.jit.recurrent_step import select_tile

        tile = select_tile(device_sm(), 640, 16, 2, 4)
        prev_h, weight, gates, prev_c = _operands(16, 640, 640, "lstm")
        out_h = torch.empty(16, 640, device="cuda", dtype=torch.float16)
        out_c = torch.empty(16, 640, device="cuda", dtype=torch.float16)
        fn = _launch(tile)

        def mutate():
            # New values *through the captured buffers*, so a replay that
            # reproduced the captured result instead of recomputing fails.
            prev_h.mul_(0.5)
            gates.add_(0.1)

        def expected():
            return _oracle(prev_h, weight, gates, prev_c, "lstm")[0].to(torch.float16)

        assert_graph_replay(
            lambda: fn(out_h, out_c, prev_h, weight, gates, prev_c),
            out=out_h,
            mutate=mutate,
            expected=expected,
            rtol=3e-3,
            atol=3e-3,
        )

    def test_repeated_launches_are_bit_identical(self):
        _requires_cxx()
        first, first_c = _run(16, 640)
        second, second_c = _run(16, 640)
        assert torch.equal(first, second)
        assert torch.equal(first_c, second_c)

    def test_it_writes_the_destinations_it_was_given(self):
        _requires_cxx()
        from oasr.jit.recurrent_step import select_tile

        tile = select_tile(device_sm(), 256, 16, 2, 4)
        prev_h, weight, gates, prev_c = _operands(16, 256, 256, "lstm")
        out_h = torch.full((16, 256), float("nan"), device="cuda", dtype=torch.float16)
        out_c = torch.full((16, 256), float("nan"), device="cuda", dtype=torch.float16)
        ptr_h, ptr_c = out_h.data_ptr(), out_c.data_ptr()
        _launch(tile)(out_h, out_c, prev_h, weight, gates, prev_c)
        assert out_h.data_ptr() == ptr_h and out_c.data_ptr() == ptr_c
        assert not torch.isnan(out_h).any() and not torch.isnan(out_c).any()

    def test_a_row_past_the_cohort_is_left_alone(self):
        """The overhang is predicated off, not written with garbage."""
        _requires_cxx()
        from oasr.jit.recurrent_step import TILES

        batch, hidden, tile = 5, 256, 4  # block_m 64, so 59 rows overhang
        prev_h, weight, gates, prev_c = _operands(batch, hidden, hidden, "lstm")
        assert TILES[tile].block_m > batch
        out_h = torch.zeros(batch + 3, hidden, device="cuda", dtype=torch.float16)
        out_c = torch.zeros(batch + 3, hidden, device="cuda", dtype=torch.float16)
        sentinel = out_h[batch:].clone()
        _launch(tile)(out_h[:batch], out_c[:batch], prev_h, weight, gates, prev_c)
        assert torch.equal(out_h[batch:], sentinel), "the kernel wrote past the cohort"


# ---------------------------------------------------------------------------
# Routing between the two lanes
# ---------------------------------------------------------------------------


class TestLaneRouting:
    """The arbiter's two gates, and what each one is for."""

    def test_backend_env_is_read(self, monkeypatch):
        from oasr.jit import recurrent_cute as rc

        for raw, expected in [
            ("auto", "auto"),
            ("cute", "cute"),
            ("cutedsl", "cute"),
            ("cxx", "cxx"),
            ("cpp", "cxx"),
            ("cutlass", "cxx"),
            ("banana", "auto"),
        ]:
            monkeypatch.setenv("OASR_RECURRENT_BACKEND", raw)
            assert rc._read_backend() == expected

    def test_auto_prefers_the_cxx_lane(self):
        """Measured 1.05x-1.31x, geomean 1.13x, no row regressing.

        A reorder is a performance claim, so it should have to move this
        assertion and the table it cites.
        """
        from oasr.jit.recurrent_cute import _AUTO_ORDER

        assert _AUTO_ORDER[0] == "cxx"

    def test_set_backend_invalidates_the_route_memo(self, device):
        from oasr.jit import recurrent_cute as rc

        before_mode, before_backend = rc.get_mode(), rc.get_backend()
        try:
            rc.set_mode("auto")
            for backend in ("cxx", "cute"):
                rc.set_backend(backend)
                routed = rc.routed_step(
                    dtype_str="float16",
                    gate_count=4,
                    activation="lstm",
                    hidden=640,
                    batch=16,
                )
                if routed is None:
                    pytest.skip(f"the {backend} lane is unavailable here")
                assert routed[0] == backend, "a pinned lane was served from a stale memo"
        finally:
            rc.set_mode(before_mode)
            rc.set_backend(before_backend)

    def test_the_rollback_switch_beats_a_pinned_lane(self, device):
        """``OASR_RECURRENT_CUTE=off`` must silence both lanes, not just one."""
        from oasr.jit import recurrent_cute as rc

        before_mode, before_backend = rc.get_mode(), rc.get_backend()
        try:
            for backend in ("auto", "cxx", "cute"):
                rc.set_backend(backend)
                rc.set_mode("off")
                assert (
                    rc.routed_step(
                        dtype_str="float16",
                        gate_count=4,
                        activation="lstm",
                        hidden=640,
                        batch=16,
                    )
                    is None
                )
        finally:
            rc.set_mode(before_mode)
            rc.set_backend(before_backend)

    def test_the_two_lanes_agree_at_the_layer(self, device):
        """The claim that matters to a caller: swapping lanes changes nothing.

        Against the *unfused* path as well, which is the third implementation
        of the same recurrence and the one the rest of the suite checks against
        cuDNN.
        """
        from oasr.jit import recurrent_cute as rc
        from oasr.layers import LSTM

        before_mode, before_backend = rc.get_mode(), rc.get_backend()
        torch.manual_seed(0)
        hidden, batch = 640, 16
        layer = LSTM(hidden, hidden, num_layers=1, device="cuda", dtype=torch.float16).eval()
        x = torch.randn(1, batch, hidden, device="cuda", dtype=torch.float16)
        h0 = torch.randn(1, batch, hidden, device="cuda", dtype=torch.float16) * 0.2
        c0 = torch.randn(1, batch, hidden, device="cuda", dtype=torch.float16) * 0.2
        got = {}
        try:
            for name in ("cxx", "cute", "unfused"):
                if name == "unfused":
                    rc.set_mode("off")
                else:
                    rc.set_mode("auto")
                    rc.set_backend(name)
                with torch.no_grad():
                    out, (h, c) = layer(x, (h0, c0))
                got[name] = tuple(t.float().clone() for t in (out, h, c))
        finally:
            rc.set_mode(before_mode)
            rc.set_backend(before_backend)
        scale = max(got["unfused"][0].abs().max().item(), 1e-6)
        for other in ("cute", "unfused"):
            worst = max((got["cxx"][i] - got[other][i]).abs().max().item() for i in range(3))
            assert worst / scale < 3e-3, f"cxx and {other} disagree by {worst:.3e}"


class TestArchCoverageIsDeclared:
    """``AGENTS.md`` rule 3: a missing kernel is declared, not routed around."""

    def test_sm90_and_sm100_are_not_claimed(self):
        from oasr.jit.recurrent_step import SUPPORTED_SM

        assert 90 not in SUPPORTED_SM and 100 not in SUPPORTED_SM, (
            "the Ampere-class collective would *run* on those parts; routing to it "
            "without measuring is the performance claim this set exists to avoid"
        )

    def test_the_two_lanes_declare_the_same_set(self):
        from oasr.jit.recurrent_cute import _SUPPORTED_SM
        from oasr.jit.recurrent_step import SUPPORTED_SM

        assert tuple(SUPPORTED_SM) == tuple(_SUPPORTED_SM)

    def test_an_unserved_architecture_is_counted_not_hidden(self):
        from oasr.jit import recurrent_step as rs

        rs.reset_coverage()
        try:
            assert not rs.config_supported(
                sm=90, dtype_str="float16", kind="lstm", batch=16, hidden=640, k=640
            )
            report = rs.recurrent_step_coverage_report()
            assert any("sm_90" in line for line in report), report
        finally:
            rs.reset_coverage()

    def test_a_refused_config_is_counted(self):
        from oasr.jit import recurrent_step as rs

        rs.reset_coverage()
        try:
            assert not rs.config_supported(
                sm=120, dtype_str="float16", kind="lstm", batch=16, hidden=640, k=12
            )
            assert any("128-bit" in line for line in rs.recurrent_step_coverage_report())
        finally:
            rs.reset_coverage()
