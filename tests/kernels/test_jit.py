#!/usr/bin/env python3
"""
Unit tests for JIT compilation infrastructure.
Verifies that JitSpec can compile and load modules correctly.
"""

import pytest


class TestJitInfrastructure:
    """Tests for JIT compilation system."""

    def test_jit_env_paths_exist(self):
        """Verify JIT environment paths are valid."""
        from oasr.jit.env import OASR_CSRC_DIR, OASR_INCLUDE_DIR

        assert OASR_CSRC_DIR.exists(), f"OASR_CSRC_DIR does not exist: {OASR_CSRC_DIR}"
        assert OASR_INCLUDE_DIR.exists(), f"OASR_INCLUDE_DIR does not exist: {OASR_INCLUDE_DIR}"

    def test_gen_activation_module(self):
        """Verify activation JIT spec can be created."""
        from oasr.jit.activation import gen_activation_module

        spec = gen_activation_module()
        assert spec.name == "activation"
        assert len(spec.sources) == 2  # activation.cu + binding

    def test_gen_norm_module(self):
        """Verify norm JIT spec can be created."""
        from oasr.jit.norm import gen_norm_module

        spec = gen_norm_module()
        assert spec.name == "norm"
        assert len(spec.sources) == 2

    def test_gen_pooling_module(self):
        """Verify pooling JIT spec can be created."""
        from oasr.jit.pooling import gen_pooling_module

        spec = gen_pooling_module()
        assert spec.name == "pooling"
        assert len(spec.sources) == 2

    def test_gen_conv_module(self):
        """Verify conv JIT spec can be created."""
        from oasr.jit.conv import gen_conv_module

        spec = gen_conv_module()
        assert spec.name == "conv"
        assert len(spec.sources) == 2

    def test_gen_conv2d_module(self):
        """Verify conv2d JIT spec can be created.

        Each CUTLASS tile config is rendered as a self-contained ``.cu`` (with
        its own ``TVM_FFI_DLL_EXPORT_TYPED_FUNC``), so the spec holds one source
        per unique compile config — count varies by target SM.
        """
        from oasr.jit.conv import gen_conv2d_module

        spec = gen_conv2d_module()
        assert spec.name == "conv2d"
        assert len(spec.sources) >= 1
        assert all(p.suffix == ".cu" for p in spec.sources)

    def test_gen_gemm_module(self):
        """Verify gemm JIT spec can be created.

        Like conv2d, each tile variant is a self-contained ``.cu``; the source
        count equals the number of unique compile configs for the target SM.
        """
        from oasr.jit.gemm import gen_gemm_module

        spec = gen_gemm_module()
        assert spec.name == "gemm"
        assert len(spec.sources) >= 1
        assert all(p.suffix == ".cu" for p in spec.sources)

    def test_gen_bmm_module(self):
        """Verify bmm JIT spec can be created."""
        from oasr.jit.gemm import gen_bmm_module

        spec = gen_bmm_module()
        assert spec.name == "bmm"
        assert len(spec.sources) >= 1
        assert all(p.suffix == ".cu" for p in spec.sources)

    def test_gen_group_gemm_module(self):
        """Verify group_gemm JIT spec can be created."""
        from oasr.jit.gemm import gen_group_gemm_module

        spec = gen_group_gemm_module()
        assert spec.name == "group_gemm"
        assert len(spec.sources) >= 1
        assert all(p.suffix == ".cu" for p in spec.sources)

    def test_gen_all_modules(self):
        """Verify AOT gen_all_modules returns all expected specs."""
        from oasr.aot import gen_all_modules

        specs = gen_all_modules()
        names = [s.name for s in specs]
        expected = {
            "activation",
            "norm",
            "pooling",
            "recurrent",
            "conv",
            "conv2d",
            "cudnn_conv2d",
            "grouped_conv2d",
            "gemm",
            "bmm",
            "group_gemm",
            "gemm_log_softmax",
            "ctc_decoder",
            "softmax",
            "topk",
            "fft",
            "features",
        }
        assert expected.issubset(set(names)), f"missing modules: {expected - set(names)}"
        assert len(specs) == len(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# The CuTeDSL stream helper, shared by every compiled CuTeDSL callable
# ---------------------------------------------------------------------------


class TestTheTargetArchIsNeverGuessedAtCompileTime:
    """``_get_cuda_arch`` used to answer ``(8, 0)`` when nothing could be detected.

    That is a real architecture, so a box where CUDA was invisible — torch
    without a runtime, ``nvidia-smi`` absent — compiled
    ``-gencode arch=compute_80,code=sm_80`` and produced a library that would not
    run on whatever card eventually appeared. Nothing reported it, and the guess
    was indistinguishable from an A100.

    The guess still exists, because it has to: the JIT config generators resolve
    a target SM at *import* time (``jit.gemm``'s ``GEMM_DEFAULT``, ``jit.conv``'s
    ``CONV2D_DEFAULT``) and the CPU test job imports the package. What changed is
    that producing a ``.so`` refuses it.
    """

    def test_detection_returns_none_rather_than_a_plausible_arch(self, monkeypatch):
        """The heart of it.  ``(8, 0)`` as a failure value is indistinguishable
        from an A100, so every caller downstream believed it."""
        import torch

        from oasr.jit import core

        def _no_smi(*args, **kwargs):
            raise OSError("nvidia-smi not found")

        monkeypatch.setattr(core, "_arch_from_env", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(core.subprocess, "check_output", _no_smi)
        assert core._detect_cuda_arch() is None

    def test_import_time_resolution_still_answers_without_a_device(self, monkeypatch):
        """The property that keeps ``import oasr`` working on a CPU box."""
        from oasr.jit import core

        monkeypatch.setattr(core, "_detect_cuda_arch", lambda: None)
        assert core._get_cuda_arch() == core._ASSUMED_ARCH

    def test_compiling_refuses_an_undetected_arch(self, monkeypatch):
        from oasr.jit import core

        monkeypatch.setattr(core, "_detect_cuda_arch", lambda: None)
        with pytest.raises(RuntimeError, match="no CUDA device was detected"):
            core.require_known_cuda_arch("module 'probe'")

    def test_the_refusal_names_the_way_out(self, monkeypatch):
        """A build box with no GPU is a legitimate case, so the error has to say
        how to proceed rather than only that it will not."""
        from oasr.jit import core

        monkeypatch.setattr(core, "_detect_cuda_arch", lambda: None)
        with pytest.raises(RuntimeError) as excinfo:
            core.require_known_cuda_arch("module 'probe'")
        assert "OASR_CUDA_ARCH_LIST" in str(excinfo.value)

    def test_a_real_build_reaches_the_guard(self, monkeypatch, tmp_path):
        """Not just that the function refuses — that ``_compile`` calls it."""
        from oasr.jit import core

        monkeypatch.setattr(core, "_detect_cuda_arch", lambda: None)
        spec = core.JitSpec(name="probe", sources=[], extra_cuda_cflags=[])
        with pytest.raises(RuntimeError, match="no CUDA device was detected"):
            spec._compile(str(tmp_path / "probe.so"))

    def test_the_detected_arch_is_accepted_here(self):
        """This box has a GPU, so nothing above may fire in the ordinary case."""
        from oasr.jit import core

        assert core._detect_cuda_arch() is not None
        core.require_known_cuda_arch("module 'probe'")


class TestTheArchOverride:
    """``OASR_CUDA_ARCH_LIST`` is documented as the manual override for JIT arch
    detection and, for the JIT gencode, did nothing at all: ``_default_cuda_cflags``
    always emits its own ``-gencode``, so ``cpp_ext`` drops the context's arch flags.

    It is now the escape hatch the compile-time refusal needs — under one rule.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("9.0", (9, 0)),
            ("9.0a", (9, 0)),
            ("10.0f", (10, 0)),
            ("8.6", (8, 6)),
            ("  8.0  ", (8, 0)),
            ("9", (9, 0)),
        ],
    )
    def test_a_single_entry_is_an_unambiguous_target(self, monkeypatch, value, expected):
        from oasr.jit import core

        monkeypatch.setenv("OASR_CUDA_ARCH_LIST", value)
        assert core._arch_from_env() == expected

    @pytest.mark.parametrize("value", ["8.0 9.0a", "8.0,9.0", "8.0 9.0 10.0a"])
    def test_several_entries_are_left_to_the_aot_path(self, monkeypatch, value):
        """Two or more cannot mean "the JIT target" — the JIT builds one module
        for one architecture. Picking the first would silently compile for the
        wrong card on a machine that has the right one."""
        from oasr.jit import core

        monkeypatch.setenv("OASR_CUDA_ARCH_LIST", value)
        assert core._arch_from_env() is None

    @pytest.mark.parametrize("value", ["", "   ", "sm_90", "hopper"])
    def test_junk_is_ignored_rather_than_crashing_the_import(self, monkeypatch, value):
        from oasr.jit import core

        monkeypatch.setenv("OASR_CUDA_ARCH_LIST", value)
        assert core._arch_from_env() is None

    def test_the_override_unblocks_a_box_with_no_device(self, monkeypatch):
        """The case the refusal exists for: a build machine with no GPU."""
        from oasr.jit import core

        monkeypatch.setattr(core, "_arch_from_env", lambda: (9, 0))
        monkeypatch.setenv("OASR_CUDA_ARCH_LIST", "9.0")
        core.require_known_cuda_arch("module 'probe'")


class TestCuteRuntimeStream:
    """The shared stream helper (``oasr.jit.cute_runtime``).

    Every compiled CuTeDSL callable needs a ``CUstream``, and the obvious
    spelling — ``CUstream(torch.cuda.current_stream().cuda_stream)`` — cost 4.1 us
    per call, which was two thirds of the recurrent step's launch and 15% of an
    FMHA call.  Correctness is the point here: the handle must identify the
    *current* stream, including a side stream, or a kernel lands on the wrong one.
    """

    def test_returns_the_current_stream(self, device):
        import torch as _t

        from oasr.jit.cute_runtime import current_stream

        default = current_stream()
        raw = _t._C._cuda_getCurrentRawStream(_t.cuda.current_device())
        assert int(default) == int(raw)

    def test_tracks_a_stream_switch(self, device):
        import torch as _t

        from oasr.jit.cute_runtime import current_stream

        outer = int(current_stream())
        side = _t.cuda.Stream()
        with _t.cuda.stream(side):
            inner = int(current_stream())
        assert inner == side.cuda_stream
        assert inner != outer, "a side stream must not be served the default handle"
        assert int(current_stream()) == outer

    def test_handles_are_cached_per_stream(self, device):
        from oasr.jit.cute_runtime import current_stream

        assert current_stream() is current_stream()

    def test_the_fmha_path_uses_it(self):
        """The FMHA hot path must go through the same helper, not rebuild it."""
        import inspect

        from oasr.functionals import attention

        src = inspect.getsource(attention)
        # The assignment, not the prose: the module documents the old spelling.
        assert "stream = _CUstream(" not in src, "the 4.1 us spelling is back on the FMHA path"
        assert src.count("stream = _current_stream()") >= 2


# ---------------------------------------------------------------------------
# The CUTLASS 2.x tile space, and the constraint that keeps it buildable
# ---------------------------------------------------------------------------


class TestTileSpaceIsBuildable:
    """A tile that CUTLASS cannot express must never enter a config space.

    This is the layer where ``block_n=16`` got in.  It passed every gate that
    existed: the tile divides its warp shape, its operands fit in shared memory,
    the variant compiled, the launcher ran, and the tuner timed it and liked it
    well enough to write a production rule.  What none of those could see is that
    CUTLASS's tensor-op epilogue folds 32 lanes into a ``kAccessRows x
    kAccessWidth`` grid computed from the *tile*, asserts nothing about the
    product, and at ``block_n=16`` gets 16 — so half the lanes write rows outside
    their own slot and the kernel returns wrong numbers at full speed.

    These run without a GPU on purpose: the config space is pure Python, so the
    CPU job can hold the line for every arch, not just the one in the box.
    """

    def _all_declared_tiles(self):
        from oasr.jit.gemm import (
            _SPLITK_PARALLEL_TILES,
            _STREAMK_TILES,
            GemmExtraTileConfigs,
            TileShapeConfigs,
        )

        return TileShapeConfigs + GemmExtraTileConfigs + _STREAMK_TILES + _SPLITK_PARALLEL_TILES

    def test_the_lane_grid_covers_a_warp_for_every_built_tile(self):
        """Every tile that survives the filter maps all 32 lanes of its warps."""
        from oasr.jit.gemm import _epilogue_covers_warp, _epilogue_output_map

        for tile in self._all_declared_tiles():
            if not _epilogue_covers_warp(tile):
                continue  # refused by _tile_is_buildable; covered below
            width, rows, iters_col, iters_row = _epilogue_output_map(tile)
            assert width * rows == 32, f"{tile}: lane grid {rows}x{width}"
            assert iters_col >= 1 and iters_row >= 1, f"{tile}: {iters_row}x{iters_col} iterations"

    def test_block_n_16_is_refused(self):
        """The tile shape that shipped wrong results, stated as a unit.

        Written against a constructed tile rather than against the list, so it
        keeps its meaning if the list ever drops the entry.
        """
        from oasr.jit.gemm import TileShape, _epilogue_covers_warp, _epilogue_output_map

        tile = TileShape(block_m=128, block_n=16, block_k=64, warp_m=32, warp_n=16, warp_k=64)
        width, rows, _, _ = _epilogue_output_map(tile)
        assert (width, rows) == (2, 8), "the lane grid derivation drifted from CUTLASS"
        assert not _epilogue_covers_warp(tile), "block_n=16 is not addressable at 8 elems/access"

    def test_a_zero_iteration_tile_is_refused(self):
        """The other way the epilogue under-covers: no access at all.

        ``RowArrangement``'s 1-D branch (taken once ``block_n / warp_n`` reaches
        8 warps) computes ``kIterationsColumn = block_n / 8 / 32``, which is 0
        below ``block_n = 256``.  It has no ``static_assert`` either, so the
        constraint has to name this case as well as the narrow-grid one.
        """
        from oasr.jit.gemm import TileShape, _epilogue_covers_warp, _epilogue_output_map

        tile = TileShape(block_m=16, block_n=128, block_k=64, warp_m=16, warp_n=16, warp_k=64)
        assert _epilogue_output_map(tile)[2] == 0
        assert not _epilogue_covers_warp(tile)

    def test_no_built_config_has_an_unbuildable_tile(self):
        """Across every family that renders from the shared tile list.

        GEMM, BMM and grouped GEMM render from ``get_unique_compile_configs``
        and Conv2D from ``get_unique_conv2d_compile_configs``, so one bad tile
        used to produce four wrong kernels.
        """
        from oasr.jit.conv import get_unique_conv2d_compile_configs
        from oasr.jit.gemm import TileShape, _epilogue_covers_warp, get_unique_compile_configs

        for sm in (75, 80, 86, 89, 120):
            spaces = {
                "gemm": get_unique_compile_configs(sm),
                "conv2d": get_unique_conv2d_compile_configs(sm),
            }
            for family, configs in spaces.items():
                for name, cfg in configs.items():
                    if not hasattr(cfg, "block_n"):
                        continue  # SM90+ config: a different epilogue entirely
                    tile = TileShape(
                        block_m=cfg.block_m,
                        block_n=cfg.block_n,
                        block_k=cfg.block_k,
                        warp_m=cfg.warp_m,
                        warp_n=cfg.warp_n,
                        warp_k=cfg.warp_k,
                    )
                    assert _epilogue_covers_warp(tile), f"sm{sm} {family}: {name}"

    def test_the_refusal_is_recorded_with_a_reason(self):
        """A dropped tile is not silent — it is the reason a tile that looks
        tuneable never shows up in the autotuner's candidate list."""
        from oasr.jit.gemm import get_all_autotune_configs, rejected_tiles

        get_all_autotune_configs(120)
        rejected = rejected_tiles()
        assert "b128x16x64_w32x16x64" in rejected
        reason = rejected["b128x16x64_w32x16x64"]
        assert "block_n=16" in reason and "wrong results" in reason

    def test_every_heuristic_rule_names_a_built_config(self):
        """The rule table is a hand-carried constant; the config space is derived.

        Dropping a tile from the space silently orphans any rule that named it,
        and an orphaned rule is not an error at runtime — ``_plan`` catches the
        ``AttributeError`` and falls back to ``GEMM_DEFAULT``, which at the one
        affected shape is 3.5x slower than the right tile.  So the join between
        the two is checked here rather than discovered in a benchmark.
        """
        from oasr.jit.gemm import (
            _GEMM_HEURISTIC_RULES_SM120,
            GEMM_DEFAULT,
            get_unique_compile_configs,
        )

        built = get_unique_compile_configs(120)
        orphans = [
            (op, N, K, m_max, choice.name)
            for (op, N, K), rules in _GEMM_HEURISTIC_RULES_SM120.items()
            for m_max, choice in rules
            if not isinstance(choice, str)
            and choice is not GEMM_DEFAULT
            and (choice.kSmVersion != 120 or choice.compile_name not in built)
        ]
        assert not orphans, f"rules naming a config SM120 never compiles: {orphans}"


# ---------------------------------------------------------------------------
# The CUTLASS 2.x architecture tag, and the trait coverage that bounds it
# ---------------------------------------------------------------------------


#: The architectures served by the CUTLASS 2.x lane.  SM90 and SM100 take the 3.x
#: collective builders, whose mainloop pipelines K itself.
_SM_2X = (75, 80, 86, 89, 120)


class TestKDecompositionsAreArchUniform:
    """Stream-K and parallel split-K exist on every CUTLASS 2.x architecture.

    Both were built inside ``_get_sm120_configs`` and nowhere else, so on an
    A100, a T4, an L40S or an RTX 30-series card they were not in the config
    space at all.  Three things followed, none of them visible from a passing
    test run: ``oasr.autotune()`` had no Stream-K arm to find on a shape that
    starves the data-parallel grid; ``OASR_GEMM_STREAMK`` and
    ``OASR_GEMM_SPLITK_PARALLEL`` were inert despite ``AGENTS.md`` documenting
    them as global build knobs; and ``gemm_activation`` had no *valid* split-K
    anywhere but SM120, because serial split-K applies its epilogue per
    K-partition and the registration refuses it for a fused activation.

    Pure Python, so the CPU job holds this for every architecture rather than
    for whichever card the runner happens to have.
    """

    def _space(self, sm):
        from oasr.jit.gemm import get_unique_compile_configs

        return get_unique_compile_configs(sm)

    @pytest.mark.parametrize("sm", _SM_2X)
    def test_every_2x_arch_gets_both_decompositions(self, sm):
        configs = self._space(sm)
        sk = [c for c in configs.values() if getattr(c, "stream_k", False)]
        pk = [c for c in configs.values() if getattr(c, "parallel_split_k", False)]
        assert sk, f"sm_{sm} has no Stream-K variant in its compile set"
        assert pk, f"sm_{sm} has no parallel split-K variant in its compile set"

    @pytest.mark.parametrize("sm", [90, 100])
    def test_the_3x_arches_get_neither(self, sm):
        """Not an oversight there: the 3.x collective mainloop pipelines K itself,
        and the SM90+ template has no Stream-K path to render one into."""
        configs = self._space(sm)
        assert not [c for c in configs.values() if getattr(c, "stream_k", False)]
        assert not [c for c in configs.values() if getattr(c, "parallel_split_k", False)]

    def test_turing_builds_them_at_two_pipeline_stages_only(self):
        """Measured, not assumed: sm_75 at three or four stages fails with
        ``incomplete type "cutlass::gemm::kernel::DefaultGemmUniversal<...>"``,
        the same 2-stage-only tensor-op specialisation that makes
        ``RecurrentArch<75>`` set ``kStages = 2``.  80/86/89/120 build at 2, 3
        and 4.  A uniform stage list would put unbuildable TUs into Turing's
        space, and one of those fails the whole module."""
        for cfg in self._space(75).values():
            if getattr(cfg, "stream_k", False) or getattr(cfg, "parallel_split_k", False):
                assert cfg.kStages == 2, (
                    f"sm_75 {cfg.compile_name} is a {cfg.kStages}-stage decomposition; "
                    f"Turing's kernel::DefaultGemm has no such specialisation"
                )

    def test_the_stage_tables_cover_exactly_the_2x_families(self):
        """A family missing from either table raises ``KeyError`` when its config
        space is generated — deliberately, because silently receiving no
        decompositions is how this became SM120-only.  An extra key is dead data
        that nothing will ever read."""
        from oasr.jit.gemm import _SM_SPLITK_PARALLEL_STAGES, _SM_STREAMK_STAGES

        assert set(_SM_STREAMK_STAGES) == set(_SM_2X)
        assert set(_SM_SPLITK_PARALLEL_STAGES) == set(_SM_2X)

    @pytest.mark.parametrize("sm", _SM_2X)
    def test_gemm_activation_has_a_valid_split_k(self, sm):
        """The consequence with teeth.

        ``oasr/tune/backends/gemm.py`` skips a ``split_k > 1`` config for
        ``gemm_activation`` unless it is the *parallel* decomposition, since
        serial split-K would apply the activation to each K-partition's partial
        sum.  With parallel split-K absent, the count of fused-activation
        split-K candidates was exactly zero on four of the five 2.x arches.
        """
        from oasr.jit.gemm import get_all_autotune_configs

        valid = [
            c
            for c in get_all_autotune_configs(sm).values()
            if getattr(c, "split_k", 1) > 1 and getattr(c, "parallel_split_k", False)
        ]
        assert valid, (
            f"sm_{sm} has no split-K candidate a fused-activation GEMM can use; "
            f"serial split-K is refused for gemm_activation by construction"
        )

    @pytest.mark.parametrize("sm", _SM_2X)
    def test_the_build_knobs_are_global(self, sm, monkeypatch):
        """``OASR_GEMM_STREAMK=0`` / ``OASR_GEMM_SPLITK_PARALLEL=0`` must take
        effect on every architecture, which is what ``AGENTS.md`` promises.  They
        were read inside the SM120 branch, so on any other card they changed
        nothing at all."""
        from oasr.jit import gemm as jit_gemm

        monkeypatch.setattr(jit_gemm, "_STREAMK_ENABLED", False)
        monkeypatch.setattr(jit_gemm, "_SPLITK_PARALLEL_ENABLED", False)
        configs = jit_gemm.get_unique_compile_configs(sm)
        assert not [c for c in configs.values() if getattr(c, "stream_k", False)]
        assert not [c for c in configs.values() if getattr(c, "parallel_split_k", False)]

    def test_a_larger_smem_budget_never_yields_fewer_tiles(self):
        """sm_80's 164 KB must admit at least what sm_86's 100 KB does.

        A relationship rather than a magic count: it catches a transposed or
        mistyped entry in the stage tables, which a fixed number would only
        catch by accident.
        """
        from oasr.jit.gemm import _SM_MAX_SMEM_BYTES

        assert _SM_MAX_SMEM_BYTES[80] > _SM_MAX_SMEM_BYTES[86]
        sk80 = [c for c in self._space(80).values() if getattr(c, "stream_k", False)]
        sk86 = [c for c in self._space(86).values() if getattr(c, "stream_k", False)]
        assert len(sk80) >= len(sk86)


class TestCutlass2xArchTagIsInstantiable:
    """A JIT target must name a CUTLASS tag that CUTLASS can build for half.

    ``device::Gemm`` / ``device::GemmUniversal`` default their ``Operator_``
    parameter from ``device::DefaultGemmConfiguration<OpClassTensorOp, ArchTag,
    ElementA, ElementB, ElementC, ElementAccumulator>``, and CUTLASS specialises
    that trait for *generic* element types at three tags only: Sm70, Sm75, Sm80.
    ``Sm86`` has none; ``Sm89``'s are FP8-only.  Naming either for fp16/bf16
    selects the undefined primary template, so every rendered TU in the gemm,
    bmm, group_gemm and gemm_log_softmax modules fails with 67 "incomplete type"
    errors — i.e. **no OASR GEMM builds at all** on A10 / A40 / L4 / L40S /
    RTX 4090.  Conv2D and the recurrent family were unaffected and green, which
    is why nothing else caught it.

    Nothing here restates the allowed set: it is parsed back out of the vendored
    CUTLASS headers, so a submodule bump that adds (or drops) a tag moves this
    test with it rather than leaving a stale constant behind.

    CPU-only and arch-independent, like the tile-space tests above — the whole
    point is to hold the line for targets that are not in the box.
    """

    #: The 2.x lane.  SM90/SM100 use the 3.x CollectiveBuilder, whose arch tags
    #: are a different constraint entirely, so they are out of scope here.
    CUTLASS_2X_TARGETS = (75, 80, 86, 89, 120)

    @staticmethod
    def _generic_tensorop_tags():
        """Arch tags with a generic-element ``DefaultGemmConfiguration``.

        "Generic" is the point: a specialisation written against concrete types
        (``float_e4m3_t``, ``int8_t``, ``double``) does not answer for half, and
        ``Sm89`` has four of those and nothing else — which is exactly how it
        looked buildable.
        """
        import re
        from pathlib import Path

        import assets

        # Gated, not asserted.  The submodule is a build prerequisite, not a
        # checkout one -- ``test-cpu.yml`` deliberately initialises no
        # submodules because it compiles nothing -- so a hard assert turned the
        # one job this test was written for red.  Going through the registry
        # makes the skip *counted*: it shows up in the end-of-run ``external
        # assets:`` table, and ``--strict-assets`` (which both GPU workflows
        # pass, and where the submodule is really present) still fails on it.
        header = Path(assets.require("CUTLASS_DIR")) / (
            "include/cutlass/gemm/device/default_gemm_configuration.h"
        )
        pattern = re.compile(
            r"struct\s+DefaultGemmConfiguration<\s*"
            r"arch::OpClassTensorOp\s*,\s*arch::(Sm\d+)\s*,\s*"
            r"ElementA\s*,\s*ElementB\s*,",
            re.MULTILINE,
        )
        tags = set(pattern.findall(header.read_text()))
        assert tags, "the DefaultGemmConfiguration parse found nothing; CUTLASS moved"
        return tags

    @staticmethod
    def _declared_arch_tags():
        """``sm -> "SmNN"`` as ``CutlassArch`` declares it, parsed from the header.

        Read from the source rather than instantiated, because the mapping is a
        C++ type alias with no Python surface — and because a *missing*
        specialisation has to be visible as a missing key, not as an exception.
        """
        import re

        from helpers import REPO_ROOT

        header = REPO_ROOT / "include" / "oasr" / "gemm" / "cutlass_gemm_configs.h"
        pattern = re.compile(
            r"struct\s+CutlassArch<(\d+)>\s*\{\s*using\s+Type\s*=\s*cutlass::arch::(Sm\d+)\s*;",
            re.MULTILINE,
        )
        return {int(sm): tag for sm, tag in pattern.findall(header.read_text())}

    def test_cutlass_specialises_the_trait_for_sm80_and_not_for_sm86_or_sm89(self):
        """The fact the mapping rests on, asserted against CUTLASS itself.

        If a CUTLASS bump ever adds half support for Sm86/Sm89, this fails and
        the collapse below becomes a choice rather than a requirement.
        """
        tags = self._generic_tensorop_tags()
        assert "Sm80" in tags, "Sm80 lost its generic tensor-op configuration"
        assert "Sm86" not in tags
        assert "Sm89" not in tags

    def test_every_2x_target_names_a_buildable_tag(self):
        """The join: declared tag ∈ what CUTLASS can actually instantiate."""
        buildable = self._generic_tensorop_tags()
        declared = self._declared_arch_tags()
        for sm in self.CUTLASS_2X_TARGETS:
            assert sm in declared, f"sm{sm} is a JIT target with no CutlassArch specialisation"
            assert declared[sm] in buildable, (
                f"CutlassArch<{sm}>::Type is cutlass::arch::{declared[sm]}, which has no "
                f"generic DefaultGemmConfiguration — every gemm/bmm/group_gemm TU for sm{sm} "
                f"fails to compile. Buildable tags: {sorted(buildable)}"
            )

    def test_ampere_and_later_collapse_onto_sm80(self):
        """Stated as a unit so the intent survives the join above.

        Keeps its meaning if CUTLASS ever grows a *non-half* Sm86 tag: the
        collapse is deliberate, not merely whatever happens to pass.
        """
        declared = self._declared_arch_tags()
        for sm in (80, 86, 89, 120):
            assert (
                declared[sm] == "Sm80"
            ), f"CutlassArch<{sm}> should map to Sm80, got {declared[sm]}"
        assert declared[75] == "Sm75", "Turing has its own m16n8k8 composition"

    def test_the_collapse_does_not_merge_the_tile_spaces(self):
        """Each target keeps its own smem budget, config space and cache key.

        The tag is the only thing that collapses.  If ``compile_name`` ever
        dropped the ``smNN`` prefix, two architectures would share one autotune
        cache entry and one JIT hash directory.
        """
        from oasr.jit.gemm import _SM_MAX_SMEM_BYTES, get_unique_compile_configs

        assert _SM_MAX_SMEM_BYTES[80] != _SM_MAX_SMEM_BYTES[86], "sm_86 took A100's smem budget"
        for sm in (80, 86, 89, 120):
            names = get_unique_compile_configs(sm)
            assert names, f"sm{sm} produced an empty config space"
            assert all(
                n.startswith(f"sm{sm}_") for n in names
            ), f"sm{sm} config names are not arch-keyed"
        assert set(get_unique_compile_configs(86)) != set(get_unique_compile_configs(80))


# ---------------------------------------------------------------------------
# The CUTLASS 3.x Conv2D config space
# ---------------------------------------------------------------------------


class TestConv3xConfigSpaceIsBuildable:
    """Conv is not GEMM, and this config space used to assume it was.

    ``CutlassConv2dConfigSm90`` was written as a field-for-field mirror of
    ``CutlassGemmConfigSm90``, and three of the borrowed decisions meant **no**
    dense Conv2D kernel compiled on sm_90 or sm_100 — the whole JIT module, 205
    errors, on every H100 / H200 / B200 / GB200:

    * the mainloop got GEMM's schedule tags, but ``conv``'s ``CollectiveBuilder``
      is ``enable_if``'d on ``conv::KernelImplicitTmaWarpSpecialized*``;
    * the K mode was flat, but implicit GEMM's K axis is the filter's (C, S, R)
      modes, so the builder wants a nested ``Shape<Int<BK>>``;
    * the M tile was scaled by a co-operating-SM count, which SM100 rejects
      outright ("Invalid TileShape_M.") — the 2-SM atom comes from the cluster.

    None of that is visible from Python, so these tests hold the two *shape*
    rules that bound the emitted space, and the arch-agnostic invariants that
    let a CPU box speak for a GPU it does not have. The C++ side is gated by
    compiling the rendered TUs; see ``.artifacts/arch_portability_audit.md`` § A2.
    """

    #: The two CUTLASS 3.x conv targets.
    TARGETS = (90, 100)

    def test_the_default_is_a_config_that_gets_compiled(self):
        """Per arch, because SM90 and SM100 are generated at different K tiles.

        The module compiles exactly what the generator emits and the functional
        API looks the default up by ``compile_name``, so a default outside the
        set is an ``AttributeError`` on the first un-tuned call — the same
        invariant ``default_config_for_sm`` carries on the GEMM side.
        """
        from oasr.jit.conv import CutlassConv2dConfigSm90, get_unique_conv2d_compile_configs

        for sm in self.TARGETS:
            default = CutlassConv2dConfigSm90(
                tile_m=128,
                tile_n=128,
                tile_k=128 if sm == 90 else 64,
                cluster_m=1,
                cluster_n=1,
                kStages=3,
                kSmVersion=sm,
            )
            built = get_unique_conv2d_compile_configs(sm)
            assert default.compile_name in built, (
                f"sm{sm}: CONV2D_DEFAULT {default.compile_name} is not in the "
                f"{len(built)} configs the module compiles"
            )

    def test_sm90_tiles_fit_the_mainloop_pipeline(self):
        """One stage has to fit, or ``StageCountAutoCarveout`` resolves to zero.

        The failure is ``"Specialization requires Stages set to value 1 or more"``
        at compile time, and one unbuildable variant fails the whole module — so
        the budget is a filter on the space, not a note.
        """
        from oasr.jit.conv import _sm90_conv_tile_ok, get_unique_conv2d_compile_configs

        for cfg in get_unique_conv2d_compile_configs(90).values():
            assert _sm90_conv_tile_ok(cfg.tile_m, cfg.tile_n, cfg.tile_k), cfg.compile_name

        # …and the predicate is the measured boundary, not a guess: 106,496 B
        # builds and the next rung (114,688 B) does not.
        assert _sm90_conv_tile_ok(256, 160, 128) and not _sm90_conv_tile_ok(256, 192, 128)
        assert _sm90_conv_tile_ok(192, 224, 128) and not _sm90_conv_tile_ok(192, 256, 128)

    def test_sm100_pairs_a_256_row_tile_only_with_the_2sm_atom(self):
        from oasr.jit.conv import _sm100_conv_tile_ok, get_unique_conv2d_compile_configs

        for cfg in get_unique_conv2d_compile_configs(100).values():
            assert _sm100_conv_tile_ok(cfg.tile_m, cfg.cluster_m), cfg.compile_name

        assert _sm100_conv_tile_ok(256, 2) and not _sm100_conv_tile_ok(256, 1)
        assert _sm100_conv_tile_ok(128, 1) and _sm100_conv_tile_ok(64, 1)

    def test_the_config_carries_no_gemm_only_axis(self):
        """``pingpong`` and ``kSMs`` are gone, and must stay gone.

        Both were inert-looking mirrors of the GEMM config. ``pingpong`` doubled
        the emitted kernel count with a schedule conv does not have on SM90;
        ``kSMs`` silently doubled the M tile into a shape SM100 refuses. A field
        that encodes something the kernel cannot express is not harmless.
        """
        from oasr.jit.conv import CutlassConv2dConfigSm90

        fields = set(CutlassConv2dConfigSm90.__dataclass_fields__)
        assert not (fields & {"pingpong", "kSMs"}), f"GEMM-only axis is back: {fields}"

    def test_every_emitted_config_has_a_unique_arch_keyed_name(self):
        """Two configs under one name would silently compile one kernel twice.

        Dropping ``pingpong`` removed it from ``compile_name`` too, so this is
        the check that the remaining fields still separate every variant.
        """
        from oasr.jit.conv import get_unique_conv2d_compile_configs

        for sm in self.TARGETS:
            cfgs = get_unique_conv2d_compile_configs(sm)
            assert cfgs, f"sm{sm} emitted nothing"
            assert all(n.startswith(f"sm{sm}_") for n in cfgs), f"sm{sm} names are not arch-keyed"
            rebuilt = {c.compile_name for c in cfgs.values()}
            assert len(rebuilt) == len(cfgs)

    def test_the_space_stays_within_the_module_build_budget(self):
        """A wide 3.x space is a first-call OOM, not merely a slow build.

        Each CUTLASS 3.x conv translation unit peaks near 3.7 GB in ``cicc`` and
        ninja defaults to nproc-way parallelism, so the emitted count is a
        resource decision. Held near the GEMM SM90 space (16) rather than left
        to grow silently; raise it from a measurement.
        """
        from oasr.jit.conv import get_unique_conv2d_compile_configs

        for sm in self.TARGETS:
            n = len(get_unique_conv2d_compile_configs(sm))
            assert n <= 24, f"sm{sm} emits {n} conv TUs; widen deliberately, with a measurement"


# ---------------------------------------------------------------------------
# Which architectures the JIT claims, and which it refuses
# ---------------------------------------------------------------------------


class TestTargetArchitectures:
    """The served set is a table, and everything reads the same table.

    It used to be a nearest-lower walk over a map that listed ``70`` and
    ``103`` as *families*, neither of which could build:

    * **sm_70** — the toolchain dropped ``compute_70`` (``nvcc
      --list-gpu-arch`` starts at ``compute_75``) and no ``CutlassArch<70>``
      exists. It is refused by name.
    * **sm_103** — CUTLASS 4.6.1 has no *dense* FP16/BF16 GEMM collective for
      ``arch::Sm103``; the SM100 dense builder's ``enable_if`` names ``Sm100``
      alone, and the only ``sm103_*`` GEMM builder is block-scaled. So there is
      no 103 *family* -- but a B300 still runs the SM100 collectives, so sm_103
      is a *raw capability served by family 100*, built as ``sm_100f``.

    Both reached ``_render_all_variants`` and died on
    ``AttributeError: 'CutlassGemmConfig' object has no attribute 'tile_m'`` --
    a 2.x config handed to the 3.x template, forty frames from the cause.

    A nearest-lower rule is also unsafe in its own right for the 3.x lane:
    ``CUTLASS_ARCH_MMA_SM100_ENABLED`` requires ``__CUDA_ARCH__ == 1000``
    exactly, so a part resolved *down* onto the 100 family would compile and
    then take ``CUTE_INVALID_CONTROL_PATH`` at run time. Refusing by name is
    the only answer that is neither a crash nor a wrong kernel.

    CPU-only: the tables are pure Python.
    """

    def test_volta_is_gone_from_every_table(self):
        """sm_70 is not a target, anywhere that names targets."""
        from helpers import REPO_ROOT

        from oasr.jit.core import _SM_FAMILY, _TARGET_SMS

        assert 70 not in _SM_FAMILY and 70 not in _TARGET_SMS

        # The C++ runtime table has to agree, or an AOT build would dispatch to
        # a family the JIT never emits.
        arch_dispatch = (REPO_ROOT / "include/oasr/common/arch_dispatch.h").read_text()
        assert "if (sm >= 70) return 70;" not in arch_dispatch
        assert "case 70:" not in arch_dispatch

        # …and so does the build default.
        cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
        assert "CMAKE_CUDA_ARCHITECTURES 70 " not in cmake

    def test_the_cpp_family_table_matches_the_python_one(self):
        """AOT and JIT must resolve a device to the *same* family.

        ``resolveSmVersion`` is the AOT half of ``_SM_FAMILY``. If it names a
        family the generators do not emit, an AOT build dispatches into a
        ``CutlassArch`` specialization that does not exist -- which is what a
        leftover ``103`` did after sm_103 moved onto the 100 family: the JIT
        built a B300 as family 100 while the C++ switch still claimed 103.
        """
        import re

        from helpers import REPO_ROOT

        from oasr.jit.core import _TARGET_SMS

        src = (REPO_ROOT / "include/oasr/common/arch_dispatch.h").read_text()
        resolve = src[src.index("inline int resolveSmVersion") :]
        resolve = resolve[: resolve.index("\n}")]
        families = {int(m) for m in re.findall(r"if \(sm >= \d+\) return (\d+);", resolve)}
        assert families == set(_TARGET_SMS), (
            f"C++ resolveSmVersion families {sorted(families)} != "
            f"Python _TARGET_SMS {sorted(_TARGET_SMS)}"
        )

        cases = {int(m) for m in re.findall(r"case (\d+):", src)}
        assert cases == set(
            _TARGET_SMS
        ), f"C++ dispatch cases {sorted(cases)} != Python _TARGET_SMS {sorted(_TARGET_SMS)}"

    def test_an_unserved_capability_raises_naming_itself(self):
        """Not a silent resolve-down, and not an AttributeError later."""
        import oasr.jit.core as core

        # sm_103 is *not* here: it is served by the 100 family via sm_100f.
        for sm, cap in ((70, (7, 0)), (110, (11, 0)), (72, (7, 2)), (62, (6, 2))):
            original = core._get_cuda_arch
            core._get_cuda_arch = lambda cap=cap: cap
            try:
                with pytest.raises(RuntimeError, match=rf"sm_{sm}\b"):
                    core._get_target_sm()
            finally:
                core._get_cuda_arch = original

    def test_every_served_capability_resolves_to_a_family_with_a_config_space(self):
        """The join: what the device map promises, the generators must supply."""
        from oasr.jit.conv import get_unique_conv2d_compile_configs
        from oasr.jit.core import _SM_FAMILY, _TARGET_SMS
        from oasr.jit.gemm import get_unique_compile_configs

        assert set(_SM_FAMILY.values()) == set(_TARGET_SMS)
        for raw, family in _SM_FAMILY.items():
            assert family in _TARGET_SMS, raw
            for name, fn in (
                ("gemm", get_unique_compile_configs),
                ("conv2d", get_unique_conv2d_compile_configs),
            ):
                cfgs = fn(family)
                assert cfgs, f"sm_{raw} -> family {family}: {name} space is empty"

    def test_the_config_spaces_refuse_an_unserved_family(self):
        """No ``else`` fallthrough: an unlisted SM must not inherit SM120's tiles."""
        from oasr.jit.conv import get_unique_conv2d_compile_configs
        from oasr.jit.gemm import get_unique_compile_configs

        for sm in (70, 103, 110, 999):
            for fn in (get_unique_compile_configs, get_unique_conv2d_compile_configs):
                with pytest.raises(ValueError, match=rf"sm_{sm}\b"):
                    fn(sm)

    def test_a_3x_family_only_serves_another_part_through_a_family_target(self):
        """The rule that makes the explicit table necessary rather than tidy.

        A capability may be served by a *different* capability's kernels only
        when those kernels actually run there. For the 2.x ``mma.sync`` lane
        that is free (sm_87/88 on the 86 family, sm_121 on the 120 one). For
        the CUTLASS 3.x lane it is not: those MMA paths are gated on an exact
        ``__CUDA_ARCH__``, so the build must name a CUDA **family** target or
        every atom is preprocessed out and the kernel traps at run time.

        sm_103 on the 100 family is the one case, and ``_GENCODE_TARGET`` is
        what makes it sound.
        """
        from oasr.jit.core import _GENCODE_TARGET, _SM_FAMILY

        THREE_X = {90, 100}
        for raw, family in _SM_FAMILY.items():
            if family in THREE_X and raw != family:
                target = _GENCODE_TARGET.get(raw, "")
                assert target.endswith("f"), (
                    f"sm_{raw} is served by the arch-exact 3.x family {family} but builds "
                    f"for {target or f'sm_{raw}a'}; its MMA paths are gated on "
                    f"__CUDA_ARCH__ == {family * 10} and would be compiled out, leaving "
                    f"CUTE_INVALID_CONTROL_PATH at run time"
                )

    def test_blackwell_ultra_rides_the_sm100_family_target(self):
        """sm_103 is served by sm_100's kernels, through a CUDA *family* target.

        Two halves, both load-bearing:

        * **The family.** CUTLASS 4.6.1 has no dense FP16/BF16 GEMM collective
          for ``arch::Sm103`` -- the SM100 dense builder's ``enable_if`` names
          ``Sm100`` alone and the only ``sm103_*`` GEMM builder is block-scaled
          -- so 103 compiles the Sm100 collectives.
        * **The target.** Those cannot be built as ``sm_103a``:
          ``CUTLASS_ARCH_MMA_SM100_ENABLED`` is gated on ``__CUDA_ARCH__ ==
          1000`` exactly, so at 1030 every tcgen05 atom is preprocessed out and
          the kernel reaches ``CUTE_INVALID_CONTROL_PATH`` at run time -- it
          *compiles*, which is what makes it dangerous. ``sm_100f`` defines
          ``__CUDA_ARCH__ == 1000`` plus the family macro, so CUTLASS enables
          ``SM100F`` and emits the same code (measured byte-identical PTX).

        sm_100 uses the same target, so both share one JIT cache entry.
        """
        import oasr.jit.core as core

        flags = {}
        original = core._get_cuda_arch
        try:
            for cap in ((10, 0), (10, 3)):
                core._get_cuda_arch = lambda cap=cap: cap
                assert core._get_target_sm() == 100
                flags[cap] = [f for f in core._default_cuda_cflags() if "gencode" in f]
        finally:
            core._get_cuda_arch = original

        assert flags[(10, 0)] == flags[(10, 3)], f"must be one binary: {flags}"
        assert "sm_100f" in flags[(10, 3)][0], (
            f"sm_103 needs the family target; an arch-conditional one compiles the "
            f"tcgen05 atoms out: {flags[(10, 3)]}"
        )
        assert "103a" not in flags[(10, 3)][0]

    def test_only_the_sm100_family_overrides_its_gencode_target(self):
        """Everything else builds for its own capability, and must keep doing so.

        A family target is not a free generalisation: ``code=sm_86`` produces
        sm_86 SASS, which will not load on an sm_87 device, and these flags
        embed no PTX to JIT from. So sm_87 / sm_88 / sm_121 stay on their own
        targets -- safe because the 2.x ``mma.sync`` lane they land on is not
        arch-conditional.
        """
        import oasr.jit.core as core

        assert set(core._GENCODE_TARGET) == {100, 103}
        original = core._get_cuda_arch
        try:
            for cap, expect in (
                ((8, 7), "compute_87,code=sm_87"),
                ((8, 8), "compute_88,code=sm_88"),
                ((9, 0), "compute_90a,code=sm_90a"),
                ((12, 1), "compute_121a,code=sm_121a"),
            ):
                core._get_cuda_arch = lambda cap=cap: cap
                got = [f for f in core._default_cuda_cflags() if "gencode" in f][0]
                assert expect in got, f"sm_{cap[0] * 10 + cap[1]}: {got}"
        finally:
            core._get_cuda_arch = original


class TestSm100GemmTileSpace:
    """Every emitted SM100 GEMM tile must be one CUTLASS can build.

    17 of 37 could not. Thirteen because ``CutlassGemmConfigSm90`` scaled the
    tile M by the co-operating-SM count -- ``BM * kSMs`` made the 256-row
    configs 512 and tripped ``static_assert(M == 128 || M == 256, "Invalid
    TileShape_M.")`` -- the *same* defect the conv config inherited from this
    one, and that A2 fixed on the conv side only. Four more because the 2-SM
    16-bit epilogue refuses a ``CtaN`` above 128 that is not a multiple of 64.

    One unbuildable variant fails the whole JIT module, so the effect was that
    **no** GEMM, BMM or grouped GEMM built on B200 at all.
    """

    def test_every_emitted_tile_satisfies_the_cutlass_constraints(self):
        from oasr.jit.gemm import _sm100_gemm_tile_ok, get_unique_compile_configs

        cfgs = get_unique_compile_configs(100)
        assert cfgs
        for name, cfg in cfgs.items():
            assert _sm100_gemm_tile_ok(cfg.tile_m, cfg.tile_n, cfg.kSMs), name

    def test_the_predicate_is_the_measured_boundary(self):
        """Stated as units so the rules survive a rewrite of the ladder."""
        from oasr.jit.gemm import _sm100_gemm_tile_ok

        # 1-SM atom: M in {64, 128}; 2-SM atom: M in {128, 256}.
        assert _sm100_gemm_tile_ok(128, 128, 1) and not _sm100_gemm_tile_ok(256, 128, 1)
        assert _sm100_gemm_tile_ok(256, 128, 2) and not _sm100_gemm_tile_ok(512, 128, 2)
        assert not _sm100_gemm_tile_ok(64, 128, 2)
        # N: a multiple of 8, at most 256.
        assert not _sm100_gemm_tile_ok(128, 512, 2) and not _sm100_gemm_tile_ok(128, 132, 1)
        # 2-SM 16-bit epilogue: N above 128 must divide by 64…
        assert not _sm100_gemm_tile_ok(128, 160, 2) and not _sm100_gemm_tile_ok(128, 224, 2)
        assert _sm100_gemm_tile_ok(128, 192, 2) and _sm100_gemm_tile_ok(128, 256, 2)
        # …and that restriction is 2-SM only.
        assert _sm100_gemm_tile_ok(128, 160, 1) and _sm100_gemm_tile_ok(128, 224, 1)

    def test_the_tile_m_is_not_scaled_by_the_sm_count(self):
        """``kSMs`` selects the schedule; it must never scale the tile.

        Read out of the header, not inferred from the Python config: the
        scaling lived in the C++ ``TileShape`` alias, so the emitted config
        still says ``tile_m=256`` either way and a Python-side assertion would
        pass while the build failed. This is the half a config-space test
        cannot see, which is why the compile matrix in
        ``.artifacts/arch_portability_audit.md`` § A11 is the real gate.
        """
        import re

        from helpers import REPO_ROOT

        src = (REPO_ROOT / "include/oasr/gemm/cutlass_gemm_configs.h").read_text()
        tile = re.search(r"struct CutlassGemmConfigSm90 \{.*?using TileShape = ([^;]+);", src, re.S)
        assert tile, "CutlassGemmConfigSm90's TileShape alias moved"
        assert "kSMs" not in tile.group(1), (
            f"TileShape scales M by kSMs again: {tile.group(1).strip()} — CUTLASS wants the "
            f"combined MMA extent, so 256 becomes 512 and trips 'Invalid TileShape_M.'"
        )

        # …and the 256-row tiles the doubling broke are in the space.
        from oasr.jit.gemm import get_unique_compile_configs

        m256 = [c for c in get_unique_compile_configs(100).values() if c.tile_m == 256]
        assert m256, "every 256-row SM100 tile was filtered out"
        assert all(c.kSMs == 2 for c in m256), "a 256-row MMA tile needs the 2-SM atom"

    def test_sm90_is_unaffected(self):
        """SM90 always passes kSMs=1, so removing the scaling changed nothing.

        Pinned because the fix touched a struct both architectures share.
        """
        from oasr.jit.gemm import get_unique_compile_configs

        cfgs = get_unique_compile_configs(90)
        assert len(cfgs) == 16
        assert all(c.kSMs == 1 for c in cfgs.values())


def _struct_body(src: str, name: str) -> str:
    """Slice one ``struct <name> { ... };`` out of a header, braces balanced."""
    start = src.index(f"struct {name} {{")
    depth, i = 0, src.index("{", start)
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    raise AssertionError(f"unbalanced braces in struct {name}")


class TestSm100BatchedAndGroupedGemm:
    """BMM and grouped GEMM had their *own* SM100 breakage, past the tile space.

    A11 fixed the shared tile ladder and ``gemm`` built -- but ``bmm`` and
    ``group_gemm`` still failed, for two unrelated reasons, and one unbuildable
    variant fails the whole JIT module either way:

    * **BMM** declared a source operand (``ElementC = ElementCD``) it never
      used. Both call sites hardcode ``beta = 0`` and ``oasr.functionals.bmm``
      has no C parameter, but a non-void ElementC reserves C-tile smem
      unconditionally at compile time. Measured carveout on SM100 at
      ``128xNx128`` fp16 -- 25600 / 33792 / 82944 / 51200 / 115712 / 67584
      bytes for N = 64…256, against a flat 17408 for void C -- left
      ``(232448 - carveout) / stage_bytes`` below the required two stages at
      N = 224 and N = 256.
    * **Grouped GEMM** named ``KernelPtrArrayTmaWarpSpecializedCooperative``
      unconditionally. That schedule has no SM100 specialization, so the
      builder fell through to its primary template and failed with
      ``has no member "CollectiveOp"``.

    Both live in C++ template arguments, so no config-space assertion can see
    them; these read the headers.
    """

    @staticmethod
    def _template_src() -> str:
        from helpers import REPO_ROOT

        return (REPO_ROOT / "include/oasr/gemm/gemm_cutlass_template_sm90.h").read_text()

    @staticmethod
    def _config_src() -> str:
        from helpers import REPO_ROOT

        return (REPO_ROOT / "include/oasr/gemm/cutlass_gemm_configs.h").read_text()

    def test_the_bmm_epilogue_declares_no_source_operand(self):
        body = _struct_body(self._template_src(), "CutlassBmmKernelSm90")
        epilogue = body[body.index("using CollectiveEpilogue") :]
        epilogue = epilogue[: epilogue.index(";")]
        # ElementC is the argument following ElementCompute.
        assert "ElementCompute, void," in " ".join(epilogue.split()), (
            "BMM's epilogue took a source operand again. D = alpha * (A @ B^T) has none, "
            "and a non-void ElementC reserves C-tile smem unconditionally, which drops the "
            "SM100 mainloop below two stages at tile N of 224 and 256."
        )

    def test_the_bmm_refuses_a_beta_it_cannot_honour(self):
        """Declare, don't ignore: no source operand means beta must be zero."""
        body = _struct_body(self._template_src(), "CutlassBmmKernelSm90")
        assert "if (beta != 0.0f)" in body and "NOT_SUPPORTED" in body, (
            "BMM must refuse a non-zero beta rather than silently return "
            "alpha * A @ B^T to a caller expecting D to be accumulated into."
        )

    def test_the_grouped_schedules_are_taken_from_the_config(self):
        body = _struct_body(self._template_src(), "CutlassGroupGemmKernelSm90")
        for alias in ("EpilogueSchedule", "MainloopSchedule"):
            line = next(ln for ln in body.splitlines() if ln.strip().startswith(f"using {alias} ="))
            assert "CutlassGemmConfig::Group" in line, (
                f"grouped {alias} is hardcoded again ({line.strip()}); the cooperative "
                f"ptr-array schedule has no SM100 specialization, so this must come from "
                f"the config's per-arch selector."
            )

    def test_the_config_maps_sm100_to_the_sm100_grouped_schedules(self):
        body = _struct_body(self._config_src(), "GemmScheduleSelector<100, kSMs, kPingpong>")
        assert "GroupSMTypeAdapter<kSMs>" in body
        for k, sched in ((1, "1SmSm100"), (2, "2SmSm100")):
            adapter = _struct_body(self._config_src(), f"GroupSMTypeAdapter<{k}>")
            assert f"KernelPtrArrayTmaWarpSpecialized{sched}" in adapter
            assert f"PtrArrayTmaWarpSpecialized{k}Sm" in adapter

    def test_sm90_keeps_the_cooperative_grouped_schedule(self):
        """The SM100 split must not disturb the arch that already worked."""
        src = self._config_src()
        primary = _struct_body(src, "GemmScheduleSelector")
        assert "KernelPtrArrayTmaWarpSpecializedCooperative" in primary
        assert "PtrArrayTmaWarpSpecializedCooperative" in primary
