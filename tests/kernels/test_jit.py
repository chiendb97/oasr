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

        from helpers import REPO_ROOT

        header = (
            REPO_ROOT
            / "3rdparty"
            / "cutlass"
            / "include"
            / "cutlass"
            / "gemm"
            / "device"
            / "default_gemm_configuration.h"
        )
        assert header.is_file(), f"CUTLASS submodule missing: {header}"
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
