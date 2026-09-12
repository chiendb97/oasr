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
