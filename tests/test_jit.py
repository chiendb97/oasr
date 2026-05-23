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
            "conv",
            "conv2d",
            "cudnn_conv2d",
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
