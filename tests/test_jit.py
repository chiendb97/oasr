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
        """Verify conv2d JIT spec can be created."""
        from oasr.jit.conv import gen_conv2d_module

        spec = gen_conv2d_module()
        assert spec.name == "conv2d"
        assert len(spec.sources) == 2

    def test_gen_gemm_module(self):
        """Verify gemm JIT spec can be created."""
        from oasr.jit.gemm import gen_gemm_module

        spec = gen_gemm_module()
        assert spec.name == "gemm"
        assert len(spec.sources) == 2

    def test_gen_bmm_module(self):
        """Verify bmm JIT spec can be created."""
        from oasr.jit.gemm import gen_bmm_module

        spec = gen_bmm_module()
        assert spec.name == "bmm"
        assert len(spec.sources) == 2

    def test_gen_group_gemm_module(self):
        """Verify group_gemm JIT spec can be created."""
        from oasr.jit.gemm import gen_group_gemm_module

        spec = gen_group_gemm_module()
        assert spec.name == "group_gemm"
        assert len(spec.sources) == 2

    def test_gen_all_modules(self):
        """Verify AOT gen_all_modules returns all expected specs."""
        from oasr.aot import gen_all_modules

        specs = gen_all_modules()
        assert len(specs) == 8
        names = [s.name for s in specs]
        assert "activation" in names
        assert "norm" in names
        assert "conv" in names
        assert "conv2d" in names
        assert "cudnn_conv2d" in names
        assert "gemm" in names
        assert "bmm" in names
        assert "group_gemm" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
