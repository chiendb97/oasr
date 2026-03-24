# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT compilation infrastructure for OASR CUDA kernels."""

from .core import JitSpec, gen_jit_spec, clear_cache, write_if_different

from oasr.compilation_context import CompilationContext

#: Module-level singleton ``CompilationContext`` used by JIT generators
#: that need architecture-aware NVCC flags.
current_compilation_context = CompilationContext()
