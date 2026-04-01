# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT compilation infrastructure for OASR CUDA kernels.

Uses Ninja for parallel builds (``cpp_ext``), file-lock protected caching
(``cubin_loader``), and Jinja2 code generation (``templates``).
"""

from .core import JitSpec, JinjaJitSpec, gen_jit_spec, gen_jinja_jit_spec, clear_cache
from .cubin_loader import write_if_different

from oasr.compilation_context import CompilationContext

#: Module-level singleton ``CompilationContext`` used by JIT generators
#: that need architecture-aware NVCC flags.
current_compilation_context = CompilationContext()
