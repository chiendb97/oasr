# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the in-tree GPU WFST beam-search decoder.

Compiles the self-contained CUDA WFST decoder (migrated from the standalone ``wfst``
project) and its TVM-FFI launcher into a JIT module, mirroring the GPU CTC decoder
(:mod:`oasr.jit.ctc_decoder`).  ``decoder.cu`` is a single translation unit that
``#include``s all kernel headers, so it is compiled whole-program (no ``-rdc``); the
host-only ``.cc`` sources compile with the host compiler and link alongside it.
"""

from . import env
from .core import JitSpec, gen_jit_spec

_WFST_DIR = env.OASR_CSRC_DIR / "decoder" / "wfst"
_WFST_TESTS_DIR = env.OASR_CSRC_DIR / "tests" / "wfst"


def gen_wfst_decoder_module() -> JitSpec:
    """Generate the JIT spec for the GPU WFST decoder."""
    sources = [
        _WFST_DIR / "decoder.cu",          # GpuDecoder impl (single-TU with kernels)
        _WFST_DIR / "graph_io.cc",         # hlg.img loader (host)
        _WFST_DIR / "wfst_decoder.cu",     # TVM-FFI launcher (opaque-handle wrappers)
        _WFST_DIR / "wfst_decoder_jit_binding.cu",  # TVM-FFI exports
    ]
    # The decoder uses device lambdas / relaxed constexpr; --use_fast_math (in the
    # default flags) matches the standalone build so device SASS — hence numerics and
    # perf — are identical. -lineinfo mirrors the standalone build (profiling only).
    return gen_jit_spec(
        "wfst_decoder",
        sources,
        extra_cuda_cflags=[
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
        ],
    )


def gen_wfst_cpu_reference_module() -> JitSpec:
    """Generate the JIT spec for the **test-only** WFST CPU reference oracle.

    Compiles the exact-semantics host reference decoder (``cpu_reference.cc``, the target
    behavior the GPU decoder must reproduce) behind a self-contained TVM-FFI launcher that
    loads its own graph image from a path.  Kept in a *separate* module from
    :func:`gen_wfst_decoder_module` so the production WFST decoder ``.so`` carries no
    reference-decoder code — this module is built only by ``tests/test_wfst_decoder.py`` to
    pin GPU-decoder parity.  All sources are host-only (no device kernels).
    """
    sources = [
        _WFST_TESTS_DIR / "cpu_reference.cc",               # exact-semantics oracle (host)
        _WFST_DIR / "graph_io.cc",                          # hlg.img loader (host)
        _WFST_TESTS_DIR / "wfst_cpu_reference.cu",          # TVM-FFI launcher (graph path in)
        _WFST_TESTS_DIR / "wfst_cpu_reference_jit_binding.cu",  # TVM-FFI exports
    ]
    return gen_jit_spec(
        "wfst_cpu_reference",
        sources,
        extra_cuda_cflags=[
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
    )
