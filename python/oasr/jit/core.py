# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Core JIT compilation infrastructure.

Mirrors FlashInfer's JIT architecture:
- ``cpp_ext`` for Ninja-based compilation with proper flag construction
- ``cubin_loader`` for concurrent-safe caching with file locking
- ``JitSpec`` / ``JinjaJitSpec`` for specifying compilation units
"""

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from . import env
from .cpp_ext import (
    generate_ninja_build_for_op,
    run_ninja,
)
from .cubin_loader import locked_compile, write_if_different

logger = logging.getLogger("oasr.jit")


# ---------------------------------------------------------------------------
# Device / SM detection
# ---------------------------------------------------------------------------


def _get_cuda_arch() -> Tuple[int, int]:
    """Detect the compute capability of the current CUDA device.

    Returns (major, minor), e.g. (8, 0) for SM80.
    """
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return (props.major, props.minor)
    except ImportError:
        pass
    # Fallback: use nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        major, minor = out.split(".")
        return (int(major), int(minor))
    except Exception:
        return (8, 0)  # Safe default: SM80 (Ampere)


def _get_target_sm() -> int:
    """Get the resolved target SM version for JIT compilation."""
    major, minor = _get_cuda_arch()
    sm = major * 10 + minor
    sm_to_arch = {
        70: 70, 75: 75, 80: 80, 86: 86, 89: 89,
        90: 90, 100: 100, 103: 103, 120: 120,
    }
    target_sm = 80
    for threshold in sorted(sm_to_arch.keys()):
        if sm >= threshold:
            target_sm = sm_to_arch[threshold]
    return target_sm


# ---------------------------------------------------------------------------
# JitSpec
# ---------------------------------------------------------------------------


class JitSpec:
    """Specification for a JIT-compiled CUDA module.

    Encapsulates source files, compiler flags, and include directories.
    Compilation uses Ninja (via ``cpp_ext``) for parallel builds with
    dependency tracking.  Concurrent builds of the same library are
    serialised via file locks (``cubin_loader``).
    """

    def __init__(
        self,
        name: str,
        sources: List[Path],
        extra_cuda_cflags: Optional[List[str]] = None,
        extra_include_dirs: Optional[List[Path]] = None,
        extra_ldflags: Optional[List[str]] = None,
    ):
        self.name = name
        self.sources = [Path(s) for s in sources]
        self.extra_cuda_cflags = extra_cuda_cflags or []
        self.extra_include_dirs = [Path(d) for d in (extra_include_dirs or [])]
        self.extra_ldflags = extra_ldflags or []

    def _content_hash(self) -> str:
        """Compute hash of all source files + flags for cache invalidation."""
        h = hashlib.sha256()
        h.update(self.name.encode())
        for flag in sorted(self.extra_cuda_cflags):
            h.update(flag.encode())
        for flag in sorted(self.extra_ldflags):
            h.update(flag.encode())
        for src in sorted(self.sources, key=str):
            if src.exists():
                h.update(src.read_bytes())
            else:
                h.update(str(src).encode())
        return h.hexdigest()[:16]

    def _get_lib_dir(self) -> Path:
        """Get the build directory for this module."""
        content_hash = self._content_hash()
        return env.OASR_JIT_DIR / self.name / content_hash

    def _get_lib_path(self) -> Path:
        """Get the path to the compiled shared library."""
        return self._get_lib_dir() / f"{self.name}.so"

    def _compile(self, lib_path: str) -> None:
        """Compile sources into a shared library using Ninja."""
        lib_path = Path(lib_path)
        build_dir = lib_path.parent
        build_dir.mkdir(parents=True, exist_ok=True)

        # Generate build.ninja
        ninja_content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_include_dirs=self.extra_include_dirs,
            extra_ldflags=self.extra_ldflags,
            build_dir=build_dir,
        )
        ninja_file = build_dir / "build.ninja"
        write_if_different(ninja_file, ninja_content)

        # Run ninja
        verbose = logger.isEnabledFor(logging.DEBUG)
        run_ninja(workdir=build_dir, ninja_file=ninja_file, verbose=verbose)

    def build_and_load(self):
        """Build if needed (with file-lock protection), then load the module.

        Returns a ``tvm_ffi.Module`` handle.  Functions exported via
        ``TVM_FFI_DLL_EXPORT_TYPED_FUNC`` are accessible as attributes
        or via indexing (e.g., ``module["layernorm"]``).
        """
        lib_path = self._get_lib_path()
        if not lib_path.exists():
            locked_compile(str(lib_path), self._compile)
        import tvm_ffi

        return tvm_ffi.load_module(str(lib_path))


# ---------------------------------------------------------------------------
# JinjaJitSpec
# ---------------------------------------------------------------------------


class JinjaJitSpec(JitSpec):
    """JitSpec that generates source files from Jinja2 templates.

    Instead of compiling static source files, this renders a Jinja2 template
    with baked-in configuration (tile sizes, SM version, etc.) to produce
    a self-contained ``.cu`` file that instantiates exactly one kernel config.
    """

    def __init__(
        self,
        name: str,
        template_name: str,
        template_vars: dict,
        extra_cuda_cflags: Optional[List[str]] = None,
        extra_include_dirs: Optional[List[Path]] = None,
        extra_ldflags: Optional[List[str]] = None,
    ):
        self.template_name = template_name
        self.template_vars = template_vars

        # Render template to generated source file
        gen_src = self._render_source(name)

        super().__init__(
            name=name,
            sources=[gen_src],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_dirs=extra_include_dirs,
            extra_ldflags=extra_ldflags,
        )

    def _render_source(self, name: str) -> Path:
        """Render the Jinja template to a generated .cu file."""
        from .templates import render_template

        rendered = render_template(self.template_name, **self.template_vars)
        gen_path = env.OASR_GEN_SRC_DIR / name / f"{name}.cu"
        write_if_different(gen_path, rendered)
        return gen_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _default_cuda_cflags() -> List[str]:
    """Standard OASR NVCC flags used by all kernel modules."""
    target_sm = _get_target_sm()
    major, minor = _get_cuda_arch()
    sm = major * 10 + minor
    return [
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-DENABLE_BF16",
        f"-DOASR_TARGET_SM={target_sm}",
        f"-gencode=arch=compute_{sm},code=sm_{sm}",
    ]


def gen_jit_spec(
    name: str,
    sources: List[Path],
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
) -> JitSpec:
    """Create a JitSpec with standard OASR compilation flags."""
    return JitSpec(
        name=name,
        sources=sources,
        extra_cuda_cflags=_default_cuda_cflags() + (extra_cuda_cflags or []),
        extra_ldflags=extra_ldflags,
    )


def gen_jinja_jit_spec(
    name: str,
    template_name: str,
    template_vars: dict,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
) -> JinjaJitSpec:
    """Create a JinjaJitSpec with standard OASR compilation flags.

    Like ``gen_jit_spec()`` but renders a Jinja template instead of using
    static source files.
    """
    return JinjaJitSpec(
        name=name,
        template_name=template_name,
        template_vars=template_vars,
        extra_cuda_cflags=_default_cuda_cflags() + (extra_cuda_cflags or []),
        extra_ldflags=extra_ldflags,
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def clear_cache() -> None:
    """Remove all JIT-compiled artifacts."""
    if env.OASR_JIT_DIR.exists():
        shutil.rmtree(env.OASR_JIT_DIR)
