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
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .split("\n")[0]
        )
        major, minor = out.split(".")
        return (int(major), int(minor))
    except Exception:
        return (8, 0)  # Safe default: SM80 (Ampere)


#: Raw compute capability -> the kernel family compiled for it.
#:
#: An explicit table, not "the highest entry at or below sm".  A nearest-lower
#: rule is only safe where the family's kernels actually run on the newer part,
#: and for the CUTLASS 3.x lane they do not: its MMA paths are gated on an exact
#: ``__CUDA_ARCH__`` (``CUTLASS_ARCH_MMA_SM100_ENABLED`` needs ``== 1000``), so
#: an sm_103 part resolved down to the 100 family would compile and then take
#: ``CUTE_INVALID_CONTROL_PATH`` at run time.  Anything absent here raises with
#: its own number instead.
#:
#: Present by family:
#:   * 2.x lane (mma.sync, forward-compatible): 75, 80, 86/87/88, 89, 120/121.
#:   * 3.x lane (TMA + wgmma/tcgen05, arch-exact): 90, 100.
#:
#: **sm_70 (Volta)** is absent because the toolchain dropped it -- ``nvcc
#: --list-gpu-arch`` starts at ``compute_75`` -- and no ``CutlassArch<70>``
#: exists.  It was listed here for a long time and never built.
#:
#: **sm_103 (Blackwell Ultra)** is served by the **100** family, not by a tag of
#: its own.  CUTLASS 4.6.1 has no *dense* FP16/BF16 GEMM collective for
#: ``arch::Sm103`` -- the SM100 dense builder's ``enable_if`` names ``Sm100``
#: alone and the only ``sm103_*`` GEMM builder is block-scaled (NVFP4/MXFP8) --
#: so OASR compiles the Sm100 collectives and reaches B300 through the CUDA
#: *family* target ``sm_100f`` instead.  See :data:`_GENCODE_TARGET`.
_SM_FAMILY = {
    75: 75,
    80: 80,
    86: 86,
    87: 86,
    88: 86,
    89: 89,
    90: 90,
    100: 100,
    103: 100,
    120: 120,
    121: 120,
}

#: The distinct families kernels are generated for, ascending.
_TARGET_SMS = tuple(sorted(set(_SM_FAMILY.values())))


#: Capabilities whose kernels must be built for a CUDA **family** target rather
#: than their own arch-conditional one, and which target that is.
#:
#: Default is the capability itself -- ``sm_87``, plain, for an sm_87 device --
#: with the arch-conditional ``a`` suffix from Hopper up, because CUTLASS 3.x
#: gates every wgmma / tcgen05 atom on ``__CUDA_ARCH_FEAT_SM90_ALL`` and friends,
#: which the plain target does not define: the kernel then *compiles* and aborts
#: at run time down ``CUTE_INVALID_CONTROL_PATH``.
#:
#: sm_100 and sm_103 are the exception.  sm_103 runs the Sm100 collectives (it
#: has no dense FP16/BF16 collective of its own), and an ``sm_103a`` build of
#: those would define ``__CUDA_ARCH__ == 1030``, where
#: ``CUTLASS_ARCH_MMA_SM100_ENABLED`` -- gated on ``== 1000`` exactly -- is off
#: and every tcgen05 atom is preprocessed out.  The family target is the
#: mechanism CUDA 13 provides for this: ``sm_100f`` defines
#: ``__CUDA_ARCH__ == 1000`` plus ``__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000``, so
#: CUTLASS enables ``SM100F`` in place of ``SM100A`` and emits the same code.
#: Measured: the PTX for one of these kernels is byte-identical between
#: ``sm_100a`` and ``sm_100f`` (31,588 lines, 162 ``tcgen05`` instructions), and
#: the whole emitted space compiles identically under both.  Both capabilities
#: use it, so they also share one JIT cache entry.
#:
#: Not applied to sm_121: ``sm_120f`` exists, but sm_121 is served by the 120
#: family's **CUTLASS 2.x** kernels, whose ``mma.sync`` is not arch-conditional,
#: so its own ``sm_121a`` target is already correct.  ``sm_90f`` does not exist.
_GENCODE_TARGET = {
    100: "100f",
    103: "100f",
}


def _gencode_target(sm: int, major: int) -> str:
    """The ``compute_X``/``sm_X`` suffix to build this capability for."""
    override = _GENCODE_TARGET.get(sm)
    if override is not None:
        return override
    return f"{sm}a" if major >= 9 else f"{sm}"


def _get_target_sm() -> int:
    """The compiled kernel family for this device, or raise naming the device.

    Raising is the fix, not a regression: the old rule defaulted to 80 and
    walked up, so an unlisted card silently got another architecture's kernels
    -- a wrong answer where "this GPU is not supported" was the right one.
    """
    major, minor = _get_cuda_arch()
    sm = major * 10 + minor
    try:
        return _SM_FAMILY[sm]
    except KeyError:
        raise RuntimeError(
            f"unsupported GPU architecture sm_{sm}: OASR compiles for "
            f"{', '.join(f'sm_{s}' for s in sorted(_SM_FAMILY))}"
        ) from None


# ---------------------------------------------------------------------------
# JitSpec
# ---------------------------------------------------------------------------


# Header roots every JIT source may transitively include.  They must be part
# of the cache key: ``build_and_load`` short-circuits on an existing library
# without consulting ninja's dependency tracking, so a header-only edit that
# is absent from the hash would silently keep loading the stale binary.
_PROJECT_HEADER_ROOTS = (
    (env.OASR_INCLUDE_DIR, (".h", ".cuh", ".inc")),
    (env.OASR_CSRC_DIR, (".h",)),
)


def _project_headers():
    for root, suffixes in _PROJECT_HEADER_ROOTS:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix in suffixes and p.is_file():
                yield root, p


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
        """Compute hash of all source files + project headers + flags for
        cache invalidation."""
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
        for root, hdr in _project_headers():
            h.update(str(hdr.relative_to(root)).encode())
            h.update(hdr.read_bytes())
        # Third-party headers count too: a CUTLASS submodule bump changes the
        # generated code for every GEMM/BMM/Conv2D module.  See
        # ``env.cutlass_version_stamp`` for why this is version.h and not the
        # whole tree.
        for inc, version_h in env.cutlass_version_stamp():
            h.update(inc.encode())
            h.update(version_h)
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
    arch = _gencode_target(sm, major)
    return [
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-DENABLE_BF16",
        f"-DOASR_TARGET_SM={target_sm}",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
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
