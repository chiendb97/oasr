# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Core JIT compilation infrastructure.

Mirrors FlashInfer's JitSpec + gen_jit_spec() pattern.
Uses NVCC for compilation and ctypes for loading.
"""

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from . import env


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


class JitSpec:
    """Specification for a JIT-compiled CUDA module.

    Encapsulates source files, compiler flags, and include directories.
    Supports both AOT and JIT compilation paths.
    """

    def __init__(
        self,
        name: str,
        sources: List[Path],
        extra_cuda_cflags: Optional[List[str]] = None,
        extra_include_dirs: Optional[List[str]] = None,
        extra_ldflags: Optional[List[str]] = None,
    ):
        self.name = name
        self.sources = [Path(s) for s in sources]
        self.extra_cuda_cflags = extra_cuda_cflags or []
        self.extra_include_dirs = extra_include_dirs or []
        self.extra_ldflags = extra_ldflags or []

    def _content_hash(self) -> str:
        """Compute hash of all source files for cache invalidation."""
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

    def _get_lib_path(self) -> Path:
        """Get the path to the compiled shared library."""
        content_hash = self._content_hash()
        cache_dir = env.OASR_JIT_CACHE_DIR / self.name / content_hash
        return cache_dir / f"lib{self.name}.so"

    def _compile(self, lib_path: Path) -> None:
        """Compile sources into a shared library using NVCC."""
        lib_path.parent.mkdir(parents=True, exist_ok=True)

        # Find NVCC
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise RuntimeError("nvcc not found. Please install CUDA toolkit.")

        # Find TVM-FFI include dir
        tvm_ffi_include = _get_tvm_ffi_include_dir()

        # Build include flags
        include_dirs = [
            str(env.OASR_INCLUDE_DIR),
            str(env.OASR_CSRC_DIR),
        ] + self.extra_include_dirs
        if tvm_ffi_include:
            include_dirs.append(tvm_ffi_include)

        include_flags = [f"-I{d}" for d in include_dirs]

        # Compile each source to object files, then link
        obj_files = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for src in self.sources:
                obj_path = Path(tmpdir) / (src.stem + ".o")
                cmd = [
                    nvcc,
                    "-c",
                    str(src),
                    "-o",
                    str(obj_path),
                    "--compiler-options",
                    "-fPIC",
                ] + self.extra_cuda_cflags + include_flags
                subprocess.check_call(cmd)
                obj_files.append(str(obj_path))

            # Link into shared library
            link_cmd = [
                nvcc,
                "-shared",
                "-o",
                str(lib_path),
            ] + obj_files + ["--compiler-options", "-fPIC"] + self.extra_ldflags
            subprocess.check_call(link_cmd)

    def build_and_load(self):
        """Build if needed, then load the shared library.

        Returns a tvm_ffi.Module handle. Functions exported via
        TVM_FFI_DLL_EXPORT_TYPED_FUNC are accessible as attributes
        or via indexing (e.g., ``module["layernorm"]``).
        """
        lib_path = self._get_lib_path()
        if not lib_path.exists():
            self._compile(lib_path)
        import tvm_ffi
        return tvm_ffi.load_module(str(lib_path))


def gen_jit_spec(
    name: str,
    sources: List[Path],
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
) -> JitSpec:
    """Create a JitSpec with standard OASR compilation flags."""
    major, minor = _get_cuda_arch()
    sm = major * 10 + minor
    # Map SM to the nearest ArchTraits specialization
    sm_to_arch = {70: 70, 75: 75, 80: 80, 86: 86, 89: 89, 90: 90, 100: 100, 103: 103, 120: 120}
    target_sm = 80  # default
    for threshold in sorted(sm_to_arch.keys()):
        if sm >= threshold:
            target_sm = sm_to_arch[threshold]
    default_flags = [
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-DENABLE_BF16",
        f"-DOASR_TARGET_SM={target_sm}",
        f"-gencode=arch=compute_{sm},code=sm_{sm}",
    ]
    include_dirs = [str(env.OASR_INCLUDE_DIR), str(env.OASR_CSRC_DIR)]
    include_dirs.extend(env.OASR_CUTLASS_INCLUDE_DIRS)
    return JitSpec(
        name=name,
        sources=sources,
        extra_cuda_cflags=default_flags + (extra_cuda_cflags or []),
        extra_include_dirs=include_dirs,
        extra_ldflags=extra_ldflags,
    )


def write_if_different(path: Path, content: str) -> bool:
    """Write *content* to *path* only if it differs from the current contents.

    Returns True if the file was (re-)written, False if it was already
    up-to-date.  This avoids unnecessary recompilation when the generated
    source has not changed.
    """
    path = Path(path)
    if path.exists():
        existing = path.read_text()
        if existing == content:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def clear_cache() -> None:
    """Remove all JIT-compiled artifacts."""
    if env.OASR_JIT_CACHE_DIR.exists():
        shutil.rmtree(env.OASR_JIT_CACHE_DIR)


def _get_tvm_ffi_include_dir() -> Optional[str]:
    """Get TVM-FFI include directory if available."""
    try:
        import tvm_ffi.libinfo
        return str(tvm_ffi.libinfo.find_include_path())
    except (ImportError, AttributeError):
        return None
