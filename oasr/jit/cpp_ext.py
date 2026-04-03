# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer's flashinfer/jit/cpp_ext.py
# (https://github.com/flashinfer-ai/flashinfer)
#
# Ninja-based compilation for CUDA/C++ extensions with proper flag
# construction, dependency tracking, and parallel builds.

import functools
import logging
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

import tvm_ffi
import torch

from . import env as jit_env
from ..compilation_context import CompilationContext

logger = logging.getLogger("oasr.jit")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def parse_env_flags(env_var_name: str) -> List[str]:
    """Parse compiler flags from an environment variable.

    Supports shell-style quoting via ``shlex.split()``.
    """
    env_flags = os.environ.get(env_var_name)
    if env_flags:
        try:
            import shlex

            return shlex.split(env_flags)
        except ValueError as e:
            logger.warning(
                "Could not parse %s with shlex: %s. Falling back to simple split.",
                env_var_name,
                e,
            )
            return env_flags.split()
    return []


def _get_glibcxx_abi_build_flags() -> List[str]:
    return ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]


# ---------------------------------------------------------------------------
# CUDA path / version detection
# ---------------------------------------------------------------------------


@functools.cache
def get_cuda_path() -> str:
    """Find the CUDA toolkit root directory."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    nvcc_path = subprocess.run(["which", "nvcc"], capture_output=True)
    if nvcc_path.returncode == 0:
        cuda_home = os.path.dirname(
            os.path.dirname(nvcc_path.stdout.decode("utf-8").strip())
        )
    else:
        cuda_home = "/usr/local/cuda"
        if not os.path.exists(cuda_home):
            raise RuntimeError(
                f"Could not find nvcc and default {cuda_home=} doesn't exist"
            )
    return cuda_home


@functools.cache
def get_cuda_version() -> str:
    """Return CUDA version string (e.g. ``"12.4"``)."""
    try:
        cuda_home = get_cuda_path()
        nvcc = os.path.join(cuda_home, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if matches:
            return matches[0]
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError):
        pass
    # Fallback to PyTorch's CUDA version
    if torch.version.cuda is not None:
        return torch.version.cuda
    raise RuntimeError(
        "nvcc not found and PyTorch is not built with CUDA support. "
        "Could not determine CUDA version."
    )


def _cuda_version_ge(version_str: str) -> bool:
    """Check if the installed CUDA version is >= *version_str*."""
    from packaging.version import Version

    return Version(get_cuda_version()) >= Version(version_str)


# ---------------------------------------------------------------------------
# Flag builders (FlashInfer-style layered construction)
# ---------------------------------------------------------------------------


def _join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def get_system_includes(cuda_home: str) -> List[str]:
    """Build the list of system include directories."""
    system_includes: List[str] = [
        sysconfig.get_path("include"),
        f"{cuda_home}/include",
    ]

    # TVM-FFI headers
    try:
        system_includes.append(str(tvm_ffi.libinfo.find_include_path()))
        system_includes.append(str(tvm_ffi.libinfo.find_dlpack_include_path()))
    except (AttributeError, TypeError):
        pass

    # OASR project headers
    system_includes.append(str(jit_env.OASR_INCLUDE_DIR.resolve()))
    system_includes.append(str(jit_env.OASR_CSRC_DIR.resolve()))

    # CUTLASS headers
    for d in jit_env.OASR_CUTLASS_INCLUDE_DIRS:
        system_includes.append(str(Path(d).resolve()))

    if cuda_home == "/usr":
        # /usr/include can mess up system includes — see FlashInfer #1793
        try:
            system_includes.remove(f"{cuda_home}/include")
        except ValueError:
            pass

    return system_includes


def build_common_cflags(
    cuda_home: str,
    extra_include_dirs: Optional[List[Path]] = None,
) -> List[str]:
    """Build flags shared by both C++ and CUDA compilation."""
    system_includes = get_system_includes(cuda_home)

    common_cflags: List[str] = []
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        common_cflags.append("-DPy_LIMITED_API=0x03090000")
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            common_cflags.append(f"-I{Path(extra_dir).resolve()}")
    for sys_dir in system_includes:
        common_cflags.append(f"-isystem {sys_dir}")

    return common_cflags


def build_cflags(
    common_cflags: List[str],
    extra_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build C++ (host) compilation flags."""
    cflags = [
        "$common_cflags",
        "-fPIC",
    ]
    if extra_cflags is not None:
        cflags += extra_cflags

    env_extra = parse_env_flags("OASR_EXTRA_CFLAGS")
    if env_extra:
        cflags += env_extra

    return cflags


def build_cuda_cflags(
    common_cflags: List[str],
    extra_cuda_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build NVCC compilation flags."""
    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    cuda_cflags += [
        "$common_cflags",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    # CUDA 12.8+ flag
    if _cuda_version_ge("12.8"):
        cuda_cflags.append("-static-global-template-stub=false")

    # Architecture flags from CompilationContext
    compilation_ctx = CompilationContext()
    global_flags = compilation_ctx.get_nvcc_flags_list()

    if extra_cuda_cflags is not None:
        # If module provides -gencode flags, use those instead of global ones
        module_has_gencode = any(
            flag.startswith("-gencode=") for flag in extra_cuda_cflags
        )
        if module_has_gencode:
            global_non_arch_flags = [
                flag for flag in global_flags if not flag.startswith("-gencode=")
            ]
            cuda_cflags += global_non_arch_flags + extra_cuda_cflags
        else:
            cuda_cflags += global_flags + extra_cuda_cflags
    else:
        cuda_cflags += global_flags

    env_extra = parse_env_flags("OASR_EXTRA_CUDAFLAGS")
    if env_extra:
        cuda_cflags += env_extra

    return cuda_cflags


# ---------------------------------------------------------------------------
# Ninja build file generation
# ---------------------------------------------------------------------------


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_dirs: Optional[List[Path]] = None,
    build_dir: Optional[Path] = None,
) -> str:
    """Generate a Ninja build file for compiling an OASR kernel module.

    Returns the content of the ``build.ninja`` file as a string.
    """
    cuda_home = get_cuda_path()
    common_cflags = build_common_cflags(cuda_home, extra_include_dirs)
    cflags = build_cflags(common_cflags, extra_cflags)
    cuda_cflags = build_cuda_cflags(common_cflags, extra_cuda_cflags)

    ldflags = [
        "-shared",
        f"-L{cuda_home}/lib64",
        f"-L{cuda_home}/lib64/stubs",
        "-lcudart",
        "-lcuda",
    ]

    env_extra_ldflags = parse_env_flags("OASR_EXTRA_LDFLAGS")
    if env_extra_ldflags:
        ldflags += env_extra_ldflags

    if extra_ldflags is not None:
        ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    nvcc = os.environ.get("OASR_NVCC", f"{cuda_home}/bin/nvcc")

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        "",
        "common_cflags = " + _join_multiline(common_cflags),
        "cflags = " + _join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + _join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "ldflags = " + _join_multiline(ldflags),
        "",
        "rule compile",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule cuda_compile",
        "  command = $nvcc --generate-dependencies-with-compile "
        "--dependency-output $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule link",
        "  command = $cxx $in $ldflags -o $out",
        "",
    ]

    # Use absolute paths so ninja files work with any workdir
    output_dir = Path(build_dir) if build_dir is not None else jit_env.OASR_JIT_DIR / name

    objects = []
    for source in sources:
        source = Path(source)
        is_cuda = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_cuda else ".o"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = f"{source.parent.name}_{source.stem}{object_suffix}"
        obj = str((output_dir / obj_name).resolve())
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {source.resolve()}")

    lines.append("")
    output_so = str((output_dir / f"{name}.so").resolve())
    lines.append(f"build {output_so}: link " + " ".join(objects))
    lines.append(f"default {output_so}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ninja execution
# ---------------------------------------------------------------------------


def _get_num_workers() -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool = False) -> None:
    """Execute a Ninja build.

    Parameters
    ----------
    workdir : Path
        Working directory for the build.
    ninja_file : Path
        Path to the ``build.ninja`` file.
    verbose : bool
        If True, print Ninja output to stdout.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers()
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
