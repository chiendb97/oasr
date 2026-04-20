#!/usr/bin/env python
"""
OASR - Open Automatic Speech Recognition

Setup script for building and installing the OASR package.
"""

import os
import sys
import subprocess
import sysconfig
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


def get_ext_suffix():
    """Get the extension suffix for the current platform."""
    return sysconfig.get_config_var('EXT_SUFFIX') or '.so'


class CMakeExtension(Extension):
    """A CMake extension module."""
    
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def _find_k2_cmake_dir():
    """Return the directory containing k2Config.cmake.

    Search order:
    1. System-wide install at /usr/local/share/cmake/k2.
    2. pip package in site-packages (fallback for environments without submodule build).
    """
    import glob
    import sysconfig as _sc
    import site

    here = os.path.dirname(os.path.abspath(__file__))

    # 1. System-wide install (e.g. /usr/local/share/cmake/k2).
    system_cmakes = ["/usr/local/share/cmake/k2", "/opt/k2-install/share/cmake/k2"]
    for system_cmake in system_cmakes:
        if os.path.isfile(os.path.join(system_cmake, "k2Config.cmake")):
            return system_cmake

    # 2. pip package layout in site-packages.
    site_dirs = []
    for scheme in ("purelib", "platlib"):
        try:
            site_dirs.append(_sc.get_path(scheme))
        except Exception:
            pass
    try:
        site_dirs.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        site_dirs.append(site.getusersitepackages())
    except AttributeError:
        pass

    for site_dir in dict.fromkeys(d for d in site_dirs if d):  # dedup, preserve order
        k2_pkg_dir = os.path.join(site_dir, "k2")
        if not os.path.isdir(k2_pkg_dir):
            continue
        # Standard k2 pip layout: k2/share/cmake/k2/k2Config.cmake
        for candidate in (
            os.path.join(k2_pkg_dir, "share", "cmake", "k2"),
            os.path.join(k2_pkg_dir, "cmake"),
            k2_pkg_dir,
        ):
            if os.path.isfile(os.path.join(candidate, "k2Config.cmake")):
                return candidate
        hits = glob.glob(
            os.path.join(k2_pkg_dir, "**", "k2Config.cmake"), recursive=True
        )
        if hits:
            return os.path.dirname(hits[0])
    return None


def _find_torch_cmake_prefix():
    """Find torch cmake prefix, even inside pip's isolated build env."""
    try:
        import torch
        return torch.utils.cmake_prefix_path
    except ImportError:
        pass

    # In pip's isolated build env, torch isn't importable directly.
    # Probe the system site-packages for the torch cmake config.
    import sysconfig as _sc
    for scheme in ("purelib", "platlib"):
        site_dir = _sc.get_path(scheme)
        candidate = os.path.join(site_dir, "torch", "share", "cmake")
        if os.path.isdir(candidate):
            return candidate

    try:
        import site
        for site_dir in site.getsitepackages():
            candidate = os.path.join(site_dir, "torch", "share", "cmake")
            if os.path.isdir(candidate):
                return candidate
    except AttributeError:
        pass

    return None


def _detect_cuda_architectures() -> str:
    """Return a semicolon-separated list of CUDA SM versions for the host GPUs.

    Queries torch.cuda to find the actual device capabilities so we never pass
    an architecture that the installed CUDA toolkit does not support (e.g.
    compute_70 was dropped in CUDA 12.x).  Falls back to 80;86;89;90 if CUDA
    is unavailable or the query fails.
    """
    _FALLBACK = "80;86;89;90"
    try:
        import torch
        if not torch.cuda.is_available():
            return _FALLBACK
        archs: set[str] = set()
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            archs.add(f"{major}{minor}")
        return ";".join(sorted(archs)) if archs else _FALLBACK
    except Exception:
        return _FALLBACK


class CMakeBuild(build_ext):
    """Build extension using CMake."""
    
    def build_extension(self, ext: CMakeExtension) -> None:
        # Ensure output directory exists
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        ext_dir.mkdir(parents=True, exist_ok=True)
        
        # Build directory
        build_dir = Path(self.build_temp).absolute()
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
        ]
        
        # Torch cmake prefix (for find_package(Torch))
        torch_prefix = _find_torch_cmake_prefix()
        if torch_prefix:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={torch_prefix}")
        
        # Build type
        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
        
        # CUDA architectures: honour explicit override, otherwise auto-detect
        # from available GPUs via PyTorch.  Fall back to a conservative modern
        # set (80+) that works on CUDA 11.8+ without requiring Volta support.
        cuda_arch = os.environ.get("CUDA_ARCHITECTURES") or _detect_cuda_architectures()
        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
        
        if os.environ.get("OASR_USE_FLASH_ATTENTION", "0") == "1":
            cmake_args.append("-DUSE_FLASH_ATTENTION=ON")
        else:
            cmake_args.append("-DUSE_FLASH_ATTENTION=OFF")

        if os.environ.get("OASR_USE_K2", "0") == "1":
            cmake_args.append("-DOASR_USE_K2=ON")
            # Resolve k2's CMake config dir and pass it explicitly so the
            # pip isolated-env CMake step doesn't need to import k2 itself.
            k2_cmake_dir = _find_k2_cmake_dir()
            if k2_cmake_dir:
                cmake_args.append(f"-Dk2_DIR={k2_cmake_dir}")
            else:
                raise RuntimeError(
                    "OASR_USE_K2=1 requires k2 to be installed. "
                    "Run: pip install k2"
                )
            # K2 source tree for internal headers (k2/csrc/, k2/torch/csrc/).
            # Required by the streaming decoder.
            k2_src = os.environ.get("K2_SOURCE_DIR", "/opt/k2-src")
            if k2_src:
                cmake_args.append(f"-DK2_SOURCE_DIR={k2_src}")
        else:
            cmake_args.append("-DOASR_USE_K2=OFF")
        
        # Build arguments
        build_args = ["--config", build_type]
        
        # Parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            cpu_count = os.cpu_count() or 1
            build_args.extend(["-j", str(cpu_count)])
        
        # Run CMake configure
        print(f"CMake configure: {' '.join(cmake_args)}")
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_dir
        )
        
        # Run CMake build (extension is built directly into ext_dir)
        print(f"CMake build: {' '.join(build_args)}")
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir
        )


def get_version() -> str:
    """Get version from package __init__.py."""
    version_file = Path(__file__).parent / "oasr" / "__init__.py"
    version = "0.1.0"
    
    if version_file.exists():
        with open(version_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    version = line.split("=")[1].strip().strip('"\'')
                    break
    
    return version


def get_long_description() -> str:
    """Get long description from README."""
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


# Requirements
install_requires = [
    "numpy>=1.20.0",
    "apache-tvm-ffi>=0.1.0",
    "jinja2>=3.0",
    "filelock>=3.0",
    "packaging>=21.0",
]

extras_require = {
    "audio": [
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
    ],
    "serving": [
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.0.0",
    ],
    "all": [
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
}


setup(
    name="oasr",
    version=get_version(),
    author="OASR Contributors",
    author_email="",
    description="Open Automatic Speech Recognition - High-performance ASR inference framework",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/oasr",
    license="Apache-2.0",
    
    # Package configuration
    packages=find_packages(include=["oasr", "oasr.*"]),
    
    # C++ extension
    ext_modules=[CMakeExtension("oasr._C", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    # Keep egg-info metadata under build/ instead of polluting the source tree
    options={"egg_info": {"egg_base": "build"}},
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    
    # Entry points
    entry_points={
        "console_scripts": [
            "oasr-server=oasr.serving.server:main",
        ],
    },
    
    # Include package data
    include_package_data=True,
    zip_safe=False,
)
