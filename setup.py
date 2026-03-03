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
        
        # CUDA architectures
        cuda_arch = os.environ.get("CUDA_ARCHITECTURES", "70;75;80;86;89;90")
        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
        
        if os.environ.get("OASR_USE_FLASH_ATTENTION", "0") == "1":
            cmake_args.append("-DUSE_FLASH_ATTENTION=ON")
        else:
            cmake_args.append("-DUSE_FLASH_ATTENTION=OFF")
        
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
    version_file = Path(__file__).parent / "python" / "oasr" / "__init__.py"
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
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    
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
