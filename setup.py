#!/usr/bin/env python
"""
OASR - Open Automatic Speech Recognition

Setup script for building and installing the OASR package.
"""

import os
import sys
import subprocess
import shutil
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
        
        # Build type
        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
        
        # CUDA architectures
        cuda_arch = os.environ.get("CUDA_ARCHITECTURES", "70;75;80;86;89;90")
        cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")
        
        # Optional features
        if os.environ.get("OASR_USE_CUTLASS", "0") == "1":
            cmake_args.append("-DUSE_CUTLASS=ON")
        else:
            cmake_args.append("-DUSE_CUTLASS=OFF")
        
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
        
        # Run CMake build
        print(f"CMake build: {' '.join(build_args)}")
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_dir
        )
        
        # Copy the built library to the package directory if needed
        # CMake outputs to source_dir/python/oasr, but setuptools expects it in ext_dir
        ext_suffix = get_ext_suffix()
        source_lib = Path(ext.sourcedir) / "python" / "oasr" / f"_C{ext_suffix}"
        target = ext_dir / f"_C{ext_suffix}"
        
        if source_lib.exists() and source_lib.resolve() != target.resolve():
            print(f"Copying {source_lib} -> {target}")
            shutil.copy2(source_lib, target)


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
