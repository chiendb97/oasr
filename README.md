# OASR - Open Automatic Speech Recognition

A high-performance open-source inference framework for ASR models built with C++, CUDA, and CUTLASS.

## Supported Models

- **Conformer** - Convolution-augmented Transformer
- **Paraformer** - Parallel Transformer for ASR
- **Branchformer** - Branch-based Transformer

## Features

- High-performance CUDA kernels for ASR inference
- CUTLASS-based optimized GEMM operations
- Python bindings via pybind11
- FastAPI serving support
- Support for various attention mechanisms (MHA, Relative Position, etc.)
- Optimized convolution modules for Conformer-style models

## Architecture

```
oasr/
├── csrc/                    # C++/CUDA source code
│   ├── kernels/             # CUDA kernel implementations
│   ├── layers/              # Layer-level abstractions
│   ├── models/              # Model implementations
│   ├── ops/                 # CUTLASS operations
│   └── utils/               # Utilities (memory, logging, etc.)
├── python/
│   └── oasr/                # Python package
│       ├── _C/              # C++ extension bindings
│       ├── layers/          # Python layer wrappers
│       ├── models/          # Python model wrappers
│       └── serving/         # FastAPI serving
├── tests/                   # Unit tests
├── examples/                # Usage examples
└── third_party/             # Third-party dependencies
```

## Requirements

- CUDA >= 11.8
- cuDNN >= 8.6
- CMake >= 3.18
- Python >= 3.8
- pybind11 >= 2.10
- CUTLASS >= 3.0

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/oasr.git
cd oasr

# Build the C++ library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python package
pip install -e .
```

## Quick Start

```python
import oasr

# Load a Conformer model
model = oasr.models.Conformer.from_pretrained("path/to/model")

# Run inference
audio = oasr.load_audio("audio.wav")
result = model.transcribe(audio)
print(result.text)
```

## License

Apache 2.0
