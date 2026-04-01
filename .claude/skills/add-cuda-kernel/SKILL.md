---
name: add-cuda-kernel
description: Step-by-step tutorial for adding new CUDA kernels to Oasr
---

# Tutorial: Adding a New Kernel to Oasr

This tutorial walks through adding a simple element-wise scale operation to Oasr. We'll implement `scale(x, factor) = x * factor` to demonstrate the complete workflow, with references to real kernels (norm, activation, conv, gemm) throughout.

## Goal

Add a new operation that scales each element of a tensor by a scalar factor:

- Input: tensor `x` and scalar `factor`
- Output: `x * factor` (element-wise)
- Support multiple dtypes (FP16, BF16, FP32)

## Step 1: Define CUDA Kernel in `include/`

Create `include/oasr/scale.cuh`:

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace oasr {

/*!
 * \brief Element-wise scale kernel
 * \tparam T Data type (half, __nv_bfloat16, float)
 * \param input Input tensor
 * \param output Output tensor
 * \param factor Scale factor
 * \param n Number of elements
 */
template <typename T>
__global__ void ScaleKernel(const T* input, T* output, T factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * factor;
  }
}

/*!
 * \brief Launch scale kernel
 * \tparam T Data type
 * \param input Input pointer
 * \param output Output pointer
 * \param factor Scale factor
 * \param n Number of elements
 * \param stream CUDA stream
 */
template <typename T>
cudaError_t ScaleLauncher(const T* input, T* output, T factor, int n,
                          cudaStream_t stream = nullptr) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  ScaleKernel<T><<<blocks, threads, 0, stream>>>(input, output, factor, n);

  return cudaGetLastError();
}

}  // namespace oasr
```

**Key points:**

- Framework-agnostic (no Torch headers)
- Uses raw pointers
- Template-based for dtype flexibility
- Only includes what's needed (cuda_runtime, cuda_fp16, cuda_bf16)
- Namespace follows `oasr::` pattern (or `oasr::<family>::` for larger families, e.g. `oasr::norm::`, `oasr::activation::`)

**Real examples:**

- `include/oasr/activation.cuh` -- `oasr::activation::GLU<T>()`, `oasr::activation::Swish<T>()`
- `include/oasr/norm.cuh` -- `oasr::norm::LayerNorm<T>()`, `oasr::norm::RMSNorm<T>()`
- `include/oasr/conv/conv1d.cuh` -- Depthwise/pointwise conv1d kernels
- `include/oasr/gemm/gemm.cuh` -- CUTLASS GEMM kernels

## Step 2: Create Launcher in `csrc/`

Create `csrc/scale.cu`:

```cpp
#include <oasr/scale.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

void scale_run(TensorView output, TensorView input, double factor) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);

  int n = 1;
  for (int i = 0; i < input.ndim(); ++i) {
    n *= input.size(i);
  }

  cudaStream_t stream = get_stream(input.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    cudaError_t status = ScaleLauncher<c_type>(
      static_cast<const c_type*>(input.data_ptr()),
      static_cast<c_type*>(output.data_ptr()),
      static_cast<c_type>(factor),
      n,
      stream
    );
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "Failed to run ScaleLauncher: " << cudaGetErrorString(status);
    return true;
  });
}
```

**Key points:**

- Includes kernel header from `include/oasr/` and `"tvm_ffi_utils.h"` (TVM-FFI utils only in `csrc/`)
- Uses `TensorView` (alias for `tvm::ffi::TensorView`) as tensor type
- Uses `Optional` (alias for `tvm::ffi::Optional<TensorView>`) for optional tensors
- **Output tensor is the first parameter** in C++ launchers (destination-passing convention)
- Uses `CHECK_INPUT(x)` macro to verify tensor is on CUDA
- Gets CUDA stream via `get_stream(device)`
- Dispatches on dtype with `DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16` (handles FP32/FP16/BF16)
- Converts TensorView to raw pointers with `static_cast<T*>(x.data_ptr())`
- Checks kernel result with `TVM_FFI_ICHECK`
- Add descriptive error messages with `<<` operator

**Available validation macros** (from `csrc/tvm_ffi_utils.h`):

| Macro | Purpose |
|-------|---------|
| `CHECK_INPUT(x)` | Verify tensor is on CUDA |
| `CHECK_DIM(expected, x)` | Verify dimensionality |
| `CHECK_DEVICE(x, y)` | Same-device check |
| `CHECK_LAST_DIM_CONTIGUOUS_INPUT(x)` | Contiguity check |

**Available dispatch macros:**

| Macro | Dtypes |
|-------|--------|
| `DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dtype, c_type, ...)` | FP32, FP16, BF16 |
| `DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, ...)` | Alias for the above |
| `DISPATCH_DLPACK_HALF_DTYPE(dtype, c_type, ...)` | FP16, BF16 only (for GEMM/conv) |

**TVM-FFI Error Handling:**

- `TVM_FFI_ICHECK(condition) << "message"` -- Assert with error message (used inside dispatch macros or when you need a simple assertion)
- `TVM_FFI_THROW(ValueError) << "message"` -- Throw ValueError with custom message (standard runtime error handling)
- `TVM_FFI_THROW(TypeError) << "message"` -- Throw TypeError
- Use `<<` to chain multiple values in the error message
- Errors are properly propagated back to Python

**When to use `TVM_FFI_THROW` vs `TVM_FFI_LOG_AND_THROW`:**

- **`TVM_FFI_THROW`**: Use for normal runtime error handling. This is the standard way to report errors that will be caught and propagated to Python.

  ```cpp
  void scale_run(TensorView output, TensorView input, double factor) {
    if (!input.device().device_type == kDLCUDA) {
      TVM_FFI_THROW(ValueError) << "Input must be a CUDA tensor";
    }
  }
  ```

- **`TVM_FFI_LOG_AND_THROW`**: Use only in cases where:
  1. The function may be called during object construction time (e.g., validation in constructors or setup methods)
  2. The exception may not be caught properly (e.g., during module initialization)
  3. The error condition almost never fails in practice (e.g., internal errors, unsupported dtype combinations in dispatch macros)

  This variant logs the error message before throwing, ensuring visibility even if the exception doesn't propagate correctly.

  ```cpp
  void check_weights_shape(std::string which_weights) const {
    if (which_weights != "gemm1" && which_weights != "gemm2") {
      // Internal error that should never happen - use LOG_AND_THROW
      TVM_FFI_LOG_AND_THROW(InternalError)
          << "Internal error: which_weights = " << which_weights;
    }
  }
  ```

**Real example** (from `csrc/activation.cu`):

```cpp
void glu(TensorView output, TensorView input) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int batch_size = input.size(0);
    unsigned int seq_len = input.size(1);
    unsigned int channels = input.size(2) / 2;

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = activation::GLU<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<c_type*>(output.data_ptr()),
            batch_size, seq_len, channels, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GLU kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
```

## Step 3: Create TVM-FFI Binding in `csrc/`

Create `csrc/scale_jit_binding.cu`:

```cpp
#include "tvm_ffi_utils.h"

// Forward declaration
void scale_run(TensorView output, TensorView input, double factor);

// Export to TVM-FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, scale_run);
```

**Key points:**

- Include `"tvm_ffi_utils.h"` for TVM-FFI macros and type aliases
- Forward declare the launcher function(s)
- Export with `TVM_FFI_DLL_EXPORT_TYPED_FUNC(exported_name, function)` -- the exported name is how Python accesses it

**Real example** (from `csrc/activation_jit_binding.cu`):

```cpp
#include "tvm_ffi_utils.h"

// Forward declarations of launcher functions
void glu(TensorView output, TensorView input);
void swish(TensorView output, TensorView input);

// TVM-FFI symbol exports
TVM_FFI_DLL_EXPORT_TYPED_FUNC(glu, glu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(swish, swish);
```

**Real example** (from `csrc/norm_jit_binding.cu`):

```cpp
#include "tvm_ffi_utils.h"

void layernorm(TensorView output, TensorView input, TensorView weight,
               Optional bias_opt, double eps);
void rmsnorm(TensorView output, TensorView input, TensorView weight,
             Optional bias_opt, double eps);
// ... more forward declarations ...

TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm, layernorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm, rmsnorm);
// ... more exports ...
```

Note: Multiple launcher functions can be exported from a single binding file, as the norm and activation families do.

## Step 4: Create JIT Generator in `oasr/jit/`

Create `oasr/jit/scale.py`:

```python
from .core import gen_jit_spec, JitSpec
from . import env


def gen_scale_module() -> JitSpec:
    """Generate JIT spec for scale kernel."""
    return gen_jit_spec(
        "scale",
        [
            env.OASR_CSRC_DIR / "scale.cu",
            env.OASR_CSRC_DIR / "scale_jit_binding.cu",
        ],
    )
```

**Key points:**

- Import `gen_jit_spec` and `JitSpec` from `.core`, paths from `.env`
- `gen_jit_spec()` auto-detects GPU architecture and sets default NVCC flags
- Source files are the launcher `.cu` + binding `.cu` from `csrc/`
- **NEVER write to package directories** -- JIT cache goes to `~/.cache/oasr/jit/`

**Real examples:**

```python
# oasr/jit/activation.py
def gen_activation_module() -> JitSpec:
    return gen_jit_spec(
        "activation",
        [env.OASR_CSRC_DIR / "activation.cu",
         env.OASR_CSRC_DIR / "activation_jit_binding.cu"],
    )

# oasr/jit/norm.py
def gen_norm_module() -> JitSpec:
    return gen_jit_spec(
        "norm",
        [env.OASR_CSRC_DIR / "norm.cu",
         env.OASR_CSRC_DIR / "norm_jit_binding.cu"],
    )
```

### (Optional) Specifying Supported CUDA Architectures

Oasr uses `CompilationContext` to manage CUDA architecture targets. This is critical because some kernels only work on specific GPU architectures (e.g., Hopper SM90, Blackwell SM100).

#### How CompilationContext Works

**Automatic Detection** (default):
```python
from oasr.compilation_context import CompilationContext

ctx = CompilationContext()
# Automatically detects all GPUs in the system
# For SM90+, adds 'a' suffix (e.g., 9.0a for Hopper)
# Result: ctx.TARGET_CUDA_ARCHS = {(9, '0a'), (10, '0a'), ...}
```

**Manual Override** (via environment variable):
```bash
export OASR_CUDA_ARCH_LIST="8.0 9.0a 10.0a"
# Now only these architectures will be compiled
```

#### Specifying Architectures in Your JIT Module

When creating a JIT module, specify which major SM versions are supported:

```python
from oasr.jit.core import gen_jit_spec
from oasr.jit import current_compilation_context

def gen_my_hopper_only_module():
    """Example: Kernel works on SM90 and later supported architectures."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        # Explicitly list supported SM versions -- no automatic future compatibility
        supported_major_versions=[9, 10, 11, 12]  # SM90, SM100, SM110, SM120
    )

    return gen_jit_spec(
        name="my_hopper_kernel",
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
    )

def gen_my_blackwell_only_module():
    """Example: Kernel only works on SM100 (Blackwell)."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]  # SM100 only
    )

    return gen_jit_spec(
        name="my_blackwell_kernel",
        sources=sources,
        extra_cuda_cflags=nvcc_flags,
    )

def gen_my_universal_module():
    """Example: Kernel works on all architectures (default)."""
    # No need to call get_nvcc_flags_list -- gen_jit_spec auto-detects
    return gen_jit_spec(
        name="my_universal_kernel",
        sources=sources,
    )
```

**What Happens:**
- If user's GPU is SM90 and they call a Hopper-only module -> Compiles and runs
- If user's GPU is SM80 and they call a Hopper-only module -> `RuntimeError: No supported CUDA architectures found for major versions [9, 10, 11, 12]`

#### Common Architecture Specifications

| Supported Versions | Architectures | Use Case |
|-------------------|---------------|----------|
| `None` | All available GPUs | Universal kernels (default) |
| `[9, 10, 11, 12]` | SM90, SM100, SM110, SM120 | Hopper, Blackwell |
| `[10, 11, 12]` | SM100, SM110, SM120 | Blackwell only |
| `[12]` | SM120 | Specific architecture only |
| `[8, 9, 10, 11, 12]` | SM80, SM90, SM100, SM110, SM120 | Ampere, Hopper, Blackwell |

## Step 5: Create Python API in `oasr/`

Create `oasr/scale.py`:

```python
import functools
from typing import Optional

import torch

from .api_logging import oasr_api


@functools.cache
def _get_scale_module():
    """Get or compile scale module (cached)."""
    from oasr.jit.scale import gen_scale_module

    return gen_scale_module().build_and_load()


@oasr_api
def scale(input: torch.Tensor, factor: float,
          out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise scale operation.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor (CUDA).
    factor : float
        Scale factor.
    out : Optional[torch.Tensor]
        Output tensor (if None, allocate new tensor).

    Returns
    -------
    output : torch.Tensor
        Scaled tensor (input * factor).

    Examples
    --------
    >>> import torch
    >>> import oasr
    >>> x = torch.randn(1024, dtype=torch.float16, device="cuda")
    >>> y = oasr.scale(x, 2.0)
    >>> torch.allclose(y, x * 2.0)
    True
    """
    if out is None:
        out = torch.empty_like(input)

    # Call TVM-FFI function (output first in C++ convention)
    _get_scale_module().run(out, input, float(factor))

    return out
```

**Key points:**

- Uses `@functools.cache` to cache the compiled module (compile once per process)
- `@oasr_api` decorator (from `oasr.api_logging`) enables debug logging
- **Destination-passing style**: Output tensor is an optional Python parameter (`out=None`) but **passed first** to the C++ TVM-FFI function
- Import the JIT module lazily inside the cached function to avoid import-time compilation

**Real example** (from `oasr/activation.py`):

```python
@functools.cache
def _get_activation_module():
    from oasr.jit.activation import gen_activation_module
    return gen_activation_module().build_and_load()


@oasr_api
def glu(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Gated Linear Unit activation."""
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device, dtype=input.dtype,
        )
    _get_activation_module().glu(out, input)  # output first!
    return out


@oasr_api
def swish(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Swish (SiLU) activation: x * sigmoid(x)."""
    if out is None:
        out = torch.empty_like(input)
    _get_activation_module().swish(out, input)
    return out
```

### (Advanced) Using `@backend_requirement` and `@supported_compute_capability` Decorators

For kernels with compute capability requirements or multiple backend choices, Oasr provides two decorators (in `oasr.utils`):

#### `@supported_compute_capability` Decorator

Marks a function with its supported CUDA compute capabilities:

```python
from oasr.utils import supported_compute_capability

@supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120])
def _my_check_function(input, output):
    """Supports SM80 (Ampere) through SM120 (Blackwell)."""
    # Validation logic here
    return True
```

#### `@backend_requirement` Decorator

Enforces backend and problem size requirements at runtime. There are three usage patterns:

**Pattern 1: Single Backend (No Backend Choices)**

For kernels with only one implementation:

```python
from oasr.utils import backend_requirement, supported_compute_capability

@supported_compute_capability([80, 86, 89, 90, 100, 103, 110, 120])
def _check_my_kernel(input, output):
    """Validate inputs. Must return True if valid."""
    if input.shape[-1] > 256:
        raise ValueError("Head dimension must be <= 256")
    return True

@backend_requirement(
    backend_checks={},  # Empty dict = no backend parameter
    common_check=_check_my_kernel,
)
def my_kernel(input, output):
    # Kernel implementation
    pass
```

**Pattern 2: Multiple Backends**

For kernels with multiple implementation backends (e.g., CUTLASS, cuDNN):

```python
@supported_compute_capability([80, 86, 89, 90])
def _cutlass_check(q, k, v, backend):
    """CUTLASS backend: Ampere through Hopper."""
    if q.shape[-1] > 256:
        raise ValueError("CUTLASS: head_dim must be <= 256")
    return True

@supported_compute_capability([75, 80, 86, 89, 90, 100])
def _cudnn_check(q, k, v, backend):
    """cuDNN backend: Turing through Blackwell."""
    return True

@backend_requirement(
    backend_checks={
        "cutlass": _cutlass_check,
        "cudnn": _cudnn_check,
    },
    common_check=None,  # Optional: shared validation for all backends
)
def attention(q, k, v, backend="cutlass"):
    if backend == "cutlass":
        # CUTLASS implementation
        pass
    elif backend == "cudnn":
        # cuDNN implementation
        pass
```

**Pattern 3: Auto Backend Selection**

For kernels that can automatically select the best backend:

```python
def _heuristic_func(suitable_backends, q, k, v, backend):
    """Return backends in order of preference."""
    if q.shape[-1] <= 128:
        preferred = ["cutlass", "cudnn"]
    else:
        preferred = ["cudnn", "cutlass"]
    return [b for b in preferred if b in suitable_backends]

@backend_requirement(
    backend_checks={
        "cutlass": _cutlass_check,
        "cudnn": _cudnn_check,
    },
    common_check=_common_validation,
    heuristic_func=_heuristic_func,  # Required when backend="auto" is used
)
def attention(q, k, v, backend="auto"):
    if backend == "auto":
        backend = attention.suitable_auto_backends[0]
    # ... rest of implementation
```

#### Features Added by `@backend_requirement`

The decorator adds these methods to the wrapped function:

```python
# Check if a backend is supported (optionally for a specific CC)
scale.is_backend_supported("cutlass")           # True/False
scale.is_backend_supported("cutlass", cc=90)    # True/False for Hopper

# Check if any backend supports this compute capability
scale.is_compute_capability_supported(90)       # True/False

# Check if a backend exists
scale.has_backend("cutlass")                    # True/False

# Check if there are multiple backend choices
scale.has_backend_choices()                     # True/False
```

#### `skip_check` Keyword Argument

The decorator adds a `skip_check` keyword argument to bypass validation for performance-critical code paths:

```python
# Normal call with validation
result = scale(x, 2.0)

# Skip validation for performance (use with caution!)
result = scale(x, 2.0, skip_check=True)
```

#### Check Function Requirements

Check functions must:
1. Accept the same arguments as the decorated function
2. Return `True` if validation passes
3. Raise `ValueError` with descriptive message if validation fails
4. Be decorated with `@supported_compute_capability` to specify supported architectures

## Step 6: Write Tests in `tests/`

Create tests following the flat `tests/test_<kernel>.py` layout. The `conftest.py` provides: `device` (CUDA, skips if unavailable), `dtype`/`dtype_all` fixtures, `batch_seq_hidden` common shapes, and `get_rtol_atol(dtype)` helper.

Create `tests/test_scale.py`:

```python
import pytest
import torch

import oasr


class TestScale:
    """Tests for oasr.scale() functional API."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("size", [128, 1024, 4096])
    def test_scale_correctness(self, dtype, size):
        """Test scale operation correctness."""
        x = torch.randn(size, dtype=dtype, device="cuda")
        factor = 3.14

        y = oasr.scale(x, factor)

        expected = x * factor
        if dtype == torch.float32:
            rtol, atol = 1e-5, 1e-6
        else:
            rtol, atol = 1e-3, 1e-3

        torch.testing.assert_close(y, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_scale_destination_passing(self, dtype):
        """Test scale with pre-allocated output."""
        x = torch.randn(1024, dtype=dtype, device="cuda")
        out = torch.empty_like(x)
        factor = 2.0

        result = oasr.scale(x, factor, out=out)

        # Should return the same tensor
        assert result.data_ptr() == out.data_ptr()

        expected = x * factor
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_scale_cpu_error(self):
        """Test that CPU tensors raise an error."""
        x = torch.randn(128, dtype=torch.float32)

        with pytest.raises(Exception):
            oasr.scale(x, 2.0)
```

**Key points:**

- Use `pytest.mark.parametrize` for multiple configurations
- Compare against reference implementation
- Set appropriate tolerances per dtype (use `get_rtol_atol()` from conftest)
- Test destination-passing style: verify `result.data_ptr() == out.data_ptr()`
- Test error cases

**Real example** (from `tests/test_activation.py`):

```python
class TestGLU:
    @pytest.mark.parametrize(
        "batch_size,seq_len,channels",
        [(2, 128, 256), (4, 256, 512)],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_glu(self, batch_size, seq_len, channels, dtype):
        x = torch.randn(batch_size, seq_len, 2 * channels, device="cuda", dtype=dtype)
        output = oasr.glu(x)
        expected = F.glu(x, dim=-1).to(dtype)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    def test_glu_destination_passing(self, dtype):
        x = torch.randn(2, 128, 512, device="cuda", dtype=torch.float16)
        out = torch.empty(2, 128, 256, device="cuda", dtype=torch.float16)
        result = oasr.glu(x, out=out)
        assert result.data_ptr() == out.data_ptr()
```

### Testing with Architecture Requirements

When your kernel has architecture requirements, add skip checks:

```python
import pytest
import torch
from oasr.utils import is_sm90a_supported

def test_hopper_kernel():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90a is not supported on this GPU")
    # Test code here
    ...
```

## Step 7: Register in AOT

Register your kernel in `oasr/aot.py` so users with `oasr-jit-cache` can skip JIT compilation.

Edit `oasr/aot.py`:

```python
def gen_all_modules() -> List:
    from oasr.jit.activation import gen_activation_module
    from oasr.jit.norm import gen_norm_module
    from oasr.jit.conv import gen_conv_module, gen_conv2d_module, gen_cudnn_conv2d_module
    from oasr.jit.gemm import gen_gemm_module, gen_bmm_module, gen_group_gemm_module
    from oasr.jit.scale import gen_scale_module  # NEW

    return [
        gen_activation_module(),
        gen_norm_module(),
        gen_conv_module(),
        gen_conv2d_module(),
        gen_cudnn_conv2d_module(),
        gen_gemm_module(),
        gen_bmm_module(),
        gen_group_gemm_module(),
        gen_scale_module(),  # NEW
    ]
```

## Step 8: Export API

Edit `oasr/__init__.py`:

```python
from .scale import scale as scale
```

Add `"scale"` to the `__all__` list.

## Step 9: Run and Test

```bash
# The kernel compiles automatically on first use
pytest tests/test_scale.py -v

# Run a single test
pytest tests/test_scale.py::TestScale::test_scale_correctness -v
```

## Step 10: Add Benchmark

**All new kernels should have benchmarks.** This helps track performance regressions and allows users to compare against other implementations.

Benchmarks live in `benchmarks/` and use the unified routines framework in `benchmarks/routines/`. Each benchmark script delegates to a `run_standalone()` function.

**Simple standalone benchmark** -- create `benchmarks/bench_scale.py`:

```python
#!/usr/bin/env python3
"""Performance benchmarks for Scale kernel."""

import argparse

import torch

import oasr
from benchmarks.routines.bench_utils import (
    BenchResult,
    OutputWriter,
    bench_fn,
    check_close,
)


def run_standalone():
    parser = argparse.ArgumentParser(description="OASR Scale Benchmark")
    parser.add_argument("--target", choices=["oasr", "pytorch", "both"], default="oasr")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    writer = OutputWriter()
    sizes = [1024, 4096, 16384, 65536, 262144]
    dtypes = [torch.float16, torch.bfloat16]

    writer.write_header("Scale Kernel Benchmark")

    for size in sizes:
        for dtype in dtypes:
            x = torch.randn(size, dtype=dtype, device="cuda")
            factor = 2.0

            if args.target in ("oasr", "both"):
                median_ms, std_ms = bench_fn(
                    lambda: oasr.scale(x, factor),
                    dry_run_iters=args.warmup,
                    num_iters=args.iters,
                )
                writer.write_result(BenchResult(
                    routine="scale", subroutine="scale",
                    backend="oasr", shape=f"{size}",
                    dtype=str(dtype), median_ms=median_ms, std_ms=std_ms,
                ))

            if args.target in ("pytorch", "both"):
                median_ms, std_ms = bench_fn(
                    lambda: x * factor,
                    dry_run_iters=args.warmup,
                    num_iters=args.iters,
                )
                writer.write_result(BenchResult(
                    routine="scale", subroutine="scale",
                    backend="pytorch", shape=f"{size}",
                    dtype=str(dtype), median_ms=median_ms, std_ms=std_ms,
                ))

    writer.finalize()


if __name__ == "__main__":
    run_standalone()
```

**Alternative: Using `oasr.testing.bench_gpu_time`** for quick benchmarks:

```python
from oasr.testing import bench_gpu_time

x = torch.randn(4096, dtype=torch.float16, device="cuda")
median_s, std_s = bench_gpu_time(
    oasr.scale, args=(x, 2.0),
    enable_cupti=True, dry_run_iters=10, repeat_iters=100,
)
print(f"Median: {median_s*1e6:.2f} us, Std: {std_s*1e6:.2f} us")
```

**For more complex kernels**, consider:

- Adding comparisons against reference implementations (e.g., PyTorch native, cuBLAS, cuDNN)
- Using the unified benchmarking framework in `benchmarks/oasr_benchmark.py` if applicable
- Testing across different problem sizes and configurations

**Benchmark utilities** (from `benchmarks/routines/bench_utils.py`):

| Function | Purpose |
|----------|---------|
| `bench_fn(fn, ...)` | Time a function, returns `(median_ms, std_ms)` |
| `profile_kernel(name, fn, ...)` | Run with NVTX markers for Nsight Compute |
| `check_close(actual, expected, ...)` | Compare tensors, returns `(passed, max_diff)` |
| `BenchResult(...)` | Structured result dataclass |
| `OutputWriter()` | Manages terminal + CSV output |

-> **For complete benchmarking guide, see [`.claude/skills/benchmark-kernel/SKILL.md`](../benchmark-kernel/SKILL.md)**

## Existing OASR Kernel Families

When adding a new kernel, look at these existing families as references:

| Family | Kernel Header | Launcher | Binding | JIT Generator | Python API |
|--------|--------------|----------|---------|---------------|------------|
| **Activation** | `include/oasr/activation.cuh` | `csrc/activation.cu` | `csrc/activation_jit_binding.cu` | `jit/activation.py` | `activation.py` |
| **Norm** | `include/oasr/norm.cuh` | `csrc/norm.cu` | `csrc/norm_jit_binding.cu` | `jit/norm.py` | `norm.py` |
| **Conv1D** | `include/oasr/conv/conv1d.cuh` | `csrc/conv.cu` | `csrc/conv_jit_binding.cu` | `jit/conv.py` | `conv.py` |
| **Conv2D** | `include/oasr/conv/conv2d.cuh` | `csrc/conv2d.cu` | `csrc/conv2d_jit_binding.cu` | `jit/conv.py` | `conv.py` |
| **GEMM** | `include/oasr/gemm/gemm.cuh` | `csrc/gemm.cu` | `csrc/gemm_jit_binding.cu` | `jit/gemm.py` | `gemm.py` |
| **BMM** | `include/oasr/gemm/bmm.cuh` | `csrc/bmm.cu` | `csrc/bmm_jit_binding.cu` | `jit/gemm.py` | `gemm.py` |
| **Group GEMM** | `include/oasr/gemm/group_gemm.cuh` | `csrc/group_gemm.cu` | `csrc/group_gemm_jit_binding.cu` | `jit/gemm.py` | `gemm.py` |

## Summary of Files Created/Modified

```
include/oasr/scale.cuh                # NEW: CUDA kernel definition
csrc/scale.cu                         # NEW: TVM-FFI launcher
csrc/scale_jit_binding.cu             # NEW: TVM-FFI binding
oasr/jit/scale.py              # NEW: JIT generator
oasr/scale.py                  # NEW: Python API
oasr/__init__.py               # MODIFIED: Export API
oasr/aot.py                    # MODIFIED: Register AOT
test_scale.py                   # NEW: Unit tests
benchmarks/bench_scale.py             # NEW: Benchmark script
```
