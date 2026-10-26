// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The TVM-FFI side of the scaled GEMM launchers, shared by both kernel
// generations' rendered translation units (``scaled_mm_cutlass_template.cu.jinja``
// for the CUTLASS 2.x ``mma.sync`` lane, ``scaled_mm_cutlass_template_sm90.cu.jinja``
// for the native sm90 / sm100 lanes): one statement of the operand contract.

#pragma once

#include <cutlass/numeric_types.h>

#include "tvm_ffi_utils.h"

namespace oasr {
namespace scaled_mm {
namespace launcher {

// The DLPack dtype of the 8-bit operands (and of a quantizing launch's output).
template <bool kFp8>
inline bool isOperandType(DLDataType t) {
    return kFp8 ? (t.code == kDLFloat8_e4m3fn && t.bits == 8) : (t.code == kDLInt && t.bits == 8);
}

template <bool kFp8>
constexpr const char* operandName() {
    return kFp8 ? "float8_e4m3fn" : "int8";
}

template <typename T>
struct ToCutlassType {
    using type = T;
};

template <>
struct ToCutlassType<half> {
    using type = cutlass::half_t;
};

template <>
struct ToCutlassType<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};

struct Problem {
    int M, N, K;
    bool a_per_token;
};

// The operand checks every launcher shares.
template <bool kFp8>
inline Problem checkOperands(TensorView output, TensorView A, TensorView B, TensorView a_scale,
                             TensorView b_scale) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(output);
    CHECK_INPUT(a_scale);
    CHECK_INPUT(b_scale);
    CHECK_CONTIGUOUS_INPUT(A);
    CHECK_CONTIGUOUS_INPUT(B);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(a_scale);
    CHECK_CONTIGUOUS_INPUT(b_scale);
    CHECK_DIM(2, B);
    TVM_FFI_ICHECK(isOperandType<kFp8>(A.dtype()) && isOperandType<kFp8>(B.dtype()))
        << "scaled_mm: A and B must be " << operandName<kFp8>();
    TVM_FFI_ICHECK(a_scale.dtype().code == kDLFloat && a_scale.dtype().bits == 32 &&
                   b_scale.dtype().code == kDLFloat && b_scale.dtype().bits == 32)
        << "scaled_mm: a_scale and b_scale must be float32";

    // Flatten leading dimensions here to avoid framework-side reshapes.
    const int K = static_cast<int>(A.size(A.ndim() - 1));
    const int M = static_cast<int>(FLATTENED_ROWS(A));
    const int N = static_cast<int>(B.size(0));
    TVM_FFI_ICHECK(B.size(1) == K)
        << "scaled_mm: B is [N, K]; got K=" << B.size(1) << " for an A of K=" << K;
    TVM_FFI_ICHECK(K % 16 == 0 && N % 8 == 0)
        << "scaled_mm needs K % 16 == 0 (16-byte operand loads) and N % 8 == 0 (16-byte output "
           "stores), got N="
        << N << " K=" << K;
    TVM_FFI_ICHECK(a_scale.numel() == M || a_scale.numel() == 1)
        << "scaled_mm: a_scale needs one float per row (" << M << ") or one per tensor (1), got "
        << a_scale.numel();
    TVM_FFI_ICHECK(b_scale.numel() == N)
        << "scaled_mm: b_scale needs one float per output channel (" << N << "), got "
        << b_scale.numel();
    TVM_FFI_ICHECK(output.numel() == static_cast<int64_t>(M) * N)
        << "scaled_mm: output must hold M*N elements";
    return Problem{M, N, K, a_scale.numel() == M};
}

// The bias pointer (or null) after checking it holds N values of ``expected``.
template <typename CutlassType>
inline const CutlassType* biasPointer(Optional bias_opt, int N, DLDataType expected,
                                      const char* who) {
    if (!bias_opt.has_value()) {
        return nullptr;
    }
    TensorView bias = bias_opt.value();
    CHECK_INPUT(bias);
    TVM_FFI_ICHECK(bias.numel() == N) << who << ": bias needs N elements";
    TVM_FFI_ICHECK(bias.dtype().code == expected.code && bias.dtype().bits == expected.bits)
        << who << ": bias must have the output (round) dtype";
    return reinterpret_cast<const CutlassType*>(bias.data_ptr());
}

// The checks of a quantizing launcher's output and multiplier.
template <bool kFp8>
inline void checkQuantOutput(TensorView output, TensorView out_mul, int N, int64_t round_dtype) {
    TVM_FFI_ICHECK(isOperandType<kFp8>(output.dtype()))
        << "scaled_mm_quant: output must be " << operandName<kFp8>();
    CHECK_INPUT(out_mul);
    CHECK_CONTIGUOUS_INPUT(out_mul);
    TVM_FFI_ICHECK(out_mul.dtype().code == kDLFloat && out_mul.dtype().bits == 32 &&
                   out_mul.numel() == N)
        << "scaled_mm_quant: out_mul needs N=" << N << " float32 values, got " << out_mul.numel();
    TVM_FFI_ICHECK(round_dtype == 0 || round_dtype == 1)
        << "scaled_mm_quant: round_dtype must be 0 (fp16) or 1 (bf16), got " << round_dtype;
}

// ``round_dtype`` code -> DLPack dtype (0 = fp16, 1 = bf16).
inline DLDataType roundDLType(int64_t round_dtype) {
    DLDataType t;
    t.code = round_dtype == 0 ? kDLFloat : kDLBfloat;
    t.bits = 16;
    t.lanes = 1;
    return t;
}

}  // namespace launcher
}  // namespace scaled_mm
}  // namespace oasr
