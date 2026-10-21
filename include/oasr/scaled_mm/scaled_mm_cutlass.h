// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Dispatch for the int8 scaled GEMM: runtime activation id -> the epilogue
// instantiation.  The tile config, the K decomposition (data-parallel or
// Stream-K) and the output dtype are fixed by the caller (one rendered
// translation unit per tile config).

#pragma once

#include <oasr/scaled_mm/scaled_mm_cutlass_template.h>

namespace oasr {
namespace scaled_mm {

#define OASR_SCALED_MM_ACTIVATIONS(CASE) \
    CASE(IDENTITY)                       \
    CASE(RELU)                           \
    CASE(SWISH)                          \
    CASE(GELU)                           \
    CASE(GELU_ERF)

// fp16/bf16 output.  ``a_per_token``: ``a_scale`` holds M floats, else one.
template <typename Config, typename ElementD, bool kStreamK>
ScaledMmStatus dispatchScaledMm(int activation, int8_t const* A, int8_t const* B,
                                float const* a_scale, bool a_per_token, float const* b_scale,
                                ElementD const* bias, ElementD* D, int M, int N, int K, int64_t lda,
                                int64_t ldb, int64_t ldd, cudaStream_t stream) {
    using ThreadMap = typename ScaledMmTileMap<Config>::type;
#define OASR_SCALED_MM_CASE(ACT)                                                          \
    case static_cast<int>(ActivationType::ACT): {                                        \
        using Epilogue = ScaledMmEpilogue<ElementD, ThreadMap, ActivationType::ACT>;     \
        return CutlassScaledMmGemm<Config, typename Epilogue::EVTD, kStreamK>::run(      \
            A, B, Epilogue::make_args(a_scale, a_per_token, b_scale, bias, D, ldd), M, N, \
            K, lda, ldb, ldd, stream);                                                    \
    }
    switch (activation) {
        OASR_SCALED_MM_ACTIVATIONS(OASR_SCALED_MM_CASE)
        default:
            return ScaledMmStatus::CANNOT_IMPLEMENT;
    }
#undef OASR_SCALED_MM_CASE
}

// int8 output for a consumer with a static input scale (``out_mul`` is its
// per-channel quantize multiplier); ``ElementT`` is the dtype the fp output
// would have had, which the activation output is rounded to first.
template <typename Config, typename ElementT>
ScaledMmStatus dispatchScaledMmQuant(int activation, int8_t const* A, int8_t const* B,
                                     float const* a_scale, bool a_per_token,
                                     float const* b_scale, ElementT const* bias,
                                     float const* out_mul, int8_t* Q, int M, int N, int K,
                                     int64_t lda, int64_t ldb, int64_t ldq, cudaStream_t stream) {
    using ThreadMap = typename ScaledMmTileMap<Config>::type;
#define OASR_SCALED_MM_QUANT_CASE(ACT)                                                          \
    case static_cast<int>(ActivationType::ACT): {                                              \
        using Epilogue = ScaledMmQuantEpilogue<ElementT, ThreadMap, ActivationType::ACT>;      \
        return CutlassScaledMmGemm<Config, typename Epilogue::EVTD, /*kStreamK=*/false>::run(  \
            A, B, Epilogue::make_args(a_scale, a_per_token, b_scale, bias, out_mul, Q, ldq), M, \
            N, K, lda, ldb, ldq, stream);                                                       \
    }
    switch (activation) {
        OASR_SCALED_MM_ACTIVATIONS(OASR_SCALED_MM_QUANT_CASE)
        default:
            return ScaledMmStatus::CANNOT_IMPLEMENT;
    }
#undef OASR_SCALED_MM_QUANT_CASE
}

#undef OASR_SCALED_MM_ACTIVATIONS

}  // namespace scaled_mm
}  // namespace oasr
