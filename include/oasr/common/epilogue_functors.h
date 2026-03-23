// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Shared epilogue functor wrappers for CUTLASS kernels (GEMM, Conv2d, etc.)
// Each functor maps (ElementCD, ElementCompute) -> a CUTLASS epilogue op type.

#pragma once

// Suppress warnings from CUTLASS headers
#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/numeric_types.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

namespace oasr {

template <int Alignment, typename ElementCD, typename ElementCompute>
struct EpilogueIdentity {
    using Op = cutlass::epilogue::thread::LinearCombination<
        ElementCD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementCD, typename ElementCompute>
struct EpilogueRelu {
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementCD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementCD, typename ElementCompute>
struct EpilogueGelu {
    using Op = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementCD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementCD, typename ElementCompute>
struct EpilogueSwish {
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementCD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

}  // namespace oasr
