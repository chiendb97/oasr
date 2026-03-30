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

// CUTLASS 3.x fusion operations
#include <cutlass/epilogue/fusion/operations.hpp>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/types.h>

namespace oasr {

//==============================================================================
// FusionEpilogueOp -- maps OASR EpilogueFunctor to CUTLASS 2.x fusion operation
//==============================================================================
template <ActivationType fusion_op, int Alignment, typename ElementD, typename ElementCompute,
          typename ElementC = ElementD>
struct FusionEpilogueOp {
    using type =
        cutlass::epilogue::thread::LinearCombination<ElementD, Alignment, ElementC, ElementCompute,
                                                     cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::IDENTITY, Alignment, ElementD, ElementCompute, ElementC> {
    using type =
        cutlass::epilogue::thread::LinearCombination<ElementD, Alignment, ElementC, ElementCompute,
                                                     cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::RELU, Alignment, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementD, Alignment, ElementC, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::GELU, Alignment, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementD, Alignment, ElementC, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::SWISH, Alignment, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementD, Alignment, ElementC, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

//==============================================================================
// FusionEpilogueOpSm90 -- maps OASR EpilogueFunctor to CUTLASS 3.x fusion operation
//==============================================================================

template <ActivationType fusion_op, typename ElementD, typename ElementCompute,
          typename ElementC = ElementD>
struct FusionEpilogueOpSm90 {
    // Default: identity (linear combination)
    using type = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC,
                                                              ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOpSm90<oasr::ActivationType::IDENTITY, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC,
                                                              ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOpSm90<oasr::ActivationType::RELU, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::ReLu, ElementD,
                                                          ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOpSm90<oasr::ActivationType::GELU, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::GELU, ElementD,
                                                          ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOpSm90<oasr::ActivationType::SWISH, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::SiLu, ElementD,
                                                          ElementCompute, ElementC, ElementCompute>;
};

}  // namespace oasr
