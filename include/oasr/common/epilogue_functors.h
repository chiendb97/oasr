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

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
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
        cutlass::epilogue::thread::LinearCombination<ElementD, Alignment, ElementCompute,
                                                     ElementCompute,
                                                     cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::IDENTITY, Alignment, ElementD, ElementCompute, ElementC> {
    using type =
        cutlass::epilogue::thread::LinearCombination<ElementD, Alignment, ElementCompute,
                                                     ElementCompute,
                                                     cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::RELU, Alignment, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::thread::LinearCombinationRelu<
        ElementD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::GELU, Alignment, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::thread::LinearCombinationGeneric<
        cutlass::epilogue::thread::GELU_taylor, ElementD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
        true>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::GELU_ERF, Alignment, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::SWISH, Alignment, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default>;
};

template <int Alignment, typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOp<ActivationType::TANH, Alignment, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::thread::LinearCombinationGeneric<
        cutlass::epilogue::thread::Tanh, ElementD, Alignment, ElementCompute, ElementCompute,
        cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
        true>;
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
    using type =
        cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::GELU_taylor, ElementD,
                                                 ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOpSm90<oasr::ActivationType::GELU_ERF, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::GELU, ElementD,
                                                          ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOpSm90<oasr::ActivationType::SWISH, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::SiLu, ElementD,
                                                          ElementCompute, ElementC, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementC>
struct FusionEpilogueOpSm90<oasr::ActivationType::TANH, ElementD, ElementCompute, ElementC> {
    using type = cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::Tanh, ElementD,
                                                          ElementCompute, ElementC, ElementCompute>;
};

//==============================================================================
// FusionEpilogueOpSm90PerColBias -- alpha * acc + bias[n], then the activation
//
// The CUTLASS 2.x path spells a bias as the C operand with a zero M-stride, so
// one length-N row broadcasts over every row of D.  A 3.x epilogue cannot say
// that through C: its C operand is a TMA load, and TMA has no zero-stride mode.
// The 3.x spelling is a fusion input instead, which is also nullptr-safe -- the
// bias leaf takes a null_default, so a null pointer contributes a literal 0 and
// one instantiation serves both the biased and the unbiased GEMM.
//==============================================================================

template <ActivationType fusion_op, typename ElementD, typename ElementCompute,
          typename ElementBias = ElementD>
struct FusionEpilogueOpSm90PerColBias {
    // Default: identity, i.e. no elementwise node above the bias add.
    using type = cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementCompute, ElementBias,
                                                              void, ElementCompute>;
};

template <typename ElementD, typename ElementCompute, typename ElementBias>
struct FusionEpilogueOpSm90PerColBias<oasr::ActivationType::IDENTITY, ElementD, ElementCompute,
                                      ElementBias> {
    using type = cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementCompute, ElementBias,
                                                              void, ElementCompute>;
};

#define OASR_SM90_PER_COL_BIAS_ELT_ACT(ACTIVATION, FN)                                            \
    template <typename ElementD, typename ElementCompute, typename ElementBias>                   \
    struct FusionEpilogueOpSm90PerColBias<oasr::ActivationType::ACTIVATION, ElementD,             \
                                          ElementCompute, ElementBias> {                          \
        using type = cutlass::epilogue::fusion::LinCombPerColBiasEltAct<                          \
            FN, ElementD, ElementCompute, ElementBias, void, ElementCompute>;                     \
    }

OASR_SM90_PER_COL_BIAS_ELT_ACT(RELU, cutlass::epilogue::thread::ReLu);
OASR_SM90_PER_COL_BIAS_ELT_ACT(GELU, cutlass::epilogue::thread::GELU_taylor);
OASR_SM90_PER_COL_BIAS_ELT_ACT(GELU_ERF, cutlass::epilogue::thread::GELU);
OASR_SM90_PER_COL_BIAS_ELT_ACT(SWISH, cutlass::epilogue::thread::SiLu);
OASR_SM90_PER_COL_BIAS_ELT_ACT(TANH, cutlass::epilogue::thread::Tanh);

#undef OASR_SM90_PER_COL_BIAS_ELT_ACT

}  // namespace oasr
