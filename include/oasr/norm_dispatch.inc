// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// FlashInfer-style dispatch macros for normalization kernels.
// Dispatches VecSize and block_size based on hidden dimension alignment.

#pragma once

#include <oasr/common/vec_dtypes.h>

/**
 * @brief Dispatch vectorized vs scalar normalization based on alignment.
 *
 * Selects VecSize based on data type (4 for FP16/BF16) when hidden_size
 * and pointers are aligned, otherwise falls back to scalar.
 */
#define OASR_DISPATCH_NORM_VEC_SIZE(T, vec_size_var, hidden_size, ptr1, ptr2, ...)    \
    do {                                                                               \
        constexpr int _vec = oasr::VecTypeTrait<T>::VecSize;                          \
        if ((hidden_size) >= _vec                                                      \
            && oasr::isAligned<T, _vec>(ptr1)                                          \
            && oasr::isAligned<T, _vec>(ptr2)) {                                       \
            constexpr int vec_size_var = _vec;                                         \
            __VA_ARGS__                                                                \
        } else {                                                                       \
            constexpr int vec_size_var = 1;                                             \
            __VA_ARGS__                                                                \
        }                                                                              \
    } while (0)

/**
 * @brief Compute warp-aligned block size for normalization kernels.
 *
 * Returns a block size in [WARP_SIZE, 1024] that is a multiple of WARP_SIZE.
 */
#define OASR_NORM_BLOCK_SIZE(elements_per_thread)                                     \
    (std::min(1024, std::max(32, (((elements_per_thread) + 31) / 32) * 32)))
