// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// FlashInfer-style dispatch macros for Conv1D kernels.
// Dispatches VecSize based on channel alignment.

#pragma once

/**
 * @brief Dispatch vectorized vs scalar Conv1D kernels based on channel alignment.
 *
 * VecSize = 4 for FP16/BF16 when channels is divisible by 4 (128-bit loads).
 * Falls back to scalar (VecSize=1) otherwise.
 */
#define OASR_DISPATCH_CONV1D_VEC_SIZE(vec_size_var, channels, max_block_size, ...)   \
    do {                                                                              \
        constexpr int _vec4 = 4;                                                      \
        if ((channels) % _vec4 == 0 && ((channels) / _vec4) <= (max_block_size)) {    \
            constexpr int vec_size_var = _vec4;                                       \
            __VA_ARGS__                                                               \
        } else {                                                                      \
            constexpr int vec_size_var = 1;                                            \
            __VA_ARGS__                                                               \
        }                                                                             \
    } while (0)
