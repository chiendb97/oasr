// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief Dispatch vectorized vs scalar activation kernels based on channel alignment.
 *
 * Uses VecSize=4 for aligned FP16/BF16 channels, otherwise scalar access.
 */
#define OASR_DISPATCH_ACTIVATION_VEC_SIZE(vec_size_var, channels, ...)                \
    do {                                                                               \
        constexpr int _vec4 = 4;                                                       \
        if ((channels) % _vec4 == 0) {                                                 \
            constexpr int vec_size_var = _vec4;                                        \
            __VA_ARGS__                                                                \
        } else {                                                                       \
            constexpr int vec_size_var = 1;                                             \
            __VA_ARGS__                                                                \
        }                                                                              \
    } while (0)
