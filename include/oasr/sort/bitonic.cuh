// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Reusable bitonic sort / top-k primitives for register-resident arrays.
//
// All primitives are __forceinline__ device functions that operate on
// register arrays (passed as `T*` whose indexing is fully constexpr after
// inlining). The N/K template parameters drive complete loop unrolling, so
// the compiler keeps the arrays in registers and emits straight-line
// compare-swap sequences with no local-memory traffic.
//
// Usage:
//   float vals[K];
//   int32_t idxs[K];
//   // ... fill ...
//   oasr::sort::bitonic_sort<K, /*Ascending=*/false>(vals, idxs);   // descending
//   oasr::sort::bitonic_topk_merge<K>(vals, idxs, other_vals, other_idxs);

#pragma once

#include <cuda_runtime.h>

#include <oasr/common/math.h>

namespace oasr {
namespace sort {

// ---------------------------------------------------------------------------
// compare_and_swap
// ---------------------------------------------------------------------------
//
// Branchless compare-and-swap. Ascending=true sorts low→high; false sorts
// high→low. The (key, aux) overload moves an auxiliary register (typically
// an int32 index) alongside the key.
template <bool Ascending>
__device__ __forceinline__ void compare_and_swap(float& a, float& b) {
    float lo = fminf(a, b);
    float hi = fmaxf(a, b);
    if constexpr (Ascending) {
        a = lo;
        b = hi;
    } else {
        a = hi;
        b = lo;
    }
}

template <bool Ascending, typename Aux>
__device__ __forceinline__ void compare_and_swap(float& a, float& b, Aux& xa, Aux& xb) {
    if constexpr (Ascending) {
        if (a > b) {
            swap(a, b);
            swap(xa, xb);
        }
    } else {
        if (a < b) {
            swap(a, b);
            swap(xa, xb);
        }
    }
}

// ---------------------------------------------------------------------------
// bitonic_merge
// ---------------------------------------------------------------------------
//
// Merge a bitonic sequence of length N into a sorted sequence of length N.
// Implemented recursively: compare arr[i] with arr[i + N/2] for i in [0, N/2),
// then merge each half. Fully unrolls when N is a compile-time constant.
template <int N, bool Ascending>
__device__ __forceinline__ void bitonic_merge(float* arr) {
    if constexpr (N > 1) {
        constexpr int kHalf = N / 2;
#pragma unroll
        for (int i = 0; i < kHalf; ++i) {
            compare_and_swap<Ascending>(arr[i], arr[i + kHalf]);
        }
        bitonic_merge<kHalf, Ascending>(arr);
        bitonic_merge<kHalf, Ascending>(arr + kHalf);
    }
}

template <int N, bool Ascending, typename Aux>
__device__ __forceinline__ void bitonic_merge(float* arr, Aux* aux) {
    if constexpr (N > 1) {
        constexpr int kHalf = N / 2;
#pragma unroll
        for (int i = 0; i < kHalf; ++i) {
            compare_and_swap<Ascending>(arr[i], arr[i + kHalf], aux[i], aux[i + kHalf]);
        }
        bitonic_merge<kHalf, Ascending>(arr, aux);
        bitonic_merge<kHalf, Ascending>(arr + kHalf, aux + kHalf);
    }
}

// ---------------------------------------------------------------------------
// bitonic_sort
// ---------------------------------------------------------------------------
//
// Sort N elements (N must be a power of two) using a bitonic sort network.
// Sorts the first half ascending, the second half descending, then merges.
template <int N, bool Ascending>
__device__ __forceinline__ void bitonic_sort(float* arr) {
    static_assert((N & (N - 1)) == 0, "N must be a power of two");
    if constexpr (N > 1) {
        constexpr int kHalf = N / 2;
        bitonic_sort<kHalf, true>(arr);
        bitonic_sort<kHalf, false>(arr + kHalf);
        bitonic_merge<N, Ascending>(arr);
    }
}

template <int N, bool Ascending, typename Aux>
__device__ __forceinline__ void bitonic_sort(float* arr, Aux* aux) {
    static_assert((N & (N - 1)) == 0, "N must be a power of two");
    if constexpr (N > 1) {
        constexpr int kHalf = N / 2;
        bitonic_sort<kHalf, true>(arr, aux);
        bitonic_sort<kHalf, false>(arr + kHalf, aux + kHalf);
        bitonic_merge<N, Ascending>(arr, aux);
    }
}

// ---------------------------------------------------------------------------
// bitonic_topk_merge
// ---------------------------------------------------------------------------
//
// Merge two sorted top-K arrays into one. Both inputs must be sorted in the
// same direction (descending for Ascending=false). After the merge, `arr0`
// holds the top-K of (arr0 ∪ arr1), again sorted in the same direction.
//
// Trick: pair arr0[i] with arr1[K-1-i] and keep the better one in arr0. The
// resulting arr0 is bitonic, so a single bitonic_merge re-sorts it.
template <int K, bool Ascending>
__device__ __forceinline__ void bitonic_topk_merge(float* arr0, const float* arr1) {
#pragma unroll
    for (int i = 0; i < K; ++i) {
        float other = arr1[K - 1 - i];
        bool other_wins = Ascending ? (other < arr0[i]) : (other > arr0[i]);
        if (other_wins) {
            arr0[i] = other;
        }
    }
    bitonic_merge<K, Ascending>(arr0);
}

template <int K, bool Ascending, typename Aux>
__device__ __forceinline__ void bitonic_topk_merge(float* arr0, Aux* aux0, const float* arr1,
                                                   const Aux* aux1) {
#pragma unroll
    for (int i = 0; i < K; ++i) {
        float other = arr1[K - 1 - i];
        Aux other_aux = aux1[K - 1 - i];
        bool other_wins = Ascending ? (other < arr0[i]) : (other > arr0[i]);
        if (other_wins) {
            arr0[i] = other;
            aux0[i] = other_aux;
        }
    }
    bitonic_merge<K, Ascending>(arr0, aux0);
}

}  // namespace sort
}  // namespace oasr
