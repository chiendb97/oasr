// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Diagnostics for the split-K / Stream-K workspace cache, exported into the
// ``gemm`` JIT module.
//
// Its own translation unit rather than the per-variant template, which is
// rendered once per tile configuration: exporting one symbol from a file that
// is compiled ~37 times is a duplicate-symbol link error.
//
// Why export it at all: the invariant this cache has to hold is a byte count,
// and a test that infers that from ``cudaMemGetInfo`` cannot tell "cached once"
// from "allocated and freed on every call" — ``cudaMallocAsync`` recycles, so
// both read as flat.  Shipping a bound with a test that cannot see it is how the
// unbounded version passed review in the first place.
//
// The cache state is a function-local static of an inline function, so it is one
// instance per shared object: these speak for every generated variant in this
// module, and not for the separate ``gemm_log_softmax`` module.

#include <oasr/common/workspace_cache.h>

#include "tvm_ffi_utils.h"

/// Number of live (device, stream, pool) keys the cache holds.
int64_t ws_cache_keys() {
    return oasr::cachedWorkspaceKeys();
}

/// Device bytes the cache has handed out and will never free.
int64_t ws_cache_bytes() {
    return oasr::cachedWorkspaceBytes();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ws_cache_keys, ws_cache_keys);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ws_cache_bytes, ws_cache_bytes);
