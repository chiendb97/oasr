// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-stream feature-ring append for the streaming step loop.
//
// This is the host-side half of `InputProcessor._distribute_streaming_features`:
// for every ready stream, decide whether its feature buffer must be grown or
// compacted, then queue the copies and submit them as one batched op.
//
// It lives in C++ because of what it is made of, not because Python is slow at
// arithmetic.  Measured on this repo, the loop's *own* control flow costs
// 1.775 us for sixteen streams while a single `buf[a:b]` costs 1.603 us — the
// cost is the Python->ATen boundary (argument parsing, `THPVariable` wrapping),
// not the interpreter.  Crossing that boundary once per tick instead of ~85
// times is the whole point: the same `at::Tensor::slice` measured 1.246 us from
// Python and 0.422 us from C++ (2.95x), so what moves here is the 0.824 us of
// binding overhead per op, not the ATen dispatch underneath it.

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace oasr {
namespace streaming {

/// Updated per-stream ring state after one tick's append.
///
/// Every vector is indexed by the caller's stream order.  A stream with no new
/// frames is passed through untouched, so the caller can write the whole result
/// back without re-deriving which rows changed.
struct AppendResult {
    std::vector<std::optional<at::Tensor>> buffers;  ///< possibly reallocated
    std::vector<int64_t> frames;                     ///< new ``feature_frames``
    std::vector<int64_t> cursors;                    ///< new ``feature_cursor``
    std::vector<int64_t> base_delta;                 ///< add to ``feature_base``
    int64_t reallocations = 0;                       ///< buffers newly allocated
};

/// Append ``new_lens[i]`` frames of ``feats[i]`` to stream ``i``'s ring buffer.
///
/// Mirrors the Python implementation exactly, including its two subtleties:
///
/// * **Compact only when the append would not otherwise fit.**  Dropping the
///   consumed prefix eagerly reallocated on 91 % of appends at steady state,
///   because a stream consumes about as many frames per tick as it gains.
/// * **A reallocation keeps the outgoing buffer alive through the copy.**  The
///   `srcs` entry holds a reference to the old storage, so the allocator cannot
///   hand it back as a later stream's `new_buf` while this copy is still
///   pending.  In Python that is done by queueing the view before rebinding the
///   attribute; here it is the `at::Tensor` refcount, which is the same
///   guarantee for the same reason.
///
/// The queued pairs are independent — each writes its own buffer's
/// ``[keep_n, keep_n + n_new)``, which no other pair reads — so they are
/// submitted with one `_foreach_copy_`, whose members are unordered relative to
/// each other.
///
/// @param buffers            per-stream buffer, or nullopt before the first append
/// @param frames             per-stream ``feature_frames``
/// @param cursors            per-stream ``feature_cursor``
/// @param feats              ``(B, T, F)`` newly extracted features
/// @param new_lens           per-stream count of valid new frames in ``feats``
/// @param feat_dim           ``F``
/// @param headroom_appends   spare appends a fresh buffer should absorb
/// @param headroom_max       absolute cap on that headroom, in frames
AppendResult appendFeatures(const std::vector<std::optional<at::Tensor>>& buffers,
                            const std::vector<int64_t>& frames,
                            const std::vector<int64_t>& cursors,
                            const at::Tensor& feats,
                            const std::vector<int64_t>& new_lens,
                            int64_t feat_dim,
                            int64_t headroom_appends,
                            int64_t headroom_max);

}  // namespace streaming
}  // namespace oasr
