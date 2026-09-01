// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "streaming/feature_ring.h"

#include <ATen/Functions.h>

#include <algorithm>
#include <stdexcept>

namespace oasr {
namespace streaming {

AppendResult appendFeatures(const std::vector<std::optional<at::Tensor>>& buffers,
                            const std::vector<int64_t>& frames,
                            const std::vector<int64_t>& cursors,
                            const at::Tensor& feats,
                            const std::vector<int64_t>& new_lens,
                            int64_t feat_dim,
                            int64_t headroom_appends,
                            int64_t headroom_max) {
    const size_t n = buffers.size();
    if (frames.size() != n || cursors.size() != n || new_lens.size() != n) {
        throw std::invalid_argument(
            "appendFeatures: buffers/frames/cursors/new_lens must be the same length");
    }
    if (n > 0 && feats.dim() != 3) {
        throw std::invalid_argument("appendFeatures: feats must be (B, T, F)");
    }
    if (n > 0 && static_cast<size_t>(feats.size(0)) < n) {
        throw std::invalid_argument("appendFeatures: feats has fewer rows than streams");
    }

    AppendResult out;
    out.buffers = buffers;
    out.frames = frames;
    out.cursors = cursors;
    out.base_delta.assign(n, 0);

    // Reserved for the worst case: every stream contributes a keep-copy and an
    // append-copy.  Growing these mid-loop would be the one allocation this
    // function has no excuse for.
    std::vector<at::Tensor> dsts;
    std::vector<at::Tensor> srcs;
    dsts.reserve(2 * n);
    srcs.reserve(2 * n);

    for (size_t i = 0; i < n; ++i) {
        const int64_t n_new = new_lens[i];
        if (n_new <= 0) {
            continue;  // nothing to append; state passes through untouched
        }
        const at::Tensor new_frames =
            feats.select(0, static_cast<int64_t>(i)).slice(0, 0, n_new);

        const std::optional<at::Tensor>& maybe_buf = buffers[i];
        const bool has_buf = maybe_buf.has_value() && maybe_buf->defined();
        const int64_t have = frames[i];
        const int64_t cursor = cursors[i];

        const bool drop_prefix =
            has_buf && cursor > 0 && have + n_new > maybe_buf->size(0);

        int64_t keep_n;
        int64_t src_start;
        int64_t old_cap;
        if (drop_prefix) {
            keep_n = have - cursor;
            src_start = cursor;
            old_cap = keep_n;
        } else {
            keep_n = have;
            src_start = 0;
            old_cap = has_buf ? maybe_buf->size(0) : 0;
        }

        const int64_t needed = keep_n + n_new;
        at::Tensor buf = has_buf ? *maybe_buf : at::Tensor();

        if (!has_buf || drop_prefix || needed > buf.size(0)) {
            const int64_t headroom = std::min(headroom_appends * n_new, headroom_max);
            const int64_t floor = has_buf ? old_cap : 128;
            const int64_t cap = std::max(needed + headroom, floor);
            at::Tensor new_buf = at::zeros({cap, feat_dim}, new_frames.options());
            if (has_buf && keep_n > 0) {
                // Queued while ``buf`` still owns a reference, so the outgoing
                // storage cannot be recycled under a pending copy.
                dsts.push_back(new_buf.slice(0, 0, keep_n));
                srcs.push_back(buf.slice(0, src_start, src_start + keep_n));
            }
            out.buffers[i] = new_buf;
            buf = new_buf;
            ++out.reallocations;
            if (drop_prefix) {
                // The cursor is rebased, so the frames it counted move into the
                // base.  Their sum is the stream's absolute input-frame index,
                // which is what the speech-activity gate reads.
                out.base_delta[i] = cursor;
                out.cursors[i] = 0;
            }
        }

        dsts.push_back(buf.slice(0, keep_n, keep_n + n_new));
        srcs.push_back(new_frames);
        out.frames[i] = keep_n + n_new;
    }

    if (dsts.size() == 1) {
        dsts[0].copy_(srcs[0]);
    } else if (!dsts.empty()) {
        at::_foreach_copy_(dsts, srcs);
    }
    return out;
}

}  // namespace streaming
}  // namespace oasr
