"""Assemble k2 Fsa lattices from the decoder's flat lattice records.

Record layout (int32 x 8, see include/wfst/decoder.h):
  {src_tok, dst_tok, label, arc_map (graph arc id), score_bits (f32), seg, lane, eps}

Segments: frame t persists as segment t+1; segment 0 holds the initial epsilon closure.
An emitting/final arc at segment s connects a frame s-1 token to a frame s token; an
epsilon arc (eps=1, label 0) connects two frame-s tokens. Token ids are arena-global and
1:1 with (lane, frame, state). States are numbered by (frame, token id) so the FSA is
topologically sorted with the super-final token last, matching k2 lattice conventions
(arc scores = graph + acoustic log-likes; aux word labels attached from the graph image
via the identity arc_map).
"""

import numpy as np
import torch

from oasr.decoder.wfst.graph_image import GraphImage


def build_lattice(records: torch.Tensor, lane: int, img: GraphImage):
    """Returns a k2.Fsa for `lane` (or None if it has no arcs). Requires k2."""
    import k2

    rec = records[records[:, 6] == lane]
    if rec.numel() == 0:
        return None
    rec = rec.numpy()
    # Epsilon closure passes can re-expand unchanged arcs -> collapse duplicates.
    _, uniq = np.unique(rec[:, [0, 1, 3, 7]], axis=0, return_index=True)
    rec = rec[np.sort(uniq)]
    src_tok, dst_tok = rec[:, 0].astype(np.int64), rec[:, 1].astype(np.int64)
    label, arc_map = rec[:, 2], rec[:, 3]
    score = rec[:, 4].view(np.float32)
    seg = rec[:, 5].astype(np.int64)
    eps = rec[:, 7].astype(np.int64)

    # Token frames: dst is always at frame == seg; src at seg-1 (emitting) or seg (eps).
    toks = np.concatenate([src_tok, dst_tok])
    frames = np.concatenate([seg - 1 + eps, seg])
    uniq, inv = np.unique(toks, return_inverse=True)
    tok_frame = np.zeros(len(uniq), dtype=np.int64)
    tok_frame[inv] = frames  # consistent: each token has a single frame

    order = np.lexsort((uniq, tok_frame))  # state id = rank by (frame, token)
    state_of = np.empty(len(uniq), dtype=np.int64)
    state_of[order] = np.arange(len(uniq))
    src_state = state_of[inv[: len(src_tok)]]
    dst_state = state_of[inv[len(src_tok) :]]

    arc_order = np.lexsort((dst_state, src_state))
    arcs = np.empty((len(rec), 4), dtype=np.int32)
    arcs[:, 0] = src_state[arc_order]
    arcs[:, 1] = dst_state[arc_order]
    arcs[:, 2] = label[arc_order]
    arcs[:, 3] = score[arc_order].view(np.int32)
    fsa = k2.Fsa(torch.from_numpy(arcs))

    # aux labels (word ids) from the image's ragged pool via the identity arc_map.
    am = arc_map[arc_order].astype(np.int64)
    starts = img.aux_row_splits[am].astype(np.int64)
    ends = img.aux_row_splits[am + 1].astype(np.int64)
    counts = ends - starts
    pool_idx = np.repeat(starts, counts) + (
        np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    )
    row_splits = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    aux = k2.RaggedTensor(
        k2.ragged.create_ragged_shape2(
            row_splits=torch.from_numpy(row_splits), cached_tot_size=int(counts.sum())
        ),
        torch.from_numpy(img.aux_pool[pool_idx].astype(np.int32)),
    )
    fsa.aux_labels = aux
    fsa.arc_map = torch.from_numpy(arc_map[arc_order].astype(np.int32))
    return fsa
