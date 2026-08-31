# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Sequence-packing layout for the Conformer encoder.

Packing concatenates several utterances (post-subsampling hidden states) into
one gapless packed row ``(1, T_total, D)`` so the expensive 16-layer encoder
runs once with attention restricted to same-utterance tokens (``cu_seqlens``)
instead of padding every utterance to the batch max length.

Two time-mixing operators need per-segment isolation:

* **Attention** consumes the *gapless* packed row + ``cu_seqlens`` — each
  segment attends only to ``[cu_seqlens[s], cu_seqlens[s+1])``.
* **Depthwise conv** (kernel ``K``, non-causal ``padding=(K-1)//2``) mixes
  ``(K-1)//2`` frames each side, so a gapless row would leak across segment
  boundaries.  We scatter the post-GLU activations into a *gapped* layout with
  ``(K-1)//2`` zero frames between segments, run one depthwise conv, then
  gather the valid frames back.  With the gap frames zeroed at the conv input,
  every segment sees clean conv-internal-zero padding at its boundaries —
  making the packed result **bit-exact to single-utterance (B=1) inference**.

All index tensors are built once per packed batch and reused across all layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PackedLayout:
    """Precomputed index tensors describing one packed encoder forward.

    Attributes
    ----------
    num_segs : int
        Number of original utterances packed into this row.
    total_tokens : int
        Summed post-subsampling length ``sum(seg_lengths)`` (host int).
    seg_lengths : Tensor
        ``(S,)`` int32 per-segment post-subsampling lengths.
    cu_seqlens : Tensor
        ``(S+1,)`` int32 prefix sum of ``seg_lengths`` — segment boundaries in
        the gapless packed row.  Feeds the varlen attention kernel.
    max_seg_len : int
        ``max(seg_lengths)`` (host int) — sizes the per-segment pos-emb slice.
    pack_src_idx : Tensor
        ``(total_tokens,)`` int64.  Flat ``b * Tp + t`` source index of each
        valid frame in the padded ``(B, Tp, D)`` embed output, ordered
        ``(segment, frame)``.  ``packed = embed.reshape(B*Tp, D)[pack_src_idx]``
        and the inverse ``index_copy_`` unpacks.
    conv_gather_idx : Tensor
        ``(total_tokens,)`` int64.  Gapped position of each gapless token
        (``token + conv_half * segment_id``); scatters into / gathers out of
        the gapped depthwise-conv buffer (non-causal conv).
    gapped_len : int
        Length of the gapped depthwise-conv buffer (host int, non-causal conv).
    conv_batched_idx : Tensor
        ``(total_tokens,)`` int64.  Flat ``segment * max_seg_len + local_pos``
        position of each gapless token in a ``(num_segs, max_seg_len)`` grid;
        scatters into / gathers out of the batched-per-segment conv buffer
        (causal conv, which has no right-context contamination).
    seg_valid_mask : Tensor
        ``(num_segs, max_seg_len)`` bool — valid (non-pad) frames of the
        batched-per-segment conv grid.
    bias_offsets : Tensor or None
        ``(num_segs+1,)`` int32 prefix sum of ``H * T_s * T_s`` — start of each
        segment's block in the packed block-diagonal varlen attention bias.
        ``None`` when ``num_heads`` was not provided (conv-only layouts).
    bias_gather_idx : Tensor or None
        ``(sum H*T_s*T_s,)`` int64.  Source index into a batched
        ``(num_segs, H, max_seg_len, max_seg_len)`` rel-pos matrix that gathers
        each segment's valid ``(H, T_s, T_s)`` block into the packed bias.
        ``None`` when ``num_heads`` was not provided (conv-only layouts).
    src_rows : int
        ``B * Tp`` — row count of the flattened embed output (unpack reshape).
    padded_t : int
        ``Tp`` — padded post-subsampling length of the embed output.
    """

    num_segs: int
    total_tokens: int
    seg_lengths: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seg_len: int
    pack_src_idx: torch.Tensor
    conv_gather_idx: torch.Tensor
    gapped_len: int
    conv_batched_idx: torch.Tensor
    seg_valid_mask: torch.Tensor
    bias_offsets: "torch.Tensor | None"
    bias_gather_idx: "torch.Tensor | None"
    src_rows: int
    padded_t: int


def build_packed_layout(
    valid_mask: torch.Tensor,
    conv_kernel: int,
    num_heads: "int | None" = None,
) -> PackedLayout:
    """Build a :class:`PackedLayout` from a padding mask.

    Parameters
    ----------
    valid_mask : Tensor
        ``(B, Tp)`` bool — ``True`` where a post-subsampling frame is valid
        (not padding).  This is ``encoder_masks.squeeze(1)``.
    conv_kernel : int
        Depthwise-conv kernel size (non-causal); the per-segment gap is
        ``(conv_kernel - 1) // 2`` zero frames.
    num_heads : int, optional
        Number of attention heads.  When given, the packed block-diagonal
        varlen attention bias index tensors (``bias_offsets`` /
        ``bias_gather_idx``) are built; pass ``None`` for conv-only layouts
        (e.g. the conv-module unit test) that never run attention.

    Notes
    -----
    Performs **one** D2H sync: the per-segment lengths are pulled once
    (``total``/``max_seg`` derived from them) so the block-diagonal bias index
    build needs no further host round-trip.  The layout is built once and
    reused across all encoder layers.
    """
    assert valid_mask.dtype == torch.bool and valid_mask.dim() == 2
    device = valid_mask.device
    B, Tp = valid_mask.shape
    conv_half = (conv_kernel - 1) // 2

    seg_lengths = valid_mask.sum(dim=1).to(torch.int32)  # (B,)
    seg_lengths_i64 = seg_lengths.to(torch.int64)
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seg_lengths, dim=0).to(torch.int32)

    # Single D2H sync: pull the per-segment lengths once (the block-diagonal
    # bias index build needs them host-side) and derive total / max_seg.
    if B == 0:
        seg_list: "list[int]" = []
        total, max_seg = 0, 0
    else:
        seg_list = seg_lengths.tolist()
        total, max_seg = sum(seg_list), max(seg_list)

    # Flat source index (b * Tp + t) of every valid frame, ordered (b, t).
    flat_src = torch.arange(B, device=device, dtype=torch.int64).unsqueeze(1) * Tp + torch.arange(
        Tp, device=device, dtype=torch.int64
    ).unsqueeze(
        0
    )  # (B, Tp)
    pack_src_idx = flat_src[valid_mask]  # (total,)

    arange_total = torch.arange(total, device=device, dtype=torch.int64)
    seg_id = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64), seg_lengths_i64
    )  # (total,)

    # Non-causal: gapped position = token + conv_half * segment_id.
    conv_gather_idx = arange_total + conv_half * seg_id
    gapped_len = total + conv_half * (B - 1) if B > 1 else total

    # Causal: flat (segment, local_pos) position in a (B, max_seg) grid.
    seg_start = cu_seqlens.to(torch.int64)[seg_id]  # (total,)
    local_pos = arange_total - seg_start
    conv_batched_idx = seg_id * max_seg + local_pos
    seg_valid_mask = torch.arange(max_seg, device=device, dtype=torch.int64).unsqueeze(
        0
    ) < seg_lengths_i64.unsqueeze(
        1
    )  # (B, max_seg)

    # Varlen packed block-diagonal bias index tensors: gather each segment's
    # valid (H, T_s, T_s) block out of a batched (B, H, max_seg, max_seg)
    # rel-pos matrix into the flat packed bias the cute varlen kernel reads.
    # Built from the host ``seg_list`` already synced above (no extra round-trip).
    bias_offsets = None
    bias_gather_idx = None
    if num_heads is not None and B > 0:
        H = int(num_heads)
        block_sizes = torch.tensor(
            [H * t * t for t in seg_list],
            dtype=torch.int32,
            device=device,
        )
        bias_offsets = torch.zeros(B + 1, dtype=torch.int32, device=device)
        bias_offsets[1:] = torch.cumsum(block_sizes, dim=0).to(torch.int32)
        h_ar = torch.arange(H, device=device, dtype=torch.int64)
        idx_chunks = []
        for s, t in enumerate(seg_list):
            if t == 0:
                continue
            ar = torch.arange(t, device=device, dtype=torch.int64)
            src = ((s * H + h_ar).view(H, 1, 1) * max_seg + ar.view(1, t, 1)) * max_seg + ar.view(
                1, 1, t
            )  # (H, t, t)
            idx_chunks.append(src.reshape(-1))
        bias_gather_idx = (
            torch.cat(idx_chunks)
            if idx_chunks
            else torch.zeros(0, dtype=torch.int64, device=device)
        )

    return PackedLayout(
        num_segs=B,
        total_tokens=total,
        seg_lengths=seg_lengths,
        cu_seqlens=cu_seqlens,
        max_seg_len=max_seg,
        pack_src_idx=pack_src_idx,
        conv_gather_idx=conv_gather_idx,
        gapped_len=gapped_len,
        conv_batched_idx=conv_batched_idx,
        seg_valid_mask=seg_valid_mask,
        bias_offsets=bias_offsets,
        bias_gather_idx=bias_gather_idx,
        src_rows=B * Tp,
        padded_t=Tp,
    )


def build_packed_layout_device(
    seg_lengths: torch.Tensor,
    padded_t: int,
    conv_kernel: int,
    num_heads: "int | None" = None,
    *,
    total_capacity: "int | None" = None,
    max_seg_capacity: "int | None" = None,
    bias_capacity: "int | None" = None,
) -> PackedLayout:
    """GPU-resident, fixed-shape twin of :func:`build_packed_layout`.

    Same layout, built without a single host round-trip and without a single
    data-dependent shape, so the packed forward becomes CUDA-graph capturable.
    :func:`build_packed_layout` cannot be captured, and measurement showed the
    blocker is not one thing but four, all of them in the layout build (the
    16-layer body captures unchanged):

    ==============================================  ===============================
    construct in :func:`build_packed_layout`         replaced here by
    ==============================================  ===============================
    ``seg_lengths.tolist()``  (D2H sync)             capacities passed in by caller
    ``flat_src[valid_mask]``  (bool index)           rank scatter, ``cu[b] + t``
    ``repeat_interleave(tensor)``                    ``searchsorted(cu_seqlens)``
    per-segment Python loop over ``seg_list``        arithmetic decomposition of a
                                                     flat index into ``(s,h,i,j)``
    ==============================================  ===============================

    The three capacities replace what the D2H was for.  They must be **at least**
    the true ``sum``, ``max`` and ``sum_s H*T_s^2`` of ``seg_lengths``; pass exact
    values for an eager call, or bucket constants for a capture.  ``None`` means
    "derive the exact value", which costs the D2H back and is only useful for
    testing this function against the host builder.

    Padding beyond the true extent is inert but not free of consequence: a
    trailing dead region is isolated the same way a real segment is (verified —
    scaling the dead frames by 100x leaves every real segment bit-exact), yet
    once it grows past a kernel-selection threshold the *shapes* change and the
    real segments move by ~1.9e-1 in bf16.  So a bucketed packed forward is
    bit-exact to itself, not to the unbucketed one.

    Parameters
    ----------
    seg_lengths : Tensor
        ``(S,)`` int32 post-subsampling length per segment.  Trailing zeros are
        allowed and cost nothing (empty ranges in ``cu_seqlens``, no bias block),
        which is how the segment axis is padded to a capture bucket.
    padded_t : int
        ``Tp``, the padded post-subsampling width of the source batch.
    conv_kernel, num_heads
        As :func:`build_packed_layout`.
    """
    assert seg_lengths.dim() == 1
    device = seg_lengths.device
    S = int(seg_lengths.numel())
    conv_half = (conv_kernel - 1) // 2
    sl64 = seg_lengths.to(torch.int64)

    cu_seqlens = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seg_lengths, dim=0).to(torch.int32)
    cu64 = cu_seqlens.to(torch.int64)

    TT = int(total_capacity) if total_capacity is not None else int(cu_seqlens[S])
    TM = int(max_seg_capacity) if max_seg_capacity is not None else int(sl64.max())
    TM = max(1, TM)

    # pack_src_idx: every valid (b, t) knows its own packed rank without a
    # cumulative count over the mask, because segments are laid out in order --
    # rank = cu_seqlens[b] + t.  Invalid frames scatter into a dump slot that is
    # sliced off, which is what replaces the boolean mask index.
    pos = torch.arange(S * padded_t, device=device, dtype=torch.int64)
    b = pos // padded_t
    t = pos - b * padded_t
    rank = torch.where(t < sl64[b], cu64[b] + t, torch.full_like(pos, TT))
    scratch = torch.zeros(TT + 1, dtype=torch.int64, device=device)
    scratch.scatter_(0, rank.clamp_(max=TT), pos)
    pack_src_idx = scratch[:TT]

    ar = torch.arange(TT, device=device, dtype=torch.int64)
    seg_id = torch.searchsorted(cu64[1:].contiguous(), ar, right=True).clamp_(max=S - 1)
    local_pos = ar - cu64[seg_id]

    conv_gather_idx = ar + conv_half * seg_id
    gapped_len = TT + conv_half * (S - 1) if S > 1 else TT
    conv_batched_idx = (seg_id * TM + local_pos).clamp_(0, S * TM - 1)
    seg_valid_mask = torch.arange(TM, device=device, dtype=torch.int64).unsqueeze(
        0
    ) < sl64.unsqueeze(1)

    bias_offsets = None
    bias_gather_idx = None
    if num_heads is not None and S > 0:
        H = int(num_heads)
        block = H * sl64 * sl64
        bias_offsets = torch.zeros(S + 1, dtype=torch.int32, device=device)
        bias_offsets[1:] = torch.cumsum(block, dim=0).to(torch.int32)
        bo64 = bias_offsets.to(torch.int64)
        BB = int(bias_capacity) if bias_capacity is not None else int(bias_offsets[S])
        # Decompose a flat packed-bias position into (segment, head, i, j) with
        # arithmetic instead of concatenating one arange per segment.  Segment
        # lengths are clamped to >= 1 so a padding segment cannot divide by zero;
        # its entries are never read, because ``bias_offsets`` bounds each block.
        p_ar = torch.arange(BB, device=device, dtype=torch.int64)
        s_id = torch.searchsorted(bo64[1:].contiguous(), p_ar, right=True).clamp_(max=S - 1)
        t_s = sl64[s_id].clamp(min=1)
        rem = p_ar - bo64[s_id]
        t_sq = t_s * t_s
        head = (rem // t_sq).clamp_(0, H - 1)
        within = rem - head * t_sq
        row = within // t_s
        col = within - row * t_s
        bias_gather_idx = (
            ((s_id * H + head) * TM + row.clamp_(max=TM - 1)) * TM + col.clamp_(max=TM - 1)
        ).clamp_(0, S * H * TM * TM - 1)

    return PackedLayout(
        num_segs=S,
        total_tokens=TT,
        seg_lengths=seg_lengths,
        cu_seqlens=cu_seqlens,
        max_seg_len=TM,
        pack_src_idx=pack_src_idx,
        conv_gather_idx=conv_gather_idx,
        gapped_len=gapped_len,
        conv_batched_idx=conv_batched_idx,
        seg_valid_mask=seg_valid_mask,
        bias_offsets=bias_offsets,
        bias_gather_idx=bias_gather_idx,
        src_rows=S * padded_t,
        padded_t=padded_t,
    )


def pack_hidden(xs: torch.Tensor, layout: PackedLayout) -> torch.Tensor:
    """Gather valid frames of padded ``(B, Tp, D)`` into gapless ``(1, T_total, D)``."""
    B, Tp, D = xs.shape
    return xs.reshape(B * Tp, D).index_select(0, layout.pack_src_idx).unsqueeze(0)


def unpack_hidden(packed: torch.Tensor, layout: PackedLayout, D: int) -> torch.Tensor:
    """Scatter gapless ``(1, T_total, D)`` back to padded ``(B, Tp, D)`` (zeros in padding)."""
    out = packed.new_zeros(layout.src_rows, D)
    out.index_copy_(0, layout.pack_src_idx, packed.squeeze(0))
    return out.reshape(layout.num_segs, layout.padded_t, D)
