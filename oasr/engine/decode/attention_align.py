# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Cross-attention DTW: an AED transcript → per-token frame spans.

An attention encoder-decoder has no frame-synchronous emission to read a time
off, so the timing has to come from where the decoder *looked*.  Whisper's
``word_timestamps=True`` is the reference procedure and this is it: take the
cross-attention of a handful of heads that were empirically found to align,
normalize and smooth them into a token × frame affinity matrix, and find the
monotonic path through it with dynamic time warping.

Two things make it work at all, and both are easy to leave out:

* **Only some heads align.** Averaging every head gives a diffuse matrix whose
  DTW path is close to a straight diagonal — plausible-looking timings that are
  really just "token k of n is at k/n of the audio".  Whisper publishes the
  head set per model size; :func:`resolve_alignment_heads` falls back to the
  upper half of the decoder stack and says so, because that is where the
  aligning heads sit in every published set.
* **The padded window is not audio.** Whisper pads every utterance to 30 s, so
  a 4 s clip is 200 real encoder frames followed by 1300 frames of silence that
  the DTW will happily walk into. The matrix is cut to the real frames first.

The DP is over anti-diagonals rather than the reference implementation's nested
Python loops: cells on one anti-diagonal depend only on the two before it, so a
``(200 tokens × 1500 frames)`` alignment is ~1700 vector steps instead of 300k
scalar ones.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "resolve_alignment_heads",
    "median_filter",
    "dtw",
    "token_frame_spans",
]

#: Median-filter width along the time axis, in encoder frames.  Whisper's value.
#: Attention peaks are spiky; without smoothing the DTW path oscillates between
#: neighbouring frames and word boundaries land a frame or two off at random.
MEDFILT_WIDTH = 7


def resolve_alignment_heads(
    declared: Optional[Sequence[Sequence[int]]],
    num_layers: int,
    num_heads: int,
) -> Tuple[List[Tuple[int, int]], bool]:
    """``(layer, head)`` pairs to average, and whether they came from the checkpoint.

    A declared set is used as-is, minus any pair outside this model's geometry
    (a mismatched ``generation_config.json``).  With nothing declared, every
    head of the **upper half** of the decoder stack is used: the published sets
    live there, and averaging the lower layers — which attend broadly rather
    than positionally — is what turns the matrix into a diagonal.
    """
    if declared:
        pairs = [
            (int(layer), int(head))
            for layer, head in declared
            if 0 <= int(layer) < num_layers and 0 <= int(head) < num_heads
        ]
        if pairs:
            return pairs, True
    first = num_layers // 2
    return [(layer, head) for layer in range(first, num_layers) for head in range(num_heads)], False


def median_filter(x: np.ndarray, width: int = MEDFILT_WIDTH) -> np.ndarray:
    """Median filter along the last axis, reflect-padded (Whisper's smoother)."""
    if width <= 1 or x.shape[-1] < width:
        return x
    pad = width // 2
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1)
    out: np.ndarray = np.median(windows, axis=-1)
    return out


def dtw(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Monotonic alignment path through an ``(N, M)`` cost matrix.

    Returns ``(text_indices, time_indices)``: two equal-length arrays naming the
    cells the optimal path visits, both non-decreasing.  Transitions are the
    standard three — diagonal, hold the row, hold the column — so every token
    gets at least one frame and every frame is claimed by exactly one token.
    """
    n, m = cost.shape
    inf = np.inf
    acc = np.full((n + 1, m + 1), inf, dtype=np.float64)
    acc[0, 0] = 0.0
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)

    # Anti-diagonal sweep: (i, j) needs (i-1, j-1) from two diagonals back and
    # (i-1, j) / (i, j-1) from one, so a whole diagonal computes at once.
    for d in range(2, n + m + 1):
        lo, hi = max(1, d - m), min(n, d - 1)
        if lo > hi:
            continue
        ii = np.arange(lo, hi + 1)
        jj = d - ii
        cand = np.stack((acc[ii - 1, jj - 1], acc[ii - 1, jj], acc[ii, jj - 1]))
        choice = cand.argmin(axis=0)
        acc[ii, jj] = cost[ii - 1, jj - 1] + cand[choice, np.arange(ii.size)]
        trace[ii, jj] = choice.astype(np.int8)

    # Walking off either edge is a forced move back along it.
    trace[0, 1:] = 2
    trace[1:, 0] = 1

    i, j = n, m
    path_i: List[int] = []
    path_j: List[int] = []
    while i > 0 or j > 0:
        path_i.append(i - 1)
        path_j.append(j - 1)
        step = int(trace[i, j])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return np.array(path_i[::-1]), np.array(path_j[::-1])


def token_frame_spans(
    weights: np.ndarray,
    num_frames: int,
    medfilt_width: int = MEDFILT_WIDTH,
) -> Optional[List[Tuple[int, int]]]:
    """``(heads, tokens, enc_frames)`` cross-attention → per-token frame spans.

    ``weights`` must already be restricted to the tokens being timed — the SOT
    prompt is not part of the transcript and its attention is not about the
    audio.  ``num_frames`` is how many encoder frames are *real*; everything
    past it is the padded 30 s window.

    Returns ``None`` when there is nothing to align (no tokens, or no audio),
    rather than a degenerate all-at-zero list.
    """
    if weights.ndim != 3 or weights.shape[1] == 0:
        return None
    frames = int(min(max(num_frames, 1), weights.shape[2]))
    w = weights[:, :, :frames].astype(np.float64)
    if w.shape[2] < 2:
        return None

    # Normalize each head over the *token* axis so a head that always attends
    # strongly does not dominate the average, then smooth along time.
    mean = w.mean(axis=1, keepdims=True)
    std = w.std(axis=1, keepdims=True)
    w = (w - mean) / np.maximum(std, 1e-9)
    w = median_filter(w, medfilt_width)

    matrix = w.mean(axis=0)  # (tokens, frames)
    text_idx, time_idx = dtw(-matrix)

    n_tokens = matrix.shape[0]
    spans: List[Tuple[int, int]] = []
    prev_end = 0
    for k in range(n_tokens):
        hits = time_idx[text_idx == k]
        if hits.size == 0:
            # The path skipped this token entirely (possible only at the
            # boundaries); give it a zero-length slot at the running position so
            # the sequence stays monotone and the token still exists.
            spans.append((prev_end, prev_end))
            continue
        start, end = int(hits[0]), int(hits[-1]) + 1
        spans.append((start, end))
        prev_end = end
    return spans
