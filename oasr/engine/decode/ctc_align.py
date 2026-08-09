# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CTC timing: the beam's own emission frames, and the oracle that checks them.

A CTC prefix beam search knows *when* it emitted each token at the moment it
emits it, and OASR's kernel now records that frame beside the token
(``ctc_decoder.cuh``: ``ctime`` in flat mode, ``time_storage`` in paged).  So the
production path here is a **read**, not a computation:
:func:`attach_emission_timings` turns those frames into spans and looks the
posterior up with a single gather at ``(frame, token)`` — both of which the beam
already decided.

The rest of this module is :func:`forced_align`, the standard Viterbi alignment
of a decoded hypothesis against its log-probs.  It was the production path
briefly and should not be again: it re-derives what the decoder already knew, at
roughly **ten times the cost of the decode it decorates**, and it cannot serve a
stream at all because the log-probs are gone by the time the transcript is
final.  It stays because it is an independent implementation of the same
question, checkable bit-for-bit against
``torchaudio.functional.forced_align`` — which makes it the oracle the kernel's
frames are validated against (``tests/test_word_timings.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .alignment import TokenAlignment, wants_word_timings

if TYPE_CHECKING:
    from ..request import Request, RequestOutput
    from .base import DecodeStrategy

__all__ = [
    "forced_align",
    "align_hypotheses",
    "attach_emission_timings",
    "token_posteriors",
]

#: Log-domain zero.  Not ``-inf``: the DP adds it to a real log-prob and takes
#: differences of the result, and ``-inf + x`` propagating through an ``argmax``
#: comparison against another ``-inf`` makes the choice depend on NumPy's tie
#: handling rather than on the model.  A finite floor keeps every comparison
#: meaningful while still being unreachable — the same reasoning as the fused
#: attention kernel's mask floor.
_NEG = -1.0e30


def _extend(tokens: Sequence[int], blank_id: int) -> np.ndarray:
    """``[y1..yL]`` → ``[blank, y1, blank, y2, ..., yL, blank]`` (length ``2L+1``)."""
    ext = np.full(2 * len(tokens) + 1, blank_id, dtype=np.int64)
    ext[1::2] = np.asarray(tokens, dtype=np.int64)
    return ext


def forced_align(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    hypotheses: Sequence[Sequence[int]],
    blank_id: int,
) -> List[Optional[List[TokenAlignment]]]:
    """Align each row's hypothesis against its log-probs.

    Parameters
    ----------
    log_probs : torch.Tensor
        ``(B, T, V)`` log-probabilities — exactly what the CTC families decode
        from, so the alignment is against the same distribution the beam saw.
    lengths : torch.Tensor
        ``(B,)`` valid frame counts.
    hypotheses : sequence of sequences of int
        One decoded token sequence per row (no blanks).  An empty row yields
        an empty alignment.
    blank_id : int
        The CTC blank.

    Returns
    -------
    list
        One entry per row: the per-token alignments, or ``None`` when no CTC
        path spells that hypothesis in that many frames (a hypothesis longer
        than the audio can express, which the beam's ``max_seq_len`` truncation
        can produce).  ``None`` rather than a partial list — a caller cannot
        tell a genuinely absent word from a dropped one.
    """
    b_n = int(log_probs.size(0))
    if b_n == 0:
        return []
    ext_rows = [_extend(h, blank_id) for h in hypotheses]
    s_lens = np.array([len(e) for e in ext_rows], dtype=np.int64)
    s_max = int(s_lens.max())
    t_lens = lengths.detach().to("cpu", torch.int64).numpy()
    t_max = int(min(int(t_lens.max()), int(log_probs.size(1))))
    if t_max <= 0:
        return [None] * b_n

    # Pad the extended sequences to a rectangle so the gather is one call.  The
    # pad value is the blank, which is always a valid column; padded states are
    # masked out of the DP by ``s_lens``, never by the gather.
    ext = np.full((b_n, s_max), blank_id, dtype=np.int64)
    for b, row in enumerate(ext_rows):
        ext[b, : len(row)] = row

    idx = torch.from_numpy(ext).to(log_probs.device).unsqueeze(1)  # (B, 1, S)
    # ``take_along_dim`` broadcasts the index over T, so the (B, T, S) view is
    # never materialised as int64 indices — for a 30 s utterance that index
    # tensor alone would be larger than the log-probs it selects from.
    gathered = torch.take_along_dim(log_probs[:, :t_max].float(), idx, dim=2)
    lp = gathered.detach().to("cpu").numpy()  # (B, t_max, S)

    return _viterbi(lp, t_lens, s_lens, ext, hypotheses)


def _viterbi(
    lp: np.ndarray,
    t_lens: np.ndarray,
    s_lens: np.ndarray,
    ext: np.ndarray,
    hypotheses: Sequence[Sequence[int]],
) -> List[Optional[List[TokenAlignment]]]:
    """Batched Viterbi over the extended label sequences; then backtrack."""
    b_n, t_max, s_max = lp.shape

    # A state is reachable in one hop from itself and from ``s-1``; the ``s-2``
    # skip crosses the separating blank and is legal only into a *label* state
    # whose predecessor label differs — the rule that forces a blank between
    # two identical adjacent labels, and the only place CTC alignment differs
    # from a plain monotonic alignment.
    skip = np.zeros((b_n, s_max), dtype=bool)
    if s_max >= 3:
        odd = np.arange(2, s_max)
        skip[:, 2:] = (odd % 2 == 1)[None, :] & (ext[:, 2:] != ext[:, :-2])
    valid = np.arange(s_max)[None, :] < s_lens[:, None]

    alpha = np.full((b_n, s_max), _NEG, dtype=np.float32)
    alpha[:, 0] = lp[:, 0, 0]
    if s_max >= 2:
        # A path may open on the first label as well as on the leading blank.
        # Guarded because a batch whose every hypothesis is empty has one
        # extended state in total, and indexing column 1 would be out of range.
        has_label = s_lens > 1
        alpha[has_label, 1] = lp[has_label, 0, 1]
    alpha[~valid] = _NEG

    # Backpointers: how far back the winning predecessor was (0, 1 or 2).
    back = np.zeros((t_max, b_n, s_max), dtype=np.int8)
    for t in range(1, t_max):
        stay = alpha
        prev1 = np.full_like(alpha, _NEG)
        prev1[:, 1:] = alpha[:, :-1]
        prev2 = np.full_like(alpha, _NEG)
        if s_max >= 3:
            prev2[:, 2:] = np.where(skip[:, 2:], alpha[:, :-2], _NEG)

        cand = np.stack((stay, prev1, prev2))  # (3, B, S)
        choice = cand.argmax(axis=0).astype(np.int8)
        best = cand.max(axis=0) + lp[:, t]
        best[~valid] = _NEG

        # A row shorter than ``t_max`` freezes: its alpha stops advancing, and
        # backtracking starts from its own last frame, so these steps are never
        # revisited.
        active = (t < t_lens)[:, None]
        alpha = np.where(active, best, alpha)
        back[t] = choice

    out: List[Optional[List[TokenAlignment]]] = []
    for b in range(b_n):
        n_tokens = len(hypotheses[b])
        if n_tokens == 0:
            out.append([])
            continue
        out.append(_backtrack(lp[b], back[:, b], alpha[b], int(t_lens[b]), int(s_lens[b])))
    return out


def _backtrack(
    lp_b: np.ndarray,
    back_b: np.ndarray,
    alpha_b: np.ndarray,
    t_len: int,
    s_len: int,
) -> Optional[List[TokenAlignment]]:
    """Walk one row's backpointers into per-token spans and posteriors."""
    t_len = min(t_len, lp_b.shape[0])
    if t_len <= 0 or s_len < 3:
        return None
    # A path ends on the final label or on the trailing blank after it.
    last = s_len - 1
    end_state = last if alpha_b[last] >= alpha_b[last - 1] else last - 1
    if alpha_b[end_state] <= _NEG / 2:
        # No CTC path spells this hypothesis in this many frames.
        return None

    path = np.empty(t_len, dtype=np.int64)
    s = end_state
    for t in range(t_len - 1, 0, -1):
        path[t] = s
        s -= int(back_b[t, s])
    path[0] = s

    n_tokens = (s_len - 1) // 2
    align: List[TokenAlignment] = []
    # Label ``k`` lives at extended state ``2k+1``; the frames assigned to it
    # are contiguous, so first/last bound the span and the mean of their frame
    # probabilities is the token's posterior.
    for k in range(n_tokens):
        state = 2 * k + 1
        frames = np.flatnonzero(path == state)
        if frames.size == 0:
            # Reachable only if the backtrack went wrong; a partially-timed
            # list is worse than none.
            return None
        probs = np.exp(lp_b[frames, state])
        align.append(
            TokenAlignment(
                token=0,  # filled by the caller, which owns the id list
                start_frame=float(frames[0]),
                end_frame=float(frames[-1] + 1),
                confidence=float(np.clip(probs.mean(), 0.0, 1.0)),
            )
        )
    return align


def align_hypotheses(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    hypotheses: Sequence[Sequence[int]],
    blank_id: int,
) -> List[Optional[List[TokenAlignment]]]:
    """:func:`forced_align` with the token ids filled in.

    :func:`forced_align` reports spans positionally; this stamps each with the
    id it aligned, which is what :func:`~oasr.engine.decode.alignment.word_timings`
    needs to render the piece back to text.
    """
    aligned = forced_align(log_probs, lengths, hypotheses, blank_id)
    out: List[Optional[List[TokenAlignment]]] = []
    for hyp, row in zip(hypotheses, aligned):
        if row is None:
            out.append(None)
            continue
        out.append(
            [
                TokenAlignment(
                    token=int(tok),
                    start_frame=a.start_frame,
                    end_frame=a.end_frame,
                    confidence=a.confidence,
                )
                for tok, a in zip(hyp, row)
            ]
        )
    return out


#: Frames past the emission frame that a token's posterior peak may sit at.
#:
#: A prefix beam commits to an extension at the **leading edge** of the label's
#: posterior peak — the first frame where it becomes a competitive extension —
#: which on a real Conformer is the peak frame itself for ~3/4 of tokens and one
#: frame before it for the rest, never later and never further.  Reading the
#: posterior strictly at the emission frame therefore under-reports a quarter of
#: them badly (0.08 where the peak is 1.00).  Two frames is exactly the observed
#: spread; widening it would start absorbing the *next* label's peak.
_PEAK_LOOKAHEAD = 1


@torch.no_grad()
def token_posteriors(
    log_probs: torch.Tensor,
    rows: Sequence[Tuple[int, Sequence[int], Sequence[int]]],
) -> List[List[float]]:
    """Per-token posterior at (or just after) each token's emission frame.

    ``max(exp(log_probs[row, t .. t + _PEAK_LOOKAHEAD, token]))``.  The frames
    and the tokens are both already known — the beam chose them — so a
    confidence costs a lookup rather than a second pass over the distribution.

    The addressing is done in NumPy and handed over as **one** flat index
    array, so the whole batch costs one host→device copy and one
    ``take`` per lookahead offset.  Building three index lists element by
    element in Python and pushing each across was, measured, most of the cost
    of the confidences — the gather itself is a rounding error.
    """
    counts = [len(toks) for _, toks, _ in rows]
    total = sum(counts)
    if total == 0:
        return [[] for _ in rows]
    t_len, vocab = int(log_probs.size(1)), int(log_probs.size(2))
    t_max = t_len - 1
    frame = np.concatenate([np.fromiter(fr, np.int64, len(fr)) for _, _, fr in rows])
    np.clip(frame, 0, t_max, out=frame)
    token = np.concatenate([np.fromiter(tk, np.int64, len(tk)) for _, tk, _ in rows])
    batch = np.repeat(np.fromiter((b for b, _, _ in rows), np.int64, len(rows)), counts)

    # One flat offset into the (B, T, V) log-probs; ``take`` on the flattened
    # view is a single kernel over it.
    base = batch * (t_len * vocab) + token
    flat = torch.from_numpy(base + frame * vocab).to(log_probs.device, non_blocking=True)
    values = log_probs.reshape(-1)
    best = values.take(flat).float()
    for step in range(1, _PEAK_LOOKAHEAD + 1):
        ahead = np.minimum(frame + step, t_max)
        nxt = torch.from_numpy(base + ahead * vocab).to(log_probs.device, non_blocking=True)
        best = torch.maximum(best, values.take(nxt).float())
    vals = best.exp_().clamp_(0.0, 1.0).cpu().tolist()
    out: List[List[float]] = []
    pos = 0
    for n in counts:
        out.append(vals[pos : pos + n])
        pos += n
    return out


def attach_emission_timings(
    strategy: "DecodeStrategy",
    requests: Sequence["Request"],
    outputs: Sequence["RequestOutput"],
    times: Sequence[Sequence[Sequence[int]]],
    log_probs: torch.Tensor,
    *,
    beam_index: Optional[Sequence[int]] = None,
) -> None:
    """Attach word timings from the beam's recorded emission frames.

    ``times`` is the decoder's ``[batch][beam][frame]`` output.  ``beam_index``
    names which beam row won per batch row — the identity for a plain CTC
    decode, and the *fusion winner* for rescoring, where timing the CTC-best
    would put one transcript's words on another's clock.
    """
    wanted = [i for i, req in enumerate(requests) if wants_word_timings(req)]
    if not wanted or not times:
        return
    rows: List[Tuple[int, List[int], List[int]]] = []
    for i in wanted:
        k = beam_index[i] if beam_index is not None else 0
        toks = outputs[i].tokens[0] if outputs[i].tokens else []
        frames = list(times[i][k]) if i < len(times) and k < len(times[i]) else []
        if toks and len(frames) == len(toks):
            rows.append((i, toks, frames))
    if not rows:
        return
    for (i, toks, frames), probs in zip(rows, token_posteriors(log_probs, rows)):
        strategy.attach_emission_alignment(outputs[i], toks, frames, probs)
