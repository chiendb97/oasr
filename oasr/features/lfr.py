# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Batched low-frame-rate (LFR) frame stacking.

FunASR/Paraformer frontends stack ``lfr_m`` consecutive fbank frames and
advance by ``lfr_n`` (80-mel LFR 7/6 → 560-dim features at a 60 ms hop).  The
reference implementation (``funasr.frontends.wav_frontend.apply_lfr``) prepends
``(lfr_m - 1) // 2`` copies of the first frame and completes the trailing
window by repeating the last frame; that is exactly a gather with the source
index clamped to ``[0, T_valid - 1]``, which is what :func:`apply_lfr_batch`
does — vectorized over a padded batch with per-row valid lengths.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def lfr_output_length(num_frames: int, lfr_n: int) -> int:
    """Number of LFR frames produced from ``num_frames`` input frames."""
    if num_frames <= 0:
        return 0
    return (num_frames + lfr_n - 1) // lfr_n


def apply_lfr_batch(
    feats: torch.Tensor,
    lengths: torch.Tensor,
    lfr_m: int,
    lfr_n: int,
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply LFR stacking to a padded feature batch.

    Parameters
    ----------
    feats : Tensor
        ``(B, T, F)`` padded features.
    max_length : int, optional
        Host-side upper bound on ``lengths``, used to size the output without
        a device→host sync.  The offline path knows this already (it is the
        padded batch width).  ``None`` reads it off the device tensor, which
        costs one blocking sync per call.
    lengths : Tensor
        ``(B,)`` valid frame counts.
    lfr_m, lfr_n : int
        Stack ``lfr_m`` frames per output frame, advance ``lfr_n``.

    Returns
    -------
    out : Tensor
        ``(B, T', F * lfr_m)`` with ``T' = ceil(max(lengths) / lfr_n)``; rows
        beyond each stream's own LFR length are zero.
    out_lengths : Tensor
        ``(B,)`` per-row LFR frame counts (``ceil(len / lfr_n)``), same dtype
        and device as ``lengths``.
    """
    if lfr_m == 1 and lfr_n == 1:
        return feats, lengths

    B, T, F = feats.shape
    device = feats.device
    lengths_dev = lengths.to(device=device, dtype=torch.long)
    out_lengths = (lengths_dev + lfr_n - 1) // lfr_n
    if max_length is not None:
        # Host-supplied bound — no sync.  ``.max().item()`` on a device tensor
        # is a blocking D2H that drains the queue, and it ran unconditionally on
        # every offline micro-batch whenever LFR is enabled (i.e. all of
        # Paraformer).  The caller already knows the padded width host-side.
        t_out = (int(max_length) + lfr_n - 1) // lfr_n
    else:
        t_out = int(out_lengths.max().item()) if B > 0 else 0
    if t_out == 0:
        empty = feats.new_zeros(B, 0, F * lfr_m)
        return empty, out_lengths.to(dtype=lengths.dtype)

    left = (lfr_m - 1) // 2
    # Source frame index for (output frame i, stack slot k): i*lfr_n + k - left,
    # clamped per row to [0, len-1] — replicating the first frame on the left
    # edge and the last valid frame on the trailing partial window, exactly as
    # the reference loop does.
    base = (
        torch.arange(t_out, device=device).unsqueeze(1) * lfr_n
        + torch.arange(lfr_m, device=device).unsqueeze(0)
        - left
    )  # (T', lfr_m)
    idx = base.reshape(1, -1).expand(B, -1)  # (B, T'*lfr_m)
    max_idx = (lengths_dev - 1).clamp(min=0).unsqueeze(1)
    idx = idx.clamp(min=0).minimum(max_idx)

    gathered = torch.gather(feats, 1, idx.unsqueeze(-1).expand(B, t_out * lfr_m, F))
    out = gathered.reshape(B, t_out, lfr_m * F)

    # Zero rows past each stream's own LFR length (gather clamped them to the
    # last valid frame, which would leak real values into the padding).
    valid = torch.arange(t_out, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)
    # In place: ``out`` is a fresh reshape of the gather result, so nothing
    # aliases it, and the out-of-place form doubled a (B, T', lfr_m*F) tensor.
    out.mul_(valid.unsqueeze(-1).to(out.dtype))
    return out, out_lengths.to(dtype=lengths.dtype)
