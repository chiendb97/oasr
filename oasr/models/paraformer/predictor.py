# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CIF predictor (FunASR ``CifPredictorV2``) — continuous integrate-and-fire.

Estimates per-frame token weights (alphas) from the encoder output, appends a
``tail_threshold`` weight at each utterance's end-of-speech position, and
integrates encoder frames into per-token acoustic embeddings.  The
:func:`cif_v1` integration is the vectorized prefix-sum formulation from
FunASR (float64 cumsum to dodge precision drift; assumes ``threshold == 1``).

Everything here runs in float32 regardless of the engine dtype — FunASR wraps
the predictor in ``autocast(False)`` for the same reason: the running integral
is numerically fragile in half precision.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from oasr.layers import Conv1dActivation, Linear, Sigmoid
from oasr.models.base import align_out_features

from .config import ParaformerModelConfig


def cif_v1(
    hidden: torch.Tensor,
    alphas: torch.Tensor,
    threshold: float,
    num_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized integrate-and-fire (FunASR ``cif_v1``).

    Parameters
    ----------
    hidden : Tensor
        ``(B, T, D)`` frames to integrate.
    alphas : Tensor
        ``(B, T)`` non-negative weights.
    threshold : float
        Fire threshold; the prefix-sum formulation requires ``1.0``.
    num_tokens : int, optional
        Token rows to return.  ``None`` sizes them as FunASR does,
        ``max(round(sum(alphas)))``, which reads that value back; a caller that
        already knows how many rows it keeps passes it and saves the sync.

    Returns
    -------
    frames : Tensor
        ``(B, U, D)`` per-token integrated embeddings, ``U = num_tokens`` or
        ``max(round(sum(alphas)))``; rows beyond a row's own fire count are zero.
    fires : Tensor
        ``(B, T)`` — ``≥ 1`` exactly at fire positions, with the running
        integral's fractional part elsewhere (the CIF "peak" signal used for
        timestamps).

    The arithmetic is FunASR's, term for term: token ``u`` integrates
    ``P[t_u] - P[t_{u-1}] + r[t_{u-1}] h[t_{u-1}] - r[t_u] h[t_u]``, where ``P``
    is the running integral of ``alpha * h``, ``t_u`` the frame of the ``u``-th
    fire and ``r`` the integral's fractional part.  Only the addressing differs:
    FunASR collects the fire frames with boolean masks, each a device-to-host
    sync (five of them, plus a ``bool(any)``), and this gathers them at
    positions found by ``searchsorted`` on the running fire count, so the
    shapes are fixed and nothing is read back but ``U``.  The terms are the
    same values combined in the same order, so the result is bit-identical.
    """
    if threshold != 1.0:
        raise ValueError(f"cif_v1 prefix-sum formulation requires threshold=1.0, got {threshold}")
    device = hidden.device
    dtype = hidden.dtype
    batch_size, len_time, hidden_size = hidden.shape

    prefix_sum = torch.cumsum(alphas, dim=1, dtype=torch.float64).to(torch.float32)
    prefix_sum_floor = torch.floor(prefix_sum)
    dislocation = torch.roll(prefix_sum, 1, dims=1)
    dislocation_floor = torch.floor(dislocation)
    # ``t[:, 0] = 0`` would stage the scalar through a synchronising copy.
    dislocation_floor[:, 0].fill_(0)
    fire_idxs = (prefix_sum_floor - dislocation_floor) > 0

    fires = fire_idxs.to(dtype) + prefix_sum - prefix_sum_floor

    if num_tokens is None:
        num_tokens = int(torch.round(alphas.sum(-1)).int().max().item())
    num_tokens = max(0, int(num_tokens))
    if num_tokens == 0 or len_time == 0:
        return torch.zeros(batch_size, num_tokens, hidden_size, dtype=dtype, device=device), fires

    prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1) * hidden, dim=1)
    remains = fires - torch.floor(fires)

    # The frame of each token's fire: the first frame whose running fire count
    # reaches ``u + 1``.  Rows that fire fewer times get ``len_time`` there,
    # clamped for the gather and zeroed by ``valid`` below.
    fire_count = fire_idxs.to(torch.int64).cumsum(dim=1)
    wanted = torch.arange(1, num_tokens + 1, device=device, dtype=torch.int64)
    t_u = torch.searchsorted(fire_count, wanted.expand(batch_size, -1).contiguous())
    valid = t_u < len_time
    t_u = t_u.clamp_(max=len_time - 1)
    t_d = t_u.unsqueeze(-1).expand(-1, -1, hidden_size)

    frames = prefix_sum_hidden.gather(1, t_d)
    remain_frames = remains.gather(1, t_u).unsqueeze(-1) * hidden.gather(1, t_d)
    # The previous fire's terms, zero before a row's first token.
    shift_frames = F.pad(frames[:, :-1], (0, 0, 1, 0))
    shift_remain_frames = F.pad(remain_frames[:, :-1], (0, 0, 1, 0))

    frames = frames - shift_frames + shift_remain_frames - remain_frames
    frame_fires = torch.where(valid.unsqueeze(-1), frames, frames.new_zeros(()))
    return frame_fires.to(dtype), fires


class CifPredictor(nn.Module):
    """FunASR ``CifPredictorV2``: conv+linear alpha head → tail append → CIF."""

    def __init__(self, config: ParaformerModelConfig) -> None:
        super().__init__()
        idim = config.predictor_idim
        self.left_padding = config.predictor_l_order
        self.right_padding = config.predictor_r_order
        self.cif_conv1d = Conv1dActivation(
            idim,
            idim,
            config.predictor_l_order + config.predictor_r_order + 1,
            activation_type="relu",
        )
        # A one-wide output has no GEMM kernel either (alignment 8), so the
        # alpha head is allocated aligned like the vocabulary heads and column
        # 0 is taken; ``ParaformerModel.load_weights`` widens the checkpoint.
        self.cif_output = Linear(idim, align_out_features(1))
        self.sigmoid = Sigmoid()
        self.threshold = config.predictor_threshold
        self.tail_threshold = config.predictor_tail_threshold

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CIF over an encoder output.

        Parameters
        ----------
        hidden : Tensor
            ``(B, T, D)`` encoder output (any float dtype; computed in fp32).
        mask : Tensor
            ``(B, T)`` — 1 on valid frames, 0 on padding.

        Returns
        -------
        acoustic_embeds : Tensor
            ``(B, U_max, D)`` float32, truncated to ``U_max = max(token_num)``.
        token_num : Tensor
            ``(B,)`` float32 — ``floor(sum(alphas))`` per row (callers round).
        alphas : Tensor
            ``(B, T+1)`` float32 tail-appended per-frame weights.
        fires : Tensor
            ``(B, T+1)`` float32 CIF peak signal (``≥ 1`` at fire positions).
        """
        # Alpha head in the module's own dtype (fp16 under the engine); the
        # integration below is always fp32.
        h = F.pad(
            hidden.to(self.cif_output.weight.dtype),
            (0, 0, self.left_padding, self.right_padding),
        )
        alpha_logits = self.cif_output(self.cif_conv1d(h))
        # Apply sigmoid before slicing the aligned alpha head. This keeps the
        # kernel input contiguous and avoids materializing a row-strided view;
        # only column 0 is a checkpoint output, the padding columns are ignored.
        alphas = self.sigmoid(alpha_logits)[..., 0]
        hidden = hidden.float()
        mask = mask.float()
        alphas = alphas.float() * mask  # sigmoid is already strictly non-negative

        hidden, alphas = self._append_tail(hidden, alphas, mask)
        token_num = torch.floor(alphas.sum(dim=-1))

        # The one read-back: how many token rows the batch keeps.  ``cif_v1``
        # integrates exactly those rather than sizing its own and being cut.
        u_max = int(token_num.max().clamp(min=0).item())
        acoustic_embeds, fires = cif_v1(hidden, alphas, self.threshold, num_tokens=u_max)
        return acoustic_embeds, token_num, alphas, fires

    def _append_tail(
        self, hidden: torch.Tensor, alphas: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add ``tail_threshold`` at each row's first pad position (FunASR
        ``tail_process_fn``): guarantees the trailing partial integral fires
        so the last token isn't dropped.  Appends one zero frame / one alpha
        column, so the time axis grows to ``T + 1``."""
        b, _, d = hidden.shape
        zeros_t = alphas.new_zeros(b, 1)
        ones_t = torch.ones_like(zeros_t)
        # 1 exactly at the first pad position of each row ((mask | first pad
        # slot) minus (mask shifted right)); rows with no padding fire at the
        # appended column.
        tail = torch.cat([ones_t, mask], dim=1) - torch.cat([mask, zeros_t], dim=1)
        alphas = torch.cat([alphas, zeros_t], dim=1) + tail * self.tail_threshold
        hidden = torch.cat([hidden, hidden.new_zeros(b, 1, d)], dim=1)
        return hidden, alphas
