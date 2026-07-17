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

from typing import Tuple

import torch
from torch import nn

from .config import ParaformerModelConfig


def cif_v1(
    hidden: torch.Tensor, alphas: torch.Tensor, threshold: float
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

    Returns
    -------
    frames : Tensor
        ``(B, U_max, D)`` per-token integrated embeddings, ``U_max =
        max(round(sum(alphas)))``; rows beyond a row's own fire count are zero.
    fires : Tensor
        ``(B, T)`` — ``≥ 1`` exactly at fire positions, with the running
        integral's fractional part elsewhere (the CIF "peak" signal used for
        timestamps).
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
    dislocation_floor[:, 0] = 0
    fire_idxs = (prefix_sum_floor - dislocation_floor) > 0

    fires = torch.zeros(batch_size, len_time, dtype=dtype, device=device)
    fires[fire_idxs] = 1
    fires = fires + prefix_sum - prefix_sum_floor

    max_label_len = int(torch.round(alphas.sum(-1)).int().max().item())
    if not bool(fire_idxs.any()):
        return (
            torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device),
            fires,
        )

    prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1) * hidden, dim=1)
    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = torch.roll(frames, 1, dims=0)

    batch_len = fire_idxs.sum(1)
    batch_idxs = torch.cumsum(batch_len, dim=0)
    shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
    shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0

    remains = fires - torch.floor(fires)
    remain_frames = remains[fire_idxs].unsqueeze(-1) * hidden[fire_idxs]
    shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
    shift_remain_frames[shift_batch_idxs] = 0

    frames = frames - shift_frames + shift_remain_frames - remain_frames

    frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device)
    indices = torch.arange(max_label_len, device=device).expand(batch_size, -1)
    frame_fires[indices < batch_len.unsqueeze(1)] = frames
    return frame_fires, fires


class CifPredictor(nn.Module):
    """FunASR ``CifPredictorV2``: conv+linear alpha head → tail append → CIF."""

    def __init__(self, config: ParaformerModelConfig) -> None:
        super().__init__()
        idim = config.predictor_idim
        self.pad = nn.ConstantPad1d((config.predictor_l_order, config.predictor_r_order), 0)
        self.cif_conv1d = nn.Conv1d(
            idim, idim, config.predictor_l_order + config.predictor_r_order + 1
        )
        self.cif_output = nn.Linear(idim, 1)
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
        h = self.pad(hidden.to(self.cif_output.weight.dtype).transpose(1, 2))
        alphas = torch.sigmoid(self.cif_output(torch.relu(self.cif_conv1d(h)).transpose(1, 2)))
        hidden = hidden.float()
        mask = mask.float()
        alphas = torch.relu(alphas.squeeze(-1).float()) * mask  # (B, T)

        hidden, alphas = self._append_tail(hidden, alphas, mask)
        token_num = torch.floor(alphas.sum(dim=-1))

        acoustic_embeds, fires = cif_v1(hidden, alphas, self.threshold)
        u_max = int(token_num.max().clamp(min=0).item())
        return acoustic_embeds[:, :u_max, :], token_num, alphas, fires

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
