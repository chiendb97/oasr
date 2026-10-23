# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Statistics collected from calibration activations.

An observer sees activations ``[..., K]`` and keeps a running statistic.  The
optional ``valid`` row mask (``[...]`` bool) is how a batched calibration would
exclude padded frames — padding after subsampling and LayerNorm is not zero and
would otherwise inflate an absmax.  The MVP calibrates one unpadded utterance at
a time, so it passes no mask; the API is there for batched calibration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class Observer(ABC):
    """A running statistic over calibration activations."""

    def __init__(self) -> None:
        self.rows = 0

    def update(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> None:
        """Fold the rows of *x* (``[..., K]``) selected by *valid* into the statistic."""
        rows = x.detach().reshape(-1, x.shape[-1])
        if valid is not None:
            rows = rows[valid.reshape(-1).to(torch.bool)]
        if rows.shape[0] == 0:
            return
        self._update(rows.float())
        self.rows += int(rows.shape[0])

    @abstractmethod
    def _update(self, rows: torch.Tensor) -> None: ...

    @abstractmethod
    def result(self) -> torch.Tensor:
        """The statistic; raises if the observer never saw a row."""


class AbsMaxPerChannel(Observer):
    """``max |x[:, j]|`` per input channel — SmoothQuant's activation statistic."""

    def __init__(self) -> None:
        super().__init__()
        self._amax: Optional[torch.Tensor] = None

    def _update(self, rows: torch.Tensor) -> None:
        m = rows.abs().amax(dim=0)
        self._amax = m if self._amax is None else torch.maximum(self._amax, m)

    def result(self) -> torch.Tensor:
        if self._amax is None:
            raise RuntimeError("the observer saw no calibration activations")
        return self._amax


class AbsMaxPerTensor(Observer):
    """``max |x|`` over everything seen — a static per-tensor activation scale."""

    def __init__(self) -> None:
        super().__init__()
        self._amax: Optional[torch.Tensor] = None

    def _update(self, rows: torch.Tensor) -> None:
        m = rows.abs().amax()
        self._amax = m if self._amax is None else torch.maximum(self._amax, m)

    def result(self) -> torch.Tensor:
        if self._amax is None:
            raise RuntimeError("the observer saw no calibration activations")
        return self._amax


class HistogramPerTensor(Observer):
    """A histogram of ``|x|`` over ``[0, upper]`` -- the input to a clipping search.

    ``upper`` comes from a first pass (:class:`AbsMaxPerTensor`); a value above it
    (a second pass over the same audio sees none) lands in the last bin.
    """

    def __init__(self, upper: float, bins: int = 4096) -> None:
        super().__init__()
        if not upper > 0:
            raise ValueError(f"HistogramPerTensor needs a positive upper bound, got {upper}")
        self.upper = float(upper)
        self.bins = int(bins)
        self._counts: Optional[torch.Tensor] = None

    def _update(self, rows: torch.Tensor) -> None:
        a = rows.abs().flatten().clamp_max(self.upper)
        h = torch.histc(a, bins=self.bins, min=0.0, max=self.upper).double()
        self._counts = h if self._counts is None else self._counts + h.to(self._counts.device)

    def result(self) -> torch.Tensor:
        if self._counts is None:
            raise RuntimeError("the observer saw no calibration activations")
        return self._counts


def clip_percentile(counts: torch.Tensor, upper: float, percentile: float) -> float:
    """The upper edge of the first bin at which the cumulative share reaches *percentile*."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    w = counts.double().cpu()
    cdf = w.cumsum(0) / w.sum()
    idx = int(torch.searchsorted(cdf, torch.tensor(percentile / 100.0, dtype=cdf.dtype)))
    idx = min(idx, counts.numel() - 1)
    return upper * (idx + 1) / counts.numel()


def clip_mse(counts: torch.Tensor, upper: float, qmax: float, fmt: str = "int8") -> float:
    """The clip that minimises the expected quantization error of the histogram.

    For a clip ``c`` (scale ``c / qmax``), a value above ``c`` costs
    ``(|x| - c)**2`` (bin centres) and one below costs the rounding error of the
    format's step at that value, ``step**2 / 12``:

    * ``int8`` -- a uniform step, ``c / qmax``;
    * ``fp8`` (e4m3, ``qmax = 448``) -- a step relative to the value: at
      ``v = |x| / scale``, ``2**(floor(log2 v) - 3)`` in the normal range and the
      subnormal step ``2**-9`` below ``2**-6``.

    Every bin edge is a candidate.
    """
    if fmt not in ("int8", "fp8"):
        raise ValueError(f"clip_mse: unknown format {fmt!r}")
    n = counts.numel()
    w = counts.double().cpu()
    centres = (torch.arange(n, dtype=torch.float64) + 0.5) * (upper / n)
    clips = torch.arange(1, n + 1, dtype=torch.float64) * (upper / n)
    below = w.cumsum(0)  # values in bins [0, i] for clip at edge i+1
    # sum over bins above clip i of w * (centre - clip)^2, for every i, via moments.
    s0 = w.flip(0).cumsum(0).flip(0)
    s1 = (w * centres).flip(0).cumsum(0).flip(0)
    s2 = (w * centres**2).flip(0).cumsum(0).flip(0)
    above0 = torch.cat([s0[1:], s0.new_zeros(1)])
    above1 = torch.cat([s1[1:], s1.new_zeros(1)])
    above2 = torch.cat([s2[1:], s2.new_zeros(1)])
    clip_err = above2 - 2 * clips * above1 + clips**2 * above0
    if fmt == "int8":
        round_err = below * (clips / qmax) ** 2 / 12.0
    else:
        round_err = _fp8_round_err(w, centres, clips, qmax)
    return float(clips[int(torch.argmin(clip_err + round_err))])


def _fp8_round_err(
    w: torch.Tensor, centres: torch.Tensor, clips: torch.Tensor, qmax: float, chunk: int = 512
) -> torch.Tensor:
    """``sum_{centre < clip} w * step_e4m3(centre)**2 / 12`` for every candidate clip."""
    out = torch.empty_like(clips)
    for start in range(0, clips.numel(), chunk):
        c = clips[start : start + chunk, None]  # [C, 1]
        scale = c / qmax
        v = centres[None, :] / scale  # [C, n] in e4m3 units
        exp = torch.floor(torch.log2(v.clamp_min(2.0**-6)))
        step = torch.exp2(exp - 3.0) * scale  # the subnormal step is 2**-9 = 2**(-6-3)
        inside = centres[None, :] < c
        out[start : start + chunk] = (w[None, :] * step**2 / 12.0 * inside).sum(1)
    return out


__all__ = [
    "AbsMaxPerChannel",
    "AbsMaxPerTensor",
    "HistogramPerTensor",
    "Observer",
    "clip_mse",
    "clip_percentile",
]
