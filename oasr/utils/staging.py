# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Asynchronously stage small host index tensors to a device.

Pinned sources avoid the stream drain caused by pageable copies. Same-stream
consumers are ordered automatically; other streams must wait explicitly. Do not
call this during graph capture because replay would retain a temporary host
allocation's address.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch

__all__ = ["to_device"]


def to_device(
    values: Sequence[int],
    *,
    dtype: torch.dtype,
    device: Union[torch.device, str],
) -> torch.Tensor:
    """Ship a host-side list of indices to ``device`` without a stream drain.

    Parameters
    ----------
    values : sequence of int
        Host-side indices — slot ids, block ids, offsets, lengths.  Small: this
        is for per-step metadata, not for payload tensors.
    dtype : torch.dtype
        Element type of the resulting tensor.
    device : torch.device or str
        Destination device.  A non-CUDA destination takes the plain path — there
        is no copy to overlap and pinning would only cost a page-lock.

    Returns
    -------
    Tensor
        ``(len(values),)`` on ``device``.  The copy is asynchronous on the
        current stream; readers on that stream need no further ordering.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    host = torch.tensor(values, dtype=dtype, pin_memory=True)
    return host.to(device, non_blocking=True)
