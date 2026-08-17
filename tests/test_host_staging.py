# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`oasr.utils.staging` — the per-step host→device index copies.

The property under test is not the values (any formulation gets those right)
but that shipping them **does not drain the stream**.  A pageable
``torch.tensor(..., device="cuda")`` synchronises before it stages, so on the
streaming path — where one of these lands immediately after the encoder forward
is issued — it parks the host until the GPU has caught up, and the host can
never run a step ahead of the device.
"""

from __future__ import annotations

import pytest
import torch

from oasr.utils.staging import to_device


def _congest(device: torch.device, iters: int = 64) -> torch.Tensor:
    """Queue enough GPU work that a synchronising call cannot hide.

    Deliberately congests the stream rather than trusting a timing threshold:
    the assertion below is about pipeline *state* (is work still outstanding),
    which is what the staging discipline actually buys.
    """
    big = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    for _ in range(iters):
        big = torch.mm(big, big)
    return big


@pytest.mark.cuda
class TestHostStaging:
    def test_values_match_the_pageable_build(self, device):
        values = [7, 0, 3, 11]
        staged = to_device(values, dtype=torch.long, device=device)
        assert staged.dtype is torch.long
        assert staged.device.type == "cuda"
        assert staged.tolist() == values

    def test_dtype_is_honoured(self, device):
        staged = to_device([1, 2, 3], dtype=torch.int32, device=device)
        assert staged.dtype is torch.int32
        assert staged.tolist() == [1, 2, 3]

    def test_does_not_drain_the_stream(self, device):
        """The copy is issued, not awaited — queued work is still outstanding."""
        torch.cuda.synchronize(device)
        keep_alive = _congest(device)  # noqa: F841 — must outlive the assertion
        staged = to_device(list(range(32)), dtype=torch.long, device=device)
        outstanding = not torch.cuda.current_stream(device).query()
        torch.cuda.synchronize(device)
        assert staged.tolist() == list(range(32))
        assert outstanding, (
            "to_device drained the stream — the staged copy must be an async DMA "
            "out of pinned memory, not a pageable copy that synchronises first"
        )

    def test_pageable_build_does_drain_the_stream(self, device):
        """The control: what the call sites used to do, and why it cost.

        Without this the test above proves nothing — a GPU fast enough to
        finish the congestion would pass it either way.
        """
        torch.cuda.synchronize(device)
        keep_alive = _congest(device)  # noqa: F841
        torch.tensor(list(range(32)), dtype=torch.long, device=device)
        assert torch.cuda.current_stream(device).query(), (
            "a pageable host->device copy is expected to synchronise the stream; "
            "if this fails the platform changed and the staging discipline needs "
            "re-measuring rather than re-asserting"
        )
        torch.cuda.synchronize(device)

    def test_cpu_device_takes_the_plain_path(self):
        staged = to_device([4, 5], dtype=torch.int32, device="cpu")
        assert staged.device.type == "cpu"
        assert not staged.is_pinned()
        assert staged.tolist() == [4, 5]

    def test_empty_sequence(self, device):
        staged = to_device([], dtype=torch.long, device=device)
        assert staged.numel() == 0
        assert staged.device.type == "cuda"
