# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""VRAM-aware capacity sizing (architecture review H4).

Two engine capacities are memory, not compute: the streaming paged-KV pool
(``EngineConfig.max_num_blocks``) and the ceiling on in-flight decoder KV for
the autoregressive families (``EngineConfig.decode_kv_budget_gib``).  Both were
plain hardcoded numbers, so the operator had to hand-compute them from layers x
heads x head_dim x dtype and either wasted VRAM or hit the crash path (an
undersized pool raises ``BlockPool exhausted`` inside the encoder forward; an
oversized one OOMs at startup).  One config could not move between a 24 GB and
an 80 GB card.

This module derives both from one measurement:

    available = total * gpu_memory_utilization - resident - activation_reserve

``resident`` is everything already on the device when the profile is taken —
model weights, the CUDA context, whatever another process holds — read from the
driver rather than assumed, which is why it needs no separate "weights" term.
``activation_reserve`` comes from a *measured* probe forward at the widest shape
the engine will run (see :func:`measure_peak_activation`), scaled by
:data:`ACTIVATION_SAFETY`, and the utilization factor is the headroom on top.

The arithmetic is deliberately separated from the torch probes: everything below
:class:`MemoryProfile` is pure integer maths and is unit-tested on CPU, so the
formula is pinned without a GPU.

What the probe does **not** cover: an AR family's prefill transient (the audio
tower plus one LM forward over the whole prompt) is not run by the probe, and
CUDA-graph capture pools are allocated after it.  Both land inside the
utilization headroom, which is why the default leaves 10% of the card unspent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

#: Headroom above the probe peak for workspaces, staging, and graph captures.
ACTIVATION_SAFETY = 1.5

#: Floor on the activation reserve.  A tiny model measures a tiny peak, and the
#: fixed costs above it do not shrink with the model.
MIN_ACTIVATION_RESERVE_BYTES = 256 * 1024**2

#: Reserve when the probe could not run at all (it raised).  A fraction of the
#: budget rather than a byte count, since a card big enough to matter also has
#: proportionally larger transients.
UNMEASURED_ACTIVATION_FRACTION = 0.25

#: Minimum blocks per stream a derived pool must afford under *unlimited*
#: history (``num_left_chunks < 0``), where any pool size is technically legal:
#: a stream capped at fewer blocks than this keeps so little attention context
#: that the transcript degrades silently.  8 blocks x 16 frames = 128 encoder
#: frames, about 5 s of audio at the common 4x-subsampling / 10 ms-hop geometry.
MIN_BLOCKS_PER_STREAM = 8

#: Audio length the offline probe forward uses when the frontend has no fixed
#: window (the Kaldi frontends, whose cost tracks the real utterance length).
PROBE_AUDIO_SECONDS = 30.0

_GIB = float(1024**3)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def read_device_memory(device: torch.device) -> "tuple[int, int]":
    """``(free, total)`` device bytes as the *driver* sees them.

    Releases the caching allocator's unused blocks first: torch keeps freed
    device memory reserved, so ``mem_get_info`` would otherwise report a probe
    forward's transient as permanently gone and the derivation would give away
    memory it actually has.
    """
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    return int(free), int(total)


def measure_peak_activation(fn: Callable[[], object], device: torch.device) -> int:
    """Peak *transient* bytes one call to ``fn`` allocates.

    Measured as the high-water mark of the caching allocator above the
    already-resident baseline, so persistent allocations made before the call
    (model weights) are excluded and anything ``fn`` leaves behind is included.
    """
    torch.cuda.synchronize(device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    return max(0, peak - baseline)


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryProfile:
    """What one device has left to spend, and on what terms.

    Parameters
    ----------
    total_bytes, free_bytes : int
        As reported by the driver (:func:`read_device_memory`) *after* the model
        is resident and the probe's transients have been released.
    activation_bytes : int
        Measured peak transient of one representative forward, or ``0`` when
        ``activation_measured`` is ``False``.
    utilization : float
        Fraction of the card the engine may occupy in total, weights included —
        ``EngineConfig.gpu_memory_utilization``.  The unspent remainder is the
        headroom for everything this profile cannot see.
    activation_measured : bool
        ``False`` when the probe forward failed; the reserve then falls back to
        :data:`UNMEASURED_ACTIVATION_FRACTION` of the budget.
    """

    total_bytes: int
    free_bytes: int
    activation_bytes: int
    utilization: float
    activation_measured: bool = True

    @property
    def resident_bytes(self) -> int:
        """Bytes already on the device: weights, context, other processes."""
        return max(0, self.total_bytes - self.free_bytes)

    @property
    def cap_bytes(self) -> int:
        """Total bytes the engine may occupy under ``utilization``."""
        return int(self.total_bytes * self.utilization)

    @property
    def budget_bytes(self) -> int:
        """Spendable bytes under the cap, before the activation reserve."""
        return max(0, self.cap_bytes - self.resident_bytes)

    @property
    def activation_reserve_bytes(self) -> int:
        """Bytes withheld for transients (activations, workspaces, graph pools)."""
        if not self.activation_measured:
            return max(
                MIN_ACTIVATION_RESERVE_BYTES,
                int(self.budget_bytes * UNMEASURED_ACTIVATION_FRACTION),
            )
        return max(
            MIN_ACTIVATION_RESERVE_BYTES,
            int(self.activation_bytes * ACTIVATION_SAFETY),
        )

    @property
    def available_bytes(self) -> int:
        """Bytes a cache may claim: the budget less the activation reserve."""
        return max(0, self.budget_bytes - self.activation_reserve_bytes)

    def describe(self) -> str:
        """One-line derivation trace — logged so the numbers are auditable."""
        return (
            f"total={self.total_bytes / _GIB:.2f}GiB "
            f"resident={self.resident_bytes / _GIB:.2f}GiB "
            f"cap={self.cap_bytes / _GIB:.2f}GiB (utilization={self.utilization:.2f}) "
            f"activation_reserve={self.activation_reserve_bytes / _GIB:.2f}GiB"
            f"{'' if self.activation_measured else ' (unmeasured — probe failed)'} "
            f"→ available={self.available_bytes / _GIB:.2f}GiB"
        )


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def bytes_per_kv_block(
    *,
    num_layers: int,
    block_size_frames: int,
    n_kv_head: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Device bytes one paged KV block costs across all layers.

    Mirrors :class:`~oasr.cache.block_pool.BlockPool`, which allocates K and V
    as two ``(num_layers, max_num_blocks, block_size_frames, n_kv_head,
    head_dim)`` tensors — hence the factor 2 and the ``num_layers`` term (one
    logical block occupies a slab in *every* layer).
    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    return int(2 * num_layers * block_size_frames * n_kv_head * head_dim * itemsize)


@dataclass(frozen=True)
class PoolSizing:
    """Result of :func:`derive_pool_blocks`."""

    blocks: int
    bytes_per_block: int
    #: ``"vram"`` — memory was the binding constraint; ``"block_table"`` — the
    #: card could afford more blocks than a full pool of streams can address.
    limited_by: str

    @property
    def pool_bytes(self) -> int:
        return self.blocks * self.bytes_per_block

    def describe(self) -> str:
        return (
            f"{self.blocks} blocks x {self.bytes_per_block / (1024 ** 2):.2f}MiB = "
            f"{self.pool_bytes / _GIB:.2f}GiB (limited by {self.limited_by})"
        )


def derive_pool_blocks(
    profile: MemoryProfile,
    bytes_per_block: int,
    *,
    min_blocks: int,
    max_blocks: int,
) -> PoolSizing:
    """Size the paged-KV pool to fit ``profile.available_bytes``.

    Parameters
    ----------
    min_blocks : int
        Fewest blocks that make the configuration viable — the eviction
        invariant (``max_batch_size * blocks_per_stream``) when history is
        bounded, else :data:`MIN_BLOCKS_PER_STREAM` per stream.
    max_blocks : int
        Most blocks a full pool of streams can address
        (``max_batch_size * blocks_per_seq``).  Beyond this the memory is
        allocated and never handed out, so the pool is capped here even when the
        card could afford more.

    Raises
    ------
    ValueError
        When even ``min_blocks`` does not fit.  Deriving a smaller pool would
        either OOM at allocation or silently degrade every transcript, so this
        fails at startup with the arithmetic and the levers attached.
    """
    if bytes_per_block <= 0:
        raise ValueError(f"bytes_per_block must be positive, got {bytes_per_block}")
    if min_blocks > max_blocks:
        raise ValueError(
            f"min_blocks ({min_blocks}) exceeds max_blocks ({max_blocks}); the "
            "retained history cannot be addressed by the block table"
        )
    affordable = profile.available_bytes // bytes_per_block
    if affordable < min_blocks:
        raise ValueError(
            "cannot size the paged KV cache from VRAM: this configuration needs "
            f"at least {min_blocks} blocks "
            f"({min_blocks * bytes_per_block / _GIB:.2f}GiB) but only "
            f"{affordable} blocks fit ({profile.available_bytes / _GIB:.2f}GiB "
            f"available). Profile: {profile.describe()}. Lower max_batch_size, "
            "shorten the retained history (num_left_chunks), raise "
            "gpu_memory_utilization, or set max_num_blocks explicitly to take "
            "responsibility for the size."
        )
    if affordable >= max_blocks:
        return PoolSizing(int(max_blocks), bytes_per_block, "block_table")
    return PoolSizing(int(affordable), bytes_per_block, "vram")


@dataclass(frozen=True)
class KvBudget:
    """Result of :func:`derive_decode_kv_budget`."""

    gib: float
    #: Rows the budget affords, when the family declares a per-row footprint.
    rows: Optional[int]
    #: ``True`` when one row did not fit and the budget was raised to hold it.
    clamped_to_one_row: bool

    def describe(self) -> str:
        rows = "unknown" if self.rows is None else str(self.rows)
        note = " (clamped: one row exceeds the available VRAM)" if self.clamped_to_one_row else ""
        return f"{self.gib:.2f}GiB ≈ {rows} in-flight rows{note}"


def derive_decode_kv_budget(
    profile: MemoryProfile,
    *,
    bytes_per_row: Optional[int] = None,
) -> KvBudget:
    """Size the in-flight decoder-KV ceiling to ``profile.available_bytes``.

    ``bytes_per_row`` is the strategy's worst-case footprint for one row
    (``DecodeStrategy.kv_bytes_per_row``); it only refines the reported row count
    and the one-row clamp.  The clamp exists so a tight card still admits work:
    a budget below one row would reject every request, which is worse than
    admitting one and letting the allocator be the judge.
    """
    available = profile.available_bytes
    clamped = False
    if bytes_per_row and available < bytes_per_row:
        available = int(bytes_per_row)
        clamped = True
    rows = int(available // bytes_per_row) if bytes_per_row else None
    return KvBudget(gib=available / _GIB, rows=rows, clamped_to_one_row=clamped)
