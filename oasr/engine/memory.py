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

An AR family's prefill is measured separately (:data:`PREFILL_PROBE_ROWS` rows,
the decoder KV it writes excluded) and charged per admitted row: the decoder-KV
budget is narrowed until the widest prefill its rows allow fits in the reserve
too.  That transient is one LM forward over every row's whole prompt, so it grows
with the rows admitted together, and an encoder-only probe does not see it.  It
used to land in the utilization headroom.  On Qwen2-Audio with int8 weights,
quantization freed 6.6 GiB, and the budget admitted 26 rows instead of 15.  Their
2.9 GiB prefill no longer fitted.  On LibriSpeech test-clean at
``max_batch_size=32``, the int8 and fp8 runs had 133 prefills rejected for OOM
between them.

CUDA-graph capture pools are allocated after the probes.  The offline and AR
ones land inside the utilization headroom, which is why the default leaves 10%
of the card unspent.  The streaming pre-warm is priced up front
(:func:`streaming_graph_reserve_bytes`), because its pool grows with the slot
count while the headroom does not: at ``max_batch_size=512`` U2++'s pre-warm
needed 2.3 GiB, the derived pool left it 0.6, and a capture failed mid-way with
the card full.
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

#: Rows the prefill probe runs.  The transient is linear in the rows: measured
#: on Qwen2-Audio (int8), 112.8-113.8 MiB per row at every width from 1 to 16.
#: So two rows measure it, and the probe's own dense decoder KV stays small.
PREFILL_PROBE_ROWS = 2

_GIB = float(1024**3)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def configure_allocator(device: torch.device, expandable_segments: bool) -> bool:
    """Turn on the caching allocator's expandable segments; ``True`` if applied.

    The caching allocator keeps every segment it ever mapped, and a segment with
    one live block cannot be returned.  Serving batches of many different
    ``(B, T)`` shapes, it ends up holding 3.7-4.7x the memory the engine uses
    (measured peak reserved vs allocated: U2++ 22.5 vs 6.0 GiB, Zipformer 29.5
    vs 6.4, Nemotron 30.3 vs 11.0).  That is memory nothing else can have, and
    the offline graph cache reads free VRAM to decide whether a capture fits,
    so the fragmentation alone locked Nemotron's cache: 535 refused captures,
    every small batch eager.  Expandable segments map and unmap pages inside
    one growing segment instead; measured with the same traffic, Nemotron's
    peak reserved fell 30.4 -> 20.5 GiB and its refusals 535 -> 3.

    Process-wide and one-way: it applies to segments mapped after the call.  A
    ``PYTORCH_CUDA_ALLOC_CONF`` that already names ``expandable_segments`` is
    the operator's choice and wins either way.
    """
    if not expandable_segments or device.type != "cuda":
        return False
    import os

    if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return False
    try:
        setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
        if setter is None:  # torch before the accelerator-generic API
            setter = torch.cuda.memory._set_allocator_settings
        setter("expandable_segments:True")
    except Exception:  # pragma: no cover - older torch, or an allocator that refuses
        logger.warning("could not enable expandable allocator segments", exc_info=True)
        return False
    return True


#: Intra-op threads the engine caps torch at when the operator set none.
DEFAULT_TORCH_THREADS = 8


def configure_host_threads(requested: Optional[int]) -> Optional[int]:
    """Set torch's intra-op thread count; return what was set, or ``None``.

    An explicit ``requested`` always applies.  Otherwise torch's default -- one
    thread per core -- is capped at :data:`DEFAULT_TORCH_THREADS`, unless the
    operator chose through ``OMP_NUM_THREADS``.  The engine's CPU tensor work is
    small and latency-bound (a batch-1 VAD LSTM, a few-MB pack into pinned
    staging), and fanning it out over 64 threads made host-bound runs about 14x
    slower whenever other tenants loaded the host, against no gain on an idle one.
    """
    import os

    if requested is not None:
        n = int(requested)
    elif "OMP_NUM_THREADS" in os.environ or torch.get_num_threads() <= DEFAULT_TORCH_THREADS:
        return None
    else:
        n = DEFAULT_TORCH_THREADS
    torch.set_num_threads(n)
    return n


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


#: Transient bytes of the streaming capture pool, in measured activation peaks.
#: Measured on U2++ with the padding lane (fp16; pool minus retained inputs and
#: outputs): captured narrowest-first, the pool kept 0.50 GiB at 32 slots, 0.64
#: at 128 and 0.80 at 512 (a 0.42 GiB peak), because every wider capture had to
#: grow it; widest-first, as the pre-warm now runs, 0.11 GiB at 512.  Two peaks
#: and the floor are what a capture taken lazily, out of that order, can cost.
GRAPH_TRANSIENT_FACTOR = 2.0
GRAPH_TRANSIENT_FLOOR_BYTES = 768 * 1024**2


def streaming_graph_reserve_bytes(
    *, io_bytes_per_row: int, capture_rows: int, activation_bytes: int
) -> int:
    """Bytes the streaming pre-warm's graph pool will hold.

    Every capture keeps its own input and output buffers alive -- a ``(B, chunk,
    V)`` log-prob buffer is 160 KiB a row for U2++ -- so that part is the
    per-row I/O times the rows the pre-warm captures (the sum of its widths
    over every cache rung).  The intermediates are shared by the pool: about
    :data:`GRAPH_TRANSIENT_FACTOR` activation peaks, never below
    :data:`GRAPH_TRANSIENT_FLOOR_BYTES`.  Measured U2++ pools (fp16, the delta
    over the same engine with graphs off): 0.33 / 1.17 / 2.27 GiB at 32 / 128 /
    512 slots captured narrowest-first, 1.57 GiB at 512 widest-first.  The
    engine prices the rows at a stream's block-table ceiling, an upper bound:
    at 512 slots on a 32 GiB card it reserved 3.45 GiB, every one of the 140
    pre-warm shapes captured, and 2.3 GiB stayed free (before: a capture ran
    out of memory with 0.01 GiB left).
    """
    retained = max(0, int(io_bytes_per_row)) * max(0, int(capture_rows))
    transient = max(
        GRAPH_TRANSIENT_FLOOR_BYTES, int(GRAPH_TRANSIENT_FACTOR * max(0, int(activation_bytes)))
    )
    return retained + transient


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
    reserve_bytes: int = 0,
) -> PoolSizing:
    """Size the paged-KV pool to fit ``profile.available_bytes - reserve_bytes``.

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
    reserve_bytes : int
        Withheld for something allocated after the pool that the profile does
        not see -- the streaming graph pre-warm
        (:func:`streaming_graph_reserve_bytes`).

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
    spendable = max(0, profile.available_bytes - max(0, int(reserve_bytes)))
    affordable = spendable // bytes_per_block
    if affordable < min_blocks:
        reserved = (
            f" after a {reserve_bytes / _GIB:.2f}GiB CUDA-graph reserve"
            if reserve_bytes > 0
            else ""
        )
        raise ValueError(
            "cannot size the paged KV cache from VRAM: this configuration needs "
            f"at least {min_blocks} blocks "
            f"({min_blocks * bytes_per_block / _GIB:.2f}GiB) but only "
            f"{affordable} blocks fit ({spendable / _GIB:.2f}GiB "
            f"available{reserved}). Profile: {profile.describe()}. Lower max_batch_size, "
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
    #: Measured prefill transient per row, when the probe ran.
    prefill_bytes_per_row: Optional[int] = None
    #: ``True`` when the prefill reserve, not the encoder's, set the budget.
    limited_by_prefill: bool = False

    def describe(self) -> str:
        rows = "unknown" if self.rows is None else str(self.rows)
        note = " (clamped: one row exceeds the available VRAM)" if self.clamped_to_one_row else ""
        if self.prefill_bytes_per_row is not None:
            note += (
                f"; prefill {self.prefill_bytes_per_row / 1024**2:.0f}MiB/row reserved"
                f"{' (binds)' if self.limited_by_prefill else ''}"
            )
        return f"{self.gib:.2f}GiB ≈ {rows} in-flight rows{note}"


def derive_decode_kv_budget(
    profile: MemoryProfile,
    *,
    bytes_per_row: Optional[int] = None,
    prefill_bytes_per_row: Optional[int] = None,
) -> KvBudget:
    """Size the in-flight decoder-KV ceiling to ``profile.available_bytes``.

    ``bytes_per_row`` is the strategy's worst-case footprint for one row
    (``DecodeStrategy.kv_bytes_per_row``); it refines the reported row count and
    the one-row clamp.  The clamp exists so a tight card still admits work: a
    budget below one row would reject every request, which is worse than
    admitting one and letting the allocator be the judge.

    ``prefill_bytes_per_row`` is the measured prefill transient per row.  With
    both per-row figures known, the budget also has to leave room for the
    prefill of every row it admits, since admission can prefill them together.
    For ``R`` rows of ``k`` bytes with a prefill of ``p`` bytes each:

        R * k + ACTIVATION_SAFETY * max(encoder peak, R * p) <= budget_bytes

    The encoder half is ``available_bytes``.  The prefill half caps the decoder
    KV at ``budget_bytes * k / (k + ACTIVATION_SAFETY * p)``.  The two transients
    are taken by ``max`` because they do not overlap: an encoder pass's
    activations are freed before its micro-batch prefills.
    """
    available = profile.available_bytes
    limited_by_prefill = False
    if bytes_per_row and prefill_bytes_per_row:
        k, p = int(bytes_per_row), int(prefill_bytes_per_row)
        by_prefill = int(profile.budget_bytes * k / (k + ACTIVATION_SAFETY * p))
        if by_prefill < available:
            available = by_prefill
            limited_by_prefill = True
    clamped = False
    if bytes_per_row and available < bytes_per_row:
        available = int(bytes_per_row)
        clamped = True
    rows = int(available // bytes_per_row) if bytes_per_row else None
    return KvBudget(
        gib=available / _GIB,
        rows=rows,
        clamped_to_one_row=clamped,
        prefill_bytes_per_row=None if prefill_bytes_per_row is None else int(prefill_bytes_per_row),
        limited_by_prefill=limited_by_prefill,
    )
