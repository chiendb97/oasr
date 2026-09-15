# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Where a tuned routing table's numbers came from — and whether they came from here.

Two kinds of routing decision live in ``oasr/jit``, and only one of them travels.

*Derived* decisions read the machine and do arithmetic against it:
:func:`oasr.jit.mlp.gated_mlp_ctas_per_sm` takes ``multi_processor_count``, the
real opt-in shared memory and ``max_threads_per_multi_processor`` and computes
wave counts; ``selectBmmTile`` takes ``getDeviceMultiProcessorCount()``. Move
the binary to another card and the answer moves with it.

*Measured* decisions are cut-offs somebody timed on one GPU and wrote down —
``_LSTM_BANDS``, the ``_TILES`` ranking, the gated-MLP candidate list. They are
not wrong, and they are not guesses; they are the best available estimate. But
the crossover they encode is a function of SM count and memory bandwidth, and
an A30 (56 SMs, 933 GB/s), an A100 (108, 1555), an L40S (142, 864) and an RTX
5090 (170, 1792) put it in four different places. Applied unchanged on a card
nobody measured, such a table is an **extrapolation**.

Identity is ``(compute capability, SM count)``, both queried exactly. Bandwidth
is the other half of what moves a band and is deliberately *not* computed —
see :class:`Machine`.

This module makes that word appear. It does not change any routing decision: the
band still applies, because the *shape* of the curve is a property of the
algorithm (a weight-streaming kernel wins at small batch, a library GEMM's
mainloop wins at large) even though the crossover between them is a property of
the machine. What it adds is that an extrapolation says so once, is counted, and
shows up in ``oasr.layers.format_gap_report()`` beside the other things this
repo declares rather than hides.

Re-measuring is the fix, one card at a time; each record names the note its
numbers came from so the next person starts from the protocol rather than from
scratch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger("oasr.jit.measured")

__all__ = [
    "Machine",
    "MeasuredOn",
    "current_machine",
    "extrapolations",
    "is_native",
    "note_extrapolation",
    "reset_extrapolations",
]


@dataclass(frozen=True)
class Machine:
    """A GPU, identified by what can be *queried exactly*.

    Compute capability and SM count, and nothing else.  Memory bandwidth belongs
    in this comparison on the merits — it is half of what moves a band — and is
    deliberately absent, because it cannot be read reliably: the obvious
    ``2 * memory_clock_rate * bus_width / 8`` is right for HBM and GDDR6 and
    **wrong for GDDR7**, whose PAM3 signalling puts the RTX 50-series at well
    over twice its reported clock.  A formula that under-reports the one card
    these tables were measured on would declare an extrapolation *on the machine
    that was measured*, which is worse than not claiming the number at all.  The
    spec figure lives in :attr:`MeasuredOn.bandwidth` as prose, for whoever
    re-measures.

    ``(sm, sms)` separates every card in scope anyway: A30 (80, 56), A100
    (80, 108), L40S (89, 142), RTX 5090 (120, 170).
    """

    name: str
    sm: int
    sms: int

    def __str__(self) -> str:
        return f"{self.name} (sm_{self.sm}, {self.sms} SMs)"


@dataclass(frozen=True)
class MeasuredOn:
    """Provenance for one tuned table."""

    #: The table(s) this record covers, spelled as they appear in the source.
    table: str
    #: The GPU the numbers were taken on.
    machine: Machine
    #: Its memory bandwidth, as prose — documentation, not an identity field.
    bandwidth: str
    #: The note holding the protocol and the raw timings.
    source: str
    #: What moves the crossover — read by whoever re-measures.
    moves_with: str


def current_machine() -> Optional[Machine]:
    """This box's :class:`Machine`, or ``None`` without a usable CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
    except Exception:  # noqa: BLE001 — diagnostics must never break the caller
        return None
    return Machine(
        name=props.name,
        sm=props.major * 10 + props.minor,
        sms=props.multi_processor_count,
    )


def is_native(record: MeasuredOn) -> bool:
    """Was *record* measured on the GPU this process is running on?

    Compared on ``(sm, sms)`` rather than the marketing name: two cards can share
    a name across refreshes, and an unrecognised device with the same capability
    and SM count is, for this purpose, the machine that was measured.
    """
    here = current_machine()
    if here is None:
        return False
    there = record.machine
    return (here.sm, here.sms) == (there.sm, there.sms)


#: ``table -> one-line description of the gap``.  One entry per table, not per
#: call: an extrapolated band is a single fact about this process, and recording
#: it per shape would bury it under the shapes it covers.
_EXTRAPOLATIONS: Dict[str, str] = {}


def note_extrapolation(record: MeasuredOn) -> bool:
    """Record that *record*'s table is being applied off the card it was measured on.

    Returns ``True`` when this is an extrapolation.  Logs at most once per table
    per process — the caller is a routing gate, and a per-shape warning would be
    both noise and a lie about the cardinality of the fact.

    With no CUDA device this records nothing and returns ``False``: there is no
    machine to have extrapolated *onto*, and the CPU test job would otherwise
    report every table in the package as applied off its card.
    """
    here = current_machine()
    if here is None:
        return False
    if (here.sm, here.sms) == (record.machine.sm, record.machine.sms):
        return False
    if record.table in _EXTRAPOLATIONS:
        return True
    where = str(here)
    _EXTRAPOLATIONS[record.table] = (
        f"measured on {record.machine}, {record.bandwidth}; running on {where}"
    )
    logger.info(
        "%s was measured on %s and is being applied on %s. The band still holds "
        "its shape — a weight-streaming kernel wins at small batch — but the "
        "crossover moves with %s. Re-measure with the protocol in %s, or A/B it "
        "with the module's env switch.",
        record.table,
        record.machine,
        where,
        record.moves_with,
        record.source,
    )
    return True


def extrapolations() -> Dict[str, str]:
    """Tuned tables this process applied off the card they were measured on."""
    return dict(_EXTRAPOLATIONS)


def reset_extrapolations() -> None:
    """Clear the record (per-test isolation, per-benchmark accounting)."""
    _EXTRAPOLATIONS.clear()
