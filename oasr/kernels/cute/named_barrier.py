# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Named-barrier IDs for the OASR attention kernels.

Notes
-----
SM80 / SM120 only need a single CTA-wide barrier between the smem-write
producers (cp.async) and the smem-read consumers (ldmatrix + mma). The
warp-specialized producer/consumer barriers from FlashAttention's
``named_barrier.py`` (WG1-3, PFull/PEmpty queue, 2CTA/MLA) are SM90+
machinery and are deliberately not ported.

PTX `bar.sync` IDs are 0..15 globally per CTA; ID 0 is the implicit
syncthreads(). We reserve ID 1 for the attention pipeline.
"""

import cutlass.pipeline as pipeline


class NamedBarrierFwd:
    """Named-barrier IDs for the FMHA forward kernel."""

    Epilogue = 1  # producer/consumer rendezvous around the Q/K/V loads.


def make_cta_sync_barrier(num_threads: int) -> pipeline.NamedBarrier:
    """Build the single CTA-wide barrier the FMHA forward kernel uses."""
    return pipeline.NamedBarrier(
        barrier_id=NamedBarrierFwd.Epilogue,
        num_threads=num_threads,
    )


__all__ = ["NamedBarrierFwd", "make_cta_sync_barrier"]
