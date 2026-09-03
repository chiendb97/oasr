# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""A swallowed CUDA-graph capture failure must not poison the process's RNG.

Capture is best-effort everywhere in ``oasr/engine``: a forward that reads a
device value host-side invalidates the capture stream, and the caches answer by
remembering the shape and running eager.  What the swallow does not undo is that
``torch.cuda.graph.__enter__`` put the device's default generator into capture
mode and ``capture_end`` — which raises alongside the body — never took it out.
The generator then stays capturing for the life of the process and every CUDA
RNG call fails, which in this engine is ``torch.multinomial`` in
``oasr/engine/generation/sampling.py``: one unrelated capture failure turns every
sampled decode into an error, far from the edit that caused it.

These tests fail against the pre-fix caches: drop the
``restore_rng_after_failed_capture`` calls and both classes below raise
``RuntimeError: Offset increment outside graph capture encountered
unexpectedly``.
"""

from __future__ import annotations

import gc

import pytest
import torch

from oasr.engine.capture_recovery import (
    recover_from_failed_capture,
    restore_rng_after_failed_capture,
)
from oasr.engine.offline_graph import FUSED, GraphedOfflineForward

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _abort_a_capture(device: torch.device, pool=None) -> None:
    """Fail a capture the way a host read inside a forward does."""
    x = torch.ones(4, 4, device=device)
    graph = torch.cuda.CUDAGraph()
    try:
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            _ = int((x * 2).sum().item())
    except Exception:
        pass
    torch.cuda.synchronize(device)


def _free_gib(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.mem_get_info(device)[0] / 2**30


def _cuda_rng_works(device: torch.device) -> bool:
    try:
        torch.randn(4, 4, device=device)
        return True
    except RuntimeError:
        return False


class _SyncingEncoder(torch.nn.Module):
    """Not capturable: the host read invalidates the capture stream."""

    def __init__(self, feat_dim=16, hidden=32, sub=2):
        super().__init__()
        self.proj = torch.nn.Linear(feat_dim, hidden)
        self.sub = sub

    def forward_offline(self, features, lengths):
        out = torch.tanh(self.proj(features))[:, :: self.sub]
        assert out.size(1) >= int(lengths.max().item()) // self.sub
        return out, (lengths // self.sub).to(torch.int32)


@cuda_only
class TestTheHazardIsReal:
    """Without a reset, an aborted capture breaks RNG for the whole process."""

    def test_an_aborted_capture_poisons_cuda_rng(self):
        dev = torch.device("cuda")
        _abort_a_capture(dev)
        assert not _cuda_rng_works(dev), (
            "torch no longer leaks capture state on an aborted capture — "
            "restore_rng_after_failed_capture and these tests can go"
        )
        assert restore_rng_after_failed_capture(dev)
        assert _cuda_rng_works(dev)

    @pytest.mark.parametrize("fix", [torch.cuda.manual_seed, None])
    def test_the_obvious_resets_do_not_clear_it(self, fix):
        """Why the recovery is a throwaway capture and not a seed call."""
        dev = torch.device("cuda")
        _abort_a_capture(dev)
        gen = torch.cuda.default_generators[dev.index or 0]
        try:
            fix(0) if fix is not None else gen.set_state(gen.get_state())
        except RuntimeError:
            pass  # set_state itself asserts "not capturing" on some builds
        assert not _cuda_rng_works(dev)
        assert restore_rng_after_failed_capture(dev)


@cuda_only
class TestTheCachesRepairWhatTheySwallow:
    def test_a_failed_offline_capture_leaves_rng_usable(self):
        """``GraphedOfflineForward`` returns ``None`` and the process is intact."""
        dev = torch.device("cuda")
        assert restore_rng_after_failed_capture(dev)  # start from a clean slate
        enc = _SyncingEncoder().to(dev).eval()
        cache = GraphedOfflineForward(device=dev, frame_granularity=64, batch_buckets=[2])
        feats = torch.randn(2, 64, 16, device=dev)
        lens = torch.tensor([64, 50], device=dev, dtype=torch.int32)

        with torch.no_grad():
            assert cache.run(FUSED, enc.forward_offline, feats, lens) is None
        assert cache.fallback_failed == 1

        assert _cuda_rng_works(dev), "the swallowed capture left the generator capturing"

    def test_sampled_decoding_still_works_after_a_failed_capture(self):
        """The concrete consequence: ``select_next_tokens`` draws on the GPU."""
        from oasr.engine.generation.sampling import select_next_tokens
        from oasr.engine.request import DecodingOptions

        dev = torch.device("cuda")
        assert restore_rng_after_failed_capture(dev)
        enc = _SyncingEncoder().to(dev).eval()
        cache = GraphedOfflineForward(device=dev, frame_granularity=64, batch_buckets=[2])
        with torch.no_grad():
            cache.run(
                FUSED,
                enc.forward_offline,
                torch.randn(2, 64, 16, device=dev),
                torch.tensor([64, 50], device=dev, dtype=torch.int32),
            )

        opts = DecodingOptions(temperature=1.0, top_p=0.9)
        assert opts.sampling, "this row must take the multinomial path or nothing is tested"
        tokens = select_next_tokens(torch.randn(1, 32, device=dev), [opts])
        assert tokens.shape == (1,) and 0 <= int(tokens[0]) < 32


@cuda_only
class TestTheAllocatorHalf:
    """The half a throwaway capture cannot fix, and the one that costs GiB.

    ``__enter__`` called ``beginAllocateToPool``; when the body raises, the
    matching ``endAllocateToPool`` never runs and the allocator keeps serving
    *every* later allocation in the process out of that private pool.  Nothing
    gives those blocks back -- not ``del``, not ``gc``, not ``empty_cache`` --
    which is how an uncapturable encoder stranded 3.2 GiB per engine.
    """

    def test_allocations_after_an_abort_are_stranded(self):
        dev = torch.device("cuda")
        pool = torch.cuda.graph_pool_handle()
        _abort_a_capture(dev, pool)

        before = _free_gib(dev)
        block = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        del block
        gc.collect()
        torch.cuda.empty_cache()
        stranded = before - _free_gib(dev)
        assert stranded > 0.2, (
            "torch no longer strands post-abort allocations "
            f"(only {stranded:.3f} GiB); capture_recovery can be simplified"
        )

        assert recover_from_failed_capture(dev, pool)
        before = _free_gib(dev)
        block = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        del block
        gc.collect()
        torch.cuda.empty_cache()
        assert before - _free_gib(dev) < 0.05, "the allocator is still bound to the pool"

    def test_the_rng_reset_is_not_a_substitute_for_releasing_the_pool(self):
        """Why ``recover_from_failed_capture`` releases *first*, then resets RNG.

        The throwaway capture ends the diversion as a side effect of its own
        ``__exit__``, so allocations made *after* it behave normally.  What it
        cannot do is hand back what the process already put in the stuck pool —
        and by then the ordering is fixed: releasing afterwards no longer
        recovers those blocks.  An engine allocates its KV pool right after the
        abort, so this is the difference between 0.0 and 3.2 GiB.
        """
        dev = torch.device("cuda")
        pool = torch.cuda.graph_pool_handle()
        before = _free_gib(dev)
        _abort_a_capture(dev, pool)
        block = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)

        assert restore_rng_after_failed_capture(dev)
        del block
        gc.collect()
        torch.cuda.empty_cache()
        assert before - _free_gib(dev) > 0.2, (
            "the throwaway capture returned blocks already in the pool; "
            "the explicit release may be redundant"
        )

    def test_releasing_first_keeps_the_bytes(self):
        """The shipped order, against the same abort."""
        dev = torch.device("cuda")
        pool = torch.cuda.graph_pool_handle()
        before = _free_gib(dev)
        _abort_a_capture(dev, pool)
        assert recover_from_failed_capture(dev, pool)

        block = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
        del block
        gc.collect()
        torch.cuda.empty_cache()
        assert before - _free_gib(dev) < 0.05, "recovery did not restore normal allocation"


@cuda_only
class TestAnUncapturableForwardCostsNothingLasting:
    """End to end: the cache declines, and the engine's VRAM comes back.

    Before the recovery this was 3.24 GiB per Nemotron engine, and six engines
    into the accuracy suite the 7B checkpoint could no longer size its decoder
    KV pool.
    """

    def test_the_engine_returns_its_memory(self):
        dev = torch.device("cuda")
        assert recover_from_failed_capture(dev, None)
        enc = _SyncingEncoder().to(dev).eval()
        before = _free_gib(dev)

        cache = GraphedOfflineForward(device=dev, frame_granularity=64, batch_buckets=[1, 8])
        with torch.no_grad():
            assert (
                cache.run(
                    FUSED,
                    enc.forward_offline,
                    torch.randn(8, 256, 16, device=dev),
                    torch.full((8,), 256, device=dev, dtype=torch.int32),
                )
                is None
            )
        assert not cache.enabled, "an uncapturable forward must switch the cache off"
        cache.release()
        del cache, enc
        gc.collect()
        torch.cuda.empty_cache()
        assert before - _free_gib(dev) < 0.05, "the failed capture stranded memory"
