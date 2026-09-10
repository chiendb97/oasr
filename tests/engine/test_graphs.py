# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture across the engine: four caches, one failure mode.

``oasr/engine/{offline_graph,graph_cache,predictor_graph,capture_recovery}.py``
each capture a different forward, and they share what makes capture hazardous:
it is *best-effort*, so a forward that reads a device value host-side
invalidates the capture stream and the cache answers by falling back to eager.
The swallowed failure is the bug -- an aborted ``torch.cuda.graph`` capture
strands every later allocation in the process and breaks CUDA RNG, so "the
transcript is still fine" is not evidence that the recovery worked.

They were four files because they were four commits. They are one file
because ``_SyncingEncoder`` -- the deliberate capture-breaker -- was copied
verbatim into two of them, three of them declared their own alias for the same
CUDA marker, and a change to the recovery path touches all four.

The shape/bucketing arithmetic runs anywhere; everything that captures is
gated by the module-level ``cuda`` marker.
"""

from __future__ import annotations

import gc

import pytest
import torch

from oasr.engine.capture_recovery import (
    recover_from_failed_capture,
    restore_rng_after_failed_capture,
)
from oasr.engine.graph_cache import (
    CACHE_BUCKET_KNEE,
    cache_bucket_ladder,
    pick_cache_bucket,
    round_up_bucket,
)
from oasr.engine.offline_graph import (
    FUSED,
    GraphedOfflineForward,
    resolve_batch_buckets,
)
from oasr.engine.predictor_graph import PredictorStepGraphCache as Cache

# ---------------------------------------------------------------------------
# The offline forward: shape buckets, bit-exact replay, and fallback accounting
# ---------------------------------------------------------------------------


class _StubConfig:
    def __init__(self, max_batch_size=32, preferred=None, explicit=None):
        self.max_batch_size = max_batch_size
        self.preferred_batch_size = preferred
        self.offline_graph_batch_buckets = explicit


def _cache(**kw):
    kw.setdefault("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    kw.setdefault("batch_buckets", [1, 2, 4, 8])
    return GraphedOfflineForward(**kw)


class _Encoder(torch.nn.Module):
    """Small capturable stand-in with the engine's offline forward signature."""

    def __init__(self, feat_dim=16, hidden=32, sub=2):
        super().__init__()
        self.proj = torch.nn.Linear(feat_dim, hidden)
        self.out = torch.nn.Linear(hidden, hidden)
        self.sub = sub

    def forward_offline(self, features, lengths):
        x = torch.tanh(self.proj(features))[:, :: self.sub]
        return self.out(x), (lengths // self.sub).to(torch.int32)


class _PaddingSensitiveEncoder(_Encoder):
    """Trailing padding reaches the valid outputs, the way Zipformer's does.

    ``SimpleDownsample`` fills its last window by replicating the final frame,
    so how many frames follow the valid ones changes what that window averages.
    Measured on real weights the leak is ~2.5e-1 in bf16.  A padding-invariant
    stub cannot fail the consistency test below, so it would not be a test.
    """

    def forward_offline(self, features, lengths):
        t = features.size(1)
        pad = (-t) % 4
        if pad:
            features = torch.cat([features, features[:, -1:].expand(-1, pad, -1)], dim=1)
        x = torch.tanh(self.proj(features))
        x = x.reshape(x.size(0), -1, 4, x.size(2)).mean(dim=2)
        return self.out(x), (lengths // 4).to(torch.int32)


class _SyncingEncoder(_Encoder):
    """Reads a device value host-side, the way the Zipformer asserts used to.

    A ``.item()`` inside a capture region raises
    ``cudaErrorStreamCaptureInvalidated``; this is the shape of every forward the
    cache must refuse rather than keep re-attempting.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = 0

    def forward_offline(self, features, lengths):
        self.calls += 1
        out, out_lengths = super().forward_offline(features, lengths)
        assert out.size(1) >= int(lengths.max().item()) // self.sub
        return out, out_lengths


# ---------------------------------------------------------------------------
# Bucketing / configuration (no CUDA)
# ---------------------------------------------------------------------------


class TestShapeBuckets:
    def test_frame_bucket_rounds_up_to_granularity(self):
        c = _cache(frame_granularity=64)
        assert c.frame_bucket(1) == 64
        assert c.frame_bucket(64) == 64
        assert c.frame_bucket(65) == 128
        assert c.frame_bucket(960) == 960

    def test_granularity_one_is_an_exact_key(self):
        """What a fixed-window frontend gets: no rounding, so no padding."""
        c = _cache(frame_granularity=1)
        for t in (1, 3000, 3001):
            assert c.frame_bucket(t) == t

    def test_pick_batch_bucket_is_the_smallest_fit(self):
        c = _cache(batch_buckets=[1, 4, 16])
        assert c.pick_batch_bucket(1) == 1
        assert c.pick_batch_bucket(2) == 4
        assert c.pick_batch_bucket(16) == 16
        assert c.pick_batch_bucket(17) is None  # oversized -> eager
        assert c.pick_batch_bucket(0) is None


class TestResolveBatchBuckets:
    def test_explicit_wins(self):
        cfg = _StubConfig(preferred=[4, 8], explicit=[3, 5])
        assert resolve_batch_buckets(cfg) == [3, 5]

    def test_preferred_is_used_when_no_explicit(self):
        """The widths the partitioner already emits, so B-padding is zero."""
        assert resolve_batch_buckets(_StubConfig(preferred=[8, 16, 32])) == [8, 16, 32]

    def test_falls_back_to_powers_of_two(self):
        assert resolve_batch_buckets(_StubConfig(max_batch_size=16)) == [1, 2, 4, 8, 16]

    def test_non_power_of_two_cap_is_included(self):
        assert resolve_batch_buckets(_StubConfig(max_batch_size=24)) == [1, 2, 4, 8, 16, 24]


class TestDisabledCache:
    def test_cpu_device_disables(self):
        c = GraphedOfflineForward(device=torch.device("cpu"), batch_buckets=[1])
        assert not c.enabled
        f = torch.zeros(1, 10, 4)
        assert c.run(FUSED, lambda a, b: (a, b), f, torch.tensor([10])) is None
        assert c.pad_time(f, torch.tensor([10]))[0] is f  # unchanged, not padded


# ---------------------------------------------------------------------------
# Capture (CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCaptureIsBitExact:
    def test_replay_matches_eager_on_the_same_padded_input(self):
        """The core guarantee: capture changes launch count, never numerics."""
        torch.manual_seed(0)
        dev = torch.device("cuda")
        enc = _Encoder().to(dev).eval()
        c = _cache(frame_granularity=64, batch_buckets=[4])
        feats = torch.randn(4, 100, 16, device=dev)
        lens = torch.tensor([100, 90, 75, 60], device=dev, dtype=torch.int32)

        with torch.no_grad():
            padded, plens = c.pad_time(feats, lens)
            want, want_len = enc.forward_offline(padded, plens)
            got = c.run(FUSED, enc.forward_offline, feats, lens)

        assert got is not None
        assert torch.equal(got[0], want)
        assert torch.equal(got[1], want_len)

    def test_batch_padding_does_not_disturb_valid_rows(self):
        """B is padded up to a bucket; rows past B_active must stay inert.

        The oracle is eager at the **same** padded width, not at B=3: widening a
        GEMM's M changes which cuBLAS kernel runs, and that alone moves the valid
        rows by an ulp (measured 2.4e-7 in fp32 here) with no contamination at
        all.  Comparing against B=3 would fold that unavoidable kernel-selection
        difference into a test whose subject is the padding rows, so it would
        fail for a reason it does not name.  ``test_eager_fallback_matches_a_graph_hit``
        below is what pins the property that actually matters end to end.
        """
        torch.manual_seed(0)
        dev = torch.device("cuda")
        enc = _Encoder().to(dev).eval()
        c = _cache(frame_granularity=64, batch_buckets=[8])  # B=3 -> bucket 8
        feats = torch.randn(3, 128, 16, device=dev)
        lens = torch.tensor([128, 100, 64], device=dev, dtype=torch.int32)

        padded = torch.zeros(8, 128, 16, device=dev)
        padded[:3] = feats
        plens = torch.full((8,), 128, device=dev, dtype=torch.int32)
        plens[:3] = lens

        with torch.no_grad():
            want, want_len = enc.forward_offline(padded, plens)
            unpadded, _ = enc.forward_offline(feats, lens)
            got = c.run(FUSED, enc.forward_offline, feats, lens)

        assert got is not None
        assert got[0].size(0) == 3
        assert torch.equal(got[0], want[:3])
        assert torch.equal(got[1], want_len[:3])
        # And the B-bucketing tax is rounding, not contamination: if this ever
        # grows past an ulp, padding is reaching the valid rows for real.
        assert (want[:3] - unpadded).abs().max() < 1e-5

    def test_replays_are_stable_across_shapes(self):
        """A later capture must not corrupt an earlier replay's returned tensor."""
        torch.manual_seed(0)
        dev = torch.device("cuda")
        enc = _Encoder().to(dev).eval()
        c = _cache(frame_granularity=64, batch_buckets=[2, 4])
        a_feats = torch.randn(2, 64, 16, device=dev)
        a_lens = torch.tensor([64, 50], device=dev, dtype=torch.int32)

        with torch.no_grad():
            first = c.run(FUSED, enc.forward_offline, a_feats, a_lens)
            assert first is not None
            held = first[0].clone()
            # Force a *different* capture, which is what invalidates a
            # pool-backed view -- not just another replay at the same key.
            c.run(
                FUSED,
                enc.forward_offline,
                torch.randn(4, 192, 16, device=dev),
                torch.tensor([192, 150, 120, 100], device=dev, dtype=torch.int32),
            )

        assert torch.equal(first[0], held)


@pytest.mark.cuda
class TestPaddingConsistency:
    def test_eager_fallback_matches_a_graph_hit(self):
        """A saturated cache must not change the answer, only the speed.

        Without ``pad_time`` on the fallback the two paths see different padded
        widths, and an encoder that is not padding-invariant then decodes an
        utterance differently depending on whether its shape happened to be
        captured.
        """
        torch.manual_seed(0)
        dev = torch.device("cuda")
        enc = _PaddingSensitiveEncoder().to(dev).eval()
        # 98 is deliberately *not* a multiple of the stub's window: that is the
        # only case where the replicated tail differs from the bucket's zeros,
        # which is exactly Zipformer's mechanism.
        feats = torch.randn(2, 98, 16, device=dev)
        lens = torch.tensor([98, 80], device=dev, dtype=torch.int32)

        served = _cache(frame_granularity=64, batch_buckets=[2])
        starved = _cache(frame_granularity=64, batch_buckets=[1])
        with torch.no_grad():
            hit = served.run(FUSED, enc.forward_offline, feats, lens)
            # The fallback path, exactly as ModelRunner._offline runs it.
            miss = enc.forward_offline(*starved.pad_time(feats, lens))
            # ...and what it would have been without pad_time, which must differ
            # or this test could not fail.
            unpadded = enc.forward_offline(feats, lens)

        assert hit is not None
        assert starved.pick_batch_bucket(2) is None  # this shape is not served
        assert torch.equal(hit[0], miss[0])
        n = min(hit[0].size(1), unpadded[0].size(1))
        assert not torch.equal(
            hit[0][:, :n], unpadded[0][:, :n]
        ), "stub is padding-invariant, so the test cannot fail"

    def test_pad_time_is_a_noop_at_an_exact_bucket(self):
        dev = torch.device("cuda")
        c = _cache(frame_granularity=64)
        f = torch.zeros(1, 128, 16, device=dev)
        lens = torch.tensor([128], device=dev, dtype=torch.int32)
        assert c.pad_time(f, lens)[0] is f

    def test_pad_time_zero_fills_the_tail(self):
        dev = torch.device("cuda")
        c = _cache(frame_granularity=64)
        f = torch.ones(2, 100, 16, device=dev)
        padded, _ = c.pad_time(f, torch.tensor([100, 100], device=dev, dtype=torch.int32))
        assert padded.shape == (2, 128, 16)
        assert torch.equal(padded[:, :100], f)
        assert not padded[:, 100:].any()


@pytest.mark.cuda
class TestFallbackAccounting:
    def test_a_failed_capture_is_never_retried(self):
        """Retrying costs a warm-up forward per call and then runs eager anyway.

        The counter proves it: the second ``run`` must add **no** forward call of
        its own beyond the one the caller would make eagerly.

        An abort is also not a per-shape verdict.  What makes a forward
        uncapturable is a host read in its *code*, so the first abort turns the
        whole cache off rather than letting every later shape abort in turn --
        each stranding the memory its attempt allocated inside a pool nothing can
        reclaim.  The second call is therefore counted under ``fallback_disabled``
        rather than ``fallback_failed``; what must not change is that it costs no
        forward of its own.
        """
        torch.manual_seed(0)
        dev = torch.device("cuda")
        enc = _SyncingEncoder().to(dev).eval()
        c = _cache(frame_granularity=64, batch_buckets=[2])
        feats = torch.randn(2, 64, 16, device=dev)
        lens = torch.tensor([64, 50], device=dev, dtype=torch.int32)

        with torch.no_grad():
            assert c.run(FUSED, enc.forward_offline, feats, lens) is None
        after_first = enc.calls
        assert c.fallback_failed == 1
        assert not c.enabled, "an uncapturable forward must switch the cache off"

        with torch.no_grad():
            assert c.run(FUSED, enc.forward_offline, feats, lens) is None
        assert enc.calls == after_first, "second attempt re-ran the capture warm-up"
        assert c.fallback_disabled == 1
        assert c.captures == 0

    def test_the_capturability_probe_runs_at_the_narrowest_shape(self):
        """The expensive question is asked at B=1, not at the production width.

        An aborted capture strands whatever it allocated inside its private pool
        -- unreachable afterwards by ``reset``, ``gc`` or ``empty_cache`` -- so
        asking "is this capturable" at B=16 costs 16x the memory of asking at
        B=1, for the same answer.
        """
        torch.manual_seed(0)
        dev = torch.device("cuda")
        enc = _SyncingEncoder().to(dev).eval()
        seen = []
        inner = enc.forward_offline

        def spy(features, lengths):
            seen.append(tuple(features.shape))
            return inner(features, lengths)

        c = _cache(frame_granularity=64, batch_buckets=[1, 16])
        with torch.no_grad():
            assert (
                c.run(
                    FUSED,
                    spy,
                    torch.randn(9, 128, 16, device=dev),
                    torch.full((9,), 128, device=dev, dtype=torch.int32),
                )
                is None
            )

        assert seen, "the probe never called the forward"
        assert all(s[0] == 1 for s in seen), f"probe widened past B=1: {seen}"
        assert all(s[1] <= 128 for s in seen), f"probe used more frames than asked: {seen}"

    def test_oversized_batch_is_counted_not_silent(self):
        dev = torch.device("cuda")
        enc = _Encoder().to(dev).eval()
        c = _cache(frame_granularity=64, batch_buckets=[2])
        with torch.no_grad():
            out = c.run(
                FUSED,
                enc.forward_offline,
                torch.randn(9, 64, 16, device=dev),
                torch.full((9,), 64, device=dev, dtype=torch.int32),
            )
        assert out is None
        assert c.fallback_oversized == 1

    def test_saturated_cache_falls_back_and_counts(self):
        torch.manual_seed(0)
        dev = torch.device("cuda")
        enc = _Encoder().to(dev).eval()
        c = _cache(frame_granularity=64, batch_buckets=[2], max_captures=1)
        lens = torch.tensor([64, 50], device=dev, dtype=torch.int32)
        with torch.no_grad():
            assert (
                c.run(FUSED, enc.forward_offline, torch.randn(2, 64, 16, device=dev), lens)
                is not None
            )
            assert (
                c.run(FUSED, enc.forward_offline, torch.randn(2, 192, 16, device=dev), lens) is None
            )
        assert c.num_captured == 1
        assert c.fallback_saturated == 1

    def test_pad_overhead_reports_the_bucketing_tax(self):
        torch.manual_seed(0)
        dev = torch.device("cuda")
        enc = _Encoder().to(dev).eval()
        c = _cache(frame_granularity=64, batch_buckets=[4])
        with torch.no_grad():  # B 2 -> 4, T 100 -> 128
            c.run(
                FUSED,
                enc.forward_offline,
                torch.randn(2, 100, 16, device=dev),
                torch.tensor([100, 90], device=dev, dtype=torch.int32),
            )
        assert c.pad_overhead == pytest.approx((4 * 128) / (2 * 100))


# --------------------------------------------------------------------------
# Recovery from a capture that failed halfway through
# --------------------------------------------------------------------------


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


@pytest.mark.cuda
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


@pytest.mark.cuda
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


@pytest.mark.cuda
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


@pytest.mark.cuda
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


# --------------------------------------------------------------------------
# The transducer predictor step, and which shapes of state can be carried
# --------------------------------------------------------------------------


class TestCapturable:
    @pytest.mark.cuda
    def test_a_bare_cuda_tensor_is_capturable(self):
        """The regression: icefall's stateless predictor state is exactly this."""
        assert Cache.capturable(torch.zeros(4, 2, device="cuda"))

    @pytest.mark.cuda
    def test_a_sequence_of_cuda_tensors_is_still_capturable(self):
        t = torch.zeros(4, 2, device="cuda")
        assert Cache.capturable((t, t))
        assert Cache.capturable([t, t])

    def test_a_cpu_state_is_not(self):
        assert not Cache.capturable(torch.zeros(4, 2))
        assert not Cache.capturable((torch.zeros(4, 2),))

    def test_non_tensor_states_are_refused(self):
        assert not Cache.capturable(None)
        assert not Cache.capturable(())
        assert not Cache.capturable([])
        assert not Cache.capturable({"h": 1})
        assert not Cache.capturable(((torch.zeros(1),),))  # nested


class TestDetach:
    @pytest.mark.cuda
    def test_a_bare_tensor_is_copied_not_aliased(self):
        """Graph memory must not escape; returning the same object let it."""
        src = torch.zeros(4, 2, device="cuda")
        out = Cache.detach(src)
        assert out is not src, "detach aliased the caller's state"
        out.fill_(1.0)
        assert float(src.abs().sum()) == 0.0, "detach returned a view"

    @pytest.mark.cuda
    def test_a_sequence_is_copied_elementwise(self):
        src = (torch.zeros(4, 2, device="cuda"), torch.zeros(4, device="cuda"))
        out = Cache.detach(src)
        assert all(a is not b for a, b in zip(src, out))
        out[0].fill_(1.0)
        assert float(src[0].abs().sum()) == 0.0


CONTEXT = 3


class _BareStatePredictor:
    """Minimal stand-in for a stateless predictor: the state is one tensor.

    Shaped like icefall's: ``(B, CONTEXT)`` of label ids, projected to ``dim``.
    """

    def __init__(self, dim=4, device="cuda"):
        # Deterministic and RNG-free on purpose.  A *failed* CUDA-graph capture
        # elsewhere in the process wedges the CUDA generator ("Offset increment
        # outside graph capture"), so a `torch.randn` here would make these tests
        # fail for a reason that has nothing to do with them.
        self.w = (
            torch.linspace(-1.0, 1.0, CONTEXT * dim, device=device)
            .reshape(CONTEXT, dim)
            .contiguous()
        )

    def advance(self, state, tok, emit):
        # Shift the label window left and append the emitted token.
        nxt = torch.roll(state, shifts=-1, dims=1)
        nxt[:, -1] = tok
        return torch.where(emit.unsqueeze(1), nxt, state)

    def predict(self, state):
        return state.float() @ self.w  # (B, CONTEXT) @ (CONTEXT, dim) -> (B, dim)


class _Joiner:
    def decoder_proj(self, x):
        return x * 2.0


@pytest.mark.cuda
class TestCapturesAndReplaysABareState:
    def _cache(self):
        return Cache(_BareStatePredictor(), _Joiner(), max_captures=4)

    def test_step_returns_a_replay_not_none(self):
        """What ``0 replays, 304 fallbacks`` looked like before the fix."""
        cache = self._cache()
        state = torch.zeros(4, CONTEXT, dtype=torch.long, device="cuda")
        tok = torch.tensor([1, 2, 3, 4], device="cuda")
        emit = torch.tensor([True, True, False, True], device="cuda")
        out = cache.step(state, tok, emit)
        assert out is not None, "bare-tensor state was refused"
        assert cache.num_captured == 1

    def test_the_returned_state_keeps_the_callers_shape(self):
        """A bare state in must not become a 1-tuple out."""
        cache = self._cache()
        state = torch.zeros(4, CONTEXT, dtype=torch.long, device="cuda")
        tok = torch.tensor([1, 2, 3, 4], device="cuda")
        emit = torch.ones(4, dtype=torch.bool, device="cuda")
        new_state, _proj = cache.step(state, tok, emit)
        assert isinstance(new_state, torch.Tensor), type(new_state)

    def test_replay_matches_the_eager_step(self):
        cache = self._cache()
        pred, join = cache._predictor, cache._joiner
        state = torch.tensor([[0, 1, 2]] * 4, device="cuda")  # (4, CONTEXT)
        tok = torch.tensor([5, 6, 7, 8], device="cuda")
        emit = torch.tensor([True, False, True, False], device="cuda")

        want_state = pred.advance(state, tok, emit)
        want_proj = join.decoder_proj(pred.predict(want_state))
        got_state, got_proj = cache.step(state, tok, emit)

        assert torch.equal(got_state, want_state)
        assert torch.allclose(got_proj, want_proj, atol=0, rtol=0)

    def test_successive_replays_carry_state_forward(self):
        """The graph writes its output back into the buffers it reads."""
        cache = self._cache()
        pred = cache._predictor
        state = torch.zeros(4, CONTEXT, dtype=torch.long, device="cuda")
        emit = torch.ones(4, dtype=torch.bool, device="cuda")
        want = state
        for step in range(5):
            tok = torch.full((4,), step + 1, device="cuda")
            want = pred.advance(want, tok, emit)
            state, _ = cache.step(state, tok, emit)
            assert torch.equal(state, want), f"diverged at step {step}"

    def test_a_second_batch_width_gets_its_own_capture(self):
        cache = self._cache()
        for b in (2, 4):
            cache.step(
                torch.zeros(b, CONTEXT, dtype=torch.long, device="cuda"),
                torch.ones(b, dtype=torch.long, device="cuda"),
                torch.ones(b, dtype=torch.bool, device="cuda"),
            )
        assert cache.num_captured == 2

    def test_a_bare_state_and_a_one_tuple_do_not_share_a_capture(self):
        """They hand the predictor different shapes, so the key must differ."""
        cache = self._cache()
        s = torch.zeros(4, CONTEXT, dtype=torch.long, device="cuda")
        assert cache._key((s,), bare=True) != cache._key((s,), bare=False)


# --------------------------------------------------------------------------
# The streaming encoder's cache-bucket ladder
# --------------------------------------------------------------------------


_N_BLOCK = 64


class TestLadderShape:
    @pytest.mark.parametrize("capacity", [64, 512, 1024, 4096, 4984, 8192])
    def test_rungs_are_kernel_tile_multiples_within_capacity(self, capacity):
        ladder = cache_bucket_ladder(capacity)
        assert ladder, "ladder must not be empty"
        assert ladder == sorted(set(ladder)), "rungs must be sorted and unique"
        for rung in ladder:
            assert rung % _N_BLOCK == 0, f"{rung} is not an N_BLOCK multiple"
            # A rung *above* capacity is the out-of-bounds read the capacity
            # exists to prevent — the block table cannot address it and the
            # relative-position table cannot index it.
            assert rung <= capacity, f"rung {rung} exceeds capacity {capacity}"

    def test_it_is_finite_and_small(self):
        """The whole point: a flat 64-frame axis is unbounded, this one is not."""
        for capacity in (4096, 65536, 1 << 20):
            ladder = cache_bucket_ladder(capacity)
            assert len(ladder) < 40, f"{capacity} produced {len(ladder)} rungs"
        # ...and it grows logarithmically, not linearly, in the capacity.
        small = len(cache_bucket_ladder(4096))
        big = len(cache_bucket_ladder(4096 * 64))
        assert big - small <= 12, (small, big)

    def test_below_the_knee_it_is_still_flat_64(self):
        """Short streams keep the fine granularity; a coarse rung there would be
        a large *relative* over-read."""
        ladder = cache_bucket_ladder(4096)
        fine = [r for r in ladder if r <= CACHE_BUCKET_KNEE]
        assert fine == list(range(0, CACHE_BUCKET_KNEE + 1, _N_BLOCK))

    def test_growth_one_restores_the_legacy_flat_ladder(self):
        ladder = cache_bucket_ladder(1024, growth=1.0)
        assert ladder == list(range(0, 1025, _N_BLOCK))


class TestPickMatchesLadder:
    """The pre-warm captures the ladder; the runtime picks with this function.

    If the two ever disagree, the pre-warm covers shapes the runtime never asks
    for and misses the ones it does — which is the tail, silently back.
    """

    @pytest.mark.parametrize("capacity", [512, 1024, 4096])
    def test_every_reachable_length_maps_onto_a_prewarmed_rung(self, capacity):
        ladder = cache_bucket_ladder(capacity)
        rungs = set(ladder)
        for cache_t1 in range(0, capacity + 1):
            got = pick_cache_bucket(cache_t1, ladder)
            assert got in rungs, f"cache_t1={cache_t1} -> {got}, off the ladder"

    @pytest.mark.parametrize("capacity", [512, 4096])
    def test_a_bucket_is_never_shorter_than_the_cache(self, capacity):
        """Handing the kernel a ``host_seqlen_max`` below the real
        ``cache_seqlens`` would truncate a stream's attention history."""
        ladder = cache_bucket_ladder(capacity)
        for cache_t1 in range(0, capacity + 1):
            assert pick_cache_bucket(cache_t1, ladder) >= cache_t1

    def test_over_read_is_bounded_by_the_growth_ratio(self):
        """The trade for a finite ladder: rungs above the knee over-read by at
        most ``growth``, which measured ~4% of a replay at 1.5."""
        ladder = cache_bucket_ladder(8192, growth=1.5)
        for cache_t1 in range(CACHE_BUCKET_KNEE, 8192, 37):
            rung = pick_cache_bucket(cache_t1, ladder)
            assert rung <= cache_t1 * 1.5 + _N_BLOCK, (cache_t1, rung)

    def test_off_ladder_falls_back_to_flat_rounding(self):
        ladder = cache_bucket_ladder(512)
        assert pick_cache_bucket(9999, ladder) == round_up_bucket(9999)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestPrewarmCoversTheRuntime:
    """End-to-end: a stream long enough to walk the whole cache ladder must not
    capture a single graph on a live tick."""

    def test_a_long_stream_captures_nothing(self, ckpt_dir):
        from oasr.engine import ASREngine, EngineConfig
        from oasr.engine.graph_cache import GraphedEncoderForward

        seen: list = []
        original = GraphedEncoderForward._capture

        def spy(self, B, T, bucket, *a, **kw):
            seen.append((B, T, bucket))
            return original(self, B, T, bucket, *a, **kw)

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device="cuda",
            dtype=torch.bfloat16,
            service_mode="streaming",
            max_batch_size=4,
            num_left_chunks=-1,  # the default: an unbounded cache axis
        )
        engine = ASREngine(cfg)
        try:
            GraphedEncoderForward._capture = spy
            seen.clear()
            samples = engine._input_processor.streaming_audio_chunk_samples
            # ~60 s of audio walks well past where the old ladder stopped.
            chunk = torch.zeros(samples, dtype=torch.float32)
            rid = engine.add_streaming_request(sample_rate=16000)
            n = int(60 * 16000 / samples)
            for j in range(n):
                engine.feed_chunk(rid, chunk, is_last=(j == n - 1))
            engine.run()
            torch.cuda.synchronize()
        finally:
            GraphedEncoderForward._capture = original
            engine.shutdown()

        assert not seen, (
            f"{len(seen)} graph(s) captured on a live tick — the pre-warm ladder "
            f"no longer covers what the runtime asks for: {seen[:5]}"
        )
