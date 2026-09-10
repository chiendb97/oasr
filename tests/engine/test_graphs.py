# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the offline-forward CUDA-Graph cache (``oasr/engine/offline_graph.py``).

The shape/bucketing half runs anywhere.  The capture half needs CUDA and pins the
three properties that make the cache safe to leave on by default:

* a replay is **bit-exact** against running the same padded input eagerly;
* a shape that falls back to eager decodes **identically** to one that is
  graph-served, which is what :meth:`GraphedOfflineForward.pad_time` is for;
* a capture that fails is remembered and **never retried**.
"""

from __future__ import annotations

import pytest
import torch

from oasr.engine.offline_graph import (
    DEFAULT_FRAME_GRANULARITY,
    FUSED,
    GraphedOfflineForward,
    resolve_batch_buckets,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kw",
        [
            {"offline_graph_frame_granularity": 0},
            {"offline_graph_max_frames": 8, "offline_graph_frame_granularity": 64},
            {"offline_graph_max_captures": 0},
            {"offline_graph_batch_buckets": [0]},
            {"offline_graph_batch_buckets": [1, 999]},
        ],
    )
    def test_rejects_incoherent_knobs(self, kw):
        from oasr.engine.config import EngineConfig

        with pytest.raises(ValueError):
            EngineConfig(ckpt_dir="/nonexistent", max_batch_size=32, **kw)

    def test_defaults_are_coherent(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/nonexistent")
        assert cfg.offline_graph_frame_granularity == DEFAULT_FRAME_GRANULARITY
        assert cfg.use_offline_cuda_graphs is True


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


@cuda_only
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


@cuda_only
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


@cuda_only
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
