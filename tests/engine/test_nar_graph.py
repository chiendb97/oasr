"""The two-stage CIF + NAR decode capture cache (``oasr.engine.nar_graph``).

Stand-in halves with the real signatures, small enough to reason about exactly:
every op is row-local, so a replay at a padded batch width must equal the eager
call on the unpadded batch bit for bit.  The real Paraformer is exercised in
``tests/models/arch/test_paraformer.py::TestCapturedDecode``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from oasr.engine.nar_graph import NarDecodeGraphs

D = 16


def _weights(enc: torch.Tensor, lens: torch.Tensor):
    """``(alphas (B, T+1), token_lens (B,))``, from the valid frames only."""
    T = enc.size(1)
    mask = (torch.arange(T, device=enc.device)[None] < lens[:, None].to(enc.device)).float()
    alphas = torch.sigmoid(enc.float().mean(-1)) * mask
    alphas = F.pad(alphas, (0, 1))
    return alphas, torch.floor(alphas.sum(-1)).to(torch.int32)


def _decode(enc, lens, alphas, token_lens, num_tokens: int):
    """``(best_lp (B, U), best_ids (B, U), fires (B, T+1))``; ``U`` shapes the output."""
    idx = torch.arange(num_tokens, device=enc.device) % alphas.size(1)
    picked = alphas[:, idx] * token_lens[:, None].float()
    best_ids = (picked > 0.25).long() + idx[None]
    return picked + enc.float().sum((1, 2))[:, None], best_ids, alphas * 2


def _graphs(**kw) -> NarDecodeGraphs:
    kw.setdefault("device", torch.device("cuda"))
    kw.setdefault("batch_buckets", [1, 2, 4])
    kw.setdefault("max_frames", 256)
    return NarDecodeGraphs(weights_fn=_weights, decode_fn=_decode, **kw)


def _batch(B: int, T: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    enc = torch.randn(B, T, D, device="cuda", generator=g, dtype=torch.float16)
    lens = torch.randint(1, T + 1, (B,), device="cuda", generator=g, dtype=torch.int32)
    lens[0] = T
    return enc, lens


def _served(graphs: NarDecodeGraphs, enc, lens, num_tokens: int):
    stage = graphs.weights(enc, lens)
    assert stage is not None
    B = enc.size(0)
    alphas, token_lens = stage.alphas[:B].clone(), stage.token_lens[:B].clone()
    decoded = graphs.decode(stage, num_tokens)
    assert decoded is not None
    return (alphas, token_lens) + tuple(t.clone() for t in decoded)


def _eager(enc, lens, num_tokens: int):
    alphas, token_lens = _weights(enc, lens)
    return (alphas, token_lens) + _decode(enc, lens, alphas, token_lens, num_tokens)


@pytest.mark.cuda
class TestReplay:
    @pytest.mark.parametrize("B", [1, 2, 3], ids=["exact1", "exact2", "padded3to4"])
    def test_replay_equals_eager_bit_for_bit(self, B):
        graphs = _graphs()
        enc, lens = _batch(B, 64, seed=B)
        for got, want in zip(_served(graphs, enc, lens, 16), _eager(enc, lens, 16)):
            assert torch.equal(got, want)

    def test_a_second_batch_reuses_the_graphs_and_sees_its_own_data(self):
        graphs = _graphs()
        first, first_lens = _batch(3, 64, seed=1)
        _served(graphs, first, first_lens, 16)
        captured = graphs.captures
        second, second_lens = _batch(3, 64, seed=2)
        for got, want in zip(
            _served(graphs, second, second_lens, 16), _eager(second, second_lens, 16)
        ):
            assert torch.equal(got, want)
        assert graphs.captures == captured
        assert graphs.hits == 4

    def test_each_token_width_is_its_own_decode_graph(self):
        graphs = _graphs()
        enc, lens = _batch(2, 64, seed=3)
        for width in (8, 16, 8):
            got = _served(graphs, enc, lens, width)
            assert got[2].shape == (2, width)
            for a, b in zip(got, _eager(enc, lens, width)):
                assert torch.equal(a, b)
        assert graphs.num_captured == 3  # one weights graph, two decode graphs

    def test_padded_rows_are_reset_in_the_shared_staging_buffer(self):
        """Batch widths 4 and 2 view the same staging memory.  The rows past a
        3-row batch must be silence of length 1 every time, whatever a wider
        batch at another key left there."""
        graphs = _graphs()
        wide, wide_lens = _batch(4, 64, seed=4)
        _served(graphs, wide, wide_lens, 8)
        narrow, narrow_lens = _batch(3, 64, seed=5)
        stage = graphs.weights(narrow, narrow_lens)
        assert stage is not None
        assert torch.count_nonzero(stage.enc[3]) == 0
        assert int(stage.lens[3]) == 1
        silence = torch.zeros(1, 64, D, device="cuda", dtype=torch.float16)
        want_alphas, _ = _weights(silence, torch.ones(1, device="cuda", dtype=torch.int32))
        assert torch.equal(stage.alphas[3:4], want_alphas)

    def test_token_bucket_rounds_up_to_the_granularity(self):
        graphs = _graphs()
        assert [graphs.token_bucket(u) for u in (0, 1, 8, 9, 17)] == [8, 8, 8, 16, 24]
        assert _graphs(token_granularity=4).token_bucket(9) == 12


@pytest.mark.cuda
class TestFallbacks:
    def test_oversized_batch_and_width_run_eager_and_are_counted(self):
        graphs = _graphs()
        assert graphs.weights(*_batch(5, 64, seed=6)) is None
        assert graphs.weights(*_batch(1, 320, seed=7)) is None
        assert graphs.fallback_oversized == 2
        assert graphs.captures == 0

    def test_cpu_device_disables_the_cache(self):
        graphs = NarDecodeGraphs(
            device=torch.device("cpu"), weights_fn=_weights, decode_fn=_decode, batch_buckets=[1]
        )
        assert not graphs.enabled
        enc = torch.zeros(1, 8, D)
        assert graphs.weights(enc, torch.ones(1, dtype=torch.int32)) is None

    def test_an_uncapturable_half_turns_the_cache_off(self):
        def reads_back(enc, lens):
            alphas, token_lens = _weights(enc, lens)
            return alphas, token_lens + int(token_lens.sum().item())

        graphs = NarDecodeGraphs(
            device=torch.device("cuda"),
            weights_fn=reads_back,
            decode_fn=_decode,
            batch_buckets=[1, 2],
        )
        enc, lens = _batch(2, 64, seed=8)
        assert graphs.weights(enc, lens) is None
        assert graphs.capture_failures == 1 and not graphs.enabled
        assert graphs.weights(enc, lens) is None
        assert graphs.fallback_disabled == 1
        # The allocator and the generator are usable afterwards.
        torch.randn(4, device="cuda")
        assert torch.equal(_weights(enc, lens)[1], _weights(enc, lens)[1])


@pytest.mark.cuda
class TestBounds:
    def test_at_the_cap_the_only_weights_graph_gives_up_its_oldest_decode(self):
        graphs = _graphs(max_captures=3)
        enc, lens = _batch(2, 64, seed=9)
        _served(graphs, enc, lens, 8)
        _served(graphs, enc, lens, 16)
        assert graphs.num_captured == 3
        got = _served(graphs, enc, lens, 24)
        assert graphs.num_captured == 3 and graphs.evictions == 1
        stage = graphs.weights(enc, lens)
        assert stage is not None and list(stage.decodes) == [16, 24]
        for a, b in zip(got, _eager(enc, lens, 24)):
            assert torch.equal(a, b)

    def test_a_new_shape_evicts_the_least_recent_weights_graph_with_its_decodes(self):
        graphs = _graphs(max_captures=3)
        enc, lens = _batch(2, 64, seed=10)
        _served(graphs, enc, lens, 8)
        _served(graphs, enc, lens, 16)
        other, other_lens = _batch(1, 128, seed=11)
        got = _served(graphs, other, other_lens, 8)
        assert graphs.num_captured == 2  # the new weights graph and its decode
        assert graphs.evictions == 1
        for a, b in zip(got, _eager(other, other_lens, 8)):
            assert torch.equal(a, b)

    def test_without_eviction_a_full_cache_runs_new_shapes_eager(self):
        graphs = _graphs(max_captures=2, evict_lru=False)
        enc, lens = _batch(2, 64, seed=12)
        _served(graphs, enc, lens, 8)
        stage = graphs.weights(enc, lens)
        assert stage is not None
        assert graphs.decode(stage, 16) is None
        assert graphs.fallback_saturated == 1

    def test_release_resets_everything_and_the_cache_captures_again(self):
        graphs = _graphs()
        enc, lens = _batch(2, 64, seed=13)
        _served(graphs, enc, lens, 8)
        graphs.release()
        assert graphs.num_captured == 0 and graphs.pool_bytes == 0
        for a, b in zip(_served(graphs, enc, lens, 8), _eager(enc, lens, 8)):
            assert torch.equal(a, b)

    def test_a_capture_that_would_starve_the_eager_path_is_refused(self, monkeypatch):
        import oasr.engine.nar_graph as nar_graph

        graphs = _graphs()
        monkeypatch.setattr(nar_graph, "read_device_memory", lambda device: (0, 1 << 40))
        enc, lens = _batch(2, 64, seed=14)
        assert graphs.weights(enc, lens) is None
        assert graphs.fallback_budget == 1 and graphs.captures == 0
        # Remembered: no second warm-up inside the retry window.
        assert graphs.weights(enc, lens) is None
        assert graphs.fallback_budget == 2
