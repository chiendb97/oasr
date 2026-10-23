#!/usr/bin/env python3
"""Tests for the decoder-side KV storage the AR families generate through.

Two layers, both pure bookkeeping and both CPU-only:

* ``DecoderKVCacheManager`` — paged block allocation;
* ``DecoderKv`` / ``PagedDecoderKv`` — the per-row state
  ``prefill``/``step``/``select``/``merge`` thread through.
"""

import pytest
import torch

from oasr.cache import DecoderKv, DecoderKVCacheManager, PagedDecoderKv


@pytest.fixture()
def mgr():
    pool = DecoderKVCacheManager.build_pool(
        num_layers=2,
        n_kv_head=2,
        head_dim=8,
        block_tokens=4,
        max_num_blocks=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return DecoderKVCacheManager(pool)


class TestLifecycle:
    def test_create_free_roundtrip(self, mgr):
        total = mgr.pool.num_total_blocks
        mgr.create("r1", max_new_tokens=10)
        assert mgr.num_active() == 1
        assert mgr.seqlen("r1") == 0
        assert mgr.pool.num_free_blocks == total - 1  # first page pre-allocated
        mgr.free("r1")
        assert mgr.num_active() == 0
        assert mgr.pool.num_free_blocks == total

    def test_prefill_allocates_pages(self, mgr):
        mgr.create("r1", max_new_tokens=4, prefill_len=6)  # 6 tokens / 4 per page → 2
        assert mgr.seqlen("r1") == 6
        assert mgr.block_tables(["r1"]).shape[1] == 2

    def test_duplicate_create_rejected(self, mgr):
        mgr.create("r1", max_new_tokens=4)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create("r1", max_new_tokens=4)

    def test_free_is_idempotent(self, mgr):
        mgr.create("r1", max_new_tokens=4)
        mgr.free("r1")
        mgr.free("r1")  # no raise

    def test_free_all(self, mgr):
        total = mgr.pool.num_total_blocks
        for rid in ("a", "b", "c"):
            mgr.create(rid, max_new_tokens=4)
        mgr.free_all()
        assert mgr.num_active() == 0
        assert mgr.pool.num_free_blocks == total


class TestGrowth:
    def test_append_returns_write_positions(self, mgr):
        mgr.create("a", max_new_tokens=16)
        mgr.create("b", max_new_tokens=16, prefill_len=3)
        for step in range(5):
            pos = mgr.append_step(["a", "b"])
            assert pos == [step, 3 + step]
        assert mgr.cache_seqlens(["a", "b"]).tolist() == [5, 8]

    def test_page_boundary_allocates_block(self, mgr):
        mgr.create("a", max_new_tokens=16)
        free0 = mgr.pool.num_free_blocks
        for _ in range(4):  # fills page 0 exactly
            mgr.append_step(["a"])
        assert mgr.pool.num_free_blocks == free0
        mgr.append_step(["a"])  # token 5 crosses into page 1
        assert mgr.pool.num_free_blocks == free0 - 1
        assert mgr.block_tables(["a"]).shape == (1, 2)

    def test_no_eviction_growth_to_exhaustion(self, mgr):
        """AR decode grows monotonically — exhaustion raises, nothing evicts."""
        mgr.create("a", max_new_tokens=1000)
        with pytest.raises(RuntimeError, match="exhausted"):
            for _ in range(mgr.pool.num_total_blocks * 4 + 1):
                mgr.append_step(["a"])

    def test_block_tables_padded_per_batch(self, mgr):
        mgr.create("a", max_new_tokens=16, prefill_len=9)  # 3 pages
        mgr.create("b", max_new_tokens=16, prefill_len=1)  # 1 page
        table = mgr.block_tables(["a", "b"])
        assert table.shape == (2, 3)
        assert table.dtype == torch.int32
        # padding entries are 0 (never dereferenced past cache_seqlens)
        assert table[1, 1:].tolist() == [0, 0]

    def test_kv_view_shape(self, mgr):
        k, v = mgr.kv_view(0)
        assert k.shape == v.shape == (8, 4, 2, 8)  # (blocks, tokens, heads, head_dim)


# ---------------------------------------------------------------------------
# Per-row KV offsets: dense state threaded through AR decoders
# ---------------------------------------------------------------------------

LAYERS, HEADS, DIM = 2, 3, 4


def _kv(batch, cap=16, starts=None):
    return DecoderKv.empty(
        LAYERS,
        batch,
        torch.device("cpu"),
        starts=None if starts is None else torch.tensor(starts, dtype=torch.int32),
        cap=cap,
    )


def _tokens(batch, t_new, fill):
    """``(B, HEADS, t_new, DIM)`` where every element of row b is ``fill + b``."""
    base = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1)
    return base + float(fill) + torch.zeros(batch, HEADS, t_new, DIM)


class TestPerRowOffsets:
    """Rows at different offsets are the whole point: without them two decode
    groups can never share a forward, and neither can a paged cache index a
    per-row block table."""

    def test_uniform_append_is_a_slice(self):
        kv = _kv(2)
        k, v, extent = kv.append(0, _tokens(2, 3, 10), _tokens(2, 3, 20))
        kv.commit(3)
        assert extent == 3
        assert k.shape == (2, HEADS, 16, DIM)
        assert kv.lens_host == [3, 3]
        torch.testing.assert_close(k[:, :, :3], _tokens(2, 3, 10))
        torch.testing.assert_close(v[:, :, :3], _tokens(2, 3, 20))

    def test_divergent_rows_scatter_to_their_own_offsets(self):
        kv = _kv(2)
        kv.append(0, _tokens(2, 4, 0), _tokens(2, 4, 0))
        kv.commit(4)
        # Retire row 1 back to offset 2 by hand -- what a merge produces.
        kv.lens = torch.tensor([4, 2], dtype=torch.int32)
        kv.lens_host = [4, 2]
        assert not kv.uniform

        k, _, extent = kv.append(0, _tokens(2, 1, 99), _tokens(2, 1, 99))
        assert extent == 5  # bounded by the *longest* row
        torch.testing.assert_close(k[0, :, 4], torch.full((HEADS, DIM), 99.0))
        torch.testing.assert_close(k[1, :, 2], torch.full((HEADS, DIM), 100.0))
        # Row 1's offset-4 slot is still untouched capacity, not row 0's write.
        torch.testing.assert_close(k[1, :, 4], torch.zeros(HEADS, DIM))

    def test_mask_kwargs_is_the_post_write_window(self):
        kv = _kv(2, starts=[0, 3])
        kv.lens = torch.tensor([5, 7], dtype=torch.int32)
        kv.lens_host = [5, 7]
        kwargs = kv.mask_kwargs(1)
        assert kwargs["kv_lens"].tolist() == [6, 8]
        assert kwargs["kv_starts"].tolist() == [0, 3]
        # Positions skip the left padding; the window does not.
        assert kv.positions().tolist() == [5, 4]

    def test_trimmed_prefill_needs_no_mask(self):
        """A prefill reads the cache trimmed and masks with the causal triangle
        alone -- handing it the whole capacity buffer plus a length vector would
        compute two orders of magnitude more score matrix for the same answer."""
        kv = _kv(2)
        assert kv.mask_kwargs(4, trimmed=True) == {}
        k, _, extent = kv.append(0, _tokens(2, 4, 1), _tokens(2, 4, 1), trim=True)
        assert extent is None and k.size(2) == 4

    def test_overflow_grows_instead_of_corrupting(self):
        kv = _kv(2, cap=4)
        kv.append(0, _tokens(2, 4, 1), _tokens(2, 4, 1))
        kv.commit(4)
        k, _, extent = kv.append(0, _tokens(2, 1, 7), _tokens(2, 1, 7))
        assert extent == 5 and k.size(2) >= 5
        torch.testing.assert_close(k[:, :, :4], _tokens(2, 4, 1))  # kept
        torch.testing.assert_close(k[:, :, 4], _tokens(2, 1, 7)[:, :, 0])

    def test_legacy_growth_refuses_divergent_rows(self):
        """``cat`` appends one width to every row, so it cannot represent them --
        which is why ``can_merge`` says no rather than producing a wrong cache."""
        kv = DecoderKv.empty(LAYERS, 2, torch.device("cpu"))
        kv.append(0, _tokens(2, 3, 1), _tokens(2, 3, 1))
        kv.commit(3)
        kv.lens_host = [3, 1]
        with pytest.raises(RuntimeError, match="different offsets"):
            kv.append(0, _tokens(2, 1, 2), _tokens(2, 1, 2))

    def test_select_keeps_rows_and_repeats(self):
        kv = _kv(3, starts=[0, 1, 2])
        kv.append(0, _tokens(3, 2, 5), _tokens(3, 2, 5))
        kv.commit(2)
        picked = kv.select(torch.tensor([2, 0, 0]))
        assert picked.lens_host == [2, 2, 2]
        assert picked.starts.tolist() == [2, 0, 0]
        src = _tokens(3, 2, 5)
        torch.testing.assert_close(picked.k[0][0:1, :, :2], src[2:3])
        torch.testing.assert_close(picked.k[0][1:3, :, :2], src[0:1].expand(2, -1, -1, -1))


class TestMerge:
    def test_merge_concatenates_rows_and_offsets(self):
        a, b = _kv(2, cap=8, starts=[0, 1]), _kv(1, cap=6, starts=[2])
        a.append(0, _tokens(2, 5, 10), _tokens(2, 5, 10))
        a.append(1, _tokens(2, 5, 10), _tokens(2, 5, 10))
        a.commit(5)
        b.append(0, _tokens(1, 2, 30), _tokens(1, 2, 30))
        b.append(1, _tokens(1, 2, 30), _tokens(1, 2, 30))
        b.commit(2)

        merged = a.merge(b)
        assert merged.lens_host == [5, 5, 2]
        assert merged.starts.tolist() == [0, 1, 2]
        assert merged.k[0].shape == (3, HEADS, 8, DIM)  # widened to the max cap
        torch.testing.assert_close(merged.k[0][:2, :, :5], _tokens(2, 5, 10))
        torch.testing.assert_close(merged.k[0][2:, :, :2], _tokens(1, 2, 30))

    def test_merged_capacity_tail_is_zeroed(self):
        """The buffer is handed to the attention kernel whole, so its untouched
        tail must be zero -- a NaN bit pattern there survives any mask through
        ``P @ V``.  Widening on merge is a fresh allocation, so it has to hold."""
        a, b = _kv(1, cap=4), _kv(1, cap=12)
        for kv, t in ((a, 4), (b, 2)):
            for layer in range(LAYERS):
                kv.append(layer, _tokens(1, t, 1), _tokens(1, t, 1))
            kv.commit(t)
        merged = a.merge(b)
        for buf in merged.k + merged.v:
            assert torch.isfinite(buf).all()
            torch.testing.assert_close(buf[1, :, 2:], torch.zeros(HEADS, 10, DIM))

    def test_merge_releases_both_sources(self):
        """Merging is a copy of both caches; holding the sources through it would
        double the transient at shapes where one row is ~0.4 GiB."""
        a, b = _kv(1), _kv(1)
        for kv in (a, b):
            kv.append(0, _tokens(1, 2, 1), _tokens(1, 2, 1))
            kv.commit(2)
        a.merge(b)
        assert a.k[0] is None and b.k[0] is None

    def test_legacy_and_capacity_states_do_not_merge(self):
        cap, legacy = _kv(1), DecoderKv.empty(LAYERS, 1, torch.device("cpu"))
        assert not cap.can_merge(legacy)
        assert not legacy.can_merge(cap)
        with pytest.raises(ValueError, match="not mergeable"):
            cap.merge(legacy)

    def test_mismatched_padding_discipline_does_not_merge(self):
        assert not _kv(1, starts=[0]).can_merge(_kv(1))


# ---------------------------------------------------------------------------
# Paged storage
# ---------------------------------------------------------------------------


def _paged(mgr, batch, prefill_len=2, capacity=12, starts=None):
    return PagedDecoderKv.create(
        mgr,
        batch,
        torch.device("cpu"),
        prefill_len=prefill_len,
        capacity=capacity,
        starts=None if starts is None else torch.tensor(starts, dtype=torch.int32),
    )


def _pool_tokens(mgr, batch, t_new, fill):
    """``(B, H_kv, t_new, D)`` shaped for ``mgr``'s pool, row ``b`` filled ``fill + b``."""
    cfg = mgr.pool.config
    base = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1)
    return base + float(fill) + torch.zeros(batch, cfg.n_kv_head, t_new, cfg.head_dim)


def _write(mgr, kv, t_new, fill):
    """One forward's worth of writes: mask first, then every layer, then commit."""
    data = _pool_tokens(mgr, kv.batch, t_new, fill)
    kv.mask_kwargs(t_new)
    for layer in range(mgr.pool.config.num_layers):
        kv.append(layer, data, data)
    kv.commit(t_new)
    return data


class TestPagedStorage:
    """A row holds only the pages it has filled, and gives them back."""

    def test_prefill_then_steps_land_where_the_table_says(self, mgr):
        kv = _paged(mgr, 1, prefill_len=3, capacity=12)
        prompt = _write(mgr, kv, 3, 5)
        token = _write(mgr, kv, 1, 9)
        assert kv.lens_host == [4]
        k_pool, _ = mgr.kv_view(0)
        page = kv.block_table()[0, 0].item()
        # The pool is frame-major, the decoder's K/V head-major.
        torch.testing.assert_close(k_pool[page, :3], prompt[0].permute(1, 0, 2))
        torch.testing.assert_close(k_pool[page, 3], token[0, :, 0])

    def test_pages_are_mapped_as_rows_fill_them(self, mgr):
        """The point of paging: a row that stops early never held the rest."""
        kv = _paged(mgr, 1, prefill_len=1, capacity=16)
        _write(mgr, kv, 1, 1)
        assert mgr.num_blocks(kv.slots[0]) == 1  # 4 tokens per page in this pool
        for _ in range(4):
            _write(mgr, kv, 1, 1)
        assert mgr.num_blocks(kv.slots[0]) == 2

    def test_select_frees_the_rows_it_drops(self, mgr):
        kv = _paged(mgr, 2, prefill_len=2, capacity=8)
        _write(mgr, kv, 2, 1)
        free_before = mgr.pool.num_free_blocks
        kept = kv.select(torch.tensor([1]))
        assert mgr.pool.num_free_blocks == free_before + 1
        assert kept.lens_host == [2] and mgr.num_active() == 1

    def test_free_is_idempotent_and_returns_everything(self, mgr):
        total = mgr.pool.num_total_blocks
        kv = _paged(mgr, 2)
        _write(mgr, kv, 2, 1)
        kv.free()
        kv.free()
        assert mgr.pool.num_free_blocks == total and mgr.num_active() == 0

    def test_merge_moves_no_pages(self, mgr):
        a = _paged(mgr, 1, prefill_len=2, capacity=8)
        b = _paged(mgr, 1, prefill_len=1, capacity=8)
        _write(mgr, a, 2, 1)
        _write(mgr, a, 1, 2)  # a is three tokens in, b is one
        _write(mgr, b, 1, 3)
        held = mgr.pool.num_free_blocks
        slots = list(a.slots) + list(b.slots)
        merged = a.merge(b)
        assert mgr.pool.num_free_blocks == held  # nothing allocated, nothing freed
        assert merged.slots == slots  # the same pages, now one group
        assert merged.lens_host == [3, 1]

    def test_select_refuses_to_alias_two_rows_onto_one_page(self, mgr):
        """The declared limit: expanding a beam grid repeats row indices, and two
        slots sharing pages would each overwrite the other's K/V."""
        kv = _paged(mgr, 1, prefill_len=1, capacity=8)
        _write(mgr, kv, 1, 1)
        with pytest.raises(RuntimeError, match="repeated row index"):
            kv.select(torch.tensor([0, 0]))

    def test_admission_reserves_the_whole_ceiling(self, mgr):
        """A pool with no eviction must refuse at admission, not run out mid-
        generation: the row it fails is one that is already half-answered."""
        pool_tokens = mgr.pool.num_total_blocks * mgr.pool.config.block_size_frames
        held = _paged(mgr, 1, prefill_len=1, capacity=pool_tokens)  # claims the pool
        assert not mgr.can_admit(max_new_tokens=1, prefill_len=1)
        with pytest.raises(Exception, match="cannot admit"):
            _paged(mgr, 1, prefill_len=1, capacity=4)
        assert held.lens_host == [0]  # keep the reservation alive for the assert

    def test_a_dropped_group_hands_its_pages_back(self, mgr):
        """The backstop: a group that goes out of scope without a select or a
        free must not strand pages — the symptom is 'BlockPool exhausted' on an
        unrelated request much later, which is unattributable."""
        total = mgr.pool.num_total_blocks
        kv = _paged(mgr, 2)
        _write(mgr, kv, 2, 1)
        assert mgr.pool.num_free_blocks < total
        del kv
        assert mgr.pool.num_free_blocks == total

    def test_a_refused_batch_leaves_no_pages_behind(self, mgr):
        free0 = mgr.pool.num_free_blocks
        pool_tokens = mgr.pool.num_total_blocks * mgr.pool.config.block_size_frames
        with pytest.raises(Exception, match="cannot admit"):
            _paged(mgr, 3, prefill_len=1, capacity=pool_tokens)
        assert mgr.pool.num_free_blocks == free0 and mgr.num_active() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# FP8 KV (``EngineConfig.quantization.kv_cache = "fp8_e4m3"``)
# ---------------------------------------------------------------------------


def _fp8_mgr(blocks=16, block_tokens=4, dim=16):
    pool = DecoderKVCacheManager.build_pool(
        num_layers=LAYERS,
        n_kv_head=HEADS,
        head_dim=dim,
        block_tokens=block_tokens,
        max_num_blocks=blocks,
        device=torch.device("cpu"),
        dtype=torch.float8_e4m3fn,
    )
    return DecoderKVCacheManager(pool)


def _rand(batch, t_new, dim, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, HEADS, t_new, dim, generator=g) * 3


def _quantized(x):
    from oasr.functionals.kv_quant import QuantizedKv, reference_quantize_kv

    return QuantizedKv(*reference_quantize_kv(x)).dequantize(torch.float32)


class TestFp8Storage:
    """Bookkeeping of a quantized cache, on CPU through the torch mirror.

    What has to hold is the same contract as the served-dtype cache -- per-row
    offsets, select, merge, page release -- plus two things of its own: the
    prompt is *written* quantized but *attended* fresh, and what a step reads
    back is exactly the quantization of what was written.
    """

    def _dense(self, batch, cap=16, starts=None):
        return DecoderKv.empty(
            LAYERS,
            batch,
            torch.device("cpu"),
            starts=None if starts is None else torch.tensor(starts, dtype=torch.int32),
            cap=cap,
            kv_cache="fp8_e4m3",
        )

    def test_a_quantized_cache_needs_a_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            DecoderKv.empty(LAYERS, 1, torch.device("cpu"), kv_cache="fp8_e4m3")
        with pytest.raises(ValueError, match="unknown kv_cache"):
            DecoderKv.empty(LAYERS, 1, torch.device("cpu"), cap=8, kv_cache="fp4")

    def test_the_prefill_attends_fresh_and_the_step_reads_the_quantization(self):
        from oasr.functionals.kv_quant import QuantizedKv

        kv = self._dense(2)
        prompt = _rand(2, 3, DIM, seed=0)
        assert kv.mask_kwargs(3) == {}, "a prefill is trimmed: causal over its own K/V"
        k, v, extent = kv.append(0, prompt, prompt)
        assert k is prompt and v is prompt and extent is None
        kv.append(1, prompt, prompt)
        kv.commit(3)

        step = _rand(2, 1, DIM, seed=1)
        assert set(kv.mask_kwargs(1)) == {"kv_lens"}
        k, v, extent = kv.append(0, step, step)
        assert isinstance(k, QuantizedKv) and k.data.dtype == torch.float8_e4m3fn
        assert extent == 4
        stored = k.dequantize(torch.float32)
        assert torch.equal(stored[:, :, :3], _quantized(prompt))
        assert torch.equal(stored[:, :, 3:4], _quantized(step))
        assert torch.all(stored[:, :, 4:] == 0), "unused capacity stays zeroed"

    def test_left_padding_keeps_its_window_through_the_prefill(self):
        kv = self._dense(2, starts=[0, 2])
        assert set(kv.mask_kwargs(3)) == {"kv_lens", "kv_starts"}

    def test_divergent_rows_scatter_to_their_own_offsets(self):
        a, b = self._dense(1), self._dense(1)
        for kv, n, seed in ((a, 5, 2), (b, 2, 3)):
            x = _rand(1, n, DIM, seed)
            for layer in range(LAYERS):
                kv.append(layer, x, x)
            kv.commit(n)
        assert a.can_merge(b)
        merged = a.merge(b)
        assert merged.lens_host == [5, 2] and merged.kv_cache == "fp8_e4m3"
        step = _rand(2, 1, DIM, seed=4)
        k, _, _ = merged.append(0, step, step)
        stored = k.dequantize(torch.float32)
        assert torch.equal(stored[0, :, 5], _quantized(step)[0, :, 0])
        assert torch.equal(stored[1, :, 2], _quantized(step)[1, :, 0])

    def test_select_keeps_scales_with_their_rows(self):
        kv = self._dense(3)
        x = _rand(3, 2, DIM, seed=5)
        for layer in range(LAYERS):
            kv.append(layer, x, x)
        kv.commit(2)
        kept = kv.select(torch.tensor([2, 0, 2]))
        step = _rand(3, 1, DIM, seed=6)
        k, _, _ = kept.append(0, step, step)
        stored = k.dequantize(torch.float32)
        want = _quantized(x)
        assert torch.equal(stored[0, :, :2], want[2]) and torch.equal(stored[1, :, :2], want[0])

    def test_a_quantized_and_a_full_precision_cache_do_not_merge(self):
        assert not self._dense(1).can_merge(_kv(1))

    def test_paged_round_trip_through_the_pool(self):
        from oasr.functionals.kv_quant import QuantizedKv

        mgr = _fp8_mgr()
        assert mgr.quantized
        kv = PagedDecoderKv.create(mgr, 2, torch.device("cpu"), prefill_len=3, capacity=12)
        prompt = _rand(2, 3, 16, seed=7)
        mask = kv.mask_kwargs(3)
        assert "block_table" not in mask, "a quantized prefill attends its own K/V densely"
        k, _, _ = kv.append(0, prompt, prompt)
        assert k is prompt
        kv.append(1, prompt, prompt)
        kv.commit(3)
        step = _rand(2, 1, 16, seed=8)
        assert "block_table" in kv.mask_kwargs(1)
        k, v, _ = kv.append(0, step, step)
        assert isinstance(k, QuantizedKv) and isinstance(v, QuantizedKv)
        table = kv.block_table()
        want = torch.cat([_quantized(prompt), _quantized(step)], dim=2)
        pool = k.dequantize(torch.float32)  # (blocks, page, H, D)
        for b in range(2):
            for t in range(4):
                page, off = int(table[b, t // 4]), t % 4
                assert torch.equal(pool[page, off], want[b, :, t])
        kv.free()
        assert mgr.pool.num_free_blocks == mgr.pool.num_blocks

    def test_the_pool_format_and_the_decoder_must_agree(self):
        from oasr.cache import build_kv

        with pytest.raises(ValueError, match="pool is"):
            build_kv(LAYERS, 1, torch.device("cpu"), prefill_len=2, cap=8, manager=_fp8_mgr())
        fp_pool = DecoderKVCacheManager(
            DecoderKVCacheManager.build_pool(
                num_layers=LAYERS,
                n_kv_head=HEADS,
                head_dim=16,
                block_tokens=4,
                max_num_blocks=8,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
        with pytest.raises(ValueError, match="pool is"):
            build_kv(
                LAYERS,
                1,
                torch.device("cpu"),
                prefill_len=2,
                cap=8,
                manager=fp_pool,
                kv_cache="fp8_e4m3",
            )

    def test_the_pool_holds_one_scale_per_slot_and_head(self):
        mgr = _fp8_mgr(blocks=5, block_tokens=4, dim=16)
        k, v = mgr.kv_view(0)
        assert k.data.shape == (5, 4, HEADS, 16) and k.scale.shape == (5, 4, HEADS)
        assert k.scale.dtype == torch.float32
        with pytest.raises(RuntimeError, match="no scales"):
            mgr_fp = DecoderKVCacheManager.build_pool(
                num_layers=1,
                n_kv_head=1,
                head_dim=8,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            mgr_fp.get_kv_scale_view(0)
