"""The C++ attention lane's single-query decode kernel (``include/oasr/attention/fmha_decode.h``).

What it inherits from the MMA path it replaces for ``T_q == 1``: the masking
contract (per-row lengths, left padding, a row with no live key is exactly zero),
GQA, dense and paged K/V at any page size, every padded head dim up to 256, fp16
and bf16, a destination with strides.  What is new: the key range is split
across a thread-block cluster and merged in the kernel, by a count that is a pure
function of shape and SM count (``AGENTS.md`` rule 11), so a replayed graph and
eager agree bit for bit.
"""

from __future__ import annotations

import math

import pytest
import torch
from helpers import FMHA_TOL, assert_graph_replay, capture_graph, device_sm, ref_fmha

from oasr.jit import fmha as jf
from oasr.jit.attention_select import AttnChoice

# =============================================================================
# The split rule (no device)
# =============================================================================


class TestSplitRule:
    @pytest.mark.parametrize(
        "ctas, t_k, want",
        [
            (6, 1536, 8),  # starved (Whisper-tiny B1): as many as a cluster holds
            (24, 1536, 6),
            (48, 1536, 3),
            (96, 1536, 2),
            (127, 1536, 2),  # just under 75 % of 170 SMs
            (128, 1536, 1),  # 75 %: the unsplit grid is enough
            (6, 383, 1),  # below the shortest range worth splitting
            (6, 512, 4),  # every split at least 128 keys
        ],
    )
    def test_examples_on_170_sms(self, ctas, t_k, want):
        assert jf.decode_num_splits(ctas, t_k, 170) == want

    def test_bounds_over_a_grid(self):
        for sms in (46, 84, 132, 170):
            for ctas in (1, 2, 3, 6, 20, 32, 64, 96, 128, 200, 640):
                for t_k in (1, 64, 383, 384, 500, 1024, 1536, 4096, 32768):
                    s = jf.decode_num_splits(ctas, t_k, sms)
                    assert 1 <= s <= jf.DECODE_MAX_SPLITS
                    if 100 * ctas >= jf.DECODE_FILL_PCT * sms or t_k < jf.DECODE_MIN_KEYS_FOR_SPLIT:
                        assert s == 1
                    if s > 1:
                        assert s * jf.DECODE_MIN_KEYS_PER_SPLIT <= t_k

    @pytest.mark.parametrize(
        "splits, t_k, want", [(8, 1536, 8), (8, 256, 2), (8, 100, 1), (16, 4096, 8), (1, 4096, 1)]
    )
    def test_a_measured_setting_is_clamped(self, splits, t_k, want):
        assert jf.clamp_decode_splits(splits, t_k) == want

    def test_supported_cells(self):
        assert jf.decode_supported(sm=120, dtype_str="float16", head_dim=64)
        assert jf.decode_supported(sm=120, dtype_str="bfloat16", head_dim=256)
        assert not jf.decode_supported(sm=89, dtype_str="float16", head_dim=64)  # no cluster
        assert not jf.decode_supported(sm=120, dtype_str="float32", head_dim=64)
        assert not jf.decode_supported(sm=120, dtype_str="float16", head_dim=12)
        assert not jf.decode_supported(sm=120, dtype_str="float16", head_dim=264)


# =============================================================================
# The kernel (CUDA)
# =============================================================================


def _requires_decode():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if device_sm() not in jf.DECODE_KERNEL_SM:
        pytest.skip(f"the decode kernel is not routed to on sm_{device_sm()}")


def _step(B, H, T, D, *, H_kv=None, dtype=torch.float16, page=0, left_pad=False, seed=0):
    """A decode step's operands, dense or paged, and the dense K/V the reference reads."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    H_kv = H_kv or H
    q = torch.randn(B, H, 1, D, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=dtype, generator=g)
    lens = torch.randint(T // 2, T + 1, (B,), device="cuda", dtype=torch.int32, generator=g)
    lens[0] = T
    starts = None
    if left_pad:
        starts = torch.randint(0, T // 4, (B,), device="cuda", dtype=torch.int32, generator=g)
        starts[0] = 0
    ops = {"q": q, "k": k, "v": v, "cache_seqlens": lens, "cache_seqstarts": starts}
    ref_kv = (k, v)
    if page:
        per = T // page
        perm = torch.randperm(B * per, device="cuda", generator=g) + 1
        table = perm.view(B, per).to(torch.int32)
        pool_k = torch.zeros(B * per + 1, page, H_kv, D, device="cuda", dtype=dtype)
        pool_v = torch.zeros_like(pool_k)
        for b in range(B):
            for i in range(per):
                rows = slice(i * page, (i + 1) * page)
                pool_k[table[b, i]] = k[b, :, rows].transpose(0, 1)
                pool_v[table[b, i]] = v[b, :, rows].transpose(0, 1)
        ops.update(k=pool_k, v=pool_v, block_table=table)
    return ops, ref_kv


def _run(ops, splits=0, **kw):
    from oasr.functionals.attention import fmha

    D = ops["q"].size(-1)
    return fmha(
        ops["q"],
        ops["k"],
        ops["v"],
        softmax_scale=1.0 / math.sqrt(D),
        cache_seqlens=ops["cache_seqlens"],
        cache_seqstarts=ops.get("cache_seqstarts"),
        block_table=ops.get("block_table"),
        backend=AttnChoice("cxx", splits=splits),
        **kw,
    )


def _ref(ops, ref_kv):
    D = ops["q"].size(-1)
    return ref_fmha(
        ops["q"],
        *ref_kv,
        1.0 / math.sqrt(D),
        cache_seqlens=ops["cache_seqlens"],
        cache_seqstarts=ops.get("cache_seqstarts"),
    )


class TestParity:
    @pytest.fixture(autouse=True)
    def _gate(self):
        _requires_decode()

    @pytest.mark.parametrize("splits", [0, 1, 3, 8])
    @pytest.mark.parametrize("page", [0, 16, 32])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_layouts_dtypes_and_splits(self, dtype, page, splits):
        ops, ref_kv = _step(5, 6, 1024, 64, dtype=dtype, page=page)
        torch.testing.assert_close(_run(ops, splits), _ref(ops, ref_kv), **FMHA_TOL)

    @pytest.mark.parametrize("D", [32, 64, 80, 96, 128, 256])
    def test_head_dims(self, D):
        """Every padded head dim; 80 runs predicated in the 96 cell."""
        ops, ref_kv = _step(3, 4, 640, D, page=16)
        for splits in (1, 4):
            torch.testing.assert_close(_run(ops, splits), _ref(ops, ref_kv), **FMHA_TOL)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gqa_and_left_padding(self, dtype):
        ops, ref_kv = _step(4, 8, 768, 128, H_kv=2, dtype=dtype, page=16, left_pad=True)
        for splits in (0, 1, 6):
            torch.testing.assert_close(_run(ops, splits), _ref(ops, ref_kv), **FMHA_TOL)

    @pytest.mark.parametrize("page", [0, 16])
    def test_a_row_with_no_live_key_is_zero(self, page):
        ops, ref_kv = _step(4, 6, 1024, 64, page=page, left_pad=True)
        ops["cache_seqlens"][1] = ops["cache_seqstarts"][1]  # an empty window
        ops["cache_seqlens"][2] = 0
        ops["cache_seqstarts"][2] = 0
        for splits in (1, 8):
            out = _run(ops, splits)
            assert torch.isfinite(out).all()
            assert torch.equal(out[1:3], torch.zeros_like(out[1:3]))
            torch.testing.assert_close(out, _ref(ops, ref_kv), **FMHA_TOL)

    def test_a_strided_destination(self):
        ops, ref_kv = _step(3, 6, 512, 64, page=16)
        wide = torch.full((3, 6, 2, 64), float("nan"), device="cuda", dtype=torch.float16)
        out = wide[:, :, :1]
        _run(ops, 2, out=out)
        torch.testing.assert_close(out, _ref(ops, ref_kv), **FMHA_TOL)
        assert torch.isnan(wide[:, :, 1:]).all()  # nothing written past the view


class TestRouting:
    """``T_q == 1`` with no bias and no mask takes the decode kernel; nothing else does."""

    @pytest.fixture(autouse=True)
    def _gate(self, monkeypatch):
        _requires_decode()
        self.calls = 0
        real = jf.get_fmha_decode_fn

        def spy(**kw):
            self.calls += 1
            return real(**kw)

        monkeypatch.setattr(jf, "get_fmha_decode_fn", spy)

    def test_a_decode_step_takes_it(self):
        ops, _ = _step(2, 6, 512, 64, page=16)
        _run(ops)
        assert self.calls == 1

    def test_two_query_rows_a_bias_or_a_mask_do_not(self):
        from oasr.functionals.attention import fmha

        ops, ref_kv = _step(2, 6, 512, 64)
        q2 = torch.cat([ops["q"], ops["q"]], dim=2)
        bias = torch.zeros(2, 6, 1, 512, device="cuda", dtype=torch.float16)
        scale = 0.125
        common = {"cache_seqlens": ops["cache_seqlens"], "backend": "cxx"}
        fmha(q2, ops["k"], ops["v"], softmax_scale=scale, **common)
        fmha(ops["q"], ops["k"], ops["v"], softmax_scale=scale, attn_bias=bias, **common)
        fmha(ops["q"], ops["k"], ops["v"], softmax_scale=scale, causal=True, **common)
        fmha(ops["q"], ops["k"], ops["v"], softmax_scale=scale, window_left=64, **common)
        assert self.calls == 0

    def test_a_cell_built_with_routing_off_still_carries_the_kernel(self, monkeypatch):
        """What a cell contains must not depend on what is routed.

        A cell generated while ``DECODE_KERNEL_SM`` was empty (a test naming the
        MMA path) was cached without the decode symbols, and every later decode
        call in the process failed to find them.
        """
        monkeypatch.setattr(jf, "DECODE_KERNEL_SM", ())
        jf.gen_fmha_module.cache_clear()
        jf._cell_module.cache_clear()
        ops, ref_kv = _step(2, 6, 512, 64, page=16)
        _run(ops)  # builds (or loads) the cell with routing off
        monkeypatch.setattr(jf, "DECODE_KERNEL_SM", (device_sm(),))
        torch.testing.assert_close(_run(ops), _ref(ops, ref_kv), **FMHA_TOL)
        assert self.calls == 1

    def test_an_unmeasured_architecture_keeps_the_mma_path(self, monkeypatch):
        monkeypatch.setattr(jf, "DECODE_KERNEL_SM", ())
        ops, ref_kv = _step(2, 6, 512, 64, page=16)
        torch.testing.assert_close(_run(ops), _ref(ops, ref_kv), **FMHA_TOL)
        assert self.calls == 0


class TestGraphs:
    @pytest.fixture(autouse=True)
    def _gate(self):
        _requires_decode()

    @pytest.mark.parametrize("splits", [1, 6])
    def test_replay_recomputes_and_equals_eager(self, splits):
        ops, ref_kv = _step(2, 6, 1024, 64, page=16)
        out = torch.empty_like(ops["q"])

        def launch():
            return _run(ops, splits, out=out)

        eager = launch().clone()
        assert torch.equal(launch(), eager)  # deterministic
        graph, _ = capture_graph(launch)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)  # the same kernel, the same merge order

        def mutate():
            ops["k"].copy_(torch.randn_like(ops["k"]))
            for b in range(ops["q"].size(0)):
                for i in range(ops["block_table"].size(1)):
                    rows = slice(i * 16, (i + 1) * 16)
                    ref_kv[0][b, :, rows] = ops["k"][ops["block_table"][b, i]].transpose(0, 1)

        assert_graph_replay(
            launch, out=out, mutate=mutate, expected=lambda: _ref(ops, ref_kv), **FMHA_TOL
        )
