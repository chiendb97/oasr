# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""FP8 KV cache: the quantize-store kernel and the fp8-KV attention variants.

The two halves of ``EngineConfig.quantization.kv_cache = "fp8_e4m3"`` at the
kernel level:

* ``oasr.functionals.kv_quant.quantize_kv_store`` -- one power-of-two scale per
  (token, kv-head), scattered into a dense capacity buffer or a paged pool.  Its
  torch mirror is the oracle, and the claim is **bit** equality.
* ``oasr.fmha`` over ``float8_e4m3fn`` K/V -- the C++ lane's fp8 variants, which
  dequantize inside the tile loop.  The sharp instrument here is not a
  tolerance: because the scales are powers of two, dequantizing is exact, so
  the fp8 kernel must equal the **served-dtype kernel fed the dequantized
  cache**, ``atol=0``.  Any difference is the fp8 gather or the in-tile
  conversion and nothing else.  A looser comparison against attention over the
  original fp16 K/V bounds what the format itself costs.

Also pinned: the resolver mirror, the predication of rows past a stream's length
(stale NaN pages must not reach the output), CUDA-graph replay, and the shapes
the lane refuses by name.
"""

from __future__ import annotations

import functools
import math

import pytest
import torch
from helpers import assert_graph_replay, device_sm

from oasr.functionals.attention import fmha
from oasr.functionals.kv_quant import (
    QuantizedKv,
    kv_cache_bytes_per_token,
    quantize_kv,
    quantize_kv_store,
    reference_quantize_kv,
)

_SUPPORTED = (80, 86, 89, 120)
FP8 = torch.float8_e4m3fn


def _requires_lane():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if device_sm() not in _SUPPORTED:
        pytest.skip(f"the C++ FMHA lane is not compiled for sm_{device_sm()}")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == FP8 else t


# ---------------------------------------------------------------------------
# Quantize-store
# ---------------------------------------------------------------------------


class TestReferenceQuantizer:
    """The numerics the kernel is pinned to, checked on CPU."""

    def test_scales_are_the_least_power_of_two_covering_the_amax(self):
        torch.manual_seed(0)
        x = torch.randn(4, 3, 9, 64) * torch.logspace(-8, 4, 9).view(1, 1, 9, 1)
        q, scale = reference_quantize_kv(x)
        mantissa, _ = torch.frexp(scale)
        assert torch.all(mantissa == 0.5), "every scale must be a power of two"
        amax = x.abs().amax(-1)
        assert torch.all(448.0 * scale >= amax)
        assert torch.all(448.0 * scale / 2 < amax), "a smaller power of two would also cover"

    def test_dequantization_is_exact_and_within_e4m3_precision(self):
        torch.manual_seed(1)
        x = torch.randn(2, 4, 33, 128) * 7.0
        q, scale = reference_quantize_kv(x)
        deq = QuantizedKv(q, scale).dequantize(torch.float32)
        # Exact: q * 2^k is representable, so fp32 and fp64 agree.
        exact = q.double() * scale.double().unsqueeze(-1)
        assert torch.equal(deq.double(), exact)
        # Normal e4m3 values round to within half an ulp: 2^-4 relative.
        big = x.abs() > scale.unsqueeze(-1) * 2.0**-6
        rel = ((deq - x).abs() / x.abs())[big]
        assert rel.max().item() <= 2.0**-4 + 1e-7

    def test_an_all_zero_vector_stays_zero(self):
        q, scale = reference_quantize_kv(torch.zeros(1, 2, 3, 16))
        assert torch.all(q.float() == 0) and torch.all(torch.isfinite(scale))


@pytest.mark.cuda
class TestQuantizeStoreKernel:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("head_dim", [16, 24, 32, 64, 128, 256])
    def test_dense_matches_the_reference_bit_for_bit(self, dtype, head_dim):
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        torch.manual_seed(head_dim)
        scales = torch.logspace(-6, 3, 7, device="cuda").view(1, 1, 7, 1)
        x = (torch.randn(3, 5, 7, head_dim, device="cuda") * scales).to(dtype)
        x[0, 0, 0] = 0
        got = quantize_kv(x)
        q, scale = reference_quantize_kv(x)
        assert torch.equal(_bits(got.data), _bits(q))
        assert torch.equal(got.scale, scale)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_paged_scatter_lands_where_the_positions_say(self, dtype):
        """A strided head-split input, scattered into a shared pool by slot."""
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        torch.manual_seed(3)
        B, T, H, D, slots = 2, 9, 4, 64, 100
        # (B, T, H, 3, D)[..., 1, :] -> (B, H, T, D): every stride non-trivial.
        k = torch.randn(B, T, H, 3, D, device="cuda", dtype=dtype)[:, :, :, 1].permute(0, 2, 1, 3)
        v = torch.randn(B, T, H, 3, D, device="cuda", dtype=dtype)[:, :, :, 2].permute(0, 2, 1, 3)
        pos = torch.randperm(slots, device="cuda")[: B * T]

        def pool():
            return (
                torch.zeros(1, H, slots, D, dtype=FP8, device="cuda"),
                torch.zeros(1, H, slots, device="cuda"),
            )

        (ko, ks), (vo, vs) = pool(), pool()
        quantize_kv_store(k, ko, ks, v, vo, vs, pos=pos)
        for x, out, sc in ((k, ko, ks), (v, vo, vs)):
            q, scale = reference_quantize_kv(x)
            flat = pos.view(B, T)
            for b in range(B):
                assert torch.equal(_bits(out[0][:, flat[b]]), _bits(q[b]))
                assert torch.equal(sc[0][:, flat[b]], scale[b])
            untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
            untouched[pos] = False
            assert torch.all(out[0][:, untouched].float() == 0)

    def test_the_cpu_path_is_the_same_function(self):
        """CPU / fp32 take the torch mirror; it must address slots identically."""
        torch.manual_seed(4)
        x = torch.randn(2, 3, 5, 32)
        out = torch.zeros(2, 3, 12, 32, dtype=FP8)
        sc = torch.zeros(2, 3, 12)
        pos = torch.tensor([[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]])
        quantize_kv_store(x, out, sc, pos=pos.reshape(-1))
        q, scale = reference_quantize_kv(x)
        assert torch.equal(_bits(out[0, :, 1:6]), _bits(q[0]))
        assert torch.equal(_bits(out[1, :, 7:12]), _bits(q[1]))
        assert torch.equal(sc[1, :, 7:12], scale[1])


def test_bytes_per_token_counts_fp8_plus_one_scale_per_head():
    common = {"num_layers": 4, "n_kv_head": 6, "head_dim": 64}
    fp16 = kv_cache_bytes_per_token(dtype=torch.float16, kv_cache=None, **common)
    fp8 = kv_cache_bytes_per_token(dtype=torch.float16, kv_cache="fp8_e4m3", **common)
    assert fp16 == 2 * 4 * 6 * 64 * 2
    assert fp8 == 2 * 4 * 6 * (64 + 4)
    assert fp16 / fp8 == pytest.approx(128 / 68)


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _config_probe():
    from oasr.jit import env
    from oasr.jit.core import gen_jit_spec

    return gen_jit_spec(
        "fmha_config_probe", [env.OASR_CSRC_DIR / "fmha_jit_binding.cu"]
    ).build_and_load()


@pytest.mark.cuda
class TestKv8TileSelectionAgrees:
    """``fmhaResolveTileKv8`` and its Python mirror, field for field, over four parts."""

    @pytest.mark.parametrize("sm", [80, 86, 89, 120])
    @pytest.mark.parametrize("head_dim", [16, 32, 48, 64, 80, 96, 128, 192, 256, 320])
    def test_python_mirror_matches_cxx(self, sm, head_dim):
        if not torch.cuda.is_available():
            pytest.skip("the probe module is JIT-built with nvcc")
        from oasr.jit.fmha import resolve_config_kv8

        out = torch.zeros(7, dtype=torch.int32)
        _config_probe().fmha_kv8_resolved_config(out, sm, 16, head_dim)
        want = resolve_config_kv8(sm, 16, head_dim)
        fields = [
            int(want.valid),
            want.block_m,
            want.block_n,
            want.num_warps,
            want.num_stages,
            int(want.q_in_regs),
            want.smem_bytes,
        ]
        assert out.tolist() == fields

    @pytest.mark.parametrize("sm", [80, 86, 89, 120])
    @pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
    def test_shipped_head_dims_keep_the_served_dtype_k_tile(self, sm, head_dim):
        """Same ``block_n`` as the fp16 kernel: the oracle below depends on it,
        because the split count and the online-softmax tiling both follow it."""
        from oasr.jit.fmha import resolve_config, resolve_config_kv8

        fp8, fp16 = resolve_config_kv8(sm, 16, head_dim), resolve_config(sm, 16, head_dim)
        assert fp8.valid and fp16.valid
        assert fp8.block_n == fp16.block_n


# ---------------------------------------------------------------------------
# Attention over an fp8 cache
# ---------------------------------------------------------------------------


def _dense_case(B, H, H_kv, T_q, T_k, D, dtype, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H, T_q, D, generator=g).to("cuda", dtype)
    k = (torch.randn(B, H_kv, T_k, D, generator=g) * 3).to("cuda", dtype)
    v = torch.randn(B, H_kv, T_k, D, generator=g).to("cuda", dtype)
    return q, k, v


def _paged_case(B, H, H_kv, T_q, D, block, lens, dtype, seed):
    """A shuffled pool, its fp8 copy, and a block table covering ``lens``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nblk = sum(-(-n // block) for n in lens) + 3
    k = (torch.randn(nblk, block, H_kv, D, generator=g) * 3).to("cuda", dtype)
    v = torch.randn(nblk, block, H_kv, D, generator=g).to("cuda", dtype)
    k8, v8 = [QuantizedKv(*reference_quantize_kv(t)) for t in (k, v)]
    perm = torch.randperm(nblk, generator=g).tolist()
    width = max(-(-n // block) for n in lens)
    table = torch.zeros(B, width, dtype=torch.int32)
    used = 0
    for b, n in enumerate(lens):
        for j in range(-(-n // block)):
            table[b, j] = perm[used]
            used += 1
    q = torch.randn(B, H, T_q, D, generator=g).to("cuda", dtype)
    seqlens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return q, (k, v), (k8, v8), table.to("cuda"), seqlens


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


@pytest.mark.cuda
class TestAgainstTheDequantizedOracle:
    """fp8 kernel == the served-dtype kernel on the dequantized cache, exactly."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("D", [32, 64, 128])
    @pytest.mark.parametrize(
        "B,H,H_kv,T_q,T_k,lens,starts",
        [
            (2, 4, 2, 1, 37, True, False),  # GQA decode step, ragged lengths
            (2, 4, 4, 5, 300, True, True),  # left padding (the speech-LLM's)
            (3, 4, 1, 70, 129, False, False),  # MQA, two Q tiles, whole extent
            (1, 6, 6, 1, 1500, True, False),  # whisper-tiny cross-attention step: splits
            (1, 20, 20, 4, 1500, False, False),  # whisper-large cross-attention prefill
            (4, 2, 2, 1, 448, True, True),
        ],
    )
    def test_dense(self, dtype, D, B, H, H_kv, T_q, T_k, lens, starts):
        _requires_lane()
        q, k, v = _dense_case(B, H, H_kv, T_q, T_k, D, dtype, seed=B * 100 + T_k)
        k8, v8 = quantize_kv(k), quantize_kv(v)
        kw = {}
        if lens:
            kw["cache_seqlens"] = torch.tensor(
                [max(1, T_k - 3 * i) for i in range(B)], dtype=torch.int32, device="cuda"
            )
        if starts:
            kw["cache_seqstarts"] = torch.tensor(
                [min(2 * i, T_k - 1) for i in range(B)], dtype=torch.int32, device="cuda"
            )
        scale = D**-0.5
        got = fmha(
            q, k8.data, v8.data, k_scale=k8.scale, v_scale=v8.scale, softmax_scale=scale, **kw
        )
        want = fmha(
            q, k8.dequantize(dtype), v8.dequantize(dtype), softmax_scale=scale, backend="cxx", **kw
        )
        assert torch.equal(got, want)
        # And what the format costs against the unquantized cache.
        full = fmha(q, k, v, softmax_scale=scale, backend="cxx", **kw)
        assert _rel_err(got, full) < 0.15

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("D", [64, 128])
    @pytest.mark.parametrize(
        "B,H,H_kv,T_q,block,lens",
        [
            (2, 4, 2, 1, 16, (37, 5)),  # table one page wider for row 0
            (3, 6, 6, 4, 16, (1500, 700, 33)),  # cross-attention prefill, splits
            (1, 20, 20, 1, 16, (1500,)),
            (2, 4, 4, 1, 64, (100, 64)),  # a page as wide as the K tile
            (2, 4, 1, 1, 8, (100, 3)),  # a page narrower than a K tile: 8 per tile
            (2, 4, 4, 1, 16, (17, 1)),  # width 2 pages, not a whole K tile
        ],
    )
    def test_paged(self, dtype, D, B, H, H_kv, T_q, block, lens):
        _requires_lane()
        q, _, (k8, v8), table, seqlens = _paged_case(
            B, H, H_kv, T_q, D, block, lens, dtype, seed=sum(lens)
        )
        scale = D**-0.5
        got = fmha(
            q,
            k8.data,
            v8.data,
            k_scale=k8.scale,
            v_scale=v8.scale,
            softmax_scale=scale,
            cache_seqlens=seqlens,
            block_table=table,
        )
        want = fmha(
            q,
            k8.dequantize(dtype),
            v8.dequantize(dtype),
            softmax_scale=scale,
            cache_seqlens=seqlens,
            block_table=table,
            backend="cxx",
        )
        assert torch.equal(got, want)

    def test_the_sdpa_reference_agrees_within_attention_tolerance(self):
        """``backend='sdpa'`` dequantizes and runs SDPA: the CPU/fp32 oracle."""
        _requires_lane()
        q, k, v = _dense_case(2, 4, 2, 3, 200, 64, torch.float16, seed=9)
        k8, v8 = quantize_kv(k), quantize_kv(v)
        lens = torch.tensor([200, 150], dtype=torch.int32, device="cuda")
        args = {
            "k_scale": k8.scale,
            "v_scale": v8.scale,
            "softmax_scale": 0.125,
            "cache_seqlens": lens,
        }
        got = fmha(q, k8.data, v8.data, **args)
        ref = fmha(q, k8.data, v8.data, backend="sdpa", **args)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.cuda
class TestMaskedRowsNeverReachTheOutput:
    """Rows past a stream's length -- and below its start -- carry garbage in a
    real pool.  They must contribute nothing: K is masked, and V must arrive as
    zeros (a stale NaN would enter through ``P @ V``, where no mask can stop it)."""

    @pytest.mark.parametrize("D", [64, 128])
    def test_nan_garbage_past_the_length_is_inert(self, D):
        _requires_lane()
        B, H, block, lens = 2, 4, 16, (37, 5)
        q, _, (k8, v8), table, seqlens = _paged_case(
            B, H, H, 1, D, block, lens, torch.float16, seed=5
        )
        args = {"softmax_scale": D**-0.5, "cache_seqlens": seqlens, "block_table": table}
        clean = fmha(q, k8.data, v8.data, k_scale=k8.scale, v_scale=v8.scale, **args)
        dirty = [QuantizedKv(t.data.clone(), t.scale.clone()) for t in (k8, v8)]
        for b, n in enumerate(lens):
            # The tail of each row's own last page: real pool memory the row
            # owns, past its length -- what a recycled page holds in production.
            for pos in range(n, -(-n // block) * block):
                page = int(table[b, pos // block])
                for t in dirty:
                    t.data.view(torch.uint8)[page, pos % block] = 0x7F  # e4m3 NaN
                    t.scale[page, pos % block] = float("nan")
        got = fmha(
            q, dirty[0].data, dirty[1].data, k_scale=dirty[0].scale, v_scale=dirty[1].scale, **args
        )
        assert torch.isfinite(got).all()
        assert torch.equal(got, clean)

    def test_an_empty_window_is_zeros_not_nan(self):
        """``seqstart == seqlen``: every key masked.  The empty-row clamp must stay
        local to the tile (AGENTS rule 10) and the row comes back zero."""
        _requires_lane()
        q, k, v = _dense_case(2, 2, 2, 1, 64, 64, torch.float16, seed=6)
        k8, v8 = quantize_kv(k), quantize_kv(v)
        lens = torch.tensor([10, 64], dtype=torch.int32, device="cuda")
        starts = torch.tensor([10, 0], dtype=torch.int32, device="cuda")
        out = fmha(
            q,
            k8.data,
            v8.data,
            k_scale=k8.scale,
            v_scale=v8.scale,
            softmax_scale=0.125,
            cache_seqlens=lens,
            cache_seqstarts=starts,
        )
        assert torch.all(out[0] == 0)
        assert torch.isfinite(out).all() and out[1].abs().sum() > 0


@pytest.mark.cuda
class TestGraphReplay:
    def test_a_captured_paged_decode_step_recomputes(self):
        """Lengths, table and cache bytes all change through the captured buffers;
        the replay must follow them (no host-side trip count, no allocation)."""
        _requires_lane()
        D, block = 64, 16
        q, _, (k8, v8), table, seqlens = _paged_case(
            3, 6, 6, 1, D, block, (1500, 700, 33), torch.float16, seed=7
        )
        out = torch.empty_like(q)

        def launch():
            fmha(
                q,
                k8.data,
                v8.data,
                k_scale=k8.scale,
                v_scale=v8.scale,
                softmax_scale=D**-0.5,
                cache_seqlens=seqlens,
                block_table=table,
                out=out,
            )

        def mutate():
            seqlens.copy_(torch.tensor([1400, 701, 20], dtype=torch.int32, device="cuda"))
            table.copy_(table.flip(0).roll(1, dims=1))
            k8.scale.mul_(2.0)
            q.normal_()

        def expected():
            return fmha(
                q,
                k8.dequantize(torch.float16),
                v8.dequantize(torch.float16),
                softmax_scale=D**-0.5,
                cache_seqlens=seqlens,
                block_table=table,
                backend="cxx",
            )

        assert_graph_replay(launch, out=out, mutate=mutate, expected=expected, atol=0, rtol=0)


@pytest.mark.cuda
class TestRefusedByName:
    """What the fp8 variants do not render raises -- never a silent fallback (rule 3)."""

    def _kv(self, D=64):
        q, k, v = _dense_case(1, 2, 2, 4, 32, D, torch.float16, seed=8)
        return q, quantize_kv(k), quantize_kv(v)

    @pytest.mark.parametrize(
        "kwargs,needle",
        [
            ({"causal": True}, "causal or windowed"),
            ({"window_left": 8, "window_right": 0}, "causal or windowed"),
            ({"attn_bias": "bias"}, "bias"),
        ],
    )
    def test_unrendered_masks(self, kwargs, needle):
        _requires_lane()
        q, k8, v8 = self._kv()
        if kwargs.get("attn_bias") == "bias":
            kwargs = {"attn_bias": torch.zeros(1, 2, 4, 32, dtype=torch.float16, device="cuda")}
        with pytest.raises(NotImplementedError, match=needle):
            fmha(
                q, k8.data, v8.data, k_scale=k8.scale, v_scale=v8.scale, softmax_scale=1.0, **kwargs
            )

    @pytest.mark.parametrize("D", [80, 96])
    def test_a_head_dim_the_gather_cannot_tile(self, D):
        _requires_lane()
        q, k8, v8 = self._kv(D)
        with pytest.raises(NotImplementedError, match="fp8 KV cannot be served"):
            fmha(q, k8.data, v8.data, k_scale=k8.scale, v_scale=v8.scale, softmax_scale=1.0)

    def test_naming_the_cutedsl_lane(self):
        _requires_lane()
        q, k8, v8 = self._kv()
        with pytest.raises(NotImplementedError, match="CuTeDSL"):
            fmha(
                q,
                k8.data,
                v8.data,
                k_scale=k8.scale,
                v_scale=v8.scale,
                softmax_scale=1.0,
                backend="cute",
            )

    def test_scales_are_required(self):
        _requires_lane()
        q, k8, v8 = self._kv()
        with pytest.raises(ValueError, match="k_scale and v_scale"):
            fmha(q, k8.data, v8.data, softmax_scale=1.0)

    def test_the_reference_serves_what_the_kernel_refuses(self):
        """A *named* reference is the one place a causal fp8 read is answered."""
        _requires_lane()
        q, k8, v8 = self._kv()
        q = q[:, :, :4]
        out = fmha(
            q,
            k8.data[:, :, :4],
            v8.data[:, :, :4],
            k_scale=k8.scale[:, :, :4],
            v_scale=v8.scale[:, :, :4],
            softmax_scale=1.0,
            causal=True,
            backend="sdpa",
        )
        assert torch.isfinite(out).all()


def test_cpu_fp32_reads_through_the_dequantizing_reference():
    torch.manual_seed(10)
    q = torch.randn(2, 4, 1, 32)
    k, v = torch.randn(2, 2, 20, 32), torch.randn(2, 2, 20, 32)
    k8, v8 = [QuantizedKv(*reference_quantize_kv(t)) for t in (k, v)]
    lens = torch.tensor([20, 11], dtype=torch.int32)
    out = fmha(
        q,
        k8.data,
        v8.data,
        k_scale=k8.scale,
        v_scale=v8.scale,
        softmax_scale=32**-0.5,
        cache_seqlens=lens,
    )
    ref = fmha(
        q,
        k8.dequantize(torch.float32),
        v8.dequantize(torch.float32),
        softmax_scale=32**-0.5,
        cache_seqlens=lens,
    )
    assert torch.equal(out, ref)
    assert math.isfinite(out.abs().max().item())
