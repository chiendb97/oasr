"""Parity tests for ``oasr.fmha`` against an SDPA reference.

These tests run against whichever backend ``OASR_ATTN_BACKEND`` selects:
* ``sdpa`` -- exercises the fallback path; verifies the public functional
  API + wrapper integration produce numerically identical results to the
  legacy SDPA call.
* ``cute`` -- exercises the SM120 CuteDSL kernel.

Run with::

    pytest tests/test_fmha.py -v                          # default backend
    OASR_ATTN_BACKEND=cute pytest tests/test_fmha.py -v  # force cute backend
"""

from __future__ import annotations

import math
import os
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Reference: a clean SDPA path that mirrors oasr.fmha_forward's contract.
# Used to compare both backends against a single source of truth.
# ---------------------------------------------------------------------------


def _ref_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    attn_bias: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, H, T_q, D = q.shape
    H_kv = k.size(1)
    T_k = k.size(2)
    if H % H_kv != 0:
        raise ValueError("H % H_kv != 0")
    if H_kv != H:
        n_repeat = H // H_kv
        k = k.repeat_interleave(n_repeat, dim=1)
        v = v.repeat_interleave(n_repeat, dim=1)

    masks = []
    if attn_bias is not None:
        masks.append(attn_bias.to(q.dtype))
    if cache_seqlens is not None:
        arange = torch.arange(T_k, device=cache_seqlens.device)
        keep = arange.unsqueeze(0) < cache_seqlens.unsqueeze(1)
        pad = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
        pad = pad.unsqueeze(1).unsqueeze(1)  # (B,1,1,T_k)
        masks.append(pad)

    full_mask = None
    if masks:
        full_mask = masks[0]
        for m in masks[1:]:
            full_mask = full_mask + m

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=full_mask,
        scale=softmax_scale,
    )


# ---------------------------------------------------------------------------
# Test parameter grid
# ---------------------------------------------------------------------------

_SHAPES = [
    # (B, H, H_kv, T_q, T_k, D)
    (1, 4, 4, 8, 16, 64),  # smallest streaming chunk
    (4, 4, 4, 8, 64, 64),  # bigger batch
    (1, 4, 4, 16, 32, 64),  # T_q not 8
    (2, 8, 8, 8, 128, 64),  # bigger H
    (2, 8, 1, 8, 64, 64),  # MQA
    (2, 8, 2, 8, 64, 64),  # GQA
    (1, 4, 4, 64, 256, 64),  # offline-ish shape
    (1, 4, 4, 16, 249, 64),  # T_k not divisible by 8 (real audio frame counts)
    (2, 4, 4, 16, 33, 64),  # tiny odd T_k
]

_DTYPES = [torch.float16]
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    _DTYPES.append(torch.bfloat16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def fmha():
    """Import oasr.fmha_forward and force a fresh backend probe."""
    # Force a re-read of the env var since other tests may have set/unset it.
    from oasr.jit.attention import set_backend_mode

    mode = os.environ.get("OASR_ATTN_BACKEND", "auto").lower()
    set_backend_mode(mode)
    from oasr import fmha

    return fmha


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_fmha_offline(fmha, cuda, dtype, shape):
    """Offline mode: no bias, no length mask."""
    B, H, H_kv, T_q, T_k, D = shape
    torch.manual_seed(0)
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    out = fmha(q, k, v, softmax_scale=scale)
    ref = _ref_fmha(q, k, v, scale)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_fmha_with_bias(fmha, cuda, dtype, shape):
    """Offline + additive bias (rel-pos style)."""
    B, H, H_kv, T_q, T_k, D = shape
    torch.manual_seed(1)
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    bias = torch.randn(B, H, T_q, T_k, device=cuda, dtype=dtype) * 0.1
    scale = 1.0 / math.sqrt(D)

    out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
    ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_fmha_with_length_mask(fmha, cuda, dtype, shape):
    """Per-stream length mask via cache_seqlens (heterogeneous)."""
    B, H, H_kv, T_q, T_k, D = shape
    torch.manual_seed(2)
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    # Half the streams get a short context (~1/4 of T_k), the rest full T_k.
    base = max(1, T_k // 4)
    seqlens = torch.tensor(
        [base if i < B // 2 else T_k for i in range(B)],
        dtype=torch.int32,
        device=cuda,
    )
    scale = 1.0 / math.sqrt(D)

    out = fmha(q, k, v, softmax_scale=scale, cache_seqlens=seqlens)
    ref = _ref_fmha(q, k, v, scale, cache_seqlens=seqlens)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_fmha_bias_and_mask(fmha, cuda, dtype, shape):
    """Combined bias + length mask (matches RelPosMHA paged-streaming usage)."""
    B, H, H_kv, T_q, T_k, D = shape
    torch.manual_seed(3)
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    bias = torch.randn(B, H, T_q, T_k, device=cuda, dtype=dtype) * 0.1
    seqlens = torch.tensor(
        [max(1, T_k - i) for i in range(B)],
        dtype=torch.int32,
        device=cuda,
    )
    scale = 1.0 / math.sqrt(D)

    out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias, cache_seqlens=seqlens)
    ref = _ref_fmha(q, k, v, scale, attn_bias=bias, cache_seqlens=seqlens)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


_PAGED_SHAPES = [
    # (B, H, H_kv, T_q, max_blocks_per_seq, D, block_size)
    (1, 4, 4, 8, 4, 64, 16),  # MHA, single stream
    (2, 4, 4, 8, 4, 64, 16),  # MHA, two streams
    (3, 8, 2, 16, 8, 64, 16),  # GQA, multi-stream, longer kv
    (1, 4, 4, 8, 8, 64, 32),  # different block_size
    (2, 8, 1, 8, 4, 64, 16),  # MQA
]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _PAGED_SHAPES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_fmha_paged_matches_sdpa(fmha, cuda, dtype, shape, with_bias):
    """Paged mode produces the same result as SDPA on gathered K/V."""
    B, H, H_kv, T_q, max_blocks_per_seq, D, block_size = shape
    T_kv_max = max_blocks_per_seq * block_size

    torch.manual_seed(11)
    num_pool_blocks = max(B * max_blocks_per_seq + 4, 16)
    k_pool = torch.randn(
        num_pool_blocks,
        block_size,
        H_kv,
        D,
        device=cuda,
        dtype=dtype,
    )
    v_pool = torch.randn(
        num_pool_blocks,
        block_size,
        H_kv,
        D,
        device=cuda,
        dtype=dtype,
    )

    # Per-stream block table picks distinct blocks.
    block_ids = torch.randperm(num_pool_blocks)[: B * max_blocks_per_seq]
    block_table = block_ids.reshape(B, max_blocks_per_seq).to(
        dtype=torch.int32,
        device=cuda,
    )
    # Per-stream cache_seqlens: vary across streams.
    cache_seqlens = torch.tensor(
        [min(T_kv_max - 4 - 2 * b, T_kv_max - 1) for b in range(B)],
        dtype=torch.int32,
        device=cuda,
    )

    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    bias = torch.randn(B, H, T_q, T_kv_max, device=cuda, dtype=dtype) * 0.1 if with_bias else None
    scale = 1.0 / math.sqrt(D)

    out = fmha(
        q,
        k_pool,
        v_pool,
        softmax_scale=scale,
        attn_bias=bias,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
    )

    # Reference: gather and call SDPA.
    block_ids_long = block_table.long()
    k_full = k_pool[block_ids_long].reshape(B, T_kv_max, H_kv, D).permute(0, 2, 1, 3)
    v_full = v_pool[block_ids_long].reshape(B, T_kv_max, H_kv, D).permute(0, 2, 1, 3)
    ref = _ref_fmha(
        q,
        k_full,
        v_full,
        scale,
        attn_bias=bias,
        cache_seqlens=cache_seqlens,
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_fmha_fp32_falls_back_to_sdpa(fmha, cuda):
    """fp32 isn't supported by the cute kernel but works via SDPA fallback."""
    q32 = torch.randn(1, 4, 8, 64, device=cuda, dtype=torch.float32)
    k32 = torch.randn(1, 4, 16, 64, device=cuda, dtype=torch.float32)
    v32 = torch.randn(1, 4, 16, 64, device=cuda, dtype=torch.float32)
    out = fmha(q32, k32, v32, softmax_scale=0.125)
    ref = _ref_fmha(q32, k32, v32, 0.125)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_fmha_gqa_validation(fmha, cuda):
    """H must be divisible by H_kv."""
    q = torch.randn(1, 8, 8, 64, device=cuda, dtype=torch.float16)
    k = torch.randn(1, 3, 16, 64, device=cuda, dtype=torch.float16)  # 8 % 3 != 0
    v = torch.randn(1, 3, 16, 64, device=cuda, dtype=torch.float16)
    with pytest.raises(ValueError, match="divisible"):
        fmha(q, k, v, softmax_scale=0.125)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mask_floor", [-1e8, -1e10, -1e12])
def test_fmha_finite_mask_floor_stays_finite(fmha, cuda, dtype, mask_floor):
    """A heavily key-padded row masked with a large *finite* floor must not NaN.

    OASR's Conformer does not mask key padding with ``-inf``; it adds a large
    finite floor (``-1e10``, see ``RelPositionMultiHeadedAttention``). In fp16
    that saturates to ``-inf`` and the old kernel was accidentally safe, but
    bf16 keeps it finite -- and the online softmax then computed its ``exp2``
    argument as ``S * c - row_max * c``. Under ``fastmath`` the compiler
    contracts that to an FMA which subtracts the *rounded* ``fl(row_max * c)``
    from a full-precision ``S * c``, so the element attaining the max came out
    **positive** by up to half a ULP -- 64 at ``row_max ~ -1e10``. ``exp2(64)``
    is 1.8e19, a few consecutive fully-masked K-blocks pushed ``acc_O`` to inf,
    and the next rescale turned ``inf * 0`` into NaN: a whole batch row of NaN
    log-probs and a silently empty transcript.

    The geometry below is the one that fails: a short valid prefix so that
    several whole 64-column K-blocks are fully masked, and a V large enough to
    overflow once P is inflated. ``mask_floor`` is swept because the half-ULP,
    and hence the blow-up, scales with it.
    """
    B, H, T, D, valid = 1, 4, 208, 64, 46
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H, T, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H, T, D, device=cuda, dtype=dtype) * 6.0
    floor = torch.zeros(B, 1, 1, T, device=cuda, dtype=torch.float32)
    floor[..., valid:] = mask_floor
    bias = (torch.randn(B, H, T, T, device=cuda, dtype=torch.float32) * 0.5) + floor
    scale = 1.0 / math.sqrt(D)
    bias = (bias * scale).to(dtype)

    out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
    assert torch.isfinite(out).all(), (
        f"{int((~torch.isfinite(out)).sum())} non-finite entries with a finite "
        f"mask floor of {mask_floor:g}"
    )
    # And it must still be *right*, not merely finite: the valid rows attend
    # only over the unmasked prefix.
    ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
    torch.testing.assert_close(out[:, :, :valid], ref[:, :, :valid], atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# Shared-memory budget / cp.async ring depth
# ---------------------------------------------------------------------------


class TestRingDepthFitsSmem:
    """The ring depth is sized to the arch, not hardcoded.

    ``num_stages`` was fixed at 3, so the smem a launch needed
    (``sQ + stages * (sK + sV)``) scaled straight off ``head_dim``.  At
    ``head_dim=128`` with a 64x64 tile that is 112 KB, over the 99 KB cap on
    sm_86 / sm_89 / sm_120, so ``can_implement`` returned False and the shape
    was refused outright — on sm_80's 163 KB it worked fine, which is why it
    read as "no head_dim-128 config" rather than as a budget bug.  Two stages
    need 80 KB and fit.  Paraformer's SANM attention is ``d_k=128``.
    """

    @staticmethod
    def _cls(arch_str: str):
        cutlass = pytest.importorskip("cutlass")
        from oasr.kernels.cute.attention.fmha_sm80 import FmhaSm80
        from oasr.kernels.cute.attention.fmha_sm120 import FmhaSm120

        del cutlass
        return {"sm_80": FmhaSm80, "sm_120": FmhaSm120}[arch_str]

    @pytest.mark.parametrize("arch_str", ["sm_80", "sm_120"])
    @pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
    def test_selected_ring_fits(self, arch_str, head_dim):
        cls = self._cls(arch_str)
        stages = cls.select_num_stages(head_dim=head_dim)
        assert stages >= 1, f"{arch_str} head_dim={head_dim} should be implementable"
        need = cls.smem_bytes(
            head_dim=head_dim, m_block_size=64, n_block_size=64, num_stages=stages
        )
        assert need <= cls._smem_capacity_in_bytes()
        # …and it must be the *deepest* one that fits, not merely a safe one.
        if stages < cls.MAX_NUM_STAGES:
            deeper = cls.smem_bytes(
                head_dim=head_dim, m_block_size=64, n_block_size=64, num_stages=stages + 1
            )
            assert deeper > cls._smem_capacity_in_bytes()

    def test_head_dim_128_is_implementable_on_a_99kb_arch(self):
        """The regression itself."""
        cutlass = pytest.importorskip("cutlass")
        cls = self._cls("sm_120")
        assert cls._smem_capacity_in_bytes() < 112 * 1024, "premise: 3 stages must not fit"
        assert cls.select_num_stages(head_dim=128) == 2
        assert cls.can_implement(dtype=cutlass.Float16, head_dim=128)

    def test_budget_uses_the_padded_head_dim(self):
        """The layouts allocate ``(head_dim + 31) // 32 * 32``; the budget must
        agree.  Costing the raw value under-counts by a third at head_dim 72 and
        can approve a config that will not launch."""
        cls = self._cls("sm_120")
        assert cls.smem_bytes(
            head_dim=72, m_block_size=64, n_block_size=64, num_stages=1
        ) == cls.smem_bytes(head_dim=96, m_block_size=64, n_block_size=64, num_stages=1)

    def test_impossible_head_dim_still_refused(self):
        """Degrading the ring is not a licence to approve anything: a head_dim
        whose *single*-stage layout overflows must still say no."""
        cutlass = pytest.importorskip("cutlass")
        cls = self._cls("sm_120")
        assert cls.select_num_stages(head_dim=512) == 0
        assert not cls.can_implement(dtype=cutlass.Float16, head_dim=512)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_head_dim_128_matches_reference(self):
        """And the shallower ring must still compute the right answer."""
        from oasr.attention import fmha

        torch.manual_seed(0)
        B, H, T, D = 2, 4, 200, 128
        q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.float16) for _ in range(3))
        lens = torch.tensor([T, T // 2], device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        out = fmha(q, k, v, softmax_scale=scale, cache_seqlens=lens)
        ref = _ref_fmha(q, k, v, scale, cache_seqlens=lens)
        assert not torch.isnan(out).any()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
