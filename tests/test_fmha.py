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
    causal: bool = False,
    cache_seqstarts: Optional[torch.Tensor] = None,
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
        if cache_seqstarts is not None:
            keep = keep & (arange.unsqueeze(0) >= cache_seqstarts.unsqueeze(1))
        pad = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
        pad = pad.unsqueeze(1).unsqueeze(1)  # (B,1,1,T_k)
        masks.append(pad)
    if causal:
        upper = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).triu(1)
        tri = torch.zeros(1, 1, T_q, T_k, dtype=q.dtype, device=q.device)
        masks.append(tri.masked_fill_(upper.view(1, 1, T_q, T_k), float("-inf")))

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


class TestInfiniteMaskFloorWithALargeBias:
    """The other end of the mask-floor question: ``-inf`` plus a *large* bias.

    ``test_fmha_finite_mask_floor_stays_finite`` above covers a large finite
    floor.  ``-inf`` is the form upstream HuggingFace models write, and it is
    accurate here only while the **finite** part of the bias stays small.  Found
    via Nemotron's Transformer-XL relative-position bias, which reaches ~±120: the
    fp16 encoder output came out 0.69 off an fp32 reference (HF's own fp16 run:
    0.004-0.02) and two LJSpeech-200 transcripts were truncated mid-word.

    Measured boundary on an RTX 5090, fp16, ``B3 H8 T122 D128``, 38%-dense mask,
    error against fp32 SDPA — accurate to ±20, broken from ±40:

    ========  ==========  =========
    range     fused       SDPA fp16
    ========  ==========  =========
    ±20       0.00093     0.00081
    ±40       **1.365**   0.00066
    ±80       **1.494**   0.00090
    ========  ==========  =========

    It is **not** the bias magnitude alone, and it is not the K remainder: at the
    same magnitude and density a *banded* mask is accurate at every ``T`` tried
    (122 / 128 / 120 / 130 / 192).  Narrowed to one query row at a time — at
    ``B=1`` exactly **one** row of 122 is wrong (0.46 absolute), and it is a row
    whose window leaves only **4** unmasked keys out of 122, so the online
    softmax rescales across ~2 entirely-``-inf`` K-blocks while carrying a large
    finite row max.  Neighbouring rows with the same 4 keys are fine, i.e. it is
    a numerical coincidence in that bookkeeping rather than a structural
    mis-index.  In the real encoder that one bad row per layer spreads over 24
    layers and the convolution's left context, which is how it reached 0.69 at
    the output.  Same family as the finite-floor bug fixed in ``08c12cc``.

    The finite-floor arm below is the property the Nemotron model *depends on*
    (``oasr.models.nemotron.encoder.MASK_FLOOR``), so it is a hard assertion; the
    ``-inf`` arm is a strict xfail, which fails the suite when the kernel is fixed
    so the workaround and this note get removed together.
    """

    @staticmethod
    def _case(cuda, dtype, floor):
        B, H, T, D, magnitude = 3, 8, 122, 128, 80.0
        torch.manual_seed(0)
        q = torch.randn(B, H, T, D, device=cuda, dtype=dtype) * 0.5
        k = torch.randn(B, H, T, D, device=cuda, dtype=dtype) * 0.5
        v = torch.randn(B, H, T, D, device=cuda, dtype=dtype) * 0.5
        # NeMo's ``chunked_limited`` window, inline so this file stays independent
        # of the model package: frames are grouped into chunks of ``right + 1 = 4``
        # and a query sees its own chunk plus the previous ``56 // 4 = 14``.  The
        # first chunk therefore leaves only 4 unmasked keys, which is the row that
        # fails.  Every row keeps its own diagonal, so none is empty and the
        # comparison is about accuracy, not empty-row handling.
        chunk = torch.arange(T, device=cuda).div(4, rounding_mode="trunc")
        diff = chunk.unsqueeze(1) - chunk.unsqueeze(0)
        keep = (diff >= 0) & (diff <= 14)
        bias = (torch.randn(B, H, T, T, device=cuda, dtype=dtype) * magnitude).masked_fill(
            ~keep.view(1, 1, T, T), floor
        )
        scale = 1.0 / math.sqrt(D)
        return q, k, v, bias, scale

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_large_bias_with_a_finite_floor_matches_sdpa(self, fmha, cuda, dtype):
        q, k, v, bias, scale = self._case(cuda, dtype, -1.0e4)
        out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
        ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "known kernel defect: -inf in attn_bias is inaccurate once the finite "
            "part of the bias exceeds ~±32 (see .artifacts/known_issues.md). Pass a "
            "large finite floor instead; remove this xfail when the kernel is fixed"
        ),
    )
    def test_large_bias_with_an_infinite_floor_matches_sdpa(self, fmha, cuda):
        from oasr.jit.attention import select_backend

        if select_backend() != "cute":
            pytest.skip("defect is in the CuteDSL kernel; SDPA fallback is accurate")
        q, k, v, bias, scale = self._case(cuda, torch.float16, float("-inf"))
        out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
        ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


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
        if stages == 0:
            # No ring depth fits at the *default* 64-wide K tile.  That is not a
            # refusal — narrowing the tile is the other half of the search, and
            # it is what makes head_dim 256 available on a 99 KB arch (a 1-deep
            # ring would also "fit" there on paper, but fails IR verification,
            # which is why MIN_NUM_STAGES is 2 and this branch exists at all).
            n_block, stages_eff = cls.select_tile(
                head_dim=head_dim,
                m_block_size=64,
                n_block_size=64,
                paged=False,
                block_size=0,
            )
            assert (
                stages_eff >= cls.MIN_NUM_STAGES
            ), f"{arch_str} head_dim={head_dim} fits at no tile at all"
            assert n_block < 64, "expected a narrowed tile, not the requested one"
            assert (
                cls.smem_bytes(
                    head_dim=head_dim,
                    m_block_size=64,
                    n_block_size=n_block,
                    num_stages=stages_eff,
                )
                <= cls._smem_capacity_in_bytes()
            )
            return
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


class TestCausal:
    """Causal masking through ``oasr.fmha``, and the block skipping under it.

    The kernel always had the element-wise causal mask (``AttentionMask``); it
    was simply never plumbed through ``get_compiled_fmha`` / ``oasr.fmha``, so
    the waist recorded "no causal mode" as a capability gap.  Plumbing it in
    alone measured **1.4-4.8x slower than SDPA**, because the mask was applied
    per element while every row block still scanned all of K — SDPA's flash path
    skips fully-masked blocks and this one did not.  Bounding ``n_block_max`` by
    the CTA's diagonal is the actual feature (qwen2-prefill shape: 282.6 ->
    199.8 us, and the fused path overtakes SDPA at T=2048).
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    @pytest.mark.parametrize(
        "T_q,T_k",
        [
            (64, 64),  # exactly one m-block
            (65, 65),  # partial trailing block, both axes
            (128, 128),  # two m-blocks: block 0 must skip block 1's K tile
            (320, 320),  # several, so skipping is the common case
            (1, 64),  # degenerate query
            (20, 64),  # non-square: top-left aligned, same as torch
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_matches_sdpa(self, T_q, T_k, dtype):
        from oasr.attention import fmha

        torch.manual_seed(0)
        B, H, D = 2, 4, 64
        q = torch.randn(B, H, T_q, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, T_k, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, T_k, D, device="cuda", dtype=dtype)
        scale = 1.0 / math.sqrt(D)
        out = fmha(q, k, v, softmax_scale=scale, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        assert not torch.isnan(out).any()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_composes_with_per_row_lengths(self):
        """Causal AND a length mask: the kernel applies both, so the skipping
        bound must be the *tighter* of the two, never the causal one alone."""
        from oasr.attention import fmha

        torch.manual_seed(1)
        B, H, T, D = 2, 4, 192, 64
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        lens = torch.tensor([T, 40], device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        out = fmha(q, k, v, softmax_scale=scale, cache_seqlens=lens, causal=True)
        ref = _ref_fmha(q, k, v, scale, cache_seqlens=lens, causal=True)
        assert not torch.isnan(out).any()
        # Rows past a stream's length have no valid key at all under the
        # intersection, so compare only where the reference is finite.
        finite = torch.isfinite(ref)
        torch.testing.assert_close(out[finite], ref[finite], atol=2e-2, rtol=2e-2)

    def test_waist_keeps_causal_on_sdpa(self):
        """Routing is a *measured* choice now, not a capability gap — the
        distinction the backend design exists to keep."""
        from oasr.layers._backend import gap_hits, policy_hits, reset_backend_stats

        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        from oasr.layers import Attention

        a = Attention(4, 64)
        q = torch.randn(2, 4, 40, 64, device="cuda", dtype=torch.float16)
        reset_backend_stats()
        a(q, q, q, is_causal=True)
        assert policy_hits().get("fmha-causal-short") == 1
        assert not gap_hits(), "causal is a measured routing choice, not a capability gap"
        reset_backend_stats()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestPerRowKeyStart:
    """Left padding: valid keys are ``[start, len)``, not ``[0, len)``.

    The kernel used to mask keys by *length* only, so a per-row key **start**
    had no form to arrive in and left-padded batches (HF's masked-generate
    convention, which is what a batched LLM prompt is) were stranded on SDPA.
    ``mCacheSeqStarts`` closes that: one more ``(B,)`` vector, compared against
    the column index in the same mask predicate that already handles the length.
    """

    @pytest.mark.parametrize(
        "B,H,T_q,T_k,D,starts,lens",
        [
            (2, 4, 64, 128, 64, [10, 30], [128, 128]),  # start inside tile 0
            (2, 4, 64, 192, 64, [70, 130], [192, 192]),  # start past a whole tile
            (3, 4, 64, 192, 64, [70, 10, 100], [180, 128, 192]),  # both ends
            (2, 8, 64, 128, 128, [33, 65], [128, 100]),  # wide heads
            (1, 4, 64, 128, 64, [0], [128]),  # degenerate: start 0
        ],
    )
    def test_matches_reference(self, B, H, T_q, T_k, D, starts, lens):
        from oasr.attention import fmha

        torch.manual_seed(0)
        q = torch.randn(B, H, T_q, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T_k, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T_k, D, device="cuda", dtype=torch.float16)
        st = torch.tensor(starts, device="cuda", dtype=torch.int32)
        ln = torch.tensor(lens, device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        out = fmha(q, k, v, softmax_scale=scale, cache_seqlens=ln, cache_seqstarts=st)
        ref = _ref_fmha(q, k, v, scale, cache_seqlens=ln, cache_seqstarts=st)
        assert not torch.isnan(out).any()
        finite = torch.isfinite(ref)
        torch.testing.assert_close(out[finite], ref[finite], atol=2e-2, rtol=2e-2)

    def test_composes_with_causal(self):
        """Qwen2 prefill needs both at once: the causal triangle *and* the
        left-pad window.  Each must be applied, so the result is the
        intersection — the case SDPA cannot express without materializing a
        mask, which is exactly why fusing it pays."""
        from oasr.attention import fmha

        torch.manual_seed(2)
        B, H, T, D = 2, 4, 128, 64
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        st = torch.tensor([40, 96], device="cuda", dtype=torch.int32)
        ln = torch.full((B,), T, device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        out = fmha(q, k, v, softmax_scale=scale, cache_seqlens=ln, cache_seqstarts=st, causal=True)
        ref = _ref_fmha(q, k, v, scale, cache_seqlens=ln, cache_seqstarts=st, causal=True)
        assert not torch.isnan(out).any()
        finite = torch.isfinite(ref)
        torch.testing.assert_close(out[finite], ref[finite], atol=2e-2, rtol=2e-2)

    def test_fully_masked_row_is_zero_not_nan(self):
        """A query row whose whole window is padding comes back zero.

        SDPA's math backend returns NaN there, and a NaN pad row is not
        harmless: in the next layer a masked key still contributes ``0 * NaN``,
        so it poisons the *real* rows.  The kernel's empty-row clamp is what
        makes left padding safe to hand it without the caller pre-opening a
        diagonal."""
        from oasr.attention import fmha

        torch.manual_seed(3)
        B, H, T, D = 1, 4, 128, 64
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        # start == len: an empty window for every row.
        st = torch.tensor([64], device="cuda", dtype=torch.int32)
        ln = torch.tensor([64], device="cuda", dtype=torch.int32)
        out = fmha(q, k, v, softmax_scale=1.0 / math.sqrt(D), cache_seqlens=ln, cache_seqstarts=st)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_no_starts_is_unchanged(self):
        """Regression: omitting ``cache_seqstarts`` must compile and run the
        same kernel as before — the predicate is const-folded out."""
        from oasr.attention import fmha

        torch.manual_seed(4)
        B, H, T, D = 2, 4, 128, 64
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        ln = torch.tensor([128, 90], device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        a = fmha(q, k, v, softmax_scale=scale, cache_seqlens=ln)
        b = fmha(q, k, v, softmax_scale=scale, cache_seqlens=ln, cache_seqstarts=None)
        torch.testing.assert_close(a, b)
        # ... and equals passing an all-zero start vector explicitly.
        zeros = torch.zeros(B, device="cuda", dtype=torch.int32)
        c = fmha(q, k, v, softmax_scale=scale, cache_seqlens=ln, cache_seqstarts=zeros)
        torch.testing.assert_close(a, c, atol=0, rtol=0)

    def test_starts_without_lens_raises(self):
        """A start with no end is not a window."""
        from oasr.attention import fmha

        q = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.float16)
        st = torch.zeros(1, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="requires cache_seqlens"):
            fmha(q, q, q, softmax_scale=0.125, cache_seqstarts=st)

    def test_paged_kv_takes_a_start_too(self):
        """The start is read before the paged/dense branch, so one predicate
        serves both.  Not a combination anything in-tree uses today — paged
        streaming history grows rightward — but the claim is cheap to pin, and
        an untested one in a kernel is how it stops being true."""
        from oasr.attention import _gather_paged_kv, _sdpa_reference, fmha

        torch.manual_seed(5)
        B, H, D, block, nblk = 2, 4, 64, 16, 8
        T_q, T_k = 32, block * nblk
        k_pool = torch.randn(B * nblk, block, H, D, device="cuda", dtype=torch.float16)
        v_pool = torch.randn(B * nblk, block, H, D, device="cuda", dtype=torch.float16)
        bt = torch.arange(B * nblk, device="cuda", dtype=torch.int32).view(B, nblk)
        q = torch.randn(B, H, T_q, D, device="cuda", dtype=torch.float16)
        ln = torch.tensor([T_k, 100], device="cuda", dtype=torch.int32)
        st = torch.tensor([20, 48], device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)

        out = fmha(
            q,
            k_pool,
            v_pool,
            softmax_scale=scale,
            cache_seqlens=ln,
            cache_seqstarts=st,
            block_table=bt,
        )
        k_dense, v_dense = _gather_paged_kv(k_pool, v_pool, bt)
        ref = _sdpa_reference(q, k_dense, v_dense, scale, None, ln, False, st)
        assert torch.isfinite(out).all()
        finite = torch.isfinite(ref)
        torch.testing.assert_close(out[finite], ref[finite], atol=2e-2, rtol=2e-2)

    def test_finite_stale_data_past_the_length_is_inert(self):
        """What the whole-buffer / paged-pool convention rests on.

        A caller may hand over a K/V tensor wider than ``cache_seqlens`` — a
        recycled paged pool, a padded feature batch, a capacity-preallocated
        decode cache.  The kernel reads up to the K *tile* boundary above the
        length, so those columns are read; they must not matter.  They do not,
        for any finite value: the length mask gives them zero softmax weight.

        ``NaN``/``Inf`` in ``v`` are the documented exception — zero weight
        still yields ``0 * NaN`` inside ``P @ V``, past any mask — which is why
        a preallocated cache has to be zeroed rather than ``empty``.  That is a
        precondition on the caller today; predicating the load against the
        length (as upstream FlashAttention does) is what would retire it.
        """
        from oasr.attention import fmha

        torch.manual_seed(7)
        B, H, D = 2, 4, 64
        T_k, L = 192, 130  # L % 64 != 0, so a partial last block exists
        q = torch.randn(B, H, 32, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T_k, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T_k, D, device="cuda", dtype=torch.float16)
        ln = torch.full((B,), L, dtype=torch.int32, device="cuda")
        scale = 1.0 / math.sqrt(D)
        base = fmha(q, k, v, softmax_scale=scale, cache_seqlens=ln)

        for fill in (0.0, 3.0, -2.0, 1e4):
            k2, v2 = k.clone(), v.clone()
            k2[:, :, L:] = fill
            v2[:, :, L:] = fill
            got = fmha(q, k2, v2, softmax_scale=scale, cache_seqlens=ln)
            torch.testing.assert_close(got, base, atol=0, rtol=0)
