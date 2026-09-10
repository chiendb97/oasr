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

from oasr.cache import PagedKVCache

# ``fmha`` is also the name of this module's fixture, so the dense function
# is imported under an alias for the packed reference below.
from oasr.functionals.attention import fmha as _dense_fmha, fmha_varlen
from oasr.layers.attention.attention import RelPositionMultiHeadedAttention

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
def test_fmha_bias_and_mask(fmha, cuda, dtype, shape):
    """Combined bias + length mask (matches RelPosMHA paged-streaming usage).

    This is the *operand-carrying* sweep, and it subsumes bias-alone and
    mask-alone shape for shape: both operands are folded into one additive
    mask before the softmax, so a shape that survives the pair survives either
    half.  What the pair cannot show is that each operand is honoured *at all*
    when the other is absent -- a launcher that dropped ``attn_bias`` whenever
    ``cache_seqlens`` was None would still pass here.  That is one case each,
    below, not a second and third pass over all nine shapes.
    """
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


@pytest.mark.parametrize("operand", ["bias", "length_mask"])
def test_each_operand_alone_is_honoured(fmha, cuda, operand):
    """One operand, no other: it must change the answer and match the reference.

    The ``!= plain`` assertion is the point. Parity alone would pass on a
    launcher that silently ignored the operand *if* the reference ignored it
    too -- so the test first proves the operand does something.
    """
    B, H, H_kv, T_q, T_k, D = 2, 8, 2, 8, 64, 64
    torch.manual_seed(1)
    dtype = _DTYPES[0]
    q = torch.randn(B, H, T_q, D, device=cuda, dtype=dtype)
    k = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    v = torch.randn(B, H_kv, T_k, D, device=cuda, dtype=dtype)
    scale = 1.0 / math.sqrt(D)

    if operand == "bias":
        kw = {"attn_bias": torch.randn(B, H, T_q, T_k, device=cuda, dtype=dtype) * 0.5}
    else:
        # Half the streams get a short context (~1/4 of T_k), the rest full T_k.
        kw = {
            "cache_seqlens": torch.tensor(
                [max(1, T_k // 4) if i < B // 2 else T_k for i in range(B)],
                dtype=torch.int32,
                device=cuda,
            )
        }

    out = fmha(q, k, v, softmax_scale=scale, **kw)
    torch.testing.assert_close(out, _ref_fmha(q, k, v, scale, **kw), atol=1e-2, rtol=1e-2)
    plain = fmha(q, k, v, softmax_scale=scale)
    assert not torch.allclose(out, plain), f"{operand} was ignored"


_PAGED_SHAPES = [
    # (B, H, H_kv, T_q, max_blocks_per_seq, D, block_size)
    (1, 4, 4, 8, 4, 64, 16),  # MHA, single stream
    (2, 4, 4, 8, 4, 64, 16),  # MHA, two streams
    (3, 8, 2, 16, 8, 64, 16),  # GQA, multi-stream, longer kv
    (1, 4, 4, 8, 8, 64, 32),  # different block_size
    (2, 8, 1, 8, 4, 64, 16),  # MQA
    # Block tables whose width is *not* a whole number of K tiles.  A paged K
    # tile spans ``N_BLOCK // block_size`` pages and the loader indexes them
    # unpredicated, so the last tile of a 3-wide table (4 pages per tile) reads
    # past the tensor and dereferences whatever followed it as a page id.  Every
    # shape above happens to divide evenly, which is why this went unseen until
    # AR decode paged its KV 16 tokens to a page.
    (2, 4, 4, 8, 3, 64, 16),  # 3 pages, 4 per tile
    (2, 4, 4, 1, 1, 64, 16),  # a decode step: one query, one page
    (1, 4, 4, 8, 5, 64, 32),  # 5 pages, 2 per tile
]


#: ``with_bias=True`` is the superset: it adds the bias-tile load, whose
#: predication is what once read ~500 elements past a short segment.  So every
#: geometry is swept with the bias on, and the no-bias branch -- the AR-decode
#: path -- gets three: an even-tile table, a short one, and a decode step.
_PAGED_CASES = [(shape, True) for shape in _PAGED_SHAPES] + [
    (_PAGED_SHAPES[0], False),  # MHA, table divides evenly into K tiles
    (_PAGED_SHAPES[5], False),  # 3 pages, 4 per tile
    (_PAGED_SHAPES[6], False),  # a decode step: one query, one page
]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape,with_bias", _PAGED_CASES)
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
    """``-inf`` in ``attn_bias`` has to mean what SDPA means by it.

    The kernel used to write the *empty-row clamp* into the carried
    ``row_max``: a K-tile with no unmasked column at all left the running max
    at ``0`` instead of ``-inf``, so every later tile whose own max ``m`` was
    negative got exponentiated about 0 rather than about ``m``.  ``P`` then
    lost the ``max(P) == 1`` invariant that makes the cast down to fp16/bf16
    before ``P @ V`` lossless -- subnormal by ``m ~ -11``, flushed to zero by
    ``m ~ -20`` -- while ``row_sum`` stayed fp32-exact, so the row came back
    mis-scaled or empty.  ``m`` is a logit: nothing but an additive bias gets
    it that far from 0, which is why plain attention -- and FlashAttention,
    whose softmax block this one mirrors -- never sees it.

    Two things therefore decide a case, and both are in the geometries below:
    a **fully masked first-visited K-tile** (the n-block loop runs descending,
    so that is the *last* tile of the row) and a **negative row max**.  A
    large finite floor (``-1e4``) never triggered it, because a finite floor
    is itself a finite row max.
    """

    # (B, H, T, D) shared by every case here so each dtype compiles once.
    SHAPE = (3, 8, 122, 128)

    @staticmethod
    def _chunked_window(T, device, chunk_size=4, history=14):
        """NeMo's ``chunked_limited`` window, inline so this file stays
        independent of the model package: frames are grouped into chunks of
        ``right + 1 = 4`` and a query sees its own chunk plus the previous
        ``56 // 4 = 14``.  The first chunk therefore leaves only 4 unmasked
        keys, and its rows are the ones that failed.  Every row keeps its own
        diagonal, so none is empty and the comparison is about accuracy, not
        empty-row handling.
        """
        chunk = torch.arange(T, device=device).div(chunk_size, rounding_mode="trunc")
        diff = chunk.unsqueeze(1) - chunk.unsqueeze(0)
        return (diff >= 0) & (diff <= history)

    @classmethod
    def _case(cls, cuda, dtype, floor, magnitude=80.0, offset=0.0, keep=None):
        B, H, T, D = cls.SHAPE
        torch.manual_seed(0)
        q = torch.randn(B, H, T, D, device=cuda, dtype=dtype) * 0.5
        k = torch.randn(B, H, T, D, device=cuda, dtype=dtype) * 0.5
        v = torch.randn(B, H, T, D, device=cuda, dtype=dtype) * 0.5
        if keep is None:
            keep = cls._chunked_window(T, cuda)
        bias = torch.randn(B, H, T, T, device=cuda, dtype=dtype) * magnitude + offset
        bias = bias.masked_fill(~keep.view(1, 1, T, T), floor)
        scale = 1.0 / math.sqrt(D)
        return q, k, v, bias, scale

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_large_bias_with_a_finite_floor_matches_sdpa(self, fmha, cuda, dtype):
        q, k, v, bias, scale = self._case(cuda, dtype, -1.0e4)
        out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
        ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_large_bias_with_an_infinite_floor_matches_sdpa(self, fmha, cuda, dtype):
        """The case the strict xfail used to pin.  Off by 1.49 on the pre-fix
        kernel, on the 4-unmasked-key rows of the first chunk."""
        q, k, v, bias, scale = self._case(cuda, dtype, float("-inf"))
        out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
        ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_the_two_floors_agree(self, fmha, cuda, dtype):
        """``-inf`` and a large finite floor are the same mask, so they must
        give the same answer -- the claim that retires the finite-floor
        workaround, rather than each arm merely being within tolerance of
        SDPA.  It is also the sharpest of these cases in bf16, where the wider
        exponent kept the pre-fix error (0.0039) inside the SDPA tolerance
        while the two floors still disagreed."""
        args_inf = self._case(cuda, dtype, float("-inf"))
        args_fin = self._case(cuda, dtype, -1.0e4)
        q, k, v, bias_inf, scale = args_inf
        bias_fin = args_fin[3]
        out_inf = fmha(q, k, v, softmax_scale=scale, attn_bias=bias_inf)
        out_fin = fmha(q, k, v, softmax_scale=scale, attn_bias=bias_fin)
        torch.testing.assert_close(out_inf, out_fin, atol=0, rtol=0)

    @pytest.mark.parametrize("offset", [-200.0, -90.0, -20.0, -11.0, 0.0, 90.0, 200.0])
    def test_the_bias_offset_does_not_matter(self, fmha, cuda, offset):
        """Shift the whole bias and watch the sign of the row max decide.

        Softmax is shift-invariant, so adding a constant to every unmasked
        logit must not move the output at all -- and on the pre-fix kernel it
        did, at every offset here that left the sparse rows' max negative.  The
        positive end passes on the broken kernel too, because the poisoned
        running max is ``max(0, m)``, which is ``m`` exactly when ``m >= 0``:
        that asymmetry is this defect's fingerprint, and it is why a symmetric
        ``randn`` bias hides half of it.  With a *constant* bias in place of
        this one's ``randn * 80``, the damage sets in at ``m ~ -11`` (fp16
        subnormals) and is total by ``m ~ -20``.
        """
        q, k, v, bias, scale = self._case(cuda, torch.float16, float("-inf"), offset=offset)
        out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
        ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_several_fully_masked_leading_tiles(self, fmha, cuda, dtype):
        """Only the first three keys survive, so the descending n-block loop
        opens on two whole ``-inf`` tiles before it reaches a live column --
        and with three keys per row the max is negative about half the time.
        Off by 2.14 pre-fix, in both dtypes.
        """
        B, H, T, D = self.SHAPE
        keep = torch.zeros(T, T, dtype=torch.bool, device=cuda)
        keep[:, :3] = True
        q, k, v, bias, scale = self._case(cuda, dtype, float("-inf"), keep=keep)
        out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
        ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    def test_empty_rows_stay_zero_next_to_a_large_bias(self, fmha, cuda):
        """A row with no unmasked key at all still comes back zero, not NaN --
        the kernel's documented empty-row behaviour, pinned on its own by
        ``TestPerRowKeyStart.test_fully_masked_row_is_zero_not_nan`` -- and its
        neighbours in the same tile are unaffected.  This is the case the
        rescale has to survive: ``row_max_prev`` is ``-inf`` while
        ``row_max_cur`` is finite, and forming that delta as
        ``exp2((-inf) - m)`` rather than special-casing it would multiply an
        all-zero accumulator by an exponent one overflow away from ``inf``.
        """
        B, H, T, D = self.SHAPE
        keep = self._chunked_window(T, cuda).clone()
        rows = torch.arange(T, device=cuda)
        empty = (rows % 8) == 3
        keep[empty] = False
        q, k, v, bias, scale = self._case(cuda, torch.float16, float("-inf"), keep=keep)
        out = fmha(q, k, v, softmax_scale=scale, attn_bias=bias)
        ref = _ref_fmha(q, k, v, scale, attn_bias=bias)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out[:, :, empty], torch.zeros_like(out[:, :, empty]))
        torch.testing.assert_close(out[:, :, ~empty], ref[:, :, ~empty], atol=2e-2, rtol=2e-2)

    def test_paged_kv_with_an_infinite_floor(self, fmha, cuda):
        """The paged path shares the softmax block, and shared is not the same
        as covered: its tiles come from a block table, so it reaches
        ``online_softmax`` through a different loader.  Off by 1.53 pre-fix."""
        from oasr.functionals.attention import gather_paged_kv

        B, H, D, block, nblk = 2, 4, 64, 16, 9
        T_q, T_k = 128, block * nblk
        torch.manual_seed(0)
        k_pool = torch.randn(B * nblk, block, H, D, device=cuda, dtype=torch.float16) * 0.5
        v_pool = torch.randn(B * nblk, block, H, D, device=cuda, dtype=torch.float16) * 0.5
        bt = torch.arange(B * nblk, device=cuda, dtype=torch.int32).view(B, nblk)
        q = torch.randn(B, H, T_q, D, device=cuda, dtype=torch.float16) * 0.5
        keep = self._chunked_window(max(T_q, T_k), cuda)[:T_q, :T_k]
        bias = torch.randn(B, H, T_q, T_k, device=cuda, dtype=torch.float16) * 80.0
        bias = bias.masked_fill(~keep.view(1, 1, T_q, T_k), float("-inf"))
        scale = 1.0 / math.sqrt(D)

        lens = torch.tensor([T_k, T_k - 20], device=cuda, dtype=torch.int32)
        out = fmha(
            q,
            k_pool,
            v_pool,
            softmax_scale=scale,
            attn_bias=bias,
            cache_seqlens=lens,
            block_table=bt,
        )
        k_dense, v_dense = gather_paged_kv(k_pool, v_pool, bt)
        ref = _ref_fmha(q, k_dense, v_dense, scale, attn_bias=bias, cache_seqlens=lens)
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
        from oasr.functionals.attention import fmha

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
    def test_matches_sdpa(self, T_q, T_k):
        """The six geometries are the block-skipping branch table.

        No dtype axis: the skip bound is integer arithmetic over block
        indices, identical in fp16 and bf16, so a second pass would re-run the
        same decision at twice the cost.
        """
        from oasr.functionals.attention import fmha

        torch.manual_seed(0)
        B, H, D = 2, 4, 64
        dtype = torch.float16
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
        from oasr.functionals.attention import fmha

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
        from oasr.functionals.attention import fmha

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
        from oasr.functionals.attention import fmha

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
        from oasr.functionals.attention import fmha

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
        from oasr.functionals.attention import fmha

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
        from oasr.functionals.attention import fmha

        q = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.float16)
        st = torch.zeros(1, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="requires cache_seqlens"):
            fmha(q, q, q, softmax_scale=0.125, cache_seqstarts=st)

    def test_paged_kv_takes_a_start_too(self):
        """The start is read before the paged/dense branch, so one predicate
        serves both.  Not a combination anything in-tree uses today — paged
        streaming history grows rightward — but the claim is cheap to pin, and
        an untested one in a kernel is how it stops being true."""
        from oasr.functionals.attention import _sdpa_reference, fmha, gather_paged_kv

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
        k_dense, v_dense = gather_paged_kv(k_pool, v_pool, bt)
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
        from oasr.functionals.attention import fmha

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


# ---------------------------------------------------------------------------
# Variable-length (sequence-packed) attention
#
# ``fmha_varlen`` packs several segments into one ``(total, H, D)`` tensor and
# restricts each segment to itself via ``cu_seqlens``.  The reference is the
# dense ``fmha`` above run per segment, so the two live together: a change to
# the dense kernel that the packed loader does not follow shows up here.
# ---------------------------------------------------------------------------


#: The packed reference accumulates per segment, so the bound is the same in
#: both served formats -- it never depended on ``dtype``, which is why the
#: parameter it used to take was ignored.
_VARLEN_TOL = {"rtol": 2e-2, "atol": 2e-2}


def _varlen_make_packed(seg_lens, H, H_kv, D, dtype, device):
    """Random packed q/k/v + cu_seqlens for the given segment lengths."""
    total = sum(seg_lens)
    q = torch.randn(total, H, D, device=device, dtype=dtype)
    k = torch.randn(total, H_kv, D, device=device, dtype=dtype)
    v = torch.randn(total, H_kv, D, device=device, dtype=dtype)
    cu = torch.zeros(len(seg_lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seg_lens, dtype=torch.int32, device=device).cumsum(0)
    return q, k, v, cu


def _build_packed_bias(q, cu, H, D, dtype, device, scale):
    """A packed block-diagonal additive bias (random) + bias_offsets."""
    seg_lens = (cu[1:] - cu[:-1]).tolist()
    sizes = [H * t * t for t in seg_lens]
    offsets = torch.zeros(len(seg_lens) + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.tensor(sizes, dtype=torch.int64, device=device).cumsum(0)
    bias = torch.randn(int(offsets[-1]), device=device, dtype=dtype) * 0.1
    return bias, offsets


def _ref_per_segment(q, k, v, cu, scale, bias, offsets, H):
    """Reference: dense fmha per segment, assembled back into packed output."""
    cu_l = cu.tolist()
    bo = offsets.tolist() if offsets is not None else None
    out = torch.empty_like(q)
    for s in range(len(cu_l) - 1):
        a, b = cu_l[s], cu_l[s + 1]
        qs = q[a:b].transpose(0, 1).unsqueeze(0)
        ks = k[a:b].transpose(0, 1).unsqueeze(0)
        vs = v[a:b].transpose(0, 1).unsqueeze(0)
        bias_s = None
        if bias is not None:
            # clone() → a fresh 16-byte-aligned allocation (the dense cute
            # kernel rejects misaligned bias slices of the packed buffer).
            bias_s = bias[bo[s] : bo[s + 1]].view(1, H, b - a, b - a).clone()
        out_s = _dense_fmha(qs, ks, vs, softmax_scale=scale, attn_bias=bias_s)
        out[a:b] = out_s.squeeze(0).transpose(0, 1)
    return out


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seg_lens", [[33], [64, 64], [33, 249, 17], [8] * 12])
@pytest.mark.parametrize("with_bias", [False, True])
def test_varlen_matches_per_segment_dense(dtype, seg_lens, with_bias, device):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unsupported")
    torch.manual_seed(0)
    H, H_kv, D = 4, 4, 64
    scale = 1.0 / math.sqrt(D)
    q, k, v, cu = _varlen_make_packed(seg_lens, H, H_kv, D, dtype, device)
    bias, offsets = (None, None)
    if with_bias:
        bias, offsets = _build_packed_bias(q, cu, H, D, dtype, device, scale)

    out = fmha_varlen(
        q,
        k,
        v,
        softmax_scale=scale,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(seg_lens),
        max_seqlen_k=max(seg_lens),
        attn_bias=bias,
        bias_offsets=offsets,
    )
    ref = _ref_per_segment(q, k, v, cu, scale, bias, offsets, H)
    assert out.shape == q.shape
    torch.testing.assert_close(out, ref, **_VARLEN_TOL)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("gqa", [(8, 2), (8, 1)])
def test_varlen_gqa(dtype, gqa, device):
    torch.manual_seed(1)
    H, H_kv = gqa
    D = 64
    scale = 1.0 / math.sqrt(D)
    seg_lens = [40, 55, 33]
    q, k, v, cu = _varlen_make_packed(seg_lens, H, H_kv, D, dtype, device)
    out = fmha_varlen(
        q,
        k,
        v,
        softmax_scale=scale,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(seg_lens),
        max_seqlen_k=max(seg_lens),
    )
    ref = _ref_per_segment(q, k, v, cu, scale, None, None, H)
    torch.testing.assert_close(out, ref, **_VARLEN_TOL)


@pytest.mark.cuda
def test_varlen_single_segment_equals_dense(device):
    dtype = torch.float16
    torch.manual_seed(2)
    H, D, T = 4, 64, 50
    scale = 1.0 / math.sqrt(D)
    q, k, v, cu = _varlen_make_packed([T], H, H, D, dtype, device)
    out = fmha_varlen(
        q,
        k,
        v,
        softmax_scale=scale,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=T,
        max_seqlen_k=T,
    )
    dense = (
        _dense_fmha(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            softmax_scale=scale,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    torch.testing.assert_close(out, dense, **_VARLEN_TOL)


# ---------------------------------------------------------------------------
# RelPos multi-head attention -- the layer that calls the kernel
#
# ``RelPositionMultiHeadedAttention`` is the only in-tree caller of the paged
# path, and its reference re-implements the WeNet rel-pos algebra (matrix_bd
# combined with the padding bias before SDPA).  That algebra is *not* the SDPA
# reference above: this is a guard against drift in the bias/mask plumbing
# between the module and the kernel it dispatches to.
# ---------------------------------------------------------------------------


# Tests use n_feat=64, n_head=4 → d_k = 16.
N_HEAD = 4  # RelPosMHA: n_feat=64, n_head=4 -> d_k=16
N_FEAT = 64
D_K = N_FEAT // N_HEAD


def _ref_qkv(attn, x):
    """Replicate the module's fused-QKV projection (head-major (B, H, T, D))."""
    B, T, _ = x.shape
    qkv = attn.linear_qkv(x)
    q, k, v = qkv.split((attn.inner_dim, attn.inner_kv_dim, attn.inner_kv_dim), dim=-1)
    q = q.view(B, T, attn.h, attn.d_k).transpose(1, 2)
    k = k.view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    v = v.view(B, T, attn.h_kv, attn.d_k).transpose(1, 2)
    return q, k, v


def _ref_offline(attn, x, mask, pos_emb):
    """SDPA reference for the offline path."""
    q, k, v = _ref_qkv(attn, x)
    n_batch_pos = pos_emb.size(0)
    p = attn.linear_pos(pos_emb).view(n_batch_pos, -1, attn.h, attn.d_k).transpose(1, 2)
    q_t = q.transpose(1, 2)
    q_u = (q_t + attn.pos_bias_u).transpose(1, 2)
    q_v = (q_t + attn.pos_bias_v).transpose(1, 2)
    matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
    attn_bias = (matrix_bd + mask.unsqueeze(1)) / math.sqrt(attn.d_k)
    out = F.scaled_dot_product_attention(
        q_u,
        k,
        v,
        attn_mask=attn_bias,
        scale=1 / math.sqrt(attn.d_k),
    )
    out = out.transpose(1, 2).contiguous().view(x.size(0), -1, attn.h * attn.d_k)
    return attn.linear_out(out)


def _ref_paged(attn, x, pos_emb, cache: PagedKVCache):
    """SDPA reference for the paged-streaming path.

    Mirrors the new path: write new K/V into a copy of the pool, gather
    up to ``host_seqlen_max + T_q`` frames, then run SDPA with the
    ``(matrix_bd + padding_bias) / sqrt(d_k)`` mask.
    """
    q, k_new, v_new = _ref_qkv(attn, x)

    cache_local = PagedKVCache(
        k_cache=cache.k_cache.clone(),
        v_cache=cache.v_cache.clone(),
        block_table=cache.block_table,
        cache_seqlens=cache.cache_seqlens,
        block_size=cache.block_size,
        host_seqlen_max=cache.host_seqlen_max,
    )
    cache_local.write_kv_chunk(k_new, v_new, offset=cache_local.cache_seqlens)
    T_kv_max = cache.host_seqlen_max + x.size(1)
    k_full, v_full = cache_local.gather_full_kv(T_kv_max)

    total_kv_lens = cache.cache_seqlens + x.size(1)
    arange = torch.arange(T_kv_max, device=cache.cache_seqlens.device)
    keep = arange.unsqueeze(0) < total_kv_lens.unsqueeze(1)  # (B, T_kv_max)
    pad_bias = torch.where(keep, 0.0, float("-inf")).to(x.dtype)
    pad_bias = pad_bias.unsqueeze(1).unsqueeze(1)  # broadcast over (H, T_q)

    n_batch_pos = pos_emb.size(0)
    p = attn.linear_pos(pos_emb).view(n_batch_pos, -1, attn.h, attn.d_k).transpose(1, 2)
    q_t = q.transpose(1, 2)
    q_u = (q_t + attn.pos_bias_u).transpose(1, 2)
    q_v = (q_t + attn.pos_bias_v).transpose(1, 2)
    matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
    attn_bias = (matrix_bd + pad_bias) / math.sqrt(attn.d_k)
    out = F.scaled_dot_product_attention(
        q_u,
        k_full,
        v_full,
        attn_mask=attn_bias,
        scale=1 / math.sqrt(attn.d_k),
    )
    out = out.transpose(1, 2).contiguous().view(x.size(0), -1, attn.h * attn.d_k)
    return attn.linear_out(out)


@pytest.fixture
def attn(device):
    torch.manual_seed(0)
    return RelPositionMultiHeadedAttention(N_HEAD, N_FEAT).to(device).eval()


@pytest.mark.parametrize("B,T", [(1, 8), (2, 6), (4, 12)])
def test_offline_path_matches_sdpa(attn, device, B, T):
    """Offline (cache=None) FlexAttention path matches SDPA reference."""
    x = torch.randn(B, T, N_FEAT, device=device)
    pos_emb = torch.randn(B, T, N_FEAT, device=device)
    mask = torch.zeros(B, 1, T, device=device)

    with torch.no_grad():
        out_new, cache = attn(x, mask, pos_emb, cache=None)
        out_ref = _ref_offline(attn, x, mask, pos_emb)

    assert cache is None
    torch.testing.assert_close(out_new, out_ref, rtol=1e-4, atol=1e-4)


def test_paged_path_matches_sdpa(attn, device):
    """Paged streaming (cache=PagedKVCache) — heterogeneous per-stream lengths.

    Two streams with different ``cache_seqlens`` (10 and 5) sharing one
    physical block pool. Verifies the per-stream length-mask + rel-pos
    bias path against the SDPA reference.
    """
    torch.manual_seed(1)
    B = 2
    T_q = 4
    max_blocks, block_size = 16, 8

    k_pool = torch.zeros(max_blocks, block_size, N_HEAD, D_K, device=device)
    v_pool = torch.zeros(max_blocks, block_size, N_HEAD, D_K, device=device)

    block_table = torch.tensor(
        [[0, 1, 2, 5, 6, 7], [3, 4, 8, 9, 10, 11]],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor([10, 5], dtype=torch.int32, device=device)

    # Pre-fill stream 0's first 10 frames and stream 1's first 5 frames.
    seed_k0 = torch.randn(10, N_HEAD, D_K, device=device)
    seed_v0 = torch.randn(10, N_HEAD, D_K, device=device)
    for t in range(10):
        phys = int(block_table[0, t // block_size].item())
        k_pool[phys, t % block_size] = seed_k0[t]
        v_pool[phys, t % block_size] = seed_v0[t]
    seed_k1 = torch.randn(5, N_HEAD, D_K, device=device)
    seed_v1 = torch.randn(5, N_HEAD, D_K, device=device)
    for t in range(5):
        phys = int(block_table[1, 0].item())
        k_pool[phys, t] = seed_k1[t]
        v_pool[phys, t] = seed_v1[t]

    x = torch.randn(B, T_q, N_FEAT, device=device)
    T_kv_max = 10 + T_q
    pos_emb = torch.randn(1, T_kv_max, N_FEAT, device=device)

    cache = PagedKVCache(
        k_cache=k_pool,
        v_cache=v_pool,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        block_size=block_size,
        host_seqlen_max=10,
    )

    k_pool_save = k_pool.clone()
    v_pool_save = v_pool.clone()

    with torch.no_grad():
        cache.k_cache.copy_(k_pool_save)
        cache.v_cache.copy_(v_pool_save)
        out_new, _ = attn(x, torch.zeros((0, 0, 0), device=device), pos_emb, cache=cache)
        cache.k_cache.copy_(k_pool_save)
        cache.v_cache.copy_(v_pool_save)
        out_ref = _ref_paged(attn, x, pos_emb, cache)

    torch.testing.assert_close(out_new, out_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.cuda
def test_a_paged_config_the_kernel_refuses_still_answers():
    """A declared gap has to *serve* the shape, not raise on it.

    The paged loader skips per-element head-dim predication, so the arch class
    refuses a head_dim off its 32-element MMA stride.  ``oasr.functionals.attention.fmha``
    raises there — the right contract for a caller naming the kernel by name —
    which leaves the waist to gather the pages and answer on SDPA, counting the
    gap so the coverage debt stays visible.  A shipped decoder's head_dim is 64
    or 128, so this is the tiny-config path; it is also the only thing standing
    between such a config and a hard failure.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from oasr.layers import Attention
    from oasr.layers._backend import gap_hits, reset_backend_stats

    torch.manual_seed(0)
    heads, head_dim, block_size, blocks = 2, 16, 16, 4
    B, T_q = 2, 1
    k_pool = torch.randn(blocks, block_size, heads, head_dim, device="cuda", dtype=torch.float16)
    v_pool = torch.randn(blocks, block_size, heads, head_dim, device="cuda", dtype=torch.float16)
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    lens = torch.tensor([20, 9], dtype=torch.int32, device="cuda")
    q = torch.randn(B, heads, T_q, head_dim, device="cuda", dtype=torch.float16)

    attn = Attention(heads, head_dim)
    reset_backend_stats()
    with torch.no_grad():
        out = attn(q, k_pool, v_pool, kv_lens=lens, block_table=table)
    assert gap_hits().get("fmha-paged-config"), "the gap was not counted"

    # Reference: gather the addressed pages and mask by length.
    dense = k_pool[table.long()].reshape(B, -1, heads, head_dim).permute(0, 2, 1, 3)
    dense_v = v_pool[table.long()].reshape(B, -1, heads, head_dim).permute(0, 2, 1, 3)
    keep = torch.arange(dense.size(2), device="cuda").unsqueeze(0) < lens.unsqueeze(1)
    ref = F.scaled_dot_product_attention(
        q.float(),
        dense.float(),
        dense_v.float(),
        attn_mask=keep.view(B, 1, 1, -1),
        scale=head_dim**-0.5,
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
