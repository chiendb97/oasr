"""Unit tests for OASR's RelPos multi-head attention.

After the refactor a single class is exposed —
:class:`oasr.layers.attention.attention.RelPositionMultiHeadedAttention` —
exercising three cache modes:

* offline (``cache=None``)
* dense streaming (``cache=(K_old, V_old)``)
* paged streaming (``cache=PagedKVCache``)

Each mode is checked against a hand-rolled reference that re-implements
the WeNet rel-pos algebra exactly (``matrix_bd`` combined with the
padding bias before SDPA). The reference and the implementation both
funnel into ``F.scaled_dot_product_attention``, so this is a regression
guard against accidental drift in the bias / mask plumbing rather than
a kernel-vs-kernel comparison.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from oasr.layers.attention.attention import (
    PagedKVCache,
    RelPositionMultiHeadedAttention,
    _paged_gather_kv,
    _paged_write_kv,
    build_paged_block_mask,
)


# FlexAttention's compiled fused kernel requires head_dim >= 16 (Triton
# block-tile constraint). Tests use n_feat=64, n_head=4 → d_k = 16.
N_HEAD = 4
N_FEAT = 64
D_K = N_FEAT // N_HEAD


def _ref_offline(attn, x, mask, pos_emb):
    """SDPA reference for the offline path."""
    q = attn._project("query", x)
    k = attn._project("key", x)
    v = attn._project("value", x)
    n_batch_pos = pos_emb.size(0)
    p = attn.linear_pos(pos_emb).view(n_batch_pos, -1, attn.h, attn.d_k).transpose(1, 2)
    q_t = q.transpose(1, 2)
    q_u = (q_t + attn.pos_bias_u).transpose(1, 2)
    q_v = (q_t + attn.pos_bias_v).transpose(1, 2)
    matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
    attn_bias = (matrix_bd + mask.unsqueeze(1)) / math.sqrt(attn.d_k)
    out = F.scaled_dot_product_attention(
        q_u, k, v, attn_mask=attn_bias, scale=1 / math.sqrt(attn.d_k),
    )
    out = out.transpose(1, 2).contiguous().view(x.size(0), -1, attn.h * attn.d_k)
    return attn.linear_out(out)


def _ref_dense(attn, x, mask, pos_emb, cache):
    """SDPA reference for the dense-streaming path."""
    q = attn._project("query", x)
    k_new = attn._project("key", x)
    v_new = attn._project("value", x)
    k_old, v_old = cache
    k_full = torch.cat([k_old, k_new], dim=-2)
    v_full = torch.cat([v_old, v_new], dim=-2)
    n_batch_pos = pos_emb.size(0)
    p = attn.linear_pos(pos_emb).view(n_batch_pos, -1, attn.h, attn.d_k).transpose(1, 2)
    q_t = q.transpose(1, 2)
    q_u = (q_t + attn.pos_bias_u).transpose(1, 2)
    q_v = (q_t + attn.pos_bias_v).transpose(1, 2)
    matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
    attn_bias = (matrix_bd + mask.unsqueeze(1)) / math.sqrt(attn.d_k)
    out = F.scaled_dot_product_attention(
        q_u, k_full, v_full, attn_mask=attn_bias, scale=1 / math.sqrt(attn.d_k),
    )
    out = out.transpose(1, 2).contiguous().view(x.size(0), -1, attn.h * attn.d_k)
    return attn.linear_out(out), (k_full, v_full)


def _ref_paged(attn, x, pos_emb, cache: PagedKVCache):
    """SDPA reference for the paged-streaming path.

    Mirrors the new path: write new K/V into a copy of the pool, gather
    up to ``host_seqlen_max + T_q`` frames, then run SDPA with the
    ``(matrix_bd + padding_bias) / sqrt(d_k)`` mask.
    """
    q = attn._project("query", x)
    k_new = attn._project("key", x)
    v_new = attn._project("value", x)

    k_pool_local = cache.k_cache.clone()
    v_pool_local = cache.v_cache.clone()
    _paged_write_kv(
        k_pool_local, v_pool_local, cache.block_table,
        cache.cache_seqlens, k_new, v_new,
    )
    T_kv_max = cache.host_seqlen_max + x.size(1)
    k_full, v_full = _paged_gather_kv(
        k_pool_local, v_pool_local, cache.block_table, T_kv_max,
    )

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
        q_u, k_full, v_full, attn_mask=attn_bias, scale=1 / math.sqrt(attn.d_k),
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
        out_new, cache = attn(x, x, x, mask, pos_emb, cache=None)
        out_ref = _ref_offline(attn, x, mask, pos_emb)

    assert cache is None
    torch.testing.assert_close(out_new, out_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("B,T_q,T_old", [(1, 4, 12), (2, 6, 16)])
def test_dense_path_matches_sdpa(attn, device, B, T_q, T_old):
    """Dense streaming (cache=tuple) FlexAttention path matches SDPA reference."""
    x = torch.randn(B, T_q, N_FEAT, device=device)
    T_kv = T_old + T_q
    pos_emb = torch.randn(B, T_kv, N_FEAT, device=device)
    mask = torch.zeros(B, 1, T_kv, device=device)
    k_old = torch.randn(B, N_HEAD, T_old, D_K, device=device)
    v_old = torch.randn(B, N_HEAD, T_old, D_K, device=device)

    with torch.no_grad():
        out_new, new_cache = attn(x, x, x, mask, pos_emb, cache=(k_old, v_old))
        out_ref, ref_cache = _ref_dense(attn, x, mask, pos_emb, (k_old, v_old))

    torch.testing.assert_close(out_new, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(new_cache[0], ref_cache[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(new_cache[1], ref_cache[1], rtol=1e-4, atol=1e-4)


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
        dtype=torch.int32, device=device,
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

    # The encoder normally builds the FlexAttention BlockMask once per
    # forward and shares it across layers. Mirror that here.
    total_kv_lens = cache_seqlens + T_q
    block_mask = build_paged_block_mask(
        total_kv_lens, B=B, T_q=T_q, T_kv_max=T_kv_max, device=device,
    )

    cache = PagedKVCache(
        k_cache=k_pool, v_cache=v_pool,
        block_table=block_table, cache_seqlens=cache_seqlens,
        block_size=block_size, host_seqlen_max=10,
        block_mask=block_mask,
    )

    k_pool_save = k_pool.clone()
    v_pool_save = v_pool.clone()

    with torch.no_grad():
        cache.k_cache.copy_(k_pool_save)
        cache.v_cache.copy_(v_pool_save)
        out_new, _ = attn(x, x, x, torch.zeros((0, 0, 0), device=device), pos_emb, cache=cache)
        cache.k_cache.copy_(k_pool_save)
        cache.v_cache.copy_(v_pool_save)
        out_ref = _ref_paged(attn, x, pos_emb, cache)

    torch.testing.assert_close(out_new, out_ref, rtol=1e-4, atol=1e-4)
