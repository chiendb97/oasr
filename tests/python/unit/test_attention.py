"""
Unit tests for OASR attention layers, using WeNet as ground truth.

These tests compare the outputs of the OASR attention implementations under
`oasr.layers.attention.attention` against the original WeNet reference
implementations in `wenet.models.transformer.attention`.

Tests are skipped automatically if WeNet is not available.
"""

from __future__ import annotations

import pytest
import torch


wenet = pytest.importorskip("wenet")

from wenet.models.transformer import attention as wenet_attn  # type: ignore  # noqa: E402
from wenet.utils import rope_utils as wenet_rope_utils  # type: ignore  # noqa: E402

from oasr.layers.attention import attention as oasr_attn  # noqa: E402


def _randn(*shape: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=torch.float32)


@pytest.mark.parametrize("n_head,n_feat,time1,time2", [(4, 16, 5, 7), (2, 8, 3, 3)])
@pytest.mark.parametrize("use_sdpa", [False])
def test_multi_headed_attention_matches_wenet(n_head, n_feat, time1, time2, use_sdpa):
    """MultiHeadedAttention outputs should match WeNet implementation."""
    torch.manual_seed(0)

    dropout = 0.0  # keep deterministic

    ref = wenet_attn.MultiHeadedAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout,
        use_sdpa=use_sdpa,
    )
    impl = oasr_attn.MultiHeadedAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout,
        use_sdpa=use_sdpa,
    )
    # Ensure identical weights
    impl.load_state_dict(ref.state_dict())

    ref.eval()
    impl.eval()

    batch = 2
    query = _randn(batch, time1, n_feat)
    key = _randn(batch, time2, n_feat)
    value = _randn(batch, time2, n_feat)

    # Padding mask: shape (batch, 1, time2), 1 = keep, 0 = mask
    mask = torch.ones(batch, 1, time2, dtype=torch.bool)
    # Set some positions to 0 to test masking
    if time2 > 1:
        mask[:, :, -1] = 0

    # Default cache (no past)
    cache = (
        torch.zeros(0, 0, 0, 0),
        torch.zeros(0, 0, 0, 0),
    )

    with torch.no_grad():
        out_ref, cache_ref = ref(query, key, value, mask, torch.empty(0), cache)
        out_impl, cache_impl = impl(query, key, value, mask, torch.empty(0), cache)

    assert out_ref.shape == out_impl.shape
    torch.testing.assert_close(out_impl, out_ref, rtol=1e-5, atol=1e-6)

    # Cache contents and shapes should also match
    assert len(cache_ref) == len(cache_impl)
    for c_ref, c_impl in zip(cache_ref, cache_impl):
        assert c_ref.shape == c_impl.shape
        torch.testing.assert_close(c_impl, c_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_head,n_feat,time", [(4, 16, 6)])
@pytest.mark.parametrize("use_sdpa", [False])
def test_rel_position_multi_headed_attention_matches_wenet(
    n_head, n_feat, time, use_sdpa
):
    """RelPositionMultiHeadedAttention should match WeNet implementation."""
    torch.manual_seed(1)

    dropout = 0.0

    ref = wenet_attn.RelPositionMultiHeadedAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout,
        use_sdpa=use_sdpa,
    )
    impl = oasr_attn.RelPositionMultiHeadedAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout,
        use_sdpa=use_sdpa,
    )
    impl.load_state_dict(ref.state_dict())

    ref.eval()
    impl.eval()

    batch = 2
    query = _randn(batch, time, n_feat)
    key = _randn(batch, time, n_feat)
    value = _randn(batch, time, n_feat)

    # Positional embedding (WeNet expects [batch, time, size])
    pos_emb = _randn(batch, time, n_feat)

    # Full attention mask (no padding masked)
    mask = torch.ones(batch, time, time, dtype=torch.bool)

    cache = (
        torch.zeros(0, 0, 0, 0),
        torch.zeros(0, 0, 0, 0),
    )

    with torch.no_grad():
        out_ref, cache_ref = ref(query, key, value, mask, pos_emb, cache)
        out_impl, cache_impl = impl(query, key, value, mask, pos_emb, cache)

    assert out_ref.shape == out_impl.shape
    torch.testing.assert_close(out_impl, out_ref, rtol=1e-5, atol=1e-6)

    assert len(cache_ref) == len(cache_impl)
    for c_ref, c_impl in zip(cache_ref, cache_impl):
        assert c_ref.shape == c_impl.shape
        torch.testing.assert_close(c_impl, c_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_head,n_feat,time_q,time_kv", [(4, 16, 3, 5)])
@pytest.mark.parametrize("use_sdpa", [False])
def test_multi_headed_cross_attention_matches_wenet(
    n_head, n_feat, time_q, time_kv, use_sdpa
):
    """MultiHeadedCrossAttention should match WeNet implementation."""
    torch.manual_seed(2)

    dropout = 0.0

    ref = wenet_attn.MultiHeadedCrossAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout,
        use_sdpa=use_sdpa,
    )
    impl = oasr_attn.MultiHeadedCrossAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout,
        use_sdpa=use_sdpa,
    )
    impl.load_state_dict(ref.state_dict())

    ref.eval()
    impl.eval()

    batch = 2
    query = _randn(batch, time_q, n_feat)
    key = _randn(batch, time_kv, n_feat)
    value = _randn(batch, time_kv, n_feat)

    # Mask shape (batch, time_q, time_kv)
    mask = torch.ones(batch, time_q, time_kv, dtype=torch.bool)
    if time_kv > 1:
        mask[:, :, -1] = 0

    cache = (
        torch.zeros(0, 0, 0, 0),
        torch.zeros(0, 0, 0, 0),
    )

    with torch.no_grad():
        out_ref, cache_ref = ref(query, key, value, mask, torch.empty(0), cache)
        out_impl, cache_impl = impl(query, key, value, mask, torch.empty(0), cache)

    assert out_ref.shape == out_impl.shape
    torch.testing.assert_close(out_impl, out_ref, rtol=1e-5, atol=1e-6)

    assert len(cache_ref) == len(cache_impl)
    for c_ref, c_impl in zip(cache_ref, cache_impl):
        assert c_ref.shape == c_impl.shape
        torch.testing.assert_close(c_impl, c_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_head,n_feat,time", [(4, 16, 5)])
@pytest.mark.parametrize("style", ["google", "llama"])
@pytest.mark.parametrize("use_sdpa", [False])
def test_rope_multi_headed_attention_matches_wenet(
    n_head, n_feat, time, style, use_sdpa
):
    """RopeMultiHeadedAttention should match WeNet implementation."""
    torch.manual_seed(3)

    dropout = 0.0

    ref = wenet_attn.RopeMultiHeadedAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout,
        use_sdpa=use_sdpa,
        style=style,
    )
    impl = oasr_attn.RopeMultiHeadedAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout,
        use_sdpa=use_sdpa,
        style=style,
    )
    impl.load_state_dict(ref.state_dict())

    ref.eval()
    impl.eval()

    batch = 2
    query = _randn(batch, time, n_feat)
    key = _randn(batch, time, n_feat)
    value = _randn(batch, time, n_feat)

    # Use WeNet's RoPE utility as the ground truth for pos_emb.
    # We need to reshape the frequencies so they broadcast correctly with
    # q/k of shape [batch, time, n_head, head_dim].
    head_dim = n_feat // n_head
    freqs_cis = wenet_rope_utils.precompute_freqs_cis(head_dim, time)
    # Original shape: [time, head_dim/2] (complex). Make it [1, time, 1, head_dim/2]
    # so that it broadcasts over batch and head dimensions.
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Mask (no positions masked)
    mask = torch.ones(batch, 1, time, dtype=torch.bool)

    cache = (
        torch.zeros(0, 0, 0, 0),
        torch.zeros(0, 0, 0, 0),
    )

    with torch.no_grad():
        out_ref, cache_ref = ref(query, key, value, mask, freqs_cis, cache)
        out_impl, cache_impl = impl(query, key, value, mask, freqs_cis, cache)

    assert out_ref.shape == out_impl.shape
    torch.testing.assert_close(out_impl, out_ref, rtol=1e-5, atol=1e-6)

    assert len(cache_ref) == len(cache_impl)
    for c_ref, c_impl in zip(cache_ref, cache_impl):
        assert c_ref.shape == c_impl.shape
        torch.testing.assert_close(c_impl, c_ref, rtol=1e-5, atol=1e-6)


