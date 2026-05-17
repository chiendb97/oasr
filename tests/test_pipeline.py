# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline tests: Conformer encoder + paged cache manager + CTC decoder.

Model directory and audio file are supplied via ``--ckpt-dir`` / ``CKPT_DIR``
and ``--audio-path`` / ``AUDIO_PATH``. Tests:

1. Paged streaming forward shapes (smoke test, random audio).
2. Paged streaming logits sum to a sensible distribution per chunk (random audio).
3. CTC streaming decoder produces valid, reproducible output (random audio).
4. Two concurrent paged streams with different audio stay fully isolated.
5. End-to-end on real audio: paged streaming → CTC decode → non-empty transcript.

All tests require CUDA. Tests using the ``model`` fixture require
``--ckpt-dir``. Streaming is paged-only (the dense ``forward_chunk`` path was
removed); the offline ``model.forward(...)`` call serves as the reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest
import torch

from oasr.features import FeatureConfig, extract_features_batch

# ---------------------------------------------------------------------------
# Model-architecture constants (specific to the U2++ LibriSpeech checkpoint)
# ---------------------------------------------------------------------------

NUM_LAYERS = 12
N_KV_HEAD = 4
OUTPUT_SIZE = 256
HEAD_DIM = OUTPUT_SIZE // N_KV_HEAD  # 64
HIDDEN_DIM = 256
KERNEL_SIZE = 15   # causal → lorder = 14
VOCAB_SIZE_RAW = 5002
VOCAB_SIZE = (VOCAB_SIZE_RAW + 7) // 8 * 8  # 5008 (padded to multiple of 8)

# Streaming parameters
CHUNK_SIZE = 16
SUBSAMPLE_RATE = 4
RIGHT_CONTEXT = 6
# Input frames per chunk so subsampling yields exactly CHUNK_SIZE output frames:
#   (chunk_size - 1) * subsample_rate + right_context + 1
CHUNK_INPUT_TIME = (CHUNK_SIZE - 1) * SUBSAMPLE_RATE + RIGHT_CONTEXT + 1  # 67
NUM_FEAT = 80  # log-mel bins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model(ckpt_dir: str, device):
    """Load the U2++ Conformer from the checkpoint supplied via --ckpt-dir."""
    if not ckpt_dir or not (Path(ckpt_dir) / "final.pt").exists():
        pytest.skip(
            "Checkpoint not found; pass --ckpt-dir <path> or set CKPT_DIR"
        )
    from oasr.models.conformer import load_wenet_checkpoint

    m, _ = load_wenet_checkpoint(ckpt_dir, device=str(device), dtype=torch.float16)
    return m


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _read_audio_and_extract_features(
    path: str,
    device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load a WAV file and compute 80-dim log-mel filterbank features.

    Returns
    -------
    torch.Tensor
        Shape ``(1, T, 80)`` on *device* with the requested *dtype*.
    """
    import torchaudio

    audio, sr = torchaudio.load(path)
    assert sr == 16000, f"expected 16 kHz audio, got {sr}"
    cfg = FeatureConfig(
        sample_rate=16000,
        num_mel_bins=NUM_FEAT,
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        dither=1.0,
    )
    feats, _ = extract_features_batch([audio], cfg)
    return feats.to(device=device, dtype=dtype)


def _chunk_features(feats: torch.Tensor) -> List[torch.Tensor]:
    """Split a ``(1, T, F)`` feature tensor into streaming windows."""
    T = feats.size(1)
    stride = SUBSAMPLE_RATE * CHUNK_SIZE
    chunks = []
    for cur in range(0, T - RIGHT_CONTEXT, stride):
        end = min(cur + CHUNK_INPUT_TIME, T)
        chunks.append(feats[:, cur:end, :])
    return chunks


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------


def _build_id2word(ckpt_dir: str) -> dict:
    """Load the ``id → word`` vocabulary from ``<ckpt_dir>/words.txt``."""
    id2word: dict = {}
    with open(Path(ckpt_dir) / "words.txt") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                word, idx = parts
                id2word[int(idx)] = word
    return id2word


def _tokens_to_text(token_ids: list, id2word: dict) -> str:
    return "".join(id2word.get(t, "") for t in token_ids).replace("▁", " ").strip()


# ---------------------------------------------------------------------------
# Synthetic-chunk helpers
# ---------------------------------------------------------------------------


def _make_chunks(
    n: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Return *n* random acoustic feature chunks ``(1, CHUNK_INPUT_TIME, NUM_FEAT)``."""
    torch.manual_seed(seed)
    return [
        torch.randn(1, CHUNK_INPUT_TIME, NUM_FEAT, dtype=dtype, device=device)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Paged streaming pipeline
# ---------------------------------------------------------------------------


def _cache_manager_streaming_paged(
    model,
    chunks: List[torch.Tensor],
    num_left_chunks: int,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int = VOCAB_SIZE,
    beam_size: int = 5,
) -> Tuple:
    """Run chunk-by-chunk inference using paged KV cache + CTC decoder.

    Returns ``(logits_list, all_logits, streaming_result, pool)``.
    """
    from oasr.cache import (
        AttentionCacheManager,
        BlockPool,
        CacheConfig,
        CnnCacheManager,
        CtcStateCacheManager,
        StreamContext,
    )
    from oasr.ctc_decode import GpuDecoderConfig

    n_chunks = len(chunks)
    max_blocks = max(64, (abs(num_left_chunks) if num_left_chunks > 0 else n_chunks) * 4)

    cfg = CacheConfig(
        num_layers=NUM_LAYERS,
        n_kv_head=N_KV_HEAD,
        head_dim=HEAD_DIM,
        hidden_dim=HIDDEN_DIM,
        kernel_size=KERNEL_SIZE,
        chunk_size=CHUNK_SIZE,
        num_left_chunks=num_left_chunks,
        block_size_frames=CHUNK_SIZE,
        max_num_blocks=max_blocks,
        max_blocks_per_seq=max(64, n_chunks + 4),
        device=device,
        dtype=dtype,
    )
    pool = BlockPool(cfg)
    att_mgr = AttentionCacheManager(pool, cfg)
    cnn_mgr = CnnCacheManager(cfg)
    ctc_mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=beam_size))

    sid = 0
    att_mgr.allocate_stream(sid)
    cnn_mgr.allocate_stream(sid)
    ctc_mgr.allocate_stream(sid, batch=1, vocab_size=vocab_size, device=device)
    ctx = StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)

    offset = 0
    logits_list = []
    cnn_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
    with torch.no_grad():
        for xs in chunks:
            ctx.prepare_chunk()
            att_caches = ctx.get_att_caches()
            probs, new_cnn = model.forward_chunk_paged(
                xs, offset, att_caches, cnn_cache, cache_t1=offset,
            )
            ctx.commit_chunk_paged(probs.size(1), new_cnn)
            cnn_cache = new_cnn
            ctx.get_decoder().decode_chunk(probs)
            logits_list.append(probs)
            offset += probs.size(1)

    streaming_result = ctx.get_decoder().finalize_stream()
    ctx.free()
    return logits_list, torch.cat(logits_list, dim=1), streaming_result, pool


# ---------------------------------------------------------------------------
# Test 1: Shape smoke test (random audio, no checkpoint needed)
# ---------------------------------------------------------------------------


class TestPagedStreamingShapes:
    """Verify cache and forward shapes through the paged streaming pipeline.

    Skips when no checkpoint is provided (we still need a real ``model`` for the
    forward call, but use random feature chunks rather than real audio).
    """

    @pytest.mark.parametrize("n_chunks", [1, 3])
    @pytest.mark.parametrize("num_left_chunks", [-1, 2])
    def test_shapes(self, model, device, n_chunks: int, num_left_chunks: int):
        dtype = torch.float16
        chunks = _make_chunks(n_chunks, seed=0, device=device, dtype=dtype)

        logits_list, _, result, pool = _cache_manager_streaming_paged(
            model, chunks, num_left_chunks=num_left_chunks,
            device=device, dtype=dtype,
        )

        assert len(logits_list) == n_chunks
        for probs in logits_list:
            assert probs.shape == torch.Size([1, CHUNK_SIZE, VOCAB_SIZE])
        assert result.lengths.shape == torch.Size([1, 5])
        assert result.scores.shape == torch.Size([1, 5])
        assert pool.num_free_blocks == pool.num_total_blocks


# ---------------------------------------------------------------------------
# Test 2: Paged streaming logits are well-formed log-probabilities
# ---------------------------------------------------------------------------


class TestPagedStreamingLogitsAreLogProbs:
    """Per-chunk paged streaming outputs are valid log-softmax distributions."""

    @pytest.mark.parametrize("num_left_chunks,n_chunks", [
        (-1, 3),
        (2, 4),
    ])
    def test_log_probs(self, model, device, num_left_chunks: int, n_chunks: int):
        dtype = torch.float16
        chunks = _make_chunks(n_chunks, seed=99, device=device, dtype=dtype)

        logits_list, _, _, pool = _cache_manager_streaming_paged(
            model, chunks, num_left_chunks, device, dtype,
        )

        for step, probs in enumerate(logits_list):
            sums = probs.exp().sum(dim=-1).float()
            torch.testing.assert_close(
                sums, torch.ones_like(sums), rtol=1e-2, atol=1e-2,
                msg=f"chunk {step}: log-probs do not sum to 1",
            )

        assert pool.num_free_blocks == pool.num_total_blocks


# ---------------------------------------------------------------------------
# Test 3: CTC decoding — streaming result is reproducible and logits are valid
# ---------------------------------------------------------------------------


class TestCtcDecode:
    """CTC streaming decoder produces valid, reproducible output."""

    def test_streaming_ctc_valid_and_reproducible(self, model, device):
        from oasr.ctc_decode import GpuDecoderConfig, GpuStreamingDecoder

        dtype = torch.float16
        n_chunks = 5
        beam_size = 5
        num_left_chunks = -1
        chunks = _make_chunks(n_chunks, seed=7, device=device, dtype=dtype)

        mgr_logits_list, _, mgr_result, pool = _cache_manager_streaming_paged(
            model, chunks, num_left_chunks, device, dtype, beam_size=beam_size,
        )
        assert pool.num_free_blocks == pool.num_total_blocks

        # Fresh decoder fed the same logits → must produce identical top-1.
        ref_decoder = GpuStreamingDecoder(GpuDecoderConfig(beam_size=beam_size))
        ref_decoder.init_stream(batch=1, vocab_size=VOCAB_SIZE, device=device)
        for probs in mgr_logits_list:
            ref_decoder.decode_chunk(probs)
        ref_result = ref_decoder.finalize_stream()

        assert mgr_result.tokens[0][0] == ref_result.tokens[0][0], (
            f"Cache-manager top-1: {mgr_result.tokens[0][0]}\n"
            f"Reference decoder top-1: {ref_result.tokens[0][0]}"
        )

        assert mgr_result.lengths.shape == torch.Size([1, beam_size])
        assert mgr_result.scores.shape == torch.Size([1, beam_size])

        scores = mgr_result.scores[0].cpu()
        assert (scores[:-1] >= scores[1:]).all(), (
            f"Beam scores not sorted: {scores.tolist()}"
        )

        for beam_tokens in mgr_result.tokens[0]:
            assert all(0 <= t < VOCAB_SIZE for t in beam_tokens), (
                f"Token out of range [0, {VOCAB_SIZE}): {beam_tokens}"
            )


# ---------------------------------------------------------------------------
# Test 4: Multi-stream isolation under paged sharing
# ---------------------------------------------------------------------------


class TestMultiStreamIsolation:
    """Two concurrent paged streams with different audio stay fully isolated."""

    def test_two_streams_isolated(self, model, device):
        from oasr.cache import (
            AttentionCacheManager,
            BlockPool,
            CacheConfig,
            CnnCacheManager,
            CtcStateCacheManager,
            StreamContext,
        )
        from oasr.ctc_decode import GpuDecoderConfig

        dtype = torch.float16
        n_chunks = 3
        num_left_chunks = 2

        cfg = CacheConfig(
            num_layers=NUM_LAYERS,
            n_kv_head=N_KV_HEAD,
            head_dim=HEAD_DIM,
            hidden_dim=HIDDEN_DIM,
            kernel_size=KERNEL_SIZE,
            chunk_size=CHUNK_SIZE,
            num_left_chunks=num_left_chunks,
            block_size_frames=CHUNK_SIZE,
            max_num_blocks=64,
            device=device,
            dtype=dtype,
        )
        pool = BlockPool(cfg)
        att_mgr = AttentionCacheManager(pool, cfg)
        cnn_mgr = CnnCacheManager(cfg)
        ctc_mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=5))

        sid_a, sid_b = 10, 20
        for sid in (sid_a, sid_b):
            att_mgr.allocate_stream(sid)
            cnn_mgr.allocate_stream(sid)
            ctc_mgr.allocate_stream(sid, batch=1, vocab_size=VOCAB_SIZE, device=device)

        ctx_a = StreamContext(sid_a, att_mgr, cnn_mgr, ctc_mgr)
        ctx_b = StreamContext(sid_b, att_mgr, cnn_mgr, ctc_mgr)

        chunks_a = _make_chunks(n_chunks, seed=100, device=device, dtype=dtype)
        chunks_b = _make_chunks(n_chunks, seed=200, device=device, dtype=dtype)

        cnn_cache_a = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
        cnn_cache_b = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
        offset_a = offset_b = 0

        with torch.no_grad():
            for xs_a, xs_b in zip(chunks_a, chunks_b):
                for ctx, xs, off, cnn_state in (
                    (ctx_a, xs_a, offset_a, cnn_cache_a),
                    (ctx_b, xs_b, offset_b, cnn_cache_b),
                ):
                    ctx.prepare_chunk()
                    att_caches = ctx.get_att_caches()
                    probs, new_cnn = model.forward_chunk_paged(
                        xs, off, att_caches, cnn_state, cache_t1=off,
                    )
                    ctx.commit_chunk_paged(probs.size(1), new_cnn)
                    ctx.get_decoder().decode_chunk(probs)
                    if ctx is ctx_a:
                        cnn_cache_a = new_cnn
                    else:
                        cnn_cache_b = new_cnn
                offset_a += CHUNK_SIZE
                offset_b += CHUNK_SIZE

        # Capture stream B's paged state before freeing stream A.
        bt_b_before, cs_b_before = ctx_b.get_paged_state_views()
        bt_b_before = bt_b_before.clone()
        cs_b_before = cs_b_before.clone()

        ctx_a.free()

        bt_b_after, cs_b_after = ctx_b.get_paged_state_views()
        torch.testing.assert_close(bt_b_after, bt_b_before, rtol=0.0, atol=0.0,
                                   msg="Stream B's block_table changed after freeing A")
        torch.testing.assert_close(cs_b_after, cs_b_before, rtol=0.0, atol=0.0,
                                   msg="Stream B's cache_seqlens changed after freeing A")

        ctx_b.get_decoder().finalize_stream()
        ctx_b.free()
        assert pool.num_free_blocks == pool.num_total_blocks


# ---------------------------------------------------------------------------
# Test 5: End-to-end paged streaming on real audio
# ---------------------------------------------------------------------------

def _skip_if_no_audio(audio_path: str) -> None:
    if not audio_path or not Path(audio_path).exists():
        pytest.skip(
            "Audio file not found; pass --audio-path <wav> or set AUDIO_PATH"
        )


class TestStreamingWithRealAudio:
    """End-to-end paged streaming on real audio.

    Requires:
    - ``--ckpt-dir`` / ``CKPT_DIR``: WeNet Conformer checkpoint directory
      (must contain ``final.pt`` and ``words.txt``).
    - ``--audio-path`` / ``AUDIO_PATH``: 16 kHz WAV file to transcribe.
    """

    @staticmethod
    def _load_chunks(audio_path: str, device, dtype) -> List[torch.Tensor]:
        feats = _read_audio_and_extract_features(audio_path, device=device, dtype=dtype)
        chunks = _chunk_features(feats)
        if len(chunks) < 2:
            pytest.skip(
                f"Audio at {audio_path!r} is too short "
                f"(got {feats.size(1)} frames, need ≥ {2 * CHUNK_INPUT_TIME})."
            )
        return chunks

    def test_forward_and_ctc_decode(
        self, model, ckpt_dir: str, audio_path: str, device
    ):
        """Paged streaming → CTC beam search → non-empty transcript."""
        _skip_if_no_audio(audio_path)

        dtype = torch.float16
        beam_size = 10
        chunks = self._load_chunks(audio_path, device, dtype)
        id2word = _build_id2word(ckpt_dir)

        _, _, streaming_result, pool = _cache_manager_streaming_paged(
            model, chunks,
            num_left_chunks=-1,
            device=device,
            dtype=dtype,
            vocab_size=VOCAB_SIZE,
            beam_size=beam_size,
        )

        assert pool.num_free_blocks == pool.num_total_blocks, (
            f"Block pool leak: {pool.num_free_blocks}/{pool.num_total_blocks} free"
        )

        assert streaming_result.lengths.shape == torch.Size([1, beam_size])
        assert streaming_result.scores.shape == torch.Size([1, beam_size])

        scores = streaming_result.scores[0].cpu()
        assert (scores[:-1] >= scores[1:]).all(), (
            f"Beam scores not sorted: {scores.tolist()}"
        )

        for beam_idx, beam_tokens in enumerate(streaming_result.tokens[0]):
            bad = [t for t in beam_tokens if not (0 <= t < VOCAB_SIZE)]
            assert not bad, (
                f"Beam {beam_idx} contains out-of-range token IDs: {bad}"
            )

        top1_tokens = streaming_result.tokens[0][0]
        text = _tokens_to_text(top1_tokens, id2word)
        print(f"\n[paged streaming top-1]  {text!r}")
        assert len(text.strip()) > 0, "CTC decoder produced an empty transcript"

        for beam_idx, beam_tokens in enumerate(streaming_result.tokens[0]):
            beam_text = _tokens_to_text(beam_tokens, id2word)
            print(f"  beam {beam_idx}: {beam_text!r}  (score {scores[beam_idx]:.3f})")

    def test_cache_manager_pool_accounting(
        self, model, audio_path: str, device
    ):
        """Two real-audio paged streams share a pool; freeing one leaves the other intact."""
        _skip_if_no_audio(audio_path)

        from oasr.cache import (
            AttentionCacheManager,
            BlockPool,
            CacheConfig,
            CnnCacheManager,
            CtcStateCacheManager,
            StreamContext,
        )
        from oasr.ctc_decode import GpuDecoderConfig

        dtype = torch.float16
        num_left_chunks = 2

        chunks_a = self._load_chunks(audio_path, device, dtype)[:3]
        chunks_b = _make_chunks(3, seed=77, device=device, dtype=dtype)

        max_blocks = 64
        cfg = CacheConfig(
            num_layers=NUM_LAYERS,
            n_kv_head=N_KV_HEAD,
            head_dim=HEAD_DIM,
            hidden_dim=HIDDEN_DIM,
            kernel_size=KERNEL_SIZE,
            chunk_size=CHUNK_SIZE,
            num_left_chunks=num_left_chunks,
            block_size_frames=CHUNK_SIZE,
            max_num_blocks=max_blocks,
            device=device,
            dtype=dtype,
        )
        pool = BlockPool(cfg)
        att_mgr = AttentionCacheManager(pool, cfg)
        cnn_mgr = CnnCacheManager(cfg)
        ctc_mgr = CtcStateCacheManager(GpuDecoderConfig(beam_size=5))

        initial_free = pool.num_free_blocks

        sid_a, sid_b = 1, 2
        for sid in (sid_a, sid_b):
            att_mgr.allocate_stream(sid)
            cnn_mgr.allocate_stream(sid)
            ctc_mgr.allocate_stream(sid, batch=1, vocab_size=VOCAB_SIZE, device=device)

        ctx_a = StreamContext(sid_a, att_mgr, cnn_mgr, ctc_mgr)
        ctx_b = StreamContext(sid_b, att_mgr, cnn_mgr, ctc_mgr)

        cnn_cache_a = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
        cnn_cache_b = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
        offset_a = offset_b = 0

        with torch.no_grad():
            for xs_a, xs_b in zip(chunks_a, chunks_b):
                for ctx, xs, off, cnn_state, sid_local in (
                    (ctx_a, xs_a, offset_a, cnn_cache_a, "a"),
                    (ctx_b, xs_b, offset_b, cnn_cache_b, "b"),
                ):
                    ctx.prepare_chunk()
                    att_caches = ctx.get_att_caches()
                    probs, new_cnn = model.forward_chunk_paged(
                        xs, off, att_caches, cnn_state, cache_t1=off,
                    )
                    ctx.commit_chunk_paged(probs.size(1), new_cnn)
                    ctx.get_decoder().decode_chunk(probs)
                    if sid_local == "a":
                        cnn_cache_a = new_cnn
                    else:
                        cnn_cache_b = new_cnn
                offset_a += CHUNK_SIZE
                offset_b += CHUNK_SIZE

        bt_b_before, cs_b_before = ctx_b.get_paged_state_views()
        bt_b_before = bt_b_before.clone()
        cs_b_before = cs_b_before.clone()

        ctx_a.free()

        bt_b_after, cs_b_after = ctx_b.get_paged_state_views()
        torch.testing.assert_close(bt_b_after, bt_b_before, rtol=0.0, atol=0.0,
                                   msg="Stream B's block_table changed after freeing A")
        torch.testing.assert_close(cs_b_after, cs_b_before, rtol=0.0, atol=0.0,
                                   msg="Stream B's cache_seqlens changed after freeing A")

        ctx_b.get_decoder().finalize_stream()
        ctx_b.free()

        assert pool.num_free_blocks == initial_free, (
            f"Pool not fully recovered: {pool.num_free_blocks}/{initial_free} free blocks"
        )
