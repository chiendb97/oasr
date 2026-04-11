# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline tests: Conformer encoder + cache manager + CTC decoder.

Model directory and audio file are supplied via ``--ckpt-dir`` / ``CKPT_DIR``
and ``--audio-path`` / ``AUDIO_PATH`` (same CLI / env-var pattern as
``TestCtcPrefixBeamSearch`` in ``test_decoder.py``).  The fixtures are
registered in ``conftest.py``.

Tests:
  1. Forward-chunk streaming shapes (smoke test, random audio).
  2. Dense cache-manager path logits match the raw reference loop (random audio).
  3. Paged cache-manager path logits match the raw reference loop (random audio).
  4. CTC streaming decoder produces valid, reproducible output (random audio).
  5. Two concurrent streams with different audio stay fully isolated (random audio).
  6. Paged path produces logits identical to dense reference on real audio.

All tests require CUDA.  Tests 1–5 use random feature tensors and run without
a checkpoint; only test 6 and the ``model`` fixture require ``--ckpt-dir``.
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

    Uses :mod:`oasr.features` (same Kaldi FBANK settings as the legacy tests:
    25 ms frame, 10 ms shift, dither 1.0).

    Returns
    -------
    torch.Tensor
        Shape ``(1, T, 80)`` on *device* with the requested *dtype*.
    """
    import torchaudio

    audio, sr = torchaudio.load(path)
    audio = audio * (1 << 15)
    cfg = FeatureConfig(
        sample_rate=int(sr),
        num_mel_bins=80,
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        dither=1.0,
    )
    feats, _ = extract_features_batch([audio], cfg)
    return feats.to(device=device, dtype=dtype)


def _chunk_features(feats: torch.Tensor) -> List[torch.Tensor]:
    """Split a ``(1, T, F)`` feature tensor into streaming windows.

    Each window is at most ``CHUNK_INPUT_TIME`` frames wide with a stride of
    ``CHUNK_SIZE * SUBSAMPLE_RATE`` frames.  The last window may be shorter
    when audio does not divide evenly (mirroring ``forward_chunk_by_chunk``).

    Returns
    -------
    list[Tensor]
        Each element has shape ``(1, ≤CHUNK_INPUT_TIME, F)``.
    """
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
    """Load the ``id → word`` vocabulary from ``<ckpt_dir>/words.txt``.

    The file format (one entry per line) is ``<word> <integer_id>``, matching
    the WeNet convention.  Special tokens (``<blank>``, ``<unk>``,
    ``<sos/eos>``) are retained so that all valid token IDs map cleanly.
    """
    id2word: dict = {}
    with open(Path(ckpt_dir) / "words.txt") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                word, idx = parts
                id2word[int(idx)] = word
    return id2word


def _tokens_to_text(token_ids: list, id2word: dict) -> str:
    """Convert a CTC top-1 token ID sequence to a readable text string.

    Sentencepiece ``▁`` (U+2581) word-boundary markers are replaced with
    spaces and the result is stripped.
    """
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
    """Return *n* random acoustic feature chunks, each ``(1, CHUNK_INPUT_TIME, NUM_FEAT)``."""
    torch.manual_seed(seed)
    return [
        torch.randn(1, CHUNK_INPUT_TIME, NUM_FEAT, dtype=dtype, device=device)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _reference_streaming(
    model,
    chunks: List[torch.Tensor],
    required_cache_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Run chunk-by-chunk inference with direct dense cache passthrough.

    Returns
    -------
    logits_list : list of Tensor
        Per-chunk log-prob outputs, each ``(1, CHUNK_SIZE, VOCAB_SIZE)``.
    all_logits : Tensor
        Concatenated log-probs ``(1, n*CHUNK_SIZE, VOCAB_SIZE)``.
    """
    att_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
    cnn_cache = torch.zeros(0, 0, 0, 0, dtype=dtype, device=device)
    offset = 0
    logits_list = []
    with torch.no_grad():
        for xs in chunks:
            probs, att_cache, cnn_cache = model.forward_chunk(
                xs, offset, required_cache_size, att_cache, cnn_cache
            )
            logits_list.append(probs)
            offset += probs.size(1)
    return logits_list, torch.cat(logits_list, dim=1)


def _cache_manager_streaming(
    model,
    chunks: List[torch.Tensor],
    required_cache_size: int,
    num_left_chunks: int,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int = VOCAB_SIZE,
    beam_size: int = 5,
) -> Tuple:
    """Run chunk-by-chunk inference through the dense cache manager + CTC decoder.

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
    with torch.no_grad():
        for xs in chunks:
            att_cache = ctx.get_att_cache()
            cnn_cache = ctx.get_cnn_cache()
            probs, new_att, new_cnn = model.forward_chunk(
                xs, offset, required_cache_size, att_cache, cnn_cache
            )
            n_frames = probs.size(1)
            ctx.commit_chunk(new_att[:, :, -n_frames:, :], new_cnn)
            ctx.get_decoder().decode_chunk(probs)
            logits_list.append(probs)
            offset += n_frames

    streaming_result = ctx.get_decoder().finalize_stream()
    ctx.free()
    return logits_list, torch.cat(logits_list, dim=1), streaming_result, pool


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

    K/V are written directly into the block pool by ``forward_chunk_paged``.

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
            probs, new_cnn = model.forward_chunk_paged(xs, offset, att_caches, cnn_cache)
            ctx.commit_chunk_paged(probs.size(1), new_cnn)
            cnn_cache = new_cnn
            ctx.get_decoder().decode_chunk(probs)
            logits_list.append(probs)
            offset += probs.size(1)

    streaming_result = ctx.get_decoder().finalize_stream()
    ctx.free()
    return logits_list, torch.cat(logits_list, dim=1), streaming_result, pool


# ---------------------------------------------------------------------------
# Test 1: Shape smoke test
# ---------------------------------------------------------------------------


class TestPipelineShapes:
    """Verify output and cache shapes at every step of the streaming pipeline."""

    @pytest.mark.parametrize("n_chunks", [1, 3])
    @pytest.mark.parametrize("num_left_chunks", [-1, 2])
    def test_shapes(self, model, device, n_chunks: int, num_left_chunks: int):
        dtype = torch.float16
        required_cache_size = CHUNK_SIZE * num_left_chunks

        from oasr.cache import (
            AttentionCacheManager,
            BlockPool,
            CacheConfig,
            CnnCacheManager,
            CtcStateCacheManager,
            StreamContext,
        )
        from oasr.ctc_decode import GpuDecoderConfig

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

        sid = 0
        att_mgr.allocate_stream(sid)
        cnn_mgr.allocate_stream(sid)
        ctc_mgr.allocate_stream(sid, batch=1, vocab_size=VOCAB_SIZE, device=device)
        ctx = StreamContext(sid, att_mgr, cnn_mgr, ctc_mgr)

        chunks = _make_chunks(n_chunks, seed=0, device=device, dtype=dtype)
        offset = 0

        with torch.no_grad():
            for i, xs in enumerate(chunks):
                att_cache = ctx.get_att_cache()
                cnn_cache = ctx.get_cnn_cache()

                if i == 0:
                    assert att_cache.shape == torch.Size([0, 0, 0, 0])
                else:
                    assert att_cache.dim() == 4
                    assert att_cache.size(0) == NUM_LAYERS
                    assert att_cache.size(1) == N_KV_HEAD
                    assert att_cache.size(3) == HEAD_DIM * 2

                assert cnn_cache.shape == torch.Size(
                    [NUM_LAYERS, 1, KERNEL_SIZE - 1, HIDDEN_DIM]
                )

                probs, new_att, new_cnn = model.forward_chunk(
                    xs, offset, required_cache_size, att_cache, cnn_cache
                )

                assert probs.shape == torch.Size([1, CHUNK_SIZE, VOCAB_SIZE])
                assert new_att.dim() == 4
                assert new_att.size(0) == NUM_LAYERS
                assert new_att.size(3) == HEAD_DIM * 2
                assert new_cnn.shape == torch.Size(
                    [NUM_LAYERS, 1, KERNEL_SIZE - 1, HIDDEN_DIM]
                )

                ctx.commit_chunk(new_att[:, :, -CHUNK_SIZE:, :], new_cnn)
                ctx.get_decoder().decode_chunk(probs)
                offset += CHUNK_SIZE

        result = ctx.get_decoder().finalize_stream()
        ctx.free()

        assert len(result.tokens) == 1
        assert len(result.tokens[0]) == 5
        assert result.lengths.shape == torch.Size([1, 5])
        assert result.scores.shape == torch.Size([1, 5])


# ---------------------------------------------------------------------------
# Test 2: Dense cache-manager path matches direct reference loop
# ---------------------------------------------------------------------------


class TestStreamingMatchesReference:
    """Dense cache-manager path produces identical logits to the raw reference."""

    @pytest.mark.parametrize("num_left_chunks,n_chunks", [
        (-1, 3),
        (2, 4),
    ])
    def test_logits_match(self, model, device, num_left_chunks, n_chunks):
        dtype = torch.float16
        required_cache_size = CHUNK_SIZE * num_left_chunks
        chunks = _make_chunks(n_chunks, seed=42, device=device, dtype=dtype)

        ref_logits_list, _ = _reference_streaming(
            model, chunks, required_cache_size, dtype, device
        )
        mgr_logits_list, _, _, pool = _cache_manager_streaming(
            model, chunks, required_cache_size, num_left_chunks, device, dtype
        )

        assert len(ref_logits_list) == len(mgr_logits_list) == n_chunks
        for step, (ref, mgr) in enumerate(zip(ref_logits_list, mgr_logits_list)):
            torch.testing.assert_close(
                mgr, ref, rtol=0.0, atol=0.0,
                msg=f"Logit mismatch at chunk {step} (num_left_chunks={num_left_chunks})",
            )

        assert pool.num_free_blocks == pool.num_total_blocks


# ---------------------------------------------------------------------------
# Test 3: Paged cache path matches direct reference loop
# ---------------------------------------------------------------------------


class TestPagedMatchesReference:
    """Paged KV cache path produces logits identical to the dense reference."""

    @pytest.mark.parametrize("num_left_chunks,n_chunks", [
        (-1, 3),
        (2, 4),
    ])
    def test_paged_logits_match_reference(self, model, device, num_left_chunks, n_chunks):
        dtype = torch.float16
        required_cache_size = CHUNK_SIZE * num_left_chunks
        chunks = _make_chunks(n_chunks, seed=99, device=device, dtype=dtype)

        ref_logits_list, _ = _reference_streaming(
            model, chunks, required_cache_size, dtype, device
        )
        paged_logits_list, _, _, pool = _cache_manager_streaming_paged(
            model, chunks, num_left_chunks, device, dtype
        )

        assert len(ref_logits_list) == len(paged_logits_list) == n_chunks
        for step, (ref, paged) in enumerate(zip(ref_logits_list, paged_logits_list)):
            torch.testing.assert_close(
                paged, ref, rtol=0.0, atol=0.0,
                msg=f"Paged logit mismatch at chunk {step} "
                    f"(num_left_chunks={num_left_chunks})",
            )

        assert pool.num_free_blocks == pool.num_total_blocks


# ---------------------------------------------------------------------------
# Test 4: CTC decoding — streaming result is reproducible and logits are valid
# ---------------------------------------------------------------------------


class TestCtcDecode:
    """CTC streaming decoder produces valid, reproducible output."""

    def test_streaming_ctc_valid_and_reproducible(self, model, device):
        from oasr.ctc_decode import GpuDecoderConfig, GpuStreamingDecoder

        dtype = torch.float16
        n_chunks = 5
        beam_size = 5
        num_left_chunks = -1
        required_cache_size = CHUNK_SIZE * num_left_chunks
        chunks = _make_chunks(n_chunks, seed=7, device=device, dtype=dtype)

        mgr_logits_list, _, mgr_result, pool = _cache_manager_streaming(
            model, chunks, required_cache_size, num_left_chunks,
            device, dtype, beam_size=beam_size,
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
# Test 5: Multi-stream isolation
# ---------------------------------------------------------------------------


class TestMultiStreamIsolation:
    """Two concurrent streams with different audio stay fully isolated."""

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
        required_cache_size = CHUNK_SIZE * num_left_chunks

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

        offset_a = offset_b = 0
        with torch.no_grad():
            for xs_a, xs_b in zip(chunks_a, chunks_b):
                for ctx, xs, off in (
                    (ctx_a, xs_a, offset_a),
                    (ctx_b, xs_b, offset_b),
                ):
                    probs, new_att, new_cnn = model.forward_chunk(
                        xs, off, required_cache_size,
                        ctx.get_att_cache(), ctx.get_cnn_cache(),
                    )
                    ctx.commit_chunk(new_att[:, :, -CHUNK_SIZE:, :], new_cnn)
                    ctx.get_decoder().decode_chunk(probs)
                offset_a += CHUNK_SIZE
                offset_b += CHUNK_SIZE

        kv_b_before = ctx_b.get_att_cache().clone()
        ctx_a.free()
        torch.testing.assert_close(
            ctx_b.get_att_cache(), kv_b_before, rtol=0.0, atol=0.0
        )

        ctx_b.get_decoder().finalize_stream()
        ctx_b.free()

        assert pool.num_free_blocks == pool.num_total_blocks

        ref_a, _ = _reference_streaming(model, chunks_a, required_cache_size, dtype, device)
        ref_b, _ = _reference_streaming(model, chunks_b, required_cache_size, dtype, device)
        assert not torch.allclose(
            torch.cat(ref_a, dim=1), torch.cat(ref_b, dim=1)
        ), "Expected streams A and B to differ for different audio."


# ---------------------------------------------------------------------------
# Test 6: Full end-to-end pipeline on real audio
# ---------------------------------------------------------------------------

def _skip_if_no_audio(audio_path: str) -> None:
    """Call pytest.skip() when the audio file is not available."""
    if not audio_path or not Path(audio_path).exists():
        pytest.skip(
            "Audio file not found; pass --audio-path <wav> or set AUDIO_PATH"
        )


class TestStreamingWithRealAudio:
    """Full end-to-end pipeline on real audio: model forward → CTC decode → text.

    Requires:
    - ``--ckpt-dir`` / ``CKPT_DIR``: WeNet Conformer checkpoint directory
      (must contain ``final.pt`` and ``words.txt``).
    - ``--audio-path`` / ``AUDIO_PATH``: 16 kHz WAV file to transcribe.

    All tests are skipped when either resource is absent.

    Tests
    -----
    test_forward_and_ctc_decode
        Runs the paged streaming pipeline end-to-end and verifies that the
        CTC decoder produces a non-empty transcript.  Prints the top-beam
        text so the output is visible in ``pytest -s``.

    test_paged_logits_match_dense_reference
        Verifies that paged ``forward_chunk_paged`` produces bit-identical
        log-probabilities to the plain ``forward_chunk`` dense reference, for
        both unlimited history and a finite left-context window.  Also checks
        that the shared block pool is fully recovered after the stream is freed.

    test_cache_manager_pool_accounting
        Allocates two concurrent streams from the same block pool, runs each
        for several chunks, frees stream A, and confirms that stream B's
        cached KV tensors are unchanged.  Then frees stream B and verifies the
        pool is back to its initial free-block count.
    """

    # ------------------------------------------------------------------
    # Helper: load features and ensure audio is long enough
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Test 6a: forward pass + CTC decoding + text conversion
    # ------------------------------------------------------------------

    def test_forward_and_ctc_decode(
        self, model, ckpt_dir: str, audio_path: str, device
    ):
        """Paged streaming → CTC beam search → non-empty transcript."""
        _skip_if_no_audio(audio_path)

        dtype = torch.float16
        beam_size = 10
        chunks = self._load_chunks(audio_path, device, dtype)
        id2word = _build_id2word(ckpt_dir)

        # Run paged streaming path with CTC decoding.
        _, _, streaming_result, pool = _cache_manager_streaming_paged(
            model, chunks,
            num_left_chunks=-1,
            device=device,
            dtype=dtype,
            vocab_size=VOCAB_SIZE,
            beam_size=beam_size,
        )

        # Pool fully recovered after ctx.free().
        assert pool.num_free_blocks == pool.num_total_blocks, (
            f"Block pool leak: {pool.num_free_blocks}/{pool.num_total_blocks} free"
        )

        # CTC result structure.
        assert streaming_result.lengths.shape == torch.Size([1, beam_size])
        assert streaming_result.scores.shape == torch.Size([1, beam_size])

        # Beam scores must be non-increasing.
        scores = streaming_result.scores[0].cpu()
        assert (scores[:-1] >= scores[1:]).all(), (
            f"Beam scores not sorted: {scores.tolist()}"
        )

        # All token IDs must be within vocabulary range.
        for beam_idx, beam_tokens in enumerate(streaming_result.tokens[0]):
            bad = [t for t in beam_tokens if not (0 <= t < VOCAB_SIZE)]
            assert not bad, (
                f"Beam {beam_idx} contains out-of-range token IDs: {bad}"
            )

        # Convert top-1 beam to text and verify it is non-empty.
        top1_tokens = streaming_result.tokens[0][0]
        text = _tokens_to_text(top1_tokens, id2word)
        print(f"\n[paged streaming top-1]  {text!r}")
        assert len(text.strip()) > 0, "CTC decoder produced an empty transcript"

        # All beams should be decodable to strings (may be empty for lower beams).
        for beam_idx, beam_tokens in enumerate(streaming_result.tokens[0]):
            beam_text = _tokens_to_text(beam_tokens, id2word)
            print(f"  beam {beam_idx}: {beam_text!r}  (score {scores[beam_idx]:.3f})")

    # ------------------------------------------------------------------
    # Test 6b: paged logits == dense reference, pool recovered
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("num_left_chunks", [-1, 4])
    def test_paged_logits_match_dense_reference(
        self, model, audio_path: str, device, num_left_chunks: int
    ):
        """Paged path is bit-identical to the dense reference on real audio."""
        _skip_if_no_audio(audio_path)

        dtype = torch.float16
        chunks = self._load_chunks(audio_path, device, dtype)
        required_cache_size = CHUNK_SIZE * num_left_chunks

        ref_logits, _ = _reference_streaming(
            model, chunks, required_cache_size, dtype, device
        )
        paged_logits, _, _, pool = _cache_manager_streaming_paged(
            model, chunks, num_left_chunks, device, dtype
        )

        assert len(ref_logits) == len(paged_logits) == len(chunks)
        for step, (ref, paged) in enumerate(zip(ref_logits, paged_logits)):
            torch.testing.assert_close(
                paged, ref, rtol=0.0, atol=0.0,
                msg=f"Paged/dense logit mismatch at chunk {step} "
                    f"(num_left_chunks={num_left_chunks})",
            )

        assert pool.num_free_blocks == pool.num_total_blocks, (
            f"Block pool leak after free: {pool.num_free_blocks}/{pool.num_total_blocks}"
        )

    # ------------------------------------------------------------------
    # Test 6c: two concurrent streams stay isolated; pool fully recovered
    # ------------------------------------------------------------------

    def test_cache_manager_pool_accounting(
        self, model, audio_path: str, device
    ):
        """Two real-audio streams share a pool; freeing one leaves the other intact."""
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
        required_cache_size = CHUNK_SIZE * num_left_chunks

        # Use 3 chunks per stream so eviction is exercised.
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

        offset_a = offset_b = 0
        with torch.no_grad():
            for xs_a, xs_b in zip(chunks_a, chunks_b):
                for ctx, xs, off in ((ctx_a, xs_a, offset_a), (ctx_b, xs_b, offset_b)):
                    probs, new_att, new_cnn = model.forward_chunk(
                        xs, off, required_cache_size,
                        ctx.get_att_cache(), ctx.get_cnn_cache(),
                    )
                    ctx.commit_chunk(new_att[:, :, -CHUNK_SIZE:, :], new_cnn)
                    ctx.get_decoder().decode_chunk(probs)
                offset_a += CHUNK_SIZE
                offset_b += CHUNK_SIZE

        # Capture stream B's KV cache before freeing stream A.
        kv_b_before = ctx_b.get_att_cache().clone()

        # Free stream A; stream B must be unaffected.
        ctx_a.free()
        torch.testing.assert_close(
            ctx_b.get_att_cache(), kv_b_before, rtol=0.0, atol=0.0,
            msg="Stream B's KV cache changed after freeing stream A",
        )

        # Decode stream B and free it; pool must be fully recovered.
        ctx_b.get_decoder().finalize_stream()
        ctx_b.free()

        assert pool.num_free_blocks == initial_free, (
            f"Pool not fully recovered: {pool.num_free_blocks}/{initial_free} free blocks"
        )
