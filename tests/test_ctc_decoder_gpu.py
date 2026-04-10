#!/usr/bin/env python3
"""
Unit tests for GPU CTC prefix beam search decoder.
"""

import pytest
import torch

import oasr
from oasr.ctc_decode import (
    GpuDecoderConfig,
    GpuDecoderResult,
    GpuStreamingDecoder,
    ctc_beam_search_decode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logp_gpu(T: int, V: int, best_path: list, device="cuda") -> torch.Tensor:
    """Build a [1, T, V] log-prob tensor with clear winner tokens.

    At each timestep t, best_path[t] gets log-prob 0.0 (prob=1) and
    all others get -1e9 (essentially 0 prob).
    """
    assert len(best_path) == T
    logp = torch.full((1, T, V), -1e9, dtype=torch.float32, device=device)
    for t, tok in enumerate(best_path):
        logp[0, t, tok] = 0.0
    return logp


def _make_batched_logp_gpu(batch_paths: list, V: int, device="cuda"):
    """Build batched log-prob tensor from multiple paths (padded to max T).

    Returns: (logp [batch, max_T, V], seq_lengths [batch])
    """
    batch = len(batch_paths)
    max_T = max(len(p) for p in batch_paths)
    logp = torch.full((batch, max_T, V), -1e9, dtype=torch.float32, device=device)
    lengths = []
    for b, path in enumerate(batch_paths):
        T = len(path)
        lengths.append(T)
        for t, tok in enumerate(path):
            logp[b, t, tok] = 0.0
        # Fill padding with blank (token 0) as dominant
        for t in range(T, max_T):
            logp[b, t, 0] = 0.0
    seq_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    return logp, seq_lengths


# ---------------------------------------------------------------------------
# Offline decode tests
# ---------------------------------------------------------------------------

@pytest.mark.cuda
class TestCtcDecoderGpuOffline:
    """Tests for offline GPU CTC prefix beam search decode."""

    def test_basic_sequence(self, device):
        """Deterministic sequence with blanks decodes correctly."""
        V = 5
        # Path: tok1, blank, tok2, blank, tok3 → [1, 2, 3]
        logp = _make_logp_gpu(5, V, [1, 0, 2, 0, 3], device)
        seq_lengths = torch.tensor([5], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(logp, seq_lengths, beam_size=3,
                                        blank_id=0, blank_threshold=1.0,
                                        max_seq_len=10)

        assert isinstance(result, GpuDecoderResult)
        assert len(result.tokens) == 1       # batch=1
        assert len(result.tokens[0]) == 3    # beam=3
        # Best beam should contain [1, 2, 3]
        assert result.tokens[0][0] == [1, 2, 3]

    def test_all_blank(self, device):
        """All-blank input produces empty output when blank_threshold < 1.

        With blank_threshold=0.99, all frames where P(blank)=1.0 are skipped
        (blank_prob > threshold), so no frames are processed and output is empty.
        """
        V = 4
        logp = _make_logp_gpu(5, V, [0, 0, 0, 0, 0], device)
        seq_lengths = torch.tensor([5], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(logp, seq_lengths, beam_size=3,
                                        blank_id=0, blank_threshold=0.99,
                                        max_seq_len=10)

        # All frames are blank with prob=1.0, exceeding threshold=0.99,
        # so they are all filtered out, resulting in empty output
        assert result.tokens[0][0] == []

    def test_repeated_tokens_with_blank(self, device):
        """Repeated tokens separated by blank produce expected output."""
        V = 4
        # Use stronger signal: tok1, blank, tok1 → [1, 1]
        # Build soft log-probs so blank frames are recognizable
        logp = _make_logp_gpu(3, V, [1, 0, 1], device)
        seq_lengths = torch.tensor([3], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(logp, seq_lengths, beam_size=3,
                                        blank_id=0, blank_threshold=1.0,
                                        max_seq_len=10)

        best = result.tokens[0][0]
        # tok1, blank, tok1 → [1, 1] (blank separates the two 1s)
        assert best == [1, 1]

    def test_batched_decode(self, device):
        """Multiple utterances decoded in parallel."""
        V = 6
        paths = [
            [1, 0, 2, 0, 3],       # → [1, 2, 3]
            [4, 0, 5],             # → [4, 5]
            [1, 0, 1, 0, 1, 0, 2], # → [1, 1, 1, 2]
        ]
        logp, seq_lengths = _make_batched_logp_gpu(paths, V, device)

        result = ctc_beam_search_decode(logp, seq_lengths, beam_size=5,
                                        blank_id=0, blank_threshold=1.0,
                                        max_seq_len=20)

        assert len(result.tokens) == 3
        assert result.tokens[0][0] == [1, 2, 3]
        assert result.tokens[1][0] == [4, 5]
        assert result.tokens[2][0] == [1, 1, 1, 2]

    def test_variable_seq_lengths(self, device):
        """Batch with different sequence lengths handles padding correctly."""
        V = 5
        # Utterance 1: length 3, utterance 2: length 6
        paths = [
            [1, 0, 2],
            [3, 0, 4, 0, 1, 0],
        ]
        logp, seq_lengths = _make_batched_logp_gpu(paths, V, device)

        result = ctc_beam_search_decode(logp, seq_lengths, beam_size=3,
                                        blank_id=0, blank_threshold=1.0,
                                        max_seq_len=10)

        assert result.tokens[0][0] == [1, 2]
        assert result.tokens[1][0] == [3, 4, 1]

    def test_output_shapes(self, device):
        """Verify output tensor shapes."""
        V, beam = 10, 5
        logp = _make_logp_gpu(8, V, [1, 0, 2, 0, 3, 0, 4, 0], device)
        seq_lengths = torch.tensor([8], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(logp, seq_lengths, beam_size=beam,
                                        blank_id=0, max_seq_len=20)

        assert result.lengths.shape == (1, beam)
        assert result.scores.shape == (1, beam)
        # Scores should be sorted descending
        scores = result.scores[0].cpu()
        for i in range(len(scores) - 1):
            if scores[i + 1].item() > -1e20:  # Skip invalid beams
                assert scores[i].item() >= scores[i + 1].item()


# ---------------------------------------------------------------------------
# Streaming decode tests
# ---------------------------------------------------------------------------

@pytest.mark.cuda
class TestCtcDecoderGpuStreaming:
    """Tests for streaming GPU CTC prefix beam search decode."""

    def test_streaming_basic(self, device):
        """Basic streaming decode produces correct output."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)
        decoder.init_stream(batch=1, vocab_size=V, device=device)

        # Feed frames one at a time: [1, 0, 2, 0, 3]
        path = [1, 0, 2, 0, 3]
        for tok in path:
            frame = torch.full((1, 1, V), -1e9, dtype=torch.float32, device=device)
            frame[0, 0, tok] = 0.0
            decoder.decode_chunk(frame)

        result = decoder.finalize_stream()
        assert result.tokens[0][0] == [1, 2, 3]

    def test_streaming_multi_frame_chunks(self, device):
        """Streaming with multi-frame chunks."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)
        decoder.init_stream(batch=1, vocab_size=V, device=device)

        # Chunk 1: [1, 0]  Chunk 2: [2, 0, 3]
        logp1 = _make_logp_gpu(2, V, [1, 0], device)
        logp2 = _make_logp_gpu(3, V, [2, 0, 3], device)
        decoder.decode_chunk(logp1)
        decoder.decode_chunk(logp2)

        result = decoder.finalize_stream()
        assert result.tokens[0][0] == [1, 2, 3]

    def test_init_stream_resets(self, device):
        """Re-calling init_stream resets all state."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)

        # First decode
        decoder.init_stream(batch=1, vocab_size=V, device=device)
        logp = _make_logp_gpu(3, V, [1, 0, 2], device)
        decoder.decode_chunk(logp)
        result1 = decoder.finalize_stream()

        # Re-init and decode different input
        decoder.init_stream(batch=1, vocab_size=V, device=device)
        logp = _make_logp_gpu(3, V, [3, 0, 4], device)
        decoder.decode_chunk(logp)
        result2 = decoder.finalize_stream()

        assert result1.tokens[0][0] == [1, 2]
        assert result2.tokens[0][0] == [3, 4]

    def test_streaming_step_counter(self, device):
        """Step counter increments correctly (blank-dominant frames are skipped)."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)
        decoder.init_stream(batch=1, vocab_size=V, device=device)

        assert decoder.step == 0

        logp = _make_logp_gpu(3, V, [1, 2, 3], device)
        decoder.decode_chunk(logp)
        assert decoder.step == 3

    def test_finalize_without_init_raises(self, device):
        """Calling finalize without init raises RuntimeError."""
        decoder = GpuStreamingDecoder()
        with pytest.raises(RuntimeError, match="init_stream"):
            decoder.finalize_stream()

    def test_decode_chunk_without_init_raises(self, device):
        """Calling decode_chunk without init raises RuntimeError."""
        V = 5
        decoder = GpuStreamingDecoder()
        logp = _make_logp_gpu(3, V, [1, 0, 2], device)
        with pytest.raises(RuntimeError, match="init_stream"):
            decoder.decode_chunk(logp)


# ---------------------------------------------------------------------------
# Workspace size tests
# ---------------------------------------------------------------------------

@pytest.mark.cuda
class TestGpuDecoderWorkspace:
    """Tests for workspace and state buffer sizing."""

    def test_workspace_size_positive(self, device):
        """Workspace size is positive."""
        mod = oasr.ctc_decode._get_ctc_decoder_module()
        size = mod.ctc_decoder_workspace_size(1, 10, 5000, 200)
        assert size > 0

    def test_state_size_positive(self, device):
        """State buffer size is positive and larger than workspace."""
        mod = oasr.ctc_decode._get_ctc_decoder_module()
        ws = mod.ctc_decoder_workspace_size(1, 10, 5000, 200)
        state = mod.ctc_decoder_state_size(1, 10, 5000, 200)
        assert state > 0
        assert state > ws  # State includes header + workspace

    def test_workspace_scales_with_batch(self, device):
        """Larger batch produces proportionally larger workspace."""
        mod = oasr.ctc_decode._get_ctc_decoder_module()
        size1 = mod.ctc_decoder_workspace_size(1, 10, 5000, 200)
        size4 = mod.ctc_decoder_workspace_size(4, 10, 5000, 200)
        assert size4 > size1

    def test_workspace_scales_with_beam(self, device):
        """Larger beam produces larger workspace."""
        mod = oasr.ctc_decode._get_ctc_decoder_module()
        size5 = mod.ctc_decoder_workspace_size(1, 5, 5000, 200)
        size20 = mod.ctc_decoder_workspace_size(1, 20, 5000, 200)
        assert size20 > size5


# ---------------------------------------------------------------------------
# High-level API integration via oasr namespace
# ---------------------------------------------------------------------------

@pytest.mark.cuda
class TestCtcDecoderApiExport:
    """Test that GPU decoder is accessible via oasr namespace."""

    def test_accessible_from_oasr(self):
        """Functions exported from oasr namespace."""
        assert hasattr(oasr, "ctc_beam_search_decode")
        assert hasattr(oasr, "GpuStreamingDecoder")
        assert hasattr(oasr, "GpuDecoderConfig")
        assert hasattr(oasr, "GpuDecoderResult")


# ---------------------------------------------------------------------------
# Paged memory tests
# ---------------------------------------------------------------------------

@pytest.mark.cuda
class TestCtcDecoderPagedOffline:
    """Tests for paged-memory GPU CTC prefix beam search (offline)."""

    def test_paged_matches_flat_basic(self, device):
        """Paged mode produces token-exact output matching flat mode."""
        V = 5
        logp = _make_logp_gpu(5, V, [1, 0, 2, 0, 3], device)
        seq_lengths = torch.tensor([5], dtype=torch.int32, device=device)

        result_flat = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=1.0, max_seq_len=10)
        result_paged = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=1.0, max_seq_len=10,
            use_paged_memory=True, page_size=4)

        assert result_paged.tokens[0][0] == [1, 2, 3]
        assert result_paged.tokens == result_flat.tokens

    def test_paged_matches_flat_batched(self, device):
        """Paged mode handles batched inputs identically to flat mode."""
        V = 6
        paths = [
            [1, 0, 2, 0, 3],
            [4, 0, 5],
            [1, 0, 1, 0, 1, 0, 2],
        ]
        logp, seq_lengths = _make_batched_logp_gpu(paths, V, device)

        result_flat = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=5, blank_id=0, blank_threshold=1.0, max_seq_len=20)
        result_paged = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=5, blank_id=0, blank_threshold=1.0, max_seq_len=20,
            use_paged_memory=True, page_size=4)

        assert result_paged.tokens == result_flat.tokens

    def test_paged_workspace_smaller_than_flat(self, device):
        """Paged workspace is smaller than flat for large max_seq_len."""
        mod = oasr.ctc_decode._get_ctc_decoder_module()
        batch, beam, vocab, max_seq = 4, 16, 5000, 1024
        flat_size = mod.ctc_decoder_workspace_size(batch, beam, vocab, max_seq)
        paged_size = mod.ctc_decoder_paged_workspace_size(batch, beam, vocab, max_seq, 16)
        assert paged_size < flat_size

    def test_paged_workspace_size_positive(self, device):
        """Paged workspace size is positive."""
        mod = oasr.ctc_decode._get_ctc_decoder_module()
        size = mod.ctc_decoder_paged_workspace_size(1, 10, 5000, 200, 16)
        assert size > 0

    def test_paged_page_size_16(self, device):
        """Default page_size=16 produces correct output."""
        V = 8
        logp = _make_logp_gpu(7, V, [1, 0, 2, 0, 3, 0, 4], device)
        seq_lengths = torch.tensor([7], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=5, blank_id=0, blank_threshold=1.0, max_seq_len=10,
            use_paged_memory=True, page_size=16)

        assert result.tokens[0][0] == [1, 2, 3, 4]

    def test_paged_long_sequence_multiple_pages(self, device):
        """Sequence longer than one page exercises multi-page access."""
        V = 6
        # 20 non-blank tokens separated by blanks → each token on a new page
        # (page_size=4 means page boundary every 4 tokens)
        tokens = [t for tok in range(1, 21) for t in [tok % V or 1, 0]]
        T = len(tokens)
        path_tokens = [t for t in tokens]
        logp = _make_logp_gpu(T, V, path_tokens, device)
        seq_lengths = torch.tensor([T], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=1, blank_id=0, blank_threshold=1.0, max_seq_len=40,
            use_paged_memory=True, page_size=4)

        # Result should be non-empty and have a positive score
        assert len(result.tokens[0][0]) > 0
        assert result.scores[0, 0].item() > -1e8


@pytest.mark.cuda
class TestCtcDecoderPagedStreaming:
    """Tests for paged-memory GPU CTC prefix beam search (streaming)."""

    def test_paged_streaming_matches_flat(self, device):
        """Paged streaming produces the same output as flat streaming."""
        V = 5
        path = [1, 0, 2, 0, 3]

        def _run(use_paged):
            config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10,
                                      use_paged_memory=use_paged, page_size=4)
            decoder = GpuStreamingDecoder(config)
            decoder.init_stream(batch=1, vocab_size=V, device=device)
            for tok in path:
                frame = torch.full((1, 1, V), -1e9, dtype=torch.float32, device=device)
                frame[0, 0, tok] = 0.0
                decoder.decode_chunk(frame)
            return decoder.finalize_stream()

        result_flat = _run(use_paged=False)
        result_paged = _run(use_paged=True)

        assert result_paged.tokens[0][0] == [1, 2, 3]
        assert result_paged.tokens == result_flat.tokens

    def test_paged_streaming_state_size_positive(self, device):
        """Paged state buffer size is positive."""
        mod = oasr.ctc_decode._get_ctc_decoder_module()
        size = mod.ctc_decoder_paged_state_size(1, 10, 5000, 200, 16)
        assert size > 0

    def test_paged_streaming_reinit(self, device):
        """Re-calling init_stream in paged mode resets state."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10,
                                  use_paged_memory=True, page_size=4)
        decoder = GpuStreamingDecoder(config)

        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [1, 0, 2], device))
        result1 = decoder.finalize_stream()

        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [3, 0, 4], device))
        result2 = decoder.finalize_stream()

        assert result1.tokens[0][0] == [1, 2]
        assert result2.tokens[0][0] == [3, 4]
