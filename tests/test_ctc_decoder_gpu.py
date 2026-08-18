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
    StreamHandle,
    StreamState,
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

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=1.0, max_seq_len=10
        )

        assert isinstance(result, GpuDecoderResult)
        assert len(result.tokens) == 1  # batch=1
        assert len(result.tokens[0]) == 3  # beam=3
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

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=0.99, max_seq_len=10
        )

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

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=1.0, max_seq_len=10
        )

        best = result.tokens[0][0]
        # tok1, blank, tok1 → [1, 1] (blank separates the two 1s)
        assert best == [1, 1]

    def test_repeat_after_skipped_blank_collapses(self, device):
        """Regression for GPU-DEC-1: a repeat *following* a skipped blank
        must still collapse.

        Path ``tok1, blank, tok1, tok1`` with ``blank_threshold < 1`` skips the
        (P=1.0) blank frame, so the step that emits the second ``tok1`` carries
        ``need_add_blank`` (a blank was skipped before it).  The previous kernel
        relabelled that freshly emitted non-blank token as "ends in blank", so
        the third (consecutive) ``tok1`` frame extended instead of collapsing,
        yielding ``[1, 1, 1]``.  CTC ground truth is ``1,blank,1,1 → [1, 1]``
        (the last two 1s collapse, the blank separates the first pair).
        """
        V = 4
        logp = _make_logp_gpu(4, V, [1, 0, 1, 1], device)
        seq_lengths = torch.tensor([4], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=0.98, max_seq_len=10
        )
        assert result.tokens[0][0] == [1, 1]

        # Paged path goes through the parallel topk_phase2_paged_kernel.
        paged = ctc_beam_search_decode(
            logp,
            seq_lengths,
            beam_size=3,
            blank_id=0,
            blank_threshold=0.98,
            max_seq_len=10,
            use_paged_memory=True,
        )
        assert paged.tokens[0][0] == [1, 1]

    def test_repeat_after_leading_skipped_blank_collapses(self, device):
        """Regression for GPU-DEC-1 (first-step variant): a repeat right after
        leading skipped blanks must collapse.

        Path ``blank, tok1, tok1`` with ``blank_threshold < 1`` skips the
        leading blank, so ``first_step`` initialises the beam at ``first_t=1``.
        The previous kernel put that initial non-blank token in the *blank*
        slot when ``first_t > 0``, so the next (consecutive) ``tok1`` extended
        to ``[1, 1]``.  CTC ground truth is ``blank,1,1 → [1]``.
        """
        V = 4
        logp = _make_logp_gpu(3, V, [0, 1, 1], device)
        seq_lengths = torch.tensor([3], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=0.98, max_seq_len=10
        )
        assert result.tokens[0][0] == [1]

    def test_batched_decode(self, device):
        """Multiple utterances decoded in parallel."""
        V = 6
        paths = [
            [1, 0, 2, 0, 3],  # → [1, 2, 3]
            [4, 0, 5],  # → [4, 5]
            [1, 0, 1, 0, 1, 0, 2],  # → [1, 1, 1, 2]
        ]
        logp, seq_lengths = _make_batched_logp_gpu(paths, V, device)

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=5, blank_id=0, blank_threshold=1.0, max_seq_len=20
        )

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

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=1.0, max_seq_len=10
        )

        assert result.tokens[0][0] == [1, 2]
        assert result.tokens[1][0] == [3, 4, 1]

    def test_output_shapes(self, device):
        """Verify output tensor shapes."""
        V, beam = 10, 5
        logp = _make_logp_gpu(8, V, [1, 0, 2, 0, 3, 0, 4, 0], device)
        seq_lengths = torch.tensor([8], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(
            logp, seq_lengths, beam_size=beam, blank_id=0, max_seq_len=20
        )

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

    def test_streaming_repeat_after_skipped_blank_collapses(self, device):
        """Regression for GPU-DEC-1 on the streaming path.

        Same scenario as the offline test: ``tok1, blank, tok1, tok1`` with
        ``blank_threshold < 1`` skips the blank frame; the repeat following the
        skip must collapse to ``[1, 1]`` rather than extend to ``[1, 1, 1]``.
        Exercised via ``decode_chunk`` (shared topk_phase2/first_step kernels),
        with CUDA graphs off and on to cover both chunk-launcher paths.
        """
        V = 4
        for use_graphs in (False, True):
            config = GpuDecoderConfig(beam_size=3, blank_id=0, blank_threshold=0.98, max_seq_len=10)
            decoder = GpuStreamingDecoder(config, use_cuda_graphs=use_graphs)
            decoder.init_stream(batch=1, vocab_size=V, device=device)
            decoder.decode_chunk(_make_logp_gpu(4, V, [1, 0, 1, 1], device))
            assert decoder.finalize_stream().tokens[0][0] == [1, 1], f"use_cuda_graphs={use_graphs}"

    def test_streaming_decodes_more_frames_than_output_cap(self, device):
        """Regression for GPU-DEC-2: streaming decodes more *frames* than the
        output-token cap (``max_seq_len``) instead of truncating.

        ``step`` counts decoded frames, not output tokens.  Here many
        blank-dominant (but not skipped, at ``blank_threshold=1.0``) frames push
        ``step`` past ``max_seq_len`` before the two real tokens at the end.  The
        pre-fix chunk loop capped ``step`` at ``max_seq_len`` and dropped the
        trailing tokens; ``select_seqs`` is now a ring of width ``max_seq_len``
        and the frame cap is gone, so the tail is decoded.  Output length stays
        bounded by ``max_seq_len`` via the in-kernel clen cap.
        """
        V, msl, n_pad = 4, 8, 12
        T = n_pad + 2
        logp = torch.full((1, T, V), -1.0, dtype=torch.float32, device=device)
        logp[0, :n_pad, 0] = 5.0  # P(blank) ~ 0.99 < 1.0 → decoded, no token
        logp[0, n_pad, 1] = 5.0  # token 1
        logp[0, n_pad + 1, 2] = 5.0  # token 2
        for use_graphs in (False, True):
            dec = GpuStreamingDecoder(
                GpuDecoderConfig(beam_size=4, blank_id=0, blank_threshold=1.0, max_seq_len=msl),
                use_cuda_graphs=use_graphs,
            )
            dec.init_stream(batch=1, vocab_size=V, device=device)
            dec.decode_chunk(logp)
            assert dec.step > msl, f"step={dec.step} should exceed cap {msl}"
            assert dec.finalize_stream().tokens[0][0] == [1, 2], f"use_cuda_graphs={use_graphs}"

    def test_streaming_graph_multiframe_chunk_with_skips(self, device):
        """Regression for the captured-graph multi-frame chunk counter race.

        A single multi-frame chunk containing many non-blank frames interleaved
        with skipped (P=1) blanks must decode identically under CUDA graphs and
        eager.  The captured graphs previously pushed ``(step, frame_idx)``
        through one reused pinned host buffer that the launcher overwrote for
        the next frame before the GPU consumed it, so multi-frame graph chunks
        saw stale counters and duplicated tokens.  Counters are now set by a
        by-value kernel on the stream before each launch.
        """
        V = 6
        # 6 groups of ``tok, blank, tok, tok`` (blank separates, then a repeat
        # collapses) → each group contributes ``tok, tok``.
        toks = [1, 2, 3, 4, 5, 1]
        path = []
        for k in toks:
            path += [k, 0, k, k]
        expected = []
        for k in toks:
            expected += [k, k]
        logp = _make_logp_gpu(len(path), V, path, device)

        def run(graphs):
            dec = GpuStreamingDecoder(
                GpuDecoderConfig(beam_size=4, blank_id=0, blank_threshold=0.98, max_seq_len=64),
                use_cuda_graphs=graphs,
            )
            dec.init_stream(batch=1, vocab_size=V, device=device)
            dec.decode_chunk(logp)  # whole sequence in ONE multi-frame chunk
            return dec.finalize_stream().tokens[0][0]

        eager = run(False)
        graph = run(True)
        assert eager == expected, f"eager={eager} expected={expected}"
        assert graph == eager, f"graph={graph} eager={eager}"

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

    def test_reuse_with_varying_batch(self, device):
        """Single decoder reused across requests with different batch sizes."""
        V = 6
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)

        # Request 1: batch=1
        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [1, 0, 2], device))
        r1 = decoder.finalize_stream()
        assert r1.tokens[0][0] == [1, 2]

        # Request 2: batch=2 (larger — triggers buffer growth)
        paths = [[3, 0, 4], [1, 0, 1]]
        logp, _ = _make_batched_logp_gpu(paths, V, device)
        decoder.init_stream(batch=2, vocab_size=V, device=device)
        decoder.decode_chunk(logp)
        r2 = decoder.finalize_stream()
        assert r2.tokens[0][0] == [3, 4]
        assert r2.tokens[1][0] == [1, 1]

        # Request 3: batch=1 again (reuses the larger buffer)
        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [4, 0, 3], device))
        r3 = decoder.finalize_stream()
        assert r3.tokens[0][0] == [4, 3]

    def test_reuse_with_varying_chunk_count(self, device):
        """Single decoder reused with different numbers of chunks."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=20)
        decoder = GpuStreamingDecoder(config)

        # Short request: 1 chunk
        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [1, 0, 2], device))
        assert decoder.finalize_stream().tokens[0][0] == [1, 2]

        # Long request: 3 chunks
        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(2, V, [1, 0], device))
        decoder.decode_chunk(_make_logp_gpu(2, V, [2, 0], device))
        decoder.decode_chunk(_make_logp_gpu(1, V, [3], device))
        assert decoder.finalize_stream().tokens[0][0] == [1, 2, 3]

    def test_buffer_reuse_no_realloc(self, device):
        """Verifies that init_stream reuses the same buffer when size matches."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)

        decoder.init_stream(batch=1, vocab_size=V, device=device)
        buf_ptr_1 = decoder._state.buffer.data_ptr()

        decoder.init_stream(batch=1, vocab_size=V, device=device)
        buf_ptr_2 = decoder._state.buffer.data_ptr()

        assert buf_ptr_1 == buf_ptr_2, "Buffer should be reused for identical params"

    def test_empty_chunk(self, device):
        """Zero-length chunk is a no-op."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)
        decoder.init_stream(batch=1, vocab_size=V, device=device)

        empty = torch.empty(1, 0, V, dtype=torch.float32, device=device)
        decoder.decode_chunk(empty)
        assert decoder.step == 0

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
# Interleaved / explicit StreamState tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCtcDecoderInterleaved:
    """Tests for interleaved multi-request decoding via explicit StreamState."""

    def test_create_state(self, device):
        """create_state returns a StreamState with correct attributes."""
        V = 5
        decoder = GpuStreamingDecoder(GpuDecoderConfig(beam_size=3, max_seq_len=10))
        state = decoder.create_state(batch=1, vocab_size=V, device=device)

        assert isinstance(state, StreamState)
        assert state.step == 0
        assert state.batch == 1
        assert state.vocab_size == V
        assert state.buffer.device.type == "cuda"

    def test_interleaved_two_requests(self, device):
        """Two requests interleaved on the same decoder produce correct output."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)

        s1 = decoder.create_state(batch=1, vocab_size=V, device=device)
        s2 = decoder.create_state(batch=1, vocab_size=V, device=device)

        # Interleave: s1 gets [1, 0, 2], s2 gets [3, 0, 4]
        decoder.decode_chunk(_make_logp_gpu(1, V, [1], device), state=s1)
        decoder.decode_chunk(_make_logp_gpu(1, V, [3], device), state=s2)
        decoder.decode_chunk(_make_logp_gpu(1, V, [0], device), state=s1)
        decoder.decode_chunk(_make_logp_gpu(1, V, [0], device), state=s2)
        decoder.decode_chunk(_make_logp_gpu(1, V, [2], device), state=s1)
        decoder.decode_chunk(_make_logp_gpu(1, V, [4], device), state=s2)

        r1 = decoder.finalize_stream(state=s1)
        r2 = decoder.finalize_stream(state=s2)

        assert r1.tokens[0][0] == [1, 2]
        assert r2.tokens[0][0] == [3, 4]

    def test_interleaved_different_lengths(self, device):
        """Interleaved requests with different frame counts."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=20)
        decoder = GpuStreamingDecoder(config)

        s_short = decoder.create_state(batch=1, vocab_size=V, device=device)
        s_long = decoder.create_state(batch=1, vocab_size=V, device=device)

        decoder.decode_chunk(_make_logp_gpu(3, V, [1, 0, 2], device), state=s_short)
        decoder.decode_chunk(_make_logp_gpu(2, V, [3, 0], device), state=s_long)

        r_short = decoder.finalize_stream(state=s_short)
        assert r_short.tokens[0][0] == [1, 2]

        decoder.decode_chunk(_make_logp_gpu(3, V, [4, 0, 1], device), state=s_long)
        r_long = decoder.finalize_stream(state=s_long)
        assert r_long.tokens[0][0] == [3, 4, 1]

    def test_reset_state_reuses_buffer(self, device):
        """reset_state reinitializes without reallocating if size fits."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)

        state = decoder.create_state(batch=1, vocab_size=V, device=device)
        buf_ptr = state.buffer.data_ptr()

        decoder.decode_chunk(_make_logp_gpu(3, V, [1, 0, 2], device), state=state)
        # step counts non-blank frames emitted; [1, 0, 2] yields 2 emit steps.
        assert state.step == 2

        decoder.reset_state(state, batch=1, vocab_size=V)
        assert state.step == 0
        assert state.buffer.data_ptr() == buf_ptr, "Buffer should be reused"

        decoder.decode_chunk(_make_logp_gpu(3, V, [3, 0, 4], device), state=state)
        r = decoder.finalize_stream(state=state)
        assert r.tokens[0][0] == [3, 4]

    def test_reset_state_grows_for_larger_batch(self, device):
        """reset_state allocates a larger buffer when batch grows."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)

        state = decoder.create_state(batch=1, vocab_size=V, device=device)
        small_bytes = state._buffer_bytes

        decoder.reset_state(state, batch=4, vocab_size=V)
        assert state._buffer_bytes >= small_bytes
        assert state.batch == 4

    def test_stream_handle(self, device):
        """StreamHandle wraps decoder + state with the standard interface."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)
        state = decoder.create_state(batch=1, vocab_size=V, device=device)

        handle = StreamHandle(decoder, state)
        assert handle.step == 0
        assert handle.config is config

        handle.decode_chunk(_make_logp_gpu(3, V, [1, 0, 2], device))
        # [1, 0, 2] emits two non-blank steps.
        assert handle.step == 2
        r = handle.finalize_stream()
        assert r.tokens[0][0] == [1, 2]

    def test_explicit_state_does_not_affect_internal(self, device):
        """Using explicit state does not touch the internal default state."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)

        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [1, 0, 2], device))

        state = decoder.create_state(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [3, 0, 4], device), state=state)

        # Each sequence has one blank in the middle, so 2 emit steps.
        assert decoder.step == 2
        assert state.step == 2

        r_internal = decoder.finalize_stream()
        r_explicit = decoder.finalize_stream(state=state)
        assert r_internal.tokens[0][0] == [1, 2]
        assert r_explicit.tokens[0][0] == [3, 4]


# ---------------------------------------------------------------------------
# Overlapped (non-blocking) interim read-back
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCtcDecoderAsyncPeek:
    """``peek_states_async`` / ``peek_states_collect`` — the read-back behind
    ``EngineConfig.overlap_partial_readback``.

    It is off by default, which is exactly why it needs a test: the launcher
    grew an ``out_times`` argument for word timings and this call site kept its
    old 11-argument shape, so every stream under the overlapped read-back died
    with a ``TypeError`` and finalised with an empty transcript — a
    default-off path failing silently in the one mode that selects it.
    """

    def test_async_peek_matches_blocking_peek(self, device):
        """The overlapped read-back returns the blocking one's hypotheses."""
        V = 5
        config = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10)
        decoder = GpuStreamingDecoder(config)

        states = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in range(3)]
        paths = [[1, 0, 2], [3, 0, 4], [2, 2, 0]]
        for state, path in zip(states, paths):
            decoder.decode_chunk(_make_logp_gpu(len(path), V, path, device), state=state)

        blocking = decoder.peek_states(states)
        handle = decoder.peek_states_async(states)
        overlapped = decoder.peek_states_collect(handle)

        assert len(overlapped) == len(blocking)
        for over, block in zip(overlapped, blocking):
            assert over.tokens == block.tokens
        assert [r.tokens[0][0] for r in overlapped] == [[1, 2], [3, 4], [2]]

    def test_async_peek_keeps_advancing_across_steps(self, device):
        """Issue at step N, collect at step N+1 — the engine's actual cadence."""
        V = 5
        decoder = GpuStreamingDecoder(GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10))
        state = decoder.create_state(batch=1, vocab_size=V, device=device)

        decoder.decode_chunk(_make_logp_gpu(2, V, [1, 0], device), state=state)
        handle = decoder.peek_states_async([state])
        # A whole further step runs before the partial is materialised: the
        # collected hypothesis is the *previous* step's, one chunk behind.
        decoder.decode_chunk(_make_logp_gpu(2, V, [2, 0], device), state=state)
        lagged = decoder.peek_states_collect(handle)

        assert lagged[0].tokens[0][0] == [1]
        assert decoder.finalize_stream(state=state).tokens[0][0] == [1, 2]

    def test_async_peek_empty_set_is_none(self, device):
        """No ready streams: no handle, no collected partials."""
        decoder = GpuStreamingDecoder(GpuDecoderConfig(beam_size=3, max_seq_len=10))
        assert decoder.peek_states_async([]) is None
        assert decoder.peek_states_collect(None) == []


# ---------------------------------------------------------------------------
# Batched read-back
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCtcDecoderBatchedReadBack:
    """``ctc_beam_search_read_state_batched`` — one launch for the whole ready set.

    The per-state launcher costs 3-4 ``cudaMemcpy2DAsync`` (flat) or one kernel
    (paged) *each*, which at streaming pool sizes is the largest source of tiny
    GPU operations in the step.  The batched form reaches state ``i`` at a
    constant byte delta off state 0, so what has to hold is that every stream
    still reads *its own* buffers: an off-by-one in the delta arithmetic returns
    a neighbouring stream's transcript, which is plausible output rather than a
    crash.  Hence the comparison against the per-state path rather than against
    expected tokens.
    """

    @staticmethod
    def _decoded_states(decoder, paths, V, device):
        states = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in paths]
        for state, path in zip(states, paths):
            decoder.decode_chunk(_make_logp_gpu(len(path), V, path, device), state=state)
        return states

    @pytest.mark.parametrize("use_paged", [False, True], ids=["flat", "paged"])
    @pytest.mark.parametrize("want_times", [False, True], ids=["tokens", "tokens+times"])
    # 65 and 130 straddle the 64-stream group the launcher batches in, which is
    # where a per-group base pointer would be reused for the wrong group.
    @pytest.mark.parametrize("n_states", [1, 3, 64, 65, 130])
    def test_matches_per_state_read(self, device, use_paged, want_times, n_states):
        V = 7
        decoder = GpuStreamingDecoder(
            GpuDecoderConfig(
                beam_size=3, blank_id=0, max_seq_len=16, use_paged_memory=use_paged, page_size=16
            )
        )
        # Distinct per stream, so a stream reading a neighbour's state is visible.
        paths = [[1 + (i % (V - 1)), 0, 1 + ((i + 1) % (V - 1))] for i in range(n_states)]
        states = self._decoded_states(decoder, paths, V, device)

        batched = decoder.peek_states(states, want_times=want_times)
        per_state = [decoder.peek_state(state=s, want_times=want_times) for s in states]

        assert len(batched) == n_states
        for i, (got, want) in enumerate(zip(batched, per_state)):
            assert got.tokens[0] == want.tokens[0], f"stream {i} tokens"
            assert torch.equal(got.lengths.cpu(), want.lengths.cpu()), f"stream {i} lengths"
            assert torch.equal(got.scores.cpu(), want.scores.cpu()), f"stream {i} scores"
            if want_times:
                assert list(got.times[0][0]) == list(want.times[0][0]), f"stream {i} times"

    def test_streams_at_different_steps(self, device):
        """Per-state ``step`` picks the parity, so a shared one reads stale beams.

        The beam state is double-buffered and the live half is ``(step - 1) % 2``.
        Streams in one ready set are at different steps — that is the normal
        condition, not an edge case — so the depths below are unequal *and* the
        tokens differ per frame: a path of one repeated token writes the same
        thing into both parities, which would let a shared-step read pass.
        """
        V = 5
        decoder = GpuStreamingDecoder(GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=16))
        states = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in range(4)]
        # Depths 1..4, and every emitted token distinct from its neighbour so the
        # two parity buffers hold genuinely different sequences.
        for depth, state in enumerate(states, start=1):
            path = [1 + (t % (V - 1)) for t in range(depth)]
            decoder.decode_chunk(_make_logp_gpu(depth, V, path, device), state=state)

        batched = decoder.peek_states(states)
        per_state = [decoder.peek_state(state=s) for s in states]
        assert [s.step for s in states] == [1, 2, 3, 4]
        assert [r.tokens[0][0] for r in batched] == [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
        assert [r.tokens[0][0] for r in batched] == [r.tokens[0][0] for r in per_state]


# ---------------------------------------------------------------------------
# Blank-mask staging
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCtcDecoderBlankMask:
    """The auto-computed blank mask must decode exactly like an explicit one.

    ``decode_chunk_batch`` reads the mask back through reused page-locked
    staging rather than a fresh pageable ``.cpu()``.  The mask decides which
    frames are *decoded at all*, so a stale or mis-strided read is a silent
    transcript change — and the buffers are grow-only and reused across steps,
    which is precisely where a stale read would come from.
    """

    def test_auto_mask_matches_explicit_mask(self, device):
        import math

        V = 6
        thresh = 0.98
        decoder = GpuStreamingDecoder(
            GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=32, blank_threshold=thresh)
        )
        auto = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in range(5)]
        explicit = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in range(5)]

        torch.manual_seed(7)
        for _ in range(3):
            log_probs = torch.randn(5, 6, V, device=device).log_softmax(-1)
            decoder.decode_chunk_batch(log_probs, auto)
            mask = log_probs[:, :, 0].lt(math.log(thresh)).to(torch.uint8).cpu().contiguous()
            decoder.decode_chunk_batch(log_probs, explicit, is_speech_mask=mask)

        assert [s.step for s in auto] == [s.step for s in explicit]
        got = decoder.peek_states(auto)
        want = decoder.peek_states(explicit)
        assert [r.tokens[0][0] for r in got] == [r.tokens[0][0] for r in want]

    def test_shrinking_then_growing_ready_set(self, device):
        """Grow-only staging: a narrower step must not read the wider one's rows."""
        V = 6
        decoder = GpuStreamingDecoder(
            GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=32, blank_threshold=0.98)
        )
        pool = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in range(8)]
        torch.manual_seed(11)
        for n_ready in (8, 2, 5, 8):
            log_probs = torch.randn(n_ready, 4, V, device=device).log_softmax(-1)
            ready = pool[:n_ready]
            before = [s.step for s in ready]
            decoder.decode_chunk_batch(log_probs, ready)
            # Every frame either decoded or skipped as blank — never more.
            assert all(0 <= s.step - b <= 4 for s, b in zip(ready, before))
        assert all(s.step > 0 for s in pool[:2])


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
            logp, seq_lengths, beam_size=3, blank_id=0, blank_threshold=1.0, max_seq_len=10
        )
        result_paged = ctc_beam_search_decode(
            logp,
            seq_lengths,
            beam_size=3,
            blank_id=0,
            blank_threshold=1.0,
            max_seq_len=10,
            use_paged_memory=True,
            page_size=4,
        )

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
            logp, seq_lengths, beam_size=5, blank_id=0, blank_threshold=1.0, max_seq_len=20
        )
        result_paged = ctc_beam_search_decode(
            logp,
            seq_lengths,
            beam_size=5,
            blank_id=0,
            blank_threshold=1.0,
            max_seq_len=20,
            use_paged_memory=True,
            page_size=4,
        )

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
            logp,
            seq_lengths,
            beam_size=5,
            blank_id=0,
            blank_threshold=1.0,
            max_seq_len=10,
            use_paged_memory=True,
            page_size=16,
        )

        assert result.tokens[0][0] == [1, 2, 3, 4]

    def test_paged_long_sequence_multiple_pages(self, device):
        """Sequence longer than one page exercises multi-page access."""
        V = 6
        # 20 non-blank tokens separated by blanks → each token on a new page
        # (page_size=4 means page boundary every 4 tokens)
        tokens = [t for tok in range(1, 21) for t in [tok % V or 1, 0]]
        T = len(tokens)
        path_tokens = list(tokens)
        logp = _make_logp_gpu(T, V, path_tokens, device)
        seq_lengths = torch.tensor([T], dtype=torch.int32, device=device)

        result = ctc_beam_search_decode(
            logp,
            seq_lengths,
            beam_size=1,
            blank_id=0,
            blank_threshold=1.0,
            max_seq_len=40,
            use_paged_memory=True,
            page_size=4,
        )

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
            config = GpuDecoderConfig(
                beam_size=3, blank_id=0, max_seq_len=10, use_paged_memory=use_paged, page_size=4
            )
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
        config = GpuDecoderConfig(
            beam_size=3, blank_id=0, max_seq_len=10, use_paged_memory=True, page_size=4
        )
        decoder = GpuStreamingDecoder(config)

        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [1, 0, 2], device))
        result1 = decoder.finalize_stream()

        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, V, [3, 0, 4], device))
        result2 = decoder.finalize_stream()

        assert result1.tokens[0][0] == [1, 2]
        assert result2.tokens[0][0] == [3, 4]


# ---------------------------------------------------------------------------
# Step 4: per-state captured CUDA Graph parity tests
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCtcStreamingCudaGraphParity:
    """Bit-exact decode parity between the captured-graph and eager paths."""

    def _decode_path(
        self,
        V: int,
        path: list,
        *,
        use_cuda_graphs: bool,
        beam: int = 3,
        max_seq_len: int = 32,
        blank_threshold: float = 0.0,
        device=torch.device("cuda"),
    ):
        """Run a synthetic CTC decode over ``path`` with the given graph mode."""
        cfg = GpuDecoderConfig(
            beam_size=beam, blank_id=0, max_seq_len=max_seq_len, blank_threshold=blank_threshold
        )
        decoder = GpuStreamingDecoder(cfg, use_cuda_graphs=use_cuda_graphs)
        decoder.init_stream(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(len(path), V, path, device))
        return decoder.finalize_stream()

    @pytest.mark.parametrize(
        "path",
        [
            [1, 0, 2, 0, 3],
            [0, 0, 1, 0, 0, 2, 3, 0],
            [1, 1, 0, 2, 2, 0, 3],
            [0, 0, 0, 0, 0],  # all-blank chunk
            [1, 2, 3, 4],  # no blanks at all
            [1] * 20,  # repeated non-blank
        ],
    )
    def test_streaming_decode_matches_eager_graph_capture(self, path, device):
        """Decoded tokens / lengths / scores match eager path bit-exactly."""
        V = 6
        r_eager = self._decode_path(
            V, path, use_cuda_graphs=False, blank_threshold=0.0, device=device
        )
        r_graph = self._decode_path(
            V, path, use_cuda_graphs=True, blank_threshold=0.0, device=device
        )
        assert (
            r_eager.tokens == r_graph.tokens
        ), f"path={path}: eager={r_eager.tokens}, graph={r_graph.tokens}"
        torch.testing.assert_close(r_eager.lengths, r_graph.lengths, rtol=0, atol=0)
        torch.testing.assert_close(r_eager.scores, r_graph.scores, rtol=0, atol=0)

    def test_streaming_decode_with_blank_threshold(self, device):
        """Blank-threshold skipping is preserved on the graph path."""
        V = 5
        path = [0, 0, 1, 0, 0, 0, 2, 0, 3, 0, 0]
        r_eager = self._decode_path(
            V, path, use_cuda_graphs=False, blank_threshold=0.5, device=device
        )
        r_graph = self._decode_path(
            V, path, use_cuda_graphs=True, blank_threshold=0.5, device=device
        )
        assert r_eager.tokens == r_graph.tokens

    def test_streaming_decode_multi_chunk_matches_eager(self, device):
        """Decode parity across multiple chunks in one stream."""
        V = 6
        chunks = [
            [1, 0, 2],
            [0, 0, 3, 0],
            [4, 0, 0, 5],
        ]

        def run(use_cuda_graphs: bool):
            cfg = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=32, blank_threshold=0.0)
            decoder = GpuStreamingDecoder(cfg, use_cuda_graphs=use_cuda_graphs)
            decoder.init_stream(batch=1, vocab_size=V, device=device)
            for c in chunks:
                decoder.decode_chunk(_make_logp_gpu(len(c), V, c, device))
            return decoder.finalize_stream()

        r_eager = run(False)
        r_graph = run(True)
        assert r_eager.tokens == r_graph.tokens

    def test_streaming_decode_state_pool_reuse_with_graphs(self, device):
        """Pooled state reuse (same shape) keeps captured graphs valid."""
        V = 5
        cfg = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=32, blank_threshold=0.0)
        decoder = GpuStreamingDecoder(cfg, use_cuda_graphs=True)

        state = decoder.create_state(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(4, V, [1, 0, 2, 3], device), state=state)
        r1 = decoder.finalize_stream(state=state)
        assert r1.tokens[0][0] == [1, 2, 3]

        # Reuse the same state — graphs should still be valid.
        decoder.reset_state(state, batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(4, V, [4, 0, 1, 2], device), state=state)
        r2 = decoder.finalize_stream(state=state)
        assert r2.tokens[0][0] == [4, 1, 2]

    def test_streaming_decode_state_realloc_releases_graphs(self, device):
        """Resetting to a larger shape releases stale graphs and recaptures."""
        cfg = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=32, blank_threshold=0.0)
        decoder = GpuStreamingDecoder(cfg, use_cuda_graphs=True)

        state = decoder.create_state(batch=1, vocab_size=5, device=device)
        decoder.decode_chunk(_make_logp_gpu(3, 5, [1, 0, 2], device), state=state)
        decoder.finalize_stream(state=state)

        # Bigger vocab forces a realloc → graphs released → recaptured.
        decoder.reset_state(state, batch=1, vocab_size=8, device=device)
        decoder.decode_chunk(_make_logp_gpu(4, 8, [3, 0, 5, 7], device), state=state)
        r = decoder.finalize_stream(state=state)
        assert r.tokens[0][0] == [3, 5, 7]


# ---------------------------------------------------------------------------
# Batched streaming decode (decode_chunk_batch) — parity vs per-stream path
# ---------------------------------------------------------------------------


@pytest.mark.cuda
class TestCtcDecoderBatchedChunk:
    """Parity tests for ``GpuStreamingDecoder.decode_chunk_batch``."""

    def _run_per_stream(self, decoder, paths, V, device):
        states = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in paths]
        for s, path in zip(states, paths):
            decoder.decode_chunk(_make_logp_gpu(len(path), V, path, device), state=s)
        return [decoder.finalize_stream(state=s) for s in states], states

    def _run_batched(self, decoder, paths, V, device):
        # All chunks must share the same T for the batched call; pad with
        # blank where shorter.
        max_T = max(len(p) for p in paths)
        padded = [p + [0] * (max_T - len(p)) for p in paths]
        logp = torch.stack(
            [_make_logp_gpu(max_T, V, p, device).squeeze(0) for p in padded],
            dim=0,
        )  # (N, T, V)
        states = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in paths]
        decoder.decode_chunk_batch(logp, states)
        return [decoder.finalize_stream(state=s) for s in states], states

    def test_batched_matches_per_stream_no_graphs(self, device):
        V = 6
        cfg = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=20, blank_threshold=0.0)
        paths = [[1, 0, 2, 0, 3], [4, 0, 5, 0, 1], [2, 0, 3, 0, 4]]

        d_per = GpuStreamingDecoder(cfg, use_cuda_graphs=False)
        d_bat = GpuStreamingDecoder(cfg, use_cuda_graphs=False)

        r_per, _ = self._run_per_stream(d_per, paths, V, device)
        r_bat, _ = self._run_batched(d_bat, paths, V, device)

        assert len(r_per) == len(r_bat)
        for a, b in zip(r_per, r_bat):
            assert a.tokens[0][0] == b.tokens[0][0]

    def test_batched_matches_per_stream_with_graphs(self, device):
        V = 6
        cfg = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=20, blank_threshold=0.0)
        paths = [[1, 0, 2], [3, 0, 4], [5, 0, 1], [2, 0, 3]]

        d_per = GpuStreamingDecoder(cfg, use_cuda_graphs=True)
        d_bat = GpuStreamingDecoder(cfg, use_cuda_graphs=True)

        r_per, _ = self._run_per_stream(d_per, paths, V, device)
        r_bat, _ = self._run_batched(d_bat, paths, V, device)

        for a, b in zip(r_per, r_bat):
            assert a.tokens[0][0] == b.tokens[0][0]

    def test_batched_step_counter_advances(self, device):
        V = 5
        # Use the default blank_threshold so the blank frame at t=1 is
        # skipped by the mask path — same behaviour as decode_chunk's
        # eager loop.
        cfg = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10, blank_threshold=0.98)
        decoder = GpuStreamingDecoder(cfg, use_cuda_graphs=False)
        paths = [[1, 0, 2], [3, 0, 4]]
        _, states = self._run_batched(decoder, paths, V, device)
        # Two non-blank frames per path → step == 2, frame_idx == 3.
        for s in states:
            assert s.step == 2
            assert s.actual_frame_idx == 3

    def test_batched_blank_mask_skips_frames(self, device):
        V = 6
        cfg = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10, blank_threshold=0.0)
        decoder = GpuStreamingDecoder(cfg, use_cuda_graphs=False)
        paths = [[1, 0, 2, 0, 3], [4, 0, 5, 0, 1]]
        N, T = 2, 5
        # Mask the middle frame for both streams; the path still encodes
        # blank at t=1 and t=3, so the masked-skip mirrors blank behaviour
        # and the final transcript stays the same.
        mask = torch.ones((N, T), dtype=torch.uint8, device="cpu")
        mask[:, 1] = 0
        mask[:, 3] = 0
        logp = torch.stack(
            [_make_logp_gpu(T, V, p, device).squeeze(0) for p in paths],
            dim=0,
        )
        states = [decoder.create_state(batch=1, vocab_size=V, device=device) for _ in paths]
        decoder.decode_chunk_batch(logp, states, is_speech_mask=mask)
        for s, path in zip(states, paths):
            r = decoder.finalize_stream(state=s)
            # Three non-blank frames produce three emit steps.
            assert s.step == 3
            assert r.tokens[0][0] == [path[0], path[2], path[4]]

    def test_batched_single_stream_matches_decode_chunk(self, device):
        V = 5
        cfg = GpuDecoderConfig(beam_size=3, blank_id=0, max_seq_len=10, blank_threshold=0.0)
        decoder = GpuStreamingDecoder(cfg, use_cuda_graphs=True)
        path = [1, 0, 2, 0, 3]
        # Single-stream batched call should match the well-tested
        # single-state path.
        s_bat = decoder.create_state(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk_batch(_make_logp_gpu(5, V, path, device), [s_bat])
        r_bat = decoder.finalize_stream(state=s_bat)

        s_per = decoder.create_state(batch=1, vocab_size=V, device=device)
        decoder.decode_chunk(_make_logp_gpu(5, V, path, device), state=s_per)
        r_per = decoder.finalize_stream(state=s_per)

        assert r_bat.tokens[0][0] == r_per.tokens[0][0]
        assert s_bat.step == s_per.step
