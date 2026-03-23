#!/usr/bin/env python3
"""
Unit tests for decoder wrappers (CTC prefix beam search).
"""

from pathlib import Path

import pytest
import torch
import torchaudio

from oasr.models.conformer import load_wenet_checkpoint
from oasr.decoder import CtcPrefixBeamSearch


class TestCtcPrefixBeamSearch:
    """Tests for the CtcPrefixBeamSearch Python wrapper."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_decode(self, ckpt_dir: str, audio_path: str, dtype: torch.dtype):
        """Decoder should match WeNet for the same checkpoint."""

        if not ckpt_dir or not Path(ckpt_dir).exists():
            pytest.skip(
                "WeNet checkpoint dir not set or not found; set WENET_CKPT_DIR env var or --ckpt-dir"
            )
        if not audio_path or not Path(audio_path).exists():
            pytest.skip(
                "Audio path not set or not found; set WENET_AUDIO_PATH env var or --audio-path"
            )

        def read_audio(path: str):
            """Read audio file."""
            audio, sr = torchaudio.load(path)
            audio = audio * (1 << 15)
            return audio, sr

        def extract_features(audio: torch.Tensor, sr: int):
            # extract feature using torchaudio.compliance.kaldi as kaldi
            feats = torchaudio.compliance.kaldi.fbank(
                audio,
                sr,
                num_mel_bins=80,
                frame_shift=10,
                frame_length=25,
                dither=1.0,
            )
            return feats

        def build_dictionary(ckpt_dir: str):
            """Build dictionary from words.txt."""
            id2word = {}
            with open(Path(ckpt_dir) / "words.txt", "r") as f:
                for line in f.readlines():
                    word, idx = line.strip().split()
                    id2word[int(idx)] = word
            return id2word

        def decode(probs: torch.Tensor, id2word: dict[int, str]):
            """Detokenize outputs."""
            decoder = CtcPrefixBeamSearch(
                blank=0, first_beam_size=10, second_beam_size=10)

            probs = probs.squeeze(0).to(device="cpu", dtype=torch.float32)

            decoder.reset()
            decoder.search(probs)
            decoder.finalize_search()

            outputs = decoder.outputs
            likelihood = decoder.likelihood

            text = "".join([id2word[idx]
                           for idx in outputs[0]]).replace("▁", " ").strip()
            return text, likelihood

        # Run the encoder on CUDA using the requested half/bfloat16 dtype.
        device = "cuda"
        oasr_model, _ = load_wenet_checkpoint(
            str(ckpt_dir), device=device, dtype=dtype)

        decoder = CtcPrefixBeamSearch(
            blank=0, first_beam_size=10, second_beam_size=10)

        torch.manual_seed(42)

        audio, sr = read_audio(audio_path)
        feats = extract_features(audio, sr)  # [T, F]
        feats = feats.unsqueeze(0)  # [1, T, F]
        feats = feats.to(device=device, dtype=dtype)

        lengths = torch.full(
            (1,), feats.size(1), dtype=torch.long, device=device
        )

        with torch.no_grad():
            probs = oasr_model(feats, lengths)
            chunk_by_chunk_probs = oasr_model.forward_chunk_by_chunk(
                feats, decoding_chunk_size=4, num_decoding_left_chunks=2)

        # CtcPrefixBeamSearch expects log-probs of shape [T, V] on CPU in
        # float32; collapse the batch dimension and move to CPU.
        probs = probs.squeeze(0).to(device="cpu", dtype=dtype)
        chunk_by_chunk_probs = chunk_by_chunk_probs.squeeze(
            0).to(device="cpu", dtype=dtype)

        id2word = build_dictionary(ckpt_dir)

        text, _ = decode(probs, id2word)
        chunk_by_chunk_text, _ = decode(chunk_by_chunk_probs, id2word)
        print(f"text: {text}")
        print(f"chunk_by_chunk_text: {chunk_by_chunk_text}")

        assert len(text.split()) > 0
        assert len(chunk_by_chunk_text.split()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
