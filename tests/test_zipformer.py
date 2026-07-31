# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Zipformer CTC model tests.

* Registry / contract tests — always run (no reference / checkpoint needed).
* Numerical parity tests — load identical random weights into the OASR port and
  the vendored icefall reference modules and assert bit-exact (within fp tol)
  outputs, for both the offline and chunk-wise streaming paths.  Skipped when the
  icefall reference source is not available locally.

Point ``ICEFALL_ZIPFORMER_DIR`` at a checkout of
``icefall/egs/librispeech/ASR/zipformer`` (containing ``zipformer.py``,
``scaling.py``, ``subsampling.py``) to enable the parity tests, or place those
files under ``/tmp/icefall_ref``.
"""

import sys

import assets
import pytest
import torch

from oasr.models import (
    CacheSpec,
    ZipformerEncoderConfig,
    ZipformerModel,
    ZipformerModelConfig,
    get_model_entry,
    list_models,
)
from oasr.models.zipformer import IcefallConverter

# --------------------------------------------------------------------------- #
# Reference harness: import the icefall zipformer modules standalone.
# --------------------------------------------------------------------------- #


def _load_reference(tmp_path):
    """Import icefall zipformer/subsampling with stubbed icefall-only deps."""
    ref = assets.require("ICEFALL_ZIPFORMER_DIR")

    stub = tmp_path / "_zip_stubs"
    stub.mkdir(exist_ok=True)
    (stub / "k2.py").write_text(
        "import torch\n"
        "def swoosh_l_forward(x):\n"
        "    z=torch.zeros((),dtype=x.dtype,device=x.device)\n"
        "    return torch.logaddexp(z,x-4.0)-0.08*x-0.035\n"
        "def swoosh_r_forward(x):\n"
        "    z=torch.zeros((),dtype=x.dtype,device=x.device)\n"
        "    return torch.logaddexp(z,x-1.0)-0.08*x-0.313261687\n"
        "swoosh_l=swoosh_l_forward\nswoosh_r=swoosh_r_forward\n"
    )
    (stub / "encoder_interface.py").write_text(
        "import torch\nclass EncoderInterface(torch.nn.Module):\n    pass\n"
    )
    icefall_pkg = stub / "icefall"
    icefall_pkg.mkdir(exist_ok=True)
    (icefall_pkg / "__init__.py").write_text("")
    (icefall_pkg / "utils.py").write_text(
        "import contextlib, torch\n"
        "def torch_autocast(*a, **k):\n    return contextlib.nullcontext()\n"
        "def make_pad_mask(lengths, max_len=0):\n"
        "    n=lengths.numel(); ml=max(int(lengths.max()), max_len)\n"
        "    e=torch.arange(ml, device=lengths.device).expand(n, ml)\n"
        "    return e >= lengths.unsqueeze(1)\n"
    )
    for p in (str(stub), ref):
        if p not in sys.path:
            sys.path.insert(0, p)
    for m in ("zipformer", "subsampling", "scaling"):
        sys.modules.pop(m, None)
    import subsampling as ref_sub  # type: ignore
    import zipformer as ref_zip  # type: ignore

    return ref_zip, ref_sub


def _tiny_encoder_config(causal=False, chunk_size=(-1,), left_context_frames=(-1,)):
    # All derived projection dims must be multiples of 8: the OASR GEMM kernels
    # (oasr.gemm / oasr.layers.Linear) require N % 8 == K % 8 == 0.  encoder_dim
    # and feedforward_dim are multiples of 32 so the 3/4 and 5/4 feed-forward
    # widths (and the 3/4 nonlin-attention hidden) stay 8-aligned.
    return ZipformerEncoderConfig(
        feature_dim=80,
        downsampling_factor=(1, 2),
        encoder_dim=(64, 96),
        num_encoder_layers=(1, 1),
        query_head_dim=(8,),
        pos_head_dim=(4,),
        value_head_dim=(6,),
        num_heads=(4, 4),
        feedforward_dim=(64, 96),
        cnn_module_kernel=(15, 15),
        pos_dim=16,
        causal=causal,
        chunk_size=chunk_size,
        left_context_frames=left_context_frames,
    )


def _build_reference(ref_zip, ref_sub, enc_cfg):
    embed = ref_sub.Conv2dSubsampling(enc_cfg.feature_dim, enc_cfg.encoder_dim[0]).eval()
    zip_enc = ref_zip.Zipformer2(
        output_downsampling_factor=enc_cfg.output_downsampling_factor,
        downsampling_factor=enc_cfg.downsampling_factor,
        encoder_dim=enc_cfg.encoder_dim,
        # training-only (per-frame dropout dim); must satisfy u <= encoder_dim.
        encoder_unmasked_dim=enc_cfg.encoder_dim,
        num_encoder_layers=enc_cfg.num_encoder_layers,
        query_head_dim=enc_cfg.query_head_dim,
        pos_head_dim=enc_cfg.pos_head_dim,
        value_head_dim=enc_cfg.value_head_dim,
        num_heads=enc_cfg.num_heads,
        feedforward_dim=enc_cfg.feedforward_dim,
        cnn_module_kernel=enc_cfg.cnn_module_kernel,
        pos_dim=enc_cfg.pos_dim,
        causal=enc_cfg.causal,
        chunk_size=enc_cfg.chunk_size,
        left_context_frames=enc_cfg.left_context_frames,
    ).eval()
    return embed, zip_enc


# --------------------------------------------------------------------------- #
# Registry / contract tests (always run)
# --------------------------------------------------------------------------- #


class TestZipformerRegistry:
    def test_registered(self):
        assert "zipformer" in list_models()
        entry = get_model_entry("zipformer")
        assert entry.model_cls is ZipformerModel
        assert isinstance(entry.converter, IcefallConverter)

    def test_contract(self):
        # Streaming-capable (causal) config: advertises the stateful backend and
        # therefore reports a cache geometry.
        cfg = ZipformerModelConfig(
            encoder=_tiny_encoder_config(causal=True, chunk_size=(8,)), vocab_size=32
        )
        model = ZipformerModel.from_config(cfg).eval()
        assert model.decode_type == "ctc"
        assert model.head is model.ctc
        assert model.encoder.streaming_kind == "stateful"
        assert isinstance(model.cache_spec, CacheSpec)
        # cache_spec from the live model and from the config must agree.
        assert model.cache_spec == cfg.cache_spec
        # output dim == max(encoder_dim)
        assert model.encoder.output_size == 96
        assert model.cache_spec.num_layers == 2  # 1 + 1
        assert model.cache_spec.conv_kernel_size == 1  # no slot-CNN cache

    def test_non_causal_config_is_offline_only(self):
        """``causal=False`` has no chunk-wise forward, so it must not claim one.

        Keeps ``streaming_kind`` and ``cache_spec`` in lockstep: an encoder that
        cannot stream reports no streaming cache, which is what lets the engine
        refuse streaming service mode at construction and skip allocating a
        paged pool it would never use.
        """
        cfg = ZipformerModelConfig(encoder=_tiny_encoder_config(), vocab_size=32)
        model = ZipformerModel.from_config(cfg).eval()
        assert model.encoder.config.causal is False
        assert model.encoder.streaming_kind == "none"
        assert model.cache_spec is None
        with pytest.raises(ValueError, match="not configured for streaming"):
            _ = model.encoder.streaming_chunk_frames

    def test_forward_shapes(self):
        # The CTC head uses the CUDA-only gemm_log_softmax kernel.
        if not torch.cuda.is_available():
            pytest.skip("CTC head (gemm_log_softmax) requires CUDA")
        device = "cuda"
        cfg = ZipformerModelConfig(encoder=_tiny_encoder_config(), vocab_size=32)
        # gemm_log_softmax supports FP16/BF16 only.
        model = ZipformerModel.from_config(cfg).eval().to(device).half()
        B, T = 2, 80
        x = torch.randn(B, T, 80, device=device, dtype=torch.float16)
        xl = torch.tensor([T, T - 8], dtype=torch.int32, device=device)
        with torch.no_grad():
            log_probs, out_lens = model.forward_offline(x, xl)
        assert log_probs.shape[0] == B and log_probs.shape[2] == 32
        # total subsampling is 4x: ((T-7)//2 + 1)//2
        assert log_probs.shape[1] == ((T - 7) // 2 + 1) // 2
        assert out_lens[0].item() == log_probs.shape[1]


# --------------------------------------------------------------------------- #
# Parity tests vs icefall reference
# --------------------------------------------------------------------------- #


class TestZipformerParity:
    """Parity vs the fp32 icefall reference.

    The OASR port runs on CUDA in FP16 (its CUDA GEMM / norm / activation kernels
    are half-precision); the icefall reference stays in FP32 on CPU.  Tolerances
    are loosened to FP16-with-FP32-accumulation reality (measured max-abs ~9e-3
    against reference values up to ~5.7), versus the exact match the pure-torch
    port used to achieve.
    """

    # FP16 (FP32 GEMM accumulation) vs FP32 reference.
    _RTOL = 2e-2
    _ATOL = 2e-2

    def test_offline_parity(self, tmp_path):
        if not torch.cuda.is_available():
            pytest.skip("OASR kernels require CUDA")
        ref_zip, ref_sub = _load_reference(tmp_path)
        sys.path.insert(0, assets.require("ICEFALL_ZIPFORMER_DIR"))
        from icefall.utils import make_pad_mask as ref_make_pad_mask  # type: ignore

        torch.manual_seed(1234)
        enc_cfg = _tiny_encoder_config()
        ref_embed, ref_enc = _build_reference(ref_zip, ref_sub, enc_cfg)  # CPU fp32

        model = ZipformerModel(ZipformerModelConfig(encoder=enc_cfg, vocab_size=32)).eval()
        # Identical weights: module names mirror icefall, so this is a strict load
        # (depthwise-conv weights are transposed [C,1,K]->[K,1,C] by the load hook).
        model.encoder.encoder_embed.load_state_dict(ref_embed.state_dict())
        model.encoder.encoder.load_state_dict(ref_enc.state_dict())
        model = model.half().cuda().eval()

        B, T = 2, 96
        x = torch.randn(B, T, enc_cfg.feature_dim)  # CPU fp32 input shared by both
        xl = torch.tensor([T, T - 10], dtype=torch.int32)

        with torch.no_grad():
            xe, xle = ref_embed(x, xl)
            spm = ref_make_pad_mask(xle)
            ref_out, ref_lens = ref_enc(xe.permute(1, 0, 2), xle, spm)
            ref_out = ref_out.permute(1, 0, 2)

            my_out, my_masks = model.encoder(x.half().cuda(), xl.cuda())
            my_lens = my_masks.squeeze(1).sum(-1)

        assert my_out.shape == ref_out.shape, (my_out.shape, ref_out.shape)
        torch.testing.assert_close(my_out.float().cpu(), ref_out, rtol=self._RTOL, atol=self._ATOL)
        assert torch.equal(my_lens.cpu().to(ref_lens.dtype), ref_lens)

    def test_streaming_parity(self, tmp_path):
        if not torch.cuda.is_available():
            pytest.skip("OASR kernels require CUDA")
        ref_zip, ref_sub = _load_reference(tmp_path)

        torch.manual_seed(4321)
        L, C = 32, 16
        enc_cfg = _tiny_encoder_config(causal=True, chunk_size=(C,), left_context_frames=(L,))
        ref_embed, ref_enc = _build_reference(ref_zip, ref_sub, enc_cfg)  # CPU fp32

        model = ZipformerModel(ZipformerModelConfig(encoder=enc_cfg, vocab_size=32)).eval()
        model.encoder.encoder_embed.load_state_dict(ref_embed.state_dict())
        model.encoder.encoder.load_state_dict(ref_enc.state_dict())
        model = model.half().cuda().eval()

        B = 2
        # Init states (port on CUDA/fp16, reference on CPU/fp32; identical zeros).
        my_states = model.get_streaming_init_states(B, device="cuda", dtype=torch.float16)
        ref_embed_state = ref_embed.get_init_states(B)
        ref_enc_states = ref_enc.get_init_states(B)

        chunk_T = 45  # -> (45-7)//2 - 3 = 16 subsampled frames
        x = torch.randn(B, chunk_T, enc_cfg.feature_dim)  # CPU fp32 shared input
        xl = torch.full((B,), chunk_T, dtype=torch.int32)

        with torch.no_grad():
            # reference one-chunk streaming
            xe, xle, _ = ref_embed.streaming_forward(x, xl, ref_embed_state)
            xe_t = xe.permute(1, 0, 2)
            spm = torch.zeros(B, L + xe_t.size(0), dtype=torch.bool)
            ref_out, ref_lens, _ = ref_enc.streaming_forward(xe_t, xle, ref_enc_states, spm)
            ref_out = ref_out.permute(1, 0, 2)

            # port one-chunk streaming
            my_hidden, my_lens, _ = model.encoder.streaming_forward(
                x.half().cuda(), xl.cuda(), my_states
            )

        assert my_hidden.shape == ref_out.shape, (my_hidden.shape, ref_out.shape)
        torch.testing.assert_close(
            my_hidden.float().cpu(), ref_out, rtol=self._RTOL, atol=self._ATOL
        )


# --------------------------------------------------------------------------- #
# Real icefall checkpoint (weights + tokenizer from an actual release)
# --------------------------------------------------------------------------- #

# Declared in tests/assets.py, including the "dangling LFS symlink" probe: an
# HF snapshot can be present with no payload, which is not a usable checkpoint.
ZIPFORMER_CKPT = assets.declared("ZIPFORMER_CKPT")
ZIP_WAV_DIR = assets.declared("WAV_DIR")


@pytest.mark.requires_assets("ZIPFORMER_CKPT")
class TestRealIcefallCheckpoint:
    """The converter against a real release, not a synthetic state dict.

    icefall ships **no config file** — ``IcefallConverter`` infers the whole
    architecture from tensor shapes, which is the one part of the checkpoint layer
    that cannot be validated with random weights.  This release documents its own
    geometry in ``exp/train.sh``, so the inference has a ground truth to be
    checked against.
    """

    #: From ``exp/train.sh`` of the ``-large-cr-ctc-20241018`` release.
    WANT_LAYERS = (2, 2, 4, 5, 4, 2)
    WANT_DIM = (192, 256, 512, 768, 512, 256)
    WANT_FF = (512, 768, 1536, 2048, 1536, 768)

    @pytest.fixture(scope="class")
    def bundle(self):
        from oasr.models.registry import load_checkpoint_bundle

        return load_checkpoint_bundle(ZIPFORMER_CKPT)

    def test_detected_and_bundled(self, bundle):
        arch, b = bundle
        assert arch == "zipformer"
        assert b.source_format == "icefall"
        assert b.decoding.default_decode_type == "ctc"
        # icefall CTC blank is id 0.
        assert b.decoding.blank_id == 0

    def test_shape_inference_matches_the_training_config(self, bundle):
        """The claim that could not be tested without real weights.

        A wrong inference builds a plausible-looking model that either dies with a
        raw shape error much later or — if the dims happen to coincide — loads and
        produces garbage.  Checked against the geometry the release itself
        documents.
        """
        _arch, b = bundle
        enc = b.model_config.encoder
        assert tuple(enc.num_encoder_layers) == self.WANT_LAYERS
        assert tuple(enc.encoder_dim) == self.WANT_DIM
        assert tuple(enc.feedforward_dim) == self.WANT_FF
        assert enc.causal is False, "this release is the non-streaming (offline) model"

    def test_vocab_comes_from_the_ctc_head(self, bundle):
        _arch, b = bundle
        # lang_bpe_500 → 500 units + blank/unk/sos-eos, padded to a multiple of 8
        # for the GEMM kernels.
        assert b.model_config.vocab_size % 8 == 0
        assert 500 <= b.model_config.vocab_size <= 512

    def test_tokenizer_travels_with_the_checkpoint(self, bundle):
        _arch, b = bundle
        assert b.tokenizer is not None, "no TokenizerSpec — decode would emit raw ids"
        assert b.tokenizer.kind == "sentencepiece"

    def test_tokenizer_is_found_from_the_exp_subdir_too(self):
        """icefall puts weights in ``exp/`` and the tokenizer in ``data/`` — siblings.

        ``_find_ckpt`` accepts ``<root>/exp``, so pointing there is natural; without
        searching the parent the bundle loaded with ``tokenizer=None`` and the
        engine silently fell back to joining raw token ids — a transcript of
        numbers, with no error anywhere.
        """
        from pathlib import Path

        from oasr.models.registry import load_checkpoint_bundle

        exp = Path(ZIPFORMER_CKPT) / "exp"
        if not exp.is_dir():
            pytest.skip("release has no exp/ subdir")
        _arch, b = load_checkpoint_bundle(exp)
        assert b.tokenizer is not None and b.tokenizer.kind == "sentencepiece"

    def test_weights_load_exactly(self, bundle):
        """No missing tensors, no dropped tensors — the inference was right.

        This is the assertion that a synthetic checkpoint cannot make: a random
        state dict is generated *from* the config, so it fits by construction.
        """
        from oasr.models.registry import instantiate_from_bundle

        arch, b = bundle
        model, cfg, report = instantiate_from_bundle(arch, b, device="cpu", dtype=torch.float32)
        assert not report.missing, f"model tensors not filled: {report.missing[:8]}"
        assert not report.dropped, f"checkpoint tensors dropped: {report.dropped[:8]}"
        assert len(report.mapped) > 500, "suspiciously few tensors mapped"
        assert sorted(model.capabilities) == ["ctc"]
        # This release is `cr-ctc`, i.e. non-causal, so it is offline-only and
        # must say so; a causal release would report "stateful" here.
        expect = "stateful" if model.encoder.config.causal else "none"
        assert model.encoder.streaming_kind == expect

    def test_feature_spec_uses_icefall_audio_scale(self, bundle):
        """icefall computes FBANK via lhotse on the [-1, 1] waveform.

        WeNet's ``audio_scale=32768`` offsets every log-mel bin by ~20.8 and
        costs the leading token of the transcript (see
        ``IcefallConverter.build_feature_spec``), so this pins the convention
        at the source rather than only through an end-to-end transcript.
        """
        _, b = bundle
        assert b.features is not None
        assert b.features.audio_scale == 1.0
        assert b.features.kind == "kaldi_fbank"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="engine requires CUDA")
    @pytest.mark.requires_assets("WAV_DIR")
    class TestEngineE2E:
        """Offline + streaming transcription of real audio on real weights."""

        def _audios(self, n=2):
            import torchaudio

            wavs = assets.require_wavs(n)
            return [torchaudio.load(w)[0].squeeze(0) for w in wavs]

        def _engine(self, mode):
            from oasr.engine import ASREngine, EngineConfig

            return ASREngine(
                EngineConfig(
                    ckpt_dir=ZIPFORMER_CKPT,
                    service_mode=mode,
                    max_batch_size=4,
                    dtype=torch.float16,
                )
            )

        # LJSpeech ground truth for LJ001-0001 / LJ001-0002. Asserted in full,
        # not by keyword: the bug this guards against (an icefall checkpoint
        # loaded with WeNet's audio_scale=32768) dropped only the *leading*
        # token -- "TING IN THE ONLY SENSE ..." and "BEING COMPARATIVELY
        # MODERN" -- which a substring check on "printing"/"modern" happily
        # passes for the second utterance.
        GROUND_TRUTH = [
            "printing in the only sense with which we are at present concerned "
            "differs from most if not from all the arts and crafts represented "
            "in the exhibition",
            "in being comparatively modern",
        ]

        def test_offline_transcribes_real_audio(self):
            eng = self._engine("offline")
            try:
                texts = eng.transcribe_offline(self._audios(2))
                texts = [t.text if hasattr(t, "text") else t for t in texts]
            finally:
                del eng
                torch.cuda.empty_cache()
            for got, want in zip(texts, self.GROUND_TRUTH):
                assert got.lower().strip() == want, f"got={got!r} want={want!r}"

        def _is_causal_release(self) -> bool:
            from oasr.models import build_model_from_checkpoint

            m = build_model_from_checkpoint(ZIPFORMER_CKPT, device="cpu", dtype=torch.float32)
            m = m[0] if isinstance(m, tuple) else m
            kind = m.encoder.streaming_kind
            del m
            return kind != "none"

        def test_streaming_agrees_with_offline(self):
            """The stateful backend on real weights.

            A non-streaming (``causal=False``) release decoded chunk-by-chunk will
            not match offline exactly — it was never trained for it — so this
            asserts the transcript is *recognisably right* rather than identical.
            """
            if not self._is_causal_release():
                pytest.skip(
                    "checkpoint is a non-causal (causal=False) release: it has no "
                    "chunk-wise forward, so streaming is refused by design"
                )
            offline = self._engine("offline")
            try:
                ref = offline.transcribe_offline(self._audios(1))[0]
                ref = ref.text if hasattr(ref, "text") else ref
            finally:
                del offline
                torch.cuda.empty_cache()

            eng = self._engine("streaming")
            try:
                got = eng.transcribe(self._audios(1)[0])
                got = got.text if hasattr(got, "text") else got
            finally:
                del eng
                torch.cuda.empty_cache()
            assert got.strip(), "streaming produced an empty transcript"
            ref_words = set(ref.lower().split())
            hit = sum(1 for w in got.lower().split() if w in ref_words)
            assert hit >= max(3, len(ref_words) // 3), f"ref={ref!r} got={got!r}"

        def test_streaming_mode_is_refused_at_construction(self):
            """A non-causal release must fail at engine build, not first request.

            Regression guard for ``ZipformerEncoder.streaming_kind``: it used to
            claim ``"stateful"`` from the config's mere existence, so an engine
            pinned to streaming built happily and then raised out of
            ``streaming_chunk_frames`` once a request arrived.
            """
            if self._is_causal_release():
                pytest.skip("checkpoint is a causal release; streaming is supported")
            with pytest.raises((ValueError, RuntimeError)) as ei:
                self._engine("streaming")
            assert "stream" in str(ei.value).lower(), ei.value
