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

import os
import sys

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


def _ref_dir():
    for cand in (os.environ.get("ICEFALL_ZIPFORMER_DIR"), "/tmp/icefall_ref"):
        if cand and os.path.exists(os.path.join(cand, "zipformer.py")):
            return cand
    return None


def _load_reference(tmp_path):
    """Import icefall zipformer/subsampling with stubbed icefall-only deps."""
    ref = _ref_dir()
    if ref is None:
        pytest.skip("icefall zipformer reference not found (set ICEFALL_ZIPFORMER_DIR)")

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
    return ZipformerEncoderConfig(
        feature_dim=80,
        downsampling_factor=(1, 2),
        encoder_dim=(48, 64),
        num_encoder_layers=(1, 1),
        query_head_dim=(8,),
        pos_head_dim=(4,),
        value_head_dim=(6,),
        num_heads=(4, 4),
        feedforward_dim=(48, 64),
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
        cfg = ZipformerModelConfig(encoder=_tiny_encoder_config(), vocab_size=32)
        model = ZipformerModel.from_config(cfg).eval()
        assert model.decode_type == "ctc"
        assert model.head is model.ctc
        assert isinstance(model.cache_spec, CacheSpec)
        # cache_spec from the live model and from the config must agree.
        assert model.cache_spec == cfg.cache_spec
        # output dim == max(encoder_dim)
        assert model.encoder.output_size == 64
        assert model.cache_spec.num_layers == 2  # 1 + 1
        assert model.cache_spec.conv_kernel_size == 1  # no slot-CNN cache

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
    def test_offline_parity(self, tmp_path):
        ref_zip, ref_sub = _load_reference(tmp_path)
        sys.path.insert(0, _ref_dir())
        from icefall.utils import make_pad_mask as ref_make_pad_mask  # type: ignore

        torch.manual_seed(1234)
        enc_cfg = _tiny_encoder_config()
        ref_embed, ref_enc = _build_reference(ref_zip, ref_sub, enc_cfg)

        model = ZipformerModel(ZipformerModelConfig(encoder=enc_cfg, vocab_size=32)).eval()
        # Identical weights: module names mirror icefall, so this is a strict load.
        model.encoder.encoder_embed.load_state_dict(ref_embed.state_dict())
        model.encoder.encoder.load_state_dict(ref_enc.state_dict())

        B, T = 2, 96
        x = torch.randn(B, T, enc_cfg.feature_dim)
        xl = torch.tensor([T, T - 10], dtype=torch.int32)

        with torch.no_grad():
            xe, xle = ref_embed(x, xl)
            spm = ref_make_pad_mask(xle)
            ref_out, ref_lens = ref_enc(xe.permute(1, 0, 2), xle, spm)
            ref_out = ref_out.permute(1, 0, 2)

            my_out, my_masks = model.encoder(x, xl)
            my_lens = my_masks.squeeze(1).sum(-1)

        assert my_out.shape == ref_out.shape, (my_out.shape, ref_out.shape)
        torch.testing.assert_close(my_out, ref_out, rtol=1e-4, atol=1e-4)
        assert torch.equal(my_lens.to(ref_lens.dtype), ref_lens)

    def test_streaming_parity(self, tmp_path):
        ref_zip, ref_sub = _load_reference(tmp_path)

        torch.manual_seed(4321)
        L, C = 32, 16
        enc_cfg = _tiny_encoder_config(
            causal=True, chunk_size=(C,), left_context_frames=(L,)
        )
        ref_embed, ref_enc = _build_reference(ref_zip, ref_sub, enc_cfg)

        model = ZipformerModel(ZipformerModelConfig(encoder=enc_cfg, vocab_size=32)).eval()
        model.encoder.encoder_embed.load_state_dict(ref_embed.state_dict())
        model.encoder.encoder.load_state_dict(ref_enc.state_dict())

        B = 2
        # Init states (port + reference, identical zeros).
        my_states = model.get_streaming_init_states(B)
        ref_embed_state = ref_embed.get_init_states(B)
        ref_enc_states = ref_enc.get_init_states(B)

        chunk_T = 45  # -> (45-7)//2 - 3 = 16 subsampled frames
        x = torch.randn(B, chunk_T, enc_cfg.feature_dim)
        xl = torch.full((B,), chunk_T, dtype=torch.int32)

        with torch.no_grad():
            # reference one-chunk streaming
            xe, xle, _ = ref_embed.streaming_forward(x, xl, ref_embed_state)
            xe_t = xe.permute(1, 0, 2)
            spm = torch.zeros(B, L + xe_t.size(0), dtype=torch.bool)
            ref_out, ref_lens, _ = ref_enc.streaming_forward(xe_t, xle, ref_enc_states, spm)
            ref_out = ref_out.permute(1, 0, 2)

            # port one-chunk streaming
            my_hidden, my_lens, _ = model.encoder.streaming_forward(x, xl, my_states)

        assert my_hidden.shape == ref_out.shape, (my_hidden.shape, ref_out.shape)
        torch.testing.assert_close(my_hidden, ref_out, rtol=1e-4, atol=1e-4)
