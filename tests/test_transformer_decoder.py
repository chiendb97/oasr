#!/usr/bin/env python3
"""Tests for the WeNet-compatible transformer decoder (U2++ AED branch).

Parity oracle: the upstream WeNet v2.0.1 decoder sources under
``/tmp/wenet_ref`` (override with ``WENET_REF_DIR``) — the same pattern as the
Zipformer tests' ``/tmp/icefall_ref``.  Tests needing them skip when absent::

    mkdir -p /tmp/wenet_ref/transformer /tmp/wenet_ref/utils
    for f in transformer/decoder.py transformer/decoder_layer.py \
             transformer/attention.py transformer/embedding.py \
             transformer/positionwise_feed_forward.py utils/mask.py \
             utils/common.py; do
      curl -sf https://raw.githubusercontent.com/wenet-e2e/wenet/v2.0.1/wenet/$f \
           -o /tmp/wenet_ref/$f
    done
"""

import os
import sys
import types

import pytest
import torch

from oasr.models.decoders import (
    BiTransformerDecoder,
    TransformerDecoderConfig,
    add_sos_eos,
    reverse_pad_list,
)

WENET_REF = os.environ.get("WENET_REF_DIR", "/tmp/wenet_ref")
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    "/data01/kilm/users/chiendb/models/asr/am/20210610_u2pp_conformer_exp_librispeech",
)


def _small_config(**overrides):
    base = dict(
        vocab_size=50,
        encoder_output_size=32,
        attention_heads=2,
        linear_units=64,
        num_blocks=2,
        r_num_blocks=2,
        sos_id=49,
        eos_id=49,
        reverse_weight=0.3,
    )
    base.update(overrides)
    return TransformerDecoderConfig(**base)


def _import_wenet_ref():
    """Import the upstream WeNet decoder from the reference source tree."""
    if not os.path.exists(os.path.join(WENET_REF, "transformer", "decoder.py")):
        pytest.skip(f"WeNet reference sources not found at {WENET_REF} (set WENET_REF_DIR)")
    import typeguard

    # WeNet v2.0.1 targets the typeguard 2.x API; shim it for 3.x/4.x installs.
    if not hasattr(typeguard, "check_argument_types"):
        typeguard.check_argument_types = lambda *a, **k: True
    for name in ("wenet", "wenet.transformer", "wenet.utils"):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        sub = "" if name == "wenet" else "/" + name.split(".")[1]
        mod.__path__ = [WENET_REF + sub]
        sys.modules[name] = mod
    from wenet.transformer.decoder import BiTransformerDecoder as WenetBiDecoder
    from wenet.utils.common import add_sos_eos as w_add_sos_eos
    from wenet.utils.common import reverse_pad_list as w_reverse_pad_list

    return WenetBiDecoder, w_add_sos_eos, w_reverse_pad_list


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


class TestInputHelpers:
    def test_add_sos_eos(self):
        ys = torch.tensor([[1, 2, 3, -1], [4, 5, 6, 7]])
        ys_in, ys_out = add_sos_eos(ys, sos=9, eos=8, ignore_id=-1)
        assert ys_in.tolist() == [[9, 1, 2, 3, 8], [9, 4, 5, 6, 7]]
        assert ys_out.tolist() == [[1, 2, 3, 8, -1], [4, 5, 6, 7, 8]]

    def test_add_sos_eos_empty_hyp(self):
        ys = torch.full((1, 2), -1, dtype=torch.long)
        ys_in, ys_out = add_sos_eos(ys, sos=9, eos=8, ignore_id=-1)
        assert ys_in.tolist() == [[9, 8, 8]]
        assert ys_out.tolist() == [[8, -1, -1]]

    def test_reverse_pad_list(self):
        ys = torch.tensor([[1, 2, 3], [4, 5, -1]])
        r = reverse_pad_list(ys, torch.tensor([3, 2]), -1)
        assert r.tolist() == [[3, 2, 1], [5, 4, -1]]

    def test_helpers_match_wenet(self):
        _, w_add, w_rev = _import_wenet_ref()
        torch.manual_seed(0)
        ys = torch.randint(1, 40, (4, 6))
        lens = torch.tensor([6, 3, 1, 5])
        for b, n in enumerate(lens.tolist()):
            ys[b, n:] = -1
        ours_in, ours_out = add_sos_eos(ys, 49, 48, -1)
        ref_in, ref_out = w_add(ys, 49, 48, -1)
        assert torch.equal(ours_in, ref_in)
        assert torch.equal(ours_out, ref_out)
        ours_r = reverse_pad_list(ys, lens, -1)
        ref_r = w_rev(ys, lens, -1.0).long()
        assert torch.equal(ours_r, ref_r)


# ---------------------------------------------------------------------------
# Decoder forward parity vs the upstream WeNet implementation
# ---------------------------------------------------------------------------


class TestWenetParity:
    def _random_inputs(self, cfg, B=3, T=17, Lmax=6, seed=0):
        torch.manual_seed(seed)
        memory = torch.randn(B, T, cfg.encoder_output_size)
        memory_lens = torch.tensor([T, T - 5, T - 9][:B])
        hyps = torch.randint(1, cfg.vocab_size - 2, (B, Lmax))
        lens = torch.tensor([Lmax, Lmax - 2, Lmax - 4][:B])
        for b, n in enumerate(lens.tolist()):
            hyps[b, n:] = -1
        return memory, memory_lens, hyps, lens

    def test_bitransformer_parity_random_weights(self):
        WenetBiDecoder, _, _ = _import_wenet_ref()
        cfg = _small_config()
        torch.manual_seed(1)
        ours = BiTransformerDecoder(cfg).eval()
        ref = WenetBiDecoder(
            vocab_size=cfg.vocab_size,
            encoder_output_size=cfg.encoder_output_size,
            attention_heads=cfg.attention_heads,
            linear_units=cfg.linear_units,
            num_blocks=cfg.num_blocks,
            r_num_blocks=cfg.r_num_blocks,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            self_attention_dropout_rate=0.0,
            src_attention_dropout_rate=0.0,
        ).eval()
        # Same key layout by construction: copy our random weights into the ref.
        ref.load_state_dict(ours.state_dict(), strict=True)

        memory, memory_lens, hyps, lens = self._random_inputs(cfg)
        ys_in, _ = add_sos_eos(hyps, cfg.sos_id, cfg.eos_id, -1)
        r_ys_in, _ = add_sos_eos(reverse_pad_list(hyps, lens, -1), cfg.sos_id, cfg.eos_id, -1)

        memory_mask = (
            torch.arange(memory.size(1)).unsqueeze(0) < memory_lens.unsqueeze(1)
        ).unsqueeze(1)
        with torch.no_grad():
            l_ours, r_ours = ours(memory, memory_lens, ys_in, lens + 1, r_ys_in)
            l_ref, r_ref, _ = ref(memory, memory_mask, ys_in, lens + 1, r_ys_in, 0.3)

        valid = torch.arange(ys_in.size(1)).unsqueeze(0) < (lens + 1).unsqueeze(1)
        assert (l_ours - l_ref).abs()[valid].max().item() < 2e-5
        assert (r_ours - r_ref).abs()[valid].max().item() < 2e-5

    def test_real_checkpoint_parity(self):
        """Bit-level oracle on the real U2++ decoder weights (fp32, CPU)."""
        WenetBiDecoder, _, _ = _import_wenet_ref()
        ckpt = os.path.join(CKPT_DIR, "final.pt")
        if not os.path.exists(ckpt):
            pytest.skip(f"no U2++ checkpoint at {CKPT_DIR} (set CKPT_DIR)")
        sd = torch.load(ckpt, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        dsd = {k[len("decoder.") :]: v for k, v in sd.items() if k.startswith("decoder.")}

        cfg = TransformerDecoderConfig(
            vocab_size=5002,
            encoder_output_size=256,
            attention_heads=4,
            linear_units=2048,
            num_blocks=3,
            r_num_blocks=3,
            sos_id=5001,
            eos_id=5001,
            reverse_weight=0.3,
        )
        ours = BiTransformerDecoder(cfg).eval()
        ours.load_state_dict(dsd, strict=True)
        ref = WenetBiDecoder(
            vocab_size=5002,
            encoder_output_size=256,
            attention_heads=4,
            linear_units=2048,
            num_blocks=3,
            r_num_blocks=3,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            self_attention_dropout_rate=0.0,
            src_attention_dropout_rate=0.0,
        ).eval()
        ref.load_state_dict(dsd, strict=True)

        torch.manual_seed(42)
        B, T = 2, 41
        memory = torch.randn(B, T, 256)
        memory_lens = torch.tensor([T, 28])
        hyps = torch.randint(3, 5000, (B, 7))
        hyps[0, 5:] = -1
        lens = torch.tensor([5, 7])
        ys_in, _ = add_sos_eos(hyps, 5001, 5001, -1)
        r_ys_in, _ = add_sos_eos(reverse_pad_list(hyps, lens, -1), 5001, 5001, -1)
        memory_mask = (torch.arange(T).unsqueeze(0) < memory_lens.unsqueeze(1)).unsqueeze(1)

        with torch.no_grad():
            l_ours, r_ours = ours(memory, memory_lens, ys_in, lens + 1, r_ys_in)
            l_ref, r_ref, _ = ref(memory, memory_mask, ys_in, lens + 1, r_ys_in, 0.3)
        valid = torch.arange(ys_in.size(1)).unsqueeze(0) < (lens + 1).unsqueeze(1)
        assert (l_ours - l_ref).abs()[valid].max().item() < 5e-5
        assert (r_ours - r_ref).abs()[valid].max().item() < 5e-5


# ---------------------------------------------------------------------------
# Incremental (AR) step vs teacher-forced forward
# ---------------------------------------------------------------------------


class TestForwardOneStep:
    def test_incremental_matches_teacher_forced(self):
        cfg = _small_config(r_num_blocks=0)
        torch.manual_seed(2)
        dec = BiTransformerDecoder(cfg).eval().left_decoder
        B, T, L = 2, 11, 5
        memory = torch.randn(B, T, cfg.encoder_output_size)
        memory_lens = torch.tensor([T, T - 4])
        tokens = torch.randint(1, cfg.vocab_size - 2, (B, L))
        ys_in = torch.cat([torch.full((B, 1), cfg.sos_id), tokens], dim=1)
        lens = torch.full((B,), L + 1, dtype=torch.long)

        with torch.no_grad():
            full = dec(memory, memory_lens, ys_in, lens)  # (B, L+1, V)
            caches = None
            step_logits = []
            for t in range(L + 1):
                out, caches = dec.forward_one_step(
                    memory, memory_lens, ys_in[:, t], offset=t, caches=caches
                )
                step_logits.append(out)
        stepped = torch.stack(step_logits, dim=1)
        assert (stepped - full).abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# ConformerModel decoder-branch weight loading
# ---------------------------------------------------------------------------


class TestConformerDecoderLoading:
    def _model_config(self, with_decoder=True):
        from oasr.models.conformer import ConformerEncoderConfig, ConformerModelConfig

        enc = ConformerEncoderConfig(
            input_size=80,
            output_size=32,
            num_blocks=1,
            attention_heads=2,
            linear_units=64,
            cnn_module_kernel=7,
            causal=True,
        )
        return ConformerModelConfig(
            encoder=enc,
            vocab_size=56,  # 8-aligned CTC head over a raw vocab of 50
            decoder=_small_config() if with_decoder else None,
        )

    def test_bitransformer_keys_map_one_to_one(self):
        from oasr.models.conformer.model import ConformerModel

        torch.manual_seed(3)
        src = ConformerModel(self._model_config())
        sd = {k: v for k, v in src.state_dict().items()}
        dst = ConformerModel(self._model_config())
        report = dst.load_weights(sd)
        assert not [k for k in report.dropped if k.startswith("decoder.")]
        for k in ("decoder.left_decoder.embed.0.weight", "decoder.right_decoder.output_layer.bias"):
            assert torch.equal(dst.state_dict()[k], sd[k])
        assert sorted(model_caps := dst.capabilities) == ["ctc", "ctc_aed_rescoring"]
        assert dst.default_decode_type == "ctc"
        assert dst.decode_type == "ctc"  # alias must not flip to the AED branch

    def test_plain_transformer_keys_remap_to_left_decoder(self):
        """WeNet ``decoder: transformer`` checkpoints key layers as
        ``decoder.decoders.*`` — they must land in ``left_decoder``."""
        from oasr.models.conformer.model import ConformerModel

        torch.manual_seed(4)
        src = ConformerModel(self._model_config())
        sd = {}
        for k, v in src.state_dict().items():
            if k.startswith("decoder.right_decoder."):
                continue  # a plain transformer has no reverse branch
            sd[k.replace("decoder.left_decoder.", "decoder.")] = v

        cfg = self._model_config()
        cfg.decoder.r_num_blocks = 0
        dst = ConformerModel(cfg)
        report = dst.load_weights(sd)
        assert not [k for k in report.dropped if k.startswith("decoder.")]
        assert torch.equal(
            dst.state_dict()["decoder.left_decoder.embed.0.weight"],
            sd["decoder.embed.0.weight"],
        )

    def test_no_decoder_config_drops_decoder_keys(self):
        from oasr.models.conformer.model import ConformerModel

        src = ConformerModel(self._model_config())
        sd = {k: v for k, v in src.state_dict().items()}
        dst = ConformerModel(self._model_config(with_decoder=False))
        report = dst.load_weights(sd)
        dropped_dec = [k for k in report.dropped if k.startswith("decoder.")]
        assert dropped_dec, "decoder.* must be reported dropped on a CTC-only config"
        assert dst.capabilities == frozenset({"ctc"})

    def test_pos_enc_buffer_not_serialized(self):
        cfg = _small_config()
        dec = BiTransformerDecoder(cfg)
        assert not [k for k in dec.state_dict() if k.endswith("pe")]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
