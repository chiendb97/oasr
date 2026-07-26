#!/usr/bin/env python3
"""Conformance tests over the model + capability registries (keystone K1).

Pure CPU, no checkpoints, no CUDA: everything here is answered from registered
metadata, so it runs everywhere and catches the class of drift that used to reach
production — a checkpoint advertising a decode family it cannot actually serve, or
a capability with no strategy behind it.

The complement to this is ``require_capability``, exercised on real model objects
by the per-family suites (``test_whisper.py``, ``test_speech_llm.py``, ...).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from oasr.engine.decode.base import _REGISTRY as DECODE_REGISTRY
from oasr.engine.decode.base import _strategy_name
from oasr.models.interfaces import CAPABILITIES, missing_members, require_capability
from oasr.models.registry import get_model_entry, list_models

# Every capability any registered model may advertise.
KNOWN_CAPABILITIES = frozenset(CAPABILITIES)


def _ctc_config():
    """Minimal stand-in for ``EngineConfig`` when resolving a strategy name."""
    return SimpleNamespace(decoder_type="ctc_cuda")


class TestCapabilityTable:
    def test_every_capability_has_a_strategy(self):
        """A capability nobody can decode is a dead advertisement."""
        for cap in KNOWN_CAPABILITIES:
            name = _strategy_name(cap, _ctc_config())
            assert name in DECODE_REGISTRY, f"capability {cap!r} resolves to unregistered {name!r}"

    def test_every_strategy_maps_back_to_a_capability(self):
        """And a strategy no capability names can never be selected."""
        reachable = {_strategy_name(cap, _ctc_config()) for cap in KNOWN_CAPABILITIES}
        # CTC is the one family that fans out by kernel (``decoder_type``).
        reachable |= {"ctc_cuda", "ctc_wfst"}
        assert set(DECODE_REGISTRY) <= reachable, (
            f"strategies unreachable from any capability: "
            f"{sorted(set(DECODE_REGISTRY) - reachable)}"
        )

    def test_specs_are_self_consistent(self):
        for cap, spec in CAPABILITIES.items():
            assert spec.capability == cap, "spec key and capability name must agree"
            assert spec.requires, f"{cap!r} declares no required members"
            assert spec.why, f"{cap!r} has no explanation to put in the error"
            assert len(set(spec.requires)) == len(spec.requires), f"{cap!r} repeats a member"


@pytest.mark.parametrize("arch", list_models())
class TestRegisteredModels:
    """Every registered architecture, checked from its declared metadata."""

    def test_capabilities_are_known_names(self, arch):
        model_cls = get_model_entry(arch).model_cls
        caps = self._declared(model_cls, "capabilities")
        if caps is None:
            pytest.skip(f"{arch} derives capabilities from a live instance")
        unknown = set(caps) - KNOWN_CAPABILITIES
        assert not unknown, (
            f"{arch} advertises capabilities with no entry in "
            f"oasr.models.interfaces.CAPABILITIES: {sorted(unknown)}"
        )

    def test_default_decode_type_is_a_capability(self, arch):
        model_cls = get_model_entry(arch).model_cls
        caps = self._declared(model_cls, "capabilities")
        default = self._declared(model_cls, "default_decode_type")
        if caps is None or default is None:
            pytest.skip(f"{arch} derives its decode metadata from a live instance")
        assert default in caps, f"{arch}: default_decode_type={default!r} not in {sorted(caps)}"

    def test_config_class_round_trips(self, arch):
        """``save_native`` writes ``asdict``; ``load_native`` calls ``from_dict``."""
        import dataclasses

        entry = get_model_entry(arch)
        cfg_cls = entry.config_cls
        assert dataclasses.is_dataclass(cfg_cls), f"{arch}: config must be a dataclass (asdict)"
        assert hasattr(cfg_cls, "from_dict"), f"{arch}: config needs from_dict (native load)"
        cfg = cfg_cls()
        restored = cfg_cls.from_dict(dataclasses.asdict(cfg))
        assert type(restored) is cfg_cls

    def test_converter_surface(self, arch):
        conv = get_model_entry(arch).converter
        for member in (
            "detect",
            "build_config",
            "load_state_dict",
            "convert",
            "build_aux",
            "build_tokenizer_spec",
            "build_feature_spec",
            "build_decoding_defaults",
        ):
            assert hasattr(conv, member), f"{arch}: converter is missing {member}()"
        for member in ("detect_specificity", "expected_unused_prefixes", "capability_drop_hints"):
            assert hasattr(conv, member), f"{arch}: converter is missing {member}"

    def test_converter_declares_its_own_identity(self, arch):
        """Class attributes, not literals buried in a hand-written ``convert()``.

        ``BaseCheckpointConverter.convert`` reads these, so a subclass that
        inherits another converter (the transducer one extends icefall's) and
        forgets to re-declare them would silently emit a bundle labelled with
        its parent's architecture.
        """
        conv = get_model_entry(arch).converter
        assert (
            conv.architecture == arch
        ), f"{arch}: converter reports architecture={conv.architecture!r}"
        assert conv.source_format, f"{arch}: converter declares no source_format"
        assert conv.default_checkpoint_name, f"{arch}: no default_checkpoint_name"
        assert conv.default_decode_type, f"{arch}: no default_decode_type"

    @staticmethod
    def _declared(model_cls, name):
        """Read a class-level ``property`` without building the model.

        Architectures that override the property with a constant expose it
        through ``fget`` on the class; ones that derive it from live submodules
        (the base implementation reads ``self.head``) cannot be answered
        statically, and those tests skip.
        """
        attr = getattr(model_cls, name, None)
        if not isinstance(attr, property):
            return attr if not callable(attr) else None
        try:
            return attr.fget(None)  # type: ignore[misc]
        except Exception:
            return None


#: Smallest config kwargs that still build each architecture on CPU.  Keeping the
#: dims tiny is what lets this run as a unit test rather than an integration one.
TINY_CONFIGS = {
    "conformer": dict(vocab_size=64),
    "zipformer": dict(vocab_size=64),
    "whisper": dict(
        encoder_layers=1,
        decoder_layers=1,
        d_model=64,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        vocab_size=64,
        max_source_positions=64,
        max_target_positions=16,
    ),
    "paraformer": dict(
        encoder_num_blocks=1,
        decoder_num_blocks=1,
        encoder_output_size=64,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_linear_units=64,
        decoder_linear_units=64,
        vocab_size=64,
        predictor_idim=64,
        decoder_att_layer_num=1,
    ),
    "transducer": dict(vocab_size=64, decoder_dim=64, joiner_dim=64),
    "speech_llm": dict(
        vocab_size=64,
        text_num_hidden_layers=1,
        text_hidden_size=64,
        text_num_attention_heads=2,
        text_num_key_value_heads=1,
        text_intermediate_size=64,
        text_max_position_embeddings=64,
        audio_encoder_layers=1,
        audio_d_model=64,
        audio_encoder_attention_heads=2,
        audio_encoder_ffn_dim=64,
        audio_max_source_positions=64,
    ),
}


def _tiny_model(arch: str):
    """Build ``arch`` at its smallest working size on CPU, or skip."""
    kwargs = TINY_CONFIGS.get(arch)
    if kwargs is None:
        pytest.skip(f"no tiny config for {arch}; add one to TINY_CONFIGS")
    entry = get_model_entry(arch)
    return entry.model_cls.from_config(entry.config_cls(**kwargs))


@pytest.mark.parametrize("arch", list_models())
def test_advertised_capabilities_are_backed_by_the_real_surface(arch):
    """Build each architecture tiny on CPU and check the table against reality.

    This is the test that makes :data:`CAPABILITIES` trustworthy: the entries are
    dotted paths written by hand from reading each strategy, so a typo or a
    renamed member would otherwise turn into a spurious rejection of a working
    checkpoint — and the suites that would catch it need a GPU and real weights.
    Here every advertised capability of every registered architecture must have
    **zero** missing members.
    """
    model = _tiny_model(arch)
    for capability in sorted(model.capabilities):
        assert missing_members(model, capability) == [], (
            f"{arch} advertises {capability!r} but its surface is incomplete — "
            f"either the model or CAPABILITIES[{capability!r}] is wrong"
        )
        require_capability(model, capability)


class TestPagedKvGeometryIsOptIn:
    """``n_kv_head`` / ``head_dim`` describe the *engine's* paged-KV layout (N4).

    They used to be abstract on ``BaseEncoder``, so Whisper, the Paraformer SANM
    encoder and the Qwen2-Audio tower each implemented two properties purely to
    satisfy the ABC — ceremony that reads to the next author as a requirement for
    every encoder.  These tests pin the replacement contract in both directions.
    """

    @pytest.mark.parametrize("arch", list_models())
    def test_cache_spec_is_none_exactly_for_offline_only_encoders(self, arch):
        model = _tiny_model(arch)
        spec = model.cache_spec
        if model.encoder.streaming_kind == "none":
            assert spec is None, (
                f"{arch} is offline-only but reports a streaming cache spec; the "
                "engine would allocate a paged pool nothing can read"
            )
        else:
            assert spec is not None
            assert spec.num_layers > 0 and spec.n_kv_head > 0 and spec.head_dim > 0

    @pytest.mark.parametrize("arch", list_models())
    def test_streaming_encoders_declare_the_geometry_offline_ones_need_not(self, arch):
        encoder = _tiny_model(arch).encoder
        if encoder.streaming_kind == "none":
            # Not "must raise" — an encoder is free to expose the dims.  The point is
            # that it is not *forced* to, and that the default failure is legible.
            try:
                encoder.n_kv_head
            except NotImplementedError as exc:
                assert "paged-KV layout" in str(exc)
        else:
            assert isinstance(encoder.n_kv_head, int)
            assert isinstance(encoder.head_dim, int)

    def test_the_default_explains_itself(self):
        """A bare encoder asked for paged geometry says why it has none."""
        model = _tiny_model("whisper")
        with pytest.raises(NotImplementedError, match=r"WhisperEncoder.*paged-KV layout"):
            model.encoder.n_kv_head
        with pytest.raises(NotImplementedError, match=r"WhisperEncoder.*paged-KV layout"):
            model.encoder.head_dim


def test_hybrid_conformer_backs_its_rescoring_capability():
    """The U2++ branch is conditional, so the default tiny conformer does not
    advertise ``ctc_aed_rescoring`` — build one that does and check its spec."""
    from oasr.models.conformer.config import ConformerModelConfig
    from oasr.models.conformer.model import ConformerModel
    from oasr.models.decoders.transformer_decoder import TransformerDecoderConfig

    cfg = ConformerModelConfig(
        vocab_size=64,
        decoder=TransformerDecoderConfig(
            vocab_size=64,
            encoder_output_size=256,
            attention_heads=2,
            linear_units=64,
            num_blocks=1,
            r_num_blocks=1,
            sos_id=63,
            eos_id=63,
            reverse_weight=0.3,
        ),
    )
    model = ConformerModel.from_config(cfg)
    assert "ctc_aed_rescoring" in model.capabilities
    assert model.default_decode_type == "ctc", "rescoring stays opt-in"
    assert missing_members(model, "ctc_aed_rescoring") == []


class TestRequireCapability:
    """The single runtime gate that replaced five bespoke ``hasattr`` gauntlets."""

    def test_none_model_is_rejected_like_any_other_gap(self):
        """The five gauntlets this replaced disagreed on ``None``: aed / llm /
        paraformer raised, transducer and CTC accepted it and failed later.  One
        rule now — no surface means rejected."""
        for capability in CAPABILITIES:
            with pytest.raises(ValueError, match="no model was supplied"):
                require_capability(None, capability)

    def test_unknown_capability_is_not_blocked(self):
        assert missing_members(SimpleNamespace(), "brand_new_family") == []
        require_capability(SimpleNamespace(), "brand_new_family")

    def test_missing_members_are_named(self):
        model = SimpleNamespace(encode_offline=lambda *a: None)  # no decoder, no config
        missing = missing_members(model, "aed")
        assert "decoder.prefill" in missing
        assert "config.sot_sequence" in missing
        assert "encode_offline" not in missing

    def test_error_names_the_capability_and_the_gap(self):
        model = SimpleNamespace(encode_offline=lambda *a: None)
        with pytest.raises(ValueError) as exc:
            require_capability(model, "aed", decode_method="aed")
        msg = str(exc.value)
        assert "decode_method='aed'" in msg
        assert "decoder.prefill" in msg
        assert "label-synchronous" in msg  # the spec's `why`

    def test_none_valued_slot_counts_as_missing(self):
        """``decoder`` is a bare annotation that reads back ``None`` when unset —
        as unusable as absent, and the old ``hasattr`` checks knew that."""
        model = SimpleNamespace(encode_offline=lambda *a: None, decoder=None, blank_id=0)
        assert "decoder.init_state" in missing_members(model, "transducer")

    def test_satisfied_surface_passes(self):
        decoder = SimpleNamespace(init_state=lambda *a: None)
        joiner = SimpleNamespace(encoder_proj=lambda *a: None, decoder_proj=lambda *a: None)
        model = SimpleNamespace(
            encode_offline=lambda *a: None, blank_id=0, decoder=decoder, joiner=joiner
        )
        assert missing_members(model, "transducer") == []
        require_capability(model, "transducer")
