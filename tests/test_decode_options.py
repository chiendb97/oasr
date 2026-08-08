# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-family decode options (H4) and the per-request option table (S9).

The property under test is the extension contract: a decode family declares its
own knobs and nothing in the engine core has to know about them.  ``EngineConfig``
keeps the old flat fields as deprecated aliases, so both spellings must resolve
to the same value, with the new one winning when both are given.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from oasr.engine.decode.options import (
    build_options,
    coerce_option_value,
    describe_options,
    option,
    option_factory,
    parse_decode_options,
)


@dataclass(frozen=True)
class _Opts:
    width: int = option(4, legacy="legacy_width", doc="how wide")
    ratio: float = option(0.5, doc="a ratio")
    name: str = option("x")
    flag: bool = option(False)
    sub: list = option_factory(list, doc="lazily built")

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width!r}")


def _cfg(**kw):
    kw.setdefault("decode_options", {})
    return SimpleNamespace(**kw)


class TestResolutionOrder:
    def test_defaults_when_nothing_is_set(self):
        o = build_options(_Opts, _cfg())
        assert (o.width, o.ratio, o.name, o.flag, o.sub) == (4, 0.5, "x", False, [])

    def test_legacy_field_is_honoured(self):
        assert build_options(_Opts, _cfg(legacy_width=9)).width == 9

    def test_decode_options_beats_the_legacy_alias(self):
        cfg = _cfg(legacy_width=9, decode_options={"width": 11})
        assert build_options(_Opts, cfg).width == 11

    def test_an_absent_legacy_field_is_not_an_error(self):
        """A family may declare an alias for a field the config never had."""
        assert build_options(_Opts, _cfg()).width == 4

    def test_the_options_dataclass_validates(self):
        with pytest.raises(ValueError, match="width must be >= 1"):
            build_options(_Opts, _cfg(decode_options={"width": 0}))

    def test_factory_default_is_per_instance(self):
        a, b = build_options(_Opts, _cfg()), build_options(_Opts, _cfg())
        a.sub.append(1)
        assert b.sub == [], "factory default leaked between instances"


class TestUnknownKeysAreRejected:
    """A misspelled knob must fail, not be ignored — the S9 failure mode."""

    def test_unknown_key_names_the_valid_ones(self):
        with pytest.raises(ValueError, match=r"unknown decode_options \['wdith'\]"):
            build_options(_Opts, _cfg(decode_options={"wdith": 3}))

    def test_options_for_a_family_that_declares_none(self):
        assert build_options(None, _cfg()) is None
        with pytest.raises(ValueError, match="declares no options"):
            build_options(None, _cfg(decode_options={"anything": 1}))


class TestStringValuesAreTyped:
    """``--decode-option k=v`` can only carry strings; the default types them.

    The serving crate deliberately forwards raw strings — typing them there
    would mean it carrying a copy of every family's option table.
    """

    @pytest.mark.parametrize(
        "key,raw,want",
        [
            ("width", "7", 7),
            ("ratio", "0.25", 0.25),
            ("flag", "true", True),
            ("flag", "0", False),
            ("name", "hello", "hello"),
        ],
    )
    def test_typed_from_the_declared_default(self, key, raw, want):
        o = build_options(_Opts, _cfg(decode_options={key: raw}))
        got = getattr(o, key)
        assert got == want and type(got) is type(want)

    def test_a_bad_value_names_the_option(self):
        with pytest.raises(ValueError, match="decode option flag="):
            build_options(_Opts, _cfg(decode_options={"flag": "maybe"}))

    def test_coerce_rejects_a_non_numeric_int(self):
        with pytest.raises(ValueError):
            coerce_option_value("abc", 1)


class TestCliParsing:
    def test_pairs_are_typed_and_validated(self):
        got = parse_decode_options(["width=8", "ratio=0.1"], _Opts)
        assert got == {"width": 8, "ratio": 0.1}

    def test_missing_equals_is_rejected(self):
        with pytest.raises(ValueError, match="expects k=v"):
            parse_decode_options(["width"], _Opts)

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="unknown decode option"):
            parse_decode_options(["nope=1"], _Opts)

    def test_describe_surfaces_defaults_and_docs(self):
        by_name = {d["name"]: d for d in describe_options(_Opts)}
        assert by_name["width"]["default"] == 4
        assert by_name["width"]["doc"] == "how wide"
        assert by_name["width"]["legacy"] == "legacy_width"


class TestRegisteredFamilies:
    """Every registered strategy's options must resolve from a bare config."""

    def _families(self):
        import oasr.engine.decode  # noqa: F401  (registers the built-ins)
        from oasr.engine.decode.base import _REGISTRY

        return sorted(_REGISTRY.items())

    def test_every_family_declares_resolvable_options(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="x")
        for name, cls in self._families():
            opts = build_options(cls.options_cls, cfg)
            assert (opts is None) == (cls.options_cls is None), name

    @pytest.mark.parametrize(
        "family,key,value,attr",
        [
            ("ctc_aed_rescoring", "ctc_weight", 0.3, "ctc_weight"),
            ("ctc_aed_rescoring", "reverse_weight", 0.7, "reverse_weight"),
            ("transducer", "max_sym_per_frame", 3, "max_sym_per_frame"),
            ("llm", "prompt", "say it", "prompt"),
            ("aed", "max_new_tokens", 64, "max_new_tokens"),
            ("ctc_wfst", "fst_path", "/tmp/HLG.pt", "fst_path"),
        ],
    )
    def test_real_family_knobs_round_trip(self, family, key, value, attr):
        from oasr.engine.config import EngineConfig
        from oasr.engine.decode.base import _REGISTRY

        cfg = EngineConfig(ckpt_dir="x", decode_options={key: value})
        opts = build_options(_REGISTRY[family].options_cls, cfg)
        assert getattr(opts, attr) == value

    @pytest.mark.parametrize(
        "family,legacy,value,attr",
        [
            ("ctc_aed_rescoring", "rescoring_ctc_weight", 0.25, "ctc_weight"),
            ("transducer", "transducer_max_sym_per_frame", 2, "max_sym_per_frame"),
            ("llm", "llm_prompt", "legacy prompt", "prompt"),
            ("aed", "max_new_tokens", 32, "max_new_tokens"),
        ],
    )
    def test_deprecated_flat_fields_still_work(self, family, legacy, value, attr):
        """The old spelling is public API and every oasr-server flag maps to it."""
        from oasr.engine.config import EngineConfig
        from oasr.engine.decode.base import _REGISTRY

        cfg = EngineConfig(ckpt_dir="x", **{legacy: value})
        opts = build_options(_REGISTRY[family].options_cls, cfg)
        assert getattr(opts, attr) == value

    def test_ctc_beam_configs_are_no_longer_built_for_every_engine(self):
        """The §3.2 leak: a Whisper engine used to construct both CTC configs.

        They are family options now, materialised by the CTC strategies'
        factories, so a config that never builds a CTC strategy never builds
        them.
        """
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="x")
        assert cfg.ctc_decoder_config is None
        assert cfg.wfst_decoder_config is None

        from oasr.engine.decode.base import _REGISTRY

        built = build_options(_REGISTRY["ctc_cuda"].options_cls, cfg)
        assert built.decoder_config is not None, "the CTC family must still get one"


class TestPerRequestOptionTable:
    """S9: one source of truth for the option keys crossing PyO3."""

    def test_coerce_uses_the_dataclass_fields(self):
        from oasr.engine.request import DecodingOptions

        assert DecodingOptions.option_keys() == (
            "n_best",
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
            "prompt",
            "task",
            "language",
        )

    def test_every_field_survives_a_dict_round_trip(self):
        from oasr.engine.request import DecodingOptions

        sent = {
            "n_best": 3,
            "max_new_tokens": 10,
            "temperature": 0.5,
            "top_k": 5,
            "top_p": 0.9,
            "prompt": "hi",
            "task": "translate",
            "language": "fr",
        }
        got = DecodingOptions.coerce(sent)
        for k, v in sent.items():
            assert getattr(got, k) == v, k

    def test_matching_wire_keys_passes(self):
        from oasr.engine.request import DecodingOptions

        DecodingOptions.assert_matches_wire_keys(DecodingOptions.option_keys())

    @pytest.mark.parametrize(
        "keys,missing_side",
        [
            (("n_best",), "only in Python"),
            (
                ("n_best", "max_new_tokens", "temperature", "top_k", "top_p", "prompt", "x"),
                "only in Rust",
            ),
        ],
    )
    def test_drift_is_caught_and_attributed(self, keys, missing_side):
        from oasr.engine.request import DecodingOptions

        with pytest.raises(ValueError, match=missing_side):
            DecodingOptions.assert_matches_wire_keys(keys)
