# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Native-format config round-trip: ``asdict`` → JSON → ``from_dict`` → equal (N6).

``oasr.checkpoints.native`` writes a config with a generic ``dataclasses.asdict``,
so the write side is uniform.  The read side used to be six hand-written
``from_dict`` methods using four different spellings of "filter to known fields",
each of which fails on the *next* edit in a different way — a hardcoded name tuple
drops added fields, ``hasattr`` admits properties and misses defaultless fields,
and a config with ``Tuple`` fields that forgot the ad-hoc ``tuple(...)`` restore
returns lists (JSON has no tuples).

The generic reader (:func:`oasr.models.base.coerce_config`) is driven by the
declared field types.  These tests are what makes that trustworthy: for every
registered architecture, a **non-default** config must survive the exact trip the
native format performs — including tuple identity, which plain equality on a
dataclass does catch (``(1, 2) != [1, 2]``).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from oasr.models.base import coerce_config
from oasr.models.registry import get_model_entry, list_models


def _round_trip(cfg):
    """The exact trip ``save_native`` / ``load_native`` performs."""
    return type(cfg).from_dict(json.loads(json.dumps(dataclasses.asdict(cfg))))


#: Non-default values per architecture, chosen to exercise the field *kinds* that
#: the old hand-written readers got wrong: nested dataclasses, ``Tuple[int, ...]``,
#: ``List[Tuple[int, int]]``, and plain scalars that a hardcoded name list could
#: silently omit.
NON_DEFAULT = {
    "conformer": lambda m: {"vocab_size": 4711},
    "zipformer": lambda m: {"vocab_size": 4711},
    "whisper": lambda m: {
        "vocab_size": 4711,
        "forced_decoder_ids": [(1, 50259), (2, 50359), (3, 50363)],
        "suppress_tokens": [1, 2, 7],
        "begin_suppress_tokens": [220, 50257],
    },
    "paraformer": lambda m: {"vocab_size": 4711},
    "transducer": lambda m: {
        "vocab_size": 4711,
        "decoder_dim": 99,
        "joiner_dim": 77,
        "context_size": 3,
        "blank_id": 5,
    },
    "speech_llm": lambda m: {"vocab_size": 4711, "eos_token_ids": [1, 2, 3]},
    # ``supported_num_lookahead_tokens`` is a ``Tuple[int, ...]`` on the *nested*
    # encoder config — the exact combination (nested dataclass + tuple field) that
    # the hand-written readers got wrong, and JSON has no tuples.
    "nemotron": lambda m: _nemotron_overrides(),
}


def _nemotron_overrides():
    """Needs a nested-config instance, which a lambda over the *outer* class
    cannot build — the ``m`` argument is the config class, not its module."""
    from oasr.models.nemotron.config import NemotronEncoderConfig

    return {
        "vocab_size": 4711,
        "blank_token_id": 4710,
        "default_prompt_id": 7,
        "encoder": NemotronEncoderConfig(
            hidden_size=64,
            num_hidden_layers=3,
            supported_num_lookahead_tokens=(0, 5, 11),
        ),
    }


@pytest.mark.parametrize("arch", list_models())
def test_default_config_round_trips(arch):
    cls = get_model_entry(arch).config_cls
    cfg = cls()
    assert _round_trip(cfg) == cfg


@pytest.mark.parametrize("arch", list_models())
def test_non_default_config_round_trips(arch):
    cls = get_model_entry(arch).config_cls
    overrides = NON_DEFAULT.get(arch)
    if overrides is None:
        pytest.skip(f"no non-default sample for {arch}; add one to NON_DEFAULT")
    cfg = cls(**overrides(cls))
    restored = _round_trip(cfg)
    assert restored == cfg, f"{arch} lost or mistyped a field on native round-trip"


@pytest.mark.parametrize("arch", list_models())
def test_every_declared_field_survives(arch):
    """Field-by-field, so a failure names the field rather than the whole config.

    Plain ``==`` on the dataclass would catch a drop, but reports only "not equal";
    the whole point of the old bugs was that they were hard to attribute.
    """
    cls = get_model_entry(arch).config_cls
    overrides = NON_DEFAULT.get(arch)
    cfg = cls(**overrides(cls)) if overrides else cls()
    restored = _round_trip(cfg)
    for f in dataclasses.fields(cls):
        before, after = getattr(cfg, f.name), getattr(restored, f.name)
        assert after == before, f"{arch}.{f.name}: {before!r} → {after!r}"
        assert type(after) is type(before), (
            f"{arch}.{f.name} changed type on round-trip: "
            f"{type(before).__name__} → {type(after).__name__}"
        )


class TestTupleRestoration:
    """JSON has no tuples, so a ``Tuple`` field must be restored from its type."""

    def test_zipformer_tuple_fields_come_back_as_tuples(self):
        from oasr.models.zipformer.config import ZipformerEncoderConfig

        enc = ZipformerEncoderConfig()
        tuple_fields = [
            f.name
            for f in dataclasses.fields(ZipformerEncoderConfig)
            if isinstance(getattr(enc, f.name), tuple)
        ]
        assert tuple_fields, "expected Tuple fields on the Zipformer encoder config"

        restored = coerce_config(
            ZipformerEncoderConfig, json.loads(json.dumps(dataclasses.asdict(enc)))
        )
        for name in tuple_fields:
            assert isinstance(getattr(restored, name), tuple), name

    def test_list_of_tuples_restores_both_levels(self):
        """Whisper's ``forced_decoder_ids: List[Tuple[int, int]]`` — the case that
        needed a hand-written coercion in every config that had one."""
        from oasr.models.whisper.config import WhisperModelConfig

        cfg = WhisperModelConfig(vocab_size=64, forced_decoder_ids=[(1, 2), (3, 4)])
        restored = _round_trip(cfg)
        assert restored.forced_decoder_ids == [(1, 2), (3, 4)]
        assert all(isinstance(p, tuple) for p in restored.forced_decoder_ids)


class TestFlatAndPolymorphicOverrides:
    """The two field kinds a type annotation genuinely cannot describe."""

    def test_conformer_accepts_a_flat_encoder_dict(self):
        """WeNet-derived dicts put encoder hyperparameters at the top level."""
        from oasr.models.conformer.config import ConformerModelConfig

        cfg = ConformerModelConfig.from_dict({"vocab_size": 7, "output_size": 256})
        assert cfg.vocab_size == 7
        assert cfg.encoder.output_size == 256

    def test_zipformer_accepts_flat_icefall_args(self):
        from oasr.models.zipformer.config import ZipformerModelConfig

        cfg = ZipformerModelConfig.from_dict({"vocab_size": 7, "encoder_dim": [64, 96]})
        assert cfg.encoder.encoder_dim == (64, 96)

    @pytest.mark.parametrize(
        "encoder_type,expected",
        [("conformer", "ConformerEncoderConfig"), ("zipformer", "ZipformerEncoderConfig")],
    )
    def test_transducer_encoder_class_follows_encoder_type(self, encoder_type, expected):
        """``encoder: Any`` — the class is decided by a *sibling key*, so this is the
        one field in the tree that legitimately needs an override hook."""
        from oasr.models.transducer.config import TransducerModelConfig

        cfg = TransducerModelConfig.from_dict(
            {"vocab_size": 7, "encoder_type": encoder_type, "encoder": {}}
        )
        assert type(cfg.encoder).__name__ == expected

    def test_transducer_scalars_no_longer_come_from_a_hardcoded_list(self):
        """The old reader listed five field names by hand and omitted
        ``model_type`` / ``encoder_type``; every scalar now comes from the fields."""
        from oasr.models.transducer.config import TransducerModelConfig

        cfg = TransducerModelConfig(
            vocab_size=1,
            encoder_type="zipformer",
            decoder_dim=8,
            joiner_dim=9,
            context_size=4,
            blank_id=3,
        )
        # keep the encoder consistent with the declared type
        from oasr.models.zipformer.config import ZipformerEncoderConfig

        cfg.encoder = ZipformerEncoderConfig()
        restored = _round_trip(cfg)
        assert restored.encoder_type == "zipformer"
        assert restored.model_type == cfg.model_type
        assert (restored.decoder_dim, restored.joiner_dim) == (8, 9)
        assert (restored.context_size, restored.blank_id) == (4, 3)


class TestCoercionEdges:
    def test_unknown_keys_are_ignored(self):
        """Checkpoint configs legitimately carry keys we do not model."""
        from oasr.models.paraformer.config import ParaformerModelConfig

        cfg = ParaformerModelConfig.from_dict({"vocab_size": 5, "trained_by": "someone"})
        assert cfg.vocab_size == 5

    def test_optional_nested_dataclass_accepts_none(self):
        from oasr.models.conformer.config import ConformerModelConfig

        assert ConformerModelConfig.from_dict({"vocab_size": 5, "decoder": None}).decoder is None

    def test_optional_nested_dataclass_recurses_when_present(self):
        from oasr.models.conformer.config import ConformerModelConfig

        cfg = ConformerModelConfig.from_dict(
            {
                "vocab_size": 5,
                "decoder": {
                    "vocab_size": 5,
                    "encoder_output_size": 256,
                    "attention_heads": 2,
                    "linear_units": 64,
                    "num_blocks": 1,
                    "r_num_blocks": 1,
                    "sos_id": 4,
                    "eos_id": 4,
                    "reverse_weight": 0.3,
                },
            }
        )
        assert cfg.decoder is not None
        assert cfg.decoder.reverse_weight == 0.3
        assert type(cfg.decoder).__name__ == "TransformerDecoderConfig"
