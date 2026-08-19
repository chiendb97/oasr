# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared checkpoint-converter assembly and weight-loading fallbacks.

Subclasses declare format metadata and implement detection, configuration, and
state-dict loading. The registry also accepts independent protocol-compatible
converters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

import torch

from .registry import DETECT_ASSET_LAYOUT


class BaseCheckpointConverter:
    """Shared scaffolding for a format-specific checkpoint reader."""

    #: Registry name of the architecture this converter produces.
    architecture: ClassVar[str] = ""
    #: Provenance label recorded on the bundle (``"wenet"``, ``"icefall"``, ...).
    source_format: ClassVar[str] = ""
    #: Weights filename used when the caller does not name one.
    default_checkpoint_name: ClassVar[str] = "model.pt"
    #: Decode family the checkpoint runs unless the caller overrides it.
    default_decode_type: ClassVar[str] = "ctc"
    #: See :func:`oasr.models.registry.resolve_architecture`.
    detect_specificity: ClassVar[int] = DETECT_ASSET_LAYOUT
    #: Key prefixes whose dropping is expected and logged at DEBUG only.
    expected_unused_prefixes: ClassVar[Tuple[str, ...]] = ()
    #: Key prefix → the capability lost when those weights are dropped.
    capability_drop_hints: ClassVar[Mapping[str, str]] = {}

    # ------------------------------------------------------------------
    # Required hooks
    # ------------------------------------------------------------------

    def detect(self, ckpt_dir: Path) -> bool:
        """Positive markers only — never a guard against another format.

        See the protocol docstring: negative guards put one format's knowledge
        inside another's converter.  Rank with :attr:`detect_specificity`.
        """
        raise NotImplementedError

    def build_config(self, ckpt_dir: Path):
        raise NotImplementedError

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def build_aux(self, ckpt_dir: Path) -> Dict[str, Any]:
        """Auxiliary buffers passed to ``Model.from_config`` (e.g. CMVN)."""
        return {}

    def build_tokenizer_spec(self, ckpt_dir: Path):
        """``TokenizerSpec`` travelling with the bundle, or ``None``."""
        return None

    def build_feature_spec(self, ckpt_dir: Path):
        """``FeatureSpec`` travelling with the bundle, or ``None``."""
        return None

    def build_decoding_defaults(self, config, ckpt_dir: Path):
        """Decoding ids/family for the bundle.

        The default reports only :attr:`default_decode_type`; override to add
        the blank / sos / eos ids.  ``ckpt_dir`` is passed because some formats
        keep those ids in the on-disk config rather than the model config
        (WeNet's ``sos/eos`` is ``output_dim - 1`` from ``train.yaml``).
        """
        from oasr.checkpoints import DecodingDefaults

        return DecodingDefaults(default_decode_type=self.default_decode_type)

    def build_config_for_convert(self, ckpt_dir: Path, state_dict):
        """Build the model config during :meth:`convert`, given the weights.

        Defaults to :meth:`build_config`, which re-reads the directory.  Formats
        that *infer* the config from tensor shapes (icefall ships no config
        file) override this to use the already-loaded ``state_dict`` instead —
        otherwise ``convert`` deserialises the same checkpoint twice, which for
        a multi-GB file is the single slowest thing it does.
        """
        return self.build_config(ckpt_dir)

    # ------------------------------------------------------------------
    # Provided
    # ------------------------------------------------------------------

    def convert(
        self,
        ckpt_dir: Path,
        checkpoint_name: Optional[str] = None,
        map_location: Any = "cpu",
    ):
        """Assemble the complete :class:`~oasr.checkpoints.ConvertedCheckpoint`."""
        from oasr.checkpoints import ConvertedCheckpoint

        ckpt_dir = Path(ckpt_dir)
        name = checkpoint_name or self.default_checkpoint_name
        # Weights first, so a shape-inferring converter can reuse them.
        state_dict = self.load_state_dict(ckpt_dir, name, map_location)
        config = self.build_config_for_convert(ckpt_dir, state_dict)
        return ConvertedCheckpoint(
            architecture=self.architecture,
            model_config=config,
            aux=self.build_aux(ckpt_dir),
            state_dict=state_dict,
            tokenizer=self.build_tokenizer_spec(ckpt_dir),
            features=self.build_feature_spec(ckpt_dir),
            decoding=self.build_decoding_defaults(config, ckpt_dir),
            source_format=self.source_format,
        )

    # ------------------------------------------------------------------
    # Shared readers
    # ------------------------------------------------------------------

    @staticmethod
    def read_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return data

    @staticmethod
    def read_yaml(path: Path) -> Dict[str, Any]:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)
        return data

    @classmethod
    def load_hf_state_dict(
        cls, ckpt_dir: Path, map_location: Any = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """HuggingFace weight loading: sharded safetensors → single → ``.bin``.

        ``weights_only=True`` on the torch path: a checkpoint is untrusted input
        and an unpickle is arbitrary code execution.
        """
        # Each reader annotates its result rather than returning the call
        # directly: `safetensors` is an optional extra, so mypy sees `Any` where
        # it is absent and a typed dict where it is installed, and an error
        # count that depends on which extras the checking box happens to have is
        # a ratchet that fails in CI and passes locally.
        ckpt_dir = Path(ckpt_dir)
        index_path = ckpt_dir / "model.safetensors.index.json"
        if index_path.exists():
            from safetensors.torch import load_file

            shards = sorted(set(cls.read_json(index_path)["weight_map"].values()))
            sd: Dict[str, torch.Tensor] = {}
            for shard in shards:
                sd.update(load_file(str(ckpt_dir / shard), device=str(map_location)))
            return sd
        st_path = ckpt_dir / "model.safetensors"
        if st_path.exists():
            from safetensors.torch import load_file

            single: Dict[str, torch.Tensor] = load_file(str(st_path), device=str(map_location))
            return single
        bin_path = ckpt_dir / "pytorch_model.bin"
        if bin_path.exists():
            binary: Dict[str, torch.Tensor] = torch.load(
                str(bin_path), map_location=map_location, weights_only=True
            )
            return binary
        raise FileNotFoundError(
            f"no model.safetensors[.index.json] or pytorch_model.bin under {ckpt_dir}"
        )

    @staticmethod
    def whisper_logmel_spec(feature_dim: int):
        """The Whisper log-mel frontend spec, shared by Whisper and Qwen2-Audio.

        Identical for both but for the mel count (80 vs 128), which is why it
        lives here rather than being written twice.
        """
        from oasr.features import FeatureSpec

        return FeatureSpec(
            kind="whisper_logmel",
            sample_rate=16000,
            feature_dim=int(feature_dim),
            frame_length_ms=25.0,  # n_fft 400 @ 16 kHz
            frame_shift_ms=10.0,  # hop 160
            dither=0.0,
            audio_scale=1.0,
        )


__all__ = ["BaseCheckpointConverter"]
