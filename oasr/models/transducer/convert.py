# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Load icefall pruned-transducer checkpoints into OASR transducer models.

Targets the icefall ``zipformer`` recipe family (Zipformer2 encoder + stateless
predictor + joiner, e.g. ``pruned_transducer_stateless7`` / ``zipformer``):
the encoder config is shape-inferred exactly like the Zipformer CTC converter,
and the predictor/joiner dims come from the ``decoder.*`` / ``joiner.*``
weight shapes.

**Selection is explicit** (``detect()`` is False): icefall experiment dirs are
claimed by the Zipformer CTC converter, and hybrid checkpoints carry *both*
branches — pass ``architecture="transducer"`` to ``from_pretrained`` /
``oasr-convert`` to decode the transducer branch.  Capability-typed model
composition (multi-paradigm Phase 2, K1) is the planned proper fix for
hybrid selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

from ..zipformer.convert import IcefallConverter, _extract_state_dict, infer_encoder_config
from .config import TransducerModelConfig
from .model import TransducerModel

if TYPE_CHECKING:
    import torch

    from oasr.checkpoints import ConvertedCheckpoint

logger = logging.getLogger(__name__)


class IcefallTransducerConverter(IcefallConverter):
    """Checkpoint converter for icefall pruned-transducer experiment dirs.

    Inherits checkpoint/tokenizer-asset discovery and the feature spec from
    :class:`~oasr.models.zipformer.convert.IcefallConverter`; overrides
    detection (explicit-only), config inference (predictor + joiner dims), and
    the bundle's decoding defaults.
    """

    #: The pruned-RNNT ``simple_*_proj`` heads are training-only; a hybrid
    #: ``ctc_output.*`` branch is a named capability (load it with
    #: ``architecture="zipformer"`` instead).
    expected_unused_prefixes: Tuple[str, ...] = ("simple_am_proj", "simple_lm_proj")
    capability_drop_hints: Dict[str, str] = {
        "ctc_output.": (
            "the CTC head branch is not loaded under architecture='transducer'; "
            "load with architecture='zipformer' for CTC decode"
        ),
        "attention_decoder.": (
            "the attention-decoder branch is not loaded; AED decode lands with "
            "the multi-paradigm Phase 2"
        ),
    }

    def detect(self, ckpt_dir: Path) -> bool:
        """Never auto-detected — see the module docstring."""
        del ckpt_dir
        return False

    def build_config(self, ckpt_dir: Path) -> TransducerModelConfig:
        """Infer encoder + predictor + joiner hyperparameters from the checkpoint."""
        ckpt = self._find_ckpt(Path(ckpt_dir))
        if ckpt is None:
            raise FileNotFoundError(f"No icefall checkpoint (*.pt) found under {ckpt_dir}")
        import torch

        sd = _extract_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
        return self.config_from_state_dict(sd)

    @staticmethod
    def config_from_state_dict(sd: Mapping[str, "torch.Tensor"]) -> TransducerModelConfig:
        try:
            emb_w = sd["decoder.embedding.weight"]  # (vocab, decoder_dim)
            joiner_out_w = sd["joiner.output_linear.weight"]  # (vocab, joiner_dim)
        except KeyError as exc:
            raise KeyError(
                f"Checkpoint has no transducer branch (missing {exc}); "
                "is this a CTC-only checkpoint? Load with architecture='zipformer'."
            ) from None
        vocab, decoder_dim = int(emb_w.shape[0]), int(emb_w.shape[1])
        joiner_dim = int(joiner_out_w.shape[1])
        conv_w = sd.get("decoder.conv.weight")  # (decoder_dim, 1, context_size)
        context_size = int(conv_w.shape[-1]) if conv_w is not None else 1
        return TransducerModelConfig(
            encoder_type="zipformer",
            encoder=infer_encoder_config(sd),
            # No 8-alignment padding: the joiner is a plain torch linear, not
            # the OASR CTC GEMM kernel with its N % 8 requirement.
            vocab_size=vocab,
            decoder_dim=decoder_dim,
            joiner_dim=joiner_dim,
            context_size=context_size,
            blank_id=0,  # icefall convention: <blk> = 0
        )

    def convert(
        self, ckpt_dir: Path, checkpoint_name: str = "pretrained.pt", map_location: Any = "cpu"
    ) -> "ConvertedCheckpoint":
        """Full conversion: config + weights + tokenizer/feature/decoding specs."""
        from oasr.checkpoints import ConvertedCheckpoint, DecodingDefaults

        ckpt_dir = Path(ckpt_dir)
        state_dict = self.load_state_dict(ckpt_dir, checkpoint_name, map_location)
        return ConvertedCheckpoint(
            architecture="transducer",
            model_config=self.config_from_state_dict(state_dict),
            aux={},
            state_dict=state_dict,
            tokenizer=self.build_tokenizer_spec(ckpt_dir),
            features=self.build_feature_spec(ckpt_dir),
            decoding=DecodingDefaults(default_decode_type="transducer", blank_id=0),
            source_format="icefall",
        )


def load_icefall_transducer_checkpoint(
    ckpt_dir: str,
    checkpoint_name: str = "pretrained.pt",
    device: str = "cpu",
    dtype=None,
) -> Tuple[TransducerModel, TransducerModelConfig]:
    """Convenience loader for an icefall pruned-transducer checkpoint dir."""
    from ..registry import build_model_from_checkpoint

    return build_model_from_checkpoint(
        ckpt_dir,
        checkpoint_name=checkpoint_name,
        device=device,
        dtype=dtype,
        architecture="transducer",
    )
