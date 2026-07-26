# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Load icefall Zipformer checkpoints into OASR.

icefall checkpoints store only the model state-dict (the architecture config
comes from CLI args), so the config defaults to the LibriSpeech "M" recipe (see
:class:`ZipformerModelConfig`); the CTC vocab is read from the checkpoint when
possible.  The architecture-specific key remapping lives in
:meth:`ZipformerModel.load_weights`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Mapping, Optional, Tuple

import torch

from ..converter import BaseCheckpointConverter
from ..registry import DETECT_ASSET_LAYOUT
from .config import ZipformerEncoderConfig, ZipformerModelConfig
from .model import ZipformerModel

if TYPE_CHECKING:
    from oasr.checkpoints import ConvertedCheckpoint

logger = logging.getLogger(__name__)

_NAMED_CANDIDATES = ("pretrained.pt", "model.pt", "checkpoint.pt", "cpu_jit.pt")
# icefall recipes universally use pos_head_dim=4; this disambiguates the
# (num_heads, *_head_dim) factorization, which is otherwise degenerate in the
# weight shapes.
_POS_HEAD_DIM = 4


def infer_encoder_config(sd: Mapping[str, torch.Tensor]) -> ZipformerEncoderConfig:
    """Infer a :class:`ZipformerEncoderConfig` from icefall checkpoint shapes.

    Recovers per-stack ``downsampling_factor`` / ``encoder_dim`` /
    ``num_encoder_layers`` / ``num_heads`` / ``query_head_dim`` /
    ``value_head_dim`` / ``feedforward_dim`` / ``cnn_module_kernel`` plus
    ``pos_dim`` from the parameter shapes, assuming ``pos_head_dim == 4``.
    ``feature_dim`` is not uniquely recoverable and defaults to 80.
    """
    stacks = sorted(
        {int(m.group(1)) for k in sd if (m := re.match(r"encoder\.encoders\.(\d+)\.", k))}
    )
    ds, edim, nlayers, nheads, qhd, vhd, ffd, cnnk = [], [], [], [], [], [], [], []
    pos_dim = None
    for i in stacks:
        pre = f"encoder.encoders.{i}."
        if (pre + "downsample.bias") in sd:
            ds.append(sd[pre + "downsample.bias"].shape[0])
            lpre = pre + "encoder.layers."
        else:
            ds.append(1)
            lpre = pre + "layers."
        nlayers.append(
            len({int(m.group(1)) for k in sd if (m := re.match(re.escape(lpre) + r"(\d+)\.", k))})
        )
        l0 = lpre + "0."
        edim.append(sd[l0 + "norm.bias"].shape[0])
        ffd.append(sd[l0 + "feed_forward2.in_proj.weight"].shape[0])
        cnnk.append(sd[l0 + "conv_module1.depthwise_conv.weight"].shape[-1])
        lin_pos = sd[l0 + "self_attn_weights.linear_pos.weight"].shape  # (H*pos_head_dim, pos_dim)
        in_proj = sd[l0 + "self_attn_weights.in_proj.weight"].shape[0]  # (2*qhd+pos_head_dim)*H
        v_in = sd[l0 + "self_attn1.in_proj.weight"].shape[0]  # H*vhd
        H = lin_pos[0] // _POS_HEAD_DIM
        nheads.append(H)
        qhd.append((in_proj // H - _POS_HEAD_DIM) // 2)
        vhd.append(v_in // H)
        pos_dim = lin_pos[1]
    # Single-value tuples for fields icefall keeps constant across stacks.
    qhd_t = (qhd[0],) if len(set(qhd)) == 1 else tuple(qhd)
    vhd_t = (vhd[0],) if len(set(vhd)) == 1 else tuple(vhd)
    return ZipformerEncoderConfig(
        feature_dim=80,
        downsampling_factor=tuple(ds),
        encoder_dim=tuple(edim),
        num_encoder_layers=tuple(nlayers),
        num_heads=tuple(nheads),
        query_head_dim=qhd_t,
        pos_head_dim=(_POS_HEAD_DIM,),
        value_head_dim=vhd_t,
        feedforward_dim=tuple(ffd),
        cnn_module_kernel=tuple(cnnk),
        pos_dim=int(pos_dim),
        causal=False,
    )


def _extract_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    """icefall checkpoints are sometimes ``{'model': sd, ...}`` and sometimes raw ``sd``."""
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


class IcefallConverter(BaseCheckpointConverter):
    """Checkpoint converter for icefall Zipformer experiment directories.

    Implements the :class:`~oasr.models.registry.CheckpointConverter` protocol.
    """

    #: Weight-drop accounting (read by the registry after ``load_weights``):
    #: the pruned-RNNT ``simple_*_proj`` heads are training-only, so dropping
    #: them is expected; the transducer / attention-decoder branches are named
    #: capability losses.
    expected_unused_prefixes: Tuple[str, ...] = ("simple_am_proj", "simple_lm_proj")
    capability_drop_hints: Dict[str, str] = {
        "decoder.": (
            "the transducer predictor branch is not loaded; transducer decode "
            "lands with the multi-paradigm Phase 1"
        ),
        "joiner.": (
            "the transducer joiner branch is not loaded; transducer decode "
            "lands with the multi-paradigm Phase 1"
        ),
        "attention_decoder.": (
            "the attention-decoder branch is not loaded; AED decode lands with "
            "the multi-paradigm Phase 2"
        ),
    }

    def _find_ckpt(self, ckpt_dir: Path, checkpoint_name: Optional[str] = None) -> Optional[Path]:
        ckpt_dir = Path(ckpt_dir)
        # icefall keeps checkpoints under exp/; search both the dir and exp/.
        search_dirs = [ckpt_dir, ckpt_dir / "exp"]
        if checkpoint_name:
            for d in search_dirs:
                if (d / checkpoint_name).exists():
                    return d / checkpoint_name
        for d in search_dirs:
            for c in _NAMED_CANDIDATES:
                if (d / c).exists():
                    return d / c
        for d in search_dirs:
            epochs = sorted(d.glob("epoch-*.pt"))
            if epochs:
                return epochs[-1]
            pts = sorted(d.glob("*.pt"))
            if pts:
                return pts[0]
        return None

    @staticmethod
    def _tokenizer_dirs_under(root: Path) -> List[Path]:
        """``root`` itself, ``root/data``, and ``root/data/lang_*``."""
        dirs = [root, root / "data"]
        if (root / "data").is_dir():
            dirs.extend(sorted((root / "data").glob("lang_*")))
        return dirs

    def _tokenizer_search_dirs(self, ckpt_dir: Path) -> List[Path]:
        """Where icefall keeps tokenizer assets.

        An icefall release puts the weights in ``<root>/exp`` and the tokenizer in
        ``<root>/data/lang_*`` — **siblings**.  So the search covers the parent
        too when the parent has a ``data/`` directory: pointing at ``exp/`` is a
        natural thing to do (it is where the weights are, and ``_find_ckpt``
        accepts it), and without this the bundle loads with ``tokenizer=None``,
        the engine falls back to joining raw ids, and the transcript comes out as
        numbers with no error anywhere.

        The checkpoint dir's own assets are searched first, so a self-contained
        directory is never overridden by a sibling.
        """
        ckpt_dir = Path(ckpt_dir)
        dirs = self._tokenizer_dirs_under(ckpt_dir)
        parent = ckpt_dir.parent
        if parent != ckpt_dir and (parent / "data").is_dir():
            dirs.extend(self._tokenizer_dirs_under(parent))
        seen, out = set(), []
        for d in dirs:
            if d.is_dir() and d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _find_tokenizer_asset(self, ckpt_dir: Path, name: str) -> Optional[Path]:
        for d in self._tokenizer_search_dirs(ckpt_dir):
            if (d / name).exists():
                return d / name
        return None

    #: only filename / asset conventions; other formats ship the same filenames, so this claim outranks a weaker one
    architecture: ClassVar[str] = "zipformer"
    source_format: ClassVar[str] = "icefall"
    default_checkpoint_name: ClassVar[str] = "pretrained.pt"
    default_decode_type: ClassVar[str] = "ctc"
    #: (see :func:`oasr.models.registry.resolve_architecture`).
    detect_specificity: ClassVar[int] = DETECT_ASSET_LAYOUT

    def detect(self, ckpt_dir: Path) -> bool:
        """Specific icefall markers only (the old "any ``.pt``" rule over-claimed).

        Positive markers: ``tokens.txt`` / ``bpe.model`` assets, an ``exp/``
        checkpoint layout, or icefall's conventional checkpoint filenames
        (``pretrained.pt``, ``epoch-*.pt``, ...).  A directory holding a single
        arbitrarily-named ``.pt`` no longer detects as icefall — pass
        ``architecture="zipformer"`` for such dirs.

        These are all filename conventions other frameworks share — a WeNet dir has
        ``final.pt``, a FunASR dir has ``model.pt`` — so this converter used to carry
        ``return False`` guards for ``train.yaml`` and ``config.yaml``, i.e. WeNet's
        and FunASR's markers hardcoded inside *icefall's* detector.  That made a 7th
        format an edit here.  Replaced by
        :attr:`detect_specificity` = ``DETECT_ASSET_LAYOUT``: those formats name
        themselves in a config file and outrank this claim, so the guards are gone
        and this method states only what icefall itself looks like.
        """
        ckpt_dir = Path(ckpt_dir)
        if self._find_tokenizer_asset(ckpt_dir, "tokens.txt") is not None:
            return True
        if self._find_tokenizer_asset(ckpt_dir, "bpe.model") is not None:
            return True
        for d in (ckpt_dir, ckpt_dir / "exp"):
            for c in _NAMED_CANDIDATES:
                if (d / c).exists():
                    return True
            if next(iter(d.glob("epoch-*.pt")), None) is not None:
                return True
        return False

    def config_from_state_dict(
        self, sd: Mapping[str, torch.Tensor], source: str = "state dict"
    ) -> ZipformerModelConfig:
        """Infer the architecture + vocab from checkpoint tensor shapes.

        Split out of :meth:`build_config` so :meth:`build_config_for_convert`
        can reuse weights that are already in memory — icefall ships no config
        file, so this is the only source of architecture, and reading the file
        twice to get it was the slowest step of a conversion.
        """
        config = ZipformerModelConfig()
        try:
            config.encoder = infer_encoder_config(sd)
        except Exception as exc:
            raise ValueError(
                f"could not infer the Zipformer architecture from {source} "
                f"({exc}). Pass an explicit model config, or check that this "
                "is an icefall Zipformer checkpoint."
            ) from exc
        w = sd.get("ctc_output.1.weight")
        if w is not None:
            vocab = int(w.shape[0])
            # GEMM kernels require N % 8 == 0; pad like the Conformer loader.
            if vocab % 8 != 0:
                vocab = (vocab // 8 + 1) * 8
            config.vocab_size = vocab
        return config

    def build_config(self, ckpt_dir: Path) -> ZipformerModelConfig:
        """Build the model config, inferring the encoder architecture + vocab from
        the checkpoint shapes."""
        config = ZipformerModelConfig()
        ckpt = self._find_ckpt(Path(ckpt_dir))
        if ckpt is not None:
            # Shape inference is the *only* source of architecture here, so a
            # failure must not silently fall back to the LibriSpeech "M"
            # defaults: that builds a plausible-looking but wrong model, which
            # then fails much later with a raw shape-mismatch error (or, if the
            # dims happen to coincide, loads and produces garbage).
            try:
                sd = _extract_state_dict(
                    torch.load(str(ckpt), map_location="cpu", weights_only=True)
                )
            except Exception as exc:
                raise ValueError(
                    f"could not read the icefall checkpoint {ckpt} to infer the "
                    f"Zipformer architecture: {exc}"
                ) from exc
            config = self.config_from_state_dict(sd, source=str(ckpt))
        logger.info(
            "Zipformer config: vocab_size=%s encoder_dim=%s num_encoder_layers=%s",
            config.vocab_size,
            config.encoder.encoder_dim,
            config.encoder.num_encoder_layers,
        )
        return config

    def build_aux(self, ckpt_dir: Path) -> dict:
        return {}

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        ckpt = self._find_ckpt(Path(ckpt_dir), checkpoint_name)
        if ckpt is None:
            raise FileNotFoundError(f"No icefall checkpoint (*.pt) found under {ckpt_dir}")
        return _extract_state_dict(
            torch.load(str(ckpt), map_location=map_location, weights_only=True)
        )

    # -- complete-bundle conversion (tokenizer / feature / decoding specs) ----

    def build_tokenizer_spec(self, ckpt_dir: Path):
        """icefall ships ``bpe.model`` (SentencePiece — the CTC ids *are* the
        piece ids) and/or ``tokens.txt``; prefer the SentencePiece model (it can
        also encode), fall back to the symbol table.  Fixes the historical gap
        where zipformer checkpoints got no symbol table at all (the engine only
        sniffed ``units.txt``/``words.txt``)."""
        from oasr.tokenizers import TokenizerSpec

        bpe = self._find_tokenizer_asset(ckpt_dir, "bpe.model")
        if bpe is not None:
            return TokenizerSpec(kind="sentencepiece", files={"model": str(bpe)})
        tokens = self._find_tokenizer_asset(ckpt_dir, "tokens.txt")
        if tokens is not None:
            return TokenizerSpec(kind="symbol_table", files={"table": str(tokens)})
        return None

    def build_feature_spec(self, ckpt_dir: Path):
        """icefall LibriSpeech recipes: 80-dim FBANK @16 kHz (dither off at inference)."""
        from oasr.features import FeatureSpec

        return FeatureSpec(
            kind="kaldi_fbank",
            sample_rate=16000,
            feature_dim=80,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            dither=0.0,
            normalize=None,
            audio_scale=32768.0,
        )

    def build_config_for_convert(self, ckpt_dir: Path, state_dict):
        """Infer from the already-loaded weights — icefall ships no config file.

        ``build_config`` would ``torch.load`` the same (multi-GB) checkpoint a
        second time just to read tensor shapes.
        """
        return self.config_from_state_dict(_extract_state_dict(state_dict))

    def build_decoding_defaults(self, config, ckpt_dir: Path):
        from oasr.checkpoints import DecodingDefaults

        # icefall CTC: <blk>=0; sos/eos unused by CTC decode.
        return DecodingDefaults(default_decode_type=self.default_decode_type, blank_id=0)


def load_icefall_checkpoint(
    ckpt_dir: str,
    checkpoint_name: str = "pretrained.pt",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[ZipformerModel, ZipformerModelConfig]:
    """Convenience loader for an icefall Zipformer checkpoint directory.

    Thin wrapper around :func:`oasr.models.registry.build_model_from_checkpoint`.
    """
    from ..registry import build_model_from_checkpoint

    return build_model_from_checkpoint(
        ckpt_dir, checkpoint_name=checkpoint_name, device=device, dtype=dtype
    )
