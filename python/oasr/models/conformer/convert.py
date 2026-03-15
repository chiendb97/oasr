# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert and load pretrained WeNet Conformer checkpoints into OASR models.

Usage (as module)::

    from oasr.models.conformer.convert import load_wenet_checkpoint

    model = load_wenet_checkpoint(
        "/path/to/20210610_u2pp_conformer_exp_librispeech",
    )

Usage (CLI)::

    python -m oasr.models.conformer.convert \
        /path/to/20210610_u2pp_conformer_exp_librispeech

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from .config import ConformerEncoderConfig, ConformerModelConfig
from .model import ConformerModel, GlobalCMVN

logger = logging.getLogger(__name__)


def _parse_wenet_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _build_config_from_wenet(raw: Dict[str, Any]) -> ConformerModelConfig:
    """Translate a WeNet ``train.yaml`` into a :class:`ConformerModelConfig`."""
    enc = raw.get("encoder_conf", {})
    encoder_cfg = ConformerEncoderConfig(
        input_size=raw.get("input_dim", 80),
        output_size=enc.get("output_size", 256),
        num_blocks=enc.get("num_blocks", 12),
        attention_heads=enc.get("attention_heads", 4),
        linear_units=enc.get("linear_units", 2048),
        cnn_module_kernel=enc.get("cnn_module_kernel", 15),
        use_cnn_module=enc.get("use_cnn_module", True),
        activation_type=enc.get("activation_type", "swish"),
        normalize_before=enc.get("normalize_before", True),
        macaron_style=enc.get("macaron_style", True),
        causal=enc.get("causal", False),
        cnn_module_norm=enc.get("cnn_module_norm", "batch_norm"),
        input_layer=enc.get("input_layer", "conv2d"),
        embed_layer_norm=False,
    )
    return ConformerModelConfig(
        encoder=encoder_cfg,
        vocab_size=raw.get("output_dim"),
    )


def _load_global_cmvn(cmvn_path: str) -> GlobalCMVN:
    """Load a WeNet JSON-format ``global_cmvn`` file into a :class:`GlobalCMVN` module."""
    with open(cmvn_path, "r") as f:
        raw = json.load(f)
    mean_stat = torch.tensor(raw["mean_stat"], dtype=torch.float32)
    var_stat = torch.tensor(raw["var_stat"], dtype=torch.float32)
    frame_num = raw["frame_num"]

    mean = mean_stat / frame_num
    variance = var_stat / frame_num - mean ** 2
    istd = 1.0 / torch.sqrt(torch.clamp(variance, min=1e-20))
    return GlobalCMVN(mean, istd)


def _filter_encoder_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Keep only ``encoder.*`` keys from a full WeNet checkpoint.

    The prefix is kept as-is since ``ConformerModel.state_dict()`` also
    nests everything under ``encoder.``.
    """
    return {k: v for k, v in state_dict.items() if k.startswith("encoder.")}


def load_wenet_checkpoint(
    ckpt_dir: str,
    checkpoint_name: str = "final.pt",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[ConformerModel, ConformerModelConfig]:
    """Load a WeNet pretrained Conformer checkpoint directory.

    Args:
        ckpt_dir: Path to the WeNet experiment directory (must contain
            ``train.yaml``, ``global_cmvn``, and the checkpoint file).
        checkpoint_name: Filename of the ``.pt`` checkpoint inside
            *ckpt_dir*.  Defaults to ``"final.pt"``.
        device: Device to map tensors onto.
        dtype: Optional dtype to cast model parameters into after loading.

    Returns:
        A tuple of ``(model, config)`` where *model* has weights loaded
        and is set to eval mode.
    """
    ckpt_dir = Path(ckpt_dir)
    yaml_path = ckpt_dir / "train.yaml"
    cmvn_path = ckpt_dir / "global_cmvn"
    ckpt_path = ckpt_dir / checkpoint_name

    for p in (yaml_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # --- 1. Parse config ---
    raw_config = _parse_wenet_yaml(str(yaml_path))
    config = _build_config_from_wenet(raw_config)
    logger.info("Encoder config: %s", config.encoder)

    # --- 2. Build GlobalCMVN (optional) ---
    global_cmvn: Optional[GlobalCMVN] = None
    if cmvn_path.exists():
        global_cmvn = _load_global_cmvn(str(cmvn_path))
        logger.info("Loaded global CMVN from %s", cmvn_path)

    # --- 3. Construct model ---
    model = ConformerModel(config, global_cmvn=global_cmvn)

    # --- 4. Load checkpoint ---
    state_dict = torch.load(str(ckpt_path), map_location=device)
    encoder_sd = _filter_encoder_state_dict(state_dict)
    logger.info(
        "Checkpoint has %d encoder keys (out of %d total)",
        len(encoder_sd),
        len(state_dict),
    )

    missing, unexpected = model.load_state_dict(encoder_sd, strict=False)

    # pos_enc.pe is a computed buffer, expected to be absent from the checkpoint
    _expected_missing = {"encoder.embed.pos_enc.pe"}
    real_missing = [k for k in missing if k not in _expected_missing]
    if real_missing:
        logger.warning("Unexpected missing keys: %s", real_missing)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected)

    model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)

    model.eval()
    logger.info("Model loaded and set to eval mode.")
    return model, config


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Load a WeNet Conformer checkpoint into OASR."
    )
    parser.add_argument(
        "ckpt_dir",
        help="Path to the WeNet experiment directory.",
    )
    parser.add_argument(
        "--checkpoint", default="final.pt",
        help="Checkpoint filename (default: final.pt).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to load onto (default: cpu).",
    )
    parser.add_argument(
        "--save", default=None,
        help="If set, save the converted OASR state dict to this path.",
    )
    args = parser.parse_args()

    model, config = load_wenet_checkpoint(
        args.ckpt_dir,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")
    print(f"Config: {config}")

    if args.save:
        save_path = Path(args.save)
        torch.save(model.state_dict(), str(save_path))
        print(f"Saved converted state dict to {save_path}")
