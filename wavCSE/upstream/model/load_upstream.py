"""
Upstream SSL model factory.

This module provides a centralized interface for loading
upstream self supervised learning speech models based on
a string identifier. It handles checkpoint validation,
device placement, and evaluation mode setup.

Currently supported upstream models include WavLM variants.
The factory design allows easy extension to additional
models such as HuBERT or wav2vec 2.0.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

from __future__ import annotations

from pathlib import Path
import logging
import torch

from model.wavlm.wavlm_loader import load_wavlm


def load_upstream_model(
    upstream_model_type: str,
    checkpoint_path: str | Path,
    device: torch.device,
):
    """
    Loads and returns an upstream model based on upstream_model_type.

    Expected upstream_model_type examples:
      - "wavlm_large"
      - "wavlm_base"

    Returns:
      torch.nn.Module (already moved to device and set to eval mode)
    """
    if upstream_model_type is None:
        raise ValueError("upstream_model_type is None")

    model_key = str(upstream_model_type).strip().lower()
    ckpt_path = Path(checkpoint_path).expanduser()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint_path not found: {ckpt_path}")

    # ---- WavLM ----
    if model_key.startswith("wavlm"):
        model = load_wavlm(checkpoint_path=ckpt_path, device=device)

        logging.info(
            "Loaded upstream model | type=%s | ckpt=%s | device=%s",
            model_key, str(ckpt_path), str(device)
        )
        return model

    # ---- Future models (HuBERT, wav2vec2, etc.) ----
    raise ValueError(
        f"Unknown upstream_model_type='{upstream_model_type}'. "
        "Supported: wavlm_base, wavlm_large"
    )
