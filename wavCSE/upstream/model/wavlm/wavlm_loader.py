# models/wavlm/wavlm_loader.py
from __future__ import annotations

from pathlib import Path
import torch

from model.wavlm.WavLM import WavLM, WavLMConfig


def load_wavlm(checkpoint_path: str | Path, device: torch.device):
    ckpt_path = Path(checkpoint_path).expanduser()

    checkpoint = torch.load(ckpt_path, map_location="cpu")  # load safely on CPU first
    cfg_model = WavLMConfig(checkpoint["cfg"])
    model = WavLM(cfg_model)
    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()
    return model
