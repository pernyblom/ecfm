from __future__ import annotations

from typing import Dict

import torch

from .model import LeWorldModel


def build_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    return LeWorldModel(cfg).to(device)
