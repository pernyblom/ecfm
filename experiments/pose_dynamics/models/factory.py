from __future__ import annotations

from typing import Dict

import torch

from .pose_dynamics import PoseDynamicsProjector


def build_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    return PoseDynamicsProjector(
        reps=data_cfg["representations"],
        cnn_channels=model_cfg["cnn_channels"],
        feature_dim=model_cfg["feature_dim"],
        hidden_dim=model_cfg.get("hidden_dim", 256),
        future_steps=data_cfg["future_steps"],
        min_depth=float(model_cfg.get("min_depth", 0.1)),
    ).to(device)
