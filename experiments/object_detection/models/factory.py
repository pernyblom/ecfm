from __future__ import annotations

from typing import Dict

import torch

from .model import MultiRepObjectDetector


def build_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    return MultiRepObjectDetector(
        representations=list(data_cfg["representations"]),
        heatmap_representations=list(data_cfg.get("heatmap_representations", ["xt_my", "yt_mx"])),
        image_size=tuple(data_cfg["image_size"]),
        backbone_cfg=dict(model_cfg.get("backbone", {})),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256)),
        heatmap_hidden_dim=int(model_cfg.get("heatmap_hidden_dim", 256)),
        box_hidden_dim=int(model_cfg.get("box_hidden_dim", 256)),
    ).to(device)
