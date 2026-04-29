from __future__ import annotations

from typing import Dict

import torch

from .model import MultiRepObjectDetector
from .centernet import CenterNetDetector
from ..utils.config import resolve_representation_image_sizes


def build_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    detector_type = str(model_cfg.get("detector", "detr_lite")).lower()
    if detector_type == "centernet":
        return CenterNetDetector(
            representations=list(data_cfg["representations"]),
            image_sizes=resolve_representation_image_sizes(data_cfg),
            frame_size=tuple(data_cfg["frame_size"]),
            backbone_cfg=dict(model_cfg.get("backbone", {})),
            hidden_dim=int(model_cfg.get("centernet_hidden_dim", model_cfg.get("fusion_hidden_dim", 128))),
            output_stride=int(model_cfg.get("output_stride", 4)),
            topk=int(model_cfg.get("topk", 100)),
            predict_velocity=bool(model_cfg.get("predict_velocity", True)),
        ).to(device)
    if detector_type not in {"detr_lite", "query", "detr-lite"}:
        raise ValueError(f"Unknown model.detector: {detector_type}")
    return MultiRepObjectDetector(
        representations=list(data_cfg["representations"]),
        heatmap_representations=list(data_cfg.get("heatmap_representations", [])),
        image_sizes=resolve_representation_image_sizes(data_cfg),
        backbone_cfg=dict(model_cfg.get("backbone", {})),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256)),
        heatmap_hidden_dim=int(model_cfg.get("heatmap_hidden_dim", 256)),
        box_hidden_dim=int(model_cfg.get("box_hidden_dim", 256)),
        num_queries=int(model_cfg.get("num_queries", 8)),
        query_hidden_dim=int(model_cfg.get("query_hidden_dim", 256)),
    ).to(device)
