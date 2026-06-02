from __future__ import annotations

from typing import Dict

import torch

from .kalman_residual import KalmanResidualForecaster
from ..utils.config import resolve_representation_image_sizes


def build_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    history_steps = int(model_cfg.get("history_steps", data_cfg.get("history_steps", 12)))
    return KalmanResidualForecaster(
        representations=list(data_cfg["representations"]),
        image_sizes=resolve_representation_image_sizes(data_cfg),
        backbone_cfg=dict(model_cfg.get("backbone", {})),
        history_steps=history_steps,
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256)),
        fusion_layers=int(model_cfg.get("fusion_layers", 1)),
        state_hidden_dim=int(model_cfg.get("state_hidden_dim", 128)),
        state_layers=int(model_cfg.get("state_layers", 2)),
        residual_hidden_dim=int(model_cfg.get("residual_hidden_dim", 256)),
        residual_layers=int(model_cfg.get("residual_layers", 2)),
        residual_scale=float(model_cfg.get("residual_scale", 1.0)),
        predict_size_residuals=bool(model_cfg.get("predict_size_residuals", True)),
        use_filter_state_features=bool(model_cfg.get("use_filter_state_features", False)),
        filter_state_feature_mode=str(model_cfg.get("filter_state_feature_mode", "full")),
        filter_covariance_features=str(model_cfg.get("filter_covariance_features", "none")),
        initial_state_source=str(model_cfg.get("initial_state_source", "last_four")),
        kalman_params=dict(cfg.get("kalman") or {}),
        cell_local_first_conv=bool(model_cfg.get("cell_local_first_conv", False)),
        cell_local_first_conv_representations=list(
            model_cfg.get("cell_local_first_conv_representations") or []
        ),
    ).to(device)
