from __future__ import annotations

from typing import Dict

import torch

from .fusion import MultiRepForecast
from .transformer import MultiRepTransformer


def build_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    model_type = model_cfg.get("type", "gru")
    if model_type == "transformer":
        return MultiRepTransformer(
            reps=data_cfg["representations"],
            cnn_channels=model_cfg["cnn_channels"],
            feature_dim=model_cfg["feature_dim"],
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nhead", 4),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 4),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
            dim_feedforward=model_cfg.get("dim_feedforward", 512),
            dropout=model_cfg.get("dropout", 0.1),
            use_past_boxes=model_cfg["use_past_boxes"],
            past_steps=data_cfg["past_steps"],
            future_steps=data_cfg["future_steps"],
            predict_past=bool(model_cfg.get("predict_past", False)),
            pos_encoding=model_cfg.get("pos_encoding", "learned"),
        ).to(device)
    if model_type in {"gru", "rnn"}:
        return MultiRepForecast(
            reps=data_cfg["representations"],
            cnn_channels=model_cfg["cnn_channels"],
            feature_dim=model_cfg["feature_dim"],
            use_past_boxes=model_cfg["use_past_boxes"],
            rnn_hidden=model_cfg["rnn_hidden"],
            rnn_layers=model_cfg["rnn_layers"],
            future_steps=data_cfg["future_steps"],
        ).to(device)
    raise ValueError(f"Unknown model type: {model_type}")
