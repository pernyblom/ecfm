from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .encoders import MultiRepEncoder
from .predictors import ForecastMLPHead, build_predictor


class LeWorldModel(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        self.ssl_context_steps = int(data_cfg["ssl_context_steps"])
        self.ssl_future_steps = int(data_cfg["ssl_future_steps"])
        self.forecast_history_steps = int(data_cfg["forecast_history_steps"])
        self.forecast_future_steps = int(data_cfg["forecast_future_steps"])
        self.latent_dim = int(model_cfg["latent_dim"])
        self.encoder = MultiRepEncoder(
            reps=data_cfg["representations"],
            encoder_type=model_cfg.get("encoder_type", "small_cnn"),
            image_size=tuple(data_cfg["image_size"]),
            latent_dim=self.latent_dim,
            cnn_channels=list(model_cfg.get("cnn_channels", [16, 32, 64])),
            vit_patch_size=int(model_cfg.get("vit_patch_size", 16)),
            vit_embed_dim=int(model_cfg.get("vit_embed_dim", 192)),
            vit_depth=int(model_cfg.get("vit_depth", 6)),
            vit_heads=int(model_cfg.get("vit_heads", 3)),
            vit_mlp_ratio=float(model_cfg.get("vit_mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        self.predictor = build_predictor(model_cfg, self.latent_dim, self.ssl_context_steps)
        forecast_cfg = cfg.get("downstream", {}).get("forecasting", {})
        self.forecast_head = None
        if bool(forecast_cfg.get("enabled", False)):
            self.forecast_head = ForecastMLPHead(
                latent_dim=self.latent_dim,
                ssl_context_steps=self.ssl_context_steps,
                box_history_steps=self.forecast_history_steps,
                future_steps=self.forecast_future_steps,
                hidden_dim=int(forecast_cfg.get("hidden_dim", 256)),
                depth=int(forecast_cfg.get("depth", 2)),
            )

    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(inputs)

    def rollout(self, context_latents: torch.Tensor) -> torch.Tensor:
        history = context_latents
        preds = []
        for _ in range(self.ssl_future_steps):
            next_latent = self.predictor(history[:, -self.ssl_context_steps :])
            preds.append(next_latent)
            history = torch.cat([history, next_latent.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    def teacher_force_predictions(self, z_seq: torch.Tensor) -> torch.Tensor:
        preds = []
        for idx in range(self.ssl_future_steps):
            history = z_seq[:, idx : idx + self.ssl_context_steps]
            preds.append(self.predictor(history))
        return torch.stack(preds, dim=1)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        past_boxes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_seq = self.encode(inputs)
        z_context = z_seq[:, : self.ssl_context_steps]
        z_future = z_seq[:, self.ssl_context_steps :]
        pred_future_teacher = self.teacher_force_predictions(z_seq)
        pred_future_rollout = self.rollout(z_context)
        out = {
            "latents": z_seq,
            "context_latents": z_context,
            "future_latents": z_future,
            "pred_future_teacher": pred_future_teacher,
            "pred_future_rollout": pred_future_rollout,
        }
        if self.forecast_head is not None and past_boxes is not None:
            out["pred_future_boxes"] = self.forecast_head(z_context, past_boxes)
        return out
