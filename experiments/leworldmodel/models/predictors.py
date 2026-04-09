from __future__ import annotations

import torch
from torch import nn


class MLPPredictor(nn.Module):
    def __init__(self, latent_dim: int, history_steps: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers = []
        in_dim = latent_dim * history_steps
        current = in_dim
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.GELU())
            current = hidden_dim
        layers.append(nn.Linear(current, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.net(history.flatten(1))


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        history_steps: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, history_steps, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, latent_dim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(history) + self.pos[:, : history.shape[1]]
        x = self.encoder(x)
        x = self.norm(x[:, -1])
        return self.head(x)


class ForecastMLPHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        ssl_context_steps: int,
        box_history_steps: int,
        future_steps: int,
        hidden_dim: int,
        depth: int,
        use_ssl_features: bool = True,
        use_history_boxes: bool = True,
    ) -> None:
        super().__init__()
        self.use_ssl_features = bool(use_ssl_features)
        self.use_history_boxes = bool(use_history_boxes)
        if not self.use_ssl_features and not self.use_history_boxes:
            raise ValueError("ForecastMLPHead requires at least one input source.")
        in_dim = 0
        if self.use_ssl_features:
            in_dim += latent_dim * ssl_context_steps
        if self.use_history_boxes:
            in_dim += box_history_steps * 4
        layers = []
        current = in_dim
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(nn.GELU())
            current = hidden_dim
        layers.append(nn.Linear(current, future_steps * 4))
        self.net = nn.Sequential(*layers)
        self.future_steps = future_steps

    def forward(
        self,
        context_latents: torch.Tensor | None,
        past_boxes: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = []
        if self.use_ssl_features:
            if context_latents is None:
                raise ValueError("context_latents is required when use_ssl_features=True")
            parts.append(context_latents.flatten(1))
        if self.use_history_boxes:
            if past_boxes is None:
                raise ValueError("past_boxes is required when use_history_boxes=True")
            parts.append(past_boxes.flatten(1))
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        boxes = self.net(x).view(-1, self.future_steps, 4)
        return boxes.sigmoid()


def build_predictor(model_cfg: dict, latent_dim: int, history_steps: int) -> nn.Module:
    predictor_type = model_cfg.get("predictor_type", "mlp")
    if predictor_type == "mlp":
        return MLPPredictor(
            latent_dim=latent_dim,
            history_steps=history_steps,
            hidden_dim=int(model_cfg.get("predictor_hidden_dim", 256)),
            depth=int(model_cfg.get("predictor_depth", 2)),
        )
    if predictor_type == "transformer":
        return TransformerPredictor(
            latent_dim=latent_dim,
            history_steps=history_steps,
            d_model=int(model_cfg.get("predictor_d_model", 256)),
            nhead=int(model_cfg.get("predictor_nhead", 4)),
            num_layers=int(model_cfg.get("predictor_layers", 4)),
            dim_feedforward=int(model_cfg.get("predictor_dim_feedforward", 512)),
            dropout=float(model_cfg.get("predictor_dropout", 0.1)),
        )
    raise ValueError(f"Unknown predictor_type: {predictor_type}")
