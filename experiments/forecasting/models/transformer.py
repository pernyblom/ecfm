from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .cnn import SmallCNN


def _sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim)
    )
    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MultiRepTransformer(nn.Module):
    def __init__(
        self,
        reps: List[str],
        cnn_channels: List[int],
        feature_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        use_past_boxes: bool,
        past_steps: int,
        future_steps: int,
        predict_past: bool,
        pos_encoding: str,
    ) -> None:
        super().__init__()
        self.reps = reps
        self.use_past_boxes = use_past_boxes
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.predict_past = predict_past

        self.encoders = nn.ModuleDict(
            {r: SmallCNN(3, cnn_channels, feature_dim) for r in reps}
        )
        input_dim = feature_dim * len(reps)
        if use_past_boxes:
            input_dim += 4
        self.input_proj = nn.Linear(input_dim, d_model)

        total_steps = past_steps + future_steps if predict_past else future_steps
        self.pos_encoding = pos_encoding
        if pos_encoding == "learned":
            self.src_pos = nn.Parameter(torch.zeros(1, past_steps, d_model))
            self.tgt_pos = nn.Parameter(torch.zeros(1, total_steps, d_model))
            self.query = nn.Parameter(torch.zeros(1, total_steps, d_model))
            nn.init.trunc_normal_(self.src_pos, std=0.02)
            nn.init.trunc_normal_(self.tgt_pos, std=0.02)
            nn.init.trunc_normal_(self.query, std=0.02)
        elif pos_encoding == "sinusoidal":
            self.register_buffer(
                "src_pos",
                _sinusoidal_positional_encoding(past_steps, d_model).unsqueeze(0),
            )
            self.register_buffer(
                "tgt_pos",
                _sinusoidal_positional_encoding(total_steps, d_model).unsqueeze(0),
            )
            self.register_buffer("query", torch.zeros(1, total_steps, d_model))
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(d_model, 4)


    def forward(self, inputs: Dict[str, torch.Tensor], past_boxes: torch.Tensor) -> torch.Tensor:
        # inputs[rep]: [B, T, C, H, W] where T == past_steps
        reps_feat = []
        for rep in self.reps:
            x = inputs[rep]
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            feat = self.encoders[rep](x).view(b, t, -1)
            reps_feat.append(feat)
        feats = torch.cat(reps_feat, dim=-1)
        if self.use_past_boxes:
            feats = torch.cat([feats, past_boxes], dim=-1)

        src = self.input_proj(feats) + self.src_pos
        tgt = self.query.expand(src.shape[0], -1, -1) + self.tgt_pos
        out = self.transformer(src, tgt)
        return self.head(out)
