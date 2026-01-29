from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .cnn import SmallCNN


class MultiRepForecast(nn.Module):
    def __init__(
        self,
        reps: List[str],
        cnn_channels: List[int],
        feature_dim: int,
        use_past_boxes: bool,
        rnn_hidden: int,
        rnn_layers: int,
        future_steps: int,
    ) -> None:
        super().__init__()
        self.reps = reps
        self.use_past_boxes = use_past_boxes
        self.future_steps = future_steps

        self.encoders = nn.ModuleDict(
            {r: SmallCNN(3, cnn_channels, feature_dim) for r in reps}
        )
        input_dim = feature_dim * len(reps)
        if use_past_boxes:
            input_dim += 4

        self.rnn = nn.GRU(
            input_dim, rnn_hidden, num_layers=rnn_layers, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_hidden, future_steps * 4),
        )

    def forward(self, inputs: Dict[str, torch.Tensor], past_boxes: torch.Tensor) -> torch.Tensor:
        # inputs[rep]: [B, T, C, H, W]
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
        out, _ = self.rnn(feats)
        last = out[:, -1]
        pred = self.head(last).view(-1, self.future_steps, 4)
        return pred
