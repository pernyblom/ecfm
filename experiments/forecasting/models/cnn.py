from __future__ import annotations

from typing import List

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, channels: List[int], feature_dim: int) -> None:
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(prev, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)
