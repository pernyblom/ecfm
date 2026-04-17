from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torchvision import models


@dataclass
class EncoderOutput:
    fmap: torch.Tensor
    pooled: torch.Tensor


class SmallCNNEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: List[int], out_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            prev = ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        fmap = self.backbone(x)
        pooled = self.proj(self.pool(fmap).flatten(1))
        return EncoderOutput(fmap=fmap, pooled=pooled)


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        net = models.resnet18(weights=None)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        fmap = self.backbone(x)
        pooled = self.proj(self.pool(fmap).flatten(1))
        return EncoderOutput(fmap=fmap, pooled=pooled)


def build_single_encoder(cfg: Dict) -> nn.Module:
    backbone_type = str(cfg.get("type", "small_cnn")).lower()
    in_channels = int(cfg.get("in_channels", 3))
    out_dim = int(cfg.get("out_dim", 128))
    if backbone_type == "small_cnn":
        return SmallCNNEncoder(in_channels, list(cfg.get("channels", [32, 64, 128])), out_dim)
    if backbone_type == "resnet18":
        return ResNet18Encoder(in_channels, out_dim)
    raise ValueError(f"Unknown backbone type: {backbone_type}")
