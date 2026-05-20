from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import re

import torch
from torch import nn
from torchvision import models

_GRID_REP_RE = re.compile(r"^(?P<base>.+)_(?P<grid_x>\d+)x(?P<grid_y>\d+)$", re.IGNORECASE)
_GRID_SPLIT_BASE_REPS = {"xy", "xt", "yt", "cstr2", "cstr3", "xt_my", "yt_mx", "events"}


@dataclass
class EncoderOutput:
    fmap: torch.Tensor
    pooled: torch.Tensor


def grid_split_from_rep_name(rep: str) -> tuple[int, int] | None:
    match = _GRID_REP_RE.match(str(rep))
    if match is None:
        return None
    if match.group("base") not in _GRID_SPLIT_BASE_REPS:
        return None
    grid_x = int(match.group("grid_x"))
    grid_y = int(match.group("grid_y"))
    if grid_x <= 0 or grid_y <= 0:
        return None
    return grid_x, grid_y


class CellLocalConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, *, grid_x: int, grid_y: int) -> None:
        super().__init__()
        self.conv = conv
        self.grid_x = int(grid_x)
        self.grid_y = int(grid_y)
        if self.grid_x <= 0 or self.grid_y <= 0:
            raise ValueError(f"grid_x and grid_y must be positive, got {grid_x}x{grid_y}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grid_x == 1 and self.grid_y == 1:
            return self.conv(x)
        _, _, h, w = x.shape
        if self.grid_x > w or self.grid_y > h:
            raise ValueError(
                f"cell-local first conv grid {self.grid_x}x{self.grid_y} is larger than "
                f"input feature size {w}x{h}."
            )
        x_edges = torch.linspace(0, w, self.grid_x + 1, device=x.device).long().tolist()
        y_edges = torch.linspace(0, h, self.grid_y + 1, device=x.device).long().tolist()
        rows: list[torch.Tensor] = []
        for gy in range(self.grid_y):
            cols: list[torch.Tensor] = []
            y0, y1 = y_edges[gy], y_edges[gy + 1]
            for gx in range(self.grid_x):
                x0, x1 = x_edges[gx], x_edges[gx + 1]
                cols.append(self.conv(x[:, :, y0:y1, x0:x1]))
            rows.append(torch.cat(cols, dim=3))
        return torch.cat(rows, dim=2)


class SmallCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        out_dim: int,
        first_conv_grid: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_channels
        for idx, ch in enumerate(channels):
            conv: nn.Module = nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1)
            if idx == 0 and first_conv_grid is not None:
                conv = CellLocalConv2d(conv, grid_x=first_conv_grid[0], grid_y=first_conv_grid[1])
            layers.append(conv)
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
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        first_conv_grid: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        net = models.resnet18(weights=None)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1: nn.Module = net.conv1
        if first_conv_grid is not None:
            conv1 = CellLocalConv2d(net.conv1, grid_x=first_conv_grid[0], grid_y=first_conv_grid[1])
        self.backbone = nn.Sequential(
            conv1,
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
    first_conv_grid_raw = cfg.get("cell_local_first_conv_grid")
    first_conv_grid = None
    if first_conv_grid_raw is not None:
        first_conv_grid = (int(first_conv_grid_raw[0]), int(first_conv_grid_raw[1]))
    if backbone_type == "small_cnn":
        return SmallCNNEncoder(
            in_channels,
            list(cfg.get("channels", [32, 64, 128])),
            out_dim,
            first_conv_grid=first_conv_grid,
        )
    if backbone_type == "resnet18":
        return ResNet18Encoder(in_channels, out_dim, first_conv_grid=first_conv_grid)
    raise ValueError(f"Unknown backbone type: {backbone_type}")
