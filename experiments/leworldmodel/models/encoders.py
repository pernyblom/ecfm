from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torchvision import models


class SmallCNNBackbone(nn.Module):
    def __init__(self, in_channels: int, channels: List[int], out_dim: int) -> None:
        super().__init__()
        layers = []
        prev = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.GELU())
            prev = ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class ResNet18Backbone(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model.fc = nn.Linear(model.fc.in_features, out_dim)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ViTTinyBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: tuple[int, int],
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        out_dim: int,
    ) -> None:
        super().__init__()
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for vit_tiny.")
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


class MultiRepEncoder(nn.Module):
    def __init__(
        self,
        reps: List[str],
        encoder_type: str,
        image_size: tuple[int, int],
        latent_dim: int,
        cnn_channels: List[int],
        vit_patch_size: int = 16,
        vit_embed_dim: int = 192,
        vit_depth: int = 6,
        vit_heads: int = 3,
        vit_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.reps = list(reps)
        in_channels = len(self.reps) * 3
        if encoder_type == "small_cnn":
            self.backbone = SmallCNNBackbone(in_channels, cnn_channels, latent_dim)
        elif encoder_type == "resnet18":
            self.backbone = ResNet18Backbone(in_channels, latent_dim)
        elif encoder_type == "vit_tiny":
            self.backbone = ViTTinyBackbone(
                in_channels=in_channels,
                image_size=image_size,
                patch_size=vit_patch_size,
                embed_dim=vit_embed_dim,
                depth=vit_depth,
                num_heads=vit_heads,
                mlp_ratio=vit_mlp_ratio,
                dropout=dropout,
                out_dim=latent_dim,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        per_rep = [inputs[rep] for rep in self.reps]
        x = torch.cat(per_rep, dim=2)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        z = self.backbone(x)
        return z.view(b, t, -1)
