from __future__ import annotations

import torch
from torch import nn


class PatchEncoder(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * patch_size * patch_size, embed_dim),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.net(patch)

