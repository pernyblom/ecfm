from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import build_single_encoder


class HeatmapHead(nn.Module):
    def __init__(self, in_dim: int, out_size: tuple[int, int], hidden_dim: int = 256) -> None:
        super().__init__()
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.out_size[0] * self.out_size[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(x.shape[0], 1, self.out_size[1], self.out_size[0])


class BoxHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        center = raw[..., :2].sigmoid()
        size = F.softplus(raw[..., 2:]).clamp(min=1.0e-4, max=1.0)
        return torch.cat([center, size], dim=-1)


class ObjectnessHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultiRepObjectDetector(nn.Module):
    def __init__(
        self,
        *,
        representations: List[str],
        heatmap_representations: List[str],
        image_size: tuple[int, int],
        backbone_cfg: Dict,
        fusion_hidden_dim: int = 256,
        heatmap_hidden_dim: int = 256,
        box_hidden_dim: int = 256,
        num_queries: int = 8,
        query_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.representations = list(representations)
        self.heatmap_representations = list(heatmap_representations)
        self.num_queries = int(num_queries)
        self.encoders = nn.ModuleDict({rep: build_single_encoder(backbone_cfg) for rep in self.representations})
        per_rep_dim = int(backbone_cfg.get("out_dim", 128))
        fused_dim = per_rep_dim * len(self.representations)
        self.fusion = nn.Sequential(nn.Linear(fused_dim, fusion_hidden_dim), nn.ReLU(inplace=True))
        heatmap_in_dim = fusion_hidden_dim + per_rep_dim
        self.heatmap_heads = nn.ModuleDict()
        for rep in self.heatmap_representations:
            self.heatmap_heads[rep] = HeatmapHead(heatmap_in_dim, image_size, heatmap_hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, query_hidden_dim)
        self.query_proj = nn.Sequential(
            nn.Linear(fusion_hidden_dim + query_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.box_head = BoxHead(fusion_hidden_dim, box_hidden_dim)
        self.objectness_head = ObjectnessHead(fusion_hidden_dim, min(box_hidden_dim, 128))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = {rep: self.encoders[rep](inputs[rep]) for rep in self.representations}
        fused = self.fusion(torch.cat([enc[rep].pooled for rep in self.representations], dim=-1))
        heatmaps = {
            rep: self.heatmap_heads[rep](torch.cat([fused, enc[rep].pooled], dim=-1))
            for rep in self.heatmap_representations
        }
        batch_size = fused.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        fused_queries = fused.unsqueeze(1).expand(-1, self.num_queries, -1)
        query_features = self.query_proj(torch.cat([fused_queries, query_embed], dim=-1))
        return {
            "fused_features": fused,
            "query_features": query_features,
            "boxes": self.box_head(query_features),
            "heatmaps": heatmaps,
            "objectness_logits": self.objectness_head(query_features),
        }
