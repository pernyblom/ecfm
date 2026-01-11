from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .patch_encoder import PatchEncoder


class EventMAE(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        plane_embed_dim: int,
        metadata_dim: int,
        num_tokens: int,
        num_planes: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = num_tokens

        self.patch_encoder = PatchEncoder(patch_size, embed_dim)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.plane_embedding = nn.Embedding(num_planes, plane_embed_dim)
        self.plane_proj = nn.Linear(plane_embed_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.patch_decoder = nn.Linear(embed_dim, patch_size * patch_size)
        self.count_decoder = nn.Linear(embed_dim, 1)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        metadata: torch.Tensor,
        plane_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_tokens = patches.shape[:2]
        if num_tokens != self.num_tokens:
            raise ValueError("num_tokens mismatch with configured model")

        patches_flat = patches.reshape(bsz * num_tokens, 1, self.patch_size, self.patch_size)
        patch_emb = self.patch_encoder(patches_flat).reshape(bsz, num_tokens, -1)
        meta_emb = self.metadata_mlp(metadata)
        token = patch_emb + meta_emb

        plane_emb = self.plane_proj(self.plane_embedding(plane_ids))
        token = token + plane_emb + self.pos_embedding

        if mask is not None:
            mask_token = self.mask_token.expand(bsz, num_tokens, -1)
            token = torch.where(mask.unsqueeze(-1), mask_token, token)

        encoded = self.encoder(token)
        pred_patch = self.patch_decoder(encoded).reshape(bsz, num_tokens, 1, self.patch_size, self.patch_size)
        pred_count = self.count_decoder(encoded)
        return pred_patch, pred_count, encoded

