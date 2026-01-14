from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .patch_encoder import PatchEncoder
from .rel_attention import RelativeBias, RelTransformerEncoder


class EventMAE(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        decoder_embed_dim: int,
        decoder_num_heads: int,
        decoder_num_layers: int,
        mlp_ratio: float,
        plane_embed_dim: int,
        metadata_dim: int,
        num_tokens: int,
        num_planes: int,
        use_pos_embedding: bool = True,
        use_relative_bias: bool = False,
        rel_bias_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = num_tokens
        self.use_pos_embedding = use_pos_embedding
        self.use_relative_bias = use_relative_bias

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

        if self.use_relative_bias:
            self.rel_bias = RelativeBias(num_heads=num_heads, hidden_dim=rel_bias_hidden_dim)
            self.encoder = RelTransformerEncoder(
                d_model=embed_dim,
                nhead=num_heads,
                mlp_ratio=mlp_ratio,
                num_layers=num_layers,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embedding = nn.Parameter(
            torch.zeros(1, num_tokens, decoder_embed_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_num_layers)

        self.patch_decoder = nn.Linear(decoder_embed_dim, 2 * patch_size * patch_size)
        self.count_decoder = nn.Linear(decoder_embed_dim, 1)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embedding, std=0.02)

    def forward(
        self,
        patches: torch.Tensor,
        metadata: torch.Tensor,
        plane_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_tokens = patches.shape[:2]
        encoded = self.encode(patches, metadata, plane_ids, mask=mask)
        decoder_in = self.decoder_proj(encoded)
        if self.use_pos_embedding:
            decoder_in = decoder_in + self.decoder_pos_embedding
        decoded = self.decoder(decoder_in)
        pred_patch = self.patch_decoder(decoded).reshape(
            bsz, num_tokens, 2, self.patch_size, self.patch_size
        )
        pred_count = self.count_decoder(decoded)
        return pred_patch, pred_count, decoded

    def encode(
        self,
        patches: torch.Tensor,
        metadata: torch.Tensor,
        plane_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, num_tokens = patches.shape[:2]
        if num_tokens != self.num_tokens:
            raise ValueError("num_tokens mismatch with configured model")

        patches_flat = patches.reshape(bsz * num_tokens, 2, self.patch_size, self.patch_size)
        patch_emb = self.patch_encoder(patches_flat).reshape(bsz, num_tokens, -1)
        meta_emb = self.metadata_mlp(metadata)
        plane_emb = self.plane_proj(self.plane_embedding(plane_ids))

        if mask is not None:
            mask_token = self.mask_token.expand(bsz, num_tokens, -1)
            patch_emb = torch.where(mask.unsqueeze(-1), mask_token, patch_emb)

        token = patch_emb + meta_emb + plane_emb
        if self.use_pos_embedding:
            token = token + self.pos_embedding

        if self.use_relative_bias:
            bias = self.rel_bias(metadata)
            return self.encoder(token, attn_bias=bias)
        return self.encoder(token)
