from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class RelativeBias(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        # metadata: [B, N, 9], use normalized x,y,dx,dy,t,dt (first 6 entries)
        pos = metadata[..., :6]
        x, y, dx, dy, t, dt = [pos[..., i] for i in range(6)]

        xi = x[:, :, None]
        xj = x[:, None, :]
        yi = y[:, :, None]
        yj = y[:, None, :]
        ti = t[:, :, None]
        tj = t[:, None, :]
        dxi = dx[:, :, None]
        dxj = dx[:, None, :]
        dyi = dy[:, :, None]
        dyj = dy[:, None, :]
        dti = dt[:, :, None]
        dtj = dt[:, None, :]

        eps = 1e-6
        feats = torch.stack(
            [
                xi - xj,
                yi - yj,
                ti - tj,
                torch.log((dxi + eps) / (dxj + eps)),
                torch.log((dyi + eps) / (dyj + eps)),
                torch.log((dti + eps) / (dtj + eps)),
            ],
            dim=-1,
        )
        bias = self.mlp(feats)  # [B, N, N, H]
        return bias.permute(0, 3, 1, 2).contiguous()  # [B, H, N, N]


class BiasMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, N, D], attn_bias: [B, H, N, N]
        bsz, num_tokens, _ = x.shape
        qkv = self.in_proj(x)
        qkv = qkv.view(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, num_tokens, self.embed_dim)
        return self.out_proj(out)


class RelTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = BiasMultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, int(d_model * mlp_ratio))
        self.linear2 = nn.Linear(int(d_model * mlp_ratio), d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.dropout(self.linear2(self.activation(self.linear1(self.norm2(x)))))
        return x


class RelTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)
        return x
