from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import build_single_encoder


def _encoder_fmap_dim(backbone_cfg: Dict) -> int:
    backbone_type = str(backbone_cfg.get("type", "small_cnn")).lower()
    if backbone_type == "resnet18":
        return 512
    if backbone_type == "small_cnn":
        channels = list(backbone_cfg.get("channels", [32, 64, 128]))
        return int(channels[-1])
    raise ValueError(f"Unknown backbone type: {backbone_type}")


class CenterNetDetector(nn.Module):
    def __init__(
        self,
        *,
        representations: List[str],
        image_sizes: Dict[str, Tuple[int, int]],
        frame_size: Tuple[int, int],
        backbone_cfg: Dict,
        hidden_dim: int = 128,
        output_stride: int = 4,
        topk: int = 100,
        predict_velocity: bool = True,
    ) -> None:
        super().__init__()
        self.representations = list(representations)
        self.image_sizes = {
            str(rep): (int(size[0]), int(size[1])) for rep, size in dict(image_sizes).items()
        }
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.output_stride = int(output_stride)
        self.topk = int(topk)
        self.predict_velocity = bool(predict_velocity)
        if self.output_stride < 1:
            raise ValueError(f"output_stride must be >= 1, got {self.output_stride}")
        if not self.representations:
            raise ValueError("representations must not be empty.")

        out_w = max(1, self.frame_size[0] // self.output_stride)
        out_h = max(1, self.frame_size[1] // self.output_stride)
        self.output_size = (out_w, out_h)
        self.encoders = nn.ModuleDict({rep: build_single_encoder(backbone_cfg) for rep in self.representations})
        in_dim = _encoder_fmap_dim(backbone_cfg) * len(self.representations)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )
        if self.predict_velocity:
            self.velocity_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 2, kernel_size=1),
            )
        else:
            self.velocity_head = None

        prior = 0.01
        nn.init.constant_(self.heatmap_head[-1].bias, -torch.log(torch.tensor((1.0 - prior) / prior)).item())

    def _decode(
        self,
        heatmap_logits: torch.Tensor,
        size_raw: torch.Tensor,
        offset_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heat = heatmap_logits.sigmoid()
        pooled = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
        peaks = heat * (heat == pooled).to(heat.dtype)
        batch_size, _, out_h, out_w = peaks.shape
        k = min(self.topk, out_h * out_w)
        scores, indices = torch.topk(peaks.flatten(1), k=k, dim=1)
        ys = torch.div(indices, out_w, rounding_mode="floor")
        xs = indices % out_w
        gather_idx = indices.unsqueeze(1).expand(-1, 2, -1)
        size = size_raw.sigmoid().flatten(2).gather(2, gather_idx).permute(0, 2, 1)
        offset = offset_raw.sigmoid().flatten(2).gather(2, gather_idx).permute(0, 2, 1)
        centers = torch.stack([xs.to(size.dtype), ys.to(size.dtype)], dim=-1) + offset
        scale = torch.tensor([out_w, out_h], device=size.device, dtype=size.dtype)
        centers = centers / scale
        boxes = torch.cat([centers.clamp(0.0, 1.0), size.clamp(1.0e-4, 1.0)], dim=-1)
        return boxes, scores

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target_hw = (self.output_size[1], self.output_size[0])
        fmaps = []
        pooled = []
        for rep in self.representations:
            enc = self.encoders[rep](inputs[rep])
            fmaps.append(F.interpolate(enc.fmap, size=target_hw, mode="bilinear", align_corners=False))
            pooled.append(enc.pooled)
        fused_map = self.fusion(torch.cat(fmaps, dim=1))
        heatmap_logits = self.heatmap_head(fused_map)
        size_raw = self.size_head(fused_map)
        offset_raw = self.offset_head(fused_map)
        boxes, scores = self._decode(heatmap_logits, size_raw, offset_raw)
        out = {
            "detector_type": "centernet",
            "fused_features": torch.cat(pooled, dim=-1),
            "centernet_heatmap_logits": heatmap_logits,
            "centernet_size_raw": size_raw,
            "centernet_offset_raw": offset_raw,
            "heatmaps": {"xy": heatmap_logits},
            "boxes": boxes,
            "scores": scores,
        }
        if self.velocity_head is not None:
            out["centernet_velocity"] = self.velocity_head(fused_map)
        return out
