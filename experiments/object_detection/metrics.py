from __future__ import annotations

from typing import Dict

import torch


def box_center_l1_px(pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    scale = torch.tensor([frame_size[0], frame_size[1]], device=pred.device, dtype=pred.dtype)
    return ((pred[:, :2] - target[:, :2]) * scale).abs().mean()


def box_iou_mean(pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    wh = torch.tensor([frame_size[0], frame_size[1]], device=pred.device, dtype=pred.dtype)
    pred_xy = pred[:, :2] * wh
    pred_wh = pred[:, 2:] * wh
    tgt_xy = target[:, :2] * wh
    tgt_wh = target[:, 2:] * wh

    pred0 = pred_xy - pred_wh / 2.0
    pred1 = pred_xy + pred_wh / 2.0
    tgt0 = tgt_xy - tgt_wh / 2.0
    tgt1 = tgt_xy + tgt_wh / 2.0

    inter0 = torch.maximum(pred0, tgt0)
    inter1 = torch.minimum(pred1, tgt1)
    inter_wh = (inter1 - inter0).clamp(min=0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    pred_area = pred_wh[:, 0].clamp(min=0.0) * pred_wh[:, 1].clamp(min=0.0)
    tgt_area = tgt_wh[:, 0].clamp(min=0.0) * tgt_wh[:, 1].clamp(min=0.0)
    union = pred_area + tgt_area - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union)).mean()


def heatmap_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = pred_logits.sigmoid() >= threshold
    tgt = target >= threshold
    inter = (pred & tgt).sum(dim=(1, 2, 3)).float()
    union = (pred | tgt).sum(dim=(1, 2, 3)).float()
    return torch.where(union > 0, inter / union, torch.zeros_like(union)).mean()


def summarize_metrics(
    preds: Dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    target_heatmaps: Dict[str, torch.Tensor],
    frame_size: tuple[int, int],
) -> Dict[str, float]:
    out = {
        "center_l1_px": float(box_center_l1_px(preds["boxes"], target_boxes, frame_size).item()),
        "box_iou": float(box_iou_mean(preds["boxes"], target_boxes, frame_size).item()),
    }
    for rep, target in target_heatmaps.items():
        out[f"heatmap_iou_{rep}"] = float(heatmap_iou(preds["heatmaps"][rep], target).item())
    return out
