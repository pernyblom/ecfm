from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _boxes_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    xy0 = boxes[:, :2] - boxes[:, 2:] / 2.0
    xy1 = boxes[:, :2] + boxes[:, 2:] / 2.0
    return torch.cat([xy0, xy1], dim=-1)


def _ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred_xyxy = _boxes_xyxy(pred_boxes)
    tgt_xyxy = _boxes_xyxy(target_boxes)

    inter0 = torch.maximum(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    inter1 = torch.minimum(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    inter_wh = (inter1 - inter0).clamp(min=0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    pred_wh = (pred_xyxy[:, 2:] - pred_xyxy[:, :2]).clamp(min=0.0)
    tgt_wh = (tgt_xyxy[:, 2:] - tgt_xyxy[:, :2]).clamp(min=0.0)
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    tgt_area = tgt_wh[:, 0] * tgt_wh[:, 1]
    union = pred_area + tgt_area - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))

    pred_center = pred_boxes[:, :2]
    tgt_center = target_boxes[:, :2]
    center_dist_sq = (pred_center - tgt_center).pow(2).sum(dim=-1)

    enc0 = torch.minimum(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    enc1 = torch.maximum(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    enc_wh = (enc1 - enc0).clamp(min=1.0e-8)
    enc_diag_sq = enc_wh.pow(2).sum(dim=-1).clamp(min=1.0e-8)

    v = (4.0 / (torch.pi**2)) * (
        torch.atan(tgt_wh[:, 0] / tgt_wh[:, 1].clamp(min=1.0e-8))
        - torch.atan(pred_wh[:, 0] / pred_wh[:, 1].clamp(min=1.0e-8))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v).clamp(min=1.0e-8)
    ciou = iou - (center_dist_sq / enc_diag_sq) - alpha * v
    return (1.0 - ciou).mean()


def compute_losses(
    preds: Dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    target_heatmaps: Dict[str, torch.Tensor],
    target_objectness: torch.Tensor,
    *,
    heatmap_weight: float = 1.0,
    box_weight: float = 1.0,
    objectness_weight: float = 1.0,
    box_l1_weight: float = 1.0,
    box_ciou_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    heat_total = torch.tensor(0.0, device=target_boxes.device)
    heat_enabled = False
    for rep, target in target_heatmaps.items():
        if rep not in preds["heatmaps"]:
            continue
        heat_enabled = True
        loss = F.binary_cross_entropy_with_logits(preds["heatmaps"][rep], target)
        heat_total = heat_total + loss
        metrics[f"heatmap_{rep}"] = float(loss.item())
    objectness_loss = F.binary_cross_entropy_with_logits(preds["objectness_logits"], target_objectness)
    pos_mask = target_objectness.bool()
    if bool(pos_mask.any()):
        pred_pos = preds["boxes"][pos_mask]
        target_pos = target_boxes[pos_mask]
        box_l1_loss = F.l1_loss(pred_pos, target_pos)
        box_ciou_loss = _ciou_loss(pred_pos, target_pos)
        box_loss = box_l1_weight * box_l1_loss + box_ciou_weight * box_ciou_loss
    else:
        box_l1_loss = torch.tensor(0.0, device=target_boxes.device)
        box_ciou_loss = torch.tensor(0.0, device=target_boxes.device)
        box_loss = torch.tensor(0.0, device=target_boxes.device)
    total = (
        heatmap_weight * heat_total
        + box_weight * box_loss
        + objectness_weight * objectness_loss
    )
    metrics["loss"] = float(total.item())
    metrics["box_regression"] = float(box_loss.item())
    metrics["box_l1"] = float(box_l1_loss.item())
    metrics["box_ciou"] = float(box_ciou_loss.item())
    if heat_enabled:
        metrics["heatmap_total"] = float(heat_total.item())
    metrics["objectness_bce"] = float(objectness_loss.item())
    return total, metrics
