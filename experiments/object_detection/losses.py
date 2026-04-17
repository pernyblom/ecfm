from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def compute_losses(
    preds: Dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    target_heatmaps: Dict[str, torch.Tensor],
    *,
    heatmap_weight: float = 1.0,
    box_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    heat_total = torch.tensor(0.0, device=target_boxes.device)
    metrics: Dict[str, float] = {}
    for rep, target in target_heatmaps.items():
        loss = F.binary_cross_entropy_with_logits(preds["heatmaps"][rep], target)
        heat_total = heat_total + loss
        metrics[f"heatmap_{rep}"] = float(loss.item())
    box_loss = F.l1_loss(preds["boxes"], target_boxes)
    total = heatmap_weight * heat_total + box_weight * box_loss
    metrics["loss"] = float(total.item())
    metrics["box_l1"] = float(box_loss.item())
    metrics["heatmap_total"] = float(heat_total.item())
    return total, metrics
