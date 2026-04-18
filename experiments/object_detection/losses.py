from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import torch
import torch.nn.functional as F


def _boxes_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    xy0 = boxes[..., :2] - boxes[..., 2:] / 2.0
    xy1 = boxes[..., :2] + boxes[..., 2:] / 2.0
    return torch.cat([xy0, xy1], dim=-1)


def _ciou_per_box(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred_xyxy = _boxes_xyxy(pred_boxes)
    tgt_xyxy = _boxes_xyxy(target_boxes)

    inter0 = torch.maximum(pred_xyxy[..., :2], tgt_xyxy[..., :2])
    inter1 = torch.minimum(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])
    inter_wh = (inter1 - inter0).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    pred_wh = (pred_xyxy[..., 2:] - pred_xyxy[..., :2]).clamp(min=0.0)
    tgt_wh = (tgt_xyxy[..., 2:] - tgt_xyxy[..., :2]).clamp(min=0.0)
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    tgt_area = tgt_wh[..., 0] * tgt_wh[..., 1]
    union = pred_area + tgt_area - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))

    pred_center = pred_boxes[..., :2]
    tgt_center = target_boxes[..., :2]
    center_dist_sq = (pred_center - tgt_center).pow(2).sum(dim=-1)

    enc0 = torch.minimum(pred_xyxy[..., :2], tgt_xyxy[..., :2])
    enc1 = torch.maximum(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])
    enc_wh = (enc1 - enc0).clamp(min=1.0e-8)
    enc_diag_sq = enc_wh.pow(2).sum(dim=-1).clamp(min=1.0e-8)

    v = (4.0 / (torch.pi**2)) * (
        torch.atan(tgt_wh[..., 0] / tgt_wh[..., 1].clamp(min=1.0e-8))
        - torch.atan(pred_wh[..., 0] / pred_wh[..., 1].clamp(min=1.0e-8))
    ).pow(2)
    alpha = v / (1.0 - iou + v).clamp(min=1.0e-8)
    ciou = iou - (center_dist_sq / enc_diag_sq) - alpha * v
    return ciou


def _pairwise_ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    q = pred_boxes.shape[0]
    n = target_boxes.shape[0]
    pred_exp = pred_boxes.unsqueeze(1).expand(q, n, 4)
    tgt_exp = target_boxes.unsqueeze(0).expand(q, n, 4)
    return 1.0 - _ciou_per_box(pred_exp, tgt_exp)


@lru_cache(maxsize=None)
def _mask_bit_count(mask: int) -> int:
    return int(mask.bit_count())


def _match_queries(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_queries, num_gt = cost.shape
    if num_gt == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=cost.device),
            torch.zeros((0,), dtype=torch.long, device=cost.device),
        )
    cost_cpu = cost.detach().cpu()
    inf = float("inf")
    dp: List[Dict[int, tuple[float, tuple[int, int] | None]]] = [{0: (0.0, None)}]
    for q in range(num_queries):
        prev = dp[-1]
        cur: Dict[int, tuple[float, tuple[int, int] | None]] = dict(prev)
        for mask, (base_cost, _) in prev.items():
            assigned = _mask_bit_count(mask)
            if assigned >= num_gt:
                continue
            for gt_idx in range(num_gt):
                if mask & (1 << gt_idx):
                    continue
                new_mask = mask | (1 << gt_idx)
                new_cost = base_cost + float(cost_cpu[q, gt_idx].item())
                old = cur.get(new_mask, (inf, None))[0]
                if new_cost < old:
                    cur[new_mask] = (new_cost, (mask, gt_idx))
        dp.append(cur)
    full_mask = (1 << num_gt) - 1
    if full_mask not in dp[-1]:
        raise RuntimeError("Failed to compute a complete query-to-gt matching.")
    mask = full_mask
    query_indices: List[int] = []
    gt_indices: List[int] = []
    for q in range(num_queries, 0, -1):
        entry = dp[q].get(mask)
        if entry is None:
            continue
        back = entry[1]
        if back is None:
            continue
        prev_mask, gt_idx = back
        if prev_mask != mask:
            query_indices.append(q - 1)
            gt_indices.append(gt_idx)
            mask = prev_mask
    query_indices.reverse()
    gt_indices.reverse()
    return (
        torch.tensor(query_indices, dtype=torch.long, device=cost.device),
        torch.tensor(gt_indices, dtype=torch.long, device=cost.device),
    )


def compute_losses(
    preds: Dict[str, torch.Tensor],
    target_boxes_list: List[torch.Tensor],
    target_heatmaps: Dict[str, torch.Tensor],
    *,
    heatmap_weight: float = 1.0,
    box_weight: float = 1.0,
    objectness_weight: float = 1.0,
    box_l1_weight: float = 1.0,
    box_ciou_weight: float = 1.0,
    match_score_weight: float = 1.0,
    match_l1_weight: float = 1.0,
    match_ciou_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, object]]:
    device = preds["boxes"].device
    metrics: Dict[str, float] = {}

    heat_total = torch.tensor(0.0, device=device)
    heat_enabled = False
    for rep, target in target_heatmaps.items():
        if rep not in preds["heatmaps"]:
            continue
        heat_enabled = True
        loss = F.binary_cross_entropy_with_logits(preds["heatmaps"][rep], target)
        heat_total = heat_total + loss
        metrics[f"heatmap_{rep}"] = float(loss.item())

    batch_size, num_queries, _ = preds["boxes"].shape
    pred_boxes = preds["boxes"]
    pred_logits = preds["objectness_logits"]
    target_objectness = torch.zeros((batch_size, num_queries), device=device, dtype=pred_logits.dtype)
    matched_pred_boxes: List[torch.Tensor] = []
    matched_gt_boxes: List[torch.Tensor] = []
    frame_matches: List[tuple[torch.Tensor, torch.Tensor]] = []
    total_matches = 0

    for batch_idx in range(batch_size):
        gt_boxes = target_boxes_list[batch_idx].to(device)
        if gt_boxes.shape[0] > num_queries:
            areas = gt_boxes[:, 2] * gt_boxes[:, 3]
            keep = torch.argsort(areas, descending=True)[:num_queries]
            gt_boxes = gt_boxes[keep]
        if gt_boxes.numel() == 0:
            frame_matches.append(
                (
                    torch.zeros((0,), dtype=torch.long, device=device),
                    torch.zeros((0,), dtype=torch.long, device=device),
                )
            )
            continue
        box_l1_cost = (pred_boxes[batch_idx].unsqueeze(1) - gt_boxes.unsqueeze(0)).abs().mean(dim=-1)
        box_ciou_cost = _pairwise_ciou_loss(pred_boxes[batch_idx], gt_boxes)
        score_cost = -pred_logits[batch_idx].sigmoid().unsqueeze(1).expand(-1, gt_boxes.shape[0])
        cost = (
            match_l1_weight * box_l1_cost
            + match_ciou_weight * box_ciou_cost
            + match_score_weight * score_cost
        )
        pred_idx, gt_idx = _match_queries(cost)
        frame_matches.append((pred_idx, gt_idx))
        if pred_idx.numel() > 0:
            target_objectness[batch_idx, pred_idx] = 1.0
            matched_pred_boxes.append(pred_boxes[batch_idx, pred_idx])
            matched_gt_boxes.append(gt_boxes[gt_idx])
            total_matches += int(pred_idx.numel())

    objectness_loss = F.binary_cross_entropy_with_logits(pred_logits, target_objectness)
    if matched_pred_boxes:
        pred_cat = torch.cat(matched_pred_boxes, dim=0)
        gt_cat = torch.cat(matched_gt_boxes, dim=0)
        box_l1_loss = F.l1_loss(pred_cat, gt_cat)
        box_ciou_loss = (1.0 - _ciou_per_box(pred_cat, gt_cat)).mean()
        box_loss = box_l1_weight * box_l1_loss + box_ciou_weight * box_ciou_loss
    else:
        box_l1_loss = torch.tensor(0.0, device=device)
        box_ciou_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)

    total = heatmap_weight * heat_total + box_weight * box_loss + objectness_weight * objectness_loss
    metrics["loss"] = float(total.item())
    metrics["box_regression"] = float(box_loss.item())
    metrics["box_l1"] = float(box_l1_loss.item())
    metrics["box_ciou"] = float(box_ciou_loss.item())
    metrics["objectness_bce"] = float(objectness_loss.item())
    metrics["num_matches"] = float(total_matches)
    if heat_enabled:
        metrics["heatmap_total"] = float(heat_total.item())

    aux: Dict[str, object] = {
        "target_objectness": target_objectness.detach(),
        "frame_matches": frame_matches,
        "matched_pred_boxes": matched_pred_boxes,
        "matched_gt_boxes": matched_gt_boxes,
    }
    return total, metrics, aux
