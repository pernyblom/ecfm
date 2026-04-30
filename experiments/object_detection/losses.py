from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence

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


def _draw_gaussian(heatmap: torch.Tensor, cx: int, cy: int, radius: int) -> None:
    height, width = heatmap.shape
    diameter = 2 * radius + 1
    x = torch.arange(diameter, device=heatmap.device, dtype=heatmap.dtype) - radius
    y = x[:, None]
    sigma = max(diameter / 6.0, 1.0e-6)
    gaussian = torch.exp(-(x[None, :].pow(2) + y.pow(2)) / (2.0 * sigma * sigma))
    left = min(cx, radius)
    right = min(width - cx - 1, radius)
    top = min(cy, radius)
    bottom = min(height - cy - 1, radius)
    if right < 0 or bottom < 0:
        return
    patch = heatmap[cy - top : cy + bottom + 1, cx - left : cx + right + 1]
    gpatch = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    torch.maximum(patch, gpatch, out=patch)


def _build_centernet_targets(
    target_boxes_list: Sequence[torch.Tensor],
    target_velocities_list: Sequence[torch.Tensor] | None,
    target_velocity_masks: Sequence[torch.Tensor] | None,
    *,
    out_h: int,
    out_w: int,
    device: torch.device,
    dtype: torch.dtype,
    gaussian_radius: int,
) -> Dict[str, torch.Tensor]:
    batch_size = len(target_boxes_list)
    heatmap = torch.zeros((batch_size, 1, out_h, out_w), device=device, dtype=dtype)
    size = torch.zeros((batch_size, 2, out_h, out_w), device=device, dtype=dtype)
    offset = torch.zeros((batch_size, 2, out_h, out_w), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, 1, out_h, out_w), device=device, dtype=dtype)
    velocity = torch.zeros((batch_size, 2, out_h, out_w), device=device, dtype=dtype)
    velocity_mask = torch.zeros((batch_size, 1, out_h, out_w), device=device, dtype=dtype)
    grid_scale = torch.tensor([out_w, out_h], device=device, dtype=dtype)
    for batch_idx, boxes_raw in enumerate(target_boxes_list):
        boxes = boxes_raw.to(device=device, dtype=dtype)
        velocities = (
            target_velocities_list[batch_idx].to(device=device, dtype=dtype)
            if target_velocities_list is not None
            else torch.zeros((boxes.shape[0], 2), device=device, dtype=dtype)
        )
        velocity_valid = (
            target_velocity_masks[batch_idx].to(device=device)
            if target_velocity_masks is not None
            else torch.zeros((boxes.shape[0],), device=device, dtype=torch.bool)
        )
        for obj_idx, box in enumerate(boxes):
            center_grid = box[:2] * grid_scale
            xy_int = torch.floor(center_grid).long()
            cx = int(xy_int[0].clamp(0, out_w - 1).item())
            cy = int(xy_int[1].clamp(0, out_h - 1).item())
            _draw_gaussian(heatmap[batch_idx, 0], cx, cy, gaussian_radius)
            size[batch_idx, :, cy, cx] = box[2:].clamp(0.0, 1.0)
            offset[batch_idx, :, cy, cx] = center_grid - xy_int.to(dtype)
            mask[batch_idx, :, cy, cx] = 1.0
            if obj_idx < velocities.shape[0] and bool(velocity_valid[obj_idx].item()):
                velocity[batch_idx, :, cy, cx] = velocities[obj_idx]
                velocity_mask[batch_idx, :, cy, cx] = 1.0
    return {
        "heatmap": heatmap,
        "size": size,
        "offset": offset,
        "mask": mask,
        "velocity": velocity,
        "velocity_mask": velocity_mask,
    }


def _centernet_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = logits.sigmoid().clamp(1.0e-4, 1.0 - 1.0e-4)
    pos = target.eq(1.0)
    neg = target.lt(1.0)
    neg_weights = (1.0 - target).pow(4)
    pos_loss = torch.log(pred) * (1.0 - pred).pow(2) * pos
    neg_loss = torch.log(1.0 - pred) * pred.pow(2) * neg_weights * neg
    num_pos = pos.float().sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (pred - target).abs().mul(mask).sum() / denom


def compute_centernet_losses(
    preds: Dict[str, torch.Tensor],
    target_boxes_list: Sequence[torch.Tensor],
    target_velocities_list: Sequence[torch.Tensor] | None = None,
    target_velocity_masks: Sequence[torch.Tensor] | None = None,
    *,
    heatmap_weight: float = 1.0,
    size_weight: float = 1.0,
    offset_weight: float = 1.0,
    velocity_weight: float = 0.0,
    gaussian_radius: int = 2,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, object]]:
    logits = preds["centernet_heatmap_logits"]
    device = logits.device
    dtype = logits.dtype
    _, _, out_h, out_w = logits.shape
    targets = _build_centernet_targets(
        target_boxes_list,
        target_velocities_list,
        target_velocity_masks,
        out_h=out_h,
        out_w=out_w,
        device=device,
        dtype=dtype,
        gaussian_radius=int(gaussian_radius),
    )
    size_pred = preds["centernet_size_raw"].sigmoid()
    offset_pred = preds["centernet_offset_raw"].sigmoid()
    heatmap_loss = _centernet_focal_loss(logits, targets["heatmap"])
    size_loss = _masked_l1(size_pred, targets["size"], targets["mask"])
    offset_loss = _masked_l1(offset_pred, targets["offset"], targets["mask"])
    velocity_loss = torch.tensor(0.0, device=device)
    if velocity_weight > 0 and "centernet_velocity" in preds:
        velocity_loss = _masked_l1(preds["centernet_velocity"], targets["velocity"], targets["velocity_mask"])
    total = (
        heatmap_weight * heatmap_loss
        + size_weight * size_loss
        + offset_weight * offset_loss
        + velocity_weight * velocity_loss
    )
    metrics = {
        "loss": float(total.item()),
        "centernet_heatmap": float(heatmap_loss.item()),
        "centernet_size": float(size_loss.item()),
        "centernet_offset": float(offset_loss.item()),
        "centernet_velocity": float(velocity_loss.item()),
        "centernet_num_objects": float(targets["mask"].sum().item()),
        "centernet_num_velocity": float(targets["velocity_mask"].sum().item()),
    }
    aux: Dict[str, object] = {"centernet_targets": targets}
    return total, metrics, aux


def compute_detr_lite_losses(
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
        with torch.no_grad():
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


def compute_losses(
    preds: Dict[str, torch.Tensor],
    target_boxes_list: List[torch.Tensor],
    target_heatmaps: Dict[str, torch.Tensor],
    *,
    target_velocities_list: Sequence[torch.Tensor] | None = None,
    target_velocity_masks: Sequence[torch.Tensor] | None = None,
    heatmap_weight: float = 1.0,
    box_weight: float = 1.0,
    objectness_weight: float = 1.0,
    box_l1_weight: float = 1.0,
    box_ciou_weight: float = 1.0,
    match_score_weight: float = 1.0,
    match_l1_weight: float = 1.0,
    match_ciou_weight: float = 1.0,
    centernet_size_weight: float = 1.0,
    centernet_offset_weight: float = 1.0,
    velocity_weight: float = 0.0,
    gaussian_radius: int = 2,
) -> tuple[torch.Tensor, Dict[str, float], Dict[str, object]]:
    if preds.get("detector_type") == "centernet":
        return compute_centernet_losses(
            preds,
            target_boxes_list,
            target_velocities_list,
            target_velocity_masks,
            heatmap_weight=heatmap_weight,
            size_weight=centernet_size_weight,
            offset_weight=centernet_offset_weight,
            velocity_weight=velocity_weight,
            gaussian_radius=gaussian_radius,
        )
    return compute_detr_lite_losses(
        preds,
        target_boxes_list,
        target_heatmaps,
        heatmap_weight=heatmap_weight,
        box_weight=box_weight,
        objectness_weight=objectness_weight,
        box_l1_weight=box_l1_weight,
        box_ciou_weight=box_ciou_weight,
        match_score_weight=match_score_weight,
        match_l1_weight=match_l1_weight,
        match_ciou_weight=match_ciou_weight,
    )
