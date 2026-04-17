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


def _boxes_xyxy_px(boxes: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    wh = torch.tensor([frame_size[0], frame_size[1]], device=boxes.device, dtype=boxes.dtype)
    xy = boxes[:, :2] * wh
    box_wh = boxes[:, 2:] * wh
    xy0 = xy - box_wh / 2.0
    xy1 = xy + box_wh / 2.0
    return torch.cat([xy0, xy1], dim=-1)


def pairwise_box_iou(pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    pred_xyxy = _boxes_xyxy_px(pred, frame_size)
    tgt_xyxy = _boxes_xyxy_px(target, frame_size)
    inter0 = torch.maximum(pred_xyxy[:, :2], tgt_xyxy[:, :2])
    inter1 = torch.minimum(pred_xyxy[:, 2:], tgt_xyxy[:, 2:])
    inter_wh = (inter1 - inter0).clamp(min=0.0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0.0) * (
        pred_xyxy[:, 3] - pred_xyxy[:, 1]
    ).clamp(min=0.0)
    tgt_area = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0.0) * (
        tgt_xyxy[:, 3] - tgt_xyxy[:, 1]
    ).clamp(min=0.0)
    union = pred_area + tgt_area - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def detection_scores(preds: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "objectness_logits" in preds:
        return preds["objectness_logits"].sigmoid()
    return torch.ones(preds["boxes"].shape[0], device=preds["boxes"].device, dtype=preds["boxes"].dtype)


def average_precision_at_iou(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    gt_present: torch.Tensor,
    frame_size: tuple[int, int],
    iou_threshold: float,
) -> torch.Tensor:
    if pred_boxes.numel() == 0:
        return torch.tensor(0.0)
    valid_gt = gt_present.bool()
    num_gt = int(valid_gt.sum().item())
    if num_gt == 0:
        return torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    target_boxes = target_boxes[order]
    valid_gt = valid_gt[order]
    ious = pairwise_box_iou(pred_boxes, target_boxes, frame_size)

    tp = ((ious >= iou_threshold) & valid_gt).float()
    fp = 1.0 - tp
    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0e-8)

    recall = torch.cat(
        [
            torch.tensor([0.0], device=recall.device, dtype=recall.dtype),
            recall,
            torch.tensor([1.0], device=recall.device, dtype=recall.dtype),
        ]
    )
    precision = torch.cat(
        [
            torch.tensor([1.0], device=precision.device, dtype=precision.dtype),
            precision,
            torch.tensor([0.0], device=precision.device, dtype=precision.dtype),
        ]
    )
    for i in range(precision.numel() - 2, -1, -1):
        precision[i] = torch.maximum(precision[i], precision[i + 1])
    delta = recall[1:] - recall[:-1]
    return torch.sum(delta * precision[1:])


def map_metrics(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    gt_present: torch.Tensor,
    frame_size: tuple[int, int],
) -> Dict[str, float]:
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    ap_values = [
        average_precision_at_iou(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            target_boxes=target_boxes,
            gt_present=gt_present,
            frame_size=frame_size,
            iou_threshold=thr,
        )
        for thr in thresholds
    ]
    return {
        "mAP_50": float(ap_values[0].item()),
        "mAP_50_95": float(torch.stack(ap_values).mean().item()),
    }


def summarize_metrics(
    preds: Dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    target_heatmaps: Dict[str, torch.Tensor],
    gt_present: torch.Tensor,
    frame_size: tuple[int, int],
) -> Dict[str, float]:
    out = {
        "center_l1_px": float(box_center_l1_px(preds["boxes"], target_boxes, frame_size).item()),
        "box_iou": float(box_iou_mean(preds["boxes"], target_boxes, frame_size).item()),
    }
    out.update(
        map_metrics(
            pred_boxes=preds["boxes"],
            pred_scores=detection_scores(preds),
            target_boxes=target_boxes,
            gt_present=gt_present,
            frame_size=frame_size,
        )
    )
    if "objectness_logits" in preds:
        target_obj = gt_present.float()
        pred_obj = preds["objectness_logits"].sigmoid()
        pred_label = pred_obj >= 0.5
        out["objectness_acc"] = float((pred_label == gt_present.bool()).float().mean().item())
        if gt_present.any():
            out["objectness_pos_mean"] = float(pred_obj[gt_present.bool()].mean().item())
        else:
            out["objectness_pos_mean"] = 0.0
        if (~gt_present.bool()).any():
            out["objectness_neg_mean"] = float(pred_obj[(~gt_present.bool())].mean().item())
        else:
            out["objectness_neg_mean"] = 0.0
    for rep, target in target_heatmaps.items():
        if rep in preds["heatmaps"]:
            out[f"heatmap_iou_{rep}"] = float(heatmap_iou(preds["heatmaps"][rep], target).item())
    return out
