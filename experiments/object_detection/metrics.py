from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def heatmap_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = pred_logits.sigmoid() >= threshold
    tgt = target >= threshold
    inter = (pred & tgt).sum(dim=(1, 2, 3)).float()
    union = (pred | tgt).sum(dim=(1, 2, 3)).float()
    return torch.where(union > 0, inter / union, torch.zeros_like(union)).mean()


def boxes_xyxy_px(boxes: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    wh = torch.tensor([frame_size[0], frame_size[1]], device=boxes.device, dtype=boxes.dtype)
    xy = boxes[..., :2] * wh
    box_wh = boxes[..., 2:] * wh
    xy0 = xy - box_wh / 2.0
    xy1 = xy + box_wh / 2.0
    return torch.cat([xy0, xy1], dim=-1)


def aligned_box_iou(pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    pred_xyxy = boxes_xyxy_px(pred, frame_size)
    tgt_xyxy = boxes_xyxy_px(target, frame_size)
    inter0 = torch.maximum(pred_xyxy[..., :2], tgt_xyxy[..., :2])
    inter1 = torch.minimum(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])
    inter_wh = (inter1 - inter0).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0.0) * (
        pred_xyxy[..., 3] - pred_xyxy[..., 1]
    ).clamp(min=0.0)
    tgt_area = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]).clamp(min=0.0) * (
        tgt_xyxy[..., 3] - tgt_xyxy[..., 1]
    ).clamp(min=0.0)
    union = pred_area + tgt_area - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def pairwise_box_iou(pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    pred_xyxy = boxes_xyxy_px(pred, frame_size)
    tgt_xyxy = boxes_xyxy_px(target, frame_size)
    inter0 = torch.maximum(pred_xyxy[:, None, :2], tgt_xyxy[None, :, :2])
    inter1 = torch.minimum(pred_xyxy[:, None, 2:], tgt_xyxy[None, :, 2:])
    inter_wh = (inter1 - inter0).clamp(min=0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0.0) * (
        pred_xyxy[:, 3] - pred_xyxy[:, 1]
    ).clamp(min=0.0)
    tgt_area = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]).clamp(min=0.0) * (
        tgt_xyxy[:, 3] - tgt_xyxy[:, 1]
    ).clamp(min=0.0)
    union = pred_area[:, None] + tgt_area[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def detection_scores(preds: Dict[str, torch.Tensor]) -> torch.Tensor:
    return preds["objectness_logits"].sigmoid()


def _matched_center_l1_px(pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    scale = torch.tensor([frame_size[0], frame_size[1]], device=pred.device, dtype=pred.dtype)
    return ((pred[:, :2] - target[:, :2]) * scale).abs().mean()


def _matched_box_iou(pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    return aligned_box_iou(pred, target, frame_size).mean()


def build_detections(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    target_boxes_list: Sequence[torch.Tensor],
    frame_keys: Sequence[str],
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    detections: list[dict] = []
    gt_by_frame: dict[str, torch.Tensor] = {}
    for frame_idx, frame_key in enumerate(frame_keys):
        gt_by_frame[frame_key] = target_boxes_list[frame_idx].detach().cpu()
        for query_idx in range(pred_boxes.shape[1]):
            detections.append(
                {
                    "frame_key": frame_key,
                    "score": float(pred_scores[frame_idx, query_idx].item()),
                    "box": pred_boxes[frame_idx, query_idx].detach().cpu(),
                }
            )
    return detections, gt_by_frame


def _prepare_detections_for_map(
    detections: list[dict],
    gt_by_frame: dict[str, torch.Tensor],
    frame_size: tuple[int, int],
) -> tuple[list[dict], dict[str, torch.Tensor], int]:
    if not detections:
        return [], {}, int(sum(int(boxes.shape[0]) for boxes in gt_by_frame.values()))
    ordered = sorted(detections, key=lambda item: item["score"], reverse=True)
    frame_to_det_indices: dict[str, list[int]] = {}
    for det_idx, det in enumerate(ordered):
        frame_to_det_indices.setdefault(det["frame_key"], []).append(det_idx)
    ious_by_frame: dict[str, torch.Tensor] = {}
    for frame_key, det_indices in frame_to_det_indices.items():
        gt_boxes = gt_by_frame[frame_key]
        if gt_boxes.numel() == 0:
            continue
        det_boxes = torch.stack([ordered[idx]["box"] for idx in det_indices], dim=0)
        ious_by_frame[frame_key] = pairwise_box_iou(det_boxes, gt_boxes, frame_size)
    return ordered, ious_by_frame, int(sum(int(boxes.shape[0]) for boxes in gt_by_frame.values()))


def average_precision_at_iou(
    ordered_detections: list[dict],
    gt_by_frame: dict[str, torch.Tensor],
    ious_by_frame: dict[str, torch.Tensor],
    iou_threshold: float,
    num_gt: int,
) -> float:
    if num_gt == 0 or not ordered_detections:
        return 0.0
    used = {
        frame_key: torch.zeros((boxes.shape[0],), dtype=torch.bool)
        for frame_key, boxes in gt_by_frame.items()
    }
    tp_vals: list[float] = []
    fp_vals: list[float] = []
    frame_offsets = {frame_key: 0 for frame_key in ious_by_frame}
    for det in ordered_detections:
        frame_key = det["frame_key"]
        gt_boxes = gt_by_frame[frame_key]
        if gt_boxes.numel() == 0:
            tp_vals.append(0.0)
            fp_vals.append(1.0)
            continue
        ious = ious_by_frame[frame_key][frame_offsets[frame_key]]
        frame_offsets[frame_key] += 1
        best_iou, best_idx = torch.max(ious, dim=0)
        if float(best_iou.item()) >= iou_threshold and not bool(used[frame_key][best_idx].item()):
            used[frame_key][best_idx] = True
            tp_vals.append(1.0)
            fp_vals.append(0.0)
        else:
            tp_vals.append(0.0)
            fp_vals.append(1.0)
    tp = torch.tensor(tp_vals, dtype=torch.float32)
    fp = torch.tensor(fp_vals, dtype=torch.float32)
    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0e-8)
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(precision.numel() - 2, -1, -1):
        precision[i] = torch.maximum(precision[i], precision[i + 1])
    delta = recall[1:] - recall[:-1]
    return float(torch.sum(delta * precision[1:]).item())


def map_metrics(
    detections: list[dict],
    gt_by_frame: dict[str, torch.Tensor],
    frame_size: tuple[int, int],
) -> Dict[str, float]:
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    ordered_detections, ious_by_frame, num_gt = _prepare_detections_for_map(detections, gt_by_frame, frame_size)
    ap_values = [
        average_precision_at_iou(
            ordered_detections=ordered_detections,
            gt_by_frame=gt_by_frame,
            ious_by_frame=ious_by_frame,
            iou_threshold=thr,
            num_gt=num_gt,
        )
        for thr in thresholds
    ]
    return {"mAP_50": ap_values[0], "mAP_50_95": float(sum(ap_values) / len(ap_values))}


def summarize_metrics(
    preds: Dict[str, torch.Tensor],
    target_boxes_list: Sequence[torch.Tensor],
    target_heatmaps: Dict[str, torch.Tensor],
    target_objectness: torch.Tensor,
    frame_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    frame_keys: Sequence[str],
    frame_size: tuple[int, int],
    include_map: bool = False,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pred_scores = detection_scores(preds)
    pred_boxes = preds["boxes"]

    matched_pred_boxes: List[torch.Tensor] = []
    matched_gt_boxes: List[torch.Tensor] = []
    for batch_idx, (pred_idx, gt_idx) in enumerate(frame_matches):
        if pred_idx.numel() == 0:
            continue
        gt_boxes = target_boxes_list[batch_idx].to(pred_boxes.device)
        matched_pred_boxes.append(pred_boxes[batch_idx, pred_idx])
        matched_gt_boxes.append(gt_boxes[gt_idx])
    if matched_pred_boxes:
        pred_cat = torch.cat(matched_pred_boxes, dim=0)
        gt_cat = torch.cat(matched_gt_boxes, dim=0)
        out["matched_center_l1_px"] = float(_matched_center_l1_px(pred_cat, gt_cat, frame_size).item())
        out["matched_box_iou"] = float(_matched_box_iou(pred_cat, gt_cat, frame_size).item())
    pred_label = pred_scores >= 0.5
    out["objectness_acc"] = float((pred_label == target_objectness.bool()).float().mean().item())
    if target_objectness.bool().any():
        out["objectness_pos_mean"] = float(pred_scores[target_objectness.bool()].mean().item())
    neg_mask = ~target_objectness.bool()
    if neg_mask.any():
        out["objectness_neg_mean"] = float(pred_scores[neg_mask].mean().item())
    if include_map:
        detections, gt_by_frame = build_detections(pred_boxes.detach(), pred_scores.detach(), target_boxes_list, frame_keys)
        out.update(map_metrics(detections, gt_by_frame, frame_size))
    for rep, target in target_heatmaps.items():
        if rep in preds["heatmaps"]:
            out[f"heatmap_iou_{rep}"] = float(heatmap_iou(preds["heatmaps"][rep], target).item())
    return out
