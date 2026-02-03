from __future__ import annotations

import torch


def ade_fde(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # pred/target: [B, T, 4] in normalized coords (cx, cy, w, h)
    pred_xy = pred[..., :2]
    tgt_xy = target[..., :2]
    dist = torch.linalg.norm(pred_xy - tgt_xy, dim=-1)  # [B, T]
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde


def _to_pixels(boxes: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    # boxes: [B, T, 4] normalized (cx, cy, w, h)
    w, h = float(frame_size[0]), float(frame_size[1])
    out = boxes.clone()
    out[..., 0] = out[..., 0] * w
    out[..., 1] = out[..., 1] * h
    out[..., 2] = out[..., 2] * w
    out[..., 3] = out[..., 3] * h
    return out


def ade_fde_center_px(
    pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_px = _to_pixels(pred, frame_size)[..., :2]
    tgt_px = _to_pixels(target, frame_size)[..., :2]
    dist = torch.linalg.norm(pred_px - tgt_px, dim=-1)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde


def ade_fde_bbox_px(
    pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_px = _to_pixels(pred, frame_size)
    tgt_px = _to_pixels(target, frame_size)
    dist = torch.linalg.norm(pred_px - tgt_px, dim=-1)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde


def miou(
    pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]
) -> torch.Tensor:
    # IoU over future timesteps, averaged over batch+time.
    pred_px = _to_pixels(pred, frame_size)
    tgt_px = _to_pixels(target, frame_size)

    pred_x0 = pred_px[..., 0] - pred_px[..., 2] / 2.0
    pred_y0 = pred_px[..., 1] - pred_px[..., 3] / 2.0
    pred_x1 = pred_px[..., 0] + pred_px[..., 2] / 2.0
    pred_y1 = pred_px[..., 1] + pred_px[..., 3] / 2.0

    tgt_x0 = tgt_px[..., 0] - tgt_px[..., 2] / 2.0
    tgt_y0 = tgt_px[..., 1] - tgt_px[..., 3] / 2.0
    tgt_x1 = tgt_px[..., 0] + tgt_px[..., 2] / 2.0
    tgt_y1 = tgt_px[..., 1] + tgt_px[..., 3] / 2.0

    inter_x0 = torch.max(pred_x0, tgt_x0)
    inter_y0 = torch.max(pred_y0, tgt_y0)
    inter_x1 = torch.min(pred_x1, tgt_x1)
    inter_y1 = torch.min(pred_y1, tgt_y1)

    inter_w = (inter_x1 - inter_x0).clamp(min=0)
    inter_h = (inter_y1 - inter_y0).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (pred_x1 - pred_x0).clamp(min=0) * (pred_y1 - pred_y0).clamp(min=0)
    tgt_area = (tgt_x1 - tgt_x0).clamp(min=0) * (tgt_y1 - tgt_y0).clamp(min=0)
    union = pred_area + tgt_area - inter_area
    iou = torch.where(union > 0, inter_area / union, torch.zeros_like(union))
    return iou.mean()
