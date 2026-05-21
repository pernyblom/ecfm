from __future__ import annotations

import torch

from experiments.forecasting.metrics import ade_fde_bbox_px, ade_fde_center_px, miou


def summarize_forecast_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    frame_size: tuple[int, int],
) -> dict[str, float]:
    ade_bb, fde_bb = ade_fde_bbox_px(pred, target, frame_size)
    ade_c, fde_c = ade_fde_center_px(pred, target, frame_size)
    return {
        "ade_bbox_px": float(ade_bb.item()),
        "fde_bbox_px": float(fde_bb.item()),
        "ade_center_px": float(ade_c.item()),
        "fde_center_px": float(fde_c.item()),
        "miou": float(miou(pred, target, frame_size).item()),
    }

