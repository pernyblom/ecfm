from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_losses(
    pred: Dict[str, torch.Tensor],
    target_centers: torch.Tensor,
    *,
    center_weight: float = 1.0,
    pose_reg_weight: float = 1.0e-3,
    intr_reg_weight: float = 1.0e-3,
    acc_reg_weight: float = 1.0e-4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    center_loss = F.l1_loss(pred["pred_centers"], target_centers)
    pose_reg = pred["pose_delta"].pow(2).mean()
    intr_reg = pred["intrinsics_delta"].pow(2).mean()
    acc_reg = pred["dynamics_acc"].pow(2).mean()

    total = (
        center_weight * center_loss
        + pose_reg_weight * pose_reg
        + intr_reg_weight * intr_reg
        + acc_reg_weight * acc_reg
    )

    metrics = {
        "loss": float(total.item()),
        "center_l1": float(center_loss.item()),
        "pose_reg": float(pose_reg.item()),
        "intr_reg": float(intr_reg.item()),
        "acc_reg": float(acc_reg.item()),
    }
    return total, metrics
