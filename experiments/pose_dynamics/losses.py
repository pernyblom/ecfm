from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def compute_losses(
    pred: Dict[str, torch.Tensor],
    target_centers: torch.Tensor,
    *,
    target_sizes: Optional[torch.Tensor] = None,
    center_weight: float = 1.0,
    size_weight: float = 0.0,
    size_min: Optional[Sequence[float]] = None,
    size_max: Optional[Sequence[float]] = None,
    pose_reg_weight: float = 1.0e-3,
    intr_reg_weight: float = 1.0e-3,
    acc_reg_weight: float = 1.0e-4,
    speed_bound: Optional[float] = None,
    speed_bound_weight: float = 0.0,
    acc_bound: Optional[float] = None,
    acc_bound_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    center_loss = F.l1_loss(pred["pred_centers"], target_centers)

    size_loss = pred["pred_centers"].new_zeros(())
    if (
        size_weight > 0.0
        and target_sizes is not None
        and size_min is not None
        and size_max is not None
    ):
        fx = pred["intrinsics_corrected"][:, 0:1].unsqueeze(1)
        fy = pred["intrinsics_corrected"][:, 1:2].unsqueeze(1)
        z = pred["points_cam"][:, :, 2:3].clamp_min(1.0e-6)
        size_min_t = target_centers.new_tensor(size_min).view(1, 1, 2)
        size_max_t = target_centers.new_tensor(size_max).view(1, 1, 2)
        size_scale = torch.cat([fx, fy], dim=-1)
        pred_size_min = size_scale * size_min_t / z
        pred_size_max = size_scale * size_max_t / z
        below = (pred_size_min - target_sizes).clamp_min(0.0)
        above = (target_sizes - pred_size_max).clamp_min(0.0)
        size_loss = (below + above).mean()

    pose_reg = pred["pose_delta"].pow(2).mean()
    intr_reg = pred["intrinsics_delta"].pow(2).mean()
    acc_reg = pred["dynamics_acc"].pow(2).mean()

    speed_bound_penalty = pred["pred_centers"].new_zeros(())
    if speed_bound_weight > 0.0 and speed_bound is not None:
        speed_norm = pred["dynamics_vel_seq"].norm(dim=-1)
        speed_bound_penalty = (speed_norm - float(speed_bound)).clamp_min(0.0).pow(2).mean()

    acc_bound_penalty = pred["pred_centers"].new_zeros(())
    if acc_bound_weight > 0.0 and acc_bound is not None:
        acc_norm = pred["dynamics_acc"].norm(dim=-1)
        acc_bound_penalty = (acc_norm - float(acc_bound)).clamp_min(0.0).pow(2).mean()

    total = (
        center_weight * center_loss
        + size_weight * size_loss
        + pose_reg_weight * pose_reg
        + intr_reg_weight * intr_reg
        + acc_reg_weight * acc_reg
        + speed_bound_weight * speed_bound_penalty
        + acc_bound_weight * acc_bound_penalty
    )

    metrics = {
        "loss": float(total.item()),
        "center_l1": float(center_loss.item()),
        "size_range": float(size_loss.item()),
        "pose_reg": float(pose_reg.item()),
        "intr_reg": float(intr_reg.item()),
        "acc_reg": float(acc_reg.item()),
        "speed_bound": float(speed_bound_penalty.item()),
        "acc_bound": float(acc_bound_penalty.item()),
    }
    return total, metrics
