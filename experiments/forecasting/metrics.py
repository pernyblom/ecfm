from __future__ import annotations

import torch


def ade_fde(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # pred/target: [B, T, 4] in normalized coords
    pred_xy = pred[..., :2]
    tgt_xy = target[..., :2]
    dist = torch.linalg.norm(pred_xy - tgt_xy, dim=-1)  # [B, T]
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde
