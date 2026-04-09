from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _step_simple_regularizer(z: torch.Tensor, target_std: float, eps: float) -> torch.Tensor:
    mean_loss = z.mean(dim=0).pow(2).mean()
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    std_loss = (std - target_std).pow(2).mean()
    return mean_loss + std_loss


def _step_vicreg_regularizer(
    z: torch.Tensor,
    target_std: float,
    eps: float,
    cov_weight: float,
) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(target_std - std).mean()
    centered = z - z.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(z.shape[0] - 1, 1)
    cov_loss = _off_diagonal(cov).pow(2).mean() if cov.shape[0] > 1 else z.new_tensor(0.0)
    return var_loss + cov_weight * cov_loss


def _step_sigreg_regularizer(
    z: torch.Tensor,
    num_projections: int,
    num_integration_steps: int,
    projection_lambda: float,
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    d = z.shape[-1]
    dirs = torch.randn(d, num_projections, device=z.device, dtype=z.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
    h = z @ dirs
    t = torch.linspace(t_min, t_max, num_integration_steps, device=z.device, dtype=z.dtype)
    ht = h.unsqueeze(-1) * t.view(1, 1, -1)
    ecf_real = torch.cos(ht).mean(dim=0)
    ecf_imag = torch.sin(ht).mean(dim=0)
    phi0 = torch.exp(-0.5 * t.square()).view(1, -1)
    weight = torch.exp(-t.square() / (2.0 * (projection_lambda ** 2))).view(1, -1)
    diff_sq = (ecf_real - phi0).square() + ecf_imag.square()
    integral = torch.trapz(weight * diff_sq, t, dim=-1)
    return integral.mean()


def anti_collapse_loss(latents: torch.Tensor, reg_cfg: Dict) -> torch.Tensor:
    reg_type = reg_cfg.get("type", "sigreg")
    eps = float(reg_cfg.get("eps", 1.0e-4))
    target_std = float(reg_cfg.get("target_std", 1.0))
    losses = []
    for t in range(latents.shape[1]):
        z = latents[:, t]
        if reg_type == "simple":
            losses.append(_step_simple_regularizer(z, target_std=target_std, eps=eps))
        elif reg_type == "vicreg":
            losses.append(
                _step_vicreg_regularizer(
                    z,
                    target_std=target_std,
                    eps=eps,
                    cov_weight=float(reg_cfg.get("cov_weight", 1.0)),
                )
            )
        elif reg_type == "sigreg":
            losses.append(
                _step_sigreg_regularizer(
                    z,
                    num_projections=int(reg_cfg.get("num_projections", 256)),
                    num_integration_steps=int(reg_cfg.get("num_integration_steps", 64)),
                    projection_lambda=float(reg_cfg.get("projection_lambda", 1.0)),
                    t_min=float(reg_cfg.get("t_min", 0.2)),
                    t_max=float(reg_cfg.get("t_max", 4.0)),
                )
            )
        else:
            raise ValueError(f"Unknown regularizer type: {reg_type}")
    return torch.stack(losses).mean()


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    future_boxes: torch.Tensor,
    train_cfg: Dict,
    reg_cfg: Dict,
) -> tuple[torch.Tensor, Dict[str, float]]:
    pred_weight = float(train_cfg.get("prediction_weight", 1.0))
    reg_weight = float(train_cfg.get("regularizer_weight", 0.1))
    forecast_weight = float(train_cfg.get("forecast_weight", 0.0))
    pred_loss = F.mse_loss(outputs["pred_future_teacher"], outputs["future_latents"])
    reg_loss = anti_collapse_loss(outputs["latents"], reg_cfg)
    total = pred_weight * pred_loss + reg_weight * reg_loss

    box_loss = outputs["latents"].new_tensor(0.0)
    if "pred_future_boxes" in outputs:
        box_loss = F.l1_loss(outputs["pred_future_boxes"], future_boxes)
        total = total + forecast_weight * box_loss

    latent_std = outputs["latents"].std(dim=0, unbiased=False).mean()
    latent_norm = outputs["latents"].norm(dim=-1).mean()
    metrics = {
        "loss": float(total.item()),
        "pred_loss": float(pred_loss.item()),
        "reg_loss": float(reg_loss.item()),
        "box_loss": float(box_loss.item()),
        "latent_std": float(latent_std.item()),
        "latent_norm": float(latent_norm.item()),
    }
    return total, metrics
