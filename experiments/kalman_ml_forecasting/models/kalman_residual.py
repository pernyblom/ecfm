from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from experiments.object_detection.models.backbones import build_single_encoder, grid_split_from_rep_name


def box_sequence_to_state(past_boxes: torch.Tensor, past_times_s: torch.Tensor) -> torch.Tensor:
    """Return [cx, cy, w, h, vx, vy, vw, vh] from the last two observed boxes."""
    last = past_boxes[:, -1]
    if past_boxes.shape[1] < 2:
        velocity = torch.zeros_like(last)
    else:
        prev = past_boxes[:, -2]
        dt = (past_times_s[:, -1] - past_times_s[:, -2]).clamp(min=1.0e-6).unsqueeze(-1)
        velocity = (last - prev) / dt
    return torch.cat([last, velocity], dim=-1)


def constant_velocity_forecast(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
) -> torch.Tensor:
    state = box_sequence_to_state(past_boxes, past_times_s)
    pos = state[:, :4]
    vel = state[:, 4:]
    dt = future_times_s - past_times_s[:, -1:].expand_as(future_times_s)
    return (pos.unsqueeze(1) + vel.unsqueeze(1) * dt.unsqueeze(-1)).clamp(0.0, 1.0)


def _encoder_cfg_for_rep(
    backbone_cfg: Dict,
    rep: str,
    *,
    cell_local_first_conv: bool,
    cell_local_first_conv_representations: List[str] | None,
) -> Dict:
    cfg = dict(backbone_cfg)
    if not cell_local_first_conv:
        return cfg
    allowed = set(cell_local_first_conv_representations or [])
    grid = grid_split_from_rep_name(rep)
    if allowed:
        if rep not in allowed:
            return cfg
        if grid is None:
            raise ValueError(f"cell_local_first_conv representation '{rep}' has no NxM suffix.")
    elif grid is None:
        return cfg
    cfg["cell_local_first_conv_grid"] = grid
    return cfg


class KalmanResidualForecaster(nn.Module):
    """Constant-velocity Kalman-style rollout with image-conditioned residuals.

    State layout is [cx, cy, w, h, vx, vy, vw, vh]. The fixed transition is a
    constant-velocity step. The learned branch predicts acceleration residuals
    for the four box channels; each step applies:

        p' = p + v dt + 0.5 a dt^2
        v' = v + a dt

    This is equivalent to x_{t+1} = F x_t + delta_x_t, where delta_x_t is
    predicted from event/RGB representations and the observed track history.
    """

    def __init__(
        self,
        *,
        representations: List[str],
        image_sizes: Dict[str, Tuple[int, int]],
        backbone_cfg: Dict,
        history_steps: int,
        fusion_hidden_dim: int = 256,
        state_hidden_dim: int = 128,
        residual_hidden_dim: int = 256,
        residual_scale: float = 1.0,
        predict_size_residuals: bool = True,
        cell_local_first_conv: bool = False,
        cell_local_first_conv_representations: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.representations = list(representations)
        self.image_sizes = {
            str(rep): (int(size[0]), int(size[1])) for rep, size in dict(image_sizes).items()
        }
        self.history_steps = int(history_steps)
        self.residual_scale = float(residual_scale)
        self.predict_size_residuals = bool(predict_size_residuals)
        self.encoders = nn.ModuleDict(
            {
                rep: build_single_encoder(
                    _encoder_cfg_for_rep(
                        backbone_cfg,
                        rep,
                        cell_local_first_conv=bool(cell_local_first_conv),
                        cell_local_first_conv_representations=cell_local_first_conv_representations,
                    )
                )
                for rep in self.representations
            }
        )
        per_rep_dim = int(backbone_cfg.get("out_dim", 128))
        fused_dim = per_rep_dim * len(self.representations)
        self.image_fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(self.history_steps * 5, state_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.ReLU(inplace=True),
        )
        residual_out_dim = 4 if self.predict_size_residuals else 2
        self.residual_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim + state_hidden_dim + 9, residual_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(residual_hidden_dim, residual_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(residual_hidden_dim, residual_out_dim),
        )

    def _history_features(self, past_boxes: torch.Tensor, past_times_s: torch.Tensor) -> torch.Tensor:
        if past_boxes.shape[1] < self.history_steps:
            pad_count = self.history_steps - past_boxes.shape[1]
            box_pad = past_boxes[:, :1].expand(-1, pad_count, -1)
            time_pad = past_times_s[:, :1].expand(-1, pad_count)
            boxes = torch.cat([box_pad, past_boxes], dim=1)
            times = torch.cat([time_pad, past_times_s], dim=1)
        else:
            boxes = past_boxes[:, -self.history_steps :]
            times = past_times_s[:, -self.history_steps :]
        rel_times = times - times[:, -1:].expand_as(times)
        return self.history_encoder(torch.cat([boxes, rel_times.unsqueeze(-1)], dim=-1).flatten(1))

    def _image_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc = [self.encoders[rep](inputs[rep]).pooled for rep in self.representations]
        return self.image_fusion(torch.cat(enc, dim=-1))

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        past_boxes: torch.Tensor,
        past_times_s: torch.Tensor,
        future_times_s: torch.Tensor,
        *,
        return_debug: bool = False,
    ):
        image_feat = self._image_features(inputs)
        history_feat = self._history_features(past_boxes, past_times_s)
        state = box_sequence_to_state(past_boxes, past_times_s)
        current_time = past_times_s[:, -1]
        preds: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for step in range(future_times_s.shape[1]):
            next_time = future_times_s[:, step]
            dt = (next_time - current_time).clamp(min=1.0e-6)
            step_feat = torch.cat([image_feat, history_feat, state, dt.unsqueeze(-1)], dim=-1)
            accel = self.residual_head(step_feat) * self.residual_scale
            if not self.predict_size_residuals:
                accel = torch.cat([accel, torch.zeros_like(accel)], dim=-1)
            pos = state[:, :4]
            vel = state[:, 4:]
            next_pos = pos + vel * dt.unsqueeze(-1) + 0.5 * accel * dt.square().unsqueeze(-1)
            next_vel = vel + accel * dt.unsqueeze(-1)
            next_pos = next_pos.clamp(0.0, 1.0)
            state = torch.cat([next_pos, next_vel], dim=-1)
            current_time = next_time
            preds.append(next_pos)
            residuals.append(accel)
        pred = torch.stack(preds, dim=1)
        if not return_debug:
            return pred
        return {
            "boxes": pred,
            "residual_accel": torch.stack(residuals, dim=1),
            "cv_boxes": constant_velocity_forecast(past_boxes, past_times_s, future_times_s),
        }

