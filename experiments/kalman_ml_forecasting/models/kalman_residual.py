from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from experiments.object_detection.models.backbones import build_single_encoder, grid_split_from_rep_name
from experiments.kalman_ml_forecasting.models.kalman_filter import kalman_filter_history


def _fit_recent_velocity(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    *,
    num_samples: int,
) -> torch.Tensor:
    if past_boxes.shape[1] < 2:
        return torch.zeros_like(past_boxes[:, -1])
    boxes = past_boxes[:, -num_samples:]
    times = past_times_s[:, -num_samples:]
    centered_t = times - times.mean(dim=1, keepdim=True)
    centered_boxes = boxes - boxes.mean(dim=1, keepdim=True)
    denom = centered_t.square().sum(dim=1, keepdim=True).clamp(min=1.0e-12)
    return (centered_boxes * centered_t.unsqueeze(-1)).sum(dim=1) / denom


def box_sequence_to_state(past_boxes: torch.Tensor, past_times_s: torch.Tensor) -> torch.Tensor:
    """Return [cx, cy, w, h, vx, vy, vw, vh] using a last-four linear fit."""
    last = past_boxes[:, -1]
    velocity = _fit_recent_velocity(past_boxes, past_times_s, num_samples=4)
    return torch.cat([last, velocity], dim=-1)


def last_two_constant_velocity_forecast(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
) -> torch.Tensor:
    last = past_boxes[:, -1]
    if past_boxes.shape[1] < 2:
        velocity = torch.zeros_like(last)
    else:
        prev = past_boxes[:, -2]
        dt = (past_times_s[:, -1] - past_times_s[:, -2]).clamp(min=1.0e-6).unsqueeze(-1)
        velocity = (last - prev) / dt
    dt = future_times_s - past_times_s[:, -1:].expand_as(future_times_s)
    return (last.unsqueeze(1) + velocity.unsqueeze(1) * dt.unsqueeze(-1)).clamp(0.0, 1.0)


def last_four_constant_velocity_forecast(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
) -> torch.Tensor:
    state = box_sequence_to_state(past_boxes, past_times_s)
    pos = state[:, :4]
    vel = state[:, 4:]
    dt = future_times_s - past_times_s[:, -1:].expand_as(future_times_s)
    return (pos.unsqueeze(1) + vel.unsqueeze(1) * dt.unsqueeze(-1)).clamp(0.0, 1.0)


def constant_velocity_forecast(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible alias for the last-four linear extrapolation baseline."""
    return last_four_constant_velocity_forecast(past_boxes, past_times_s, future_times_s)


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


def _filter_state_indices(mode: str) -> list[int]:
    if mode == "full":
        return list(range(8))
    if mode == "center_position":
        return [0, 1]
    if mode == "center_velocity":
        return [4, 5]
    if mode == "velocities":
        return [4, 5, 6, 7]
    raise ValueError(
        "filter_state_feature_mode must be one of: full, center_position, center_velocity, velocities."
    )


def _make_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_layers: int,
    final_activation: bool = False,
) -> nn.Sequential:
    hidden_layers = int(hidden_layers)
    if hidden_layers < 0:
        raise ValueError(f"hidden_layers must be >= 0, got {hidden_layers}.")
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        layers.append(nn.ReLU(inplace=True))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, int(output_dim)))
    if final_activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


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
        fusion_layers: int = 1,
        history_feature_mode: str = "raw",
        state_hidden_dim: int = 128,
        state_layers: int = 2,
        residual_hidden_dim: int = 256,
        residual_layers: int = 2,
        residual_scale: float = 1.0,
        predict_size_residuals: bool = True,
        use_filter_state_features: bool = False,
        filter_state_feature_mode: str = "full",
        filter_covariance_features: str = "none",
        initial_state_source: str = "last_four",
        kalman_params: Dict | None = None,
        cell_local_first_conv: bool = False,
        cell_local_first_conv_representations: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.representations = list(representations)
        self.image_sizes = {
            str(rep): (int(size[0]), int(size[1])) for rep, size in dict(image_sizes).items()
        }
        self.history_steps = int(history_steps)
        self.history_feature_mode = str(history_feature_mode).lower()
        self.residual_scale = float(residual_scale)
        self.predict_size_residuals = bool(predict_size_residuals)
        self.use_filter_state_features = bool(use_filter_state_features)
        self.filter_state_feature_mode = str(filter_state_feature_mode).lower()
        self.filter_state_feature_indices = _filter_state_indices(self.filter_state_feature_mode)
        self.filter_covariance_features = str(filter_covariance_features).lower()
        self.initial_state_source = str(initial_state_source).lower()
        self.kalman_params = dict(kalman_params or {})
        if self.filter_covariance_features not in {"none", "diag", "full"}:
            raise ValueError(
                "filter_covariance_features must be one of: none, diag, full."
            )
        if self.initial_state_source not in {"last_four", "kalman_filter"}:
            raise ValueError("initial_state_source must be one of: last_four, kalman_filter.")
        if int(fusion_layers) < 1:
            raise ValueError("fusion_layers must be >= 1.")
        if int(state_layers) < 1:
            raise ValueError("state_layers must be >= 1.")
        if int(residual_layers) < 0:
            raise ValueError("residual_layers must be >= 0.")
        if self.history_feature_mode not in {"raw", "relative", "none"}:
            raise ValueError("history_feature_mode must be one of: raw, relative, none.")
        has_fusion_inputs = bool(
            self.representations
            or self.use_filter_state_features
            or self.filter_covariance_features != "none"
        )
        if (
            not has_fusion_inputs
            and self.history_feature_mode == "none"
        ):
            raise ValueError(
                "At least one learned feature source is required: representation, "
                "filter state/covariance features, or history_feature_mode raw/relative."
            )
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
        filter_state_dim = len(self.filter_state_feature_indices)
        if self.use_filter_state_features:
            fused_dim += filter_state_dim
        if self.filter_covariance_features == "diag":
            fused_dim += filter_state_dim
        elif self.filter_covariance_features == "full":
            fused_dim += filter_state_dim * filter_state_dim
        self.image_fusion = None
        fusion_dim = 0
        if fused_dim > 0:
            self.image_fusion = _make_mlp(
                input_dim=fused_dim,
                hidden_dim=fusion_hidden_dim,
                output_dim=fusion_hidden_dim,
                hidden_layers=max(0, int(fusion_layers) - 1),
                final_activation=True,
            )
            fusion_dim = int(fusion_hidden_dim)
        self.history_encoder = None
        history_dim = 0
        if self.history_feature_mode != "none":
            self.history_encoder = _make_mlp(
                input_dim=self.history_steps * 5,
                hidden_dim=state_hidden_dim,
                output_dim=state_hidden_dim,
                hidden_layers=max(0, int(state_layers) - 1),
                final_activation=True,
            )
            history_dim = int(state_hidden_dim)
        residual_out_dim = 4 if self.predict_size_residuals else 2
        self.residual_head = _make_mlp(
            input_dim=fusion_dim + history_dim + 9,
            hidden_dim=residual_hidden_dim,
            output_dim=residual_out_dim,
            hidden_layers=int(residual_layers),
            final_activation=False,
        )

    def _history_features(self, past_boxes: torch.Tensor, past_times_s: torch.Tensor) -> torch.Tensor:
        if self.history_encoder is None:
            raise RuntimeError("_history_features called with history_feature_mode='none'.")
        if past_boxes.shape[1] < self.history_steps:
            pad_count = self.history_steps - past_boxes.shape[1]
            box_pad = past_boxes[:, :1].expand(-1, pad_count, -1)
            time_pad = past_times_s[:, :1].expand(-1, pad_count)
            boxes = torch.cat([box_pad, past_boxes], dim=1)
            times = torch.cat([time_pad, past_times_s], dim=1)
        else:
            boxes = past_boxes[:, -self.history_steps :]
            times = past_times_s[:, -self.history_steps :]
        if self.history_feature_mode == "relative":
            relative_positions = boxes[..., :2] - boxes[:, -1:, :2].expand_as(boxes[..., :2])
            boxes = torch.cat([relative_positions, boxes[..., 2:]], dim=-1)
        rel_times = times - times[:, -1:].expand_as(times)
        return self.history_encoder(torch.cat([boxes, rel_times.unsqueeze(-1)], dim=-1).flatten(1))

    def _image_features(
        self,
        inputs: Dict[str, torch.Tensor],
        filter_state: torch.Tensor | None = None,
        filter_cov: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        enc = [self.encoders[rep](inputs[rep]).pooled for rep in self.representations]
        if self.use_filter_state_features:
            if filter_state is None:
                raise ValueError("filter_state is required when use_filter_state_features=True.")
            enc.append(filter_state[:, self.filter_state_feature_indices])
        if self.filter_covariance_features != "none":
            if filter_cov is None:
                raise ValueError("filter_cov is required when filter_covariance_features is enabled.")
            idx = torch.tensor(
                self.filter_state_feature_indices,
                device=filter_cov.device,
                dtype=torch.long,
            )
            selected_cov = filter_cov.index_select(1, idx).index_select(2, idx)
            if self.filter_covariance_features == "diag":
                enc.append(torch.diagonal(selected_cov, dim1=-2, dim2=-1))
            else:
                enc.append(selected_cov.flatten(1))
        if not enc:
            return None
        if self.image_fusion is None:
            raise RuntimeError("image_fusion is missing even though image/filter features were provided.")
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
        filter_state = None
        filter_cov = None
        needs_filter = (
            self.use_filter_state_features
            or self.filter_covariance_features != "none"
            or self.initial_state_source == "kalman_filter"
        )
        if needs_filter:
            filter_state, filter_cov = kalman_filter_history(past_boxes, past_times_s, self.kalman_params)
        image_feat = self._image_features(inputs, filter_state, filter_cov)
        history_feat = (
            self._history_features(past_boxes, past_times_s)
            if self.history_feature_mode != "none"
            else None
        )
        if self.initial_state_source == "kalman_filter":
            if filter_state is None:
                raise RuntimeError("filter_state was not computed for kalman_filter initial_state_source.")
            state = filter_state
        else:
            state = box_sequence_to_state(past_boxes, past_times_s)
        current_time = past_times_s[:, -1]
        preds: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for step in range(future_times_s.shape[1]):
            next_time = future_times_s[:, step]
            dt = (next_time - current_time).clamp(min=1.0e-6)
            step_parts = [state, dt.unsqueeze(-1)]
            if image_feat is not None:
                step_parts.insert(0, image_feat)
            if history_feat is not None:
                insert_idx = 1 if image_feat is not None else 0
                step_parts.insert(insert_idx, history_feat)
            step_feat = torch.cat(step_parts, dim=-1)
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
        debug = {
            "boxes": pred,
            "residual_accel": torch.stack(residuals, dim=1),
            "last4_boxes": last_four_constant_velocity_forecast(past_boxes, past_times_s, future_times_s),
            "last2_boxes": last_two_constant_velocity_forecast(past_boxes, past_times_s, future_times_s),
            "cv_boxes": last_four_constant_velocity_forecast(past_boxes, past_times_s, future_times_s),
        }
        if filter_state is not None:
            debug["filter_state"] = filter_state
        if filter_cov is not None:
            debug["filter_cov"] = filter_cov
        return debug
