from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn

from .kalman_filter import DEFAULT_KALMAN_CONFIG


NOISE_PARAM_KEYS = [
    "initial_pos_std",
    "initial_size_std",
    "initial_vel_std",
    "process_pos_std",
    "process_size_std",
    "process_vel_std",
    "process_size_vel_std",
    "measurement_pos_std",
    "measurement_size_std",
]


def constant_velocity_generator(*, device=None, dtype=None) -> torch.Tensor:
    """Continuous-time CV generator for [cx, cy, w, h, vx, vy, vw, vh]."""
    generator = torch.zeros((8, 8), device=device, dtype=dtype)
    generator[:4, 4:] = torch.eye(4, device=device, dtype=dtype)
    return generator


class CoupledBoxKalmanFilter(nn.Module):
    """Differentiable box Kalman filter with a fully coupled transition.

    The learned matrix is a continuous-time generator A.  A transition over an
    arbitrary timestamp interval is F(dt) = exp(A * dt), avoiding the need to
    assume that every sequence has exactly the same sampling period.
    """

    def __init__(
        self,
        kalman_cfg: Dict[str, Any] | None = None,
        *,
        optimize_noise: bool = False,
        min_std: float = 1.0e-6,
        max_std: float = 10.0,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        cfg = dict(DEFAULT_KALMAN_CONFIG)
        if kalman_cfg:
            cfg.update({key: value for key, value in kalman_cfg.items() if key in cfg})
        self.optimize_noise = bool(optimize_noise)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        if self.min_std <= 0 or self.max_std < self.min_std:
            raise ValueError("Expected 0 < min_std <= max_std.")

        self.dynamics_generator = nn.Parameter(
            constant_velocity_generator(device=device, dtype=dtype)
        )
        self.log_std = nn.ParameterDict()
        for key in NOISE_PARAM_KEYS:
            value = min(max(float(cfg[key]), self.min_std), self.max_std)
            parameter = nn.Parameter(
                torch.tensor(math.log(value), device=device, dtype=dtype),
                requires_grad=self.optimize_noise,
            )
            self.log_std[key] = parameter

    def noise_tensors(self) -> dict[str, torch.Tensor]:
        return {key: value.exp() for key, value in self.log_std.items()}

    def transition(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.to(device=self.dynamics_generator.device, dtype=self.dynamics_generator.dtype)
        return torch.matrix_exp(dt[..., None, None] * self.dynamics_generator)

    def _std_vectors(self, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = {key: value.to(device=device, dtype=dtype) for key, value in self.noise_tensors().items()}
        initial = torch.stack(
            [params["initial_pos_std"]] * 2
            + [params["initial_size_std"]] * 2
            + [params["initial_vel_std"]] * 4
        )
        process = torch.stack(
            [params["process_pos_std"]] * 2
            + [params["process_size_std"]] * 2
            + [params["process_vel_std"]] * 2
            + [params["process_size_vel_std"]] * 2
        )
        measurement = torch.stack(
            [params["measurement_pos_std"]] * 2 + [params["measurement_size_std"]] * 2
        )
        return initial, process, measurement

    def _predict(
        self,
        state: torch.Tensor,
        covariance: torch.Tensor,
        dt: torch.Tensor,
        process_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.transition(dt).to(device=state.device, dtype=state.dtype)
        state = torch.bmm(transition, state.unsqueeze(-1)).squeeze(-1)
        q_scale = torch.stack([dt] * 4 + [torch.ones_like(dt)] * 4, dim=1)
        q_diag = (process_std.unsqueeze(0) * q_scale.clamp(min=1.0e-6)).square()
        covariance = (
            torch.bmm(torch.bmm(transition, covariance), transition.transpose(1, 2))
            + torch.diag_embed(q_diag)
        )
        return state, covariance

    @staticmethod
    def _update(
        state: torch.Tensor,
        covariance: torch.Tensor,
        measurement: torch.Tensor,
        measurement_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(state.shape[0])
        h = torch.zeros((batch, 4, 8), device=state.device, dtype=state.dtype)
        h[:, :, :4] = torch.eye(4, device=state.device, dtype=state.dtype)
        r = torch.diag_embed(measurement_std.unsqueeze(0).expand(batch, -1).square())
        residual = measurement - torch.bmm(h, state.unsqueeze(-1)).squeeze(-1)
        innovation_cov = torch.bmm(torch.bmm(h, covariance), h.transpose(1, 2)) + r
        gain = torch.linalg.solve(innovation_cov, torch.bmm(h, covariance)).transpose(1, 2)
        state = state + torch.bmm(gain, residual.unsqueeze(-1)).squeeze(-1)
        eye = torch.eye(8, device=state.device, dtype=state.dtype).unsqueeze(0).expand(batch, -1, -1)
        correction = eye - torch.bmm(gain, h)
        covariance = (
            torch.bmm(torch.bmm(correction, covariance), correction.transpose(1, 2))
            + torch.bmm(torch.bmm(gain, r), gain.transpose(1, 2))
        )
        state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
        return state, covariance

    def filter_history(
        self, past_boxes: torch.Tensor, past_times_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(past_boxes.shape[0])
        initial_std, process_std, measurement_std = self._std_vectors(
            device=past_boxes.device, dtype=past_boxes.dtype
        )
        state = torch.cat(
            [past_boxes[:, 0], torch.zeros((batch, 4), device=past_boxes.device, dtype=past_boxes.dtype)],
            dim=-1,
        )
        covariance = torch.diag_embed(initial_std.unsqueeze(0).expand(batch, -1).square())
        state, covariance = self._update(state, covariance, past_boxes[:, 0], measurement_std)
        for index in range(1, past_boxes.shape[1]):
            dt = (past_times_s[:, index] - past_times_s[:, index - 1]).clamp(min=1.0e-6)
            state, covariance = self._predict(state, covariance, dt, process_std)
            state, covariance = self._update(state, covariance, past_boxes[:, index], measurement_std)
        return state, covariance

    def forward(
        self,
        past_boxes: torch.Tensor,
        past_times_s: torch.Tensor,
        future_times_s: torch.Tensor,
    ) -> torch.Tensor:
        state, covariance = self.filter_history(past_boxes, past_times_s)
        _, process_std, _ = self._std_vectors(device=past_boxes.device, dtype=past_boxes.dtype)
        current_time = past_times_s[:, -1]
        predictions: list[torch.Tensor] = []
        for index in range(future_times_s.shape[1]):
            next_time = future_times_s[:, index]
            dt = (next_time - current_time).clamp(min=1.0e-6)
            state, covariance = self._predict(state, covariance, dt, process_std)
            state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
            predictions.append(state[:, :4])
            current_time = next_time
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def clamp_(self, *, max_dynamics_abs: float) -> None:
        self.dynamics_generator.clamp_(-float(max_dynamics_abs), float(max_dynamics_abs))
        lo, hi = math.log(self.min_std), math.log(self.max_std)
        for value in self.log_std.values():
            value.clamp_(lo, hi)

    def snapshot(self, *, reference_dt: float) -> dict[str, Any]:
        with torch.no_grad():
            generator = self.dynamics_generator.detach().cpu()
            transition = torch.matrix_exp(generator * float(reference_dt))
            noise = {key: float(value.exp().detach().cpu()) for key, value in self.log_std.items()}
        return {
            "state_layout": ["cx", "cy", "w", "h", "vx", "vy", "vw", "vh"],
            "parameterization": "F(dt) = matrix_exp(dynamics_generator * dt)",
            "reference_dt_s": float(reference_dt),
            "dynamics_generator": generator.tolist(),
            "reference_transition": transition.tolist(),
            "noise": noise,
        }
