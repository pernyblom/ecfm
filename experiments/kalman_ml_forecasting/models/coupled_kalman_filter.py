from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn

from .kalman_covariance import CovarianceParameters
from .kalman_filter import (
    channel_mask, constant_velocity_generator, kalman_config_from_dict,
    kalman_cv_forecast_tensor_params, kalman_filter_history_tensor_params,
    _std_vector_from_tensors,
)


NOISE_PARAM_KEYS = [
    "initial_pos_std", "initial_size_std", "initial_vel_std",
    "process_pos_std", "process_size_std", "process_vel_std", "process_size_vel_std",
    "measurement_pos_std", "measurement_size_std",
]


class CoupledBoxKalmanFilter(nn.Module):
    """8-D fixed CV, learned separate, or learned full dynamics and covariance.

    Custom dynamics use F(dt) = exp(A * dt). Covariances are learned independently
    of dynamics; the legacy constructor still learns full A with fixed scalar noise.
    """

    def __init__(
        self, kalman_cfg: Dict[str, Any] | None = None, *,
        optimize_dynamics: bool = True, optimize_noise: bool = False,
        dynamics: str | None = None,
        optimize_initial: bool | None = None,
        optimize_process: bool | None = None,
        optimize_measurement: bool | None = None,
        min_std: float = 1.0e-6, max_std: float = 10.0,
        device=None, dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        source = dict(kalman_cfg or {})
        if source.get("motion_model") == "constant_acceleration":
            raise ValueError("This optimizer is 8-D only; use optimize_kalman_backprop.py for 12-D constant_acceleration.")
        self.dynamics = dynamics or source.get("dynamics_structure", "full")
        if self.dynamics not in {"constant_velocity", "per_channel", "full"}:
            raise ValueError("dynamics must be constant_velocity, per_channel, or full.")
        self.optimize_dynamics = bool(optimize_dynamics and self.dynamics != "constant_velocity")
        self.optimize_noise = bool(optimize_noise)
        self.min_std, self.max_std = float(min_std), float(max_std)
        if not (math.isfinite(min_std) and math.isfinite(max_std) and 0 < min_std <= max_std):
            raise ValueError("Expected finite 0 < min_std <= max_std.")
        configured_generator = source.get("dynamics_generator")
        generator = (constant_velocity_generator(device=device, dtype=dtype)
                     if configured_generator is None or self.dynamics == "constant_velocity"
                     else torch.as_tensor(configured_generator, device=device, dtype=dtype))
        source.update(motion_model="coupled", dynamics_generator=generator.tolist(),
                      dynamics_structure="per_channel" if self.dynamics == "per_channel" else "full")
        cfg = kalman_config_from_dict(source)
        self.dynamics_generator = nn.Parameter(generator.clone(), requires_grad=self.optimize_dynamics)
        self.register_buffer("dynamics_mask", channel_mask(8, device=device) if self.dynamics == "per_channel"
                             else torch.ones((8, 8), device=device, dtype=torch.bool), persistent=False)
        learning = {"initial": optimize_initial, "process": optimize_process, "measurement": optimize_measurement}
        learning = {key: self.optimize_noise if value is None else bool(value) for key, value in learning.items()}
        self.structures = {prefix: cfg[f"{prefix}_covariance_structure"] for prefix in learning}
        for prefix in learning:
            if cfg[f"{prefix}_covariance"] is not None and self.structures[prefix] == "legacy_diagonal":
                self.structures[prefix] = "diagonal"
        self.log_std = nn.ParameterDict()
        for key in NOISE_PARAM_KEYS:
            value = float(cfg[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"kalman.{key} must be finite and positive.")
            value = min(max(value, self.min_std), self.max_std)
            prefix = key.split("_")[0]
            self.log_std[key] = nn.Parameter(torch.tensor(math.log(value), device=device, dtype=dtype),
                                            requires_grad=learning[prefix] and self.structures[prefix] == "legacy_diagonal")
        self.covariances = nn.ModuleDict()
        for prefix, structure in self.structures.items():
            if structure == "legacy_diagonal":
                continue
            value = cfg[f"{prefix}_covariance"]
            covariance = (torch.diag(_std_vector_from_tensors(self.noise_tensors(), prefix=prefix,
                                                               motion_model="constant_velocity").square())
                          if value is None else torch.tensor(value, device=device, dtype=dtype))
            self.covariances[prefix] = CovarianceParameters(
                covariance, structure=structure, optimize=learning[prefix],
                min_std=self.min_std, max_std=self.max_std,
            )

    def effective_generator(self) -> torch.Tensor:
        return self.dynamics_generator * self.dynamics_mask

    def noise_tensors(self) -> dict[str, torch.Tensor]:
        return {key: value.exp() for key, value in self.log_std.items()}

    def tensors(self) -> dict[str, torch.Tensor]:
        return {**self.noise_tensors(), **{f"{key}_covariance": module() for key, module in self.covariances.items()}}

    def transition(self, dt: torch.Tensor) -> torch.Tensor:
        dt = dt.to(device=self.dynamics_generator.device, dtype=self.dynamics_generator.dtype)
        return torch.matrix_exp(dt[..., None, None] * self.effective_generator())

    def filter_history(self, past_boxes: torch.Tensor, past_times_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return kalman_filter_history_tensor_params(
            past_boxes, past_times_s, self.tensors(),
            motion_model="constant_velocity" if self.dynamics == "constant_velocity" else "coupled",
            dynamics_generator=self.effective_generator(),
        )

    def forward(self, past_boxes: torch.Tensor, past_times_s: torch.Tensor,
                future_times_s: torch.Tensor) -> torch.Tensor:
        return kalman_cv_forecast_tensor_params(
            past_boxes, past_times_s, future_times_s, self.tensors(),
            motion_model="constant_velocity" if self.dynamics == "constant_velocity" else "coupled",
            dynamics_generator=self.effective_generator(),
        )

    def covariance_penalty(self) -> torch.Tensor:
        return sum((module.correlation_penalty() for module in self.covariances.values()),
                   self.dynamics_generator.new_zeros(()))

    @torch.no_grad()
    def clamp_(self, *, max_dynamics_abs: float) -> None:
        if self.optimize_dynamics:
            self.dynamics_generator.clamp_(-float(max_dynamics_abs), float(max_dynamics_abs))
        self.dynamics_generator.mul_(self.dynamics_mask)
        for value in self.log_std.values():
            if value.requires_grad:
                value.clamp_(math.log(self.min_std), math.log(self.max_std))
        for module in self.covariances.values():
            module.clamp_()

    def snapshot(self, *, reference_dt: float) -> dict[str, Any]:
        with torch.no_grad():
            generator = self.effective_generator().detach().cpu()
            transition = torch.matrix_exp(generator * float(reference_dt))
            noise = {key: float(value.exp().detach().cpu()) for key, value in self.log_std.items()}
            matrices = {f"{key}_covariance": module().detach().cpu().tolist() for key, module in self.covariances.items()}
        runtime_config = {
            "enabled": True,
            "motion_model": "constant_velocity" if self.dynamics == "constant_velocity" else "coupled",
            "dynamics_structure": "per_channel" if self.dynamics == "per_channel" else "full",
            **{f"{key}_covariance_structure": value for key, value in self.structures.items()},
            **noise, **matrices,
        }
        if self.dynamics != "constant_velocity":
            runtime_config["dynamics_generator"] = generator.tolist()
        return {
            "schema_version": 2,
            "state_layout": ["cx", "cy", "w", "h", "vx", "vy", "vw", "vh"],
            "parameterization": "F(dt) = matrix_exp(dynamics_generator * dt)",
            "process_covariance_time_scaling": "Q(dt) = D Q_base D^T; D = diag(dt,dt,dt,dt,1,1,1,1)",
            "dynamics": self.dynamics,
            "reference_dt_s": float(reference_dt),
            "dynamics_generator": generator.tolist(),
            "reference_transition": transition.tolist(),
            "noise": noise,
            "kalman_config": runtime_config,
        }
