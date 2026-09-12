from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


DEFAULT_KALMAN_CONFIG: dict[str, Any] = {
    "enabled": True,
    "motion_model": "constant_velocity",
    "initial_pos_std": 0.05,
    "initial_size_std": 0.05,
    "initial_vel_std": 1.0,
    "initial_accel_std": 1.0,
    "initial_size_accel_std": 1.0,
    "process_pos_std": 0.001,
    "process_size_std": 0.001,
    "process_vel_std": 0.1,
    "process_size_vel_std": 0.1,
    "process_accel_std": 0.1,
    "process_size_accel_std": 0.1,
    "measurement_pos_std": 0.01,
    "measurement_size_std": 0.01,
    "dynamics_generator": None,
}


def _validate_dynamics_generator(value: Any) -> list[list[float]]:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("kalman.dynamics_generator must be an 8x8 numeric matrix.") from exc
    if tensor.shape != (8, 8):
        raise ValueError(
            f"kalman.dynamics_generator must have shape [8, 8], got {list(tensor.shape)}."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("kalman.dynamics_generator must contain only finite values.")
    return tensor.tolist()


def constant_velocity_generator(*, device=None, dtype=None) -> torch.Tensor:
    """Continuous-time CV generator for [cx, cy, w, h, vx, vy, vw, vh]."""
    generator = torch.zeros((8, 8), device=device, dtype=dtype)
    generator[:4, 4:] = torch.eye(4, device=device, dtype=dtype)
    return generator


def kalman_config_from_dict(cfg: Dict[str, Any] | None) -> dict[str, Any]:
    out = dict(DEFAULT_KALMAN_CONFIG)
    if cfg:
        for key, value in cfg.items():
            if key not in out:
                continue
            if key == "motion_model":
                value = str(value).lower()
                if value not in {"constant_velocity", "constant_acceleration", "coupled"}:
                    raise ValueError(
                        "kalman.motion_model must be 'constant_velocity', "
                        "'constant_acceleration', or 'coupled'."
                    )
                out[key] = value
            elif key == "dynamics_generator":
                out[key] = None if value is None else _validate_dynamics_generator(value)
            elif isinstance(out[key], bool):
                out[key] = bool(value)
            else:
                out[key] = float(value)
    if out["motion_model"] == "coupled":
        if out["dynamics_generator"] is None:
            raise ValueError("kalman.motion_model 'coupled' requires kalman.dynamics_generator.")
    elif out["dynamics_generator"] is not None:
        raise ValueError("kalman.dynamics_generator is only valid when kalman.motion_model is 'coupled'.")
    return out


def _std_vector(params: Dict[str, Any], *, prefix: str, device, dtype) -> torch.Tensor:
    if prefix == "initial":
        values = [
                float(params["initial_pos_std"]),
                float(params["initial_pos_std"]),
                float(params["initial_size_std"]),
                float(params["initial_size_std"]),
                float(params["initial_vel_std"]),
                float(params["initial_vel_std"]),
                float(params["initial_vel_std"]),
                float(params["initial_vel_std"]),
            ]
        if params["motion_model"] == "constant_acceleration":
            values += [float(params["initial_accel_std"])] * 2 + [float(params["initial_size_accel_std"])] * 2
        return torch.tensor(values, device=device, dtype=dtype)
    if prefix == "process":
        values = [
                float(params["process_pos_std"]),
                float(params["process_pos_std"]),
                float(params["process_size_std"]),
                float(params["process_size_std"]),
                float(params["process_vel_std"]),
                float(params["process_vel_std"]),
                float(params["process_size_vel_std"]),
                float(params["process_size_vel_std"]),
            ]
        if params["motion_model"] == "constant_acceleration":
            values += [float(params["process_accel_std"])] * 2 + [float(params["process_size_accel_std"])] * 2
        return torch.tensor(values, device=device, dtype=dtype)
    if prefix == "measurement":
        return torch.tensor(
            [
                float(params["measurement_pos_std"]),
                float(params["measurement_pos_std"]),
                float(params["measurement_size_std"]),
                float(params["measurement_size_std"]),
            ],
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unknown std prefix: {prefix}")


def kalman_std_tensors_from_config(
    cfg: Dict[str, Any] | None,
    *,
    device,
    dtype,
) -> dict[str, torch.Tensor]:
    params = kalman_config_from_dict(cfg)
    return {
        key: torch.tensor(float(value), device=device, dtype=dtype)
        for key, value in params.items()
        if key not in {"enabled", "motion_model", "dynamics_generator"}
    }


def _std_vector_from_tensors(params: Dict[str, torch.Tensor], *, prefix: str, motion_model: str) -> torch.Tensor:
    if prefix == "initial":
        values = [
                params["initial_pos_std"],
                params["initial_pos_std"],
                params["initial_size_std"],
                params["initial_size_std"],
                params["initial_vel_std"],
                params["initial_vel_std"],
                params["initial_vel_std"],
                params["initial_vel_std"],
            ]
        if motion_model == "constant_acceleration":
            values += [params["initial_accel_std"]] * 2 + [params["initial_size_accel_std"]] * 2
        return torch.stack(values)
    if prefix == "process":
        values = [
                params["process_pos_std"],
                params["process_pos_std"],
                params["process_size_std"],
                params["process_size_std"],
                params["process_vel_std"],
                params["process_vel_std"],
                params["process_size_vel_std"],
                params["process_size_vel_std"],
            ]
        if motion_model == "constant_acceleration":
            values += [params["process_accel_std"]] * 2 + [params["process_size_accel_std"]] * 2
        return torch.stack(values)
    if prefix == "measurement":
        return torch.stack(
            [
                params["measurement_pos_std"],
                params["measurement_pos_std"],
                params["measurement_size_std"],
                params["measurement_size_std"],
            ]
        )
    raise ValueError(f"Unknown std prefix: {prefix}")


def _transition(
    dt: torch.Tensor,
    motion_model: str,
    dynamics_generator: Any = None,
) -> torch.Tensor:
    if motion_model == "coupled":
        if dynamics_generator is None:
            raise ValueError("Coupled Kalman transition requires a dynamics generator.")
        generator = torch.as_tensor(dynamics_generator, device=dt.device, dtype=dt.dtype)
        if generator.shape != (8, 8):
            raise ValueError(f"Expected an 8x8 dynamics generator, got {tuple(generator.shape)}.")
        return torch.matrix_exp(dt[..., None, None] * generator)
    batch = int(dt.shape[0])
    state_dim = 12 if motion_model == "constant_acceleration" else 8
    f = torch.eye(state_dim, device=dt.device, dtype=dt.dtype).unsqueeze(0).repeat(batch, 1, 1)
    f[:, 0, 4] = dt
    f[:, 1, 5] = dt
    f[:, 2, 6] = dt
    f[:, 3, 7] = dt
    if motion_model == "constant_acceleration":
        for channel in range(4):
            f[:, channel, 8 + channel] = 0.5 * dt.square()
            f[:, 4 + channel, 8 + channel] = dt
    return f


def _predict(
    state: torch.Tensor,
    cov: torch.Tensor,
    dt: torch.Tensor,
    q_base: torch.Tensor,
    motion_model: str,
    dynamics_generator: Any = None,
):
    f = _transition(dt, motion_model, dynamics_generator)
    state = torch.bmm(f, state.unsqueeze(-1)).squeeze(-1)
    scales = [dt] * 4 + [torch.ones_like(dt)] * 4
    if motion_model == "constant_acceleration":
        scales += [torch.ones_like(dt)] * 4
    q_scale = torch.stack(scales, dim=1)
    q_diag = (q_base.unsqueeze(0) * q_scale.clamp(min=1.0e-6)).square()
    q = torch.diag_embed(q_diag)
    cov = torch.bmm(torch.bmm(f, cov), f.transpose(1, 2)) + q
    return state, cov


def _update(state: torch.Tensor, cov: torch.Tensor, measurement: torch.Tensor, r_diag: torch.Tensor):
    batch = int(state.shape[0])
    state_dim = int(state.shape[1])
    h = torch.zeros((batch, 4, state_dim), device=state.device, dtype=state.dtype)
    h[:, 0, 0] = 1.0
    h[:, 1, 1] = 1.0
    h[:, 2, 2] = 1.0
    h[:, 3, 3] = 1.0
    r = torch.diag_embed(r_diag.unsqueeze(0).expand(batch, -1).square())
    residual = measurement - torch.bmm(h, state.unsqueeze(-1)).squeeze(-1)
    s = torch.bmm(torch.bmm(h, cov), h.transpose(1, 2)) + r
    k = torch.linalg.solve(s, torch.bmm(h, cov)).transpose(1, 2)
    state = state + torch.bmm(k, residual.unsqueeze(-1)).squeeze(-1)
    eye = torch.eye(state_dim, device=state.device, dtype=state.dtype).unsqueeze(0).expand(batch, -1, -1)
    kh = torch.bmm(k, h)
    # Joseph form is a little more expensive, but keeps P symmetric/positive for tuned extremes.
    cov = torch.bmm(torch.bmm(eye - kh, cov), (eye - kh).transpose(1, 2)) + torch.bmm(torch.bmm(k, r), k.transpose(1, 2))
    state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
    return state, cov


# Public shared primitives used by learnable and fixed Kalman runtimes.
predict_kalman_state = _predict
update_kalman_state = _update


def kalman_filter_history(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    params: Dict[str, Any] | None = None,
    *,
    _validated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = dict(params or {}) if _validated else kalman_config_from_dict(params)
    batch = int(past_boxes.shape[0])
    device = past_boxes.device
    dtype = past_boxes.dtype
    motion_model = str(cfg["motion_model"])
    state_dim = 12 if motion_model == "constant_acceleration" else 8
    state = torch.zeros((batch, state_dim), device=device, dtype=dtype)
    state[:, :4] = past_boxes[:, 0]
    init_std = _std_vector(cfg, prefix="initial", device=device, dtype=dtype)
    process_std = _std_vector(cfg, prefix="process", device=device, dtype=dtype)
    meas_std = _std_vector(cfg, prefix="measurement", device=device, dtype=dtype)
    cov = torch.diag_embed(init_std.unsqueeze(0).expand(batch, -1).square())
    state, cov = _update(state, cov, past_boxes[:, 0], meas_std)
    for idx in range(1, past_boxes.shape[1]):
        dt = (past_times_s[:, idx] - past_times_s[:, idx - 1]).clamp(min=1.0e-6)
        state, cov = _predict(
            state, cov, dt, process_std, motion_model, cfg.get("dynamics_generator")
        )
        state, cov = _update(state, cov, past_boxes[:, idx], meas_std)
    return state, cov


def kalman_forecast(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
    params: Dict[str, Any] | None = None,
) -> torch.Tensor:
    cfg = kalman_config_from_dict(params)
    state, cov = kalman_filter_history(
        past_boxes, past_times_s, cfg, _validated=True
    )
    process_std = _std_vector(cfg, prefix="process", device=past_boxes.device, dtype=past_boxes.dtype)
    current_time = past_times_s[:, -1]
    preds: list[torch.Tensor] = []
    for idx in range(future_times_s.shape[1]):
        next_time = future_times_s[:, idx]
        dt = (next_time - current_time).clamp(min=1.0e-6)
        state, cov = _predict(
            state,
            cov,
            dt,
            process_std,
            str(cfg["motion_model"]),
            cfg.get("dynamics_generator"),
        )
        state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
        preds.append(state[:, :4])
        current_time = next_time
    return torch.stack(preds, dim=1)


def kalman_cv_forecast(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
    params: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for the configured Kalman forecast."""
    return kalman_forecast(past_boxes, past_times_s, future_times_s, params)


def kalman_filter_history_tensor_params(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    params: Dict[str, torch.Tensor],
    *,
    motion_model: str = "constant_velocity",
    dynamics_generator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(past_boxes.shape[0])
    device = past_boxes.device
    dtype = past_boxes.dtype
    state = torch.cat(
        [
            past_boxes[:, 0],
            torch.zeros((batch, 8 if motion_model == "constant_acceleration" else 4), device=device, dtype=dtype),
        ],
        dim=-1,
    )
    init_std = _std_vector_from_tensors(params, prefix="initial", motion_model=motion_model).to(device=device, dtype=dtype)
    process_std = _std_vector_from_tensors(params, prefix="process", motion_model=motion_model).to(device=device, dtype=dtype)
    meas_std = _std_vector_from_tensors(params, prefix="measurement", motion_model=motion_model).to(device=device, dtype=dtype)
    cov = torch.diag_embed(init_std.unsqueeze(0).expand(batch, -1).square())
    state, cov = _update(state, cov, past_boxes[:, 0], meas_std)
    for idx in range(1, past_boxes.shape[1]):
        dt = (past_times_s[:, idx] - past_times_s[:, idx - 1]).clamp(min=1.0e-6)
        state, cov = _predict(state, cov, dt, process_std, motion_model, dynamics_generator)
        state, cov = _update(state, cov, past_boxes[:, idx], meas_std)
    return state, cov


def kalman_cv_forecast_tensor_params(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
    params: Dict[str, torch.Tensor],
    *,
    motion_model: str = "constant_velocity",
    dynamics_generator: torch.Tensor | None = None,
) -> torch.Tensor:
    state, cov = kalman_filter_history_tensor_params(
        past_boxes,
        past_times_s,
        params,
        motion_model=motion_model,
        dynamics_generator=dynamics_generator,
    )
    process_std = _std_vector_from_tensors(params, prefix="process", motion_model=motion_model).to(
        device=past_boxes.device,
        dtype=past_boxes.dtype,
    )
    current_time = past_times_s[:, -1]
    preds: list[torch.Tensor] = []
    for idx in range(future_times_s.shape[1]):
        next_time = future_times_s[:, idx]
        dt = (next_time - current_time).clamp(min=1.0e-6)
        state, cov = _predict(state, cov, dt, process_std, motion_model, dynamics_generator)
        state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
        preds.append(state[:, :4])
        current_time = next_time
    return torch.stack(preds, dim=1)


class ConfiguredBoxKalmanFilter(nn.Module):
    """Fixed, device-aware runtime for a configured box Kalman filter."""

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = kalman_config_from_dict(cfg)
        generator = self.config.get("dynamics_generator")
        self.register_buffer(
            "dynamics_generator",
            None if generator is None else torch.tensor(generator, dtype=torch.float32),
        )
        for key, value in self.config.items():
            if key in {"enabled", "motion_model", "dynamics_generator"}:
                continue
            self.register_buffer(
                f"_noise_{key}", torch.tensor(float(value), dtype=torch.float32), persistent=False
            )

    @property
    def motion_model(self) -> str:
        return str(self.config["motion_model"])

    @property
    def state_dim(self) -> int:
        return 12 if self.motion_model == "constant_acceleration" else 8

    def _noise_tensors(self) -> dict[str, torch.Tensor]:
        return {
            key: getattr(self, f"_noise_{key}")
            for key in self.config
            if key not in {"enabled", "motion_model", "dynamics_generator"}
        }

    def transition(self, dt: torch.Tensor) -> torch.Tensor:
        return _transition(dt, self.motion_model, self.dynamics_generator)

    def transition_state(self, state: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        transition = self.transition(dt).to(device=state.device, dtype=state.dtype)
        return torch.bmm(transition, state.unsqueeze(-1)).squeeze(-1)

    def filter_history(
        self, past_boxes: torch.Tensor, past_times_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kalman_filter_history_tensor_params(
            past_boxes,
            past_times_s,
            self._noise_tensors(),
            motion_model=self.motion_model,
            dynamics_generator=self.dynamics_generator,
        )

    def forward(
        self,
        past_boxes: torch.Tensor,
        past_times_s: torch.Tensor,
        future_times_s: torch.Tensor,
    ) -> torch.Tensor:
        params = self._noise_tensors()
        state, covariance = kalman_filter_history_tensor_params(
            past_boxes,
            past_times_s,
            params,
            motion_model=self.motion_model,
            dynamics_generator=self.dynamics_generator,
        )
        process_std = _std_vector_from_tensors(
            params, prefix="process", motion_model=self.motion_model
        ).to(device=past_boxes.device, dtype=past_boxes.dtype)
        current_time = past_times_s[:, -1]
        predictions: list[torch.Tensor] = []
        for index in range(future_times_s.shape[1]):
            next_time = future_times_s[:, index]
            dt = (next_time - current_time).clamp(min=1.0e-6)
            state, covariance = _predict(
                state,
                covariance,
                dt,
                process_std,
                self.motion_model,
                self.dynamics_generator,
            )
            state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
            predictions.append(state[:, :4])
            current_time = next_time
        return torch.stack(predictions, dim=1)
