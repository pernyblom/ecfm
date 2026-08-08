from __future__ import annotations

from typing import Any, Dict

import torch


DEFAULT_KALMAN_CONFIG: dict[str, float | bool | str] = {
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
}


def kalman_config_from_dict(cfg: Dict[str, Any] | None) -> dict[str, float | bool | str]:
    out = dict(DEFAULT_KALMAN_CONFIG)
    if cfg:
        for key, value in cfg.items():
            if key not in out:
                continue
            if key == "motion_model":
                value = str(value).lower()
                if value not in {"constant_velocity", "constant_acceleration"}:
                    raise ValueError("kalman.motion_model must be 'constant_velocity' or 'constant_acceleration'.")
                out[key] = value
            elif isinstance(out[key], bool):
                out[key] = bool(value)
            else:
                out[key] = float(value)
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
        if key not in {"enabled", "motion_model"}
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


def _transition(dt: torch.Tensor, motion_model: str) -> torch.Tensor:
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


def _predict(state: torch.Tensor, cov: torch.Tensor, dt: torch.Tensor, q_base: torch.Tensor, motion_model: str):
    f = _transition(dt, motion_model)
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


def kalman_filter_history(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    params: Dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = kalman_config_from_dict(params)
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
        state, cov = _predict(state, cov, dt, process_std, motion_model)
        state, cov = _update(state, cov, past_boxes[:, idx], meas_std)
    return state, cov


def kalman_cv_forecast(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
    params: Dict[str, Any] | None = None,
) -> torch.Tensor:
    state, cov = kalman_filter_history(past_boxes, past_times_s, params)
    cfg = kalman_config_from_dict(params)
    process_std = _std_vector(cfg, prefix="process", device=past_boxes.device, dtype=past_boxes.dtype)
    current_time = past_times_s[:, -1]
    preds: list[torch.Tensor] = []
    for idx in range(future_times_s.shape[1]):
        next_time = future_times_s[:, idx]
        dt = (next_time - current_time).clamp(min=1.0e-6)
        state, cov = _predict(state, cov, dt, process_std, str(cfg["motion_model"]))
        state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
        preds.append(state[:, :4])
        current_time = next_time
    return torch.stack(preds, dim=1)


def kalman_filter_history_tensor_params(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    params: Dict[str, torch.Tensor],
    *,
    motion_model: str = "constant_velocity",
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
        state, cov = _predict(state, cov, dt, process_std, motion_model)
        state, cov = _update(state, cov, past_boxes[:, idx], meas_std)
    return state, cov


def kalman_cv_forecast_tensor_params(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
    params: Dict[str, torch.Tensor],
    *,
    motion_model: str = "constant_velocity",
) -> torch.Tensor:
    state, cov = kalman_filter_history_tensor_params(past_boxes, past_times_s, params, motion_model=motion_model)
    process_std = _std_vector_from_tensors(params, prefix="process", motion_model=motion_model).to(
        device=past_boxes.device,
        dtype=past_boxes.dtype,
    )
    current_time = past_times_s[:, -1]
    preds: list[torch.Tensor] = []
    for idx in range(future_times_s.shape[1]):
        next_time = future_times_s[:, idx]
        dt = (next_time - current_time).clamp(min=1.0e-6)
        state, cov = _predict(state, cov, dt, process_std, motion_model)
        state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
        preds.append(state[:, :4])
        current_time = next_time
    return torch.stack(preds, dim=1)
