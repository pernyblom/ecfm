from __future__ import annotations

from typing import Any, Dict

import torch


DEFAULT_KALMAN_CONFIG: dict[str, float | bool] = {
    "enabled": True,
    "initial_pos_std": 0.05,
    "initial_size_std": 0.05,
    "initial_vel_std": 1.0,
    "process_pos_std": 0.001,
    "process_size_std": 0.001,
    "process_vel_std": 0.1,
    "process_size_vel_std": 0.1,
    "measurement_pos_std": 0.01,
    "measurement_size_std": 0.01,
}


def kalman_config_from_dict(cfg: Dict[str, Any] | None) -> dict[str, float | bool]:
    out = dict(DEFAULT_KALMAN_CONFIG)
    if cfg:
        for key, value in cfg.items():
            if key not in out:
                continue
            if isinstance(out[key], bool):
                out[key] = bool(value)
            else:
                out[key] = float(value)
    return out


def _std_vector(params: Dict[str, float | bool], *, prefix: str, device, dtype) -> torch.Tensor:
    if prefix == "initial":
        return torch.tensor(
            [
                float(params["initial_pos_std"]),
                float(params["initial_pos_std"]),
                float(params["initial_size_std"]),
                float(params["initial_size_std"]),
                float(params["initial_vel_std"]),
                float(params["initial_vel_std"]),
                float(params["initial_vel_std"]),
                float(params["initial_vel_std"]),
            ],
            device=device,
            dtype=dtype,
        )
    if prefix == "process":
        return torch.tensor(
            [
                float(params["process_pos_std"]),
                float(params["process_pos_std"]),
                float(params["process_size_std"]),
                float(params["process_size_std"]),
                float(params["process_vel_std"]),
                float(params["process_vel_std"]),
                float(params["process_size_vel_std"]),
                float(params["process_size_vel_std"]),
            ],
            device=device,
            dtype=dtype,
        )
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
        if key != "enabled"
    }


def _std_vector_from_tensors(params: Dict[str, torch.Tensor], *, prefix: str) -> torch.Tensor:
    if prefix == "initial":
        return torch.stack(
            [
                params["initial_pos_std"],
                params["initial_pos_std"],
                params["initial_size_std"],
                params["initial_size_std"],
                params["initial_vel_std"],
                params["initial_vel_std"],
                params["initial_vel_std"],
                params["initial_vel_std"],
            ]
        )
    if prefix == "process":
        return torch.stack(
            [
                params["process_pos_std"],
                params["process_pos_std"],
                params["process_size_std"],
                params["process_size_std"],
                params["process_vel_std"],
                params["process_vel_std"],
                params["process_size_vel_std"],
                params["process_size_vel_std"],
            ]
        )
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


def _transition(dt: torch.Tensor) -> torch.Tensor:
    batch = int(dt.shape[0])
    f = torch.eye(8, device=dt.device, dtype=dt.dtype).unsqueeze(0).repeat(batch, 1, 1)
    f[:, 0, 4] = dt
    f[:, 1, 5] = dt
    f[:, 2, 6] = dt
    f[:, 3, 7] = dt
    return f


def _predict(state: torch.Tensor, cov: torch.Tensor, dt: torch.Tensor, q_base: torch.Tensor):
    f = _transition(dt)
    state = torch.bmm(f, state.unsqueeze(-1)).squeeze(-1)
    q_scale = torch.stack(
        [dt, dt, dt, dt, torch.ones_like(dt), torch.ones_like(dt), torch.ones_like(dt), torch.ones_like(dt)],
        dim=1,
    )
    q_diag = (q_base.unsqueeze(0) * q_scale.clamp(min=1.0e-6)).square()
    q = torch.diag_embed(q_diag)
    cov = torch.bmm(torch.bmm(f, cov), f.transpose(1, 2)) + q
    return state, cov


def _update(state: torch.Tensor, cov: torch.Tensor, measurement: torch.Tensor, r_diag: torch.Tensor):
    batch = int(state.shape[0])
    h = torch.zeros((batch, 4, 8), device=state.device, dtype=state.dtype)
    h[:, 0, 0] = 1.0
    h[:, 1, 1] = 1.0
    h[:, 2, 2] = 1.0
    h[:, 3, 3] = 1.0
    r = torch.diag_embed(r_diag.unsqueeze(0).expand(batch, -1).square())
    residual = measurement - torch.bmm(h, state.unsqueeze(-1)).squeeze(-1)
    s = torch.bmm(torch.bmm(h, cov), h.transpose(1, 2)) + r
    k = torch.linalg.solve(s, torch.bmm(h, cov)).transpose(1, 2)
    state = state + torch.bmm(k, residual.unsqueeze(-1)).squeeze(-1)
    eye = torch.eye(8, device=state.device, dtype=state.dtype).unsqueeze(0).expand(batch, -1, -1)
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
    state = torch.zeros((batch, 8), device=device, dtype=dtype)
    state[:, :4] = past_boxes[:, 0]
    init_std = _std_vector(cfg, prefix="initial", device=device, dtype=dtype)
    process_std = _std_vector(cfg, prefix="process", device=device, dtype=dtype)
    meas_std = _std_vector(cfg, prefix="measurement", device=device, dtype=dtype)
    cov = torch.diag_embed(init_std.unsqueeze(0).expand(batch, -1).square())
    state, cov = _update(state, cov, past_boxes[:, 0], meas_std)
    for idx in range(1, past_boxes.shape[1]):
        dt = (past_times_s[:, idx] - past_times_s[:, idx - 1]).clamp(min=1.0e-6)
        state, cov = _predict(state, cov, dt, process_std)
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
        state, cov = _predict(state, cov, dt, process_std)
        state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
        preds.append(state[:, :4])
        current_time = next_time
    return torch.stack(preds, dim=1)


def kalman_filter_history_tensor_params(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(past_boxes.shape[0])
    device = past_boxes.device
    dtype = past_boxes.dtype
    state = torch.cat(
        [
            past_boxes[:, 0],
            torch.zeros((batch, 4), device=device, dtype=dtype),
        ],
        dim=-1,
    )
    init_std = _std_vector_from_tensors(params, prefix="initial").to(device=device, dtype=dtype)
    process_std = _std_vector_from_tensors(params, prefix="process").to(device=device, dtype=dtype)
    meas_std = _std_vector_from_tensors(params, prefix="measurement").to(device=device, dtype=dtype)
    cov = torch.diag_embed(init_std.unsqueeze(0).expand(batch, -1).square())
    state, cov = _update(state, cov, past_boxes[:, 0], meas_std)
    for idx in range(1, past_boxes.shape[1]):
        dt = (past_times_s[:, idx] - past_times_s[:, idx - 1]).clamp(min=1.0e-6)
        state, cov = _predict(state, cov, dt, process_std)
        state, cov = _update(state, cov, past_boxes[:, idx], meas_std)
    return state, cov


def kalman_cv_forecast_tensor_params(
    past_boxes: torch.Tensor,
    past_times_s: torch.Tensor,
    future_times_s: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    state, cov = kalman_filter_history_tensor_params(past_boxes, past_times_s, params)
    process_std = _std_vector_from_tensors(params, prefix="process").to(
        device=past_boxes.device,
        dtype=past_boxes.dtype,
    )
    current_time = past_times_s[:, -1]
    preds: list[torch.Tensor] = []
    for idx in range(future_times_s.shape[1]):
        next_time = future_times_s[:, idx]
        dt = (next_time - current_time).clamp(min=1.0e-6)
        state, cov = _predict(state, cov, dt, process_std)
        state = torch.cat([state[:, :4].clamp(0.0, 1.0), state[:, 4:]], dim=-1)
        preds.append(state[:, :4])
        current_time = next_time
    return torch.stack(preds, dim=1)
