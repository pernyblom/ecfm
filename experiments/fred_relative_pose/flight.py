"""Camera-relative multirotor state and differentiable short-horizon dynamics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class FlightState:
    """Batched anchor state in camera axes (right, down, forward).

    Lengths are in assumed body diameters. Angular velocity is in rad/s,
    and thrust is specific thrust in body diameters/s² (shape [B, 1]).
    """

    position: torch.Tensor
    up: torch.Tensor
    velocity: torch.Tensor
    angular_velocity: torch.Tensor
    thrust: torch.Tensor

    def map(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> FlightState:
        """Apply a batch or time-axis transformation to each state component."""
        return FlightState(*(transform(getattr(self, field.name)) for field in fields(self)))


@dataclass(frozen=True)
class FlightTrajectory:
    position: torch.Tensor
    up: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor


def _rotate(state: FlightState, time: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula using sinc factors at zero angular velocity."""
    extra_axes = (1,) * (time.ndim - 2)
    up = state.up.reshape(-1, *extra_axes, 3)
    omega = state.angular_velocity.reshape_as(up)
    cross = torch.linalg.cross(omega, up)
    angle = omega.norm(dim=-1, keepdim=True) * time
    return (up + time * torch.sinc(angle / math.pi) * cross
            + 0.5 * (time * torch.sinc(angle / (2 * math.pi))).square() * torch.linalg.cross(omega, cross))


def rollout_flight(state: FlightState, times_s: torch.Tensor, gravity_up: torch.Tensor,
                   gravity: float, linear_drag: float = 0.0) -> FlightTrajectory:
    """Propagate [B, T] elapsed times under constant rate, thrust and linear drag.

    Velocity obeys dv/dt = thrust * up(t) - gravity * gravity_up - drag * v.
    Integrate the fixed-up contribution analytically and the change from
    rotation with four-point Gauss–Legendre quadrature over each query time.
    """
    time = times_s[..., None]
    fixed_drive = (state.thrust * state.up - gravity * gravity_up)[:, None]
    if linear_drag > 0:
        # Evaluate in terms of the dimensionless drag*time, using the series
        # for the second integral where subtracting expm1 loses precision.
        x = linear_drag * time
        decay = (-x).exp()
        velocity_time = -torch.expm1(-x) / linear_drag
        drive_time = torch.where(x.abs() < 0.01, time.square() * (0.5 - x / 6 + x.square() / 24),
                                 (time - velocity_time) / linear_drag)
    else:
        decay, velocity_time, drive_time = 1.0, time, 0.5 * time.square()

    nodes = time.new_tensor([0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558]).view(1, 1, 4, 1)
    weights = time.new_tensor([0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226]).view(1, 1, 4, 1)
    node_time = time.unsqueeze(-2) * nodes
    tail = time.unsqueeze(-2) - node_time
    force_delta = state.thrust[:, None, None] * (_rotate(state, node_time) - state.up[:, None, None])
    if linear_drag > 0:
        velocity_kernel = (-linear_drag * tail).exp()
        position_kernel = -torch.expm1(-linear_drag * tail) / linear_drag
    else:
        velocity_kernel, position_kernel = 1.0, tail
    velocity = (state.velocity[:, None] * decay + fixed_drive * velocity_time
                + time * (weights * velocity_kernel * force_delta).sum(dim=-2))
    position = (state.position[:, None] + state.velocity[:, None] * velocity_time + fixed_drive * drive_time
                + time * (weights * position_kernel * force_delta).sum(dim=-2))
    up = _rotate(state, time)
    acceleration = state.thrust[:, None] * up - gravity * gravity_up[:, None] - linear_drag * velocity
    return FlightTrajectory(position, up, velocity, acceleration)
