"""Kalman fusion of box and pose observations in the event-camera image."""

from __future__ import annotations

import torch
from torch import nn

from .flight import FlightState
from .model import RelativePoseModel, project_body, repeat_camera


TIME_UNIT_S = 0.1
LOG_SIZE_SCALE = 0.25


def box_coordinates(boxes: torch.Tensor) -> torch.Tensor:
    """Convert normalized boxes to center and scaled log-width/log-height."""
    return torch.cat((boxes[..., :2], boxes[..., 2:].clamp_min(1e-5).log() * LOG_SIZE_SCALE), dim=-1)


def pose_measurements(model: RelativePoseModel, state: FlightState, camera: dict,
                      frame_size: tuple[int, int]) -> torch.Tensor:
    """Measure box coordinate, rate, and acceleration from each separate pose state."""
    batch, history, _ = state.position.shape
    flat = state.map(lambda value: value.flatten(0, 1))
    camera = repeat_camera(camera, history)
    dt = TIME_UNIT_S / 4
    queries = flat.position.new_tensor([-dt, 0., dt]).expand(batch * history, -1)
    rollout = model.propagate(flat, queries, camera["gravity_up"])
    pixels = project_body(rollout.position, rollout.up, camera, model.half_height)
    before, center, after = box_coordinates(pixels / pixels.new_tensor(frame_size * 2)).unbind(dim=1)
    # Rates are per TIME_UNIT_S. Only nearby projections of this observation are used.
    rate = (after - before) * (TIME_UNIT_S / (2 * dt))
    acceleration = (after - 2 * center + before) * (TIME_UNIT_S / dt) ** 2
    return torch.stack((center, rate, acceleration), dim=-1).unflatten(0, (batch, history))


class PoseBoxKalman(nn.Module):
    """Four scalar-coordinate Kalman filters with three kinematic states each.

    The detector observes position. The projected pose observes position, velocity
    and acceleration. The two same-frame position measurements may be correlated.
    """

    def __init__(self, *, use_pose: bool):
        super().__init__()
        self.use_pose = use_pose
        self.log_initial_std = nn.Parameter(torch.tensor([.025, .045, .025]).log().repeat(4, 1))
        self.log_process_std = nn.Parameter(torch.tensor([.002, .003, .007]).log().repeat(4, 1))
        measurement_std = [.005, .008, .016, .018] if use_pose else [.005]
        self.log_measurement_std = nn.Parameter(torch.tensor(measurement_std).log().repeat(4, 1))
        self.log_acceleration_decay = nn.Parameter(torch.full((4,), .35).log())
        if use_pose:
            self.position_correlation = nn.Parameter(torch.full((4,), .6).atanh())

    def _transition_terms(self, seconds: torch.Tensor) -> tuple[torch.Tensor, ...]:
        time = seconds.unsqueeze(-1) / TIME_UNIT_S
        decay_rate = self.log_acceleration_decay.exp()
        decay = (-time * decay_rate).exp()
        velocity_from_acceleration = -torch.expm1(-time * decay_rate) / decay_rate
        position_from_acceleration = (time - velocity_from_acceleration) / decay_rate
        return time.expand_as(decay), position_from_acceleration, velocity_from_acceleration, decay

    def _transition(self, seconds: torch.Tensor) -> torch.Tensor:
        time, pos_a, vel_a, decay = self._transition_terms(seconds)
        one, zero = torch.ones_like(time), torch.zeros_like(time)
        return torch.stack((one, time, pos_a, zero, one, vel_a, zero, zero, decay), dim=-1).unflatten(-1, (3, 3))

    def _measurements(self, box: torch.Tensor, pose: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        variance = (2 * self.log_measurement_std).exp()
        if not self.use_pose:
            return box.unsqueeze(-1), variance
        if pose is None or pose.shape != (*box.shape, 3):
            raise ValueError("Pose fusion requires position, rate, and acceleration for every observed box")
        box_var, pose_var, rate_var, acc_var = variance.unbind(dim=-1)
        covariance = .95 * self.position_correlation.tanh() * (box_var * pose_var).sqrt()
        difference_var = box_var + pose_var - 2 * covariance
        # Generalized least squares for the two observations of the same position.
        fused = box + (box_var - covariance) / difference_var * (pose[..., 0] - box)
        fused_var = (box_var * pose_var - covariance.square()) / difference_var
        return torch.cat((fused.unsqueeze(-1), pose[..., 1:]), dim=-1), torch.stack(
            (fused_var, rate_var, acc_var), dim=-1,
        )

    def filter(self, past: torch.Tensor, past_time: torch.Tensor,
               pose: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the filtered mean and covariance at the last observed time."""
        coordinates = box_coordinates(past)
        measurements, measurement_var = self._measurements(coordinates, pose)
        initial_position = coordinates[:, 0]
        zero = torch.zeros_like(initial_position)
        mean = torch.stack((initial_position, zero, zero), dim=-1)
        covariance = torch.diag_embed((2 * self.log_initial_std).exp()).expand(len(past), -1, -1, -1)
        process = torch.diag_embed((2 * self.log_process_std).exp())
        deltas = torch.diff(past_time, dim=1)
        transition = self._transition(deltas)
        measurement_noise = torch.diag_embed(measurement_var)
        for step in range(past.shape[1]):
            if step:
                f = transition[:, step - 1]
                elapsed = deltas[:, step - 1, None, None, None] / TIME_UNIT_S
                mean = (f @ mean.unsqueeze(-1)).squeeze(-1)
                covariance = f @ covariance @ f.transpose(-2, -1) + elapsed * process
            observed = measurements[:, step]
            if self.use_pose:
                innovation_cov = covariance + measurement_noise
                gain = torch.linalg.solve_ex(innovation_cov, covariance).result.transpose(-2, -1)
            else:
                gain = covariance[..., :1] / (covariance[..., 0, 0] + measurement_var[:, 0])[..., None, None]
            observed_dims = observed.shape[-1]
            innovation = observed - mean[..., :observed_dims]
            mean = mean + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
            covariance = covariance - gain @ covariance[..., :observed_dims, :]
            covariance = (covariance + covariance.transpose(-2, -1)) * .5
        return mean, covariance

    def forward(self, past: torch.Tensor, past_time: torch.Tensor, future_time: torch.Tensor,
                pose: torch.Tensor | None = None) -> torch.Tensor:
        state, _ = self.filter(past, past_time, pose)
        time, pos_a, _, _ = self._transition_terms(future_time - past_time[:, -1:])
        position, velocity, acceleration = state.unbind(-1)
        coordinates = position[:, None] + time * velocity[:, None] + pos_a * acceleration[:, None]
        sizes = (coordinates[..., 2:] / LOG_SIZE_SCALE).clamp(min=-11.5, max=0.).exp()
        return torch.cat((coordinates[..., :2], sizes), dim=-1)

    @torch.no_grad()
    def clamp_(self) -> None:
        for log_std in (self.log_initial_std, self.log_process_std, self.log_measurement_std):
            log_std.clamp_(-9., 1.)
        self.log_acceleration_decay.clamp_(-4., 1.5)
