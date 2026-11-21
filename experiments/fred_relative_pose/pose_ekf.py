"""Camera-frame 3D pose filtering with nonlinear image box measurements."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .flight import FlightState, FlightTrajectory
from .model import RelativePoseModel, project_body, tangent_basis
from .pose_kalman import TIME_UNIT_S, box_coordinates


NEAR_DEPTH = .6  # Body diameters, matching the ellipsoid projector.
STATE_DIM = 13


def pose_observations(model: RelativePoseModel, state: FlightState, gravity_up: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Map physical measurements to the EKF coordinates using a fixed range scale per tracklet."""
    right, forward = tangent_basis(gravity_up)
    axes = torch.stack((right, forward), dim=-1)
    gravity_up = gravity_up[:, None]
    relative_depth = (state.position[..., 2:] - NEAR_DEPTH).clamp_min(.2)
    range_scale = relative_depth[:, 0]
    scale = range_scale[:, None]
    radial_rate = state.velocity[..., 2:] / relative_depth
    acceleration = (state.thrust * state.up - model.gravity * gravity_up
                    - model.linear_drag * state.velocity)
    vertical = (state.up * gravity_up).sum(-1, keepdim=True)
    tilt = (state.up @ axes) / vertical
    up_rate = torch.linalg.cross(state.angular_velocity, state.up)
    tilt_rate = (up_rate @ axes - tilt * (up_rate * gravity_up).sum(-1, keepdim=True)) / vertical
    observations = torch.cat((
        state.position[..., :2] / scale, (relative_depth / scale).log(),
        state.velocity[..., :2] / scale * TIME_UNIT_S, radial_rate * TIME_UNIT_S,
        acceleration[..., :2] / scale * TIME_UNIT_S**2,
        (acceleration[..., 2:] / relative_depth - radial_rate.square()) * TIME_UNIT_S**2,
        tilt, tilt_rate * TIME_UNIT_S,
    ), dim=-1)
    return observations, range_scale


def measurement_update(mean: torch.Tensor, covariance: torch.Tensor, innovation: torch.Tensor,
                       jacobian: torch.Tensor, variance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a sensor update from its innovation, linearization, and diagonal noise."""
    cross_covariance = covariance @ jacobian.transpose(-2, -1)
    noise = torch.diag_embed(variance)
    innovation_covariance = jacobian @ cross_covariance + noise
    gain = torch.linalg.solve_ex(innovation_covariance, cross_covariance.transpose(-2, -1)).result.transpose(-2, -1)
    mean = mean + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
    # Joseph form preserves positive semidefiniteness when measurement noise is small.
    residual_map = torch.eye(mean.shape[-1], device=mean.device, dtype=mean.dtype) - gain @ jacobian
    covariance = residual_map @ covariance @ residual_map.transpose(-2, -1) + gain @ noise @ gain.transpose(-2, -1)
    return mean, (covariance + covariance.transpose(-2, -1)) * .5


class PoseEKF3D(nn.Module):
    """Damped 3D acceleration and tilt-rate state with pose and optional box updates.

    State order is [x, y, log(z-near), v_x, v_y, log-depth rate, a_x, a_y,
    log-depth acceleration, up slope x/y, up slope rate x/y]. Cartesian
    lateral terms use a fixed range scale and rates use a fixed time unit.
    """

    def __init__(self, frame_size: tuple[int, int], half_height: float, *, use_boxes: bool):
        super().__init__()
        self.half_height = half_height
        self.use_boxes = use_boxes
        self.register_buffer("pixel_scale", torch.tensor(frame_size * 2, dtype=torch.float32))
        self.log_initial_std = nn.Parameter(torch.tensor(
            [.025, .025, .10, .05, .05, .10, .035, .035, .06, .3, .3, .2, .2],
        ).log())
        self.log_process_std = nn.Parameter(torch.tensor(
            [.003, .003, .015, .004, .004, .015, .006, .006, .02, .08, .08, .06, .06],
        ).log())
        self.log_pose_std = nn.Parameter(torch.tensor(
            [.004, .004, .045, .018, .018, .045, .018, .018, .035, .27, .27, .17, .17],
        ).log())
        self.log_acceleration_decay = nn.Parameter(torch.full((3,), .38).log())
        self.log_tilt_rate_decay = nn.Parameter(torch.full((2,), .4).log())
        if use_boxes:
            self.log_box_std = nn.Parameter(torch.tensor([.003, .003, .014, .014]).log())
            self.box_shared_error = nn.Parameter(torch.zeros(4))
            self.register_buffer("geometry_offsets", torch.cat((torch.zeros(1, 5), torch.eye(5), -torch.eye(5))))

    def _transition(self, seconds: torch.Tensor) -> torch.Tensor:
        dt = seconds.unsqueeze(-1) / TIME_UNIT_S
        decay_a = self.log_acceleration_decay.exp()
        retained_a = (-dt * decay_a).exp()
        from_acceleration = -torch.expm1(-dt * decay_a) / decay_a
        decay_tilt_rate = self.log_tilt_rate_decay.exp()
        retained_tilt_rate = (-dt * decay_tilt_rate).exp()
        diagonal = torch.cat((torch.ones_like(dt).expand(*dt.shape[:-1], 6), retained_a,
                              torch.ones_like(retained_tilt_rate), retained_tilt_rate), dim=-1)
        transition = torch.diag_embed(diagonal)
        transition[..., :3, 3:6] = torch.diag_embed(dt.expand_as(retained_a))
        transition[..., :3, 6:9] = torch.diag_embed((dt - from_acceleration) / decay_a)
        transition[..., 3:6, 6:9] = torch.diag_embed(from_acceleration)
        transition[..., 9:11, 11:13] = torch.diag_embed(-torch.expm1(-dt * decay_tilt_rate) / decay_tilt_rate)
        return transition

    @staticmethod
    def _geometry(mean: torch.Tensor) -> torch.Tensor:
        return torch.cat((mean[..., :3], mean[..., 9:11]), dim=-1)

    @staticmethod
    def _decode_geometry(geometry: torch.Tensor, range_scale: torch.Tensor,
                         gravity_up: torch.Tensor, axes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = range_scale[:, None]
        depth = scale * geometry[..., 2:3].clamp(-4., 4.).exp() + NEAR_DEPTH
        position = torch.cat((geometry[..., :2] * scale, depth), dim=-1)
        up = F.normalize(gravity_up[:, None] + geometry[..., 3:] @ axes, dim=-1)
        return position, up

    def _project(self, geometry: torch.Tensor, range_scale: torch.Tensor, camera: dict,
                 axes: torch.Tensor) -> torch.Tensor:
        position, up = self._decode_geometry(geometry, range_scale, camera["gravity_up"], axes)
        return project_body(position, up, camera, self.half_height) / self.pixel_scale

    def _box_linearization(self, mean: torch.Tensor, range_scale: torch.Tensor,
                           camera: dict, axes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = .001
        perturbed = self._geometry(mean)[:, None] + epsilon * self.geometry_offsets
        samples = box_coordinates(self._project(perturbed, range_scale, camera, axes))
        geometry_jacobian = ((samples[:, 1:6] - samples[:, 6:]) / (2 * epsilon)).transpose(-1, -2)
        jacobian = torch.cat((geometry_jacobian[..., :3], F.pad(geometry_jacobian[..., 3:], (6, 2))), dim=-1)
        return samples[:, 0], jacobian

    def filter(self, past: torch.Tensor, past_time: torch.Tensor, pose: torch.Tensor,
               range_scale: torch.Tensor, camera: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the 3D posterior in filter coordinates and its full covariance."""
        if pose.shape != (*past.shape[:2], STATE_DIM):
            raise ValueError("One 3D pose observation is required per history timestamp")
        mean = torch.cat((pose[:, 0, :3], torch.zeros_like(pose[:, 0, 3:9]),
                          pose[:, 0, 9:11], torch.zeros_like(pose[:, 0, 11:])), dim=-1)
        covariance = torch.diag((2 * self.log_initial_std).exp()).expand(len(mean), -1, -1)
        process = torch.diag((2 * self.log_process_std).exp())
        pose_variance = (2 * self.log_pose_std).exp()
        identity = torch.eye(STATE_DIM, device=mean.device, dtype=mean.dtype)
        deltas = torch.diff(past_time, dim=1)
        transition = self._transition(deltas)
        if self.use_boxes:
            axes = torch.stack(tangent_basis(camera["gravity_up"]), dim=1)
            boxes = box_coordinates(past)
            pose_boxes = box_coordinates(self._project(self._geometry(pose), range_scale, camera, axes))
            box_variance = (2 * self.log_box_std).exp()
            shared = .95 * self.box_shared_error.sigmoid()
        for i, observation in enumerate(pose.unbind(dim=1)):
            if i:
                matrix = transition[:, i - 1]
                elapsed = deltas[:, i - 1, None, None] / TIME_UNIT_S
                mean = (matrix @ mean.unsqueeze(-1)).squeeze(-1)
                covariance = matrix @ covariance @ matrix.transpose(-2, -1) + elapsed * process
            mean, covariance = measurement_update(mean, covariance, observation - mean, identity, pose_variance)
            if self.use_boxes:
                prediction, jacobian = self._box_linearization(mean, range_scale, camera, axes)
                # Condition on part of the apparent box error already observed through the pose.
                innovation = boxes[:, i] - prediction - shared * (pose_boxes[:, i] - prediction)
                mean, covariance = measurement_update(
                    mean, covariance, innovation, (1 - shared)[..., None] * jacobian, box_variance,
                )
        return mean, covariance

    def rollout(self, mean: torch.Tensor, times_s: torch.Tensor, range_scale: torch.Tensor,
                gravity_up: torch.Tensor) -> FlightTrajectory:
        """Return future 3D position, up, velocity and acceleration in body-diameter units."""
        states = (self._transition(times_s) @ mean[:, None, :, None]).squeeze(-1)
        axes = torch.stack(tangent_basis(gravity_up), dim=1)
        position, up = self._decode_geometry(self._geometry(states), range_scale, gravity_up, axes)
        scale = range_scale[:, None]
        depth_margin = position[..., 2:] - NEAR_DEPTH
        # The depth guard is constant beyond its limits, so its derivatives vanish there.
        within_guard = states[..., 2:3].abs() < 4.
        log_rate = states[..., 5:6] * within_guard
        velocity = torch.cat((states[..., 3:5] * scale, depth_margin * log_rate), dim=-1) / TIME_UNIT_S
        depth_acceleration = depth_margin * (states[..., 8:9] * within_guard + log_rate.square())
        acceleration = torch.cat((states[..., 6:8] * scale, depth_acceleration), dim=-1) / TIME_UNIT_S**2
        return FlightTrajectory(position, up, velocity, acceleration)

    def forward(self, past: torch.Tensor, past_time: torch.Tensor, future_time: torch.Tensor,
                pose: torch.Tensor, range_scale: torch.Tensor, camera: dict) -> torch.Tensor:
        mean, _ = self.filter(past, past_time, pose, range_scale, camera)
        trajectory = self.rollout(mean, future_time - past_time[:, -1:], range_scale, camera["gravity_up"])
        return project_body(trajectory.position, trajectory.up, camera, self.half_height) / self.pixel_scale

    @torch.no_grad()
    def clamp_(self) -> None:
        stds = (self.log_initial_std, self.log_process_std, self.log_pose_std)
        if self.use_boxes:
            stds += (self.log_box_std,)
        for log_std in stds:
            log_std.clamp_(-8., 1.)
        self.log_acceleration_decay.clamp_(-4., 1.5)
        self.log_tilt_rate_decay.clamp_(-4., 1.5)
