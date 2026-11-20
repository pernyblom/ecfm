"""Differentiable camera, yaw-symmetric body and thrust rollout."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.kalman_ml_forecasting.utils.config import resolve_event_image_options

from .flight import FlightState, FlightTrajectory, rollout_flight


def tangent_basis(up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([1, 0, 0], dtype=up.dtype, device=up.device)
    right = F.normalize(x - up[..., :1] * up, dim=-1)
    return right, torch.linalg.cross(up, right)


def shifted_up(up: torch.Tensor, xy: torch.Tensor, tangent_limit: float) -> torch.Tensor:
    right, forward = tangent_basis(up)
    tangent = torch.tanh(xy)
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1)
    return F.normalize(up + tangent_limit * (tangent[..., :1] * right + tangent[..., 1:] * forward), dim=-1)


class SceneCameras(nn.Module):
    def __init__(self, frame_size, up_priors, config: dict):
        super().__init__()
        prior = torch.as_tensor(up_priors, dtype=torch.float32)
        if prior.ndim != 2 or prior.shape[1] != 3 or not torch.isfinite(prior).all():
            raise ValueError("Expected finite per-scene three-vector priors")
        if (torch.linalg.vector_norm(prior[:, 1:], dim=-1) <= 1e-3).any():
            raise ValueError("Scene up priors need a y or z component")
        self.register_buffer("up_prior", F.normalize(prior, dim=-1))
        self.register_buffer("frame_size", torch.as_tensor(frame_size, dtype=torch.float32))
        self.offsets = nn.Parameter(torch.zeros(len(prior), 6))
        self.focal_prior = float(config["focal_prior_px"])
        self.focal_log_limit = float(config["focal_log_limit"])
        self.center_limit = float(config["principal_point_limit"])
        self.up_limit = math.tan(math.radians(float(config["up_prior_limit_deg"])))

    def forward(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        offsets = self.offsets[ids]
        focal = self.focal_prior * (self.focal_log_limit * offsets[:, :2].tanh()).exp()
        center = self.frame_size * (0.5 + self.center_limit * offsets[:, 2:4].tanh())
        up = shifted_up(self.up_prior[ids], offsets[:, 4:6], self.up_limit)
        return {"focal": focal, "center": center, "gravity_up": up, "camera_offsets": offsets}


def project_points(xyz: torch.Tensor, camera: dict) -> torch.Tensor:
    """Project [batch, time, xyz] points to event-camera pixels."""
    xy = xyz[..., :2] / xyz[..., 2:].clamp_min(0.6)
    return xy * camera["focal"][:, None] + camera["center"][:, None]


def project_body(positions: torch.Tensor, up: torch.Tensor, camera: dict, half_height: float) -> torch.Tensor:
    """Exact perspective bounds of an oblate ellipsoid, in pixel cx/cy/w/h.

    Accept one up vector per trajectory or a vector at every query time.
    A tangent plane x - q*z = 0 gives the two extrema of projected x/z.
    Use the expanded discriminant to avoid subtraction of large depth powers.
    """
    radius_sq = 0.25
    thickness = half_height ** 2 - radius_sq
    z = positions[..., 2:].clamp_min(0.6)
    xy = positions[..., :2]
    up = up[:, None] if up.ndim == 2 else up
    u_xy, u_z = up[..., :2], up[..., 2:]
    s_ii = radius_sq + thickness * u_xy.square()
    s_zz = radius_sq + thickness * u_z.square()
    s_iz = thickness * u_xy * u_z
    denominator = z.square() - s_zz
    disc = z.square() * s_ii + xy.square() * s_zz - 2 * xy * z * s_iz + s_iz.square() - s_zz * s_ii
    center = (xy * z - s_iz) / denominator * camera["focal"][:, None] + camera["center"][:, None]
    size = 2 * disc.clamp_min(1e-8).sqrt() / denominator * camera["focal"][:, None]
    return torch.cat((center, size), dim=-1)


def history_slope(values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Least-squares derivative with a free intercept and actual timestamps."""
    dt = times - times.mean(dim=1, keepdim=True)
    return (values * dt[..., None]).sum(dim=1) / dt.square().sum(dim=1, keepdim=True).clamp_min(1e-7)


class EventEncoder(nn.Sequential):
    def __init__(self, channels: int):
        layers = []
        for width in (24, 40, 64, 80):
            layers.extend((nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, width), nn.SiLU()))
            channels = width
        super().__init__(*layers, nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
                         nn.Linear(80 * 9, 96), nn.LayerNorm(96), nn.SiLU())


class RelativePoseModel(nn.Module):
    def __init__(self, config: dict, up_priors):
        super().__init__()
        data, model = config["data"], config["model"]
        self.cameras = SceneCameras(data["frame_size"], up_priors, config["camera"])
        self.gravity = 9.81 / float(model["reference_diameter_m"])
        self.half_height = float(model["body_half_height"])
        if not 0 < self.half_height <= 0.5 or self.gravity <= 0:
            raise ValueError("Expected a positive reference size and an oblate body")
        self.tilt_limit = math.tan(math.radians(float(model["tilt_limit_deg"])))
        self.seed_history = int(model["velocity_history_steps"])
        if self.seed_history < 2:
            raise ValueError("Velocity initialization needs at least two history steps")
        self.use_box_motion = bool(model["use_box_motion"])
        self.encode_box_history = self.use_box_motion and bool(model.get("encode_box_history", False))
        self.history_steps = max(1, round(data["history_ms"] / (1000 * data["label_period_s"])))
        self.range_guard_s = float(data["forecast_ms"]) / 1000
        self.angular_limit = math.radians(float(model.get("angular_speed_limit_deg_s", 0)))
        self.velocity_limit = float(model["velocity_limit_px_s"])
        self.event_dropout = float(model.get("event_dropout", 0))
        self.linear_drag = float(model.get("linear_drag_per_s", 0))
        if not 0 <= self.event_dropout < 1 or self.linear_drag < 0:
            raise ValueError("Event dropout must be in [0, 1) and linear drag must be nonnegative")
        if not math.isfinite(self.angular_limit) or self.angular_limit < 0:
            raise ValueError("Angular speed limit must be finite and nonnegative")
        channels = resolve_event_image_options(data)["representation_channels"]
        self.encoders = nn.ModuleDict({rep: EventEncoder(channels[rep]) for rep in data["representations"]}
                                      if model["use_events"] else {})
        feature_dim = 4 + 3 * self.use_box_motion + 5 * self.history_steps * self.encode_box_history + 96 * len(self.encoders)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.LayerNorm(128), nn.SiLU(),
                                  nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 11 if self.angular_limit else 9))
        nn.init.normal_(self.head[-1].weight, std=0.002)
        nn.init.zeros_(self.head[-1].bias)

    def estimate_state(self, batch: dict, *, mask_events: bool = False) -> FlightState:
        """Infer the anchor state from past boxes, event cutouts, and scene IDs."""
        return self._estimate_state(batch, self.cameras(batch["scene"]), mask_events)[0]

    def propagate(self, state: FlightState, times_s: torch.Tensor, gravity_up: torch.Tensor) -> FlightTrajectory:
        return rollout_flight(state, times_s, gravity_up, self.gravity, self.linear_drag)

    def _estimate_state(self, batch: dict, camera: dict, mask_events: bool) -> tuple[FlightState, torch.Tensor]:
        past, times = batch["past"], batch["past_time"]
        frame = self.cameras.frame_size
        base = past[:, -1]
        base_size = (base[:, 2:] * frame).clamp_min(1)
        base_center = base[:, :2] * frame
        features = [2 * base[:, :2] - 1, (base_size / 30).log()]
        motion = torch.zeros_like(base[:, :3])
        if self.use_box_motion:
            history = past[:, -self.seed_history:]
            history_size = (history[..., 2:] * frame).clamp_min(1).log().mean(dim=-1, keepdim=True)
            motion = history_slope(torch.cat((history[..., :2] * frame, history_size), dim=-1),
                                   times[:, -self.seed_history:])
            features.append(torch.cat((motion[:, :2] / 250, motion[:, 2:]), dim=-1).clamp(-3, 3))
        if self.encode_box_history:
            if past.shape[1] != self.history_steps:
                raise ValueError(f"Expected {self.history_steps} box history steps, got {past.shape[1]}")
            offsets = (past[..., :2] * frame - base_center[:, None]) / 64
            relative_size = ((past[..., 2:] * frame).clamp_min(1) / base_size[:, None]).log()
            features.append(torch.cat((offsets, relative_size, times[..., None] * 2.5), dim=-1).clamp(-5, 5).flatten(1))
        first_event = len(features)
        for rep, encoder in self.encoders.items():
            features.append(encoder(batch["events"][rep]) if not mask_events else base.new_zeros((len(base), 96)))
        if self.encoders and self.training and self.event_dropout:
            # Drop whole event inputs occasionally to train the masked-event diagnostic.
            keep = (torch.rand((len(base), 1), device=base.device) >= self.event_dropout).float()
            features[first_event:] = [feature * keep for feature in features[first_event:]]
        controls = self.head(torch.cat(features, dim=-1)).float()

        up = shifted_up(camera["gravity_up"], controls[:, :2], self.tilt_limit)
        ray_xy = (base_center + 8 * controls[:, 2:4].tanh() - camera["center"]) / camera["focal"]
        ray = torch.cat((ray_xy, torch.ones_like(ray_xy[:, :1])), dim=-1)
        # The local projection differential gives a size-based range initialization.
        support = 0.25 * (1 + ray_xy.square()) + (self.half_height ** 2 - 0.25) * (
            up[:, :2] - ray_xy * up[:, 2:]).square()
        depth = ((2 * support.clamp_min(1e-6).sqrt() * camera["focal"] / base_size).log().mean(dim=-1, keepdim=True)
                 + 0.4 * controls[:, 4:5].tanh()).exp().clamp_min(0.8)
        position = ray * depth
        image_velocity = motion[:, :2].clamp(-1100, 1100) + self.velocity_limit * controls[:, 5:7].tanh()
        range_rate = -motion[:, 2:3].clamp(-1.3, 1.3) + 1.15 * controls[:, 7:8].tanh()
        # Set the approach guard from the training horizon so state inference
        # is independent of the timestamps subsequently used for propagation.
        range_rate = range_rate.clamp_min(-0.75 / self.range_guard_s)
        lateral = image_velocity * depth / camera["focal"]
        velocity = torch.cat((lateral, torch.zeros_like(depth)), dim=-1) + position * range_rate
        thrust = self.gravity * (0.85 * controls[:, 8:9].tanh()).exp()
        omega = torch.zeros_like(up)
        if self.angular_limit:
            right, forward = tangent_basis(up)
            components = controls[:, 9:11].tanh()
            components = components / components.norm(dim=-1, keepdim=True).clamp_min(1)
            omega = self.angular_limit * (components[:, :1] * right + components[:, 1:] * forward)
        return FlightState(position, up, velocity, omega, thrust), controls

    def forward(self, batch: dict, *, mask_events: bool = False) -> dict:
        camera = self.cameras(batch["scene"])
        state, controls = self._estimate_state(batch, camera, mask_events)
        future = batch["future_time"]
        query = torch.cat((torch.zeros_like(future[:, :1]), future), dim=1)
        trajectory = self.propagate(state, query, camera["gravity_up"])
        constant_velocity = state.position[:, None] + state.velocity[:, None] * query[..., None]
        boxes = project_body(trajectory.position, trajectory.up, camera, self.half_height)
        return {**camera, "state": state, "trajectory": trajectory, "position": state.position,
                "velocity": state.velocity, "up": state.up, "angular_velocity": state.angular_velocity,
                "acceleration": trajectory.acceleration[:, 0], "thrust": state.thrust,
                "rollout": trajectory.position, "boxes_px": boxes, "controls": controls,
                "cv_rollout": constant_velocity}
