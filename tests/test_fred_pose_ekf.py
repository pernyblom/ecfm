import pytest
import torch
from torch.nn import functional as F

from experiments.fred_relative_pose.flight import FlightState
from experiments.fred_relative_pose.model import RelativePoseModel, project_body, tangent_basis
from experiments.fred_relative_pose.pose_ekf import PoseEKF3D, pose_observations
from experiments.fred_relative_pose.pose_kalman import box_coordinates
from experiments.fred_relative_pose.train import DEFAULT_CONFIG
from experiments.kalman_ml_forecasting.utils.config import load_config


def camera_for(reference):
    return {"focal": reference.new_tensor([[1050., 1200.]]),
            "center": reference.new_tensor([[635., 354.]]),
            "gravity_up": F.normalize(reference.new_tensor([[.05, -.9, .4]]), dim=-1)}


def test_3d_observation_roundtrip_preserves_physical_state():
    position = torch.tensor([[[3., -1., 20.], [4., -2., 19.]]], dtype=torch.float64)
    camera = camera_for(position)
    up = F.normalize(position.new_tensor([[[.2, -.8, .45], [.1, -.7, .5]]]), dim=-1)
    velocity = position.new_tensor([[[3., -2., -1.], [4., -2., -1.]]])
    omega = position.new_tensor([[[.3, -.1, .08], [.4, -.08, .02]]])
    thrust = position.new_tensor([[[15.], [14.]]])
    state = FlightState(position, up, velocity, omega, thrust)
    model = RelativePoseModel(load_config(DEFAULT_CONFIG), [[0., -1., 0.]])
    ekf = PoseEKF3D((1280, 720), model.half_height, use_boxes=True).double()
    obs, scale = pose_observations(model, state, camera["gravity_up"])
    trajectory = ekf.rollout(obs[:, 1], position.new_zeros((1, 1)), scale, camera["gravity_up"])
    torch.testing.assert_close(trajectory.position, position[:, 1:])
    torch.testing.assert_close(trajectory.up, up[:, 1:])
    torch.testing.assert_close(trajectory.velocity, velocity[:, 1:])
    physical_acceleration = thrust * up - model.gravity * camera["gravity_up"][:, None] - model.linear_drag * velocity
    torch.testing.assert_close(trajectory.acceleration, physical_acceleration[:, 1:])

    dt = 1e-5
    nearby = ekf.rollout(obs[:, 1], position.new_tensor([[-dt, dt]]), scale, camera["gravity_up"])
    torch.testing.assert_close((nearby.up[:, 1] - nearby.up[:, 0]) / (2 * dt),
                               torch.linalg.cross(omega[:, 1], up[:, 1]), atol=1e-7, rtol=1e-7)


def test_box_jacobian_matches_differentiable_projection():
    ekf = PoseEKF3D((1280, 720), .24, use_boxes=True).double()
    mean = torch.tensor([[.12, -.18, .2, .1, .04, -.03, .02, .02, .01, .3, -.2, .1, .1]], dtype=torch.float64)
    camera = camera_for(mean)
    axes = torch.stack(tangent_basis(camera["gravity_up"]), dim=1)
    scale = mean.new_tensor([[12.]])
    box, computed = ekf._box_linearization(mean, scale, camera, axes)

    def project(state):
        return box_coordinates(ekf._project(ekf._geometry(state)[:, None], scale, camera, axes))[0, 0]

    torch.testing.assert_close(box[0], project(mean))
    expected = torch.autograd.functional.jacobian(project, mean)[:, 0]
    torch.testing.assert_close(computed[0], expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("use_boxes", [False, True])
def test_3d_rollout_projects_body_and_learns_from_screen_loss(use_boxes):
    ekf = PoseEKF3D((1280, 720), .24, use_boxes=use_boxes).double()
    pose = torch.tensor([.1, -.05, 0., .02, .01, .02, .01, 0., -.005, .03, .07, .02, .01],
                        dtype=torch.float64).repeat(1, 3, 1)
    pose[:, 1] *= .95
    past = pose.new_tensor([[[.57, .50, .043, .045], [.58, .52, .046, .047], [.59, .53, .05, .05]]])
    times = pose.new_tensor([[-.093, -.041, 0.]])
    future = pose.new_tensor([[.05, .4, .8]])
    scale = pose.new_tensor([[14.]])
    camera = camera_for(pose)
    mean, covariance = ekf.filter(past, times, pose, scale, camera)
    assert (torch.linalg.eigvalsh(covariance) > 0).all()
    forecast = ekf(past, times, future, pose, scale, camera)
    trajectory = ekf.rollout(mean, future, scale, camera["gravity_up"])
    assert (trajectory.position[..., 2] > 0).all()
    torch.testing.assert_close(forecast * ekf.pixel_scale,
                               project_body(trajectory.position, trajectory.up, camera, ekf.half_height))
    changed = past.clone()
    changed[0, 1, 0] += .006
    changed_forecast = ekf(changed, times, future, pose, scale, camera)
    if use_boxes:
        assert not torch.allclose(changed_forecast, forecast)
    else:
        torch.testing.assert_close(changed_forecast, forecast)
    forecast.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ekf.parameters())
