import pytest
import torch

from experiments.fred_relative_pose.pose_kalman import PoseBoxKalman, TIME_UNIT_S, box_coordinates


@pytest.mark.parametrize("use_pose", [False, True])
def test_filter_matches_direct_measurement_update(use_pose):
    model = PoseBoxKalman(use_pose=use_pose).double()
    boxes = torch.tensor([[[.43, .54, .03, .042], [.435, .539, .031, .04], [.44, .54, .032, .04]]],
                         dtype=torch.float64)
    times = torch.tensor([[-.086, -.041, 0.]], dtype=boxes.dtype)
    xyz = box_coordinates(boxes)
    pose = torch.stack((xyz + .001, torch.full_like(xyz, .01), torch.full_like(xyz, -.002)), dim=-1)
    expected_mean, expected_cov = [], []
    variance = (2 * model.log_measurement_std).exp()
    for axis in range(4):
        if use_pose:
            h = torch.eye(3, dtype=boxes.dtype)[[0, 0, 1, 2]]
            r = variance[axis].diag()
            corr = .95 * model.position_correlation[axis].tanh()
            r[0, 1] = r[1, 0] = corr * (variance[axis, 0] * variance[axis, 1]).sqrt()
        else:
            h = torch.eye(3, dtype=boxes.dtype)[:1]
            r = variance[axis].diag()
        mean = torch.stack((xyz[0, 0, axis], boxes.new_zeros(()), boxes.new_zeros(())))
        cov = (2 * model.log_initial_std[axis]).exp().diag()
        for i in range(boxes.shape[1]):
            if i:
                dt = (times[0, i] - times[0, i - 1]) / TIME_UNIT_S
                generator = boxes.new_tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
                generator[2, 2] = -model.log_acceleration_decay[axis].exp()
                f = torch.matrix_exp(generator * dt)
                mean = f @ mean
                cov = f @ cov @ f.T + (2 * model.log_process_std[axis]).exp().diag() * dt
            box = xyz[0, i, axis:axis + 1]
            z = torch.cat((box, pose[0, i, axis])) if use_pose else box
            gain = torch.linalg.solve(h @ cov @ h.T + r, h @ cov).T
            mean = mean + gain @ (z - h @ mean)
            cov = cov - gain @ h @ cov
        expected_mean.append(mean)
        expected_cov.append(cov)
    filtered_mean, filtered_cov = model.filter(boxes, times, pose if use_pose else None)
    torch.testing.assert_close(filtered_mean[0], torch.stack(expected_mean))
    torch.testing.assert_close(filtered_cov[0], torch.stack(expected_cov))
    assert (torch.linalg.eigvalsh(filtered_cov) > 0).all()


def test_rollout_uses_pose_measurements_across_the_history():
    model = PoseBoxKalman(use_pose=True)
    past = torch.tensor([.42, .43, .04, .05]).repeat(1, 12, 1)
    times = torch.arange(-11, 1).float()[None] / 30
    queries = torch.tensor([[.0333, .4, .8]])
    coords = box_coordinates(past)
    pose = torch.stack((coords, torch.zeros_like(coords), torch.zeros_like(coords)), dim=-1)
    forecast = model(past, times, queries, pose)
    torch.testing.assert_close(forecast, past[:, :3])
    pose[:, 6, 0, 1:] = .02
    changed = model(past, times, queries, pose)
    assert (changed[..., 0] - forecast[..., 0] > 1e-5).all()
    torch.testing.assert_close(changed[..., 1:], forecast[..., 1:])
    changed.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
