import pytest
import torch

from experiments.pose_dynamics.losses import compute_losses


def test_pose_dynamics_losses_include_size_and_bound_penalties():
    pred = {
        "pred_centers": torch.tensor([[[0.5, 0.5], [0.6, 0.6]]], dtype=torch.float32),
        "pose_delta": torch.zeros(1, 6, dtype=torch.float32),
        "intrinsics_delta": torch.zeros(1, 4, dtype=torch.float32),
        "dynamics_acc": torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]], dtype=torch.float32),
        "dynamics_vel_seq": torch.tensor([[[0.0, 0.0, 0.0], [6.0, 8.0, 0.0]]], dtype=torch.float32),
        "intrinsics_corrected": torch.tensor([[1.0, 1.0, 0.5, 0.5]], dtype=torch.float32),
        "points_cam": torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
    }
    target_centers = pred["pred_centers"].clone()
    target_sizes = torch.tensor([[[0.05, 0.05], [0.90, 0.90]]], dtype=torch.float32)

    total, metrics = compute_losses(
        pred,
        target_centers,
        target_sizes=target_sizes,
        size_weight=1.0,
        size_min=(0.2, 0.2),
        size_max=(0.4, 0.4),
        speed_bound=5.0,
        speed_bound_weight=1.0,
        acc_bound=2.0,
        acc_bound_weight=1.0,
    )

    assert total.item() > 0.0
    assert metrics["center_l1"] == pytest.approx(0.0)
    assert metrics["size_range"] == pytest.approx(0.32499998807907104)
    assert metrics["speed_bound"] == pytest.approx(12.5)
    assert metrics["acc_bound"] == pytest.approx(4.5)
