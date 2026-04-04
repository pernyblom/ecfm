import torch

from experiments.pose_dynamics.models.pose_dynamics import PoseDynamicsProjector


def test_model_predictions_depend_on_past_centers():
    torch.manual_seed(0)

    model = PoseDynamicsProjector(
        reps=["xt_my", "yt_mx"],
        cnn_channels=[8, 16],
        feature_dim=16,
        hidden_dim=32,
        history_steps=2,
        future_steps=3,
        min_depth=0.1,
    )
    model.eval()

    inputs = {
        "xt_my": torch.rand(2, 3, 32, 32),
        "yt_mx": torch.rand(2, 3, 32, 32),
    }
    intrinsics = torch.tensor([[1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 0.5, 0.5]], dtype=torch.float32)
    camera_pose = torch.zeros(2, 6, dtype=torch.float32)
    dt = torch.full((2, 3), 0.1, dtype=torch.float32)

    past_centers_a = torch.tensor(
        [
            [[0.10, 0.20], [0.15, 0.25], [0.20, 0.30]],
            [[0.30, 0.40], [0.35, 0.45], [0.40, 0.50]],
        ],
        dtype=torch.float32,
    )
    past_centers_b = torch.tensor(
        [
            [[0.60, 0.10], [0.55, 0.15], [0.50, 0.20]],
            [[0.20, 0.70], [0.25, 0.65], [0.30, 0.60]],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        pred_a = model(inputs, past_centers_a, intrinsics, camera_pose, dt)
        pred_b = model(inputs, past_centers_b, intrinsics, camera_pose, dt)

    assert pred_a["pred_centers"].shape == (2, 3, 2)
    assert not torch.allclose(pred_a["pred_centers"], pred_b["pred_centers"])
