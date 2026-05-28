from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import TrackKalmanForecastDataset
from experiments.kalman_ml_forecasting.models.kalman_filter import (
    kalman_cv_forecast,
    kalman_cv_forecast_tensor_params,
    kalman_filter_history,
    kalman_std_tensors_from_config,
)
from experiments.kalman_ml_forecasting.models.kalman_residual import (
    KalmanResidualForecaster,
    constant_velocity_forecast,
)
from experiments.kalman_ml_forecasting.optimize_kalman import (
    _objective_score,
    _parse_objective_weights,
)
from experiments.kalman_ml_forecasting.utils.config import resolve_representation_image_sizes


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_constant_velocity_forecast_uses_last_four_linear_fit() -> None:
    past = torch.tensor(
        [[[0.0, 0.1, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1], [0.3, 0.4, 0.1, 0.1]]]
    )
    past_t = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    future_t = torch.tensor([[4.0, 5.0]])

    pred = constant_velocity_forecast(past, past_t, future_t)

    assert torch.allclose(pred[0, :, :2], torch.tensor([[0.4, 0.5], [0.5, 0.6]]), atol=1e-6)


def test_kalman_cv_forecast_shapes() -> None:
    past = torch.tensor(
        [
            [[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]],
            [[0.4, 0.5, 0.2, 0.2], [0.5, 0.6, 0.2, 0.2]],
        ],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    future_t = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])

    pred = kalman_cv_forecast(past, past_t, future_t)

    assert pred.shape == (2, 3, 4)


def test_kalman_measurement_trust_changes_velocity_estimate() -> None:
    past = torch.tensor(
        [[[0.0, 0.5, 0.1, 0.1], [0.1, 0.5, 0.1, 0.1], [0.7, 0.5, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0, 2.0]])
    low_noise_state, _ = kalman_filter_history(
        past,
        past_t,
        {"measurement_pos_std": 1.0e-4, "process_vel_std": 1.0, "initial_vel_std": 2.0},
    )
    high_noise_state, _ = kalman_filter_history(
        past,
        past_t,
        {"measurement_pos_std": 0.5, "process_vel_std": 1.0e-3, "initial_vel_std": 0.01},
    )

    assert low_noise_state[0, 4] > high_noise_state[0, 4]


def test_kalman_tensor_parameters_receive_gradients() -> None:
    past = torch.tensor(
        [[[0.0, 0.5, 0.1, 0.1], [0.1, 0.5, 0.1, 0.1], [0.7, 0.5, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0, 2.0]])
    future_t = torch.tensor([[3.0]])
    params = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in kalman_std_tensors_from_config(None, device=past.device, dtype=past.dtype).items()
    }

    pred = kalman_cv_forecast_tensor_params(past, past_t, future_t, params)
    loss = pred[..., 0].sum()
    loss.backward()

    assert params["measurement_pos_std"].grad is not None


def test_optimize_kalman_objective_weights_support_maximize_and_weighted_score() -> None:
    metrics = {"fde_center_px": 10.0, "ade_center_px": 4.0, "miou": 0.25}

    maximize = _parse_objective_weights(None, "miou", True)
    weighted = _parse_objective_weights("fde_center_px=1,ade_center_px=0.5,miou=-100", "fde_center_px", False)

    assert maximize == {"miou": -1.0}
    assert _objective_score(metrics, maximize) == -0.25
    assert _objective_score(metrics, weighted) == -13.0


def test_kalman_residual_forecaster_forward_shapes() -> None:
    model = KalmanResidualForecaster(
        representations=["cstr3", "xt_my"],
        image_sizes={"cstr3": (8, 8), "xt_my": (8, 8)},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_hidden_dim=16,
        state_hidden_dim=8,
        residual_hidden_dim=16,
    )
    inputs = {
        "cstr3": torch.zeros((2, 3, 8, 8)),
        "xt_my": torch.zeros((2, 3, 8, 8)),
    }
    past = torch.tensor(
        [
            [[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]],
            [[0.4, 0.5, 0.2, 0.2], [0.5, 0.6, 0.2, 0.2]],
        ],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    future_t = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])

    out = model(inputs, past, past_t, future_t, return_debug=True)

    assert out["boxes"].shape == (2, 3, 4)
    assert out["residual_accel"].shape == (2, 3, 4)
    assert out["cv_boxes"].shape == (2, 3, 4)


def test_kalman_residual_forecaster_can_fuse_single_rep_with_filter_state() -> None:
    model = KalmanResidualForecaster(
        representations=["cstr3"],
        image_sizes={"cstr3": (8, 8)},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_hidden_dim=16,
        state_hidden_dim=8,
        residual_hidden_dim=16,
        use_filter_state_features=True,
        kalman_params={"measurement_pos_std": 0.01},
    )
    inputs = {"cstr3": torch.zeros((2, 3, 8, 8))}
    past = torch.tensor(
        [
            [[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]],
            [[0.4, 0.5, 0.2, 0.2], [0.5, 0.6, 0.2, 0.2]],
        ],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    future_t = torch.tensor([[2.0], [2.0]])

    out = model(inputs, past, past_t, future_t, return_debug=True)

    assert model.image_fusion[0].in_features == 24
    assert out["boxes"].shape == (2, 1, 4)
    assert out["filter_state"].shape == (2, 8)


def test_kalman_residual_forecaster_can_use_kalman_initial_state_and_covariance_features() -> None:
    model = KalmanResidualForecaster(
        representations=["cstr3"],
        image_sizes={"cstr3": (8, 8)},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_hidden_dim=16,
        state_hidden_dim=8,
        residual_hidden_dim=16,
        initial_state_source="kalman_filter",
        filter_covariance_features="diag",
        kalman_params={"measurement_pos_std": 0.01},
    )
    inputs = {"cstr3": torch.zeros((1, 3, 8, 8))}
    past = torch.tensor(
        [[[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0]])
    future_t = torch.tensor([[2.0]])

    out = model(inputs, past, past_t, future_t, return_debug=True)

    assert model.image_fusion[0].in_features == 24
    assert out["boxes"].shape == (1, 1, 4)
    assert out["filter_state"].shape == (1, 8)
    assert out["filter_cov"].shape == (1, 8, 8)


def test_kalman_residual_forecaster_allows_filter_only_fusion() -> None:
    model = KalmanResidualForecaster(
        representations=[],
        image_sizes={},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_hidden_dim=16,
        state_hidden_dim=8,
        residual_hidden_dim=16,
        use_filter_state_features=True,
        initial_state_source="kalman_filter",
    )
    past = torch.tensor(
        [[[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0]])
    future_t = torch.tensor([[2.0]])

    out = model({}, past, past_t, future_t, return_debug=True)

    assert model.image_fusion[0].in_features == 8
    assert out["boxes"].shape == (1, 1, 4)
    assert out["filter_state"].shape == (1, 8)


def test_kalman_residual_forecaster_can_use_center_velocity_filter_features() -> None:
    model = KalmanResidualForecaster(
        representations=[],
        image_sizes={},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_hidden_dim=16,
        state_hidden_dim=8,
        residual_hidden_dim=16,
        use_filter_state_features=True,
        filter_state_feature_mode="center_velocity",
    )
    past = torch.tensor(
        [[[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0]])
    future_t = torch.tensor([[2.0]])

    out = model({}, past, past_t, future_t, return_debug=True)

    assert model.image_fusion[0].in_features == 2
    assert out["boxes"].shape == (1, 1, 4)
    assert out["filter_state"].shape == (1, 8)


def test_kalman_residual_forecaster_covariance_uses_selected_velocity_state() -> None:
    model = KalmanResidualForecaster(
        representations=[],
        image_sizes={},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_hidden_dim=16,
        state_hidden_dim=8,
        residual_hidden_dim=16,
        filter_state_feature_mode="velocities",
        filter_covariance_features="full",
    )
    past = torch.tensor(
        [[[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0]])
    future_t = torch.tensor([[2.0]])

    out = model({}, past, past_t, future_t, return_debug=True)

    assert model.image_fusion[0].in_features == 16
    assert out["boxes"].shape == (1, 1, 4)
    assert out["filter_cov"].shape == (1, 8, 8)


def test_kalman_residual_forecaster_rejects_empty_reps_without_filter_features() -> None:
    try:
        KalmanResidualForecaster(
            representations=[],
            image_sizes={},
            backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
            history_steps=2,
        )
    except ValueError as exc:
        assert "At least one representation" in str(exc)
    else:
        raise AssertionError("Expected empty representations without filter features to fail.")


def test_kalman_config_resolves_empty_representation_sizes() -> None:
    assert resolve_representation_image_sizes({"representations": []}) == {}


def test_track_kalman_dataset_builds_anchor_sample(tmp_path: Path) -> None:
    labels = tmp_path / "labels" / "seq" / "Event_YOLO"
    images = tmp_path / "images" / "seq"
    for idx, t in enumerate([0, 1000000, 2000000, 3000000], start=1):
        stem = f"Video_0_frame_{t}"
        _write_text(labels / f"{stem}.txt", "0 0.5 0.5 0.1 0.1\n")
        _write_image(images / f"{stem}_cstr3.png")
    _write_text(
        tmp_path / "labels" / "seq" / "cleaned_tracks.txt",
        "\n".join(
            [
                "0.0,1,10,20,4,6",
                "1.0,1,12,22,4,6",
                "2.0,1,14,24,4,6",
                "3.0,1,16,26,4,6",
            ]
        ),
    )

    dataset = TrackKalmanForecastDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        frame_size=(100, 100),
        representations=["cstr3"],
        image_sizes={"cstr3": (8, 8)},
        history_ms=1000.0,
        forecast_ms=1000.0,
        folders=["seq"],
        label_time_unit=1e-6,
        track_time_unit=1.0,
        time_align="none",
        verify_render_manifest=False,
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.inputs["cstr3"].shape == (3, 8, 8)
    assert sample.past_boxes.shape == (2, 4)
    assert sample.future_boxes.shape == (1, 4)
    assert sample.frame_key == "seq/Video_0_frame_1000000"


def test_track_kalman_dataset_uses_dataset_event_frames(tmp_path: Path) -> None:
    labels = tmp_path / "labels" / "seq" / "Event_YOLO"
    frames = tmp_path / "labels" / "seq" / "Event" / "Frames"
    for t in [0, 1000000, 2000000, 3000000]:
        stem = f"Video_0_frame_{t}"
        _write_text(labels / f"{stem}.txt", "0 0.5 0.5 0.1 0.1\n")
        _write_image(frames / f"{stem}.png")
    _write_text(
        tmp_path / "labels" / "seq" / "cleaned_tracks.txt",
        "\n".join(
            [
                "0.0,1,10,20,4,6",
                "1.0,1,12,22,4,6",
                "2.0,1,14,24,4,6",
                "3.0,1,16,26,4,6",
            ]
        ),
    )

    dataset = TrackKalmanForecastDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        frame_size=(100, 100),
        representations=["event_frames"],
        image_sizes={"event_frames": (8, 8)},
        history_ms=1000.0,
        forecast_ms=1000.0,
        folders=["seq"],
        label_time_unit=1e-6,
        track_time_unit=1.0,
        time_align="none",
        verify_render_manifest=True,
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.inputs["event_frames"].shape == (3, 8, 8)
    assert Path(sample.input_paths["event_frames"]).parts[-3:] == ("Event", "Frames", "Video_0_frame_1000000.png")


def test_track_kalman_dataset_allows_empty_representations(tmp_path: Path) -> None:
    labels = tmp_path / "labels" / "seq" / "Event_YOLO"
    for t in [0, 1000000, 2000000, 3000000]:
        _write_text(labels / f"Video_0_frame_{t}.txt", "0 0.5 0.5 0.1 0.1\n")
    _write_text(
        tmp_path / "labels" / "seq" / "cleaned_tracks.txt",
        "\n".join(
            [
                "0.0,1,10,20,4,6",
                "1.0,1,12,22,4,6",
                "2.0,1,14,24,4,6",
                "3.0,1,16,26,4,6",
            ]
        ),
    )

    dataset = TrackKalmanForecastDataset(
        images_root=tmp_path / "images",
        labels_root=tmp_path / "labels",
        frame_size=(100, 100),
        representations=[],
        image_sizes={},
        history_ms=1000.0,
        forecast_ms=1000.0,
        folders=["seq"],
        label_time_unit=1e-6,
        track_time_unit=1.0,
        time_align="none",
        verify_render_manifest=True,
    )

    assert len(dataset) == 2
    assert dataset[0].inputs == {}
