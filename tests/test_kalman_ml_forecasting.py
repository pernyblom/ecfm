from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import (
    TrackKalmanForecastDataset,
    _augment_boxes,
    _load_image,
    _normalize_event_count_channels,
)
from experiments.kalman_ml_forecasting.models.kalman_filter import (
    ConfiguredBoxKalmanFilter,
    kalman_cv_forecast,
    kalman_config_from_dict,
    kalman_forecast,
    kalman_cv_forecast_tensor_params,
    kalman_filter_history,
    kalman_std_tensors_from_config,
)
from experiments.kalman_ml_forecasting.models.coupled_kalman_filter import (
    CoupledBoxKalmanFilter,
    constant_velocity_generator,
)
from experiments.kalman_ml_forecasting.models.kalman_residual import (
    KalmanResidualForecaster,
    box_sequence_to_state,
    constant_velocity_forecast,
)
from experiments.kalman_ml_forecasting.coupled_report_to_config import kalman_config_from_report
from experiments.kalman_ml_forecasting.optimize_kalman import (
    _objective_score,
    _parse_objective_weights,
)
from experiments.kalman_ml_forecasting.train import _box_augmentation_for_split
from experiments.kalman_ml_forecasting.utils.config import (
    resolve_representation_image_sizes,
    resolve_spatial_cutout_config,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_coupled_kalman_starts_at_constant_velocity_transition() -> None:
    model = CoupledBoxKalmanFilter()
    dt = torch.tensor([0.1, 0.25])

    transition = model.transition(dt)
    expected = torch.eye(8).unsqueeze(0).repeat(2, 1, 1)
    expected[:, :4, 4:] = torch.eye(4).unsqueeze(0) * dt[:, None, None]

    assert torch.allclose(model.dynamics_generator, constant_velocity_generator())
    assert torch.allclose(transition, expected, atol=1.0e-6)


def test_coupled_kalman_backpropagates_into_transition_and_optional_noise() -> None:
    past = torch.tensor(
        [[[0.40, 0.40, 0.10, 0.10], [0.42, 0.41, 0.11, 0.10], [0.45, 0.43, 0.12, 0.11]]]
    )
    past_times = torch.tensor([[0.0, 0.1, 0.2]])
    future_times = torch.tensor([[0.3, 0.4]])

    fixed_noise = CoupledBoxKalmanFilter(optimize_noise=False)
    fixed_noise(past, past_times, future_times).sum().backward()
    assert fixed_noise.dynamics_generator.grad is not None
    assert all(parameter.grad is None for parameter in fixed_noise.log_std.values())

    learned_noise = CoupledBoxKalmanFilter(optimize_noise=True)
    learned_noise(past, past_times, future_times).sum().backward()
    assert learned_noise.dynamics_generator.grad is not None
    assert all(parameter.grad is not None for parameter in learned_noise.log_std.values())


def test_coupled_config_requires_a_finite_8x8_generator() -> None:
    try:
        kalman_config_from_dict({"motion_model": "coupled"})
    except ValueError as exc:
        assert "requires kalman.dynamics_generator" in str(exc)
    else:
        raise AssertionError("Expected a missing coupled generator to fail.")

    try:
        kalman_config_from_dict(
            {"motion_model": "coupled", "dynamics_generator": [[0.0] * 8] * 7}
        )
    except ValueError as exc:
        assert "shape [8, 8]" in str(exc)
    else:
        raise AssertionError("Expected a malformed coupled generator to fail.")


def test_coupled_cv_generator_matches_constant_velocity_filter() -> None:
    past = torch.tensor(
        [[[0.3, 0.4, 0.1, 0.2], [0.32, 0.41, 0.11, 0.19], [0.35, 0.43, 0.12, 0.18]]]
    )
    past_t = torch.tensor([[0.0, 0.1, 0.25]])
    future_t = torch.tensor([[0.35, 0.55]])
    coupled_cfg = {
        "motion_model": "coupled",
        "dynamics_generator": constant_velocity_generator().tolist(),
    }

    expected = kalman_forecast(past, past_t, future_t)
    actual = kalman_forecast(past, past_t, future_t, coupled_cfg)

    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_coupled_optimizer_snapshot_reproduces_configured_runtime() -> None:
    generator = constant_velocity_generator()
    generator[0, 1] = 0.15
    generator[5, 4] = -0.2
    model = CoupledBoxKalmanFilter(
        {"dynamics_generator": generator.tolist()}, optimize_dynamics=False
    )
    past = torch.tensor(
        [[[0.3, 0.4, 0.1, 0.2], [0.32, 0.41, 0.11, 0.19], [0.35, 0.43, 0.12, 0.18]]]
    )
    past_t = torch.tensor([[0.0, 0.1, 0.25]])
    future_t = torch.tensor([[0.35, 0.55]])
    runtime = ConfiguredBoxKalmanFilter(
        model.snapshot(reference_dt=0.1)["kalman_config"]
    )

    assert torch.allclose(
        runtime(past, past_t, future_t), model(past, past_t, future_t), atol=1.0e-6
    )


def test_coupled_report_converts_to_runtime_config() -> None:
    generator = constant_velocity_generator().tolist()
    noise = {
        "initial_pos_std": 0.01,
        "initial_size_std": 0.02,
        "initial_vel_std": 0.03,
        "process_pos_std": 0.04,
        "process_size_std": 0.05,
        "process_vel_std": 0.06,
        "process_size_vel_std": 0.07,
        "measurement_pos_std": 0.08,
        "measurement_size_std": 0.09,
    }
    report = {
        "best": {
            "model": {
                "state_layout": ["cx", "cy", "w", "h", "vx", "vy", "vw", "vh"],
                "parameterization": "F(dt) = matrix_exp(dynamics_generator * dt)",
                "dynamics_generator": generator,
                "noise": noise,
            }
        }
    }

    config = kalman_config_from_report(report)

    assert config["motion_model"] == "coupled"
    assert config["dynamics_generator"] == generator
    assert config["measurement_size_std"] == 0.09


def test_box_augmentation_changes_offset_and_size_within_configured_bounds() -> None:
    boxes = np.asarray([[0.5, 0.5, 0.2, 0.1], [0.4, 0.6, 0.1, 0.2]], dtype=np.float32)
    original = boxes.copy()
    cfg = {
        "enabled": True,
        "center_offset_fraction": [0.2, 0.1],
        "size_scale_range": [0.8, 1.3],
        "clip_to_frame": False,
    }

    augmented = _augment_boxes(boxes, cfg, rng=np.random.default_rng(7))

    np.testing.assert_array_equal(boxes, original)
    assert not np.array_equal(augmented, original)
    assert np.all(np.abs(augmented[:, 0] - boxes[:, 0]) <= boxes[:, 2] * 0.2 + 1e-7)
    assert np.all(np.abs(augmented[:, 1] - boxes[:, 1]) <= boxes[:, 3] * 0.1 + 1e-7)
    scales = augmented[:, 2:4] / boxes[:, 2:4]
    assert np.all(scales >= 0.8 - 1e-6)
    assert np.all(scales <= 1.3 + 1e-6)


def test_box_augmentation_split_selection_defaults_to_train() -> None:
    data_cfg = {"box_augmentation": {"enabled": True, "live": True}}
    assert _box_augmentation_for_split(data_cfg, "train")["enabled"] is True
    assert _box_augmentation_for_split(data_cfg, "val") == {}

    data_cfg["box_augmentation"]["splits"] = ["train_eval", "test"]
    assert _box_augmentation_for_split(data_cfg, "train") == {}
    assert _box_augmentation_for_split(data_cfg, "train_eval")["live"] is True


def test_live_box_augmentation_is_redrawn_without_mutating_cached_sample() -> None:
    dataset = TrackKalmanForecastDataset.__new__(TrackKalmanForecastDataset)
    dataset.representations = []
    dataset.image_sizes = {}
    dataset.source_image_sizes = {}
    dataset.frame_size = (100.0, 100.0)
    dataset.spatial_cutout = {}
    dataset.box_augmentation = {
        "enabled": True,
        "live": True,
        "center_offset_fraction": [0.25, 0.25],
        "size_scale_range": [0.75, 1.25],
    }
    cached_past = np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    cached_future = np.asarray([[0.6, 0.5, 0.2, 0.2]], dtype=np.float32)
    dataset.samples = [
        {
            "input_paths": {},
            "past_boxes": cached_past.copy(),
            "future_boxes": cached_future.copy(),
            "past_times_s": np.asarray([0.0], dtype=np.float32),
            "future_times_s": np.asarray([1.0], dtype=np.float32),
            "folder": "",
            "anchor_stem": "frame_0",
            "anchor_time_s": 0.0,
            "track_id": 1,
        }
    ]

    first = dataset[0]
    second = dataset[0]

    assert not torch.equal(first.past_boxes, second.past_boxes)
    np.testing.assert_array_equal(dataset.samples[0]["past_boxes"], cached_past)
    np.testing.assert_array_equal(dataset.samples[0]["future_boxes"], cached_future)


def test_cache_box_augmentation_is_applied_reproducibly() -> None:
    cfg = {
        "enabled": True,
        "cache": True,
        "center_offset_fraction": 0.2,
        "size_scale_range": [0.8, 1.2],
    }
    sample = {
        "past_boxes": np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
        "future_boxes": np.asarray([[0.6, 0.5, 0.2, 0.2]], dtype=np.float32),
    }
    datasets = []
    for _ in range(2):
        dataset = TrackKalmanForecastDataset.__new__(TrackKalmanForecastDataset)
        dataset.seed = 19
        dataset.box_augmentation = dict(cfg)
        dataset.samples = [{key: value.copy() for key, value in sample.items()}]
        dataset._apply_cache_box_augmentation()
        datasets.append(dataset)

    assert not np.array_equal(datasets[0].samples[0]["past_boxes"], sample["past_boxes"])
    np.testing.assert_array_equal(
        datasets[0].samples[0]["past_boxes"], datasets[1].samples[0]["past_boxes"]
    )
    np.testing.assert_array_equal(
        datasets[0].samples[0]["future_boxes"], datasets[1].samples[0]["future_boxes"]
    )


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


def test_kalman_constant_acceleration_forecast_uses_12d_state() -> None:
    times = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    x = 0.1 + 0.02 * times + 0.01 * times.square()
    past = torch.stack(
        [x, torch.full_like(x, 0.5), torch.full_like(x, 0.1), torch.full_like(x, 0.1)], dim=-1
    )
    params = {
        "motion_model": "constant_acceleration", "measurement_pos_std": 1.0e-5,
        "initial_vel_std": 1.0, "initial_accel_std": 1.0,
        "process_pos_std": 1.0e-6, "process_vel_std": 1.0e-6, "process_accel_std": 1.0e-6,
    }
    state, cov = kalman_filter_history(past, times, params)
    pred = kalman_cv_forecast(past, times, torch.tensor([[4.0, 5.0]]), params)

    assert state.shape == (1, 12)
    assert cov.shape == (1, 12, 12)
    assert state[0, 8] > 0.0
    assert pred.shape == (1, 2, 4)


def test_kalman_constant_acceleration_tensor_parameters_receive_gradients() -> None:
    past = torch.tensor([[[0.1, 0.5, 0.1, 0.1], [0.13, 0.5, 0.1, 0.1], [0.18, 0.5, 0.1, 0.1]]])
    past_t = torch.tensor([[0.0, 1.0, 2.0]])
    params = {
        key: value.detach().clone().requires_grad_(True)
        for key, value in kalman_std_tensors_from_config(
            {"motion_model": "constant_acceleration"}, device=past.device, dtype=past.dtype
        ).items()
    }
    pred = kalman_cv_forecast_tensor_params(
        past, past_t, torch.tensor([[3.0]]), params, motion_model="constant_acceleration"
    )
    pred[..., 0].sum().backward()

    assert params["initial_accel_std"].grad is not None
    assert params["process_accel_std"].grad is not None


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


def test_sample_decorrelation_score_can_penalize_mean_acceleration() -> None:
    x = np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float64)
    y = np.asarray([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    base = TrackKalmanForecastDataset._score_decorrelation_stats(
        float(x.shape[0]),
        x.sum(axis=0),
        y.sum(axis=0),
        x.T @ x,
        x.T @ y,
        y.T @ y,
        ridge_lambda=1.0e-3,
        corr_weight=0.0,
        r2_weight=0.0,
        mean_accel_weight=0.0,
    )
    penalized = TrackKalmanForecastDataset._score_decorrelation_stats(
        float(x.shape[0]),
        x.sum(axis=0),
        y.sum(axis=0),
        x.T @ x,
        x.T @ y,
        y.T @ y,
        ridge_lambda=1.0e-3,
        corr_weight=0.0,
        r2_weight=0.0,
        mean_accel_weight=0.5,
    )

    assert base["mean_accel_norm"] == 2.0
    assert penalized["score"] == 1.0


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


def test_kalman_residual_forecaster_can_roll_out_coupled_transition() -> None:
    generator = constant_velocity_generator()
    generator[0, 0] = 0.2
    kalman_cfg = {
        "motion_model": "coupled",
        "dynamics_generator": generator.tolist(),
    }
    model = KalmanResidualForecaster(
        representations=[],
        image_sizes={},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        history_feature_mode="raw",
        state_layers=0,
        residual_layers=0,
        rollout_transition="configured_kalman",
        kalman_params=kalman_cfg,
    )
    for parameter in model.residual_head.parameters():
        torch.nn.init.zeros_(parameter)
    past = torch.tensor([[[0.2, 0.3, 0.1, 0.1], [0.25, 0.31, 0.1, 0.1]]])
    past_t = torch.tensor([[0.0, 0.5]])
    future_t = torch.tensor([[0.75]])
    initial_state = box_sequence_to_state(past, past_t)
    runtime = ConfiguredBoxKalmanFilter(kalman_cfg)
    expected_state = runtime.transition_state(initial_state, torch.tensor([0.25]))

    out = model({}, past, past_t, future_t, return_debug=True)

    assert torch.allclose(out["boxes"][:, 0], expected_state[:, :4].clamp(0.0, 1.0))
    assert out["kalman_boxes"].shape == (1, 1, 4)
    assert not any(
        name.startswith("kalman_filter") for name, _ in model.named_parameters()
    )


def test_kalman_residual_forecaster_encodes_image_sequence_with_gru() -> None:
    model = KalmanResidualForecaster(
        representations=["cstr3"],
        image_sizes={"cstr3": (8, 8)},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_hidden_dim=16,
        state_hidden_dim=8,
        residual_hidden_dim=16,
        representation_sequences={"cstr3": {"length": 3, "stride": 1}},
        temporal_aggregation_cfg={"type": "gru", "hidden_dim": 12},
    )
    inputs = {"cstr3": torch.zeros((2, 3, 3, 8, 8))}
    past = torch.tensor(
        [
            [[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]],
            [[0.4, 0.5, 0.2, 0.2], [0.5, 0.6, 0.2, 0.2]],
        ],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    future_t = torch.tensor([[2.0], [2.0]])

    out = model(inputs, past, past_t, future_t)

    assert out.shape == (2, 1, 4)
    assert model.image_fusion[0].in_features == 12


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


def test_kalman_residual_forecaster_can_use_center_full_filter_features_without_fusion_mlp() -> None:
    model = KalmanResidualForecaster(
        representations=[],
        image_sizes={},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_layers=0,
        history_feature_mode="none",
        residual_layers=0,
        use_filter_state_features=True,
        filter_state_feature_mode="center_full",
    )
    past = torch.tensor(
        [[[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0]])
    future_t = torch.tensor([[2.0]])

    out = model({}, past, past_t, future_t, return_debug=True)

    assert isinstance(model.image_fusion, torch.nn.Identity)
    assert model.residual_head[0].in_features == 13
    assert out["boxes"].shape == (1, 1, 4)
    assert out["filter_state"].shape == (1, 8)


def test_kalman_residual_forecaster_can_use_center_position_filter_features() -> None:
    model = KalmanResidualForecaster(
        representations=[],
        image_sizes={},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        fusion_hidden_dim=16,
        state_hidden_dim=8,
        residual_hidden_dim=16,
        use_filter_state_features=True,
        filter_state_feature_mode="center_position",
        filter_covariance_features="diag",
    )
    past = torch.tensor(
        [[[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0]])
    future_t = torch.tensor([[2.0]])

    out = model({}, past, past_t, future_t, return_debug=True)

    assert model.image_fusion[0].in_features == 4
    assert out["boxes"].shape == (1, 1, 4)
    assert out["filter_state"].shape == (1, 8)
    assert out["filter_cov"].shape == (1, 8, 8)


def test_kalman_residual_forecaster_can_frame_center_filter_state_positions() -> None:
    filter_state = torch.tensor(
        [[0.25, 0.75, 0.2, 0.3, 1.0, -1.0, 0.1, -0.1]],
        dtype=torch.float32,
    )
    expected = {
        "full": torch.tensor([[-0.5, 0.5, 0.2, 0.3, 1.0, -1.0, 0.1, -0.1]]),
        "center_full": torch.tensor([[-0.5, 0.5, 1.0, -1.0]]),
        "center_position": torch.tensor([[-0.5, 0.5]]),
        "center_velocity": torch.tensor([[1.0, -1.0]]),
    }
    for mode, expected_features in expected.items():
        model = KalmanResidualForecaster(
            representations=[],
            image_sizes={},
            backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
            history_steps=2,
            fusion_layers=0,
            history_feature_mode="none",
            use_filter_state_features=True,
            filter_state_feature_mode=mode,
            filter_state_center_position_normalization="frame_centered",
        )

        features = model._image_features({}, filter_state=filter_state)

        assert torch.allclose(features, expected_features, atol=1e-6)


def test_kalman_residual_forecaster_can_use_raw_box_history_without_history_mlp() -> None:
    model = KalmanResidualForecaster(
        representations=[],
        image_sizes={},
        backbone_cfg={"type": "small_cnn", "in_channels": 3, "channels": [4, 8], "out_dim": 16},
        history_steps=2,
        state_layers=0,
        residual_layers=0,
        history_feature_mode="raw",
    )
    past = torch.tensor(
        [[[0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    past_t = torch.tensor([[0.0, 1.0]])
    future_t = torch.tensor([[2.0]])

    out = model({}, past, past_t, future_t, return_debug=True)

    assert isinstance(model.history_encoder, torch.nn.Identity)
    assert model.residual_head[0].in_features == 19
    assert out["boxes"].shape == (1, 1, 4)


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
            history_feature_mode="none",
        )
    except ValueError as exc:
        assert "At least one learned feature source" in str(exc)
    else:
        raise AssertionError("Expected empty representations without filter features to fail.")


def test_kalman_config_resolves_empty_representation_sizes() -> None:
    assert resolve_representation_image_sizes({"representations": []}) == {}


def test_spatial_cutout_supports_per_representation_overrides() -> None:
    data_cfg = {
        "representations": ["cstr2", "xt", "yt", "rgb"],
        "image_sizes": {
            "cstr2": [640, 360],
            "xt": [640, 64],
            "yt": [64, 360],
            "rgb": [640, 360],
        },
        "spatial_cutout": {
            "mode": "fixed_pixels",
            "size_px": [64, 64],
            "fill": 0.25,
            "by_representation": {
                "cstr2": {"size_px": [160, 120]},
                "xt": {"size_px": [192, 999]},
                "yt": {"size_px": [999, 144]},
                "rgb": {"mode": "none"},
            },
        },
    }

    assert resolve_representation_image_sizes(data_cfg) == {
        "cstr2": (160, 120),
        "xt": (192, 64),
        "yt": (64, 144),
        "rgb": (640, 360),
    }
    assert resolve_spatial_cutout_config(data_cfg["spatial_cutout"], "cstr2") == {
        "mode": "fixed_pixels",
        "size_px": [160, 120],
        "fill": 0.25,
    }


def test_spatial_cutout_grid_alias_uses_base_representation_override() -> None:
    cutout = {
        "mode": "fixed_pixels",
        "size_px": [64, 64],
        "by_representation": {"xt_my": {"size_px": [128, 32]}},
    }

    assert resolve_spatial_cutout_config(cutout, "xt_my_10x10")["size_px"] == [128, 32]


def test_event_count_min_max_normalization_runs_after_fixed_cutout(tmp_path: Path) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 17
    image[:, :, 2] = 23
    image[:, :, 1] = 250
    image[1:3, 1:3, 1] = np.asarray([[10, 20], [30, 40]], dtype=np.uint8)
    path = tmp_path / "sample_cstr3.png"
    Image.fromarray(image).save(path)

    loaded = _load_image(
        path,
        (2, 2),
        source_size=(4, 4),
        rep="cstr3",
        frame_size=(4, 4),
        anchor_box=np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        spatial_cutout={
            "mode": "fixed_pixels",
            "size_px": [2, 2],
            "event_count_normalization": "min_max",
        },
    )

    torch.testing.assert_close(
        loaded[1],
        torch.tensor([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(loaded[0], torch.full((2, 2), 17.0 / 255.0))
    torch.testing.assert_close(loaded[2], torch.full((2, 2), 23.0 / 255.0))


def test_event_count_normalization_preserves_xt_my_position_channel() -> None:
    arr = np.asarray(
        [
            [[0.0, 0.2, 0.0], [0.05, 0.4, 0.1]],
            [[0.1, 0.6, 0.05], [0.2, 0.8, 0.2]],
        ],
        dtype=np.float32,
    )

    normalized = _normalize_event_count_channels(
        arr,
        rep="xt_my_10x10",
        cfg={"event_count_normalization": "max"},
    )

    np.testing.assert_array_equal(normalized[:, :, 1], arr[:, :, 1])
    np.testing.assert_allclose(normalized[:, :, 0], arr[:, :, 0] / 0.2)
    np.testing.assert_allclose(normalized[:, :, 2], arr[:, :, 2] / 0.2)


def test_event_count_normalization_supports_joint_and_explicit_channels() -> None:
    arr = np.asarray([[[1.0, 4.0, 2.0], [3.0, 8.0, 5.0]]], dtype=np.float32)

    normalized = _normalize_event_count_channels(
        arr,
        rep="custom",
        cfg={
            "event_count_normalization": {
                "mode": "joint_min_max",
                "channels": ["red", "blue"],
            }
        },
    )

    np.testing.assert_allclose(normalized[:, :, 0], [[0.0, 0.5]])
    np.testing.assert_allclose(normalized[:, :, 2], [[0.25, 1.0]])
    np.testing.assert_array_equal(normalized[:, :, 1], arr[:, :, 1])


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


def test_track_kalman_dataset_builds_causal_representation_sequence(tmp_path: Path) -> None:
    labels = tmp_path / "labels" / "seq" / "Event_YOLO"
    images = tmp_path / "images" / "seq"
    for t in [0, 1000000, 2000000, 3000000]:
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
        representation_sequences={"cstr3": {"length": 2, "stride": 1}},
    )

    sample = dataset[0]

    assert sample.inputs["cstr3"].shape == (2, 3, 8, 8)
    assert [Path(path).stem for path in sample.input_paths["cstr3"]] == [
        "Video_0_frame_0_cstr3",
        "Video_0_frame_1000000_cstr3",
    ]


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
