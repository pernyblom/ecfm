import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.coupled_report_to_config import kalman_config_from_report
from experiments.kalman_ml_forecasting.models.coupled_kalman_filter import CoupledBoxKalmanFilter
from experiments.kalman_ml_forecasting.models.kalman_filter import (
    ConfiguredBoxKalmanFilter, channel_mask, kalman_config_from_dict,
    kalman_filter_history, kalman_forecast, predict_kalman_state,
)
from experiments.kalman_ml_forecasting.models.kalman_residual import KalmanResidualForecaster
from experiments.kalman_ml_forecasting.optimize_coupled_kalman import _tune_indices
from experiments.kalman_ml_forecasting.optimize_kalman_backprop import KalmanStdParameters
from experiments.kalman_ml_forecasting.models.kalman_filter import kalman_cv_forecast_tensor_params


def history(dtype=torch.float64):
    past = torch.tensor([[[.3, .4, .1, .2], [.32, .41, .11, .19], [.35, .43, .12, .18]]], dtype=dtype)
    return past, torch.tensor([[0., .1, .25]], dtype=dtype), torch.tensor([[.35, .55]], dtype=dtype)


def variant(dynamics, structure, **kwargs):
    return CoupledBoxKalmanFilter({
        "initial_covariance_structure": structure,
        "process_covariance_structure": structure,
        "measurement_covariance_structure": "diagonal" if structure == "per_channel" else structure,
    }, dynamics=dynamics, optimize_noise=True, dtype=torch.float64, **kwargs)


@pytest.mark.parametrize("dynamics", ["constant_velocity", "per_channel", "full"])
@pytest.mark.parametrize("structure", ["legacy_diagonal", "diagonal", "per_channel", "full"])
def test_variants_train_export_and_run_as_residual(dynamics, structure):
    model = variant(dynamics, structure)
    past, times, future = history()
    # Every structure starts at exactly the legacy CV filter.
    assert torch.allclose(model(past, times, future), kalman_forecast(past, times, future), atol=1e-10)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss = (model(past, times, future) - .45).square().sum()
    loss.backward()
    for module in model.covariances.values():
        assert torch.isfinite(module.log_diagonal.grad).all()
        if module.raw_off_diagonal.numel():
            assert torch.isfinite(module.raw_off_diagonal.grad).all()
            assert module.raw_off_diagonal.grad.abs().max() > 0
    if dynamics == "per_channel":
        assert torch.count_nonzero(model.dynamics_generator.grad[~channel_mask(8)]) == 0
    optimizer.step()
    model.clamp_(max_dynamics_abs=20)
    config = kalman_config_from_report({"best": {"model": model.snapshot(reference_dt=.1)}})
    # Test actual YAML serialization and float32 buffer import precision.
    config = yaml.safe_load(yaml.safe_dump(config))
    runtime = ConfiguredBoxKalmanFilter(config).double()
    assert torch.allclose(runtime(past, times, future), model(past, times, future), atol=2e-7)
    assert torch.allclose(kalman_forecast(past, times, future, config), model(past, times, future), atol=1e-10)
    residual = KalmanResidualForecaster(
        representations=[], representation_channels={}, use_filter_state_features=True,
        image_sizes={}, backbone_cfg={}, history_steps=3,
        filter_covariance_features="full", initial_state_source="kalman_filter",
        rollout_transition="configured_kalman", kalman_params=config,
    ).double()
    out = residual({}, past, times, future, return_debug=True)
    assert torch.isfinite(out["boxes"]).all()
    assert torch.allclose(out["kalman_boxes"], runtime(past, times, future), atol=1e-10)
    assert not list(residual.kalman_filter.parameters())


def test_per_channel_matches_four_independent_reference_filters():
    model = variant("per_channel", "per_channel")
    with torch.no_grad():
        model.dynamics_generator[4, 4] = -.2
        for module in model.covariances.values():
            module.raw_off_diagonal.fill_(.02)
    past, times, _ = history()
    actual_state, actual_cov = model.filter_history(past, times)
    matrices = model.tensors()
    for channel in range(4):
        indices = torch.tensor([channel, channel + 4])
        state = torch.tensor([past[0, 0, channel], 0.], dtype=past.dtype)
        covariance = matrices["initial_covariance"][indices[:, None], indices[None, :]]
        process = matrices["process_covariance"][indices[:, None], indices[None, :]]
        measurement = matrices["measurement_covariance"][channel, channel]
        generator = model.effective_generator()[indices[:, None], indices[None, :]]
        for index in range(3):
            if index:
                dt = times[0, index] - times[0, index - 1]
                transition = torch.matrix_exp(generator * dt)
                scale = torch.diag(torch.stack([dt, dt.new_ones(())]))
                state = transition @ state
                covariance = transition @ covariance @ transition.T + scale @ process @ scale
            gain = covariance[:, 0] / (covariance[0, 0] + measurement)
            state = state + gain * (past[0, index, channel] - state[0])
            correction = torch.eye(2, dtype=past.dtype) - gain[:, None] @ state.new_tensor([[1., 0.]])
            covariance = correction @ covariance @ correction.T + gain[:, None] @ gain[None, :] * measurement
            state = torch.cat([state[:1].clamp(0, 1), state[1:]])
        assert torch.allclose(actual_state[0, indices], state, atol=1e-10)
        assert torch.allclose(actual_cov[0][indices[:, None], indices[None, :]], covariance, atol=1e-10)
    assert torch.count_nonzero(actual_cov[0][~channel_mask(8)]) == 0


def test_global_covariance_couples_estimates_with_fixed_separate_dynamics():
    past, times, future = history()
    changed = past.clone()
    changed[:, 1:, 2] += .05
    local = variant("constant_velocity", "per_channel")
    assert torch.equal(local(past, times, future)[:, :, 0], local(changed, times, future)[:, :, 0])
    joint = variant("constant_velocity", "full")
    with torch.no_grad():
        module = joint.covariances["initial"]
        module.raw_off_diagonal[(module.rows == 4) & (module.cols == 2)] = .1
    assert not torch.allclose(joint(past, times, future)[:, :, 0], joint(changed, times, future)[:, :, 0])


def test_matrix_process_noise_retains_legacy_time_scaling():
    dtype = torch.float64
    factor = torch.eye(8, dtype=dtype)
    factor[4, 0] = .3
    q = factor @ factor.T
    state = torch.zeros((1, 8), dtype=dtype)
    dt = torch.tensor([.2], dtype=dtype)
    _, covariance = predict_kalman_state(state, torch.zeros((1, 8, 8), dtype=dtype), dt,
                                        torch.ones(8, dtype=dtype), "constant_velocity", process_covariance=q)
    scale = torch.diag(torch.tensor([.2] * 4 + [1.] * 4, dtype=dtype))
    assert torch.allclose(covariance[0], scale @ q @ scale)


def test_independent_learning_controls_and_fixed_covariance():
    model = variant("constant_velocity", "full", optimize_initial=False, optimize_measurement=False)
    assert not model.dynamics_generator.requires_grad
    assert not model.covariances["initial"].log_diagonal.requires_grad
    assert model.covariances["process"].log_diagonal.requires_grad
    assert not model.covariances["measurement"].raw_off_diagonal.requires_grad
    initial = model.covariances["initial"]().detach().clone()
    model(*history()).sum().backward()
    model.clamp_(max_dynamics_abs=20)
    assert torch.equal(initial, model.covariances["initial"]())


def test_dense_covariance_is_positive_over_long_history():
    model = variant("constant_velocity", "full", min_std=1e-8)
    with torch.no_grad():
        for module in model.covariances.values():
            module.raw_off_diagonal.fill_(.02)
    times = torch.arange(120, dtype=torch.float64)[None, :] * .013
    past = torch.tensor([.3, .4, .1, .2], dtype=torch.float64)[None, None, :].expand(1, 120, 4).clone()
    past[:, :, 0] += .01 * times
    state, covariance = model.filter_history(past, times)
    assert torch.isfinite(state).all()
    assert torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-12)
    assert torch.linalg.eigvalsh(covariance).min() > 0


@pytest.mark.parametrize("kind", ["shape", "nan", "asymmetric", "indefinite", "forbidden", "unknown"])
def test_invalid_covariance_configuration_fails(kind):
    matrix = torch.eye(8, dtype=torch.float64)
    cfg = {"initial_covariance_structure": "full"}
    if kind == "shape":
        matrix = torch.eye(4)
    elif kind == "nan":
        matrix[0, 0] = float("nan")
    elif kind == "asymmetric":
        matrix[0, 1] = .1
    elif kind == "indefinite":
        matrix[0, 0] = -1
    elif kind == "forbidden":
        matrix[0, 1] = matrix[1, 0] = .1
        cfg["initial_covariance_structure"] = "per_channel"
    elif kind == "unknown":
        cfg["initial_covariance_struture"] = "full"
    cfg["initial_covariance"] = matrix.tolist()
    with pytest.raises(ValueError):
        kalman_config_from_dict(cfg)


def test_existing_12d_optimizer_runtime_and_export_still_work():
    model = KalmanStdParameters({"motion_model": "constant_acceleration"}, min_std=1e-6, max_std=10, device="cpu")
    past, times, future = history(torch.float32)
    prediction = kalman_cv_forecast_tensor_params(past, times, future, model.tensors(), motion_model=model.motion_model)
    prediction.sum().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad) for value in model.log_std.values())
    config = kalman_config_from_report({"best": {"params": model.as_config()}})
    runtime = ConfiguredBoxKalmanFilter(config)
    assert runtime.state_dim == 12
    assert runtime.filter_history(past, times)[1].shape == (1, 12, 12)
    assert torch.allclose(prediction, runtime(past, times, future))
    with pytest.raises(ValueError, match="8-D only"):
        CoupledBoxKalmanFilter(config)


def test_sequence_tuning_split_preserves_folder_boundaries():
    samples = [{"folder": folder, "track_id": track} for folder in range(4) for track in range(3)]
    train, val = _tune_indices(samples, mode="sequence", val_fraction=.25, seed=123)
    assert {samples[i]["folder"] for i in train}.isdisjoint({samples[i]["folder"] for i in val})


@pytest.mark.parametrize("dynamics", ["constant_velocity", "per_channel", "full", "constant_acceleration"])
def test_cli_train_export_and_residual_config(tmp_path, dynamics):
    labels = tmp_path / "labels"
    for folder in ("a", "b", "c"):
        sequence = labels / folder
        (sequence / "Event_YOLO").mkdir(parents=True)
        for index in range(5):
            (sequence / "Event_YOLO" / f"Video_0_frame_{index * 1000000}.txt").write_text("0 .3 .4 .1 .2\n")
        (sequence / "cleaned_tracks.txt").write_text("\n".join(f"{i},1,{25+i},{30+i},10,20" for i in range(5)))
    config = tmp_path / "base.yaml"
    split = tmp_path / "train.txt"
    split.write_text("a\nb\nc\n")
    config.write_text(yaml.safe_dump({
        "data": {"labels_root": str(labels), "representations": [], "frame_size": [100, 100],
                 "history_ms": 1000, "forecast_ms": 1000, "time_align": "none",
                 "split_files": {"train": str(split)}},
        "kalman": {"motion_model": "constant_acceleration" if dynamics == "constant_acceleration" else "constant_velocity"},
        "model": {"use_filter_state_features": True},
    }))
    report, residual = tmp_path / "report.json", tmp_path / "residual.yaml"
    command = [sys.executable, str(ROOT / "experiments/kalman_ml_forecasting/optimize_coupled_kalman.py"),
               "--config", str(config), "--epochs", "1", "--batch-size", "2", "--tune-split", "sequence",
               "--dynamics", dynamics, "--covariance-structure", "full", "--optimize-noise",
               "--output-json", str(report)]
    if dynamics == "constant_acceleration":
        command = [sys.executable, str(ROOT / "experiments/kalman_ml_forecasting/optimize_kalman_backprop.py"),
                   "--config", str(config), "--epochs", "1", "--batch-size", "2", "--output-json", str(report)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(report.read_text())["history"]
    result = subprocess.run([sys.executable, str(ROOT / "experiments/kalman_ml_forecasting/coupled_report_to_config.py"),
                             str(report), "--base-config", str(config), "--use-for-residual-rollout",
                             "--training-output-dir", str(tmp_path / "residual_training"),
                             "--output", str(residual)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    exported = yaml.safe_load(residual.read_text())
    ca = dynamics == "constant_acceleration"
    assert exported["model"]["rollout_transition"] == ("constant_velocity" if ca else "configured_kalman")
    assert exported["kalman"]["process_covariance_structure"] == ("legacy_diagonal" if ca else "full")
    assert ConfiguredBoxKalmanFilter(exported["kalman"]).state_dim == (12 if ca else 8)
    assert exported["train"]["resume_checkpoint"] is None
    assert exported["train"]["checkpoint_dir"] == str(tmp_path / "residual_training")
