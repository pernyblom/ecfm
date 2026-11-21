import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from experiments.fred_relative_pose.coordinate_tracks import coordinate_tracks
from experiments.fred_relative_pose.data import PoseTracklets, _ForecastTracklets
from experiments.fred_relative_pose.eval import check_test_split
from experiments.fred_relative_pose.flight import FlightState, rollout_flight
from experiments.fred_relative_pose.model import RelativePoseModel, project_body, project_points, tangent_basis
from experiments.fred_relative_pose.train import DEFAULT_CONFIG, history_reprojection_loss, reprojection_loss
from experiments.kalman_ml_forecasting.utils.config import load_config


def test_final_test_split_rejects_training_and_validation_sequences():
    checkpoint = {"splits": {"train": ["1"], "val": ["2"]},
                  "scenes": {"by_folder": {"1": "scene", "2": "scene", "3": "scene"}}}
    check_test_split(checkpoint, ["3"])
    for folder in ("1", "2"):
        with pytest.raises(ValueError, match="training or checkpoint selection"):
            check_test_split(checkpoint, [folder, "3"])


def test_projected_body_bounds_match_sampled_ellipsoid():
    dtype = torch.float64
    center = torch.tensor([[[1.1, -0.8, 3.5]]], dtype=dtype)
    up = F.normalize(torch.tensor([[0.3, -0.7, 0.5]], dtype=dtype), dim=-1)
    camera = {"focal": torch.tensor([[880, 970]], dtype=dtype),
              "center": torch.tensor([[661, 345]], dtype=dtype)}
    right, forward = tangent_basis(up)
    theta, phi = torch.meshgrid(torch.linspace(0, torch.pi, 121, dtype=dtype),
                                torch.linspace(0, 2 * torch.pi, 241, dtype=dtype), indexing="ij")
    a = (theta.sin() * phi.cos()).reshape(-1, 1)
    b = (theta.sin() * phi.sin()).reshape(-1, 1)
    c = theta.cos().reshape(-1, 1)
    points = center + (0.5 * a * right + 0.5 * b * forward + 0.24 * c * up)[None]
    pixels = project_points(points, camera)
    lo, hi = pixels.amin(dim=1), pixels.amax(dim=1)
    expected = torch.cat(((lo + hi) / 2, hi - lo), dim=-1)
    torch.testing.assert_close(project_body(center, up, camera, 0.24)[:, 0], expected, atol=0.06, rtol=0)


def test_reprojection_gradients_reach_pose_and_selected_camera_only():
    config = load_config(Path(DEFAULT_CONFIG))
    config["model"].update(use_events=False, linear_drag_per_s=0.65)
    model = RelativePoseModel(config, [[0, -1, 0], [0, -0.7, 0.7]])
    past_time = torch.arange(-11, 1).float()[None] / 30
    future_time = torch.arange(1, 25).float()[None] / 30
    past = torch.tensor([0.51, 0.63, 0.05, 0.04]).repeat(1, 12, 1)
    future = torch.tensor([0.56, 0.67, 0.06, 0.06]).repeat(1, 24, 1)
    batch = {"past": past, "past_time": past_time, "future_time": future_time, "scene": torch.tensor([1])}
    result = model(batch)
    torch.testing.assert_close(result["up"].norm(dim=-1), torch.ones(1))
    torch.testing.assert_close(result["position"], result["rollout"][:, 0])
    loss = reprojection_loss(result, dict(batch, future=future), model.cameras.frame_size, config["loss"])
    loss.backward()
    gradient = model.cameras.offsets.grad
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[0]) == 0
    assert torch.count_nonzero(gradient[1, :4]) == 4
    assert model.head[-1].weight.grad[:2].abs().sum() > 0
    assert model.head[-1].weight.grad[9:11].abs().sum() > 0


def test_rotating_thrust_matches_circular_integrals():
    up = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
    omega = math.pi / 2
    zero = torch.zeros_like(up)
    state = FlightState(zero, up, zero, up.new_tensor([[0, 0, omega]]), up.new_tensor([[2]]))
    query = up.new_tensor([[0, 0.5, 1]])
    trajectory = rollout_flight(state, query, up, gravity=0)
    torch.testing.assert_close(trajectory.up.norm(dim=-1), torch.ones_like(query))
    torch.testing.assert_close(trajectory.position[:, 0], zero)
    torch.testing.assert_close(trajectory.velocity[:, 0], zero)
    torch.testing.assert_close(trajectory.up[0, -1], up.new_tensor([1, 0, 0]))
    torch.testing.assert_close(trajectory.velocity[0, -1], up.new_tensor([2 / omega, -2 / omega, 0]))
    torch.testing.assert_close(trajectory.position[0, -1], up.new_tensor([2 / omega - 2 / omega**2, -2 / omega**2, 0]),
                               atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(trajectory.acceleration[0, -1], up.new_tensor([2, 0, 0]))


def test_state_uses_causal_inputs_and_supports_different_rollout_times():
    config = load_config(DEFAULT_CONFIG)
    model = RelativePoseModel(config, [[0, -1, 0]]).eval()
    past_time = torch.arange(-11, 1).float()[None] / 30
    past = torch.tensor([0.48, 0.6, 0.05, 0.04]).repeat(1, 12, 1)
    causal = {"past": past, "past_time": past_time, "scene": torch.tensor([0]),
              "events": {"event_stack": torch.randn(1, 12, 64, 64)}}
    with torch.inference_mode():
        state = model.estimate_state(causal)
        short = model(dict(causal, future_time=torch.tensor([[0.1]])))
        long = model(dict(causal, future_time=torch.tensor([[0.1, 0.8]])))
    torch.testing.assert_close(state.up.norm(dim=-1), torch.ones(1))
    torch.testing.assert_close((state.angular_velocity * state.up).sum(dim=-1), torch.zeros(1))
    assert 0 < state.angular_velocity.norm() <= math.radians(120)
    assert state.thrust.min() > 0
    torch.testing.assert_close(state.position, short["position"])
    torch.testing.assert_close(short["boxes_px"], long["boxes_px"][:, :2])


def test_pose_history_uses_earlier_events_and_matches_buffered_inference():
    config = load_config(DEFAULT_CONFIG)
    config["data"]["history_event_cutouts"] = True
    model = RelativePoseModel(config, [[0, -1, 0]]).eval()
    past = torch.tensor([0.48, 0.6, 0.05, 0.04]).repeat(1, 12, 1)
    past[..., 0] += torch.arange(12) / 1000
    crops = torch.rand(1, 12, 12, 16, 16, requires_grad=True)
    batch = {"past": past, "past_time": torch.arange(-11, 1)[None] / 30,
             "future_time": torch.arange(1, 5)[None] / 30, "future": past[:, -1:].repeat(1, 4, 1),
             "scene": torch.tensor([0]), "events": {"event_stack": crops}}
    result = model(batch)
    training_loss = reprojection_loss(result, batch, model.cameras.frame_size, config["loss"])
    rollout_loss = history_reprojection_loss(model, result, batch, config["loss"])
    for loss in (training_loss, rollout_loss):
        gradient, = torch.autograd.grad(loss, crops, retain_graph=True)
        assert torch.isfinite(gradient).all()
        assert (gradient.abs().flatten(2).sum(-1) > 0).all()

    with torch.inference_mode():
        # Independent per-box calls can be cached as the track is observed.
        observed = model.estimate_observation(
            past.flatten(0, 1), {"event_stack": crops.flatten(0, 1)}, model.cameras(batch["scene"].repeat(12)),
        ).map(lambda x: x.unflatten(0, (1, 12)))
        state = model.fuse_observations(past, batch["past_time"], model.cameras(batch["scene"]), observed)
    torch.testing.assert_close(result["pose_history"].up, observed.up)
    torch.testing.assert_close(result["state"].position, state.position)
    torch.testing.assert_close(result["state"].velocity, state.velocity)
    torch.testing.assert_close(result["state"].up.norm(dim=-1), torch.ones(1))


def test_pose_history_pairs_each_crop_with_its_own_box_and_timestamp(tmp_path, monkeypatch):
    root = tmp_path / "fred"
    folder = root / "3"
    events = folder / "Event"
    events.mkdir(parents=True)
    (events / "events.raw").touch()
    (events / "events.raw.tmp_index").touch()
    times = [(i + 1) * 33333 + i % 7 for i in range(50)]
    (folder / "coordinates.txt").write_text(
        "".join(f"{t / 1e6:.6f}: {100 + i}, 100, {150 + i}, 135\n" for i, t in enumerate(times)),
        encoding="utf-8",
    )

    def preview(self, rep, reference, box):
        return torch.tensor([reference["time_us"], box[0]], dtype=torch.float64)

    monkeypatch.setattr(_ForecastTracklets, "_prepared_input", preview)
    data = load_config(DEFAULT_CONFIG)["data"]
    data.update(labels_root=str(root), coordinate_track_cache=str(tmp_path / "coords"),
                cache_dir=str(tmp_path / "tracklets"), instance_cache={}, fallback_frame_timestamps=True,
                history_event_cutouts=True, forecast_ms=400, event_window_ms=100)
    dataset = PoseTracklets(data, ["3"], {"names": ["test"], "by_folder": {"3": "test"}},
                            split="train", seed=1)
    sample = dataset[0]
    crops = sample["events"]["event_stack"]
    first = times.index(round(sample["anchor_time"] * 1e6)) - 11
    assert crops[:, 0].tolist() == times[first:first + 12]
    torch.testing.assert_close(crops[:, 1], sample["past"][:, 0].double())


def test_coordinate_fallback_splits_at_ambiguous_frames(tmp_path):
    root = tmp_path / "fred"
    folder = root / "3"
    folder.mkdir(parents=True)
    labels = folder / "coordinates.txt"
    lines = []
    for frame in range(46):
        time = (frame + 1) * 0.033333
        lines.append(f"{time:.6f}: {100 + frame}, 100, {145 + frame}, 122\n")
        if frame == 14:
            lines.append(f"{time:.6f}: 200, 100, 245, 122\n")
    labels.write_text("".join(lines), encoding="utf-8")
    cache = tmp_path / "cache"
    data = load_config(DEFAULT_CONFIG)["data"]
    data.update(labels_root=str(root), coordinate_track_cache=str(cache), forecast_ms=400,
                event_window_ms=100, cache_dir=str(tmp_path / "tracklets"), instance_cache={}, fallback_frame_timestamps=True)
    path = coordinate_tracks(data, "3")
    assert path.parent == cache
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 31
    assert {int(row[1]) for row in rows} == {300000}
    assert float(rows[0][0]) > 15 * 0.033333
    scenes = {"names": ["test"], "by_folder": {"3": "test"}}
    dataset = PoseTracklets(data, ["3"], scenes, split="val", seed=21, use_events=False)
    assert len(dataset) == 8
    assert dataset.forecast.image_window_ms == 100
    assert dataset[0]["past"].shape == (12, 4)
    assert dataset[0]["future"].shape == (12, 4)
