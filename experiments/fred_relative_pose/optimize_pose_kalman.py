"""Fit image-plane Kalman and 3D Kalman variants for 800 ms FRED forecasts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader

from experiments.kalman_ml_forecasting.models.kalman_residual import last_four_constant_velocity_forecast
from experiments.kalman_ml_forecasting.optimize_coupled_kalman import _metric_tensors, _weighted_loss
from experiments.kalman_ml_forecasting.utils.config import read_split_file

from .data import PoseTracklets, to_device, tracklet_key
from .eval import check_test_split
from .model import RelativePoseModel
from .pose_ekf import PoseEKF3D, pose_observations
from .pose_kalman import LOG_SIZE_SCALE, TIME_UNIT_S, PoseBoxKalman, pose_measurements
from .train import save_checkpoint, write_json


HORIZON_MS = 800
# Scale the three terms to comparable magnitudes using round pixel scales.
OBJECTIVE = {"ade_bbox_px": 1 / 30, "fde_bbox_px": 1 / 65, "miou": -1 / .65}
FILTERS = {"box_only": False, "box_pose": True}
FILTERS_3D = {"pose_3d": False, "box_pose_3d": True}
CAMERA_KEYS = ("focal", "center", "gravity_up")
Filter = Union[PoseBoxKalman, PoseEKF3D]


@torch.no_grad()
def extract_observations(checkpoint: dict, model: RelativePoseModel, split: str, device: torch.device,
                         workers: int, space: str = "image") -> tuple[dict[str, torch.Tensor], dict]:
    data = copy.deepcopy(checkpoint["config"]["data"])
    scenes, splits = checkpoint["scenes"], checkpoint["splits"]
    settings = checkpoint["config"]["train"]
    seed = settings["seed"] + ("train", "val", "test").index(split)
    anchors = None
    if split == "test":
        folders = read_split_file(Path(data["split_files"]["test"]))
        check_test_split(checkpoint, folders)
    else:
        folders = splits[split]
        original = PoseTracklets(data, folders, scenes, split=split, seed=seed)
        if original.sample_digest != splits["tracklet_sha256"][split]:
            raise ValueError(f"{split} anchors differ from the frozen checkpoint")
        anchors = {tracklet_key(sample) for sample in original.forecast.samples}
        del original
    data["forecast_ms"] = HORIZON_MS
    data["max_samples_per_sequence"].pop(split, None)
    dataset = PoseTracklets(data, folders, scenes, split=split, seed=seed, anchors=anchors)
    samples = dataset.forecast.samples
    order = sorted(range(len(samples)), key=lambda i: (int(samples[i]["folder"]), samples[i]["anchor_time_s"]))
    loader = DataLoader(dataset, batch_size=settings["batch_size"], sampler=order, num_workers=workers,
                        pin_memory=device.type == "cuda")
    frame = tuple(data["frame_size"])
    scale = torch.tensor(frame * 2, device=device)
    fallback = dataset.forecast.fallback
    fitted = {scene for scene, row in checkpoint["cameras"].items() if row["fitted"]}
    rows = []
    start = time.perf_counter()
    for step, raw in enumerate(loader, 1):
        batch = to_device(raw, device)
        result = model(batch)
        if space == "3d":
            observations, range_scale = pose_observations(model, result["pose_history"], result["gravity_up"])
            geometry = {"range_scale": range_scale.cpu(), **{key: result[key].cpu() for key in CAMERA_KEYS}}
        else:
            observations = pose_measurements(model, result["pose_history"], result, frame)
            geometry = {}
        rows.append({
            **{key: raw[key] for key in ("past", "past_time", "future", "future_time")},
            "pose": observations.cpu(),
            **geometry,
            "pose_history_model": (result["boxes_px"][:, 1:] / scale).cpu(),
            "sequence": torch.tensor([int(folder) for folder in raw["folder"]]),
            "coordinate_tracks": torch.tensor([folder in fallback for folder in raw["folder"]]),
            "camera_prior": torch.tensor([scenes["by_folder"][folder] not in fitted for folder in raw["folder"]]),
        })
        if step == 1 or step % 200 == 0:
            rate = min(step * settings["batch_size"], len(dataset)) / (time.perf_counter() - start)
            print(f"{split} measurements {step}/{len(loader)} at {rate:.0f} tracklets/s", flush=True)
    observations = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    if not all(torch.isfinite(observations[key]).all() for key in ("pose", "pose_history_model")):
        raise ValueError(f"Nonfinite frozen pose measurements on {split}")
    metadata = {"requested_sequences": folders, "tracklets_by_sequence": dataset.count_by_folder,
                "tracklet_sha256": dataset.sample_digest,
                "source_anchor_count": len(anchors) if anchors is not None else None}
    return observations, metadata


def get_observations(output: Path, checkpoint: dict, model: RelativePoseModel, split: str,
                     provenance: dict, device: torch.device, workers: int) -> tuple[dict, dict]:
    path = output / f"{split}_measurements.pt"
    if path.exists():
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if entry["provenance"] != provenance:
            raise ValueError(f"{path} belongs to a different checkpoint or observation protocol")
        print(f"Reusing {path}", flush=True)
    else:
        data, metadata = extract_observations(checkpoint, model, split, device, workers,
                                              space=provenance.get("space", "image"))
        entry = {"provenance": provenance, "data": data, "metadata": metadata}
        save_checkpoint(path, entry)
    return {key: value.to(device) for key, value in entry["data"].items()}, entry["metadata"]


def predict(model: Filter, data: dict, indices: torch.Tensor) -> torch.Tensor:
    boxes = data["past"][indices]
    past_time = data["past_time"][indices]
    future_time = data["future_time"][indices]
    if isinstance(model, PoseEKF3D):
        camera = {key: data[key][indices] for key in CAMERA_KEYS}
        return model(boxes, past_time, future_time, data["pose"][indices], data["range_scale"][indices], camera)
    pose = data["pose"][indices] if model.use_pose else None
    return model(boxes, past_time, future_time, pose)


@torch.no_grad()
def score(data: dict, indices: torch.Tensor, models: dict[str, Filter],
          frame: tuple[int, int], batch_size: int, *, baselines: bool = False) -> dict:
    totals = {}
    for batch in indices.split(batch_size):
        predictions = {name: predict(model, data, batch) for name, model in models.items()}
        if baselines:
            predictions["pose_history_model"] = data["pose_history_model"][batch]
            predictions["last4"] = last_four_constant_velocity_forecast(
                data["past"][batch], data["past_time"][batch], data["future_time"][batch],
            )
        for name, prediction in predictions.items():
            metrics = _metric_tensors(prediction, data["future"][batch], frame)
            totals[name] = totals.get(name, torch.zeros(len(metrics), device=indices.device)) + (
                torch.stack(tuple(metrics.values())) * len(batch)
            )
    return {name: dict(zip(metrics, (total / len(indices)).tolist())) for name, total in totals.items()}


def fit(train: dict, val: dict, frame: tuple[int, int], args: argparse.Namespace,
        device: torch.device, half_height: float) -> tuple[dict[str, Filter], dict]:
    if args.space == "3d":
        models = {name: PoseEKF3D(frame, half_height, use_boxes=boxes).to(device)
                  for name, boxes in FILTERS_3D.items()}
    else:
        models = {name: PoseBoxKalman(use_pose=pose).to(device) for name, pose in FILTERS.items()}
    optimizer = torch.optim.Adam([p for model in models.values() for p in model.parameters()], lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .1)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    val_indices = torch.arange(len(val["past"]), device=device)
    history = []
    best = {}
    states = {}
    for epoch in range(args.epochs + 1):
        start = time.perf_counter()
        if epoch:
            for indices in torch.randperm(len(train["past"]), device=device, generator=generator).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                for model in models.values():
                    loss = _weighted_loss(_metric_tensors(predict(model, train, indices), train["future"][indices], frame),
                                          OBJECTIVE)
                    loss.backward()
                for model in models.values():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                for model in models.values():
                    model.clamp_()
            schedule.step()
        scores = score(val, val_indices, models, frame, args.batch_size, baselines=epoch == 0)
        history.append({"epoch": epoch, "validation": scores, "seconds": time.perf_counter() - start})
        summary = []
        for name, model in models.items():
            metrics = scores[name]
            value = float(_weighted_loss(metrics, OBJECTIVE)) - OBJECTIVE["miou"]
            if name not in best or value < best[name]["objective"]:
                best[name] = {"epoch": epoch, "objective": value, "metrics": metrics}
                states[name] = copy.deepcopy(model.state_dict())
            summary.append(f"{name} {metrics['ade_bbox_px']:.2f}/{metrics['fde_bbox_px']:.2f}px "
                           f"IoU {metrics['miou']:.3f} best epoch {best[name]['epoch']}")
        print(f"epoch {epoch:02d} {history[-1]['seconds']:.1f}s | " + " | ".join(summary), flush=True)
        write_json(args.output_dir / "validation.json", {"epochs": history, "best": best})
        save_checkpoint(args.output_dir / "best.pt", {"states": states, "best": best})
    for name, model in models.items():
        model.load_state_dict(states[name])
        model.eval()
    return models, {"epochs": history, "best": best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--space", choices=("image", "3d"), default="image",
                        help="State coordinates used for filtering and rollout")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=.015)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--run-test-on-best", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        parser.error("Positive epochs, batch size, and learning rate and nonnegative workers are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_sequences, val_sequences = (set(checkpoint["splits"][key]) for key in ("train", "val"))
    data = checkpoint["config"]["data"]
    canonical = set(read_split_file(Path(data["split_files"]["train"])))
    test = read_split_file(Path(data["split_files"]["test"]))
    check_test_split(checkpoint, test)
    if train_sequences & val_sequences or (train_sequences | val_sequences) - canonical:
        raise ValueError("Use disjoint checkpoint train/validation subsets of the canonical training list")
    model = RelativePoseModel(checkpoint["config"], checkpoint["scenes"]["up_priors"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    if model.pose_history_head is None:
        raise ValueError("The frozen checkpoint must provide per-frame pose history")
    provenance = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  "forecast_ms": HORIZON_MS, "time_unit_s": TIME_UNIT_S, "log_size_scale": LOG_SIZE_SCALE,
                  "measurement_version": 1}
    if args.space == "3d":
        provenance["space"] = args.space
    frame = tuple(data["frame_size"])
    split_report = {}
    observations = []
    for split in ("train", "val"):
        values, split_report[split] = get_observations(args.output_dir, checkpoint, model, split,
                                                     provenance, device, args.workers)
        observations.append(values)
    print(f"Fitting on {len(observations[0]['past'])} train and {len(observations[1]['past'])} validation windows",
          flush=True)
    filters, optimization = fit(*observations, frame, args, device, model.half_height)
    report = {"checkpoint": str(args.checkpoint), **provenance, "frame_size_px": frame,
              "objective_weights": OBJECTIVE, "optimization": optimization, "splits": split_report,
              "learning_rate": args.lr, "seed": args.seed, "batch_size": args.batch_size,
              "half_height_body_diameters": model.half_height,
              "reference_diameter_m": checkpoint["config"]["model"]["reference_diameter_m"],
              "filter_parameters": {name: {key: value.tolist() for key, value in f.state_dict().items()}
                                    for name, f in filters.items()}}
    write_json(args.output_dir / "report.json", report)
    if args.run_test_on_best:
        test_data, split_report["test"] = get_observations(args.output_dir, checkpoint, model, "test",
                                                         provenance, device, args.workers)
        indices = torch.arange(len(test_data["past"]), device=device)
        groups = {"all": indices, "cleaned_tracks": indices[~test_data["coordinate_tracks"]],
                  "coordinate_tracks": indices[test_data["coordinate_tracks"]],
                  "fitted_camera": indices[~test_data["camera_prior"]],
                  "camera_prior": indices[test_data["camera_prior"]]}
        results = {name: {"tracklets": len(group), "metrics": score(test_data, group, filters, frame,
                                                                    args.batch_size, baselines=True)}
                   for name, group in groups.items() if len(group)}
        report["results"] = results
        split_report["test"]["by_sequence"] = dict(Counter(test_data["sequence"].tolist()))
        write_json(args.output_dir / "report.json", report)
        for name, metrics in results["all"]["metrics"].items():
            print(f"{name} | box ADE/FDE {metrics['ade_bbox_px']:.2f}/{metrics['fde_bbox_px']:.2f}px | "
                  f"mIoU {metrics['miou']:.3f}", flush=True)


if __name__ == "__main__":
    main()
