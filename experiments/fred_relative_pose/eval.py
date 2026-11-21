"""Score frozen pose checkpoints on a sequence-disjoint FRED test split."""

from __future__ import annotations

import argparse
import copy
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.kalman_ml_forecasting.metrics import summarize_forecast_metrics
from experiments.kalman_ml_forecasting.models.kalman_residual import last_four_constant_velocity_forecast
from experiments.kalman_ml_forecasting.utils.config import read_split_file

from .data import PoseTracklets, to_device
from .model import RelativePoseModel
from .train import write_json


TRAINING_SOURCE_KEYS = {
    "coordinate_track_cache", "coordinate_track_method", "fallback_frame_timestamps",
    "split_reference", "sample_seed_by_sequence",
}


def check_test_split(checkpoint: dict, folders: list[str]) -> None:
    if not folders or len(set(folders)) != len(folders):
        raise ValueError("Test split must list unique sequence IDs")
    splits = checkpoint["splits"]
    overlaps = set(folders) & (set(splits["train"]) | set(splits["val"]))
    if overlaps:
        raise ValueError(f"Test sequences were used for training or checkpoint selection: {sorted(overlaps, key=int)}")
    unknown = set(folders) - checkpoint["scenes"]["by_folder"].keys()
    if unknown:
        raise ValueError(f"Test sequences lack scene metadata: {sorted(unknown, key=int)}")


@torch.inference_mode()
def evaluate(paths: list[Path], split_file: Path | None, output: Path, device: torch.device, workers: int,
             *, forecast_ms: int | None = None, compare_training_sources: bool = False) -> dict:
    checkpoints = {path.parent.name: torch.load(path, map_location="cpu", weights_only=True) for path in paths}
    if len(checkpoints) != len(paths) or "last4" in checkpoints:
        raise ValueError("Each checkpoint must come from a distinctly named run directory")
    first = next(iter(checkpoints.values()))
    data = copy.deepcopy(first["config"]["data"])
    scenes = first["scenes"]
    if split_file is None:
        split_file = Path(data["split_files"]["test"])
    folders = read_split_file(split_file)
    # A single test stream can serve both input modes by taking its last crop.
    ignored = {"history_event_cutouts"}
    if compare_training_sources:
        ignored.update(TRAINING_SOURCE_KEYS)
    comparison_data = {key: value for key, value in data.items()
                       if key not in ignored}
    for checkpoint in checkpoints.values():
        check_test_split(checkpoint, folders)
        training_data = {key: value for key, value in checkpoint["config"]["data"].items()
                         if key not in ignored}
        if training_data != comparison_data or checkpoint["scenes"] != scenes:
            raise ValueError("Paired checkpoints must share the same data and scene configuration")
        if not compare_training_sources and checkpoint["splits"] != first["splits"]:
            raise ValueError("Paired checkpoints must share the same training and validation tracklets")
    data.get("max_samples_per_sequence", {}).pop("test", None)
    if forecast_ms is not None:
        data["forecast_ms"] = forecast_ms

    models = {}
    for name, checkpoint in checkpoints.items():
        model = RelativePoseModel(checkpoint["config"], scenes["up_priors"]).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        models[name] = model
    fitted_by_run = {run: {name for name, row in checkpoint["cameras"].items() if row["fitted"]}
                     for run, checkpoint in checkpoints.items()}
    fitted = next(iter(fitted_by_run.values()))
    common_cameras = all(cameras == fitted for cameras in fitted_by_run.values())
    if not compare_training_sources and not common_cameras:
        raise ValueError("Paired checkpoints must agree on which camera scenes were fitted")

    settings = first["config"]["train"]
    data["history_event_cutouts"] = any(model.pose_history_head is not None for model in models.values())
    dataset = PoseTracklets(data, folders, scenes, split="test", seed=settings["seed"] + 2,
                            use_events=any(model.encoders for model in models.values()))
    samples = dataset.forecast.samples
    order = sorted(range(len(samples)), key=lambda i: (int(samples[i]["folder"]), samples[i]["anchor_time_s"]))
    batch_size = min(checkpoint["config"]["train"]["batch_size"] for checkpoint in checkpoints.values())
    loader = DataLoader(dataset, batch_size=batch_size, sampler=order, num_workers=workers,
                        pin_memory=device.type == "cuda")
    frame = tuple(data["frame_size"])
    scale = torch.tensor(frame * 2, device=device)
    fallback = dataset.forecast.fallback
    totals = defaultdict(lambda: defaultdict(float))
    counts = Counter()
    started = time.perf_counter()
    for step, raw in enumerate(loader, 1):
        batch = to_device(raw, device)
        target = batch["future"]
        anchor_batch = batch
        if data["history_event_cutouts"]:
            anchor_batch = {**batch, "events": {rep: crops[:, -1] for rep, crops in batch["events"].items()}}
        predictions = {
            name: model(batch if model.pose_history_head is not None else anchor_batch)["boxes_px"][:, 1:] / scale
            for name, model in models.items()
        }
        predictions["last4"] = last_four_constant_velocity_forecast(batch["past"], batch["past_time"], batch["future_time"])
        groups = defaultdict(list)
        for i, folder in enumerate(raw["folder"]):
            groups["all"].append(i)
            groups["coordinate_tracks" if folder in fallback else "cleaned_tracks"].append(i)
            if common_cameras:
                groups["fitted_camera" if scenes["by_folder"][folder] in fitted else "camera_prior"].append(i)
        for group, indices in groups.items():
            counts[group] += len(indices)
            selection = slice(None) if len(indices) == len(target) else torch.tensor(indices, device=device)
            for name, boxes in predictions.items():
                scores = summarize_forecast_metrics(boxes[selection], target[selection], frame)
                for key, value in scores.items():
                    totals[group, name][key] += len(indices) * value
        if step == 1 or step % settings["log_every"] == 0:
            print(f"test {step}/{len(loader)} | {counts['all']} tracklets | "
                  f"{counts['all'] / (time.perf_counter() - started):.1f} tracklets/s", flush=True)
    report = {
        "split_file": str(split_file), "requested_sequences": folders,
        "tracklets_by_sequence": dict(sorted(dataset.count_by_folder.items(), key=lambda pair: int(pair[0]))),
        "unscored_sequences": [folder for folder in folders if folder not in dataset.count_by_folder],
        "tracklet_sha256": dataset.sample_digest, "forecast_ms": data["forecast_ms"], "frame_size_px": list(frame),
        "test_coordinate_track_method": (data.get("coordinate_track_method", "conservative")
                                         if data.get("coordinate_track_cache") else None),
        "checkpoints": {path.parent.name: {
            "path": str(path), "epoch": checkpoints[path.parent.name]["epoch"],
            "training_forecast_ms": checkpoints[path.parent.name]["config"]["data"]["forecast_ms"],
        } for path in paths},
        "results": {group: {"tracklets": count, "metrics": {
            name: {key: total / count for key, total in totals[group, name].items()} for name in predictions
        }} for group, count in counts.items()},
    }
    if common_cameras:
        report["camera_prior_sequences"] = [folder for folder in folders if scenes["by_folder"][folder] not in fitted]
    else:
        report["camera_prior_sequences_by_checkpoint"] = {
            run: [folder for folder in folders if scenes["by_folder"][folder] not in cameras]
            for run, cameras in fitted_by_run.items()
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, help="Defaults to the test list in the checkpoint config")
    parser.add_argument("--forecast-ms", type=int, choices=(400, 800), help="Rollout horizon with the checkpoint frozen")
    parser.add_argument("--compare-training-sources", action="store_true",
                        help="Allow differing training annotation sources. The first checkpoint sets the test source.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    report = evaluate(args.checkpoint, args.split_file, args.output, torch.device(args.device), args.workers,
                      forecast_ms=args.forecast_ms, compare_training_sources=args.compare_training_sources)
    for name, metrics in report["results"]["all"]["metrics"].items():
        print(f"{name} | center ADE {metrics['ade_center_px']:.2f}px | center FDE {metrics['fde_center_px']:.2f}px")


if __name__ == "__main__":
    main()
