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
def evaluate(paths: list[Path], split_file: Path | None, output: Path, device: torch.device, workers: int) -> dict:
    checkpoints = {path.parent.name: torch.load(path, map_location="cpu", weights_only=True) for path in paths}
    if len(checkpoints) != len(paths) or "last4" in checkpoints:
        raise ValueError("Each checkpoint must come from a distinctly named run directory")
    first = next(iter(checkpoints.values()))
    data = copy.deepcopy(first["config"]["data"])
    scenes = first["scenes"]
    if split_file is None:
        split_file = Path(data["split_files"]["test"])
    folders = read_split_file(split_file)
    for checkpoint in checkpoints.values():
        check_test_split(checkpoint, folders)
        if checkpoint["config"]["data"] != data or checkpoint["scenes"] != scenes:
            raise ValueError("Paired checkpoints must share the same data and scene configuration")
        if checkpoint["splits"] != first["splits"]:
            raise ValueError("Paired checkpoints must share the same training and validation tracklets")
    data.get("max_samples_per_sequence", {}).pop("test", None)

    models = {}
    for name, checkpoint in checkpoints.items():
        model = RelativePoseModel(checkpoint["config"], scenes["up_priors"]).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        models[name] = model
    fitted = {name for name, row in first["cameras"].items() if row["fitted"]}
    for checkpoint in checkpoints.values():
        if {name for name, row in checkpoint["cameras"].items() if row["fitted"]} != fitted:
            raise ValueError("Paired checkpoints must agree on which camera scenes were fitted")

    settings = first["config"]["train"]
    dataset = PoseTracklets(data, folders, scenes, split="test", seed=settings["seed"] + 2,
                            use_events=any(model.encoders for model in models.values()))
    samples = dataset.forecast.samples
    order = sorted(range(len(samples)), key=lambda i: (int(samples[i]["folder"]), samples[i]["anchor_time_s"]))
    loader = DataLoader(dataset, batch_size=settings["batch_size"], sampler=order, num_workers=workers,
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
        predictions = {name: model(batch)["boxes_px"][:, 1:] / scale for name, model in models.items()}
        predictions["last4"] = last_four_constant_velocity_forecast(batch["past"], batch["past_time"], batch["future_time"])
        groups = defaultdict(list)
        for i, folder in enumerate(raw["folder"]):
            groups["all"].append(i)
            groups["coordinate_tracks" if folder in fallback else "cleaned_tracks"].append(i)
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
        "camera_prior_sequences": [folder for folder in folders if scenes["by_folder"][folder] not in fitted],
        "checkpoints": {path.parent.name: {"path": str(path), "epoch": checkpoints[path.parent.name]["epoch"]}
                        for path in paths},
        "results": {group: {"tracklets": count, "metrics": {
            name: {key: total / count for key, total in totals[group, name].items()} for name in predictions
        }} for group, count in counts.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, help="Defaults to the test list in the checkpoint config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    report = evaluate(args.checkpoint, args.split_file, args.output, torch.device(args.device), args.workers)
    for name, metrics in report["results"]["all"]["metrics"].items():
        print(f"{name} | center ADE {metrics['ade_center_px']:.2f}px | center FDE {metrics['fde_center_px']:.2f}px")


if __name__ == "__main__":
    main()
