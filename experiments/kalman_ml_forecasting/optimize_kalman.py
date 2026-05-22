from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import TrackKalmanForecastDataset
from experiments.kalman_ml_forecasting.metrics import summarize_forecast_metrics
from experiments.kalman_ml_forecasting.models.kalman_filter import (
    DEFAULT_KALMAN_CONFIG,
    kalman_config_from_dict,
    kalman_cv_forecast,
)
from experiments.kalman_ml_forecasting.models.kalman_residual import last_two_constant_velocity_forecast
from experiments.kalman_ml_forecasting.utils.config import (
    load_config,
    read_split_file,
    resolve_representation_image_sizes,
)


SEARCH_RANGES: dict[str, tuple[float, float]] = {
    "initial_pos_std": (1.0e-3, 2.0e-1),
    "initial_size_std": (1.0e-3, 2.0e-1),
    "initial_vel_std": (1.0e-2, 5.0),
    "process_pos_std": (1.0e-5, 5.0e-2),
    "process_size_std": (1.0e-5, 5.0e-2),
    "process_vel_std": (1.0e-3, 2.0),
    "process_size_vel_std": (1.0e-3, 2.0),
    "measurement_pos_std": (1.0e-4, 2.0e-1),
    "measurement_size_std": (1.0e-4, 2.0e-1),
}


def _build_train_dataset(cfg: Dict[str, Any], *, max_samples: int | None) -> TrackKalmanForecastDataset:
    data_cfg = cfg["data"]
    split_files = data_cfg.get("split_files")
    folders = read_split_file(Path(split_files["train"])) if split_files else None
    return TrackKalmanForecastDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        frame_size=tuple(data_cfg["frame_size"]),
        representations=list(data_cfg["representations"]),
        image_sizes=resolve_representation_image_sizes(data_cfg),
        history_ms=float(data_cfg.get("history_ms", 400.0)),
        forecast_ms=float(data_cfg.get("forecast_ms", 800.0)),
        folders=folders,
        labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
        tracks_file=data_cfg.get("tracks_file", "cleaned_tracks.txt"),
        label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
        track_time_unit=float(data_cfg.get("track_time_unit", 1.0)),
        time_align=data_cfg.get("time_align", "auto"),
        image_window_ms=float(data_cfg.get("image_window_ms", 400.0)),
        image_window_mode=data_cfg.get("image_window_mode", "trailing"),
        verify_render_manifest=False,
        render_manifest_name=data_cfg.get("render_manifest_name", "render_manifest.json"),
        window_tolerance_ms=float(data_cfg.get("window_tolerance_ms", 5.0)),
        label_period_s=data_cfg.get("label_period_s"),
        max_tracks=data_cfg.get("max_tracks_train", data_cfg.get("max_tracks")),
        max_samples=max_samples,
        seed=int(data_cfg.get("seed", 123)),
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
        filter_missing_representations=False,
        require_representations=False,
    )


def _split_indices_by_track(
    samples: list[dict],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    by_group: dict[tuple[str, int], list[int]] = {}
    for idx, sample in enumerate(samples):
        key = (str(sample["folder"]), int(sample["track_id"]))
        by_group.setdefault(key, []).append(idx)
    groups = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_count = max(1, int(round(len(groups) * val_fraction))) if groups else 0
    val_groups = set(groups[:val_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for group, indices in by_group.items():
        if group in val_groups:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return train_indices, val_indices


def _limit_indices(indices: list[int], *, max_count: int | None, seed: int) -> list[int]:
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    rng = random.Random(seed)
    out = list(indices)
    rng.shuffle(out)
    return out[:max_count]


def _stack_batch(samples: list[dict], indices: list[int], *, device: torch.device):
    past_boxes = torch.tensor([samples[i]["past_boxes"] for i in indices], dtype=torch.float32, device=device)
    future_boxes = torch.tensor([samples[i]["future_boxes"] for i in indices], dtype=torch.float32, device=device)
    past_times = torch.tensor([samples[i]["past_times_s"] for i in indices], dtype=torch.float32, device=device)
    future_times = torch.tensor([samples[i]["future_times_s"] for i in indices], dtype=torch.float32, device=device)
    return past_boxes, future_boxes, past_times, future_times


def _weighted_mean(rows: list[dict[str, float]], weights: list[int]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    total = float(sum(weights))
    return {
        key: sum(row[key] * weight for row, weight in zip(rows, weights) if key in row) / total
        for key in keys
    }


def _evaluate(
    samples: list[dict],
    indices: list[int],
    *,
    frame_size: tuple[int, int],
    params: Dict[str, Any] | None,
    batch_size: int,
    device: torch.device,
    baseline: str,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    weights: list[int] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        past_boxes, future_boxes, past_times, future_times = _stack_batch(samples, batch_indices, device=device)
        if baseline == "kalman":
            pred = kalman_cv_forecast(past_boxes, past_times, future_times, params)
        elif baseline == "last2":
            pred = last_two_constant_velocity_forecast(past_boxes, past_times, future_times)
        else:
            raise ValueError(f"Unknown baseline: {baseline}")
        rows.append(summarize_forecast_metrics(pred, future_boxes, frame_size))
        weights.append(len(batch_indices))
    return _weighted_mean(rows, weights)


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return 10.0 ** rng.uniform(math.log10(lo), math.log10(hi))


def _sample_params(rng: random.Random, base_cfg: Dict[str, Any]) -> dict[str, float | bool]:
    params = kalman_config_from_dict(base_cfg)
    params["enabled"] = True
    for key, bounds in SEARCH_RANGES.items():
        params[key] = _log_uniform(rng, bounds[0], bounds[1])
    return params


def _format_yaml(params: Dict[str, Any]) -> str:
    lines = ["kalman:"]
    for key in DEFAULT_KALMAN_CONFIG:
        value = params[key]
        if isinstance(value, bool):
            lines.append(f"  {key}: {str(value).lower()}")
        else:
            lines.append(f"  {key}: {float(value):.8g}")
    return "\n".join(lines)


def _parse_objective_weights(raw: str | None, objective: str, maximize_objective: bool) -> dict[str, float]:
    if raw is None or not str(raw).strip():
        return {objective: -1.0 if maximize_objective else 1.0}
    out: dict[str, float] = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid objective weight '{item}'. Expected comma-separated metric=weight pairs."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid objective weight '{item}': empty metric name.")
        out[key] = float(value)
    if not out:
        raise ValueError("objective weights must contain at least one metric=weight pair.")
    return out


def _objective_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    missing = [key for key in weights if key not in metrics]
    if missing:
        available = ", ".join(sorted(metrics))
        raise KeyError(f"Objective metric(s) missing: {missing}. Available metrics: {available}")
    return float(sum(float(metrics[key]) * float(weight) for key, weight in weights.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-search CV Kalman filter trust parameters on the training split."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tune-val-fraction", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="fde_center_px")
    parser.add_argument(
        "--maximize-objective",
        action="store_true",
        help="Maximize --objective by minimizing its negative. Ignored when --objective-weights is set.",
    )
    parser.add_argument(
        "--objective-weights",
        type=str,
        default=None,
        help=(
            "Comma-separated metric=weight pairs for the minimized score. "
            "Use negative weights for metrics to maximize, e.g. miou=-100,fde_center_px=1."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit before the track split.")
    parser.add_argument("--max-tune-train-samples", type=int, default=None)
    parser.add_argument("--max-tune-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    frame_size = tuple(int(v) for v in cfg["data"]["frame_size"])
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    objective_weights = _parse_objective_weights(
        args.objective_weights,
        objective=str(args.objective),
        maximize_objective=bool(args.maximize_objective),
    )
    objective_label = (
        args.objective
        if args.objective_weights is None
        else ",".join(f"{key}={value:g}" for key, value in objective_weights.items())
    )
    dataset = _build_train_dataset(cfg, max_samples=args.max_samples)
    if len(dataset.samples) == 0:
        raise RuntimeError("Training dataset yielded zero Kalman ML samples.")

    train_idx, val_idx = _split_indices_by_track(
        dataset.samples,
        val_fraction=float(args.tune_val_fraction),
        seed=int(args.seed),
    )
    total_groups = len({(str(sample["folder"]), int(sample["track_id"])) for sample in dataset.samples})
    print(f"Total train split Kalman tracklets: {len(dataset.samples)}")
    print(f"Total train split (folder, track_id) groups: {total_groups}")
    print(f"Tune train samples before optional cap: {len(train_idx)}")
    print(f"Tune val samples before optional cap: {len(val_idx)}")
    train_idx = _limit_indices(train_idx, max_count=args.max_tune_train_samples, seed=int(args.seed) + 1)
    val_idx = _limit_indices(val_idx, max_count=args.max_tune_val_samples, seed=int(args.seed) + 2)
    if not train_idx or not val_idx:
        raise RuntimeError(
            f"Need non-empty tune train and tune val splits, got {len(train_idx)} and {len(val_idx)} samples."
        )
    print(f"Tune train samples: {len(train_idx)}")
    print(f"Tune val samples: {len(val_idx)}")
    print(f"Objective score minimized: {objective_label}")

    configured = kalman_config_from_dict(cfg.get("kalman"))
    last2_train = _evaluate(
        dataset.samples,
        train_idx,
        frame_size=frame_size,
        params=None,
        batch_size=int(args.batch_size),
        device=device,
        baseline="last2",
    )
    last2_val = _evaluate(
        dataset.samples,
        val_idx,
        frame_size=frame_size,
        params=None,
        batch_size=int(args.batch_size),
        device=device,
        baseline="last2",
    )
    configured_val = _evaluate(
        dataset.samples,
        val_idx,
        frame_size=frame_size,
        params=configured,
        batch_size=int(args.batch_size),
        device=device,
        baseline="kalman",
    )
    configured_train = _evaluate(
        dataset.samples,
        train_idx,
        frame_size=frame_size,
        params=configured,
        batch_size=int(args.batch_size),
        device=device,
        baseline="kalman",
    )
    print(f"Last-two CV tune train: {json.dumps(last2_train, sort_keys=True)}")
    print(f"Last-two CV tune val:   {json.dumps(last2_val, sort_keys=True)}")
    print(f"Configured Kalman train:{json.dumps(configured_train, sort_keys=True)}")
    print(f"Configured Kalman val:  {json.dumps(configured_val, sort_keys=True)}")

    rng = random.Random(int(args.seed))
    configured_row: dict[str, Any] = {
        "trial": "configured",
        "params": configured,
        "train": configured_train,
        "val": configured_val,
        "train_score": _objective_score(configured_train, objective_weights),
        "val_score": _objective_score(configured_val, objective_weights),
    }
    trials: list[dict[str, Any]] = [configured_row]
    best: dict[str, Any] = configured_row
    print(
        f"configured initial best score={best['val_score']:.4f} "
        f"train_score={best['train_score']:.4f}"
    )
    for trial in range(int(args.trials)):
        params = _sample_params(rng, cfg.get("kalman") or {})
        train_metrics = _evaluate(
            dataset.samples,
            train_idx,
            frame_size=frame_size,
            params=params,
            batch_size=int(args.batch_size),
            device=device,
            baseline="kalman",
        )
        val_metrics = _evaluate(
            dataset.samples,
            val_idx,
            frame_size=frame_size,
            params=params,
            batch_size=int(args.batch_size),
            device=device,
            baseline="kalman",
        )
        row = {
            "trial": trial,
            "params": params,
            "train": train_metrics,
            "val": val_metrics,
            "train_score": _objective_score(train_metrics, objective_weights),
            "val_score": _objective_score(val_metrics, objective_weights),
        }
        trials.append(row)
        score = float(row["val_score"])
        if score < float(best["val_score"]):
            best = row
            print(
                f"trial {trial:04d} new best score={score:.4f} "
                f"train_score={row['train_score']:.4f} "
                f"val_metrics={json.dumps(val_metrics, sort_keys=True)}"
            )

    print("Best Kalman params:")
    print(_format_yaml(best["params"]))
    print(f"Best tune train score: {best['train_score']:.6f}")
    print(f"Best tune val score:   {best['val_score']:.6f}")
    print(f"Best tune train metrics: {json.dumps(best['train'], sort_keys=True)}")
    print(f"Best tune val metrics:   {json.dumps(best['val'], sort_keys=True)}")
    print(f"Last-two tune val:       {json.dumps(last2_val, sort_keys=True)}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "objective": args.objective,
            "objective_weights": objective_weights,
            "best": best,
            "last2_train": last2_train,
            "last2_val": last2_val,
            "configured_train": configured_train,
            "configured_val": configured_val,
            "trials": trials,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
