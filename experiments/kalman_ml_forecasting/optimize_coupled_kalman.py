from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forecasting.metrics import ade_fde_bbox_px, ade_fde_center_px, miou
from experiments.kalman_ml_forecasting.models.coupled_kalman_filter import CoupledBoxKalmanFilter
from experiments.kalman_ml_forecasting.models.kalman_residual import last_four_constant_velocity_forecast
from experiments.kalman_ml_forecasting.optimize_kalman import (
    _build_split_dataset,
    _build_train_dataset,
    _limit_indices,
    _objective_score,
    _parse_objective_weights,
    _split_indices_by_track,
    _stack_batch,
)
from experiments.kalman_ml_forecasting.utils.config import load_config


def _metric_tensors(
    prediction: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]
) -> dict[str, torch.Tensor]:
    ade_bbox, fde_bbox = ade_fde_bbox_px(prediction, target, frame_size)
    ade_center, fde_center = ade_fde_center_px(prediction, target, frame_size)
    return {
        "ade_bbox_px": ade_bbox,
        "fde_bbox_px": fde_bbox,
        "ade_center_px": ade_center,
        "fde_center_px": fde_center,
        "miou": miou(prediction, target, frame_size),
    }


def _weighted_loss(metrics: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    missing = [key for key in weights if key not in metrics]
    if missing:
        raise KeyError(f"Objective metric(s) missing: {missing}. Available: {sorted(metrics)}")
    return sum(metrics[key] * float(weight) for key, weight in weights.items())


def _mean_metrics(rows: list[dict[str, float]], counts: list[int]) -> dict[str, float]:
    total = float(sum(counts))
    return {
        key: sum(row[key] * count for row, count in zip(rows, counts)) / total
        for key in rows[0]
    }


def _evaluate(
    samples: list[dict],
    indices: list[int],
    *,
    model: CoupledBoxKalmanFilter | None,
    frame_size: tuple[int, int],
    objective_weights: dict[str, float],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    rows: list[dict[str, float]] = []
    counts: list[int] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            past, future, past_times, future_times = _stack_batch(samples, batch_indices, device=device)
            prediction = (
                model(past, past_times, future_times)
                if model is not None
                else last_four_constant_velocity_forecast(past, past_times, future_times)
            )
            tensors = _metric_tensors(prediction, future, frame_size)
            rows.append({key: float(value.detach().cpu()) for key, value in tensors.items()})
            counts.append(len(batch_indices))
    metrics = _mean_metrics(rows, counts)
    return metrics, _objective_score(metrics, objective_weights)


def _train_epoch(
    samples: list[dict],
    indices: list[int],
    *,
    model: CoupledBoxKalmanFilter,
    optimizer: torch.optim.Optimizer,
    frame_size: tuple[int, int],
    objective_weights: dict[str, float],
    batch_size: int,
    device: torch.device,
    seed: int,
    transition_l2: float,
    gradient_clip: float,
    max_dynamics_abs: float,
) -> float:
    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    cv_generator = torch.zeros((8, 8), device=device)
    cv_generator[:4, 4:] = torch.eye(4, device=device)
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(shuffled), batch_size):
        batch_indices = shuffled[start : start + batch_size]
        past, future, past_times, future_times = _stack_batch(samples, batch_indices, device=device)
        prediction = model(past, past_times, future_times)
        loss = _weighted_loss(_metric_tensors(prediction, future, frame_size), objective_weights)
        if transition_l2:
            loss = loss + float(transition_l2) * (model.dynamics_generator - cv_generator).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        optimizer.step()
        model.clamp_(max_dynamics_abs=max_dynamics_abs)
        total_loss += float(loss.detach().cpu()) * len(batch_indices)
        total_count += len(batch_indices)
    return total_loss / max(1, total_count)


def _median_dt(samples: list[dict], indices: list[int]) -> float:
    deltas: list[float] = []
    for index in indices:
        times = list(samples[index]["past_times_s"]) + list(samples[index]["future_times_s"])
        deltas.extend(float(b - a) for a, b in zip(times, times[1:]) if b > a)
    return float(torch.tensor(deltas).median()) if deltas else 1.0 / 30.0


def _new_model(args: argparse.Namespace, cfg: dict[str, Any], device: torch.device) -> CoupledBoxKalmanFilter:
    return CoupledBoxKalmanFilter(
        cfg.get("kalman"),
        optimize_noise=bool(args.optimize_noise),
        min_std=float(args.min_std),
        max_std=float(args.max_std),
        device=device,
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a fully coupled continuous-time box Kalman transition from FRED "
            "cleaned_tracks.txt tracklets."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tune-val-fraction", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="fde_center_px")
    parser.add_argument("--maximize-objective", action="store_true")
    parser.add_argument("--objective-weights", type=str, default=None)
    parser.add_argument("--optimize-noise", action="store_true")
    parser.add_argument("--transition-l2", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--max-dynamics-abs", type=float, default=20.0)
    parser.add_argument("--min-std", type=float, default=1.0e-6)
    parser.add_argument("--max-std", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tune-train-samples", type=int, default=None)
    parser.add_argument("--max-tune-val-samples", type=int, default=None)
    parser.add_argument("--run-test-on-best", action="store_true")
    parser.add_argument("--test-split-key", type=str, default="test")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--reference-dt", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.config)
    tracks_file = str(cfg["data"].get("tracks_file", "cleaned_tracks.txt"))
    if Path(tracks_file).name != "cleaned_tracks.txt":
        raise ValueError(
            "This experiment intentionally requires data.tracks_file: cleaned_tracks.txt; "
            f"got {tracks_file!r}."
        )
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    frame_size = tuple(int(value) for value in cfg["data"]["frame_size"])
    objective_weights = _parse_objective_weights(
        args.objective_weights, args.objective, args.maximize_objective
    )

    dataset = _build_train_dataset(cfg, max_samples=args.max_samples)
    if not dataset.samples:
        raise RuntimeError("Training dataset yielded zero cleaned-track Kalman samples.")
    train_indices, val_indices = _split_indices_by_track(
        dataset.samples, val_fraction=args.tune_val_fraction, seed=args.seed
    )
    train_indices = _limit_indices(
        train_indices, max_count=args.max_tune_train_samples, seed=args.seed + 1
    )
    val_indices = _limit_indices(val_indices, max_count=args.max_tune_val_samples, seed=args.seed + 2)
    if not train_indices or not val_indices:
        raise RuntimeError(f"Need non-empty train/validation sets; got {len(train_indices)}/{len(val_indices)}.")
    reference_dt = float(args.reference_dt) if args.reference_dt is not None else _median_dt(dataset.samples, train_indices)

    print(f"Track source: {tracks_file}")
    print(f"Tune samples: train={len(train_indices)} val={len(val_indices)}")
    print(f"Device: {device}; optimize_noise={args.optimize_noise}; reference_dt={reference_dt:.9g}s")
    print(f"Objective weights: {json.dumps(objective_weights, sort_keys=True)}")

    model = _new_model(args, cfg, device)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)
    initial_train, initial_train_score = _evaluate(
        dataset.samples, train_indices, model=model, frame_size=frame_size,
        objective_weights=objective_weights, batch_size=args.batch_size, device=device,
    )
    initial_val, initial_val_score = _evaluate(
        dataset.samples, val_indices, model=model, frame_size=frame_size,
        objective_weights=objective_weights, batch_size=args.batch_size, device=device,
    )
    last4_val, last4_val_score = _evaluate(
        dataset.samples, val_indices, model=None, frame_size=frame_size,
        objective_weights=objective_weights, batch_size=args.batch_size, device=device,
    )
    print(f"Initial train score={initial_train_score:.6f} metrics={json.dumps(initial_train, sort_keys=True)}")
    print(f"Initial val score={initial_val_score:.6f} metrics={json.dumps(initial_val, sort_keys=True)}")
    print(f"Last-four val score={last4_val_score:.6f} metrics={json.dumps(last4_val, sort_keys=True)}")

    best = {
        "epoch": -1, "train": initial_train, "val": initial_val,
        "train_score": initial_train_score, "val_score": initial_val_score,
        "model": model.snapshot(reference_dt=reference_dt),
        "state_dict": copy.deepcopy(model.state_dict()),
    }
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        train_loss = _train_epoch(
            dataset.samples, train_indices, model=model, optimizer=optimizer,
            frame_size=frame_size, objective_weights=objective_weights,
            batch_size=args.batch_size, device=device, seed=args.seed + epoch,
            transition_l2=args.transition_l2, gradient_clip=args.gradient_clip,
            max_dynamics_abs=args.max_dynamics_abs,
        )
        train_metrics, train_score = _evaluate(
            dataset.samples, train_indices, model=model, frame_size=frame_size,
            objective_weights=objective_weights, batch_size=args.batch_size, device=device,
        )
        val_metrics, val_score = _evaluate(
            dataset.samples, val_indices, model=model, frame_size=frame_size,
            objective_weights=objective_weights, batch_size=args.batch_size, device=device,
        )
        row = {
            "epoch": epoch, "train_loss": train_loss, "train": train_metrics, "val": val_metrics,
            "train_score": train_score, "val_score": val_score,
            "model": model.snapshot(reference_dt=reference_dt),
        }
        history.append(row)
        marker = ""
        if val_score < best["val_score"]:
            best = {**row, "state_dict": copy.deepcopy(model.state_dict())}
            marker = " new_best"
        print(
            f"epoch {epoch:04d} loss={train_loss:.6f} train_score={train_score:.6f} "
            f"val_score={val_score:.6f}{marker} val_metrics={json.dumps(val_metrics, sort_keys=True)}"
        )

    model.load_state_dict(best.pop("state_dict"))
    test_result = None
    if args.run_test_on_best:
        test_dataset = _build_split_dataset(cfg, split_key=args.test_split_key, max_samples=args.max_test_samples)
        test_indices = list(range(len(test_dataset.samples)))
        if not test_indices:
            raise RuntimeError(f"Configured {args.test_split_key!r} split yielded zero samples.")
        test_metrics, test_score = _evaluate(
            test_dataset.samples, test_indices, model=model, frame_size=frame_size,
            objective_weights=objective_weights, batch_size=args.batch_size, device=device,
        )
        last4_test, last4_test_score = _evaluate(
            test_dataset.samples, test_indices, model=None, frame_size=frame_size,
            objective_weights=objective_weights, batch_size=args.batch_size, device=device,
        )
        test_result = {
            "split": args.test_split_key, "samples": len(test_indices), "metrics": test_metrics,
            "score": test_score, "last4_metrics": last4_test, "last4_score": last4_test_score,
        }
        print(f"Best test score={test_score:.6f} metrics={json.dumps(test_metrics, sort_keys=True)}")

    result = {
        "track_source": tracks_file,
        "optimize_noise": bool(args.optimize_noise),
        "objective_weights": objective_weights,
        "initial": {"train": initial_train, "val": initial_val},
        "last4_val": {"metrics": last4_val, "score": last4_val_score},
        "best": best,
        "test": test_result,
        "history": history,
    }
    print(f"Best epoch: {best['epoch']}; val score: {best['val_score']:.6f}")
    print("Best reference transition matrix:")
    print(json.dumps(best["model"]["reference_transition"], indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
