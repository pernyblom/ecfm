from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forecasting.metrics import ade_fde_bbox_px, ade_fde_center_px, miou
from experiments.kalman_ml_forecasting.models.kalman_filter import (
    DEFAULT_KALMAN_CONFIG,
    kalman_config_from_dict,
    kalman_cv_forecast_tensor_params,
)
from experiments.kalman_ml_forecasting.models.kalman_residual import last_four_constant_velocity_forecast
from experiments.kalman_ml_forecasting.optimize_kalman import (
    _build_train_dataset,
    _format_yaml,
    _limit_indices,
    _objective_score,
    _parse_objective_weights,
    _split_indices_by_track,
    _stack_batch,
)
from experiments.kalman_ml_forecasting.utils.config import load_config


OPTIMIZED_PARAM_KEYS = [key for key in DEFAULT_KALMAN_CONFIG if key != "enabled"]


class KalmanStdParameters(nn.Module):
    def __init__(self, cfg: Dict[str, Any] | None, *, min_std: float, max_std: float, device) -> None:
        super().__init__()
        params = kalman_config_from_dict(cfg)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.log_std = nn.ParameterDict()
        for key in OPTIMIZED_PARAM_KEYS:
            value = min(max(float(params[key]), self.min_std), self.max_std)
            self.log_std[key] = nn.Parameter(torch.tensor(math.log(value), dtype=torch.float32, device=device))

    def tensors(self) -> dict[str, torch.Tensor]:
        return {key: self.log_std[key].exp() for key in OPTIMIZED_PARAM_KEYS}

    def as_config(self) -> dict[str, float | bool]:
        out: dict[str, float | bool] = {"enabled": True}
        with torch.no_grad():
            for key in OPTIMIZED_PARAM_KEYS:
                out[key] = float(self.log_std[key].exp().detach().cpu().item())
        return out

    @torch.no_grad()
    def clamp_(self) -> None:
        lo = math.log(self.min_std)
        hi = math.log(self.max_std)
        for param in self.log_std.values():
            param.clamp_(lo, hi)


def _metric_tensors(pred: torch.Tensor, target: torch.Tensor, frame_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    ade_bb, fde_bb = ade_fde_bbox_px(pred, target, frame_size)
    ade_c, fde_c = ade_fde_center_px(pred, target, frame_size)
    return {
        "ade_bbox_px": ade_bb,
        "fde_bbox_px": fde_bb,
        "ade_center_px": ade_c,
        "fde_center_px": fde_c,
        "miou": miou(pred, target, frame_size),
    }


def _weighted_loss(metrics: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    missing = [key for key in weights if key not in metrics]
    if missing:
        available = ", ".join(sorted(metrics))
        raise KeyError(f"Objective metric(s) missing: {missing}. Available metrics: {available}")
    loss = None
    for key, weight in weights.items():
        term = metrics[key] * float(weight)
        loss = term if loss is None else loss + term
    if loss is None:
        raise ValueError("objective weights cannot be empty.")
    return loss


def _mean_float_metrics(rows: list[dict[str, float]], weights: list[int]) -> dict[str, float]:
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
    model: KalmanStdParameters,
    frame_size: tuple[int, int],
    objective_weights: dict[str, float],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    rows: list[dict[str, float]] = []
    row_weights: list[int] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            past_boxes, future_boxes, past_times, future_times = _stack_batch(samples, batch_indices, device=device)
            pred = kalman_cv_forecast_tensor_params(past_boxes, past_times, future_times, model.tensors())
            metrics_t = _metric_tensors(pred, future_boxes, frame_size)
            rows.append({key: float(value.detach().cpu().item()) for key, value in metrics_t.items()})
            row_weights.append(len(batch_indices))
    metrics = _mean_float_metrics(rows, row_weights)
    return metrics, _objective_score(metrics, objective_weights)


def _evaluate_last_four(
    samples: list[dict],
    indices: list[int],
    *,
    frame_size: tuple[int, int],
    objective_weights: dict[str, float],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    rows: list[dict[str, float]] = []
    row_weights: list[int] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            past_boxes, future_boxes, past_times, future_times = _stack_batch(samples, batch_indices, device=device)
            pred = last_four_constant_velocity_forecast(past_boxes, past_times, future_times)
            metrics_t = _metric_tensors(pred, future_boxes, frame_size)
            rows.append({key: float(value.detach().cpu().item()) for key, value in metrics_t.items()})
            row_weights.append(len(batch_indices))
    metrics = _mean_float_metrics(rows, row_weights)
    return metrics, _objective_score(metrics, objective_weights)


def _train_epoch(
    samples: list[dict],
    indices: list[int],
    *,
    model: KalmanStdParameters,
    optimizer: torch.optim.Optimizer,
    frame_size: tuple[int, int],
    objective_weights: dict[str, float],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> float:
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(shuffled), batch_size):
        batch_indices = shuffled[start : start + batch_size]
        past_boxes, future_boxes, past_times, future_times = _stack_batch(samples, batch_indices, device=device)
        pred = kalman_cv_forecast_tensor_params(past_boxes, past_times, future_times, model.tensors())
        loss = _weighted_loss(_metric_tensors(pred, future_boxes, frame_size), objective_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.clamp_()
        total_loss += float(loss.detach().cpu().item()) * len(batch_indices)
        total_count += len(batch_indices)
    return total_loss / max(1, total_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backprop-optimize CV Kalman filter parameters from the configured values."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tune-val-fraction", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="fde_center_px")
    parser.add_argument("--maximize-objective", action="store_true")
    parser.add_argument("--objective-weights", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tune-train-samples", type=int, default=None)
    parser.add_argument("--max-tune-val-samples", type=int, default=None)
    parser.add_argument("--min-std", type=float, default=1.0e-6)
    parser.add_argument("--max-std", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
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
    print(f"Total train split Kalman tracklets: {len(dataset.samples)}")
    print(f"Total train split (folder, track_id) groups: {len({(str(s['folder']), int(s['track_id'])) for s in dataset.samples})}")
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

    last4_train_metrics, last4_train_score = _evaluate_last_four(
        dataset.samples,
        train_idx,
        frame_size=frame_size,
        objective_weights=objective_weights,
        batch_size=int(args.batch_size),
        device=device,
    )
    last4_val_metrics, last4_val_score = _evaluate_last_four(
        dataset.samples,
        val_idx,
        frame_size=frame_size,
        objective_weights=objective_weights,
        batch_size=int(args.batch_size),
        device=device,
    )
    print(f"Last-four train score: {last4_train_score:.6f} metrics={json.dumps(last4_train_metrics, sort_keys=True)}")
    print(f"Last-four val score:   {last4_val_score:.6f} metrics={json.dumps(last4_val_metrics, sort_keys=True)}")

    model = KalmanStdParameters(
        cfg.get("kalman"),
        min_std=float(args.min_std),
        max_std=float(args.max_std),
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    initial_train_metrics, initial_train_score = _evaluate(
        dataset.samples,
        train_idx,
        model=model,
        frame_size=frame_size,
        objective_weights=objective_weights,
        batch_size=int(args.batch_size),
        device=device,
    )
    initial_val_metrics, initial_val_score = _evaluate(
        dataset.samples,
        val_idx,
        model=model,
        frame_size=frame_size,
        objective_weights=objective_weights,
        batch_size=int(args.batch_size),
        device=device,
    )
    print(f"Initial train score: {initial_train_score:.6f} metrics={json.dumps(initial_train_metrics, sort_keys=True)}")
    print(f"Initial val score:   {initial_val_score:.6f} metrics={json.dumps(initial_val_metrics, sort_keys=True)}")

    best = {
        "epoch": -1,
        "params": model.as_config(),
        "train": initial_train_metrics,
        "val": initial_val_metrics,
        "train_score": initial_train_score,
        "val_score": initial_val_score,
    }
    history: list[dict[str, Any]] = []
    for epoch in range(int(args.epochs)):
        train_loss = _train_epoch(
            dataset.samples,
            train_idx,
            model=model,
            optimizer=optimizer,
            frame_size=frame_size,
            objective_weights=objective_weights,
            batch_size=int(args.batch_size),
            device=device,
            seed=int(args.seed) + epoch,
        )
        train_metrics, train_score = _evaluate(
            dataset.samples,
            train_idx,
            model=model,
            frame_size=frame_size,
            objective_weights=objective_weights,
            batch_size=int(args.batch_size),
            device=device,
        )
        val_metrics, val_score = _evaluate(
            dataset.samples,
            val_idx,
            model=model,
            frame_size=frame_size,
            objective_weights=objective_weights,
            batch_size=int(args.batch_size),
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
            "val": val_metrics,
            "train_score": train_score,
            "val_score": val_score,
            "params": model.as_config(),
        }
        history.append(row)
        marker = ""
        if val_score < float(best["val_score"]):
            best = row
            marker = " new_best"
        print(
            f"epoch {epoch:04d} loss={train_loss:.6f} "
            f"train_score={train_score:.6f} val_score={val_score:.6f}{marker} "
            f"val_metrics={json.dumps(val_metrics, sort_keys=True)}"
        )

    print("Best Kalman params:")
    print(_format_yaml(best["params"]))
    print(f"Best epoch: {best['epoch']}")
    print(f"Best tune train score: {best['train_score']:.6f}")
    print(f"Best tune val score:   {best['val_score']:.6f}")
    print(f"Best tune train metrics: {json.dumps(best['train'], sort_keys=True)}")
    print(f"Best tune val metrics:   {json.dumps(best['val'], sort_keys=True)}")
    print(f"Last-four tune train score: {last4_train_score:.6f}")
    print(f"Last-four tune val score:   {last4_val_score:.6f}")
    print(f"Last-four tune train metrics: {json.dumps(last4_train_metrics, sort_keys=True)}")
    print(f"Last-four tune val metrics:   {json.dumps(last4_val_metrics, sort_keys=True)}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "objective": args.objective,
                    "objective_weights": objective_weights,
                    "last4": {
                        "train": last4_train_metrics,
                        "val": last4_val_metrics,
                        "train_score": last4_train_score,
                        "val_score": last4_val_score,
                    },
                    "initial": {
                        "train": initial_train_metrics,
                        "val": initial_val_metrics,
                        "train_score": initial_train_score,
                        "val_score": initial_val_score,
                    },
                    "best": best,
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
