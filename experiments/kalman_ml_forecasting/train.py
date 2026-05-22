from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import (
    KalmanForecastSample,
    TrackKalmanForecastDataset,
)
from experiments.kalman_ml_forecasting.metrics import summarize_forecast_metrics
from experiments.kalman_ml_forecasting.models.factory import build_model
from experiments.kalman_ml_forecasting.models.kalman_filter import kalman_cv_forecast, kalman_config_from_dict
from experiments.kalman_ml_forecasting.utils.config import (
    load_config,
    read_split_file,
    resolve_representation_image_sizes,
)


@dataclass
class Batch:
    inputs: Dict[str, torch.Tensor]
    past_boxes: torch.Tensor
    future_boxes: torch.Tensor
    past_times_s: torch.Tensor
    future_times_s: torch.Tensor
    frame_keys: List[str]
    frame_times_s: List[float]
    track_ids: List[int]
    input_paths: List[Dict[str, str]]


def _collate(batch: List[KalmanForecastSample]) -> Batch:
    reps = batch[0].inputs.keys()
    return Batch(
        inputs={rep: torch.stack([b.inputs[rep] for b in batch], dim=0) for rep in reps},
        past_boxes=torch.stack([b.past_boxes for b in batch], dim=0),
        future_boxes=torch.stack([b.future_boxes for b in batch], dim=0),
        past_times_s=torch.stack([b.past_times_s for b in batch], dim=0),
        future_times_s=torch.stack([b.future_times_s for b in batch], dim=0),
        frame_keys=[b.frame_key for b in batch],
        frame_times_s=[b.frame_time_s for b in batch],
        track_ids=[b.track_id for b in batch],
        input_paths=[b.input_paths for b in batch],
    )


def _build_dataset(cfg: Dict, split: str) -> TrackKalmanForecastDataset:
    data_cfg = cfg["data"]
    split_files = data_cfg.get("split_files")
    folders = read_split_file(Path(split_files[split])) if split_files else None
    max_samples = data_cfg.get(f"max_samples_{split}", data_cfg.get("max_samples"))
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
        verify_render_manifest=bool(data_cfg.get("verify_render_manifest", True)),
        render_manifest_name=data_cfg.get("render_manifest_name", "render_manifest.json"),
        window_tolerance_ms=float(data_cfg.get("window_tolerance_ms", 5.0)),
        label_period_s=data_cfg.get("label_period_s"),
        max_tracks=data_cfg.get(f"max_tracks_{split}", data_cfg.get("max_tracks")),
        max_samples=max_samples,
        seed=int(data_cfg.get("seed", 123)) + (0 if split == "train" else 1),
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
        filter_missing_representations=bool(data_cfg.get("filter_missing_representations", True)),
    )


def _make_loader(dataset, *, batch_size: int, shuffle: bool, train_cfg: Dict) -> DataLoader:
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": _collate,
        "pin_memory": bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        if train_cfg.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(train_cfg["prefetch_factor"])
    return DataLoader(**kwargs)


def _mean_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row[key] for row in rows if key in row) / len(rows) for key in keys}


def _run_epoch(*, model, loader, device: torch.device, optimizer, cfg: Dict, train: bool) -> Dict[str, float]:
    train_cfg = cfg["train"]
    frame_size = tuple(cfg["data"]["frame_size"])
    loss_fn = nn.SmoothL1Loss(beta=float(train_cfg.get("smooth_l1_beta", 0.05)))
    residual_weight = float(train_cfg.get("residual_l2_weight", 0.0))
    kalman_cfg = kalman_config_from_dict(cfg.get("kalman"))
    model.train(mode=train)
    rows: List[Dict[str, float]] = []
    for step, batch in enumerate(loader):
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.inputs.items()}
        past_boxes = batch.past_boxes.to(device, non_blocking=True)
        future_boxes = batch.future_boxes.to(device, non_blocking=True)
        past_times_s = batch.past_times_s.to(device, non_blocking=True)
        future_times_s = batch.future_times_s.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(
                inputs,
                past_boxes,
                past_times_s,
                future_times_s,
                return_debug=True,
            )
            pred = out["boxes"]
            loss = loss_fn(pred, future_boxes)
            if residual_weight > 0:
                loss = loss + residual_weight * out["residual_accel"].square().mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(train_cfg.get("grad_clip_norm", 10.0)),
                )
                optimizer.step()
        with torch.no_grad():
            metrics = summarize_forecast_metrics(pred.detach(), future_boxes, frame_size)
            kalman_boxes = kalman_cv_forecast(
                past_boxes,
                past_times_s,
                future_times_s,
                kalman_cfg,
            )
            kalman_metrics = summarize_forecast_metrics(kalman_boxes.detach(), future_boxes, frame_size)
            last4_metrics = summarize_forecast_metrics(out["last4_boxes"].detach(), future_boxes, frame_size)
            row = {"loss": float(loss.item()), **metrics}
            row.update({f"kalman_{key}": value for key, value in kalman_metrics.items()})
            row.update({f"last4_{key}": value for key, value in last4_metrics.items()})
            rows.append(row)
        if step % int(train_cfg.get("log_every", 20)) == 0:
            phase = "train" if train else "val"
            print(
                f"{phase} step {step} loss {row['loss']:.4f} "
                f"ADE_C {row['ade_center_px']:.2f} FDE_C {row['fde_center_px']:.2f} "
                f"mIoU {row['miou']:.4f} "
                f"KALMAN_ADE_C {row['kalman_ade_center_px']:.2f} "
                f"LAST4_ADE_C {row['last4_ade_center_px']:.2f}"
            )
    return _mean_rows(rows)


def _save_checkpoint(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Building train dataset...")
    train_set = _build_dataset(cfg, "train")
    print("Building val dataset...")
    val_set = _build_dataset(cfg, "val")
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    train_cfg = cfg["train"]
    device = torch.device(train_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    start_epoch = 0
    best_val = None
    resume = args.resume_checkpoint or train_cfg.get("resume_checkpoint")
    if resume:
        state = torch.load(Path(resume), map_location=device)
        model.load_state_dict(state["model"])
        if "optim" in state:
            optimizer.load_state_dict(state["optim"])
        start_epoch = int(state.get("epoch", -1)) + 1
        best_val = state.get("best_val")
        print(f"Resumed {resume} at epoch {start_epoch}")

    train_loader = _make_loader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        train_cfg=train_cfg,
    )
    val_loader = _make_loader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        train_cfg=train_cfg,
    )

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/kalman_ml_forecasting_ckpt"))
    metrics_jsonl = Path(train_cfg.get("metrics_jsonl", ckpt_dir / "metrics.jsonl"))
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        print(f"Epoch {epoch}/{int(train_cfg['epochs']) - 1}")
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            cfg=cfg,
            train=True,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=optimizer,
                cfg=cfg,
                train=False,
            )
        print(f"train {json.dumps(train_metrics, sort_keys=True)}")
        print(f"val   {json.dumps(val_metrics, sort_keys=True)}")
        metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n")

        val_loss = val_metrics.get("loss")
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "config": cfg,
        }
        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            state["best_val"] = best_val
            _save_checkpoint(ckpt_dir / "best.pt", state)
        ckpt_every = int(train_cfg.get("checkpoint_every", 1))
        if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
            _save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", state)


if __name__ == "__main__":
    main()
