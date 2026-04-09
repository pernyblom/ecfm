from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forecasting.metrics import ade_fde_bbox_px, ade_fde_center_px, miou
from experiments.leworldmodel.data.dataset import (
    LeWorldModelSample,
    LeWorldModelTrackDataset,
)
from experiments.leworldmodel.losses import compute_losses
from experiments.leworldmodel.models.factory import build_model
from experiments.leworldmodel.utils.config import load_config


def _collate_samples(batch: List[LeWorldModelSample]):
    if not batch:
        return batch
    reps = batch[0].inputs.keys()
    inputs = {rep: torch.stack([sample.inputs[rep] for sample in batch], dim=0) for rep in reps}
    past_boxes = torch.stack([sample.past_boxes for sample in batch], dim=0)
    future_boxes = torch.stack([sample.future_boxes for sample in batch], dim=0)
    image_frame_keys = [sample.image_frame_keys for sample in batch]
    box_frame_keys = [sample.box_frame_keys for sample in batch]
    image_frame_times = [sample.image_frame_times for sample in batch]
    box_frame_times = [sample.box_frame_times for sample in batch]
    anchor_frame_keys = [sample.anchor_frame_key for sample in batch]
    track_ids = [sample.track_id for sample in batch]
    return type(
        "Batch",
        (),
        {
            "inputs": inputs,
            "past_boxes": past_boxes,
            "future_boxes": future_boxes,
            "image_frame_keys": image_frame_keys,
            "box_frame_keys": box_frame_keys,
            "image_frame_times": image_frame_times,
            "box_frame_times": box_frame_times,
            "anchor_frame_keys": anchor_frame_keys,
            "track_ids": track_ids,
        },
    )


def _read_split_file(path: Path) -> List[str]:
    items: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(line.strip("/"))
    return items


def _build_dataset(data_cfg: Dict, folders: List[str], *, max_samples_key: str, seed_offset: int):
    return LeWorldModelTrackDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        representations=data_cfg["representations"],
        ssl_context_steps=int(data_cfg["ssl_context_steps"]),
        ssl_future_steps=int(data_cfg["ssl_future_steps"]),
        ssl_future_offset_steps=int(data_cfg.get("ssl_future_offset_steps", 1)),
        forecast_history_steps=int(data_cfg["forecast_history_steps"]),
        forecast_future_steps=int(data_cfg["forecast_future_steps"]),
        stride=int(data_cfg.get("stride", 1)),
        image_size=tuple(data_cfg["image_size"]),
        folders=folders,
        labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
        tracks_file=data_cfg.get("tracks_file", "cleaned_tracks.txt"),
        label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
        track_time_unit=float(data_cfg.get("track_time_unit", 1.0)),
        time_align=data_cfg.get("time_align", "start"),
        frame_size=tuple(data_cfg["frame_size"]) if data_cfg.get("frame_size") else None,
        max_frame_gap_s=data_cfg.get("max_frame_gap_s", 0.05),
        max_tracks=data_cfg.get("max_tracks"),
        max_samples=data_cfg.get(max_samples_key),
        seed=int(data_cfg.get("seed", 123)) + seed_offset,
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
    )


def _empty_metric_accumulator() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "pred_loss": 0.0,
        "reg_loss": 0.0,
        "box_loss": 0.0,
        "latent_std": 0.0,
        "latent_norm": 0.0,
        "ade_bbox_px": 0.0,
        "fde_bbox_px": 0.0,
        "ade_center_px": 0.0,
        "fde_center_px": 0.0,
        "miou": 0.0,
        "count": 0.0,
    }


def _update_forecast_metrics(metrics: Dict[str, float], pred_boxes: torch.Tensor, future_boxes: torch.Tensor, frame_size):
    ade_bb, fde_bb = ade_fde_bbox_px(pred_boxes, future_boxes, frame_size)
    ade_c, fde_c = ade_fde_center_px(pred_boxes, future_boxes, frame_size)
    miou_val = miou(pred_boxes, future_boxes, frame_size)
    metrics["ade_bbox_px"] += float(ade_bb.item())
    metrics["fde_bbox_px"] += float(fde_bb.item())
    metrics["ade_center_px"] += float(ade_c.item())
    metrics["fde_center_px"] += float(fde_c.item())
    metrics["miou"] += float(miou_val.item())


def _finalize_metrics(sums: Dict[str, float]) -> Dict[str, float]:
    count = max(int(sums["count"]), 1)
    out = {}
    for key, value in sums.items():
        if key == "count":
            continue
        out[key] = value / count
    return out


def _append_metrics_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _run_epoch(model, loader, optimizer, device, train_cfg, reg_cfg, frame_size, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    sums = _empty_metric_accumulator()
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            inputs = {k: v.to(device) for k, v in batch.inputs.items()}
            past_boxes = batch.past_boxes.to(device)
            future_boxes = batch.future_boxes.to(device)

            outputs = model(inputs, past_boxes=past_boxes)
            loss, metrics = compute_losses(outputs, future_boxes, train_cfg, reg_cfg)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for key in ("loss", "pred_loss", "reg_loss", "box_loss", "latent_std", "latent_norm"):
                sums[key] += metrics[key]
            if "pred_future_boxes" in outputs:
                _update_forecast_metrics(sums, outputs["pred_future_boxes"].detach(), future_boxes, frame_size)
            sums["count"] += 1.0

            if train and step % int(train_cfg.get("log_every", 20)) == 0:
                msg = (
                    f"step {step} loss {metrics['loss']:.5f} pred {metrics['pred_loss']:.5f} "
                    f"reg {metrics['reg_loss']:.5f} box {metrics['box_loss']:.5f} "
                    f"lat_std {metrics['latent_std']:.5f}"
                )
                if "pred_future_boxes" in outputs:
                    ade_bb, fde_bb = ade_fde_bbox_px(outputs["pred_future_boxes"].detach(), future_boxes, frame_size)
                    msg += f" ADE_BB {ade_bb.item():.2f} FDE_BB {fde_bb.item():.2f}"
                print(msg)
    return _finalize_metrics(sums)


def build_loaders(cfg: Dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    split_files = data_cfg.get("split_files")
    if not split_files:
        raise ValueError("data.split_files is required.")
    train_folders = _read_split_file(Path(split_files["train"]))
    val_folders = _read_split_file(Path(split_files["val"]))

    train_set = _build_dataset(data_cfg, train_folders, max_samples_key="max_samples_train", seed_offset=0)
    val_set = _build_dataset(data_cfg, val_folders, max_samples_key="max_samples_val", seed_offset=1)
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=_collate_samples,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=_collate_samples,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    reg_cfg = cfg.get("regularizer", {"type": "sigreg"})
    frame_size = tuple(int(v) for v in data_cfg["frame_size"])

    train_loader, val_loader = build_loaders(cfg)

    device = torch.device(train_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/leworldmodel_ckpt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(train_cfg.get("metrics_jsonl", ckpt_dir / "metrics.jsonl"))
    ckpt_every = int(train_cfg.get("checkpoint_every", 1))
    best_val = None
    start_epoch = 0

    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optim"])
            start_epoch = int(state.get("epoch", 0)) + 1
            best_val = state.get("best_val")
            print(f"Resumed checkpoint {resume_path} at epoch {start_epoch}")

    epochs = int(train_cfg["epochs"])
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}/{epochs - 1}")
        train_metrics = _run_epoch(
            model, train_loader, optimizer, device, train_cfg, reg_cfg, frame_size, train=True
        )
        val_metrics = _run_epoch(
            model, val_loader, optimizer, device, train_cfg, reg_cfg, frame_size, train=False
        )
        print(
            "train "
            f"loss {train_metrics['loss']:.5f} pred {train_metrics['pred_loss']:.5f} "
            f"reg {train_metrics['reg_loss']:.5f} box {train_metrics['box_loss']:.5f} "
            f"lat_std {train_metrics['latent_std']:.5f}"
        )
        print(
            "val   "
            f"loss {val_metrics['loss']:.5f} pred {val_metrics['pred_loss']:.5f} "
            f"reg {val_metrics['reg_loss']:.5f} box {val_metrics['box_loss']:.5f} "
            f"lat_std {val_metrics['latent_std']:.5f}"
        )
        if cfg.get("downstream", {}).get("forecasting", {}).get("enabled", False):
            print(
                "val   "
                f"ADE_BB {val_metrics['ade_bbox_px']:.2f} FDE_BB {val_metrics['fde_bbox_px']:.2f} "
                f"ADE_C {val_metrics['ade_center_px']:.2f} FDE_C {val_metrics['fde_center_px']:.2f} "
                f"mIoU {val_metrics['miou']:.4f}"
            )

        _append_metrics_jsonl(
            metrics_path,
            {"epoch": epoch, "train": train_metrics, "val": val_metrics},
        )

        val_loss = val_metrics["loss"]
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )

        if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "config": cfg,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )


if __name__ == "__main__":
    main()
