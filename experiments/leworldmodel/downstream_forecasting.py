from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forecasting.metrics import ade_fde_bbox_px, ade_fde_center_px, miou
from experiments.leworldmodel.models.factory import build_model
from experiments.leworldmodel.train import build_loaders
from experiments.leworldmodel.utils.config import load_config


def _append_metrics_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _run_epoch(model, loader, optimizer, device, frame_size, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    sums = {
        "loss": 0.0,
        "ade_bbox_px": 0.0,
        "fde_bbox_px": 0.0,
        "ade_center_px": 0.0,
        "fde_center_px": 0.0,
        "miou": 0.0,
        "count": 0.0,
    }
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.inputs.items()}
            past_boxes = batch.past_boxes.to(device)
            future_boxes = batch.future_boxes.to(device)
            outputs = model(inputs, past_boxes=past_boxes)
            pred_boxes = outputs["pred_future_boxes"]
            loss = F.l1_loss(pred_boxes, future_boxes)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            ade_bb, fde_bb = ade_fde_bbox_px(pred_boxes.detach(), future_boxes, frame_size)
            ade_c, fde_c = ade_fde_center_px(pred_boxes.detach(), future_boxes, frame_size)
            miou_val = miou(pred_boxes.detach(), future_boxes, frame_size)
            sums["loss"] += float(loss.item())
            sums["ade_bbox_px"] += float(ade_bb.item())
            sums["fde_bbox_px"] += float(fde_bb.item())
            sums["ade_center_px"] += float(ade_c.item())
            sums["fde_center_px"] += float(fde_c.item())
            sums["miou"] += float(miou_val.item())
            sums["count"] += 1.0
    count = max(int(sums["count"]), 1)
    return {k: v / count for k, v in sums.items() if k != "count"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("downstream", {}).setdefault("forecasting", {})["enabled"] = True
    forecast_cfg = cfg["downstream"]["forecasting"]
    if not bool(forecast_cfg.get("use_ssl_features", True)) and not bool(
        forecast_cfg.get("use_history_boxes", True)
    ):
        raise ValueError("downstream.forecasting must enable at least one of use_ssl_features/use_history_boxes.")
    train_cfg = cfg["train"]
    frame_size = tuple(int(v) for v in cfg["data"]["frame_size"])

    train_loader, val_loader = build_loaders(cfg)
    device = torch.device(train_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)

    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        print(f"Unexpected keys ignored: {unexpected}")
    if missing:
        print(f"Missing keys allowed: {missing}")

    if bool(forecast_cfg.get("freeze_ssl_backbone", True)):
        for name, param in model.named_parameters():
            if not name.startswith("forecast_head."):
                param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(forecast_cfg.get("lr", train_cfg.get("lr", 2.0e-4))),
        weight_decay=float(forecast_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0))),
    )

    ckpt_dir = Path(forecast_cfg.get("checkpoint_dir", "outputs/leworldmodel_forecast_ckpt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(forecast_cfg.get("metrics_jsonl", ckpt_dir / "metrics.jsonl"))
    best_val = None

    epochs = int(forecast_cfg.get("epochs", 20))
    for epoch in range(epochs):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, frame_size, train=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, frame_size, train=False)
        print(
            f"epoch {epoch} "
            f"train loss {train_metrics['loss']:.5f} "
            f"val loss {val_metrics['loss']:.5f} "
            f"val ADE_BB {val_metrics['ade_bbox_px']:.2f} "
            f"val FDE_BB {val_metrics['fde_bbox_px']:.2f} "
            f"val ADE_C {val_metrics['ade_center_px']:.2f} "
            f"val FDE_C {val_metrics['fde_center_px']:.2f} "
            f"val mIoU {val_metrics['miou']:.4f}"
        )
        _append_metrics_jsonl(metrics_path, {"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if best_val is None or val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "config": cfg,
                    "ssl_checkpoint": str(args.checkpoint),
                },
                ckpt_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
