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

from experiments.object_detection.data.dataset import DetectionSample, FredDetectionDataset
from experiments.object_detection.losses import compute_losses
from experiments.object_detection.metrics import (
    detection_scores,
    map_metrics,
    summarize_metrics,
)
from experiments.object_detection.models.factory import build_model
from experiments.object_detection.utils.config import load_config
from experiments.object_detection.visualization import save_sample_visualization


def _read_split_file(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(line.strip("/"))
    return out


def _collate(batch: List[DetectionSample]):
    reps = batch[0].inputs.keys()
    heatmap_reps = batch[0].heatmaps.keys()
    return type(
        "Batch",
        (),
        {
            "inputs": {rep: torch.stack([b.inputs[rep] for b in batch], dim=0) for rep in reps},
            "box_xywh": torch.stack([b.box_xywh for b in batch], dim=0),
            "heatmaps": {
                rep: torch.stack([b.heatmaps[rep] for b in batch], dim=0) for rep in heatmap_reps
            },
            "frame_keys": [b.frame_key for b in batch],
            "frame_times_s": [b.frame_time_s for b in batch],
            "selected_box_index": [b.selected_box_index for b in batch],
            "all_boxes_xywh": [b.all_boxes_xywh for b in batch],
            "input_paths": [b.input_paths for b in batch],
        },
    )


def _build_dataset(cfg: Dict, split: str) -> FredDetectionDataset:
    data_cfg = cfg["data"]
    split_files = data_cfg.get("split_files")
    folders = None
    if split_files:
        folders = _read_split_file(Path(split_files[split]))
    max_samples = data_cfg.get(f"max_samples_{split}", data_cfg.get("max_samples"))
    return FredDetectionDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        representations=list(data_cfg["representations"]),
        heatmap_representations=list(data_cfg.get("heatmap_representations", [])),
        image_size=tuple(data_cfg["image_size"]),
        frame_size=tuple(data_cfg["frame_size"]),
        folders=folders,
        labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
        label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
        image_window_ms=float(data_cfg.get("image_window_ms", 33.333)),
        image_window_mode=data_cfg.get("image_window_mode", "trailing"),
        verify_render_manifest=bool(data_cfg.get("verify_render_manifest", True)),
        render_manifest_name=data_cfg.get("render_manifest_name", "render_manifest.json"),
        window_tolerance_ms=float(data_cfg.get("window_tolerance_ms", 2.0)),
        require_boxes=bool(data_cfg.get("require_boxes", True)),
        select_box=data_cfg.get("select_box", "largest"),
        max_samples=max_samples,
        seed=int(data_cfg.get("seed", 123)),
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
    )


def _mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def _run_epoch(*, model, loader, device: torch.device, optimizer, cfg: Dict, train: bool) -> Dict[str, float]:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model.train(mode=train)
    rows: List[Dict[str, float]] = []
    pred_boxes_all = []
    pred_scores_all = []
    target_boxes_all = []
    gt_present_all = []
    for step, batch in enumerate(loader):
        inputs = {k: v.to(device) for k, v in batch.inputs.items()}
        target_boxes = batch.box_xywh.to(device)
        target_heatmaps = {k: v.to(device) for k, v in batch.heatmaps.items()}
        gt_present = torch.tensor(
            [idx >= 0 for idx in batch.selected_box_index],
            device=device,
            dtype=torch.bool,
        )
        target_objectness = gt_present.float()
        with torch.set_grad_enabled(train):
            preds = model(inputs)
            loss, loss_metrics = compute_losses(
                preds,
                target_boxes,
                target_heatmaps,
                target_objectness,
                heatmap_weight=float(train_cfg.get("heatmap_weight", 1.0)),
                box_weight=float(train_cfg.get("box_weight", 1.0)),
                objectness_weight=float(train_cfg.get("objectness_weight", 1.0)),
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        metrics = summarize_metrics(
            preds,
            target_boxes,
            target_heatmaps,
            gt_present,
            tuple(data_cfg["frame_size"]),
        )
        rows.append({**loss_metrics, **metrics})
        pred_boxes_all.append(preds["boxes"].detach().cpu())
        pred_scores_all.append(detection_scores(preds).detach().cpu())
        target_boxes_all.append(target_boxes.detach().cpu())
        gt_present_all.append(gt_present.detach().cpu())
        if step % int(train_cfg.get("log_every", 20)) == 0:
            phase = "train" if train else "val"
            print(
                f"{phase} step {step} loss {loss_metrics['loss']:.4f} "
                f"box_l1 {loss_metrics['box_l1']:.4f} "
                f"obj_bce {loss_metrics['objectness_bce']:.4f} "
                f"center_l1_px {metrics['center_l1_px']:.2f} "
                f"box_iou {metrics['box_iou']:.4f} "
                f"mAP_50 {metrics['mAP_50']:.4f} "
                f"mAP_50:95 {metrics['mAP_50_95']:.4f}"
            )
    out = _mean_dict(rows)
    if pred_boxes_all:
        out.update(
            map_metrics(
                pred_boxes=torch.cat(pred_boxes_all, dim=0),
                pred_scores=torch.cat(pred_scores_all, dim=0),
                target_boxes=torch.cat(target_boxes_all, dim=0),
                gt_present=torch.cat(gt_present_all, dim=0),
                frame_size=tuple(data_cfg["frame_size"]),
            )
        )
    return out


@torch.no_grad()
def _export_visualizations(model, loader, device: torch.device, cfg: Dict, epoch: int) -> None:
    train_cfg = cfg["train"]
    vis_every = int(train_cfg.get("vis_every", 0))
    if vis_every <= 0 or (epoch + 1) % vis_every != 0:
        return
    batch = next(iter(loader), None)
    if batch is None:
        return
    inputs = {k: v.to(device) for k, v in batch.inputs.items()}
    preds = model(inputs)
    output_dir = Path(train_cfg.get("vis_output_dir", "outputs/object_detection_vis")) / f"epoch_{epoch:03d}"
    max_samples = min(int(train_cfg.get("vis_samples", 8)), batch.box_xywh.shape[0])
    backdrop_rep = str(train_cfg.get("vis_backdrop_rep", "cstr3"))
    for i in range(max_samples):
        stem = batch.frame_keys[i].replace("/", "_")
        save_sample_visualization(
            output_dir=output_dir,
            stem=stem,
            inputs={rep: batch.inputs[rep][i] for rep in batch.inputs},
            pred_boxes=preds["boxes"][i].cpu(),
            target_box=batch.box_xywh[i].cpu(),
            pred_heatmaps={rep: preds["heatmaps"][rep][i].cpu() for rep in preds.get("heatmaps", {})},
            target_heatmaps={rep: batch.heatmaps[rep][i].cpu() for rep in batch.heatmaps if rep in preds.get("heatmaps", {})},
            xy_backdrop_rep=backdrop_rep,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_set = _build_dataset(cfg, "train")
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

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=_collate,
    )

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/object_detection_ckpt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = Path(train_cfg.get("metrics_jsonl", ckpt_dir / "metrics.jsonl"))

    best_val = None
    for epoch in range(int(train_cfg["epochs"])):
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
        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch, "best_val": best_val, "config": cfg},
                ckpt_dir / "best.pt",
            )
        ckpt_every = int(train_cfg.get("checkpoint_every", 1))
        if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch, "best_val": best_val, "config": cfg},
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )
        _export_visualizations(model, val_loader, device, cfg, epoch)


if __name__ == "__main__":
    main()
