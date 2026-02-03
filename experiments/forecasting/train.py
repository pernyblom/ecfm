from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.forecasting.data.dataset import ForecastDataset, split_dataset
from experiments.forecasting.data.track_dataset import TrackForecastDataset
from experiments.forecasting.metrics import ade_fde_bbox_px, ade_fde_center_px, miou
from experiments.forecasting.models.fusion import MultiRepForecast
from experiments.forecasting.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    frame_size = data_cfg.get("frame_size")
    if frame_size is None:
        raise ValueError("data.frame_size must be set for pixel-based metrics.")
    frame_size_t = (int(frame_size[0]), int(frame_size[1]))

    def _read_split_file(path: Path) -> list[str]:
        items: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(line.strip("/"))
        return items

    split_files = data_cfg.get("split_files")
    supervision = data_cfg.get("supervision", "yolo")
    if split_files:
        train_folders = _read_split_file(Path(split_files["train"]))
        val_folders = _read_split_file(Path(split_files["val"]))
        if supervision == "tracks":
            train_set = TrackForecastDataset(
                images_root=Path(data_cfg["images_root"]),
                labels_root=Path(data_cfg["labels_root"]),
                representations=data_cfg["representations"],
                past_steps=data_cfg["past_steps"],
                future_steps=data_cfg["future_steps"],
                stride=data_cfg["stride"],
                image_size=tuple(data_cfg["image_size"]),
                folders=train_folders,
                labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
                tracks_file=data_cfg.get("tracks_file", "tracks.txt"),
                label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
                track_time_unit=float(data_cfg.get("track_time_unit", 1.0)),
                time_align=data_cfg.get("time_align", "start"),
                frame_size=tuple(data_cfg["frame_size"]) if data_cfg.get("frame_size") else None,
            )
            val_set = TrackForecastDataset(
                images_root=Path(data_cfg["images_root"]),
                labels_root=Path(data_cfg["labels_root"]),
                representations=data_cfg["representations"],
                past_steps=data_cfg["past_steps"],
                future_steps=data_cfg["future_steps"],
                stride=data_cfg["stride"],
                image_size=tuple(data_cfg["image_size"]),
                folders=val_folders,
                labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
                tracks_file=data_cfg.get("tracks_file", "tracks.txt"),
                label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
                track_time_unit=float(data_cfg.get("track_time_unit", 1.0)),
                time_align=data_cfg.get("time_align", "start"),
                frame_size=tuple(data_cfg["frame_size"]) if data_cfg.get("frame_size") else None,
            )
        else:
            train_set = ForecastDataset(
                images_root=Path(data_cfg["images_root"]),
                labels_root=Path(data_cfg["labels_root"]),
                representations=data_cfg["representations"],
                past_steps=data_cfg["past_steps"],
                future_steps=data_cfg["future_steps"],
                stride=data_cfg["stride"],
                image_size=tuple(data_cfg["image_size"]),
                select_box=data_cfg.get("select_box", "largest"),
                drop_empty=bool(data_cfg.get("drop_empty", True)),
                folders=train_folders,
                labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
            )
            val_set = ForecastDataset(
                images_root=Path(data_cfg["images_root"]),
                labels_root=Path(data_cfg["labels_root"]),
                representations=data_cfg["representations"],
                past_steps=data_cfg["past_steps"],
                future_steps=data_cfg["future_steps"],
                stride=data_cfg["stride"],
                image_size=tuple(data_cfg["image_size"]),
                select_box=data_cfg.get("select_box", "largest"),
                drop_empty=bool(data_cfg.get("drop_empty", True)),
                folders=val_folders,
                labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
            )
    else:
        if supervision == "tracks":
            dataset = TrackForecastDataset(
                images_root=Path(data_cfg["images_root"]),
                labels_root=Path(data_cfg["labels_root"]),
                representations=data_cfg["representations"],
                past_steps=data_cfg["past_steps"],
                future_steps=data_cfg["future_steps"],
                stride=data_cfg["stride"],
                image_size=tuple(data_cfg["image_size"]),
                labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
                tracks_file=data_cfg.get("tracks_file", "tracks.txt"),
                label_time_unit=float(data_cfg.get("label_time_unit", 1e-6)),
                track_time_unit=float(data_cfg.get("track_time_unit", 1.0)),
                time_align=data_cfg.get("time_align", "start"),
                frame_size=tuple(data_cfg["frame_size"]) if data_cfg.get("frame_size") else None,
            )
        else:
            dataset = ForecastDataset(
                images_root=Path(data_cfg["images_root"]),
                labels_root=Path(data_cfg["labels_root"]),
                representations=data_cfg["representations"],
                past_steps=data_cfg["past_steps"],
                future_steps=data_cfg["future_steps"],
                stride=data_cfg["stride"],
                image_size=tuple(data_cfg["image_size"]),
                select_box=data_cfg.get("select_box", "largest"),
                drop_empty=bool(data_cfg.get("drop_empty", True)),
            )
        train_set, val_set = split_dataset(
            dataset, train_split=data_cfg["train_split"], seed=data_cfg["seed"]
        )

    train_loader = DataLoader(
        train_set, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0
    )

    device = torch.device(
        train_cfg.get("device", "cpu")
        if torch.cuda.is_available()
        else "cpu"
    )
    model = MultiRepForecast(
        reps=data_cfg["representations"],
        cnn_channels=model_cfg["cnn_channels"],
        feature_dim=model_cfg["feature_dim"],
        use_past_boxes=model_cfg["use_past_boxes"],
        rnn_hidden=model_cfg["rnn_hidden"],
        rnn_layers=model_cfg["rnn_layers"],
        future_steps=data_cfg["future_steps"],
    ).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    loss_fn = nn.L1Loss()

    for epoch in range(train_cfg["epochs"]):
        model.train()
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.inputs.items()}
            past_boxes = batch.past_boxes.to(device)
            future = batch.future_boxes.to(device)

            pred = model(inputs, past_boxes)
            loss = loss_fn(pred, future)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if step % train_cfg["log_every"] == 0:
                ade_bb, fde_bb = ade_fde_bbox_px(pred.detach(), future, frame_size_t)
                ade_c, fde_c = ade_fde_center_px(pred.detach(), future, frame_size_t)
                miou_val = miou(pred.detach(), future, frame_size_t)
                print(
                    f"epoch {epoch} step {step} loss {loss.item():.4f} "
                    f"ADE_BB {ade_bb.item():.2f} FDE_BB {fde_bb.item():.2f} "
                    f"ADE_C {ade_c.item():.2f} FDE_C {fde_c.item():.2f} "
                    f"mIoU {miou_val.item():.4f}"
                )

        model.eval()
        with torch.no_grad():
            losses = []
            ade_bb_vals = []
            fde_bb_vals = []
            ade_c_vals = []
            fde_c_vals = []
            miou_vals = []
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.inputs.items()}
                past_boxes = batch.past_boxes.to(device)
                future = batch.future_boxes.to(device)
                pred = model(inputs, past_boxes)
                loss = loss_fn(pred, future)
                ade_bb, fde_bb = ade_fde_bbox_px(pred, future, frame_size_t)
                ade_c, fde_c = ade_fde_center_px(pred, future, frame_size_t)
                miou_val = miou(pred, future, frame_size_t)
                losses.append(loss.item())
                ade_bb_vals.append(ade_bb.item())
                fde_bb_vals.append(fde_bb.item())
                ade_c_vals.append(ade_c.item())
                fde_c_vals.append(fde_c.item())
                miou_vals.append(miou_val.item())
            if losses:
                print(
                    f"val epoch {epoch} loss {sum(losses)/len(losses):.4f} "
                    f"ADE_BB {sum(ade_bb_vals)/len(ade_bb_vals):.2f} "
                    f"FDE_BB {sum(fde_bb_vals)/len(fde_bb_vals):.2f} "
                    f"ADE_C {sum(ade_c_vals)/len(ade_c_vals):.2f} "
                    f"FDE_C {sum(fde_c_vals)/len(fde_c_vals):.2f} "
                    f"mIoU {sum(miou_vals)/len(miou_vals):.4f}"
                )


if __name__ == "__main__":
    main()
