from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forecasting.data.dataset import ForecastDataset, split_dataset
from experiments.forecasting.data.track_dataset import TrackForecastDataset
from experiments.forecasting.metrics import ade_fde_bbox_px, ade_fde_center_px, miou
from experiments.forecasting.models.fusion import MultiRepForecast
from experiments.forecasting.utils.config import load_config


def _collate_samples(batch):
    if not batch:
        return batch
    inputs = {}
    for rep in batch[0].inputs:
        inputs[rep] = torch.stack([b.inputs[rep] for b in batch], dim=0)
    past_boxes = torch.stack([b.past_boxes for b in batch], dim=0)
    future_boxes = torch.stack([b.future_boxes for b in batch], dim=0)
    frame_keys = [b.frame_keys for b in batch]
    return type("Batch", (), {"inputs": inputs, "past_boxes": past_boxes, "future_boxes": future_boxes, "frame_keys": frame_keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]
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

    supervision = data_cfg.get("supervision", "yolo")
    split_files = data_cfg.get("split_files")
    if split_files:
        val_folders = _read_split_file(Path(split_files["val"]))
        if supervision == "tracks":
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
        _, val_set = split_dataset(
            dataset, train_split=data_cfg["train_split"], seed=data_cfg["seed"]
        )
    loader = DataLoader(
        val_set,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_samples,
    )

    device = torch.device(
        eval_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu"
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

    if args.checkpoint and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)

    model.eval()
    with torch.no_grad():
        ade_bb_vals = []
        fde_bb_vals = []
        ade_c_vals = []
        fde_c_vals = []
        miou_vals = []
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.inputs.items()}
            past_boxes = batch.past_boxes.to(device)
            future = batch.future_boxes.to(device)
            pred = model(inputs, past_boxes)
            ade_bb, fde_bb = ade_fde_bbox_px(pred, future, frame_size_t)
            ade_c, fde_c = ade_fde_center_px(pred, future, frame_size_t)
            miou_val = miou(pred, future, frame_size_t)
            ade_bb_vals.append(ade_bb.item())
            fde_bb_vals.append(fde_bb.item())
            ade_c_vals.append(ade_c.item())
            fde_c_vals.append(fde_c.item())
            miou_vals.append(miou_val.item())
        if ade_bb_vals:
            print(
                f"ADE_BB {sum(ade_bb_vals)/len(ade_bb_vals):.2f} "
                f"FDE_BB {sum(fde_bb_vals)/len(fde_bb_vals):.2f} "
                f"ADE_C {sum(ade_c_vals)/len(ade_c_vals):.2f} "
                f"FDE_C {sum(fde_c_vals)/len(fde_c_vals):.2f} "
                f"mIoU {sum(miou_vals)/len(miou_vals):.4f}"
            )


if __name__ == "__main__":
    main()
