from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import TrackKalmanForecastDataset
from experiments.kalman_ml_forecasting.utils.config import (
    load_config,
    read_split_file,
    resolve_representation_image_sizes,
)


def _select_folders(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[str] | None:
    if args.folder:
        folders = [str(folder).strip("/\\") for folder in args.folder]
    else:
        split_files = cfg["data"].get("split_files") or {}
        if args.split_file is not None:
            folders = read_split_file(args.split_file)
        elif split_files.get(args.split):
            folders = read_split_file(Path(split_files[args.split]))
        else:
            folders = None
    if folders is not None and args.max_folders is not None:
        folders = folders[: max(0, int(args.max_folders))]
    return folders


def _build_dataset(
    cfg: Dict[str, Any],
    *,
    folders: List[str] | None,
    max_samples: int | None,
) -> TrackKalmanForecastDataset:
    data_cfg = cfg["data"]
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
        min_track_duration_ms=data_cfg.get("min_track_duration_ms"),
        max_tracks=data_cfg.get("max_tracks"),
        max_samples=max_samples,
        seed=int(data_cfg.get("seed", 123)),
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
        filter_missing_representations=bool(data_cfg.get("filter_missing_representations", True)),
        spatial_cutout=dict(data_cfg.get("spatial_cutout") or {}),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr_u8, mode="RGB")


def _load_original(path: str, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, resample=Image.BILINEAR)
    return img


def _draw_anchor_box(
    img: Image.Image,
    *,
    rep: str,
    box: torch.Tensor,
    frame_size: tuple[int, int],
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    img_w, img_h = out.size
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    cx = float(box[0]) * frame_w
    cy = float(box[1]) * frame_h
    bw = float(box[2]) * frame_w
    bh = float(box[3]) * frame_h
    rep_l = rep.lower()
    if rep_l.startswith("xt"):
        x = cx / frame_w * img_w
        half = bw / frame_w * img_w / 2.0
        draw.rectangle([x - half, 0, x + half, img_h - 1], outline=(255, 230, 0), width=2)
    elif rep_l.startswith("yt"):
        y = cy / frame_h * img_h
        half = bh / frame_h * img_h / 2.0
        draw.rectangle([0, y - half, img_w - 1, y + half], outline=(255, 230, 0), width=2)
    else:
        x = cx / frame_w * img_w
        y = cy / frame_h * img_h
        half_w = bw / frame_w * img_w / 2.0
        half_h = bh / frame_h * img_h / 2.0
        draw.rectangle([x - half_w, y - half_h, x + half_w, y + half_h], outline=(255, 230, 0), width=2)
    return out


def _make_contact_sheet(images: list[tuple[str, Image.Image]], *, cell_width: int = 320) -> Image.Image:
    if not images:
        raise ValueError("Cannot make a contact sheet without images.")
    label_h = 22
    thumbs: list[tuple[str, Image.Image]] = []
    for label, img in images:
        scale = min(1.0, float(cell_width) / max(1, img.size[0]))
        thumb_size = (max(1, int(round(img.size[0] * scale))), max(1, int(round(img.size[1] * scale))))
        thumbs.append((label, img.resize(thumb_size, resample=Image.BILINEAR)))
    width = max(thumb.size[0] for _, thumb in thumbs)
    height = sum(thumb.size[1] + label_h for _, thumb in thumbs)
    sheet = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    y = 0
    for label, thumb in thumbs:
        draw.text((4, y + 4), label, fill=(235, 235, 235))
        sheet.paste(thumb, (0, y + label_h))
        y += thumb.size[1] + label_h
    return sheet


def _sample_indices(dataset: TrackKalmanForecastDataset, args: argparse.Namespace) -> list[int]:
    indices = list(range(len(dataset)))
    if args.track_id:
        wanted = {int(track_id) for track_id in args.track_id}
        indices = [idx for idx in indices if int(dataset.samples[idx]["track_id"]) in wanted]
    if args.start_index > 0:
        indices = indices[int(args.start_index) :]
    if args.count is not None:
        indices = indices[: max(0, int(args.count))]
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write Kalman ML representation tensors after configured spatial cutout masking."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--folder", type=str, action="append", default=None)
    parser.add_argument("--max-folders", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset construction before writing.")
    parser.add_argument("--count", type=int, default=16, help="Number of selected samples to write.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--track-id", type=int, action="append", default=None)
    parser.add_argument("--representation", type=str, action="append", default=None)
    parser.add_argument("--include-original", action="store_true", help="Also write unmasked resized inputs.")
    parser.add_argument("--draw-anchor-box", action="store_true", help="Draw the final-history box on output images.")
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kalman_ml_spatial_cutouts"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    cutout_cfg = dict(data_cfg.get("spatial_cutout") or {})
    folders = _select_folders(args, cfg)
    dataset = _build_dataset(cfg, folders=folders, max_samples=args.max_samples)
    reps = list(args.representation or data_cfg["representations"])
    missing_reps = [rep for rep in reps if rep not in data_cfg["representations"]]
    if missing_reps:
        raise ValueError(f"Requested representations are not in data.representations: {missing_reps}")
    indices = _sample_indices(dataset, args)
    if not indices:
        raise RuntimeError("No samples selected.")

    image_sizes = resolve_representation_image_sizes(data_cfg)
    frame_size = (int(data_cfg["frame_size"][0]), int(data_cfg["frame_size"][1]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Writing spatial cutout inspection: samples={len(indices)}, reps={reps}, "
        f"cutout={json.dumps(cutout_cfg, sort_keys=True)}"
    )

    summary = {
        "config": str(args.config),
        "split": args.split,
        "folders": folders,
        "representations": reps,
        "spatial_cutout": cutout_cfg,
        "samples": [],
    }
    for write_idx, dataset_idx in enumerate(indices):
        sample = dataset[dataset_idx]
        meta = dataset.samples[dataset_idx]
        sample_name = f"{write_idx:04d}_{_safe_name(sample.frame_key)}_track_{int(sample.track_id)}"
        sample_dir = args.output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        anchor_box = sample.past_boxes[-1]
        sheet_images: list[tuple[str, Image.Image]] = []
        sample_summary = {
            "dataset_index": int(dataset_idx),
            "frame_key": sample.frame_key,
            "track_id": int(sample.track_id),
            "anchor_time_s": float(sample.frame_time_s),
            "anchor_box_cxcywh_norm": [float(v) for v in anchor_box.tolist()],
            "outputs": {},
        }
        for rep in reps:
            if rep not in sample.inputs:
                continue
            cutout_img = _tensor_to_pil(sample.inputs[rep])
            if args.draw_anchor_box:
                cutout_img = _draw_anchor_box(cutout_img, rep=rep, box=anchor_box, frame_size=frame_size)
            cutout_path = sample_dir / f"{_safe_name(rep)}_cutout.png"
            cutout_img.save(cutout_path)
            sheet_images.append((f"{rep} cutout", cutout_img))
            rep_outputs = {"cutout": str(cutout_path)}
            if args.include_original:
                original_img = _load_original(meta["input_paths"][rep], image_sizes[rep])
                if args.draw_anchor_box:
                    original_img = _draw_anchor_box(original_img, rep=rep, box=anchor_box, frame_size=frame_size)
                original_path = sample_dir / f"{_safe_name(rep)}_original.png"
                original_img.save(original_path)
                sheet_images.append((f"{rep} original", original_img))
                rep_outputs["original"] = str(original_path)
            sample_summary["outputs"][rep] = rep_outputs
        if sheet_images and not args.no_contact_sheet:
            sheet_path = sample_dir / "contact_sheet.png"
            _make_contact_sheet(sheet_images).save(sheet_path)
            sample_summary["contact_sheet"] = str(sheet_path)
        summary["samples"].append(sample_summary)
        print(f"Wrote {sample_dir}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
