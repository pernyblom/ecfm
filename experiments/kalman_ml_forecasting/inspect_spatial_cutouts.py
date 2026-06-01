from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import numpy as np
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


def _size_pair_from_config(value: Any, *, name: str) -> tuple[float, float]:
    if isinstance(value, dict):
        if "width" in value or "height" in value:
            return float(value.get("width", value.get("x", 0.0))), float(value.get("height", value.get("y", 0.0)))
        if "x" in value or "y" in value:
            return float(value.get("x", 0.0)), float(value.get("y", 0.0))
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a number or a two-item [width, height] value.")
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, scalar


def _base_representation_name(rep: str) -> str:
    match = re.match(r"^(?P<base>.+)_\d+x\d+$", str(rep), flags=re.IGNORECASE)
    return match.group("base") if match is not None else str(rep)


def _load_original(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _crop_with_padding(
    img: Image.Image,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    fill: float,
) -> Image.Image:
    width = max(1, int(width))
    height = max(1, int(height))
    fill_u8 = int(round(max(0.0, min(1.0, float(fill))) * 255.0))
    out = Image.new("RGB", (width, height), (fill_u8, fill_u8, fill_u8))
    src_left = max(0, int(left))
    src_top = max(0, int(top))
    src_right = min(img.size[0], int(left) + width)
    src_bottom = min(img.size[1], int(top) + height)
    if src_right <= src_left or src_bottom <= src_top:
        return out
    patch = img.crop((src_left, src_top, src_right, src_bottom))
    out.paste(patch, (src_left - int(left), src_top - int(top)))
    return out


def _spatial_cutout_image(
    img: Image.Image,
    *,
    rep: str,
    box: Any,
    cfg: Dict[str, Any],
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    mode = str(cfg.get("mode", "none")).lower()
    if mode in {"none", "disabled", "off", "false"}:
        return img.copy(), (0, 0, img.size[0], img.size[1])
    img_w, img_h = img.size
    cx = float(box[0]) * img_w
    cy = float(box[1]) * img_h
    bw_img = float(box[2]) * img_w
    bh_img = float(box[3]) * img_h
    if mode in {"box_fraction", "box_scale", "box"}:
        scale = float(cfg.get("scale", cfg.get("box_scale", cfg.get("fraction", 1.0))))
        if scale <= 0.0:
            raise ValueError("data.spatial_cutout scale/fraction must be > 0.")
        cut_w = max(1, int(round(bw_img * scale)))
        cut_h = max(1, int(round(bh_img * scale)))
    elif mode in {"fixed", "fixed_pixels", "fixed_px"}:
        size_value = cfg.get("size_px", cfg.get("fixed_size_px", cfg.get("size", None)))
        if size_value is None:
            raise ValueError("data.spatial_cutout fixed mode requires size_px or fixed_size_px.")
        cut_w_raw, cut_h_raw = _size_pair_from_config(size_value, name="data.spatial_cutout.size_px")
        cut_w = max(1, int(round(cut_w_raw)))
        cut_h = max(1, int(round(cut_h_raw)))
    else:
        raise ValueError(
            "data.spatial_cutout.mode must be one of: none, box_scale, box_fraction, fixed_pixels, fixed."
        )

    base_rep = _base_representation_name(rep).lower()
    if base_rep.startswith("xt"):
        width = cut_w
        height = img_h
        left = int(round(cx - width / 2.0))
        top = 0
    elif base_rep.startswith("yt"):
        width = img_w
        height = cut_h
        left = 0
        top = int(round(cy - height / 2.0))
    else:
        width = cut_w
        height = cut_h
        left = int(round(cx - width / 2.0))
        top = int(round(cy - height / 2.0))
    cutout = _crop_with_padding(
        img,
        left=left,
        top=top,
        width=width,
        height=height,
        fill=float(cfg.get("fill", cfg.get("fill_value", 0.0))),
    )
    return cutout, (left, top, width, height)


def _draw_anchor_box(
    img: Image.Image,
    *,
    rep: str,
    box: Any,
    source_size: tuple[int, int],
    crop: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    img_w, img_h = out.size
    crop_left, crop_top, _, _ = crop
    source_w, source_h = float(source_size[0]), float(source_size[1])
    cx = float(box[0]) * source_w - crop_left
    cy = float(box[1]) * source_h - crop_top
    bw = float(box[2]) * source_w
    bh = float(box[3]) * source_h
    rep_l = rep.lower()
    if rep_l.startswith("xt"):
        x = cx
        half = bw / 2.0
        draw.rectangle([x - half, 0, x + half, img_h - 1], outline=(255, 230, 0), width=2)
    elif rep_l.startswith("yt"):
        y = cy
        half = bh / 2.0
        draw.rectangle([0, y - half, img_w - 1, y + half], outline=(255, 230, 0), width=2)
    else:
        x = cx
        y = cy
        half_w = bw / 2.0
        half_h = bh / 2.0
        draw.rectangle([x - half_w, y - half_h, x + half_w, y + half_h], outline=(255, 230, 0), width=2)
    return out


def _make_contact_sheet(images: list[tuple[str, Image.Image]]) -> Image.Image:
    if not images:
        raise ValueError("Cannot make a contact sheet without images.")
    label_h = 22
    width = max(img.size[0] for _, img in images)
    height = sum(img.size[1] + label_h for _, img in images)
    sheet = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    y = 0
    for label, img in images:
        draw.text((4, y + 4), label, fill=(235, 235, 235))
        sheet.paste(img, (0, y + label_h))
        y += img.size[1] + label_h
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
        description="Write source representation crops using the configured spatial cutout."
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
    parser.add_argument("--include-original", action="store_true", help="Also write unmasked source inputs.")
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
        meta = dataset.samples[dataset_idx]
        folder = str(meta["folder"])
        frame_key = f"{folder}/{meta['anchor_stem']}" if folder else str(meta["anchor_stem"])
        track_id = int(meta["track_id"])
        sample_name = f"{write_idx:04d}_{_safe_name(frame_key)}_track_{track_id}"
        sample_dir = args.output_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        anchor_box = np.asarray(meta["past_boxes"], dtype=np.float32)[-1]
        sheet_images: list[tuple[str, Image.Image]] = []
        sample_summary = {
            "dataset_index": int(dataset_idx),
            "frame_key": frame_key,
            "track_id": track_id,
            "anchor_time_s": float(meta["anchor_time_s"]),
            "anchor_box_cxcywh_norm": [float(v) for v in anchor_box.tolist()],
            "outputs": {},
        }
        for rep in reps:
            if rep not in meta["input_paths"]:
                continue
            original_img = _load_original(meta["input_paths"][rep])
            cutout_img, crop = _spatial_cutout_image(
                original_img,
                rep=rep,
                box=anchor_box,
                cfg=cutout_cfg,
            )
            if args.draw_anchor_box:
                cutout_img = _draw_anchor_box(
                    cutout_img,
                    rep=rep,
                    box=anchor_box,
                    source_size=original_img.size,
                    crop=crop,
                )
            cutout_path = sample_dir / f"{_safe_name(rep)}_cutout.png"
            cutout_img.save(cutout_path)
            sheet_images.append((f"{rep} cutout", cutout_img))
            rep_outputs = {
                "cutout": str(cutout_path),
                "cutout_size": [int(cutout_img.size[0]), int(cutout_img.size[1])],
                "source_size": [int(original_img.size[0]), int(original_img.size[1])],
                "crop_left_top_width_height": [int(value) for value in crop],
            }
            if args.include_original:
                if args.draw_anchor_box:
                    original_img = _draw_anchor_box(
                        original_img,
                        rep=rep,
                        box=anchor_box,
                        source_size=original_img.size,
                    )
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
