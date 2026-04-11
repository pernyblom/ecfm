from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.curve_fit_forecasting.core import forecast_sample
from experiments.curve_fit_forecasting.data.track_dataset import TrackCurveForecastDataset
from experiments.forecasting.metrics import ade_fde_bbox_px, ade_fde_center_px, miou
from experiments.forecasting.utils.config import load_config


def _read_split_file(path: Path) -> list[str]:
    items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(line.strip("/"))
    return items


def _build_dataset(cfg: Dict, split: str) -> TrackCurveForecastDataset:
    data_cfg = cfg["data"]
    split_files = data_cfg.get("split_files")
    folders = None
    if split_files:
        if split not in split_files:
            raise ValueError(f"Split '{split}' not found in config data.split_files.")
        folders = _read_split_file(Path(split_files[split]))
    max_samples_key = f"max_samples_{split}"
    max_tracks_key = f"max_tracks_{split}"
    return TrackCurveForecastDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        frame_size=tuple(data_cfg["frame_size"]),
        image_window_ms=float(data_cfg.get("image_window_ms", 400.0)),
        history_ms=float(data_cfg.get("history_ms", 400.0)),
        forecast_ms=float(data_cfg.get("forecast_ms", 400.0)),
        representations=list(data_cfg.get("representations", ["xt_my", "yt_mx"])),
        folders=folders,
        labels_subdir=data_cfg.get("labels_subdir", "Event_YOLO"),
        tracks_file=data_cfg.get("tracks_file", "cleaned_tracks.txt"),
        label_time_unit=float(data_cfg.get("label_time_unit", 1.0e-6)),
        track_time_unit=float(data_cfg.get("track_time_unit", 1.0)),
        time_align=data_cfg.get("time_align", "start"),
        image_window_mode=data_cfg.get("image_window_mode", "trailing"),
        verify_render_manifest=bool(data_cfg.get("verify_render_manifest", True)),
        render_manifest_name=data_cfg.get("render_manifest_name", "render_manifest.json"),
        window_tolerance_ms=float(data_cfg.get("window_tolerance_ms", 5.0)),
        label_period_s=data_cfg.get("label_period_s"),
        max_tracks=data_cfg.get(max_tracks_key, data_cfg.get("max_tracks")),
        max_samples=data_cfg.get(max_samples_key, data_cfg.get("max_samples")),
        seed=int(data_cfg.get("seed", 123)),
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
    )


def _render_boxes_image(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    past_boxes: torch.Tensor,
    frame_size: tuple[int, int],
    fit_centers_xy: torch.Tensor | None = None,
    pred_centers_xy: torch.Tensor | None = None,
    backdrop: Image.Image | None = None,
    curve_width: int = 1,
) -> Image.Image:
    if backdrop is None:
        img = Image.new("RGB", frame_size, (0, 0, 0))
    else:
        img = backdrop.resize(frame_size, resample=Image.BILINEAR).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = frame_size

    def _to_xyxy(box: torch.Tensor) -> list[float]:
        cx, cy, bw, bh = [float(v) for v in box]
        cx *= w
        cy *= h
        bw *= w
        bh *= h
        return [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

    for box in past_boxes:
        draw.rectangle(_to_xyxy(box), outline=(0, 0, 255), width=2)
    for box in gt_boxes:
        draw.rectangle(_to_xyxy(box), outline=(0, 255, 0), width=2)
    for box in pred_boxes:
        draw.rectangle(_to_xyxy(box), outline=(255, 255, 0), width=2)

    def _curve_points(curve_xy: torch.Tensor) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for xy in curve_xy:
            x = float(xy[0]) * w
            y = float(xy[1]) * h
            pts.append((x, y))
        return pts

    if fit_centers_xy is not None and fit_centers_xy.numel() > 0:
        pts = _curve_points(fit_centers_xy)
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 128, 0), width=curve_width)
    if pred_centers_xy is not None and pred_centers_xy.numel() > 0:
        pts = _curve_points(pred_centers_xy)
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 0, 255), width=curve_width)
    return img


def _parse_frame_key(key: str) -> tuple[str, str]:
    if "/" in key:
        folder, stem = key.split("/", 1)
        return folder, stem
    return "", key


def _load_backdrop(
    images_root: Path,
    frame_key: str,
    rep: str | None,
) -> Image.Image | None:
    if not rep:
        return None
    folder, stem = _parse_frame_key(frame_key)
    base = images_root / folder if folder else images_root
    for ext in (".png", ".jpg", ".jpeg"):
        path = base / f"{stem}_{rep}{ext}"
        if path.exists():
            return Image.open(path).convert("RGB")
    return None


def _draw_xy_curve_on_image(
    img: Image.Image,
    curve_xy: torch.Tensor,
    *,
    color: tuple[int, int, int],
    width: int,
) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    pts = [(float(xy[0]) * w, float(xy[1]) * h) for xy in curve_xy]
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)
    return out


def _draw_plane_curve_on_image(
    img: Image.Image,
    *,
    history_times_px: torch.Tensor | List[float] | object,
    observed_fit_px: torch.Tensor | List[float] | object,
    time_axis: str,
    color: tuple[int, int, int],
    width: int,
) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    times = [float(v) for v in history_times_px]
    coords = [float(v) for v in observed_fit_px]
    pts: list[tuple[float, float]] = []
    for t, s in zip(times, coords):
        if time_axis == "x":
            pts.append((t, s))
        else:
            pts.append((s, t))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    split = args.split or eval_cfg.get("split", "val")
    dataset = _build_dataset(cfg, split=split)
    frame_size = (int(data_cfg["frame_size"][0]), int(data_cfg["frame_size"][1]))
    images_root = Path(data_cfg["images_root"])
    vis_dir = Path(eval_cfg.get("vis_output_dir", "outputs/curve_fit_forecast_vis"))
    max_visualizations = int(eval_cfg.get("max_visualizations", 0))
    vis_backdrop_rep = eval_cfg.get("vis_backdrop_rep")
    vis_curve_width = int(eval_cfg.get("vis_curve_width", 1))
    vis_save_plane_overlays = bool(eval_cfg.get("vis_save_plane_overlays", True))
    vis_plane_reps = list(eval_cfg.get("vis_plane_reps", ["xt_my", "yt_mx", "cstr3"]))
    if max_visualizations > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    ade_bb_vals: List[float] = []
    fde_bb_vals: List[float] = []
    ade_c_vals: List[float] = []
    fde_c_vals: List[float] = []
    miou_vals: List[float] = []
    fit_source_counts: Dict[str, int] = {}
    xt_rmse_vals: List[float] = []
    yt_rmse_vals: List[float] = []
    xt_event_points: List[int] = []
    yt_event_points: List[int] = []
    results_rows: List[Dict[str, object]] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        result = forecast_sample(
            xt_path=Path(sample.input_paths["xt_my"]),
            yt_path=Path(sample.input_paths["yt_mx"]),
            past_boxes=sample.past_boxes,
            past_times_s=sample.past_times_s,
            future_times_s=sample.future_times_s,
            anchor_time_s=sample.frame_time_s,
            image_window_ms=float(data_cfg.get("image_window_ms", 400.0)),
            image_window_mode=data_cfg.get("image_window_mode", "trailing"),
            params=cfg.get("curve_fit", {}),
        )

        pred = result.pred_boxes.unsqueeze(0)
        target = sample.future_boxes.unsqueeze(0)
        ade_bb, fde_bb = ade_fde_bbox_px(pred, target, frame_size)
        ade_c, fde_c = ade_fde_center_px(pred, target, frame_size)
        miou_val = miou(pred, target, frame_size)
        ade_bb_vals.append(float(ade_bb))
        fde_bb_vals.append(float(fde_bb))
        ade_c_vals.append(float(ade_c))
        fde_c_vals.append(float(fde_c))
        miou_vals.append(float(miou_val))
        xt_rmse_vals.append(float(result.xt.history_rmse_px))
        yt_rmse_vals.append(float(result.yt.history_rmse_px))
        xt_event_points.append(int(result.debug["xt_event_points"]))
        yt_event_points.append(int(result.debug["yt_event_points"]))
        for key in (result.xt.source, result.yt.source):
            fit_source_counts[key] = fit_source_counts.get(key, 0) + 1

        results_rows.append(
            {
                "frame_key": sample.frame_key,
                "track_id": sample.track_id,
                "ade_bbox_px": float(ade_bb),
                "fde_bbox_px": float(fde_bb),
                "ade_center_px": float(ade_c),
                "fde_center_px": float(fde_c),
                "miou": float(miou_val),
                **result.debug,
            }
        )

        if idx < max_visualizations:
            stem_prefix = f"{idx:04d}_{sample.track_id}_{sample.frame_key.replace('/', '_')}"
            backdrop = _load_backdrop(images_root, sample.frame_key, vis_backdrop_rep)
            img = _render_boxes_image(
                pred_boxes=result.pred_boxes,
                gt_boxes=sample.future_boxes,
                past_boxes=sample.past_boxes,
                frame_size=frame_size,
                fit_centers_xy=result.fit_centers_xy,
                pred_centers_xy=result.pred_centers_xy,
                backdrop=backdrop,
                curve_width=vis_curve_width,
            )
            img.save(vis_dir / f"{stem_prefix}.png")

            if vis_save_plane_overlays:
                if "xt_my" in vis_plane_reps:
                    xt_img = Image.open(sample.input_paths["xt_my"]).convert("RGB")
                    xt_overlay = _draw_plane_curve_on_image(
                        xt_img,
                        history_times_px=result.xt.history_times_px,
                        observed_fit_px=result.xt.observed_fit,
                        time_axis=result.xt.time_axis,
                        color=(255, 128, 0),
                        width=vis_curve_width,
                    )
                    xt_overlay.save(vis_dir / f"{stem_prefix}_xt_my_fit.png")

                if "yt_mx" in vis_plane_reps:
                    yt_img = Image.open(sample.input_paths["yt_mx"]).convert("RGB")
                    yt_overlay = _draw_plane_curve_on_image(
                        yt_img,
                        history_times_px=result.yt.history_times_px,
                        observed_fit_px=result.yt.observed_fit,
                        time_axis=result.yt.time_axis,
                        color=(255, 128, 0),
                        width=vis_curve_width,
                    )
                    yt_overlay.save(vis_dir / f"{stem_prefix}_yt_mx_fit.png")

                if "cstr3" in vis_plane_reps:
                    cstr3 = _load_backdrop(images_root, sample.frame_key, "cstr3")
                    if cstr3 is not None:
                        cstr3_overlay = _draw_xy_curve_on_image(
                            cstr3,
                            result.fit_centers_xy,
                            color=(255, 128, 0),
                            width=vis_curve_width,
                        )
                        cstr3_overlay.save(vis_dir / f"{stem_prefix}_cstr3_fit.png")

        if (idx + 1) % max(1, int(eval_cfg.get("log_every", 100))) == 0:
            print(f"{split}: processed {idx + 1}/{len(dataset)} samples")

    if not ade_bb_vals:
        raise RuntimeError("No samples evaluated.")

    summary = {
        "split": split,
        "num_samples": len(ade_bb_vals),
        "ADE_BB": sum(ade_bb_vals) / len(ade_bb_vals),
        "FDE_BB": sum(fde_bb_vals) / len(fde_bb_vals),
        "ADE_C": sum(ade_c_vals) / len(ade_c_vals),
        "FDE_C": sum(fde_c_vals) / len(fde_c_vals),
        "mIoU": sum(miou_vals) / len(miou_vals),
        "xt_history_rmse_px": sum(xt_rmse_vals) / len(xt_rmse_vals),
        "yt_history_rmse_px": sum(yt_rmse_vals) / len(yt_rmse_vals),
        "xt_event_points": sum(xt_event_points) / len(xt_event_points),
        "yt_event_points": sum(yt_event_points) / len(yt_event_points),
        "fit_sources": fit_source_counts,
    }
    print(
        f"{split} ADE_BB {summary['ADE_BB']:.2f} FDE_BB {summary['FDE_BB']:.2f} "
        f"ADE_C {summary['ADE_C']:.2f} FDE_C {summary['FDE_C']:.2f} "
        f"mIoU {summary['mIoU']:.4f}"
    )
    print(
        f"{split} XT_RMSE {summary['xt_history_rmse_px']:.2f} "
        f"YT_RMSE {summary['yt_history_rmse_px']:.2f} "
        f"XT_pts {summary['xt_event_points']:.1f} YT_pts {summary['yt_event_points']:.1f}"
    )

    metrics_path = eval_cfg.get("metrics_output")
    if metrics_path:
        out_path = Path(metrics_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "samples": results_rows}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
