from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import TrackKalmanForecastDataset
from experiments.kalman_ml_forecasting.models.factory import build_model
from experiments.kalman_ml_forecasting.models.kalman_filter import kalman_config_from_dict, kalman_cv_forecast
from experiments.kalman_ml_forecasting.models.kalman_residual import last_four_constant_velocity_forecast
from experiments.kalman_ml_forecasting.utils.config import (
    load_config,
    resolve_representation_image_sizes,
)


def _boxes_to_xyxy(boxes: torch.Tensor, frame_size: tuple[int, int]) -> torch.Tensor:
    w, h = float(frame_size[0]), float(frame_size[1])
    cx = boxes[:, 0].clamp(0, 1) * w
    cy = boxes[:, 1].clamp(0, 1) * h
    bw = boxes[:, 2].clamp(0, 1) * w
    bh = boxes[:, 3].clamp(0, 1) * h
    return torch.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dim=-1)


def _draw_boxes(
    draw,
    boxes: torch.Tensor,
    frame_size: tuple[int, int],
    *,
    outline: tuple[int, int, int],
    width: int,
) -> None:
    for x0, y0, x1, y1 in _boxes_to_xyxy(boxes, frame_size).tolist():
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=width)


def _draw_center_polyline(
    draw,
    boxes: torch.Tensor,
    frame_size: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    if boxes.shape[0] < 2:
        return
    w, h = float(frame_size[0]), float(frame_size[1])
    points = [(float(box[0]) * w, float(box[1]) * h) for box in boxes]
    draw.line(points, fill=fill, width=width)


def _render_forecast_overlay(
    *,
    past_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    frame_size: tuple[int, int],
    backdrop,
    cv_boxes: torch.Tensor | None = None,
):
    from PIL import Image, ImageDraw

    if backdrop is None:
        img = Image.new("RGB", frame_size, (0, 0, 0))
    else:
        img = backdrop.resize(frame_size, resample=Image.BILINEAR).convert("RGB")
    draw = ImageDraw.Draw(img)

    if cv_boxes is not None:
        _draw_center_polyline(draw, cv_boxes, frame_size, fill=(0, 255, 255), width=3)
        _draw_boxes(draw, cv_boxes, frame_size, outline=(0, 255, 255), width=2)
    _draw_center_polyline(draw, past_boxes, frame_size, fill=(0, 90, 255), width=4)
    _draw_center_polyline(draw, gt_boxes, frame_size, fill=(0, 255, 0), width=4)
    _draw_center_polyline(draw, pred_boxes, frame_size, fill=(255, 230, 0), width=4)
    _draw_boxes(draw, past_boxes, frame_size, outline=(0, 90, 255), width=2)
    _draw_boxes(draw, gt_boxes, frame_size, outline=(0, 255, 0), width=3)
    _draw_boxes(draw, pred_boxes, frame_size, outline=(255, 230, 0), width=3)
    return img


def _to_gif(frames: List["PIL.Image.Image"], out_path: Path, duration_ms: int) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _parse_frame_key(key: str) -> tuple[str, str]:
    if "/" in key:
        folder, stem = key.split("/", 1)
        return folder, stem
    return "", key


def _parse_rgb_time(path: Path) -> float | None:
    import re

    match = re.search(r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)$", path.stem)
    if not match:
        return None
    hh, mm, ss, frac = match.groups()
    try:
        return int(hh) * 3600.0 + int(mm) * 60.0 + int(ss) + int(frac.ljust(6, "0")[:6]) / 1_000_000.0
    except ValueError:
        return None


def _nearest_dataset_rgb(
    labels_root: Path,
    folder: str,
    rep: str,
    frame_time_s: float,
):
    rgb_dir_name = {"rgb": "RGB", "padded_rgb": "PADDED_RGB"}.get(rep.lower())
    if rgb_dir_name is None:
        return None
    rgb_dir = labels_root / folder / rgb_dir_name if folder else labels_root / rgb_dir_name
    if not rgb_dir.exists():
        return None
    files = [
        path
        for pattern in ("*.jpg", "*.png", "*.jpeg")
        for path in sorted(rgb_dir.glob(pattern))
        if not path.name.startswith(".") and not path.name.startswith("._")
    ]
    parsed = [(path, _parse_rgb_time(path)) for path in files]
    parsed = [(path, t) for path, t in parsed if t is not None]
    if not parsed:
        return files[0] if files else None
    base = parsed[0][1]
    return min(parsed, key=lambda item: abs((item[1] - base) - frame_time_s))[0]


def _nearest_dataset_event_frame(
    labels_root: Path,
    folder: str,
    stem: str,
    frame_time_s: float,
    label_time_unit: float,
):
    if stem:
        frames_dir = labels_root / folder / "Event" / "Frames" if folder else labels_root / "Event" / "Frames"
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = frames_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        if frames_dir.exists():
            files = [
                path
                for pattern in ("*.png", "*.jpg", "*.jpeg")
                for path in sorted(frames_dir.glob(pattern))
                if not path.name.startswith(".") and not path.name.startswith("._")
            ]
            parsed = []
            for path in files:
                try:
                    time_raw = int(path.stem.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                parsed.append((float(time_raw) * label_time_unit, path))
            if parsed:
                return min(parsed, key=lambda item: abs(item[0] - frame_time_s))[1]
    return None


def _load_backdrop(
    *,
    images_root: Path,
    labels_root: Path,
    frame_key: str,
    frame_time_s: float,
    rep: str,
    frame_size: tuple[int, int],
    label_time_unit: float = 1.0e-6,
):
    from PIL import Image

    folder, stem = _parse_frame_key(frame_key)
    base = images_root / folder if folder else images_root
    for ext in (".png", ".jpg", ".jpeg"):
        img_path = base / f"{stem}_{rep}{ext}"
        if img_path.exists():
            return Image.open(img_path).convert("RGB").resize(frame_size, resample=Image.BILINEAR)
    rgb_path = _nearest_dataset_rgb(labels_root, folder, rep, frame_time_s)
    if rgb_path is not None and rgb_path.exists():
        return Image.open(rgb_path).convert("RGB").resize(frame_size, resample=Image.BILINEAR)
    if rep.lower() in {"event_frames", "event_frame"}:
        event_path = _nearest_dataset_event_frame(labels_root, folder, stem, frame_time_s, label_time_unit)
        if event_path is not None and event_path.exists():
            return Image.open(event_path).convert("RGB").resize(frame_size, resample=Image.BILINEAR)
    return None


def _build_dataset(cfg: Dict, folder: str) -> TrackKalmanForecastDataset:
    data_cfg = cfg["data"]
    return TrackKalmanForecastDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        frame_size=tuple(data_cfg["frame_size"]),
        representations=list(data_cfg["representations"]),
        image_sizes=resolve_representation_image_sizes(data_cfg),
        history_ms=float(data_cfg.get("history_ms", 400.0)),
        forecast_ms=float(data_cfg.get("forecast_ms", 800.0)),
        folders=[str(folder).strip("/")],
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
        max_tracks=None,
        max_samples=None,
        seed=int(data_cfg.get("seed", 123)),
        cache_dir=Path(data_cfg["cache_dir"]) if data_cfg.get("cache_dir") else None,
        filter_missing_representations=bool(data_cfg.get("filter_missing_representations", True)),
        spatial_cutout=dict(data_cfg.get("spatial_cutout") or {}),
    )


def _iter_selected_track_ids(
    track_ids: Iterable[int],
    *,
    requested: list[int] | None,
    max_tracks: int | None,
) -> list[int]:
    ids = sorted(set(int(track_id) for track_id in track_ids))
    if requested:
        requested_set = set(requested)
        ids = [track_id for track_id in ids if track_id in requested_set]
    if max_tracks is not None and max_tracks > 0:
        ids = ids[:max_tracks]
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-visualize Kalman ML forecasts as one GIF per track in a folder."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--folder", type=str, required=True, help="FRED folder ID, e.g. 8.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kalman_ml_forecasting_track_vis"),
    )
    parser.add_argument("--backdrop-rep", type=str, default="cstr3")
    parser.add_argument("--track-id", type=int, action="append", default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--max-frames-per-track", type=int, default=200)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--include-cv", action="store_true", help="Also draw configured Kalman CV baseline in cyan.")
    parser.add_argument("--include-last4", action="store_true", help="Draw the last-four linear extrapolation baseline in cyan instead.")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Use the configured Kalman CV baseline as the prediction; no checkpoint is required.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    kalman_cfg = kalman_config_from_dict(cfg.get("kalman"))
    frame_size = data_cfg.get("frame_size")
    if frame_size is None:
        raise ValueError("data.frame_size must be set for visualization.")
    frame_size_t = (int(frame_size[0]), int(frame_size[1]))

    dataset = _build_dataset(cfg, args.folder)
    samples_by_track: dict[int, list[int]] = defaultdict(list)
    for idx, sample_meta in enumerate(dataset.samples):
        samples_by_track[int(sample_meta["track_id"])].append(idx)
    track_ids = _iter_selected_track_ids(
        samples_by_track.keys(),
        requested=args.track_id,
        max_tracks=args.max_tracks,
    )
    if not track_ids:
        raise RuntimeError(f"No track samples found for folder={args.folder}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if not args.baseline_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --baseline-only is set.")
        model = build_model(cfg, device)
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
        model.eval()

    output_dir = args.output_dir / str(args.folder).strip("/")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with torch.no_grad():
        for track_id in track_ids:
            frames = []
            indices = samples_by_track[track_id]
            if args.max_frames_per_track and args.max_frames_per_track > 0:
                indices = indices[: args.max_frames_per_track]
            for sample_idx in indices:
                sample = dataset[sample_idx]
                inputs = {rep: sample.inputs[rep].unsqueeze(0).to(device) for rep in sample.inputs}
                past_boxes = sample.past_boxes.unsqueeze(0).to(device)
                future_boxes = sample.future_boxes.unsqueeze(0).to(device)
                past_times_s = sample.past_times_s.unsqueeze(0).to(device)
                future_times_s = sample.future_times_s.unsqueeze(0).to(device)
                kalman_boxes = kalman_cv_forecast(past_boxes, past_times_s, future_times_s, kalman_cfg)
                last4_boxes = last_four_constant_velocity_forecast(past_boxes, past_times_s, future_times_s)
                if args.baseline_only:
                    pred_boxes = kalman_boxes
                    cv_overlay = None
                else:
                    pred_boxes = model(inputs, past_boxes, past_times_s, future_times_s)
                    if args.include_last4:
                        cv_overlay = last4_boxes
                    else:
                        cv_overlay = kalman_boxes if args.include_cv else None
                backdrop = _load_backdrop(
                    images_root=Path(data_cfg["images_root"]),
                    labels_root=Path(data_cfg["labels_root"]),
                    frame_key=sample.frame_key,
                    frame_time_s=sample.frame_time_s,
                    rep=args.backdrop_rep,
                    frame_size=frame_size_t,
                    label_time_unit=float(data_cfg.get("label_time_unit", 1.0e-6)),
                )
                frames.append(
                    _render_forecast_overlay(
                        past_boxes=past_boxes[0].cpu(),
                        pred_boxes=pred_boxes[0].cpu(),
                        gt_boxes=future_boxes[0].cpu(),
                        cv_boxes=None if cv_overlay is None else cv_overlay[0].cpu(),
                        frame_size=frame_size_t,
                        backdrop=backdrop,
                    )
                )
            suffix = "kalman" if args.baseline_only else "model"
            out_path = output_dir / f"folder_{args.folder}_track_{track_id}_{suffix}.gif"
            _to_gif(frames, out_path, args.duration_ms)
            written += 1
            print(f"Wrote {out_path} ({len(frames)} frames)")

    print(f"Wrote {written} track GIFs to {output_dir}")


if __name__ == "__main__":
    main()
