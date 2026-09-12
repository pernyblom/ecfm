from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import TrackKalmanForecastDataset
from experiments.kalman_ml_forecasting.compose_track_layers import (
    compose_image_layers,
    iter_composed_layer_files,
    parse_composition,
    write_gif_iter,
    write_mp4_iter,
)
from experiments.kalman_ml_forecasting.models.factory import build_model
from experiments.kalman_ml_forecasting.models.kalman_filter import kalman_config_from_dict, kalman_forecast
from experiments.kalman_ml_forecasting.models.kalman_residual import last_four_constant_velocity_forecast
from experiments.kalman_ml_forecasting.utils.config import (
    load_config,
    resolve_representation_image_sizes,
    resolve_representation_sequences,
    resolve_representation_source_image_sizes,
)


# Keep these in one place: PIL uses RGB (not OpenCV's BGR).  Prediction is
# deliberately drawn before ground truth in composites so ground truth remains
# visible where the two forecasts overlap.
LAYER_COLORS = {
    "past": (0, 90, 255, 255),
    "ground_truth": (0, 255, 0, 255),
    "prediction": (255, 230, 0, 255),
    "baseline": (0, 255, 255, 255),
}

# These projections do not share the spatial x/y canvas used by boxes and raw
# events. They are exported at their rendered resolution for synchronized
# side-by-side comparison, never resized to data.frame_size.
NATIVE_RESOLUTION_LAYER_REPS = {"xt", "xt_my", "yt", "yt_mx"}


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
    outline: tuple[int, ...],
    width: int,
) -> None:
    for x0, y0, x1, y1 in _boxes_to_xyxy(boxes, frame_size).tolist():
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=width)


def _draw_center_polyline(
    draw,
    boxes: torch.Tensor,
    frame_size: tuple[int, int],
    *,
    fill: tuple[int, ...],
    width: int,
) -> None:
    if boxes.shape[0] < 2:
        return
    w, h = float(frame_size[0]), float(frame_size[1])
    points = [(float(box[0]) * w, float(box[1]) * h) for box in boxes]
    draw.line(points, fill=fill, width=width)


def _render_box_layer(
    boxes: torch.Tensor,
    frame_size: tuple[int, int],
    color: tuple[int, int, int, int],
    *,
    line_width: int = 4,
    box_width: int = 3,
):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _draw_center_polyline(draw, boxes, frame_size, fill=color, width=line_width)
    _draw_boxes(draw, boxes, frame_size, outline=color, width=box_width)
    return img


def _render_forecast_layers(
    *,
    past_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    frame_size: tuple[int, int],
    backdrop,
    cv_boxes: torch.Tensor | None = None,
):
    from PIL import Image

    background = (
        Image.new("RGBA", frame_size, (0, 0, 0, 0))
        if backdrop is None
        else backdrop.resize(frame_size, resample=Image.BILINEAR).convert("RGBA")
    )
    layers = {
        "background": background,
        "history_boxes": _render_box_layer(past_boxes, frame_size, LAYER_COLORS["past"], box_width=2),
        "prediction_boxes": _render_box_layer(pred_boxes, frame_size, LAYER_COLORS["prediction"]),
        "gt_boxes": _render_box_layer(gt_boxes, frame_size, LAYER_COLORS["ground_truth"]),
    }
    if cv_boxes is not None:
        layers["cv_boxes"] = _render_box_layer(
            cv_boxes, frame_size, LAYER_COLORS["baseline"], line_width=3, box_width=2
        )
    composite = background.copy()
    for name in ("history_boxes", "cv_boxes", "prediction_boxes", "gt_boxes"):
        if name in layers:
            composite.alpha_composite(layers[name])
    layers["composite"] = composite
    return layers


def _load_raw_events(labels_root: Path, folder: str, event_source: str):
    """Load one FRED event sequence as [x, y, timestamp, polarity]."""
    import atexit
    from scripts.render_evt3_yolo_frames import (
        _load_events_from_npz,
        _load_events_from_raw,
        _parse_geometry,
        _read_raw_meta_or_empty,
        _read_ts_shift_us,
    )

    event_dir = labels_root / folder / "Event" if folder else labels_root / "Event"
    raw_path = event_dir / "events.raw"
    npz_path = event_dir / "output_events.npz"
    cleanup = None
    if event_source == "npz" or (event_source == "auto" and npz_path.exists()):
        events, _, meta, _, _, _ = _load_events_from_npz(
            raw_path, event_unit=1.0, ts_shift_us=_read_ts_shift_us(raw_path)
        )
    else:
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing FRED event stream: {raw_path}")
        events, _, meta, _, temporary_path = _load_events_from_raw(
            raw_path, endian="little", decode_chunk_mb=64.0, event_unit=1.0,
            ts_shift_us=_read_ts_shift_us(raw_path), show_progress=True,
        )
        if temporary_path is not None:
            # Keep the mmap alive while rendering, but close its Windows file
            # handle explicitly before unlinking the decoded temporary file.
            cleaned = False

            def cleanup() -> None:
                nonlocal cleaned
                if cleaned:
                    return
                cleaned = True
                current = events
                seen: set[int] = set()
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    mmap_handle = getattr(current, "_mmap", None)
                    if mmap_handle is not None:
                        mmap_handle.close()
                        break
                    current = getattr(current, "base", None)
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    # A third-party Windows process may briefly retain the
                    # file. Avoid an atexit traceback; the OS temp directory
                    # can safely reclaim this decoded cache later.
                    pass

            atexit.register(cleanup)
    width, height = _parse_geometry(meta or _read_raw_meta_or_empty(raw_path))
    if width is None or height is None:
        if events.size == 0:
            raise ValueError("Cannot infer event sensor geometry from an empty stream.")
        width, height = int(events[:, 0].max()) + 1, int(events[:, 1].max()) + 1
    return events, (int(width), int(height)), cleanup


def _render_raw_event_subframes(
    events: np.ndarray,
    sensor_size: tuple[int, int],
    frame_size: tuple[int, int],
    start_s: float,
    end_s: float,
    slowdown: int,
    event_time_unit: float,
    transparent: bool,
):
    from PIL import Image

    if end_s <= start_s:
        end_s = start_s + 1.0 / 30.0
    boundaries = np.linspace(start_s, end_s, slowdown + 1) / event_time_unit
    timestamps = events[:, 2]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        lo = int(np.searchsorted(timestamps, left, side="left"))
        hi = int(np.searchsorted(timestamps, right, side="left"))
        selected = events[lo:hi]
        mode = "RGBA" if transparent else "RGB"
        background = (0, 0, 0, 0) if transparent else (0, 0, 0)
        image = Image.new(mode, sensor_size, background)
        if selected.size:
            xs = selected[:, 0].astype(np.int64)
            ys = selected[:, 1].astype(np.int64)
            ps = np.clip(selected[:, 3].astype(np.int64), 0, 1)
            valid = (xs >= 0) & (xs < sensor_size[0]) & (ys >= 0) & (ys < sensor_size[1])
            colors = ((0, 90, 255, 255), (255, 70, 40, 255)) if transparent else ((0, 90, 255), (255, 70, 40))
            pixels = np.asarray(image).copy()
            pixels[ys[valid], xs[valid]] = np.asarray(colors, dtype=np.uint8)[ps[valid]]
            image = Image.fromarray(pixels, mode=mode)
        yield image.resize(frame_size, resample=Image.NEAREST)


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
    preserve_native_size: bool = False,
):
    from PIL import Image

    def load(path: Path):
        with Image.open(path) as source:
            image = source.convert("RGB")
        if preserve_native_size:
            return image
        return image.resize(frame_size, resample=Image.BILINEAR)

    folder, stem = _parse_frame_key(frame_key)
    base = images_root / folder if folder else images_root
    for ext in (".png", ".jpg", ".jpeg"):
        img_path = base / f"{stem}_{rep}{ext}"
        if img_path.exists():
            return load(img_path)
    rgb_path = _nearest_dataset_rgb(labels_root, folder, rep, frame_time_s)
    if rgb_path is not None and rgb_path.exists():
        return load(rgb_path)
    if rep.lower() in {"event_frames", "event_frame"}:
        event_path = _nearest_dataset_event_frame(labels_root, folder, stem, frame_time_s, label_time_unit)
        if event_path is not None and event_path.exists():
            return load(event_path)
    return None


def _build_dataset(cfg: Dict, folder: str) -> TrackKalmanForecastDataset:
    data_cfg = cfg["data"]
    return TrackKalmanForecastDataset(
        images_root=Path(data_cfg["images_root"]),
        labels_root=Path(data_cfg["labels_root"]),
        frame_size=tuple(data_cfg["frame_size"]),
        representations=list(data_cfg["representations"]),
        image_sizes=resolve_representation_image_sizes(data_cfg),
        source_image_sizes=resolve_representation_source_image_sizes(data_cfg),
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
        representation_sequences=resolve_representation_sequences(data_cfg),
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
        description="Render reusable Kalman ML forecast layers and optional GIF/MP4 outputs."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--folder", type=str, required=True, help="FRED folder ID, e.g. 8.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kalman_ml_forecasting_track_vis"),
    )
    parser.add_argument(
        "--backdrop-rep", type=str, default="none",
        help="Legacy shortcut used only when --composition is omitted (default: none).",
    )
    parser.add_argument(
        "--layer-rep", action="append", default=None,
        help="Also export a representation as a named layer; repeatable (for example padded_rgb).",
    )
    parser.add_argument("--track-id", type=int, action="append", default=None)
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--max-frames-per-track", type=int, default=200)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--output-gif", action="store_true", help="Write the default composite GIF.")
    parser.add_argument("--output-mp4", action="store_true", help="Write the default composite MP4.")
    parser.add_argument(
        "--output-composition-gif", action="store_true",
        help="Write the active composition as a separately named GIF.",
    )
    parser.add_argument(
        "--output-composition-mp4", action="store_true",
        help="Write the active composition as a separately named MP4.",
    )
    parser.add_argument(
        "--composition", default=None,
        help="Complete semicolon-separated layer chain in back-to-front order.",
    )
    parser.add_argument(
        "--slowdown", type=int, default=1,
        help="Output frames per dataset frame. Static layers repeat; raw events are newly sliced.",
    )
    parser.add_argument(
        "--raw-events", action="store_true",
        help="Render timestamp-aligned raw events as a two-polarity layer.",
    )
    parser.add_argument(
        "--event-source", choices=("auto", "npz", "raw"), default="auto",
        help="Prefer the much faster output_events.npz cache when available.",
    )
    parser.add_argument(
        "--event-time-unit", type=float, default=None,
        help="Seconds per raw event timestamp unit (default: data.label_time_unit).",
    )
    parser.add_argument(
        "--transparent-events", action="store_true",
        help="Use transparency instead of black behind raw events.",
    )
    parser.add_argument(
        "--split-layers", action="store_true",
        help="Deprecated no-op: reusable PNG layers are now always written.",
    )
    parser.add_argument("--include-cv", action="store_true", help="Also draw configured Kalman CV baseline in cyan.")
    parser.add_argument("--include-last4", action="store_true", help="Draw the last-four linear extrapolation baseline in cyan instead.")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Use the configured Kalman CV baseline as the prediction; no checkpoint is required.",
    )
    args = parser.parse_args()
    if args.slowdown < 1:
        parser.error("--slowdown must be at least 1")
    if args.duration_ms < 1:
        parser.error("--duration-ms must be at least 1")
    if args.composition is not None:
        composition_order = parse_composition(args.composition)
    else:
        composition_order = [
            "background" if args.backdrop_rep.lower() == "none" else args.backdrop_rep.lower()
        ]
        if args.raw_events:
            composition_order.append("raw_events")
        composition_order.append("history_boxes")
        if args.include_cv or args.include_last4:
            composition_order.append("cv_boxes")
        composition_order.extend(("prediction_boxes", "gt_boxes"))
    if "composite" in composition_order:
        parser.error("--composition cannot contain the derived 'composite' layer.")

    generated_layer_names = {
        "background", "raw_events", "history_boxes", "prediction_boxes", "cv_boxes", "gt_boxes"
    }
    representation_layers = {
        name for name in composition_order if name not in generated_layer_names
    }
    representation_layers.update(rep.lower() for rep in (args.layer_rep or []))

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
    raw_events = None
    sensor_size = None
    raw_events_cleanup = None
    if args.raw_events or "raw_events" in composition_order:
        raw_events, sensor_size, raw_events_cleanup = _load_raw_events(
            Path(data_cfg["labels_root"]), str(args.folder).strip("/"), args.event_source
        )
        if raw_events.size and np.any(np.diff(raw_events[:, 2]) < 0):
            raw_events = raw_events[np.argsort(raw_events[:, 2], kind="stable")]
    event_time_unit = float(
        args.event_time_unit
        if args.event_time_unit is not None
        else data_cfg.get("label_time_unit", 1.0e-6)
    )
    written = 0
    with torch.no_grad():
        for track_id in track_ids:
            suffix = "kalman" if args.baseline_only else "model"
            layers_root = output_dir / f"folder_{args.folder}_track_{track_id}_{suffix}_layers"
            # A shorter rerun must not leave stale tail frames from an older run.
            if layers_root.exists():
                for stale_path in layers_root.glob("*/*.png"):
                    stale_path.unlink()
            frame_index = 0
            indices = samples_by_track[track_id]
            if args.max_frames_per_track and args.max_frames_per_track > 0:
                indices = indices[: args.max_frames_per_track]
            previous_time_s = None
            for sample_idx in indices:
                sample = dataset[sample_idx]
                inputs = {rep: sample.inputs[rep].unsqueeze(0).to(device) for rep in sample.inputs}
                past_boxes = sample.past_boxes.unsqueeze(0).to(device)
                future_boxes = sample.future_boxes.unsqueeze(0).to(device)
                past_times_s = sample.past_times_s.unsqueeze(0).to(device)
                future_times_s = sample.future_times_s.unsqueeze(0).to(device)
                kalman_boxes = kalman_forecast(past_boxes, past_times_s, future_times_s, kalman_cfg)
                last4_boxes = last_four_constant_velocity_forecast(past_boxes, past_times_s, future_times_s)
                want_cv_layer = "cv_boxes" in composition_order or args.include_cv or args.include_last4
                if args.baseline_only:
                    pred_boxes = kalman_boxes
                    cv_overlay = kalman_boxes if want_cv_layer else None
                else:
                    pred_boxes = model(inputs, past_boxes, past_times_s, future_times_s)
                    if args.include_last4:
                        cv_overlay = last4_boxes
                    else:
                        cv_overlay = kalman_boxes if want_cv_layer else None
                layers = _render_forecast_layers(
                        past_boxes=past_boxes[0].cpu(),
                        pred_boxes=pred_boxes[0].cpu(),
                        gt_boxes=future_boxes[0].cpu(),
                        cv_boxes=None if cv_overlay is None else cv_overlay[0].cpu(),
                        frame_size=frame_size_t,
                        backdrop=None,
                    )
                layers.pop("composite", None)
                for layer_rep_l in sorted(representation_layers):
                    if layer_rep_l == "none":
                        continue
                    layer_image = _load_backdrop(
                        images_root=Path(data_cfg["images_root"]),
                        labels_root=Path(data_cfg["labels_root"]),
                        frame_key=sample.frame_key,
                        frame_time_s=sample.frame_time_s,
                        rep=layer_rep_l,
                        frame_size=frame_size_t,
                        label_time_unit=float(data_cfg.get("label_time_unit", 1.0e-6)),
                        preserve_native_size=layer_rep_l in NATIVE_RESOLUTION_LAYER_REPS,
                    )
                    if layer_image is None:
                        raise FileNotFoundError(
                            f"Could not resolve composition layer {layer_rep_l!r} for {sample.frame_key}."
                        )
                    layers[layer_rep_l] = layer_image.convert("RGBA")
                event_frames = None
                if raw_events is not None and sensor_size is not None:
                    if previous_time_s is None:
                        # Use the configured/inferred label cadence for the first sample.
                        period = float(data_cfg.get("label_period_s") or (1.0 / 30.0))
                        start_time_s = sample.frame_time_s - period
                    else:
                        start_time_s = previous_time_s
                    event_frames = iter(_render_raw_event_subframes(
                        raw_events, sensor_size, frame_size_t, start_time_s,
                        sample.frame_time_s, args.slowdown, event_time_unit,
                        args.transparent_events,
                    ))
                for _ in range(args.slowdown):
                    current_layers = dict(layers)
                    if event_frames is not None:
                        current_layers["raw_events"] = next(event_frames)
                    current_layers["composite"] = compose_image_layers(
                        current_layers, composition_order
                    )
                    for name, image in current_layers.items():
                        layer_dir = layers_root / name
                        layer_dir.mkdir(parents=True, exist_ok=True)
                        image.save(layer_dir / f"{frame_index:06d}.png")
                    frame_index += 1
                previous_time_s = sample.frame_time_s
            stem = output_dir / f"folder_{args.folder}_track_{track_id}_{suffix}"
            composite_order = ["composite"]
            if args.output_gif:
                write_gif_iter(
                    iter_composed_layer_files(layers_root, composite_order),
                    stem.with_suffix(".gif"), args.duration_ms,
                )
            if args.output_mp4:
                write_mp4_iter(
                    iter_composed_layer_files(layers_root, composite_order),
                    stem.with_suffix(".mp4"), 1000.0 / args.duration_ms,
                )
            if args.output_composition_gif:
                write_gif_iter(
                    iter_composed_layer_files(layers_root, composite_order),
                    Path(f"{stem}_composition.gif"), args.duration_ms,
                )
            if args.output_composition_mp4:
                write_mp4_iter(
                    iter_composed_layer_files(layers_root, composite_order),
                    Path(f"{stem}_composition.mp4"), 1000.0 / args.duration_ms,
                )
            written += 1
            print(f"Rendered folder={args.folder} track={track_id} ({frame_index} frames)")

    print(f"Rendered {written} tracks to {output_dir}")
    if raw_events_cleanup is not None:
        raw_events_cleanup()


if __name__ == "__main__":
    main()
