from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import (
    _parse_frame_time_raw,
    _read_tracks,
)
from experiments.kalman_ml_forecasting.utils.config import load_config, read_split_file


@dataclass
class FrameItem:
    stem: str
    time_s: float


@dataclass
class AccelSample:
    folder: str
    track_id: int
    anchor_stem: str
    anchor_time_s: float
    cx_px: float
    cy_px: float
    ax_px_s2: float
    ay_px_s2: float
    vx_px_s: float
    vy_px_s: float
    fit_rmse_px: float
    num_points: int


def _discover_frames(labels_dir: Path, *, label_time_unit: float) -> List[FrameItem]:
    if not labels_dir.exists():
        return []
    out = []
    for path in sorted(labels_dir.glob("*.txt")):
        time_raw = _parse_frame_time_raw(path.stem)
        if time_raw is None:
            continue
        out.append(FrameItem(stem=path.stem, time_s=float(time_raw) * label_time_unit))
    out.sort(key=lambda item: item.time_s)
    return out


def _resolve_label_period_s(frames: List[FrameItem], configured: Optional[float]) -> float:
    if configured is not None:
        return float(configured)
    if len(frames) < 2:
        raise ValueError("Need at least two frames to infer label_period_s.")
    diffs = np.diff(np.asarray([frame.time_s for frame in frames], dtype=np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("Could not infer positive label period from frame timestamps.")
    return float(np.median(diffs))


def _align_track_times(track_times: np.ndarray, label_times: np.ndarray, *, time_align: str) -> np.ndarray:
    times = track_times.copy()
    if time_align == "start":
        return times + (label_times[0] - times[0])
    if time_align == "auto":
        shift = label_times[0] - times[0]
        no_shift = int(np.sum((label_times >= times[0]) & (label_times <= times[-1])))
        shifted = int(np.sum((label_times >= times[0] + shift) & (label_times <= times[-1] + shift)))
        return times + shift if shifted > no_shift else times
    if time_align == "none":
        return times
    raise ValueError(f"Unknown time_align: {time_align}")


def _window_spec(*, history_ms: float, forecast_ms: float, label_period_s: float) -> tuple[int, int]:
    history_steps = max(1, int(round((history_ms / 1000.0) / label_period_s)))
    future_steps = max(1, int(round((forecast_ms / 1000.0) / label_period_s)))
    return history_steps, future_steps


def _fit_constant_acceleration(
    times_s: np.ndarray,
    centers_px: np.ndarray,
    *,
    anchor_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rel_t = times_s.astype(np.float64) - float(anchor_time_s)
    design = np.stack(
        [
            np.ones_like(rel_t),
            rel_t,
            0.5 * rel_t * rel_t,
        ],
        axis=1,
    )
    coeff, *_ = np.linalg.lstsq(design, centers_px.astype(np.float64), rcond=None)
    pred = design @ coeff
    rmse = float(np.sqrt(np.mean(np.sum((pred - centers_px) ** 2, axis=1))))
    pos = coeff[0]
    vel = coeff[1]
    accel = coeff[2]
    return pos, vel, accel, rmse


def _collect_folder_samples(
    *,
    folder: str,
    cfg: Dict,
    max_tracks: Optional[int],
    max_samples: Optional[int],
) -> List[AccelSample]:
    data_cfg = cfg["data"]
    labels_root = Path(data_cfg["labels_root"])
    labels_subdir = data_cfg.get("labels_subdir", "Event_YOLO")
    tracks_file = data_cfg.get("tracks_file", "cleaned_tracks.txt")
    label_time_unit = float(data_cfg.get("label_time_unit", 1.0e-6))
    track_time_unit = float(data_cfg.get("track_time_unit", 1.0))
    time_align = str(data_cfg.get("time_align", "auto"))
    history_ms = float(data_cfg.get("history_ms", 400.0))
    forecast_ms = float(data_cfg.get("forecast_ms", 800.0))
    frame_w, frame_h = [float(value) for value in data_cfg["frame_size"]]
    label_period_s = data_cfg.get("label_period_s")

    labels_dir = labels_root / folder / labels_subdir if folder else labels_root / labels_subdir
    tracks_path = labels_root / folder / tracks_file if folder else labels_root / tracks_file
    frames = _discover_frames(labels_dir, label_time_unit=label_time_unit)
    if not frames:
        return []
    tracks = _read_tracks(tracks_path)
    if not tracks:
        return []

    period_s = _resolve_label_period_s(frames, None if label_period_s is None else float(label_period_s))
    history_steps, future_steps = _window_spec(
        history_ms=history_ms,
        forecast_ms=forecast_ms,
        label_period_s=period_s,
    )
    window = history_steps + future_steps + 1
    label_times = np.asarray([frame.time_s for frame in frames], dtype=np.float64)
    label_stems = [frame.stem for frame in frames]
    out: List[AccelSample] = []

    track_items = sorted(tracks.items(), key=lambda item: int(item[0]))
    if max_tracks is not None and max_tracks > 0:
        track_items = track_items[: int(max_tracks)]
    for track_id, rows in track_items:
        times = np.asarray([row[0] for row in rows], dtype=np.float64) * track_time_unit
        if times.size < 3:
            continue
        times = _align_track_times(times, label_times, time_align=time_align)
        mask = (label_times >= times[0]) & (label_times <= times[-1])
        if not np.any(mask):
            continue
        idxs = np.nonzero(mask)[0]
        if idxs.size < window:
            continue
        query_times = label_times[idxs]
        xs = np.interp(query_times, times, np.asarray([row[1] for row in rows], dtype=np.float64))
        ys = np.interp(query_times, times, np.asarray([row[2] for row in rows], dtype=np.float64))
        ws = np.interp(query_times, times, np.asarray([row[3] for row in rows], dtype=np.float64))
        hs = np.interp(query_times, times, np.asarray([row[4] for row in rows], dtype=np.float64))
        centers = np.stack(
            [
                np.clip(xs + ws / 2.0, 0.0, frame_w),
                np.clip(ys + hs / 2.0, 0.0, frame_h),
            ],
            axis=1,
        )
        stems = [label_stems[idx] for idx in idxs]
        for start in range(0, len(stems) - window + 1):
            end = start + window
            anchor_idx = start + history_steps
            anchor_time = float(query_times[anchor_idx])
            pos, vel, accel, rmse = _fit_constant_acceleration(
                query_times[start:end],
                centers[start:end],
                anchor_time_s=anchor_time,
            )
            out.append(
                AccelSample(
                    folder=folder,
                    track_id=int(track_id),
                    anchor_stem=stems[anchor_idx],
                    anchor_time_s=anchor_time,
                    cx_px=float(pos[0]),
                    cy_px=float(pos[1]),
                    ax_px_s2=float(accel[0]),
                    ay_px_s2=float(accel[1]),
                    vx_px_s=float(vel[0]),
                    vy_px_s=float(vel[1]),
                    fit_rmse_px=rmse,
                    num_points=window,
                )
            )
            if max_samples is not None and len(out) >= max_samples:
                return out
    return out


def _sample_arrays(samples: List[AccelSample]) -> Dict[str, np.ndarray]:
    return {
        "folder": np.asarray([sample.folder for sample in samples]),
        "track_id": np.asarray([sample.track_id for sample in samples], dtype=np.int64),
        "anchor_stem": np.asarray([sample.anchor_stem for sample in samples]),
        "anchor_time_s": np.asarray([sample.anchor_time_s for sample in samples], dtype=np.float64),
        "cx_px": np.asarray([sample.cx_px for sample in samples], dtype=np.float32),
        "cy_px": np.asarray([sample.cy_px for sample in samples], dtype=np.float32),
        "ax_px_s2": np.asarray([sample.ax_px_s2 for sample in samples], dtype=np.float32),
        "ay_px_s2": np.asarray([sample.ay_px_s2 for sample in samples], dtype=np.float32),
        "vx_px_s": np.asarray([sample.vx_px_s for sample in samples], dtype=np.float32),
        "vy_px_s": np.asarray([sample.vy_px_s for sample in samples], dtype=np.float32),
        "fit_rmse_px": np.asarray([sample.fit_rmse_px for sample in samples], dtype=np.float32),
        "num_points": np.asarray([sample.num_points for sample in samples], dtype=np.int64),
    }


def _write_csv(path: Path, samples: List[AccelSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AccelSample.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def _bin_vectors(
    samples: List[AccelSample],
    *,
    grid_cols: int,
    grid_rows: int,
    bin_x_attr: str,
    bin_y_attr: str,
    x_attr: str,
    y_attr: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> dict[str, np.ndarray]:
    sum_ax = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    sum_ay = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    counts = np.zeros((grid_rows, grid_cols), dtype=np.int64)
    x_span = max(float(x_max) - float(x_min), 1.0e-9)
    y_span = max(float(y_max) - float(y_min), 1.0e-9)
    for sample in samples:
        bin_x = float(getattr(sample, bin_x_attr))
        bin_y = float(getattr(sample, bin_y_attr))
        col = min(grid_cols - 1, max(0, int((bin_x - x_min) / x_span * grid_cols)))
        row = min(grid_rows - 1, max(0, int((bin_y - y_min) / y_span * grid_rows)))
        sum_ax[row, col] += float(getattr(sample, x_attr))
        sum_ay[row, col] += float(getattr(sample, y_attr))
        counts[row, col] += 1
    mean_ax = np.divide(sum_ax, counts, out=np.zeros_like(sum_ax), where=counts > 0)
    mean_ay = np.divide(sum_ay, counts, out=np.zeros_like(sum_ay), where=counts > 0)
    return {"mean_ax": mean_ax, "mean_ay": mean_ay, "counts": counts}


def _magnitude_color(value: float, min_value: float, max_value: float) -> tuple[int, int, int]:
    if max_value <= min_value:
        return (40, 80, 180)
    t = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))
    return (
        int(40 + 210 * t),
        int(90 * (1.0 - t) + 40 * t),
        int(210 * (1.0 - t) + 30 * t),
    )


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    vector: tuple[float, float],
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    sx, sy = start
    vx, vy = vector
    ex = sx + vx
    ey = sy + vy
    draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
    angle = math.atan2(vy, vx)
    head_len = max(5.0, min(18.0, math.hypot(vx, vy) * 0.28))
    left = (
        ex - head_len * math.cos(angle - math.pi / 6.0),
        ey - head_len * math.sin(angle - math.pi / 6.0),
    )
    right = (
        ex - head_len * math.cos(angle + math.pi / 6.0),
        ey - head_len * math.sin(angle + math.pi / 6.0),
    )
    draw.polygon([(ex, ey), left, right], fill=color)


def _render_vector_field(
    *,
    samples: List[AccelSample],
    grid_cols: int,
    grid_rows: int,
    output_path: Path,
    bin_x_attr: str,
    bin_y_attr: str,
    x_attr: str,
    y_attr: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    title: str,
    colorbar_label: str,
    summary_prefix: str,
    max_output_width: int,
    min_count: int,
    arrow_scale: Optional[float],
    bound_percentile: float,
    magnitude_min: Optional[float] = None,
    magnitude_max: Optional[float] = None,
    invert_y: bool = False,
) -> dict[str, float | int]:
    x_min = float(x_min)
    x_max = float(x_max)
    y_min = float(y_min)
    y_max = float(y_max)
    x_span = max(x_max - x_min, 1.0e-9)
    y_span = max(y_max - y_min, 1.0e-9)
    bins = _bin_vectors(
        samples,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        bin_x_attr=bin_x_attr,
        bin_y_attr=bin_y_attr,
        x_attr=x_attr,
        y_attr=y_attr,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    mean_ax = bins["mean_ax"]
    mean_ay = bins["mean_ay"]
    counts = bins["counts"]
    valid = counts >= int(min_count)
    mags = np.sqrt(mean_ax * mean_ax + mean_ay * mean_ay)
    nonzero = mags[valid & (mags > 0)]
    cell_w = x_span / grid_cols
    cell_h = y_span / grid_rows
    if arrow_scale is None:
        reference = float(np.median(nonzero)) if nonzero.size else 1.0
        arrow_scale = 0.35 * min(cell_w, cell_h) / max(reference, 1.0e-9)
    bound_percentile = max(0.0, min(100.0, float(bound_percentile)))
    auto_min_mag = float(np.min(nonzero)) if nonzero.size else 0.0
    auto_max_mag = float(np.percentile(nonzero, bound_percentile)) if nonzero.size else 0.0
    min_mag = auto_min_mag if magnitude_min is None else float(magnitude_min)
    max_mag = auto_max_mag if magnitude_max is None else float(magnitude_max)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except Exception:
        plt = None

    if plt is not None:
        xs = x_min + (np.arange(grid_cols, dtype=np.float64) + 0.5) * cell_w
        ys = y_min + (np.arange(grid_rows, dtype=np.float64) + 0.5) * cell_h
        xx, yy = np.meshgrid(xs, ys)
        masked_ax = np.where(valid, mean_ax, np.nan)
        masked_ay = np.where(valid, mean_ay, np.nan)
        masked_mag = np.where(valid, mags, np.nan)
        fig_w = max(8.0, min(16.0, float(max_output_width) / 100.0))
        fig_h = fig_w * y_span / x_span
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        density = np.ma.masked_where(counts <= 0, counts)
        extent = (x_min, x_max, y_max, y_min) if invert_y else (x_min, x_max, y_min, y_max)
        ax.imshow(
            density,
            extent=extent,
            cmap="Greys",
            alpha=0.22,
            interpolation="nearest",
            origin="upper" if invert_y else "lower",
        )
        quiver = ax.quiver(
            xx,
            yy,
            masked_ax * float(arrow_scale),
            masked_ay * float(arrow_scale),
            masked_mag,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            cmap="viridis",
            norm=Normalize(vmin=min_mag, vmax=max_mag) if max_mag > min_mag else None,
            width=0.0035,
            headwidth=3.5,
            headlength=5.0,
            headaxislength=4.5,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim((y_max, y_min) if invert_y else (y_min, y_max))
        ax.set_aspect("equal")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}, n={len(samples)}, grid={grid_cols}x{grid_rows}")
        cbar = fig.colorbar(quiver, ax=ax, fraction=0.026, pad=0.02)
        cbar.set_label(colorbar_label)
        ax.text(
            0.01,
            0.99,
            f"cell shade = sample count; min_count={min_count}; arrow_scale={float(arrow_scale):.4g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return {
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "min_count": int(min_count),
            "arrow_scale": float(arrow_scale),
            "max_count": int(counts.max()) if counts.size else 0,
            "valid_cells": int(valid.sum()),
            f"{summary_prefix}_magnitude_bound": max_mag,
            f"{summary_prefix}_magnitude_min": min_mag,
            f"{summary_prefix}_bound_percentile": bound_percentile,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "renderer": "matplotlib",
        }

    scale = min(1.0, float(max_output_width) / float(x_span))
    image_size = (max(1, int(round(x_span * scale))), max(1, int(round(y_span * scale))))
    img = Image.new("RGB", image_size, (248, 248, 248))
    draw = ImageDraw.Draw(img)

    for col in range(grid_cols + 1):
        x = int(round(col * cell_w * scale))
        draw.line([(x, 0), (x, image_size[1])], fill=(225, 225, 225), width=1)
    for row in range(grid_rows + 1):
        y = int(round(row * cell_h * scale))
        draw.line([(0, y), (image_size[0], y)], fill=(225, 225, 225), width=1)

    max_count = int(counts.max()) if counts.size else 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if not valid[row, col]:
                continue
            cx = (col + 0.5) * cell_w * scale
            cy = (row + 0.5) * cell_h * scale
            if not invert_y:
                cy = image_size[1] - cy
            vx = mean_ax[row, col] * float(arrow_scale) * scale
            vy = mean_ay[row, col] * float(arrow_scale) * scale
            if not invert_y:
                vy = -vy
            mag = float(mags[row, col])
            color = _magnitude_color(mag, min_mag, max_mag)
            width = 1 + int(3 * counts[row, col] / max(1, max_count))
            _draw_arrow(draw, (cx, cy), (vx, vy), color=color, width=width)

    title = f"{title}, n={len(samples)}, grid={grid_cols}x{grid_rows}, min_count={min_count}"
    draw.rectangle([(0, 0), (min(image_size[0], 760), 24)], fill=(248, 248, 248))
    draw.text((8, 6), title, fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return {
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "min_count": int(min_count),
        "arrow_scale": float(arrow_scale),
        "max_count": max_count,
        "valid_cells": int(valid.sum()),
        f"{summary_prefix}_magnitude_bound": max_mag,
        f"{summary_prefix}_magnitude_min": min_mag,
        f"{summary_prefix}_bound_percentile": bound_percentile,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "renderer": "pil",
    }


def _select_folders(args: argparse.Namespace, cfg: Dict) -> List[str]:
    if args.folder:
        return [str(folder).strip("/\\") for folder in args.folder]
    if args.split_file is not None:
        folders = read_split_file(args.split_file)
    else:
        split_files = cfg["data"].get("split_files") or {}
        if args.split not in split_files or not split_files[args.split]:
            raise ValueError(f"Split '{args.split}' is not configured in data.split_files.")
        folders = read_split_file(Path(split_files[args.split]))
    if args.max_folders is not None:
        folders = folders[: max(0, int(args.max_folders))]
    return folders


def _resolve_axis_bounds(
    values: np.ndarray,
    *,
    axis_min: Optional[float],
    axis_max: Optional[float],
    percentile: float,
) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot resolve bounds from an empty value array.")
    percentile = max(0.0, min(100.0, float(percentile)))
    auto_min = float(np.percentile(values, 100.0 - percentile))
    auto_max = float(np.percentile(values, percentile))
    if axis_min is not None:
        auto_min = float(axis_min)
    if axis_max is not None:
        auto_max = float(axis_max)
    if auto_max <= auto_min:
        pad = max(1.0, abs(auto_min) * 0.05)
        auto_min -= pad
        auto_max += pad
    return auto_min, auto_max


def _resolve_velocity_bounds(
    samples: List[AccelSample],
    *,
    velocity_bin_min: Optional[float],
    velocity_bin_max: Optional[float],
    velocity_bin_abs: Optional[float],
    velocity_bin_x_min: Optional[float],
    velocity_bin_x_max: Optional[float],
    velocity_bin_x_abs: Optional[float],
    velocity_bin_y_min: Optional[float],
    velocity_bin_y_max: Optional[float],
    velocity_bin_y_abs: Optional[float],
    percentile: float,
) -> tuple[float, float, float, float]:
    shared_min = velocity_bin_min
    shared_max = velocity_bin_max
    if velocity_bin_abs is not None:
        if shared_min is None:
            shared_min = -abs(float(velocity_bin_abs))
        if shared_max is None:
            shared_max = abs(float(velocity_bin_abs))
    x_abs_min = -abs(float(velocity_bin_x_abs)) if velocity_bin_x_abs is not None else shared_min
    x_abs_max = abs(float(velocity_bin_x_abs)) if velocity_bin_x_abs is not None else shared_max
    y_abs_min = -abs(float(velocity_bin_y_abs)) if velocity_bin_y_abs is not None else shared_min
    y_abs_max = abs(float(velocity_bin_y_abs)) if velocity_bin_y_abs is not None else shared_max
    x_min = velocity_bin_x_min if velocity_bin_x_min is not None else x_abs_min
    x_max = velocity_bin_x_max if velocity_bin_x_max is not None else x_abs_max
    y_min = velocity_bin_y_min if velocity_bin_y_min is not None else y_abs_min
    y_max = velocity_bin_y_max if velocity_bin_y_max is not None else y_abs_max
    vx_values = np.asarray([sample.vx_px_s for sample in samples], dtype=np.float64)
    vy_values = np.asarray([sample.vy_px_s for sample in samples], dtype=np.float64)
    resolved_x_min, resolved_x_max = _resolve_axis_bounds(
        vx_values,
        axis_min=x_min,
        axis_max=x_max,
        percentile=percentile,
    )
    resolved_y_min, resolved_y_max = _resolve_axis_bounds(
        vy_values,
        axis_min=y_min,
        axis_max=y_max,
        percentile=percentile,
    )
    return resolved_x_min, resolved_x_max, resolved_y_min, resolved_y_max


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate center acceleration at forecast anchors by fitting a constant-acceleration "
            "curve over each sample's history+future window, then render a position-conditioned vector field."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--folder", type=str, action="append", default=None)
    parser.add_argument("--max-folders", type=int, default=None)
    parser.add_argument("--max-tracks-per-folder", type=int, default=None)
    parser.add_argument("--max-samples-per-folder", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grid-cols", type=int, default=24)
    parser.add_argument("--grid-rows", type=int, default=14)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--accel-arrow-scale", type=float, default=None)
    parser.add_argument("--velocity-arrow-scale", type=float, default=None)
    parser.add_argument(
        "--vector-bound-percentile",
        type=float,
        default=95.0,
        help="Dataset percentile used as the color upper bound for velocity and acceleration magnitudes.",
    )
    parser.add_argument(
        "--velocity-bin-min",
        dest="velocity_bin_min",
        type=float,
        default=None,
        help=(
            "Lower velocity axis bound applied to both axes for the velocity-to-acceleration "
            "field, in px/s. Axis-specific flags override this."
        ),
    )
    parser.add_argument(
        "--velocity-bin-max",
        dest="velocity_bin_max",
        type=float,
        default=None,
        help=(
            "Upper velocity axis bound applied to both axes for the velocity-to-acceleration "
            "field, in px/s. Axis-specific flags override this."
        ),
    )
    parser.add_argument(
        "--velocity-bin-abs",
        "--velocity-bin-mirrored",
        dest="velocity_bin_abs",
        type=float,
        default=None,
        help=(
            "Symmetric velocity bound applied to both axes, in px/s. For example 1200 "
            "means [-1200, 1200]. Axis-specific flags override this."
        ),
    )
    parser.add_argument(
        "--velocity-bin-x-min",
        "--velocity-bin-horizontal-min",
        dest="velocity_bin_x_min",
        type=float,
        default=None,
        help="Lower horizontal velocity axis bound for vx bins, in px/s.",
    )
    parser.add_argument(
        "--velocity-bin-x-max",
        "--velocity-bin-horizontal-max",
        dest="velocity_bin_x_max",
        type=float,
        default=None,
        help="Upper horizontal velocity axis bound for vx bins, in px/s.",
    )
    parser.add_argument(
        "--velocity-bin-x-abs",
        "--velocity-bin-horizontal-abs",
        "--velocity-bin-x-mirrored",
        "--velocity-bin-horizontal-mirrored",
        dest="velocity_bin_x_abs",
        type=float,
        default=None,
        help="Symmetric horizontal velocity bound for vx bins, in px/s.",
    )
    parser.add_argument(
        "--velocity-bin-y-min",
        "--velocity-bin-vertical-min",
        dest="velocity_bin_y_min",
        type=float,
        default=None,
        help="Lower vertical velocity axis bound for vy bins, in px/s.",
    )
    parser.add_argument(
        "--velocity-bin-y-max",
        "--velocity-bin-vertical-max",
        dest="velocity_bin_y_max",
        type=float,
        default=None,
        help="Upper vertical velocity axis bound for vy bins, in px/s.",
    )
    parser.add_argument(
        "--velocity-bin-y-abs",
        "--velocity-bin-vertical-abs",
        "--velocity-bin-y-mirrored",
        "--velocity-bin-vertical-mirrored",
        dest="velocity_bin_y_abs",
        type=float,
        default=None,
        help="Symmetric vertical velocity bound for vy bins, in px/s.",
    )
    parser.add_argument(
        "--velocity-bin-bound-percentile",
        type=float,
        default=99.0,
        help="Percentile used for automatic velocity axis bounds when min/max are omitted.",
    )
    parser.add_argument("--max-output-width", type=int, default=1280)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kalman_ml_acceleration_field"))
    args = parser.parse_args()
    combined_invalid = (
        args.velocity_bin_min is not None
        and args.velocity_bin_max is not None
        and float(args.velocity_bin_max) <= float(args.velocity_bin_min)
    )
    for name in ("velocity_bin_abs", "velocity_bin_x_abs", "velocity_bin_y_abs"):
        value = getattr(args, name)
        if value is not None and float(value) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than 0.")
    shared_min = args.velocity_bin_min
    shared_max = args.velocity_bin_max
    if args.velocity_bin_abs is not None:
        if shared_min is None:
            shared_min = -abs(float(args.velocity_bin_abs))
        if shared_max is None:
            shared_max = abs(float(args.velocity_bin_abs))
    x_abs_min = -abs(float(args.velocity_bin_x_abs)) if args.velocity_bin_x_abs is not None else shared_min
    x_abs_max = abs(float(args.velocity_bin_x_abs)) if args.velocity_bin_x_abs is not None else shared_max
    y_abs_min = -abs(float(args.velocity_bin_y_abs)) if args.velocity_bin_y_abs is not None else shared_min
    y_abs_max = abs(float(args.velocity_bin_y_abs)) if args.velocity_bin_y_abs is not None else shared_max
    x_min = args.velocity_bin_x_min if args.velocity_bin_x_min is not None else x_abs_min
    x_max = args.velocity_bin_x_max if args.velocity_bin_x_max is not None else x_abs_max
    y_min = args.velocity_bin_y_min if args.velocity_bin_y_min is not None else y_abs_min
    y_max = args.velocity_bin_y_max if args.velocity_bin_y_max is not None else y_abs_max
    if combined_invalid:
        raise ValueError("--velocity-bin-max must be greater than --velocity-bin-min.")
    if x_min is not None and x_max is not None and float(x_max) <= float(x_min):
        raise ValueError("--velocity-bin-x-max must be greater than --velocity-bin-x-min.")
    if y_min is not None and y_max is not None and float(y_max) <= float(y_min):
        raise ValueError("--velocity-bin-y-max must be greater than --velocity-bin-y-min.")

    cfg = load_config(args.config)
    folders = _select_folders(args, cfg)
    if not folders:
        raise RuntimeError("No folders selected.")

    samples: List[AccelSample] = []
    for folder in folders:
        remaining = None if args.max_samples is None else max(0, int(args.max_samples) - len(samples))
        if remaining == 0:
            break
        per_folder_limit = args.max_samples_per_folder
        if remaining is not None:
            per_folder_limit = remaining if per_folder_limit is None else min(int(per_folder_limit), remaining)
        folder_samples = _collect_folder_samples(
            folder=folder,
            cfg=cfg,
            max_tracks=args.max_tracks_per_folder,
            max_samples=per_folder_limit,
        )
        samples.extend(folder_samples)
        print(f"{folder}: {len(folder_samples)} acceleration samples")

    if not samples:
        raise RuntimeError("No acceleration samples found.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _sample_arrays(samples)
    npz_path = output_dir / f"{args.split}_center_acceleration_samples.npz"
    csv_path = output_dir / f"{args.split}_center_acceleration_samples.csv"
    accel_png_path = output_dir / f"{args.split}_center_acceleration_field.png"
    velocity_png_path = output_dir / f"{args.split}_center_velocity_to_acceleration_field.png"
    meta_path = output_dir / f"{args.split}_center_motion_field_summary.json"

    np.savez_compressed(npz_path, **arrays)
    _write_csv(csv_path, samples)
    frame_size = tuple(int(value) for value in cfg["data"]["frame_size"])
    accel_vis_summary = _render_vector_field(
        samples=samples,
        grid_cols=int(args.grid_cols),
        grid_rows=int(args.grid_rows),
        output_path=accel_png_path,
        bin_x_attr="cx_px",
        bin_y_attr="cy_px",
        x_attr="ax_px_s2",
        y_attr="ay_px_s2",
        x_min=0.0,
        x_max=float(frame_size[0]),
        y_min=0.0,
        y_max=float(frame_size[1]),
        x_label="x position (px)",
        y_label="y position (px)",
        title="Center acceleration field",
        colorbar_label="mean acceleration magnitude (px/s^2)",
        summary_prefix="accel",
        max_output_width=int(args.max_output_width),
        min_count=int(args.min_count),
        arrow_scale=args.accel_arrow_scale,
        bound_percentile=float(args.vector_bound_percentile),
        invert_y=True,
    )
    velocity_x_min, velocity_x_max, velocity_y_min, velocity_y_max = _resolve_velocity_bounds(
        samples,
        velocity_bin_min=args.velocity_bin_min,
        velocity_bin_max=args.velocity_bin_max,
        velocity_bin_abs=args.velocity_bin_abs,
        velocity_bin_x_min=args.velocity_bin_x_min,
        velocity_bin_x_max=args.velocity_bin_x_max,
        velocity_bin_x_abs=args.velocity_bin_x_abs,
        velocity_bin_y_min=args.velocity_bin_y_min,
        velocity_bin_y_max=args.velocity_bin_y_max,
        velocity_bin_y_abs=args.velocity_bin_y_abs,
        percentile=float(args.velocity_bin_bound_percentile),
    )
    velocity_vis_summary = _render_vector_field(
        samples=samples,
        grid_cols=int(args.grid_cols),
        grid_rows=int(args.grid_rows),
        output_path=velocity_png_path,
        bin_x_attr="vx_px_s",
        bin_y_attr="vy_px_s",
        x_attr="ax_px_s2",
        y_attr="ay_px_s2",
        x_min=velocity_x_min,
        x_max=velocity_x_max,
        y_min=velocity_y_min,
        y_max=velocity_y_max,
        x_label="vx at anchor (px/s)",
        y_label="vy at anchor (px/s, image-down positive)",
        title="Center velocity to acceleration field",
        colorbar_label="mean acceleration magnitude (px/s^2)",
        summary_prefix="velocity_to_accel",
        max_output_width=int(args.max_output_width),
        min_count=int(args.min_count),
        arrow_scale=args.velocity_arrow_scale,
        bound_percentile=float(args.vector_bound_percentile),
        invert_y=True,
    )
    accel_mag = np.sqrt(arrays["ax_px_s2"] ** 2 + arrays["ay_px_s2"] ** 2)
    velocity_mag = np.sqrt(arrays["vx_px_s"] ** 2 + arrays["vy_px_s"] ** 2)
    summary = {
        "num_folders": len(folders),
        "num_samples": len(samples),
        "split": args.split,
        "npz": str(npz_path),
        "csv": str(csv_path),
        "acceleration_png": str(accel_png_path),
        "velocity_png": str(velocity_png_path),
        "mean_accel_mag_px_s2": float(np.mean(accel_mag)),
        "median_accel_mag_px_s2": float(np.median(accel_mag)),
        "p95_accel_mag_px_s2": float(np.percentile(accel_mag, 95)),
        "mean_velocity_mag_px_s": float(np.mean(velocity_mag)),
        "median_velocity_mag_px_s": float(np.median(velocity_mag)),
        "p95_velocity_mag_px_s": float(np.percentile(velocity_mag, 95)),
        "acceleration_visualization": accel_vis_summary,
        "velocity_to_acceleration_visualization": velocity_vis_summary,
    }
    meta_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
