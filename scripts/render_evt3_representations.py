from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecfm.utils.evt3_vis import events_to_image, write_image
from scripts.render_evt3_yolo_frames import (
    _apply_transform,
    _build_rgb_index,
    _default_output_size_for_rep,
    _find_rgb_frame,
    _load_events_from_npz,
    _load_events_from_raw,
    _parse_geometry,
    _parse_image_sizes,
    _read_raw_meta_or_empty,
    _read_rgb_image,
    _read_ts_shift_us,
    _render_histogram_grid,
    _representation_list,
    _resize_to,
    _to_grayscale,
)


VALID_REPRESENTATIONS = {
    "events",
    "xy",
    "xt",
    "yt",
    "xy_p45",
    "xy_m45",
    "yt_p45",
    "yt_m45",
    "cstr2",
    "cstr3",
    "rgb",
    "grayscale",
    "gray",
    "xt_my",
    "yt_mx",
}


def _load_events(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict, dict, Optional[Path], Optional[float], bool]:
    ts_shift_us = args.ts_shift_us if args.ts_shift_us is not None else _read_ts_shift_us(args.raw)
    event_source = str(args.event_source).lower()
    if event_source == "npz":
        events, t, meta, counters, applied_shift, already_shifted = _load_events_from_npz(
            args.raw,
            event_unit=float(args.event_unit),
            ts_shift_us=ts_shift_us,
        )
        return events, t, meta, counters, None, applied_shift, already_shifted
    if event_source == "auto" and args.raw.with_name("output_events.npz").exists():
        events, t, meta, counters, applied_shift, already_shifted = _load_events_from_npz(
            args.raw,
            event_unit=float(args.event_unit),
            ts_shift_us=ts_shift_us,
        )
        return events, t, meta, counters, None, applied_shift, already_shifted

    events, t, meta, counters, tmp_path = _load_events_from_raw(
        args.raw,
        endian=args.endian,
        decode_chunk_mb=float(args.decode_chunk_mb),
        event_unit=float(args.event_unit),
        ts_shift_us=ts_shift_us,
        show_progress=bool(args.show_progress),
    )
    return events, t, meta, counters, tmp_path, ts_shift_us, False


def _frame_slices(
    num_events: int,
    *,
    events_per_frame: int,
    stride_events: int,
    max_frames: Optional[int],
) -> list[tuple[int, int]]:
    if events_per_frame <= 0:
        raise ValueError("--events-per-frame must be positive.")
    if stride_events <= 0:
        raise ValueError("--stride-events must be positive.")
    slices: list[tuple[int, int]] = []
    start = 0
    while start < num_events:
        end = min(num_events, start + events_per_frame)
        if end > start:
            slices.append((start, end))
        if max_frames is not None and len(slices) >= max(0, int(max_frames)):
            break
        start += stride_events
    return slices


def _time_frame_slices(
    timestamps: np.ndarray,
    *,
    window: float,
    stride_window: float,
    max_frames: Optional[int],
) -> list[tuple[int, int, float, float]]:
    if window <= 0:
        raise ValueError("--window must be positive.")
    if stride_window <= 0:
        raise ValueError("--stride-window must be positive.")
    if timestamps.size == 0:
        return []

    slices: list[tuple[int, int, float, float]] = []
    start_time = float(timestamps[0])
    end_time = float(timestamps[-1])
    frame_start = start_time
    while frame_start <= end_time:
        frame_end = frame_start + float(window)
        idx0 = int(np.searchsorted(timestamps, frame_start, side="left"))
        idx1 = int(np.searchsorted(timestamps, frame_end, side="left"))
        if idx1 > idx0:
            slices.append((idx0, idx1, frame_start, frame_end))
        if max_frames is not None and len(slices) >= max(0, int(max_frames)):
            break
        frame_start += float(stride_window)
    return slices


def _resolve_frame_windows(
    timestamps: np.ndarray,
    args: argparse.Namespace,
) -> tuple[str, list[tuple[int, int, float, float]]]:
    use_count = args.events_per_frame is not None
    use_time = args.window is not None
    if use_count == use_time:
        raise ValueError("Pass exactly one of --events-per-frame or --window.")

    if use_count:
        ranges = _frame_slices(
            int(timestamps.shape[0]),
            events_per_frame=int(args.events_per_frame),
            stride_events=int(args.stride_events or args.events_per_frame),
            max_frames=args.max_frames,
        )
        windows = []
        for start, end in ranges:
            t0 = float(timestamps[start]) if end > start else 0.0
            t1 = float(timestamps[end - 1]) if end > start else t0
            windows.append((start, end, t0, t1))
        return "events", windows

    return (
        "time",
        _time_frame_slices(
            timestamps,
            window=float(args.window),
            stride_window=float(args.stride_window or args.window),
            max_frames=args.max_frames,
        ),
    )


def _target_size_for_rep(
    rep: str,
    img: np.ndarray,
    *,
    image_sizes: dict[str, tuple[int, int]],
    retain_spatial_dimensions: bool,
    output_size: Optional[list[int]],
    sensor_width: int,
    sensor_height: int,
    temporal_bins: int,
) -> Optional[tuple[int, int]]:
    if rep in image_sizes and img.shape[1] == image_sizes[rep][0] and img.shape[0] == image_sizes[rep][1]:
        return None
    if rep in image_sizes:
        return image_sizes[rep]
    if retain_spatial_dimensions:
        return _default_output_size_for_rep(
            rep,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            temporal_bins=temporal_bins,
        )
    if output_size is not None:
        return int(output_size[0]), int(output_size[1])
    return None


def render_event_representations(args: argparse.Namespace) -> None:
    representations = _representation_list(args.representation)
    if not representations:
        raise ValueError("At least one representation must be requested.")
    unknown = sorted(set(representations) - VALID_REPRESENTATIONS)
    if unknown:
        raise ValueError(f"Unknown representation(s): {unknown}")

    image_sizes = _parse_image_sizes(args.image_sizes)
    unknown_size_reps = sorted(set(image_sizes) - set(representations))
    if unknown_size_reps:
        raise ValueError(f"--image-sizes contains representations not requested by --representation: {unknown_size_reps}")

    event_reps = [rep for rep in representations if rep not in {"rgb", "grayscale", "gray"}]
    tmp_events_path: Optional[Path] = None
    events: np.ndarray = np.zeros((0, 4), dtype=np.float32)
    t: np.ndarray = np.zeros((0,), dtype=np.float32)
    try:
        if event_reps:
            events, t, meta, counters, tmp_events_path, applied_ts_shift_us, npz_already_shifted = _load_events(args)
        else:
            meta = _read_raw_meta_or_empty(args.raw)
            events = np.zeros((0, 4), dtype=np.float32)
            t = np.zeros((0,), dtype=np.float32)
            counters = {}
            applied_ts_shift_us = None
            npz_already_shifted = False

        width, height = _parse_geometry(meta)
        if args.width is not None:
            width = args.width
        if args.height is not None:
            height = args.height
        if width is None or height is None:
            raise ValueError("Could not determine geometry; pass --width/--height.")

        frame_mode, frame_ranges = _resolve_frame_windows(t if event_reps else np.zeros((0,), dtype=np.float32), args)
        if event_reps and not frame_ranges:
            raise RuntimeError("No event frames were generated from the input.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        rgb_index = _build_rgb_index(args.rgb_dir, label_unit=float(args.event_unit)) if args.rgb_dir else []
        manifest = {
            "raw": str(args.raw),
            "output_dir": str(args.output_dir),
            "rgb_dir": None if args.rgb_dir is None else str(args.rgb_dir),
            "render_params": {
                "frame_mode": frame_mode,
                "events_per_frame": None if args.events_per_frame is None else int(args.events_per_frame),
                "stride_events": None
                if args.events_per_frame is None
                else int(args.stride_events or args.events_per_frame),
                "window": None if args.window is None else float(args.window),
                "stride_window": None if args.window is None else float(args.stride_window or args.window),
                "event_unit": float(args.event_unit),
                "ts_shift_us": None if args.ts_shift_us is None else float(args.ts_shift_us),
                "applied_ts_shift_us": None if applied_ts_shift_us is None else float(applied_ts_shift_us),
                "npz_timestamps_already_shifted": bool(npz_already_shifted),
                "endian": args.endian,
                "width": None if args.width is None else int(args.width),
                "height": None if args.height is None else int(args.height),
                "pixel_size": int(args.pixel_size),
                "representation": representations,
                "temporal_bins": int(args.temporal_bins),
                "spatial_bins": int(args.spatial_bins),
                "output_size": None if args.output_size is None else [int(v) for v in args.output_size],
                "image_sizes": {rep: [int(w), int(h)] for rep, (w, h) in sorted(image_sizes.items())},
                "retain_spatial_dimensions": bool(args.retain_spatial_dimensions),
                "grid_x": int(args.grid_x),
                "grid_y": int(args.grid_y),
                "transform": args.transform,
                "transform_scale": args.transform_scale,
                "transform_eps": float(args.transform_eps),
                "max_frames": None if args.max_frames is None else int(args.max_frames),
            },
            "sensor_geometry": {"width": int(width), "height": int(height)},
            "files": [],
        }

        for frame_idx, (start, end, t0, t1) in enumerate(frame_ranges):
            ev = events[start:end]
            stem = f"frame_{frame_idx:06d}_{int(round(t1)):012d}"
            file_entry = {
                "stem": stem,
                "event_start_index": int(start),
                "event_end_index": int(end),
                "window_start_render_units": t0,
                "window_end_render_units": t1,
                "num_events": int(end - start),
                "representations": {},
            }
            for rep in representations:
                if rep in {"rgb", "grayscale", "gray"}:
                    rgb_path = _find_rgb_frame(rgb_index, t1)
                    if rgb_path is None:
                        print(f"Warning: no RGB frame found for {stem}")
                        continue
                    img = _read_rgb_image(rgb_path)
                    if rep in {"grayscale", "gray"}:
                        img = _to_grayscale(img)
                elif rep == "events":
                    if args.grid_x == 1 and args.grid_y == 1:
                        img = events_to_image(ev, int(width), int(height), pixel_size=int(args.pixel_size))
                    else:
                        img = _render_histogram_grid(
                            ev,
                            width=int(width),
                            height=int(height),
                            t0=t0,
                            dt=max(0.0, t1 - t0),
                            plane="xy",
                            time_bins=1,
                            patch_size=int(args.spatial_bins),
                            grid_x=int(args.grid_x),
                            grid_y=int(args.grid_y),
                            retain_spatial_dimensions=bool(args.retain_spatial_dimensions),
                            output_size=image_sizes.get(rep),
                        )
                else:
                    img = _render_histogram_grid(
                        ev,
                        width=int(width),
                        height=int(height),
                        t0=t0,
                        dt=max(0.0, t1 - t0),
                        plane=rep,
                        time_bins=int(args.temporal_bins),
                        patch_size=int(args.spatial_bins),
                        grid_x=int(args.grid_x),
                        grid_y=int(args.grid_y),
                        retain_spatial_dimensions=bool(args.retain_spatial_dimensions),
                        output_size=image_sizes.get(rep),
                    )

                img = _apply_transform(
                    img,
                    rep=rep,
                    transform=args.transform,
                    time_horizontal=rep.startswith("yt"),
                    scale_mode=args.transform_scale,
                    eps=float(args.transform_eps),
                )
                target_size = _target_size_for_rep(
                    rep,
                    img,
                    image_sizes=image_sizes,
                    retain_spatial_dimensions=bool(args.retain_spatial_dimensions),
                    output_size=args.output_size,
                    sensor_width=int(width),
                    sensor_height=int(height),
                    temporal_bins=int(args.temporal_bins),
                )
                if target_size is not None:
                    img = _resize_to(img, target_size)
                final_path = write_image(args.output_dir / f"{stem}_{rep}.png", img)
                file_entry["representations"][rep] = {
                    "path": str(final_path),
                    "image_size": [int(img.shape[1]), int(img.shape[0])],
                    "time_horizontal": bool(rep.startswith("yt")),
                }
            if file_entry["representations"]:
                manifest["files"].append(file_entry)
            if args.show_progress and (frame_idx + 1) % 100 == 0:
                print(f"rendered {frame_idx + 1}/{len(frame_ranges)} frames")

        if any(counters.values()):
            print("Ignored word types:")
            for key, value in counters.items():
                if value:
                    print(f"  {key}: {value}")
        manifest_path = args.output_dir / "render_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {manifest_path}")
    finally:
        if isinstance(events, np.memmap):
            mm = getattr(events, "_mmap", None)
            del t
            del events
            if mm is not None:
                mm.close()
        if tmp_events_path is not None:
            try:
                tmp_events_path.unlink(missing_ok=True)
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode an EVT3 .raw file and render label-free event-image representations by event-count or time windows."
    )
    parser.add_argument("raw", type=Path, help="Path to .raw file")
    parser.add_argument("output_dir", type=Path, help="Output directory for rendered representation images")
    parser.add_argument(
        "--representation",
        type=str,
        default="events",
        help="Representation(s) to render, separated by ';' (default: events).",
    )
    parser.add_argument(
        "--events-per-frame",
        type=int,
        default=None,
        help="Number of collected events to include in each rendered frame.",
    )
    parser.add_argument(
        "--stride-events",
        type=int,
        default=None,
        help="Event-count stride between frame starts (default: same as --events-per-frame).",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        help=(
            "Time window length in event timestamp units. With the default --event-unit 1.0, "
            "this is microseconds for FRED-style EVT3 timestamps."
        ),
    )
    parser.add_argument(
        "--stride-window",
        type=float,
        default=None,
        help="Time stride between frame starts in event timestamp units (default: same as --window).",
    )
    parser.add_argument("--event-unit", type=float, default=1.0)
    parser.add_argument("--ts-shift-us", type=float, default=None)
    parser.add_argument("--endian", choices=["little", "big"], default="little")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--pixel-size", type=int, default=1)
    parser.add_argument("--temporal-bins", type=int, default=64)
    parser.add_argument("--spatial-bins", type=int, default=32)
    parser.add_argument("--output-size", type=int, nargs=2, default=None)
    parser.add_argument(
        "--image-sizes",
        type=str,
        default="",
        help="Per-representation output sizes as rep=WIDTHxHEIGHT entries separated by ';'.",
    )
    parser.add_argument("--retain-spatial-dimensions", action="store_true")
    parser.add_argument("--grid-x", type=int, default=1)
    parser.add_argument("--grid-y", type=int, default=1)
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["none", "spectrogram", "spectrum2d", "spectrogram_bin", "spectrum2d_bin"],
    )
    parser.add_argument(
        "--transform-scale",
        type=str,
        default="linear",
        choices=["linear", "log", "db"],
    )
    parser.add_argument("--transform-eps", type=float, default=1e-6)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--decode-chunk-mb", type=float, default=64.0)
    parser.add_argument(
        "--event-source",
        type=str,
        default="auto",
        choices=["raw", "npz", "auto"],
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    render_event_representations(args)


if __name__ == "__main__":
    main()
