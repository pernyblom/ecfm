"""Estimate a fixed CSTR3 count scale from a FRED training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_evt3_yolo_frames import (
    _load_events_from_npz,
    _load_events_from_raw,
    _parse_geometry,
    _parse_label_time,
    _read_ts_shift_us,
)


def _read_split(path: Path) -> list[str]:
    return [line.strip().strip("/\\") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _window_bounds(label_time: float, window: float, mode: str) -> tuple[float, float]:
    if mode == "trailing":
        start, end = label_time - window, label_time
    elif mode == "center":
        start, end = label_time - window / 2.0, label_time + window / 2.0
    elif mode == "leading":
        start, end = label_time, label_time + window
    else:
        raise ValueError(f"Unknown window mode: {mode}")
    start = max(0.0, start)
    return start, max(start, end)


def _grow_histogram(histogram: np.ndarray, size: int) -> np.ndarray:
    if size <= histogram.size:
        return histogram
    grown = np.zeros(size, dtype=np.int64)
    grown[: histogram.size] = histogram
    return grown


def _histogram_percentile(histogram: np.ndarray, percentile: float) -> int:
    total = int(histogram.sum())
    if total == 0:
        return 0
    target = max(1, int(np.ceil(total * float(percentile) / 100.0)))
    return int(np.searchsorted(np.cumsum(histogram), target, side="left"))


def _load_folder_events(folder_root: Path, args: argparse.Namespace):
    raw_path = folder_root / "Event" / "events.raw"
    shift = args.ts_shift_us
    if shift is None and raw_path.exists():
        shift = _read_ts_shift_us(raw_path)
    source = args.event_source
    if source == "auto":
        source = "npz" if raw_path.with_name("output_events.npz").exists() else "raw"
    if source == "npz":
        events, times, meta, _, _, _ = _load_events_from_npz(
            raw_path, event_unit=args.event_unit, ts_shift_us=shift
        )
        return events, times, meta, None
    events, times, meta, _, temporary_path = _load_events_from_raw(
        raw_path,
        endian=args.endian,
        decode_chunk_mb=args.decode_chunk_mb,
        event_unit=args.event_unit,
        ts_shift_us=shift,
        show_progress=args.show_progress,
    )
    return events, times, meta, temporary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--fred-root", type=Path, default=Path("datasets/FRED"))
    parser.add_argument("--window", type=float, default=33333.0)
    parser.add_argument("--window-mode", choices=["trailing", "center", "leading"], default="trailing")
    parser.add_argument("--label-unit", type=float, default=1.0)
    parser.add_argument("--event-unit", type=float, default=1.0)
    parser.add_argument("--ts-shift-us", type=float, default=None)
    parser.add_argument("--event-source", choices=["auto", "npz", "raw"], default="auto")
    parser.add_argument("--endian", choices=["little", "big"], default="little")
    parser.add_argument("--decode-chunk-mb", type=float, default=64.0)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[99.0, 99.9, 99.99])
    parser.add_argument("--recommended-percentile", type=float, default=99.9)
    parser.add_argument("--only-with-boxes", action="store_true")
    parser.add_argument("--max-label-files-per-folder", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    requested_percentiles = sorted(set(args.percentiles + [args.recommended_percentile]))
    if any(value <= 0.0 or value > 100.0 for value in requested_percentiles):
        raise ValueError("Percentiles must be in (0, 100].")
    histogram = np.zeros(1, dtype=np.int64)
    folder_summaries = []
    total_windows = 0

    for folder in _read_split(args.split_file):
        folder_root = args.fred_root / folder
        labels_dir = folder_root / "Event_YOLO"
        temporary_path = None
        events = None
        times = None
        try:
            events, times, meta, temporary_path = _load_folder_events(folder_root, args)
            width, height = _parse_geometry(meta)
            if width is None or height is None:
                raise ValueError(f"Could not determine sensor geometry for folder {folder}.")
            label_files = sorted(
                labels_dir.glob("*.txt"),
                key=lambda path: (_parse_label_time(path) is None, _parse_label_time(path) or 0, path.name),
            )
            if args.only_with_boxes:
                label_files = [path for path in label_files if path.stat().st_size > 0]
            if args.max_label_files_per_folder is not None:
                label_files = label_files[: max(0, args.max_label_files_per_folder)]
            folder_windows = 0
            for label_path in label_files:
                label_time_raw = _parse_label_time(label_path)
                if label_time_raw is None:
                    continue
                label_time = float(label_time_raw) * args.label_unit
                start, end = _window_bounds(label_time, args.window * args.label_unit, args.window_mode)
                begin_idx = int(np.searchsorted(times, start, side="left"))
                end_idx = int(np.searchsorted(times, end, side="left"))
                window_events = events[begin_idx:end_idx]
                if window_events.size:
                    x = window_events[:, 0].astype(np.int64)
                    y = window_events[:, 1].astype(np.int64)
                    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                    _, counts = np.unique(y[valid] * int(width) + x[valid], return_counts=True)
                    if counts.size:
                        count_histogram = np.bincount(counts)
                        histogram = _grow_histogram(histogram, count_histogram.size)
                        histogram[: count_histogram.size] += count_histogram
                folder_windows += 1
                total_windows += 1
            folder_summaries.append({"folder": folder, "windows": folder_windows})
            print(f"{folder}: {folder_windows} windows")
        finally:
            if isinstance(events, np.memmap):
                mmap = getattr(events, "_mmap", None)
                del times
                del events
                if mmap is not None:
                    mmap.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    percentile_counts = {
        f"{value:g}": _histogram_percentile(histogram, value)
        for value in requested_percentiles
    }
    observed = np.nonzero(histogram)[0]
    result = {
        "split_file": str(args.split_file),
        "fred_root": str(args.fred_root),
        "window": float(args.window),
        "window_mode": args.window_mode,
        "label_unit": float(args.label_unit),
        "event_unit": float(args.event_unit),
        "active_pixel_observations": int(histogram.sum()),
        "windows": int(total_windows),
        "percentile_counts": percentile_counts,
        "observed_max_count": int(observed[-1]) if observed.size else 0,
        "recommended_percentile": float(args.recommended_percentile),
        "recommended_max_count": percentile_counts[f"{args.recommended_percentile:g}"],
        "folders": folder_summaries,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
