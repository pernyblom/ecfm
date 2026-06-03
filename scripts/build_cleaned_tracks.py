from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import groupby
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


LOGGER = logging.getLogger("build_cleaned_tracks")


@dataclass
class TrackRow:
    t: float
    source_id: int
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class ReadStats:
    total_lines: int = 0
    blank_lines: int = 0
    short_lines: int = 0
    parse_errors: int = 0
    non_positive_boxes: int = 0

    @property
    def skipped_lines(self) -> int:
        return self.blank_lines + self.short_lines + self.parse_errors + self.non_positive_boxes


def _read_raw_tracks(path: Path) -> Tuple[List[TrackRow], ReadStats]:
    out: List[TrackRow] = []
    stats = ReadStats()
    for line in path.read_text(encoding="utf-8").splitlines():
        stats.total_lines += 1
        line = line.strip()
        if not line:
            stats.blank_lines += 1
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            stats.short_lines += 1
            continue
        try:
            t = float(parts[0])
            source_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except ValueError:
            stats.parse_errors += 1
            continue
        if w <= 0 or h <= 0:
            stats.non_positive_boxes += 1
            continue
        out.append(TrackRow(t=t, source_id=source_id, x=x, y=y, w=w, h=h))
    return out, stats


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _infer_time_unit(rows: List[TrackRow]) -> float:
    times = sorted({row.t for row in rows})
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    median_dt = _median(deltas)
    if median_dt is not None and median_dt > 1.0:
        return 1e-6
    return 1.0


def _center_distance(a: TrackRow, b: TrackRow) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def _time_groups(rows: Iterable[TrackRow]) -> Iterable[Tuple[float, List[TrackRow]]]:
    for t, group in groupby(rows, key=lambda row: row.t):
        yield t, list(group)


def _split_source_id_rows(
    rows: List[TrackRow],
    *,
    dt: float,
    gap_mult: float,
    max_center_jump: float,
) -> List[List[TrackRow]]:
    rows = sorted(rows, key=lambda row: (row.t, row.x, row.y, row.w, row.h))
    active: List[List[TrackRow]] = []
    finished: List[List[TrackRow]] = []
    max_gap = dt * gap_mult

    for t, detections in _time_groups(rows):
        still_active: List[List[TrackRow]] = []
        for tracklet in active:
            if t - tracklet[-1].t <= max_gap:
                still_active.append(tracklet)
            else:
                finished.append(tracklet)
        active = still_active

        candidates: List[Tuple[float, int, int]] = []
        for track_idx, tracklet in enumerate(active):
            last = tracklet[-1]
            gap = t - last.t
            if gap <= 0 or gap > max_gap:
                continue
            for det_idx, det in enumerate(detections):
                distance = _center_distance(last, det)
                if distance <= max_center_jump:
                    candidates.append((distance, track_idx, det_idx))

        used_tracks = set()
        used_detections = set()
        for _, track_idx, det_idx in sorted(candidates):
            if track_idx in used_tracks or det_idx in used_detections:
                continue
            active[track_idx].append(detections[det_idx])
            used_tracks.add(track_idx)
            used_detections.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                active.append([det])

    finished.extend(active)
    return finished


def _find_tracklets(
    rows: List[TrackRow],
    *,
    dt: float,
    gap_mult: float,
    max_center_jump: float,
) -> List[List[TrackRow]]:
    by_source_id: Dict[int, List[TrackRow]] = {}
    for row in rows:
        by_source_id.setdefault(row.source_id, []).append(row)

    tracklets: List[List[TrackRow]] = []
    for source_id in sorted(by_source_id):
        tracklets.extend(
            _split_source_id_rows(
                by_source_id[source_id],
                dt=dt,
                gap_mult=gap_mult,
                max_center_jump=max_center_jump,
            )
        )
    tracklets.sort(key=lambda item: (item[0].t, item[0].source_id, -len(item)))
    return tracklets


def _parse_time_unit(value: str) -> Optional[float]:
    if value == "auto":
        return None
    return float(value)


def _format_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.3g} us"
    if value < 1.0:
        return f"{value * 1e3:.3g} ms"
    return f"{value:.3g} s"


def _configure_logging(*, log_file: Optional[Path], verbose: bool, quiet: bool) -> None:
    handlers: List[logging.Handler] = []
    if not quiet:
        handlers.append(logging.StreamHandler())
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cleaned_tracks.txt from tracks.txt tracklets."
    )
    parser.add_argument(
        "--fred-root",
        type=Path,
        default=Path("datasets/FRED"),
        help="Root path of FRED dataset (default: datasets/FRED)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="tracks.txt",
        help="Filename to parse (default: tracks.txt)",
    )
    parser.add_argument(
        "--track-time-unit",
        type=str,
        default="auto",
        help="Multiplier for input timestamps, or auto (default: auto).",
    )
    parser.add_argument(
        "--frame-dt",
        type=float,
        default=0.033333,
        help="Expected frame delta in seconds (default: 0.033333)",
    )
    parser.add_argument(
        "--gap-mult",
        type=float,
        default=1.5,
        help="Gap multiplier to split segments (default: 1.5)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=10,
        help="Minimum segment length to keep (default: 10)",
    )
    parser.add_argument(
        "--max-center-jump",
        type=float,
        default=160.0,
        help="Maximum center movement in pixels between frames (default: 160).",
    )
    parser.add_argument(
        "--id-stride",
        type=int,
        default=100000,
        help="Per-folder output ID stride (default: 100000).",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Optional folder names to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cleaned_tracks.txt",
        help="Output filename per folder (default: cleaned_tracks.txt)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path for a detailed processing log.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include extra per-source debug details in the log.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not write log messages to stdout. Useful together with --log-file.",
    )
    args = parser.parse_args()
    _configure_logging(log_file=args.log_file, verbose=args.verbose, quiet=args.quiet)

    requested_folders = set(args.folders) if args.folders else None
    explicit_time_unit = _parse_time_unit(args.track_time_unit)

    if not args.fred_root.exists():
        raise FileNotFoundError(f"FRED root does not exist: {args.fred_root}")
    if not args.fred_root.is_dir():
        raise NotADirectoryError(f"FRED root is not a directory: {args.fred_root}")

    folders = sorted(
        [p for p in args.fred_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    LOGGER.info("Scanning FRED root: %s", args.fred_root)
    LOGGER.info(
        "Source=%s output=%s frame_dt=%s gap_mult=%g max_gap=%s min_len=%d "
        "max_center_jump=%g id_stride=%d time_unit=%s",
        args.filename,
        args.output,
        _format_seconds(args.frame_dt),
        args.gap_mult,
        _format_seconds(args.frame_dt * args.gap_mult),
        args.min_len,
        args.max_center_jump,
        args.id_stride,
        args.track_time_unit,
    )
    if requested_folders is not None:
        LOGGER.info("Folder filter enabled: %s", ", ".join(sorted(requested_folders)))
    LOGGER.info("Found %d numeric folders before filtering.", len(folders))

    processed_folders = 0
    missing_inputs = 0
    empty_inputs = 0
    folders_without_output = 0
    folders_written = 0
    total_valid_rows = 0
    total_written_rows = 0
    total_tracklets = 0
    total_kept_tracklets = 0

    for folder in folders:
        if requested_folders is not None and folder.name not in requested_folders:
            LOGGER.debug("Skipping folder %s: not in --folders filter.", folder.name)
            continue
        processed_folders += 1
        path = folder / args.filename
        if not path.exists():
            missing_inputs += 1
            LOGGER.info("Skipping folder %s: missing input file %s.", folder.name, path)
            continue
        rows, read_stats = _read_raw_tracks(path)
        if not rows:
            empty_inputs += 1
            LOGGER.info(
                "Skipping folder %s: no valid rows in %s "
                "(lines=%d skipped=%d short=%d parse_errors=%d non_positive_boxes=%d).",
                folder.name,
                path,
                read_stats.total_lines,
                read_stats.skipped_lines,
                read_stats.short_lines,
                read_stats.parse_errors,
                read_stats.non_positive_boxes,
            )
            continue
        total_valid_rows += len(rows)
        time_unit = explicit_time_unit if explicit_time_unit is not None else _infer_time_unit(rows)
        unique_times = sorted({row.t for row in rows})
        median_dt = _median([b - a for a, b in zip(unique_times, unique_times[1:]) if b > a])
        source_ids = {row.source_id for row in rows}
        if explicit_time_unit is None:
            if time_unit == 1.0:
                time_reason = "auto kept seconds because median positive timestamp delta is <= 1"
            else:
                time_reason = "auto interpreted timestamps as microseconds because median positive timestamp delta is > 1"
        else:
            time_reason = "explicit --track-time-unit"
        LOGGER.info(
            "Folder %s: read %d valid rows from %s across %d source IDs and %d timestamps "
            "(skipped=%d, median_dt=%s before scaling).",
            folder.name,
            len(rows),
            path,
            len(source_ids),
            len(unique_times),
            read_stats.skipped_lines,
            "n/a" if median_dt is None else f"{median_dt:g}",
        )
        LOGGER.info("Folder %s: using time_unit=%g (%s).", folder.name, time_unit, time_reason)
        if time_unit != 1.0:
            rows = [
                TrackRow(
                    t=row.t * time_unit,
                    source_id=row.source_id,
                    x=row.x,
                    y=row.y,
                    w=row.w,
                    h=row.h,
                )
                for row in rows
            ]
            LOGGER.info("Folder %s: scaled timestamps by %g.", folder.name, time_unit)
        tracklets = _find_tracklets(
            rows,
            dt=args.frame_dt,
            gap_mult=args.gap_mult,
            max_center_jump=args.max_center_jump,
        )
        total_tracklets += len(tracklets)
        if LOGGER.isEnabledFor(logging.DEBUG):
            lengths_by_source: Dict[int, List[int]] = {}
            for tracklet in tracklets:
                lengths_by_source.setdefault(tracklet[0].source_id, []).append(len(tracklet))
            for source_id, lengths in sorted(lengths_by_source.items()):
                LOGGER.debug(
                    "Folder %s source_id=%s: produced %d candidate tracklets, "
                    "length min/median/max=%d/%s/%d.",
                    folder.name,
                    source_id,
                    len(lengths),
                    min(lengths),
                    _median(lengths),
                    max(lengths),
                )
        out_lines: List[str] = []
        tracklet_id = 0
        base_id = int(folder.name) * args.id_stride
        kept = 0
        dropped_short = 0
        dropped_rows = 0
        for tracklet in tracklets:
            if len(tracklet) < args.min_len:
                dropped_short += 1
                dropped_rows += len(tracklet)
                continue
            track_id = base_id + tracklet_id
            tracklet_id += 1
            kept += 1
            for row in sorted(tracklet, key=lambda item: item.t):
                out_lines.append(f"{row.t},{track_id},{row.x},{row.y},{row.w},{row.h}")
        if out_lines:
            out_path = folder / args.output
            out_path.write_text("\n".join(out_lines), encoding="utf-8")
            folders_written += 1
            total_written_rows += len(out_lines)
            total_kept_tracklets += kept
            LOGGER.info(
                "Folder %s: wrote %s (%d rows, %d kept tracklets, %d dropped short "
                "tracklets containing %d rows, base_id=%d).",
                folder.name,
                out_path,
                len(out_lines),
                kept,
                dropped_short,
                dropped_rows,
                base_id,
            )
        else:
            folders_without_output += 1
            LOGGER.info(
                "Folder %s: wrote nothing because all %d candidate tracklets were shorter "
                "than min_len=%d (%d rows dropped).",
                folder.name,
                len(tracklets),
                args.min_len,
                dropped_rows,
            )

    LOGGER.info(
        "Done. processed=%d written=%d missing_inputs=%d empty_inputs=%d no_output=%d "
        "valid_rows=%d candidate_tracklets=%d kept_tracklets=%d written_rows=%d.",
        processed_folders,
        folders_written,
        missing_inputs,
        empty_inputs,
        folders_without_output,
        total_valid_rows,
        total_tracklets,
        total_kept_tracklets,
        total_written_rows,
    )
    if args.log_file is not None:
        LOGGER.info("Detailed log written to %s", args.log_file)


if __name__ == "__main__":
    main()
