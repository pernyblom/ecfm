from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import groupby
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def _read_raw_tracks(path: Path) -> List[TrackRow]:
    out: List[TrackRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            t = float(parts[0])
            source_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        out.append(TrackRow(t=t, source_id=source_id, x=x, y=y, w=w, h=h))
    return out


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
    args = parser.parse_args()
    requested_folders = set(args.folders) if args.folders else None
    explicit_time_unit = _parse_time_unit(args.track_time_unit)

    folders = sorted(
        [p for p in args.fred_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    for folder in folders:
        if requested_folders is not None and folder.name not in requested_folders:
            continue
        path = folder / args.filename
        if not path.exists():
            continue
        rows = _read_raw_tracks(path)
        if not rows:
            continue
        time_unit = explicit_time_unit if explicit_time_unit is not None else _infer_time_unit(rows)
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
        tracklets = _find_tracklets(
            rows,
            dt=args.frame_dt,
            gap_mult=args.gap_mult,
            max_center_jump=args.max_center_jump,
        )
        out_lines: List[str] = []
        tracklet_id = 0
        base_id = int(folder.name) * args.id_stride
        kept = 0
        for tracklet in tracklets:
            if len(tracklet) < args.min_len:
                continue
            track_id = base_id + tracklet_id
            tracklet_id += 1
            kept += 1
            for row in sorted(tracklet, key=lambda item: item.t):
                out_lines.append(f"{row.t},{track_id},{row.x},{row.y},{row.w},{row.h}")
        if out_lines:
            out_path = folder / args.output
            out_path.write_text("\n".join(out_lines), encoding="utf-8")
            print(
                f"{folder.name}: wrote {out_path} "
                f"({len(out_lines)} rows, {kept} tracklets, time_unit={time_unit:g})"
            )


if __name__ == "__main__":
    main()
