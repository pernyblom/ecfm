from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Sample:
    t: float
    x0: float
    y0: float
    x1: float
    y1: float


def _read_coordinates(path: Path) -> List[Sample]:
    out: List[Sample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        ts_part, coords_part = line.split(":", 1)
        ts_part = ts_part.strip()
        coords_part = coords_part.strip()
        if not ts_part or not coords_part:
            continue
        try:
            t = float(ts_part)
        except ValueError:
            continue
        parts = [p.strip() for p in coords_part.split(",")]
        if len(parts) < 4:
            continue
        try:
            x0, y0, x1, y1 = map(float, parts[:4])
        except ValueError:
            continue
        out.append(Sample(t=t, x0=x0, y0=y0, x1=x1, y1=y1))
    out.sort(key=lambda s: s.t)
    return out


def _find_segments(samples: List[Sample], *, dt: float, gap_mult: float) -> List[List[Sample]]:
    if not samples:
        return []
    segments: List[List[Sample]] = []
    cur: List[Sample] = [samples[0]]
    seen_ts = {samples[0].t}
    for prev, cur_s in zip(samples, samples[1:]):
        gap = cur_s.t - prev.t
        duplicate = cur_s.t in seen_ts
        if duplicate or gap > dt * gap_mult or gap <= 0:
            segments.append(cur)
            cur = [cur_s]
            seen_ts = {cur_s.t}
        else:
            cur.append(cur_s)
            seen_ts.add(cur_s.t)
    if cur:
        segments.append(cur)
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cleaned_tracks.txt from coordinates segments."
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
        default="coordinates.txt",
        help="Filename to parse (default: coordinates.txt)",
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
        "--output",
        type=str,
        default="cleaned_tracks.txt",
        help="Output filename per folder (default: cleaned_tracks.txt)",
    )
    args = parser.parse_args()

    folders = sorted(
        [p for p in args.fred_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    for folder in folders:
        path = folder / args.filename
        if not path.exists():
            continue
        samples = _read_coordinates(path)
        segments = _find_segments(samples, dt=args.frame_dt, gap_mult=args.gap_mult)
        out_lines: List[str] = []
        seg_id = 0
        base_id = int(folder.name) * 1000
        for seg in segments:
            if len(seg) < args.min_len:
                continue
            track_id = base_id + seg_id
            seg_id += 1
            for s in seg:
                w = s.x1 - s.x0
                h = s.y1 - s.y0
                if w < 0 or h < 0:
                    continue
                out_lines.append(
                    f"{s.t},{track_id},{s.x0},{s.y0},{w},{h}"
                )
        if out_lines:
            out_path = folder / args.output
            out_path.write_text("\n".join(out_lines), encoding="utf-8")
            print(f"{folder.name}: wrote {out_path} ({len(out_lines)} rows)")


if __name__ == "__main__":
    main()
