from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class Sample:
    t: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


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


def _find_segments(
    samples: List[Sample], *, dt: float, gap_mult: float
) -> List[List[Sample]]:
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


def _velocity_outliers(
    segment: List[Sample], *, dt: float, z_threshold: float, mult_threshold: float
) -> List[Tuple[int, float, float]]:
    if len(segment) < 3:
        return []
    centers = np.array([[s.cx, s.cy] for s in segment], dtype=np.float32)
    d = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
    v = d / dt
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median))) + 1e-6
    z = (v - median) / mad
    outliers = []
    for i, (vi, zi) in enumerate(zip(v, z), start=1):
        if zi > z_threshold or vi > median * mult_threshold:
            outliers.append((i, float(vi), float(zi)))
    return outliers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find continuous coordinate segments and flag velocity outliers."
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
        "--z-threshold",
        type=float,
        default=6.0,
        help="MAD z-threshold for velocity outliers (default: 6.0)",
    )
    parser.add_argument(
        "--mult-threshold",
        type=float,
        default=3.0,
        help="Velocity multiplier threshold vs median (default: 3.0)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=10,
        help="Minimum segment length to consider (default: 10)",
    )
    args = parser.parse_args()

    folders = sorted(
        [p for p in args.fred_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    total_reasonable = 0
    total_unreasonable = 0

    for folder in folders:
        path = folder / args.filename
        if not path.exists():
            continue
        samples = _read_coordinates(path)
        segments = _find_segments(samples, dt=args.frame_dt, gap_mult=args.gap_mult)
        bad_segments = []
        for seg in segments:
            if len(seg) < args.min_len:
                continue
            outliers = _velocity_outliers(
                seg,
                dt=args.frame_dt,
                z_threshold=args.z_threshold,
                mult_threshold=args.mult_threshold,
            )
            if outliers:
                bad_segments.append((seg, outliers))
            else:
                total_reasonable += 1

        if bad_segments:
            total_unreasonable += len(bad_segments)
            print(f"{folder.name}: {len(bad_segments)} unreasonable segments")
            for seg, outliers in bad_segments[:5]:
                t0 = seg[0].t
                t1 = seg[-1].t
                print(
                    f"  segment len={len(seg)} t=[{t0:.6f}, {t1:.6f}] "
                    f"outliers={len(outliers)} (first: idx={outliers[0][0]} "
                    f"v={outliers[0][1]:.2f} z={outliers[0][2]:.2f})"
                )

    print(f"Reasonable segments: {total_reasonable}")
    print(f"Unreasonable segments: {total_unreasonable}")


if __name__ == "__main__":
    main()
