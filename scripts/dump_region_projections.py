from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="datasets/THU-EACT-50-CHL")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="outputs/tmp_regions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=346)
    parser.add_argument("--image-height", type=int, default=260)
    parser.add_argument("--time-bins", type=int, default=64)
    parser.add_argument("--min-spatial-frac", type=float, default=0.1)
    parser.add_argument("--max-spatial-frac", type=float, default=1.0)
    parser.add_argument("--min-temporal-frac", type=float, default=0.1)
    parser.add_argument("--max-temporal-frac", type=float, default=1.0)
    return parser.parse_args()


def load_file_list(root: Path, split: str) -> List[Path]:
    list_path = root / f"{split}.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing split list: {list_path}")
    entries = [line.strip().split()[0] for line in list_path.read_text().splitlines() if line.strip()]
    files = []
    for entry in entries:
        name = Path(entry).name
        candidate = root / name
        if candidate.exists():
            files.append(candidate)
    if not files:
        raise FileNotFoundError("No dataset files found for split list")
    return files


def normalize_time(events: np.ndarray) -> Tuple[np.ndarray, float]:
    t = events[:, 2]
    t_min = float(t.min())
    t_max = float(t.max())
    denom = max(1e-6, t_max - t_min)
    events = events.copy()
    events[:, 2] = (t - t_min) / denom
    return events, denom


def sample_region(
    rng: np.random.Generator,
    image_width: int,
    image_height: int,
    min_spatial_frac: float,
    max_spatial_frac: float,
    min_temporal_frac: float,
    max_temporal_frac: float,
) -> Dict[str, float]:
    min_spatial_frac = max(0.0, min(min_spatial_frac, 1.0))
    max_spatial_frac = max(min_spatial_frac, min(max_spatial_frac, 1.0))
    min_temporal_frac = max(0.0, min(min_temporal_frac, 1.0))
    max_temporal_frac = max(min_temporal_frac, min(max_temporal_frac, 1.0))

    min_dx = max(1, int(round(image_width * min_spatial_frac)))
    max_dx = max(min_dx, int(round(image_width * max_spatial_frac)))
    min_dy = max(1, int(round(image_height * min_spatial_frac)))
    max_dy = max(min_dy, int(round(image_height * max_spatial_frac)))
    min_dt = min_temporal_frac
    max_dt = max_temporal_frac

    dx = int(rng.integers(min_dx, max_dx + 1))
    dy = int(rng.integers(min_dy, max_dy + 1))
    dt = float(rng.uniform(min_dt, max_dt))

    x = int(rng.integers(0, max(1, image_width - dx + 1)))
    y = int(rng.integers(0, max(1, image_height - dy + 1)))
    t = float(rng.uniform(0.0, max(1e-6, 1.0 - dt)))

    return {"x": x, "y": y, "t": t, "dx": dx, "dy": dy, "dt": dt}


def _hist_xy(events: np.ndarray, region: Dict[str, float], polarity: int) -> np.ndarray:
    x0, y0 = int(region["x"]), int(region["y"])
    dx, dy = int(region["dx"]), int(region["dy"])
    t0, dt = float(region["t"]), float(region["dt"])

    mask = (
        (events[:, 0] >= x0)
        & (events[:, 0] < x0 + dx)
        & (events[:, 1] >= y0)
        & (events[:, 1] < y0 + dy)
        & (events[:, 2] >= t0)
        & (events[:, 2] < t0 + dt)
        & (events[:, 3] == polarity)
    )
    sub = events[mask]
    hist = np.zeros((dy, dx), dtype=np.float32)
    if sub.shape[0] == 0:
        return hist
    x = (sub[:, 0] - x0).astype(np.int64)
    y = (sub[:, 1] - y0).astype(np.int64)
    np.add.at(hist, (y, x), 1.0)
    return hist


def _hist_xt(events: np.ndarray, region: Dict[str, float], polarity: int, time_bins: int) -> np.ndarray:
    x0, y0 = int(region["x"]), int(region["y"])
    dx, dy = int(region["dx"]), int(region["dy"])
    t0, dt = float(region["t"]), float(region["dt"])

    mask = (
        (events[:, 0] >= x0)
        & (events[:, 0] < x0 + dx)
        & (events[:, 1] >= y0)
        & (events[:, 1] < y0 + dy)
        & (events[:, 2] >= t0)
        & (events[:, 2] < t0 + dt)
        & (events[:, 3] == polarity)
    )
    sub = events[mask]
    hist = np.zeros((time_bins, dx), dtype=np.float32)
    if sub.shape[0] == 0:
        return hist
    x = (sub[:, 0] - x0).astype(np.int64)
    t_norm = (sub[:, 2] - t0) / max(dt, 1e-6)
    t_idx = np.clip((t_norm * time_bins).astype(np.int64), 0, time_bins - 1)
    np.add.at(hist, (t_idx, x), 1.0)
    return hist


def _hist_yt(events: np.ndarray, region: Dict[str, float], polarity: int, time_bins: int) -> np.ndarray:
    x0, y0 = int(region["x"]), int(region["y"])
    dx, dy = int(region["dx"]), int(region["dy"])
    t0, dt = float(region["t"]), float(region["dt"])

    mask = (
        (events[:, 0] >= x0)
        & (events[:, 0] < x0 + dx)
        & (events[:, 1] >= y0)
        & (events[:, 1] < y0 + dy)
        & (events[:, 2] >= t0)
        & (events[:, 2] < t0 + dt)
        & (events[:, 3] == polarity)
    )
    sub = events[mask]
    hist = np.zeros((time_bins, dy), dtype=np.float32)
    if sub.shape[0] == 0:
        return hist
    y = (sub[:, 1] - y0).astype(np.int64)
    t_norm = (sub[:, 2] - t0) / max(dt, 1e-6)
    t_idx = np.clip((t_norm * time_bins).astype(np.int64), 0, time_bins - 1)
    np.add.at(hist, (t_idx, y), 1.0)
    return hist


def normalize_two_channel(hist0: np.ndarray, hist1: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    max0 = float(hist0.max()) if hist0.size > 0 else 0.0
    max1 = float(hist1.max()) if hist1.size > 0 else 0.0
    out0 = hist0 / max0 if max0 > 0 else hist0
    out1 = hist1 / max1 if max1 > 0 else hist1
    return np.stack([out0, out1], axis=0), [max0, max1]


def save_channel_png(hist: np.ndarray, path: Path) -> None:
    img = np.clip(hist * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = load_file_list(root, args.split)
    if args.num_samples < len(files):
        files = list(rng.choice(files, size=args.num_samples, replace=False))

    for idx, path in enumerate(files):
        events = np.load(path).astype(np.float32)
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError(f"Unexpected event shape in {path}: {events.shape}")
        events, seq_len_sec = normalize_time(events)
        region = sample_region(
            rng,
            args.image_width,
            args.image_height,
            min_spatial_frac=args.min_spatial_frac,
            max_spatial_frac=args.max_spatial_frac,
            min_temporal_frac=args.min_temporal_frac,
            max_temporal_frac=args.max_temporal_frac,
        )

        hist_xy_0 = _hist_xy(events, region, polarity=0)
        hist_xy_1 = _hist_xy(events, region, polarity=1)
        xy_norm, xy_max = normalize_two_channel(hist_xy_0, hist_xy_1)

        hist_xt_0 = _hist_xt(events, region, polarity=0, time_bins=args.time_bins)
        hist_xt_1 = _hist_xt(events, region, polarity=1, time_bins=args.time_bins)
        xt_norm, xt_max = normalize_two_channel(hist_xt_0, hist_xt_1)

        hist_yt_0 = _hist_yt(events, region, polarity=0, time_bins=args.time_bins)
        hist_yt_1 = _hist_yt(events, region, polarity=1, time_bins=args.time_bins)
        yt_norm, yt_max = normalize_two_channel(hist_yt_0, hist_yt_1)

        stem = f"sample_{idx:03d}_{path.stem}"
        sample_dir = out_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_dir / "xy.npy", xy_norm.astype(np.float32))
        np.save(sample_dir / "xt.npy", xt_norm.astype(np.float32))
        np.save(sample_dir / "yt.npy", yt_norm.astype(np.float32))

        save_channel_png(xy_norm[0], sample_dir / "xy_p0.png")
        save_channel_png(xy_norm[1], sample_dir / "xy_p1.png")
        save_channel_png(xt_norm[0], sample_dir / "xt_p0.png")
        save_channel_png(xt_norm[1], sample_dir / "xt_p1.png")
        save_channel_png(yt_norm[0], sample_dir / "yt_p0.png")
        save_channel_png(yt_norm[1], sample_dir / "yt_p1.png")

        dt_sec = float(region["dt"]) * seq_len_sec
        t0_sec = float(region["t"]) * seq_len_sec
        rate_xy = [
            float(hist_xy_0.sum()) / max(1e-6, region["dx"] * region["dy"] * dt_sec),
            float(hist_xy_1.sum()) / max(1e-6, region["dx"] * region["dy"] * dt_sec),
        ]
        rate_xt = [
            float(hist_xt_0.sum()) / max(1e-6, region["dx"] * dt_sec),
            float(hist_xt_1.sum()) / max(1e-6, region["dx"] * dt_sec),
        ]
        rate_yt = [
            float(hist_yt_0.sum()) / max(1e-6, region["dy"] * dt_sec),
            float(hist_yt_1.sum()) / max(1e-6, region["dy"] * dt_sec),
        ]

        meta = {
            "file": str(path),
            "image_width": args.image_width,
            "image_height": args.image_height,
            "time_bins": args.time_bins,
            "sequence_duration_sec": seq_len_sec,
            "region": region,
            "region_t0_sec": t0_sec,
            "region_dt_sec": dt_sec,
            "plane_max": {"xy": xy_max, "xt": xt_max, "yt": yt_max},
            "event_counts": {
                "xy": [float(hist_xy_0.sum()), float(hist_xy_1.sum())],
                "xt": [float(hist_xt_0.sum()), float(hist_xt_1.sum())],
                "yt": [float(hist_yt_0.sum()), float(hist_yt_1.sum())],
            },
            "event_rates": {"xy": rate_xy, "xt": rate_xt, "yt": rate_yt},
        }
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(files)} samples to {out_dir}")


if __name__ == "__main__":
    main()
