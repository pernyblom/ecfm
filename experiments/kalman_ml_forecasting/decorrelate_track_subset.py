from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.acceleration_field import (
    AccelSample,
    _collect_folder_samples,
    _select_folders,
)
from experiments.kalman_ml_forecasting.utils.config import load_config

TrackKey = tuple[str, int]


def _group_by_track(samples: Iterable[AccelSample]) -> dict[TrackKey, list[AccelSample]]:
    out: dict[TrackKey, list[AccelSample]] = {}
    for sample in samples:
        out.setdefault((sample.folder, int(sample.track_id)), []).append(sample)
    return out


def _feature_target_arrays(
    track_samples: dict[TrackKey, list[AccelSample]],
    selected: Iterable[TrackKey],
    *,
    frame_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    rows_x = []
    rows_y = []
    for key in selected:
        for sample in track_samples[key]:
            rows_x.append(
                [
                    sample.cx_px / frame_w,
                    sample.cy_px / frame_h,
                    sample.vx_px_s,
                    sample.vy_px_s,
                ]
            )
            rows_y.append([sample.ax_px_s2, sample.ay_px_s2])
    return np.asarray(rows_x, dtype=np.float64), np.asarray(rows_y, dtype=np.float64)


def _standardize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1.0e-9)


def _decorrelation_score(
    track_samples: dict[TrackKey, list[AccelSample]],
    selected: Iterable[TrackKey],
    *,
    frame_size: tuple[int, int],
    ridge_lambda: float,
    corr_weight: float,
    r2_weight: float,
) -> dict[str, float]:
    selected = list(selected)
    x_raw, y_raw = _feature_target_arrays(track_samples, selected, frame_size=frame_size)
    if x_raw.shape[0] < 4:
        return {
            "score": float("inf"),
            "samples": float(x_raw.shape[0]),
            "mean_abs_corr": float("inf"),
            "mean_r2": float("inf"),
        }
    x = _standardize(x_raw)
    y = _standardize(y_raw)
    corr = np.abs((x.T @ y) / max(1, x.shape[0] - 1))
    xtx = x.T @ x
    ridge = float(ridge_lambda) * np.eye(xtx.shape[0], dtype=np.float64)
    beta = np.linalg.solve(xtx + ridge, x.T @ y)
    pred = x @ beta
    sse = np.sum((y - pred) ** 2, axis=0)
    sst = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - sse / np.maximum(sst, 1.0e-9)
    mean_abs_corr = float(np.mean(corr))
    mean_r2 = float(np.mean(np.maximum(r2, 0.0)))
    return {
        "score": float(corr_weight * mean_abs_corr + r2_weight * mean_r2),
        "samples": float(x.shape[0]),
        "mean_abs_corr": mean_abs_corr,
        "mean_r2": mean_r2,
    }


def _progress_iter(items, *, desc: str, enabled: bool):
    if not enabled:
        yield from items
        return
    try:
        from tqdm import tqdm
    except ImportError:
        total = len(items)
        start = time.monotonic()
        for idx, item in enumerate(items, start=1):
            if idx == 1 or idx == total or idx % max(1, total // 20) == 0:
                elapsed = time.monotonic() - start
                print(f"{desc}: {idx}/{total} elapsed {elapsed:.1f}s")
            yield item
        return
    yield from tqdm(items, desc=desc, unit="removal", dynamic_ncols=True)


def _select_decorrelated_tracks(
    track_samples: dict[TrackKey, list[AccelSample]],
    *,
    frame_size: tuple[int, int],
    target_tracks: int,
    min_track_samples: int,
    greedy_candidates: int,
    seed: int,
    ridge_lambda: float,
    corr_weight: float,
    r2_weight: float,
    show_progress: bool,
) -> tuple[list[TrackKey], dict[str, float]]:
    eligible = [
        key
        for key, samples in sorted(track_samples.items())
        if len(samples) >= int(min_track_samples)
    ]
    if not eligible:
        raise RuntimeError("No tracks have enough acceleration samples for selection.")
    target_tracks = max(1, min(int(target_tracks), len(eligible)))
    rng = np.random.default_rng(int(seed))
    selected = list(eligible)
    current = _decorrelation_score(
        track_samples,
        selected,
        frame_size=frame_size,
        ridge_lambda=ridge_lambda,
        corr_weight=corr_weight,
        r2_weight=r2_weight,
    )
    removal_count = max(0, len(selected) - target_tracks)
    if removal_count == 0:
        return sorted(selected), current
    removals = range(removal_count)
    for _ in _progress_iter(removals, desc="Selecting decorrelated tracks", enabled=show_progress):
        candidates = list(selected)
        if 0 < greedy_candidates < len(candidates):
            candidates = [candidates[idx] for idx in rng.choice(len(candidates), size=greedy_candidates, replace=False)]
        best_remove = None
        best_score = None
        for key in candidates:
            trial = [item for item in selected if item != key]
            score = _decorrelation_score(
                track_samples,
                trial,
                frame_size=frame_size,
                ridge_lambda=ridge_lambda,
                corr_weight=corr_weight,
                r2_weight=r2_weight,
            )
            if best_score is None or score["score"] < best_score["score"]:
                best_remove = key
                best_score = score
        if best_remove is None or best_score is None:
            break
        selected.remove(best_remove)
        current = best_score
    return sorted(selected), current


def _write_track_manifest(path: Path, selected: list[TrackKey], track_samples: dict[TrackKey, list[AccelSample]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "track_id", "num_accel_samples"])
        writer.writeheader()
        for folder, track_id in selected:
            writer.writerow(
                {
                    "folder": folder,
                    "track_id": int(track_id),
                    "num_accel_samples": len(track_samples[(folder, track_id)]),
                }
            )


def _write_folder_track_split(path: Path, selected: list[TrackKey]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for folder, track_id in selected:
            f.write(f"{folder},{int(track_id)}\n")


def _write_filtered_tracks(
    *,
    cfg: Dict,
    selected: list[TrackKey],
    output_root: Path,
    output_tracks_file: str,
) -> None:
    data_cfg = cfg["data"]
    labels_root = Path(data_cfg["labels_root"])
    source_tracks_file = data_cfg.get("tracks_file", "cleaned_tracks.txt")
    selected_by_folder: dict[str, set[int]] = {}
    for folder, track_id in selected:
        selected_by_folder.setdefault(folder, set()).add(int(track_id))
    for folder, track_ids in selected_by_folder.items():
        source_path = labels_root / folder / source_tracks_file if folder else labels_root / source_tracks_file
        if not source_path.exists():
            continue
        target_path = output_root / folder / output_tracks_file if folder else output_root / output_tracks_file
        target_path.parent.mkdir(parents=True, exist_ok=True)
        kept_lines = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                track_id = int(float(parts[1]))
            except ValueError:
                continue
            if track_id in track_ids:
                kept_lines.append(line)
        target_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select a global track subset whose fitted center acceleration is less linearly "
            "predictable from anchor position and velocity."
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
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--target-tracks", type=int, default=None)
    parser.add_argument("--min-track-samples", type=int, default=8)
    parser.add_argument("--greedy-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    parser.add_argument("--corr-weight", type=float, default=1.0)
    parser.add_argument("--r2-weight", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true", help="Disable selection progress output.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kalman_ml_decorrelated_tracks"))
    parser.add_argument(
        "--write-filtered-tracks",
        action="store_true",
        help="Write per-folder filtered track files under output-dir/filtered_tracks.",
    )
    parser.add_argument("--output-tracks-file", type=str, default="decorrelated_tracks.txt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    folders = _select_folders(args, cfg)
    if not folders:
        raise RuntimeError("No folders selected.")
    samples: list[AccelSample] = []
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
    track_samples = _group_by_track(samples)
    eligible_count = sum(1 for values in track_samples.values() if len(values) >= int(args.min_track_samples))
    if args.target_tracks is None:
        if not 0.0 < float(args.keep_fraction) <= 1.0:
            raise ValueError("--keep-fraction must be in (0, 1].")
        target_tracks = max(1, int(round(eligible_count * float(args.keep_fraction))))
    else:
        target_tracks = int(args.target_tracks)
    frame_size = tuple(int(value) for value in cfg["data"]["frame_size"])
    all_score = _decorrelation_score(
        track_samples,
        [key for key, values in track_samples.items() if len(values) >= int(args.min_track_samples)],
        frame_size=frame_size,
        ridge_lambda=float(args.ridge_lambda),
        corr_weight=float(args.corr_weight),
        r2_weight=float(args.r2_weight),
    )
    selected, selected_score = _select_decorrelated_tracks(
        track_samples,
        frame_size=frame_size,
        target_tracks=target_tracks,
        min_track_samples=int(args.min_track_samples),
        greedy_candidates=int(args.greedy_candidates),
        seed=int(args.seed),
        ridge_lambda=float(args.ridge_lambda),
        corr_weight=float(args.corr_weight),
        r2_weight=float(args.r2_weight),
        show_progress=not bool(args.no_progress),
    )

    output_dir = args.output_dir
    manifest_csv = output_dir / f"{args.split}_decorrelated_tracks.csv"
    track_split_txt = output_dir / f"{args.split}_decorrelated_track_split.txt"
    summary_json = output_dir / f"{args.split}_decorrelated_summary.json"
    _write_track_manifest(manifest_csv, selected, track_samples)
    _write_folder_track_split(track_split_txt, selected)
    if args.write_filtered_tracks:
        _write_filtered_tracks(
            cfg=cfg,
            selected=selected,
            output_root=output_dir / "filtered_tracks",
            output_tracks_file=str(args.output_tracks_file),
        )

    summary = {
        "split": args.split,
        "num_folders": len(folders),
        "num_acceleration_samples": len(samples),
        "num_tracks_total": len(track_samples),
        "num_tracks_eligible": eligible_count,
        "num_tracks_selected": len(selected),
        "min_track_samples": int(args.min_track_samples),
        "target_tracks": int(target_tracks),
        "all_tracks_score": all_score,
        "selected_tracks_score": selected_score,
        "manifest_csv": str(manifest_csv),
        "track_split_txt": str(track_split_txt),
        "filtered_tracks_root": str(output_dir / "filtered_tracks") if args.write_filtered_tracks else None,
        "method": (
            "Greedy track removal. Score is weighted mean absolute Pearson correlation "
            "and ridge linear R2 from [cx, cy, vx, vy] to [ax, ay]."
        ),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
