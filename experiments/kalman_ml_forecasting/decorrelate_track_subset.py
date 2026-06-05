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


def _sample_features(sample: AccelSample, *, frame_size: tuple[int, int], feature_mode: str) -> list[float]:
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    cx_norm = sample.cx_px / frame_w
    cy_norm = sample.cy_px / frame_h
    dx = cx_norm - 0.5
    dy = cy_norm - 0.5
    vx = float(sample.vx_px_s)
    vy = float(sample.vy_px_s)
    if feature_mode == "raw":
        return [cx_norm, cy_norm, vx, vy]
    if feature_mode == "centered":
        return [dx, dy, vx, vy]
    if feature_mode == "motion_priors":
        return [dx, dy, vx, vy, float(np.hypot(dx, dy)), float(np.hypot(vx, vy))]
    raise ValueError("feature_mode must be one of: raw, centered, motion_priors.")


def _group_by_track(samples: Iterable[AccelSample]) -> dict[TrackKey, list[AccelSample]]:
    out: dict[TrackKey, list[AccelSample]] = {}
    for sample in samples:
        out.setdefault((sample.folder, int(sample.track_id)), []).append(sample)
    return out


def _track_stats(samples: list[AccelSample], *, frame_size: tuple[int, int], feature_mode: str) -> dict[str, np.ndarray | float]:
    x_rows = [_sample_features(sample, frame_size=frame_size, feature_mode=feature_mode) for sample in samples]
    y_rows = [[sample.ax_px_s2, sample.ay_px_s2] for sample in samples]
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    return {
        "n": float(x.shape[0]),
        "sum_x": x.sum(axis=0),
        "sum_y": y.sum(axis=0),
        "sum_xx": x.T @ x,
        "sum_xy": x.T @ y,
        "sum_yy": y.T @ y,
    }


def _add_stats(left: dict[str, np.ndarray | float], right: dict[str, np.ndarray | float], *, sign: float = 1.0) -> dict[str, np.ndarray | float]:
    return {key: left[key] + sign * right[key] for key in left}


def _sum_stats(stats: Iterable[dict[str, np.ndarray | float]]) -> dict[str, np.ndarray | float]:
    stats = list(stats)
    if not stats:
        raise ValueError("Cannot sum empty stats.")
    out = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else float(value) for key, value in stats[0].items()}
    for item in stats[1:]:
        out = _add_stats(out, item)
    return out


def _score_stats(
    stats: dict[str, np.ndarray | float],
    *,
    ridge_lambda: float,
    corr_weight: float,
    r2_weight: float,
    mean_accel_weight: float,
) -> dict[str, float]:
    n = float(stats["n"])
    if n < 4:
        return {
            "score": float("inf"),
            "samples": n,
            "mean_abs_corr": float("inf"),
            "mean_r2": float("inf"),
            "mean_accel_norm": float("inf"),
        }
    sum_x = np.asarray(stats["sum_x"], dtype=np.float64)
    sum_y = np.asarray(stats["sum_y"], dtype=np.float64)
    sum_xx = np.asarray(stats["sum_xx"], dtype=np.float64)
    sum_xy = np.asarray(stats["sum_xy"], dtype=np.float64)
    sum_yy = np.asarray(stats["sum_yy"], dtype=np.float64)
    mean_y = sum_y / n
    centered_xx = sum_xx - np.outer(sum_x, sum_x) / n
    centered_xy = sum_xy - np.outer(sum_x, sum_y) / n
    centered_yy = sum_yy - np.outer(sum_y, sum_y) / n
    std_x = np.sqrt(np.maximum(np.diag(centered_xx) / n, 1.0e-18))
    std_y = np.sqrt(np.maximum(np.diag(centered_yy) / n, 1.0e-18))
    xtx = centered_xx / np.outer(std_x, std_x)
    xty = centered_xy / np.outer(std_x, std_y)
    yty = centered_yy / np.outer(std_y, std_y)
    corr = np.abs(xty / max(1.0, n - 1.0))
    ridge = float(ridge_lambda) * np.eye(xtx.shape[0], dtype=np.float64)
    beta = np.linalg.solve(xtx + ridge, xty)
    sse = np.diag(yty - 2.0 * beta.T @ xty + beta.T @ xtx @ beta)
    sst = np.maximum(np.diag(yty), 1.0e-9)
    r2 = 1.0 - sse / np.maximum(sst, 1.0e-9)
    mean_abs_corr = float(np.mean(corr))
    mean_r2 = float(np.mean(np.maximum(r2, 0.0)))
    mean_accel_norm = float(np.linalg.norm(mean_y))
    return {
        "score": float(
            corr_weight * mean_abs_corr
            + r2_weight * mean_r2
            + mean_accel_weight * mean_accel_norm
        ),
        "samples": n,
        "mean_abs_corr": mean_abs_corr,
        "mean_r2": mean_r2,
        "mean_accel_norm": mean_accel_norm,
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
    mean_accel_weight: float,
    feature_mode: str,
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
    per_track_stats = {
        key: _track_stats(track_samples[key], frame_size=frame_size, feature_mode=feature_mode)
        for key in eligible
    }
    total_stats = _sum_stats(per_track_stats[key] for key in selected)
    current = _score_stats(
        total_stats,
        ridge_lambda=float(ridge_lambda),
        corr_weight=float(corr_weight),
        r2_weight=float(r2_weight),
        mean_accel_weight=float(mean_accel_weight),
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
        best_stats = None
        for key in candidates:
            trial_stats = _add_stats(total_stats, per_track_stats[key], sign=-1.0)
            score = _score_stats(
                trial_stats,
                ridge_lambda=ridge_lambda,
                corr_weight=corr_weight,
                r2_weight=r2_weight,
                mean_accel_weight=mean_accel_weight,
            )
            if best_score is None or score["score"] < best_score["score"]:
                best_remove = key
                best_score = score
                best_stats = trial_stats
        if best_remove is None or best_score is None or best_stats is None:
            break
        selected.remove(best_remove)
        current = best_score
        total_stats = best_stats
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
    parser.add_argument(
        "--mean-accel-weight",
        type=float,
        default=0.0,
        help="Weight for penalizing the raw norm of mean fitted acceleration in px/s^2.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["raw", "centered", "motion_priors"],
        default="motion_priors",
        help=(
            "Features used when scoring acceleration predictability. raw uses normalized image "
            "position and velocity; centered moves the position origin to image center; "
            "motion_priors also adds distance-to-center and speed."
        ),
    )
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
    eligible_keys = [
        key
        for key, values in track_samples.items()
        if len(values) >= int(args.min_track_samples)
    ]
    if not eligible_keys:
        raise RuntimeError("No tracks have enough acceleration samples for selection.")
    all_stats = _sum_stats(
        _track_stats(track_samples[key], frame_size=frame_size, feature_mode=str(args.feature_mode))
        for key in eligible_keys
    )
    all_score = _score_stats(
        all_stats,
        ridge_lambda=float(args.ridge_lambda),
        corr_weight=float(args.corr_weight),
        r2_weight=float(args.r2_weight),
        mean_accel_weight=float(args.mean_accel_weight),
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
        mean_accel_weight=float(args.mean_accel_weight),
        feature_mode=str(args.feature_mode),
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
        "feature_mode": str(args.feature_mode),
        "min_track_samples": int(args.min_track_samples),
        "target_tracks": int(target_tracks),
        "ridge_lambda": float(args.ridge_lambda),
        "corr_weight": float(args.corr_weight),
        "r2_weight": float(args.r2_weight),
        "mean_accel_weight": float(args.mean_accel_weight),
        "all_tracks_score": all_score,
        "selected_tracks_score": selected_score,
        "manifest_csv": str(manifest_csv),
        "track_split_txt": str(track_split_txt),
        "filtered_tracks_root": str(output_dir / "filtered_tracks") if args.write_filtered_tracks else None,
        "method": (
            "Greedy track removal using per-track sufficient statistics. Score is weighted "
            "mean absolute Pearson correlation and ridge linear R2 from the configured "
            "feature mode to [ax, ay], plus an optional raw mean-acceleration norm penalty."
        ),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
