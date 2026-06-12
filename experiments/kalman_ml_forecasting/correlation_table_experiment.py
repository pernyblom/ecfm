from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import TrackKalmanForecastDataset
from experiments.kalman_ml_forecasting.train import (
    _build_dataset,
    _split_file_label,
    _split_train_eval_folders,
)
from experiments.kalman_ml_forecasting.utils.config import load_config


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "decorrelated",
        "stage",
        "samples",
        "mean_abs_corr",
        "mean_r2",
        "mean_accel_norm",
        "feature_mode",
        "method",
        "keep_fraction",
        "target_samples",
        "random_subset_fraction",
        "random_subset_samples",
        "seed",
        "random_subset_seed",
        "random_seed",
        "source",
        "folders",
    ]
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extras)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fieldnames + extras
                }
            )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _cfg_for_correlation(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    data_cfg = cfg.setdefault("data", {})
    data_cfg["representations"] = []
    data_cfg["filter_missing_representations"] = True
    return cfg


def _decorrelation_cfg(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(base_cfg.get("data", {}).get("decorrelation") or {})


def _cfg_before_decorrelation(base_cfg: Dict[str, Any], split: str) -> Dict[str, Any]:
    cfg = _cfg_for_correlation(base_cfg)
    decorr = _decorrelation_cfg(base_cfg)
    decorr["enabled"] = True
    decorr["splits"] = [split]
    decorr["method"] = "greedy"
    decorr["keep_fraction"] = 1.0
    decorr["target_samples"] = None
    cfg["data"]["decorrelation"] = decorr
    return cfg


def _cfg_after_decorrelation(base_cfg: Dict[str, Any], split: str) -> Dict[str, Any]:
    cfg = _cfg_for_correlation(base_cfg)
    decorr = _decorrelation_cfg(base_cfg)
    decorr["enabled"] = True
    decorr["splits"] = [split]
    cfg["data"]["decorrelation"] = decorr
    return cfg


def _split_folders(
    cfg: Dict[str, Any],
    split: str,
    train_folders: list[str] | None,
    train_eval_folders: list[str] | None,
) -> tuple[list[str] | None, str | None]:
    if split == "train":
        return train_folders, None
    if split == "train_eval" and train_eval_folders is not None:
        return train_eval_folders, None
    if split == "train_eval":
        return None, str(cfg["data"].get("train_eval_source_split", "train"))
    return None, split


def _build_split_dataset(
    cfg: Dict[str, Any],
    split: str,
    train_folders: list[str] | None,
    train_eval_folders: list[str] | None,
) -> TrackKalmanForecastDataset:
    folders_override, split_file_key = _split_folders(cfg, split, train_folders, train_eval_folders)
    return _build_dataset(
        cfg,
        split,
        split_file_key=split_file_key,
        folders_override=folders_override,
    )


def _source_label(
    cfg: Dict[str, Any],
    split: str,
    folders_override: list[str] | None,
    split_file_key: str | None,
) -> str:
    if split == "train_eval" and folders_override is not None:
        return "held-out folders from data.split_files.train"
    key = split_file_key or split
    return f"data.split_files.{key} ({_split_file_label(cfg, key)})"


def _motion_summary(accel: np.ndarray, velocity: np.ndarray) -> dict[str, float]:
    if accel.size == 0:
        return {}
    abs_accel = np.linalg.norm(accel, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    cross = velocity[:, 0] * accel[:, 1] - velocity[:, 1] * accel[:, 0]
    turning_accel = np.divide(
        np.abs(cross),
        speed,
        out=np.zeros_like(abs_accel),
        where=speed > 1.0e-9,
    )
    return {
        "mean_abs_accel": float(np.mean(abs_accel)),
        "median_abs_accel": float(np.median(abs_accel)),
        "p90_abs_accel": float(np.percentile(abs_accel, 90.0)),
        "mean_turning_accel": float(np.mean(turning_accel)),
        "median_turning_accel": float(np.median(turning_accel)),
        "p90_turning_accel": float(np.percentile(turning_accel, 90.0)),
    }


def _score_dataset(dataset: TrackKalmanForecastDataset, cfg: Dict[str, Any]) -> dict[str, Any]:
    if len(dataset) == 0:
        raise RuntimeError("Cannot score motion-prior correlation for an empty dataset.")
    decorr = _decorrelation_cfg(cfg)
    x, y, velocity = dataset._sample_decorrelation_arrays()
    n = float(x.shape[0])
    score = TrackKalmanForecastDataset._score_decorrelation_stats(
        n,
        x.sum(axis=0),
        y.sum(axis=0),
        x.T @ x,
        x.T @ y,
        y.T @ y,
        ridge_lambda=float(decorr.get("ridge_lambda", 1.0e-3)),
        corr_weight=float(decorr.get("corr_weight", 1.0)),
        r2_weight=float(decorr.get("r2_weight", 1.0)),
        mean_accel_weight=float(decorr.get("mean_accel_weight", 0.0)),
    )
    score.update(_motion_summary(y, velocity))
    score["samples"] = int(len(dataset))
    score["feature_dim"] = int(x.shape[1]) if x.ndim == 2 else 0
    score["target_dim"] = int(y.shape[1]) if y.ndim == 2 else 0
    return score


def _row_metadata(
    cfg: Dict[str, Any],
    *,
    split: str,
    decorrelated: bool,
    stage: str,
    folders_override: list[str] | None,
    split_file_key: str | None,
) -> dict[str, Any]:
    decorr = _decorrelation_cfg(cfg)
    return {
        "split": split,
        "decorrelated": "yes" if decorrelated else "no",
        "stage": stage,
        "feature_mode": str(decorr.get("feature_mode", "motion_priors")),
        "method": str(decorr.get("method", "greedy")),
        "keep_fraction": decorr.get("keep_fraction"),
        "target_samples": decorr.get("target_samples"),
        "random_subset_fraction": decorr.get("random_subset_fraction", 1.0),
        "random_subset_samples": decorr.get("random_subset_samples"),
        "seed": decorr.get("seed"),
        "random_subset_seed": decorr.get("random_subset_seed"),
        "random_seed": decorr.get("random_seed"),
        "source": _source_label(cfg, split, folders_override, split_file_key),
        "folders": None if folders_override is None else len(folders_override),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    base_cfg = load_config(args.config)
    train_cfg_for_split = _cfg_for_correlation(base_cfg)
    train_folders, train_eval_folders = _split_train_eval_folders(train_cfg_for_split)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for split in args.split:
        split_cfg = _cfg_for_correlation(base_cfg)
        folders_override, split_file_key = _split_folders(
            split_cfg,
            split,
            train_folders,
            train_eval_folders,
        )
        before_cfg = _cfg_before_decorrelation(base_cfg, split)
        after_cfg = _cfg_after_decorrelation(base_cfg, split)
        print(f"Building {split} candidate-pool dataset...")
        before_dataset = _build_split_dataset(before_cfg, split, train_folders, train_eval_folders)
        before_row = {
            **_row_metadata(
                before_cfg,
                split=split,
                decorrelated=False,
                stage="candidate_pool",
                folders_override=folders_override,
                split_file_key=split_file_key,
            ),
            **_score_dataset(before_dataset, before_cfg),
        }
        rows.append(before_row)
        print(
            f"{split} before decorrelation: samples={before_row['samples']} "
            f"mean_abs_corr={before_row['mean_abs_corr']:.6g} mean_r2={before_row['mean_r2']:.6g} "
            f"mean_accel_norm={before_row['mean_accel_norm']:.6g}"
        )

        print(f"Building {split} decorrelated dataset...")
        after_dataset = _build_split_dataset(after_cfg, split, train_folders, train_eval_folders)
        after_row = {
            **_row_metadata(
                after_cfg,
                split=split,
                decorrelated=True,
                stage="decorrelated",
                folders_override=folders_override,
                split_file_key=split_file_key,
            ),
            **_score_dataset(after_dataset, after_cfg),
        }
        rows.append(after_row)
        print(
            f"{split} after decorrelation: samples={after_row['samples']} "
            f"mean_abs_corr={after_row['mean_abs_corr']:.6g} mean_r2={after_row['mean_r2']:.6g} "
            f"mean_accel_norm={after_row['mean_accel_norm']:.6g}"
        )

    result = {
        "config": str(args.config),
        "splits": list(args.split),
        "rows": rows,
        "outputs": {
            "csv": str(output_dir / "correlation_table_rows.csv"),
            "json": str(output_dir / "correlation_table_summary.json"),
            "jsonl": str(output_dir / "correlation_table_rows.jsonl"),
        },
    }
    _write_csv(output_dir / "correlation_table_rows.csv", rows)
    _write_jsonl(output_dir / "correlation_table_rows.jsonl", rows)
    (output_dir / "correlation_table_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the motion-prior correlation experiment for paper table tab:correlation. "
            "Outputs table-ready CSV, JSON, and JSONL rows before and after sample decorrelation."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Split to evaluate. Repeatable. Defaults to train, train_eval, and test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kalman_ml_correlation_table"),
    )
    args = parser.parse_args()
    if args.split is None:
        args.split = ["train", "train_eval", "test"]
    result = run_experiment(args)
    print(json.dumps(_jsonable(result["outputs"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
