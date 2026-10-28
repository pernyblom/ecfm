"""Export the configured split's original-track forecast windows as JSON Lines."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.kalman_ml_forecasting.data.track_dataset import (
    TrackKalmanForecastDataset,
    _parse_frame_time_raw,
    _read_tracks,
)
from experiments.kalman_ml_forecasting.train import _split_train_eval_folders
from experiments.kalman_ml_forecasting.utils.config import load_config, read_split_file


class ExportTrackletDataset(TrackKalmanForecastDataset):
    """Use the standard sampler/selection, with full precision and full windows."""

    def __init__(self, *, minimum_duration_s: float = 1.2, **kwargs):
        if not math.isfinite(minimum_duration_s) or minimum_duration_s < 1.2:
            raise ValueError("minimum_duration_s must be finite and at least 1.2.")
        self.minimum_duration_s = minimum_duration_s
        self.skipped_short_windows = 0
        super().__init__(**kwargs)

    def _resolve_label_period_s(self, frames):
        if self.label_period_s is not None:
            period = self.label_period_s
        else:
            raw_times = np.asarray([_parse_frame_time_raw(frame.stem) for frame in frames], dtype=np.int64)
            differences = np.diff(raw_times)
            positive = differences[differences > 0]
            if positive.size == 0:
                raise ValueError("Need at least two distinct label timestamps to infer the period.")
            period = float(np.median(positive)) * self.label_time_unit
        if not math.isfinite(period) or period <= 0:
            raise ValueError("label_period_s must be finite and positive.")
        return period

    def _window_spec(self, label_period_s):
        history_steps, future_steps = super()._window_spec(label_period_s)
        # Match the training sampler's nearest-interval convention: FRED's
        # 36 intervals span 1.199988 s and count as a nominal 1.2 s window.
        total_steps = max(1, int(round(self.minimum_duration_s / label_period_s)))
        return history_steps, max(future_steps, total_steps - history_steps)

    def _build_samples(self):
        samples = super()._build_samples()
        frame_indices = {
            folder: {frame.stem: i for i, frame in enumerate(frames)}
            for folder, frames in self.frames_by_folder.items()
        }
        periods = {
            folder: self._resolve_label_period_s(self.frames_by_folder[folder])
            for folder in {sample["folder"] for sample in samples}
        }
        kept = []
        for sample in samples:
            folder = sample["folder"]
            anchor = frame_indices[folder][sample["anchor_stem"]]
            num_past = len(sample["past_times_s"])
            num_future = len(sample["future_times_s"])
            frames = self.frames_by_folder[folder][anchor - num_past + 1:anchor + num_future + 1]
            # Recover original float64 label times: cached training metadata uses
            # float32, which loses sub-frame precision in long recordings.
            times = np.asarray([frame.time_s for frame in frames], dtype=np.float64)
            # Subtract integer timestamps first to avoid cancellation when a
            # recording starts at a large absolute timestamp.
            duration = round((_parse_frame_time_raw(frames[-1].stem)
                              - _parse_frame_time_raw(frames[0].stem)) * self.label_time_unit, 12)
            period = periods[folder]
            minimum_steps = max(1, int(round(self.minimum_duration_s / period)))
            # Judge duration against the rounded interval count, allowing one
            # timestamp tick of quantization per interval. Genuinely truncated
            # windows are still removed before sample selection.
            minimum_span = minimum_steps * period
            tolerance = max(minimum_steps * self.label_time_unit, 1e-12)
            if duration + tolerance < minimum_span:
                self.skipped_short_windows += 1
                continue
            sample["past_times_s"] = times[:num_past]
            sample["future_times_s"] = times[num_past:]
            sample["label_stems"] = [frame.stem for frame in frames]
            sample["duration_s"] = duration
            kept.append(sample)
        # This runs before the base class's sample caps and decorrelation.
        return kept


def _split_folders(cfg, split):
    data = cfg["data"]
    holdout = bool((data.get("train_eval_split") or {}).get("enabled", False))
    if split in {"train", "train_eval"} and holdout:
        train, train_eval = _split_train_eval_folders(cfg)
        return (train if split == "train" else train_eval), "train"
    key = str(data.get("train_eval_source_split", "train")) if split == "train_eval" else split
    path = (data.get("split_files") or {}).get(key)
    if not path:
        raise ValueError(f"Export requires an explicit data.split_files.{key}.")
    return read_split_file(Path(path)), key


def _track_id_map(labels_root, folders):
    """Reserve every original ID before assigning IDs to later collisions."""
    keys = sorted(
        (folder, track_id)
        for folder in folders
        for track_id in _read_tracks(labels_root / folder / "tracks.txt")
    )
    next_id = max((track_id for _, track_id in keys), default=-1) + 1
    used = set()
    result = {}
    for key in keys:
        track_id = key[1]
        if track_id in used:
            result[key] = next_id
            next_id += 1
        else:
            result[key] = track_id
            used.add(track_id)
    return result


def export_tracklets(cfg, *, split, output_dir, seed=None, config_path=None):
    """Write tracklets.jsonl, a CSV index and manifest.json; return the manifest.

    Paths follow the training CLI convention (relative to the working directory).
    The config is not mutated. No images, RAW events or dataset caches are opened.
    """
    cfg = deepcopy(cfg)
    data = cfg["data"]
    if seed is not None:
        data["seed"] = int(seed)
    base_seed = int(data.get("seed", 123))
    dataset_seed = base_seed + {"train": 0, "train_eval": 1, "val": 2, "test": 3}.get(split, 4)
    folders, split_file_key = _split_folders(cfg, split)
    folders = list(dict.fromkeys(folders))
    labels_root = Path(data["labels_root"])
    skipped = [folder for folder in folders if not (labels_root / folder / "tracks.txt").is_file()]
    skipped_set = set(skipped)
    folders = [folder for folder in folders if folder not in skipped_set]
    ids = _track_id_map(labels_root, folders)
    decorrelation = dict(data.get("decorrelation") or {})
    selected_splits = decorrelation.get("splits", ["train"])
    if isinstance(selected_splits, str):
        selected_splits = [selected_splits]
    if not decorrelation.get("enabled", False) or not ({split, "all"} & set(selected_splits)):
        decorrelation = {}
    else:
        decorrelation["_split_name"] = split
    minimum_duration_s = max(1.2, float(data.get("min_track_duration_ms") or 1200.0) / 1000.0)
    dataset = ExportTrackletDataset(
        minimum_duration_s=minimum_duration_s,
        images_root=labels_root,
        labels_root=labels_root,
        frame_size=tuple(data["frame_size"]),
        representations=[],
        image_sizes={},
        folders=folders,
        tracks_file="tracks.txt",
        labels_subdir=data.get("labels_subdir", "Event_YOLO"),
        label_time_unit=float(data.get("label_time_unit", 1e-6)),
        track_time_unit=float(data.get("track_time_unit", 1.0)),
        time_align=data.get("time_align", "auto"),
        label_period_s=data.get("label_period_s"),
        history_ms=float(data.get("history_ms", 400.0)),
        forecast_ms=float(data.get("forecast_ms", 800.0)),
        max_tracks=data.get(f"max_tracks_{split}", data.get("max_tracks")),
        max_samples=data.get(f"max_samples_{split}", data.get("max_samples")),
        sample_decorrelation=decorrelation,
        seed=dataset_seed,
        require_representations=False,
        verify_render_manifest=False,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_tracks = set()
    index_fields = ["tracklet_id", "folder", "track_id", "original_track_id", "start_time_s",
                    "anchor_time_s", "end_time_s", "duration_s", "num_intervals", "past_points", "future_points"]
    with (output_dir / "tracklets.jsonl").open("w", encoding="utf-8", newline="\n") as stream, \
            (output_dir / "tracklets.csv").open("w", encoding="utf-8", newline="") as index:
        writer = csv.DictWriter(index, fieldnames=index_fields)
        writer.writeheader()
        for sample in dataset.samples:
            folder, original_id = sample["folder"], int(sample["track_id"])
            key = (folder, original_id)
            exported_tracks.add(key)
            # Content identity survives selection order, seed changes and ID remapping.
            identity = [folder, original_id, sample["anchor_stem"], sample["label_stems"]]
            tracklet_id = "tl_" + hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode("utf-8")).hexdigest()
            row = {
                "tracklet_id": tracklet_id,
                "folder": folder,
                "source_tracks_file": (Path(folder) / "tracks.txt").as_posix(),
                "track_id": ids[key],
                "original_track_id": original_id,
                "anchor_stem": sample["anchor_stem"],
                "anchor_time_s": sample["anchor_time_s"],
                "duration_s": sample["duration_s"],
                "num_intervals": len(sample["label_stems"]) - 1,
                "label_stems": sample["label_stems"],
                "past_times_s": sample["past_times_s"].tolist(),
                "future_times_s": sample["future_times_s"].tolist(),
                "past_boxes": sample["past_boxes"].tolist(),
                "future_boxes": sample["future_boxes"].tolist(),
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            writer.writerow({
                **{name: row[name] for name in index_fields if name in row},
                "start_time_s": row["past_times_s"][0],
                "end_time_s": row["future_times_s"][-1],
                "past_points": len(row["past_times_s"]),
                "future_points": len(row["future_times_s"]),
            })
    manifest = {
        "format": "kalman_ml_tracklets_v1",
        "config": str(config_path) if config_path is not None else None,
        "split": split,
        "split_file": str(data["split_files"][split_file_key]),
        "seed": base_seed,
        "dataset_seed": dataset_seed,
        "labels_root": str(labels_root),
        "tracks_file": "tracks.txt",
        "minimum_duration_s": minimum_duration_s,
        "configured_history_ms": float(data.get("history_ms", 400.0)),
        "configured_forecast_ms": float(data.get("forecast_ms", 800.0)),
        "window_policy": "Configured history/forecast steps; extend future to the nominal minimum rounded to the nearest label interval. Allow one timestamp tick per interval of duration quantization.",
        "box_format": "normalized [cx, cy, w, h], interpolated from original tracks.txt; no augmentation",
        "frame_size": list(data["frame_size"]),
        "label_time_unit": dataset.label_time_unit,
        "track_time_unit": dataset.track_time_unit,
        "time_align": dataset.time_align,
        "label_period_s": dataset.label_period_s,
        "selection": {
            "train_eval_split": data.get("train_eval_split") or {},
            "max_tracks": dataset.max_tracks,
            "max_samples": dataset.max_samples,
            "decorrelation": decorrelation,
            "representation_filtering": False,
        },
        "eligible_folders": folders,
        "exported_folders": sorted({folder for folder, _ in exported_tracks}),
        "skipped_folders_without_tracks_txt": skipped,
        "skipped_short_windows": dataset.skipped_short_windows,
        "tracklets": len(dataset),
        "tracks": len(exported_tracks),
        "track_id_mapping": [
            {"folder": folder, "original_track_id": original_id, "track_id": ids[(folder, original_id)]}
            for folder, original_id in sorted(exported_tracks)
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, help="Override data.seed (the usual split offset is still applied).")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_tracklets(load_config(args.config), split=args.split, seed=args.seed,
                               output_dir=args.output_dir, config_path=args.config)
    print(f"Exported {manifest['tracklets']} tracklets from {manifest['tracks']} tracks to {args.output_dir}")
    print(f"Skipped {len(manifest['skipped_folders_without_tracks_txt'])} folders without tracks.txt")


if __name__ == "__main__":
    main()
