"""FRED forecast tracklets and scene-stratified sequence splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.kalman_ml_forecasting.data.track_dataset import FrameItem, TrackKalmanForecastDataset, _read_tracks
from experiments.kalman_ml_forecasting.utils.config import (
    load_config,
    read_split_file,
    resolve_event_image_options,
    resolve_representation_image_sizes,
    resolve_representation_source_image_sizes,
)

from .coordinate_tracks import coordinate_tracks


def load_scenes(path: Path) -> dict:
    manifest = load_config(path)
    scene_by_folder = {}
    names = list(manifest["scenes"])
    for scene, folders in manifest["scenes"].items():
        for sequence in folders:
            folder = str(sequence)
            if folder in scene_by_folder:
                raise ValueError(f"Sequence {folder} appears in multiple scenes")
            scene_by_folder[folder] = scene
    priors = manifest.get("camera_up", {})
    if set(priors) - set(names):
        raise ValueError("camera_up contains an unknown scene")
    return {"by_folder": scene_by_folder, "names": names,
            "up_priors": [priors.get(scene, [0, -1, 0]) for scene in names]}


def make_splits(data: dict, scenes: dict, seed: int) -> dict:
    root = Path(data["labels_root"])
    source = (sorted(scenes["by_folder"], key=int) if data.get("all_sequences")
              else read_split_file(Path(data["split_files"]["train"])))
    if len(source) != len(set(source)):
        raise ValueError("Training sequence list contains duplicates")
    unknown = set(source) - scenes["by_folder"].keys()
    if unknown:
        raise ValueError(f"Missing scene assignments for {sorted(unknown)}")
    excluded = {}
    by_scene = defaultdict(list)
    annotation_sources = {}
    events_root = Path(data.get("events_root", root))
    for folder in source:
        tracks = root / folder / data.get("tracks_file", "cleaned_tracks.txt")
        raw = events_root / folder / "Event" / "events.raw"
        if data["event_image_source"] == "indexed" and not (raw.is_file() and raw.with_suffix(".raw.tmp_index").is_file()):
            excluded[folder] = "missing indexed events"
        elif tracks.is_file() and tracks.stat().st_size:
            annotation_sources[folder] = "cleaned tracks"
            by_scene[scenes["by_folder"][folder]].append(folder)
        elif coordinate_tracks(data, folder) is not None:
            annotation_sources[folder] = "coordinate-derived tracks"
            by_scene[scenes["by_folder"][folder]].append(folder)
        else:
            excluded[folder] = "missing or empty eligible tracks"
    reference_file = data.get("split_reference")
    if reference_file:
        reference = json.loads(Path(reference_file).read_text(encoding="utf-8"))
        train = set(reference["train"])
        val = set(reference["val"])
        eligible = annotation_sources.keys()
        if train & val or (train | val) - set(source) or eligible - (train | val):
            raise ValueError("Split reference must assign all eligible sequences to disjoint canonical splits")
        return {"train": sorted(train & eligible, key=int), "val": sorted(val & eligible, key=int),
                "excluded": excluded, "annotation_sources": annotation_sources}
    rng = np.random.default_rng(seed)
    fraction = float(data["val_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("data.val_fraction must be between 0 and 1")
    train, val = [], []
    for folders in by_scene.values():
        order = rng.permutation(folders).tolist()
        count = min(len(order) - 1, max(1, round(len(order) * fraction)))
        val.extend(order[:count])
        train.extend(order[count:])
    return {"train": sorted(train, key=int), "val": sorted(val, key=int), "excluded": excluded,
            "annotation_sources": annotation_sources}


class _ForecastTracklets(TrackKalmanForecastDataset):
    def __init__(self, *, fallback: dict[str, Path], fallback_frame_timestamps: bool, **kwargs):
        self.fallback = fallback
        self.fallback_frame_timestamps = fallback_frame_timestamps
        super().__init__(**kwargs)

    def _tracks_path(self, folder: str) -> Path:
        return self.fallback.get(folder) or super()._tracks_path(folder)

    def _discover_frames(self) -> dict[str, list[FrameItem]]:
        frames = super()._discover_frames()
        if self.fallback_frame_timestamps:
            # A few coordinate-labeled sequences have native event frames without Event_YOLO.
            for folder, track_path in self.fallback.items():
                if frames.get(folder):
                    continue
                native = self._build_event_frame_index(folder)
                if native:
                    frames[folder] = [FrameItem(path.stem, time) for time, path in native]
                elif self.event_image_source == "indexed":
                    times = sorted({row[0] for track in _read_tracks(track_path).values() for row in track})
                    frames[folder] = [FrameItem(f"Video_{folder}_{round(time / self.label_time_unit)}", time) for time in times]
            # Keep selection order aligned with the requested split list.
            frames = {folder: frames[folder] for folder in self.folders if folder in frames}
        return frames

    def _cache_key(self) -> str:
        base = super()._cache_key()
        if not self.fallback:
            return base
        paths = "\n".join(str(path.resolve()) for path in sorted(self.fallback.values()))
        if self.fallback_frame_timestamps:
            paths += "\nfallback_frame_timestamps"
        return hashlib.sha256(f"{base}\n{paths}".encode()).hexdigest()


class PoseTracklets(Dataset):
    def __init__(self, data: dict, folders: list[str], scenes: dict, *, split: str, seed: int, use_events: bool = True):
        self.scene_ids = {name: index for index, name in enumerate(scenes["names"])}
        self.scene_by_folder = scenes["by_folder"]
        self.use_events = use_events
        self.history_event_cutouts = bool(data.get("history_event_cutouts", False))
        self.reps = list(data["representations"])
        if not self.reps or any(rep in {"rgb", "padded_rgb", "event_frames", "event_frame"} for rep in self.reps):
            raise ValueError("Use rendered event representations for the pose model inputs")
        event_options = resolve_event_image_options(data)
        if self.history_event_cutouts and (not use_events or event_options["event_image_source"] != "indexed"):
            raise ValueError("History event cutouts require indexed event inputs")
        if event_options["instance_cache"].get("mode", "lazy") != "lazy":
            raise ValueError("Pose tracklets support the forecasting cutout cache in lazy mode")
        tracks_file = data.get("tracks_file", "cleaned_tracks.txt")
        root = Path(data["labels_root"])
        fallback = {}
        for folder in folders:
            tracks = root / folder / tracks_file
            if not tracks.is_file() or not tracks.stat().st_size:
                path = coordinate_tracks(data, folder)
                if path is not None:
                    fallback[folder] = path
        self.forecast = _ForecastTracklets(
            fallback=fallback,
            fallback_frame_timestamps=bool(data.get("fallback_frame_timestamps", False)),
            images_root=Path(data["images_root"]), labels_root=Path(data["labels_root"]),
            frame_size=tuple(data["frame_size"]), folders=folders,
            tracks_file=tracks_file,
            image_sizes=resolve_representation_image_sizes(data),
            source_image_sizes=resolve_representation_source_image_sizes(data),
            representations=self.reps, spatial_cutout=data.get("spatial_cutout"),
            label_period_s=data["label_period_s"], label_time_unit=data.get("label_time_unit", 1e-6),
            time_align=data.get("time_align", "none"), history_ms=data["history_ms"],
            image_window_ms=data.get("event_window_ms", data["history_ms"]), forecast_ms=data["forecast_ms"],
            require_representations=False, verify_render_manifest=event_options["event_image_source"] == "rendered",
            cache_dir=Path(data["cache_dir"]), seed=seed, **event_options,
        )
        self._select(data, split, seed)
        self.forecast.frames_by_folder.clear()
        self.forecast._folder_event_frame_indices.clear()
        print(f"{split}: {len(self)} tracklets across {len(self.count_by_folder)} sequences and "
              f"{len(self.count_by_scene)} scenes", flush=True)

    def _select(self, data: dict, split: str, seed: int) -> None:
        rng = np.random.default_rng(seed)
        period = float(data["label_period_s"])
        min_px = float(data.get("min_anchor_size_px", 6))
        frame = np.asarray(data["frame_size"], dtype=np.float32)
        by_folder = defaultdict(list)
        for sample in self.forecast.samples:
            by_folder[sample["folder"]].append(sample)
        selected = []
        limit = data.get("max_samples_per_sequence", {}).get(split)
        for folder, rows in by_folder.items():
            if self.history_event_cutouts:
                frames = self.forecast.frames_by_folder[folder]
                frame_indices = {item.stem: index for index, item in enumerate(frames)}
            folder_rng = np.random.default_rng([seed, int(folder)]) if data.get("sample_seed_by_sequence") else rng
            anchor = np.asarray([sample["past_boxes"][-1] for sample in rows])
            center, extent = np.split(anchor, 2, axis=1)
            interior = (center - extent / 2 > 1 / frame) & (center + extent / 2 < 1 - 1 / frame)
            eligible = np.flatnonzero(interior.all(1) & (extent * frame >= min_px).all(1))
            count = 0
            for index in folder_rng.permutation(eligible):
                sample = rows[index]
                times = np.concatenate((sample["past_times_s"], sample["future_times_s"]))
                gaps = np.diff(times)
                if (gaps > 2.1 * period).any() or (gaps <= 0).any():
                    continue
                if times[-1] - times[0] > (len(times) - 1) * period * 1.12:
                    continue
                if any(not np.isfinite(sample[key]).all() or (sample[key][:, 2:] <= 0).any()
                       for key in ("past_boxes", "future_boxes")):
                    continue
                if self.use_events:
                    folder, stem, time = sample["folder"], sample["anchor_stem"], sample["anchor_time_s"]
                    refs = {rep: self.forecast._resolve_input_path(folder, stem, rep, time) for rep in self.reps}
                    if any(ref is None for ref in refs.values()):
                        continue
                    self.forecast._validate_manifest_entry(folder, stem)
                    sample = dict(sample, input_paths=refs)
                    if self.history_event_cutouts:
                        anchor_index = frame_indices[stem]
                        first = anchor_index + 1 - len(sample["past_boxes"])
                        # Use the original timestamp precision, as for the anchor cache key.
                        sample["earlier_event_times_us"] = [
                            round(item.time_s * 1e6) for item in frames[first:anchor_index]
                        ]
                selected.append(sample)
                count += 1
                if limit and count >= limit:
                    break
        self.forecast.samples = selected
        keys = (f"{s['folder']}/{s['track_id']}/{s['anchor_stem']}" for s in selected)
        self.sample_digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
        self.count_by_folder = dict(Counter(sample["folder"] for sample in selected))
        self.count_by_scene = dict(Counter(self.scene_by_folder[sample["folder"]] for sample in selected))
        if not selected:
            raise ValueError(f"No {split} tracklets satisfy the data configuration")

    def __len__(self) -> int:
        return len(self.forecast)

    def __getitem__(self, index: int) -> dict:
        sample = self.forecast[index]
        metadata = self.forecast.samples[index]
        folder = metadata["folder"]
        scene = self.scene_by_folder[folder]
        events = sample.inputs
        if self.history_event_cutouts:
            boxes = metadata["past_boxes"]
            times = metadata["earlier_event_times_us"]
            events = {}
            for rep, anchor in sample.inputs.items():
                reference = metadata["input_paths"][rep]
                crops = [self.forecast._prepared_input(rep, {**reference, "time_us": time}, box)
                         for time, box in zip(times, boxes[:-1])]
                events[rep] = torch.stack([*crops, anchor])
        return {"events": events, "past": sample.past_boxes, "future": sample.future_boxes,
                "past_time": sample.past_times_s - sample.frame_time_s,
                "future_time": sample.future_times_s - sample.frame_time_s,
                "scene": self.scene_ids[scene], "folder": folder, "track_id": sample.track_id,
                "anchor_time": sample.frame_time_s, "frame_key": sample.frame_key}


def to_device(batch: dict, device: torch.device) -> dict:
    return {"events": {rep: x.to(device, non_blocking=True) for rep, x in batch["events"].items()},
            **{key: batch[key].to(device, non_blocking=True)
               for key in ("past", "future", "past_time", "future_time", "scene")}}
