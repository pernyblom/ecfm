from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class CurveForecastSample:
    input_paths: Dict[str, str]
    past_boxes: torch.Tensor
    future_boxes: torch.Tensor
    past_times_s: torch.Tensor
    future_times_s: torch.Tensor
    frame_key: str
    frame_time_s: float
    track_id: int


@dataclass
class FrameItem:
    stem: str
    time_s: float


def _parse_frame_time(name: str) -> Optional[int]:
    parts = name.split("_frame_")
    if len(parts) < 2:
        return None
    tail = parts[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def _read_tracks(path: Path) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    tracks: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    if not path.exists():
        return tracks
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            t = float(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except ValueError:
            continue
        tracks.setdefault(track_id, []).append((t, x, y, w, h))
    for track_id, items in tracks.items():
        items.sort(key=lambda item: item[0])
        tracks[track_id] = items
    return tracks


class TrackCurveForecastDataset(torch.utils.data.Dataset):
    CACHE_VERSION = 1

    def __init__(
        self,
        images_root: Path,
        labels_root: Path,
        frame_size: Tuple[int, int],
        image_window_ms: float,
        history_ms: float,
        forecast_ms: float,
        representations: List[str],
        folders: Optional[List[str]] = None,
        labels_subdir: str = "Event_YOLO",
        tracks_file: str = "cleaned_tracks.txt",
        label_time_unit: float = 1e-6,
        track_time_unit: float = 1.0,
        time_align: str = "start",
        image_window_mode: str = "trailing",
        label_period_s: Optional[float] = None,
        max_tracks: Optional[int] = None,
        max_samples: Optional[int] = None,
        seed: int = 123,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.frame_size = (float(frame_size[0]), float(frame_size[1]))
        self.image_window_ms = float(image_window_ms)
        self.history_ms = float(history_ms)
        self.forecast_ms = float(forecast_ms)
        self.representations = list(representations)
        self.folders = folders
        self.labels_subdir = labels_subdir
        self.tracks_file = tracks_file
        self.label_time_unit = float(label_time_unit)
        self.track_time_unit = float(track_time_unit)
        self.time_align = time_align
        self.image_window_mode = image_window_mode
        self.label_period_s = None if label_period_s is None else float(label_period_s)
        self.max_tracks = max_tracks
        self.max_samples = max_samples
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise ValueError("frame_size must contain positive width and height.")
        if self.image_window_ms <= 0 or self.history_ms <= 0 or self.forecast_ms <= 0:
            raise ValueError("image_window_ms, history_ms and forecast_ms must be > 0.")
        if not self.representations:
            raise ValueError("representations must not be empty.")
        if self.image_window_mode not in {"trailing", "center", "leading"}:
            raise ValueError(f"Unknown image_window_mode: {self.image_window_mode}")

        self.frames_by_folder = self._discover_frames()
        self.allowed_tracks = self._select_track_subset()
        cached = self._load_cache()
        if cached is not None:
            self.samples = cached
        else:
            self.samples = self._build_samples()
            self._apply_sample_limit()
            self._save_cache()

    def _labels_dir(self, folder: str) -> Path:
        if self.folders is None:
            return self.labels_root
        return self.labels_root / folder / self.labels_subdir

    def _images_dir(self, folder: str) -> Path:
        if self.folders is None:
            return self.images_root
        return self.images_root / folder

    def _tracks_path(self, folder: str) -> Path:
        if self.folders is None:
            return self.labels_root / self.tracks_file
        return self.labels_root / folder / self.tracks_file

    def _discover_frames(self) -> Dict[str, List[FrameItem]]:
        out: Dict[str, List[FrameItem]] = {}
        if self.folders is None:
            items: List[FrameItem] = []
            for txt in self.labels_root.glob("*.txt"):
                time_raw = _parse_frame_time(txt.stem)
                if time_raw is None:
                    continue
                if not self._has_all_reps("", txt.stem):
                    continue
                items.append(FrameItem(stem=txt.stem, time_s=float(time_raw) * self.label_time_unit))
            items.sort(key=lambda x: x.time_s)
            out[""] = items
            return out

        for folder in self.folders:
            labels_dir = self._labels_dir(folder)
            if not labels_dir.exists():
                continue
            items: List[FrameItem] = []
            for txt in labels_dir.glob("*.txt"):
                time_raw = _parse_frame_time(txt.stem)
                if time_raw is None:
                    continue
                if not self._has_all_reps(folder, txt.stem):
                    continue
                items.append(FrameItem(stem=txt.stem, time_s=float(time_raw) * self.label_time_unit))
            items.sort(key=lambda x: x.time_s)
            out[folder] = items
        return out

    def _select_track_subset(self) -> Optional[set[tuple[str, int]]]:
        if self.max_tracks is None:
            return None
        keys: List[tuple[str, int]] = []
        for folder in self.frames_by_folder.keys():
            tracks = _read_tracks(self._tracks_path(folder))
            for track_id in tracks.keys():
                keys.append((folder, int(track_id)))
        if not keys:
            return set()
        rng = np.random.default_rng(self.seed)
        rng.shuffle(keys)
        return set(keys[: self.max_tracks])

    def _has_all_reps(self, folder: str, stem: str) -> bool:
        base = self._images_dir(folder)
        for rep in self.representations:
            if not (base / f"{stem}_{rep}.png").exists():
                return False
        return True

    def _resolve_label_period_s(self, frames: List[FrameItem]) -> float:
        if self.label_period_s is not None:
            return self.label_period_s
        if len(frames) < 2:
            raise ValueError("Need at least two frames to infer label_period_s.")
        diffs = np.diff(np.asarray([f.time_s for f in frames], dtype=np.float64))
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            raise ValueError("Could not infer positive label period from frame timestamps.")
        return float(np.median(diffs))

    def _window_spec(self, label_period_s: float) -> tuple[int, int]:
        history_steps = max(1, int(round((self.history_ms / 1000.0) / label_period_s)))
        future_steps = max(1, int(round((self.forecast_ms / 1000.0) / label_period_s)))
        return history_steps, future_steps

    def _align_track_times(self, track_times: np.ndarray, label_times: np.ndarray) -> np.ndarray:
        times = track_times.copy()
        if self.time_align == "start":
            shift = label_times[0] - times[0]
            times = times + shift
        elif self.time_align == "auto":
            shift = label_times[0] - times[0]
            no_shift_count = int(np.sum((label_times >= times[0]) & (label_times <= times[-1])))
            shift_count = int(
                np.sum((label_times >= times[0] + shift) & (label_times <= times[-1] + shift))
            )
            if shift_count > no_shift_count:
                times = times + shift
        elif self.time_align != "none":
            raise ValueError(f"Unknown time_align: {self.time_align}")
        return times

    def _build_samples(self):
        samples = []
        frame_w, frame_h = self.frame_size

        print("Building curve-fit forecasting samples...")
        for folder, frames in self.frames_by_folder.items():
            if not frames:
                continue
            tracks = _read_tracks(self._tracks_path(folder))
            if not tracks:
                continue
            label_period_s = self._resolve_label_period_s(frames)
            history_steps, future_steps = self._window_spec(label_period_s)
            window = history_steps + future_steps + 1

            label_times = np.asarray([f.time_s for f in frames], dtype=np.float64)
            label_stems = [f.stem for f in frames]

            for track_id, items in tracks.items():
                if self.allowed_tracks is not None and (folder, int(track_id)) not in self.allowed_tracks:
                    continue
                times = np.asarray([t for t, *_ in items], dtype=np.float64) * self.track_time_unit
                xs = np.asarray([x for _, x, _, _, _ in items], dtype=np.float64)
                ys = np.asarray([y for _, _, y, _, _ in items], dtype=np.float64)
                ws = np.asarray([w for _, _, _, w, _ in items], dtype=np.float64)
                hs = np.asarray([h for _, _, _, _, h in items], dtype=np.float64)
                if times.size == 0:
                    continue

                times = self._align_track_times(times, label_times)
                mask = (label_times >= times[0]) & (label_times <= times[-1])
                if not np.any(mask):
                    continue
                idxs = np.nonzero(mask)[0]
                query_times = label_times[idxs]

                xq = np.interp(query_times, times, xs)
                yq = np.interp(query_times, times, ys)
                wq = np.interp(query_times, times, ws)
                hq = np.interp(query_times, times, hs)

                cx = (xq + wq / 2.0) / frame_w
                cy = (yq + hq / 2.0) / frame_h
                bw = wq / frame_w
                bh = hq / frame_h
                boxes = np.stack([cx, cy, bw, bh], axis=-1)
                stems = [label_stems[i] for i in idxs]

                for start in range(0, len(stems) - window + 1):
                    end = start + window
                    window_stems = stems[start:end]
                    window_times = query_times[start:end]
                    window_boxes = boxes[start:end]
                    anchor_idx = history_steps
                    anchor_stem = window_stems[anchor_idx]
                    if not self._has_all_reps(folder, anchor_stem):
                        continue
                    input_paths = {
                        rep: str(self._images_dir(folder) / f"{anchor_stem}_{rep}.png")
                        for rep in self.representations
                    }
                    samples.append(
                        (
                            folder,
                            int(track_id),
                            input_paths,
                            window_boxes,
                            window_times,
                            anchor_idx,
                            anchor_stem,
                        )
                    )
        return samples

    def _apply_sample_limit(self) -> None:
        if self.max_samples is None or self.max_samples <= 0 or not self.samples:
            return
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(self.samples))[: self.max_samples]
        self.samples = [self.samples[i] for i in perm]

    def _cache_key(self) -> str:
        payload = {
            "images_root": str(self.images_root),
            "labels_root": str(self.labels_root),
            "frame_size": self.frame_size,
            "image_window_ms": self.image_window_ms,
            "history_ms": self.history_ms,
            "forecast_ms": self.forecast_ms,
            "representations": self.representations,
            "folders": self.folders,
            "labels_subdir": self.labels_subdir,
            "tracks_file": self.tracks_file,
            "label_time_unit": self.label_time_unit,
            "track_time_unit": self.track_time_unit,
            "time_align": self.time_align,
            "image_window_mode": self.image_window_mode,
            "label_period_s": self.label_period_s,
            "max_tracks": self.max_tracks,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "cache_version": self.CACHE_VERSION,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"curve_fit_samples_{self._cache_key()}.pkl"

    def _load_cache(self):
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self) -> None:
        path = self._cache_path()
        if path is None:
            return
        try:
            with path.open("wb") as f:
                pickle.dump(self.samples, f)
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CurveForecastSample:
        folder, track_id, input_paths, window_boxes, window_times, anchor_idx, anchor_stem = self.samples[idx]
        past_boxes = window_boxes[: anchor_idx + 1]
        future_boxes = window_boxes[anchor_idx + 1 :]
        past_times_s = window_times[: anchor_idx + 1]
        future_times_s = window_times[anchor_idx + 1 :]
        frame_key = f"{folder}/{anchor_stem}" if folder else anchor_stem
        return CurveForecastSample(
            input_paths=dict(input_paths),
            past_boxes=torch.tensor(past_boxes, dtype=torch.float32),
            future_boxes=torch.tensor(future_boxes, dtype=torch.float32),
            past_times_s=torch.tensor(past_times_s, dtype=torch.float32),
            future_times_s=torch.tensor(future_times_s, dtype=torch.float32),
            frame_key=frame_key,
            frame_time_s=float(window_times[anchor_idx]),
            track_id=int(track_id),
        )
