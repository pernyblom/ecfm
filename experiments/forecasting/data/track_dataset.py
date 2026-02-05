from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class TrackSample:
    inputs: Dict[str, torch.Tensor]
    past_boxes: torch.Tensor
    future_boxes: torch.Tensor
    frame_keys: List[str]
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
    num = ""
    for ch in tail:
        if ch.isdigit():
            num += ch
        else:
            break
    return int(num) if num else None


def _load_image(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size[0], size[1]):
        img = img.resize(size, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


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


def _infer_frame_size(tracks: Dict[int, List[Tuple[float, float, float, float, float]]]) -> Tuple[int, int]:
    max_x = 0.0
    max_y = 0.0
    for items in tracks.values():
        for _, x, y, w, h in items:
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
    width = int(np.ceil(max_x)) if max_x > 0 else 0
    height = int(np.ceil(max_y)) if max_y > 0 else 0
    return width, height


class TrackForecastDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_root: Path,
        labels_root: Path,
        representations: List[str],
        past_steps: int,
        future_steps: int,
        stride: int,
        image_size: Tuple[int, int],
        folders: Optional[List[str]] = None,
        labels_subdir: str = "Event_YOLO",
        tracks_file: str = "tracks.txt",
        label_time_unit: float = 1e-6,
        track_time_unit: float = 1.0,
        time_align: str = "start",
        frame_size: Optional[Tuple[int, int]] = None,
        max_tracks: Optional[int] = None,
        max_samples: Optional[int] = None,
        seed: int = 123,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.representations = representations
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.stride = stride
        self.image_size = image_size
        self.folders = folders
        self.labels_subdir = labels_subdir
        self.tracks_file = tracks_file
        self.label_time_unit = label_time_unit
        self.track_time_unit = track_time_unit
        self.time_align = time_align
        self.frame_size_override = frame_size
        self.max_tracks = max_tracks
        self.max_samples = max_samples
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None

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
        frames: Dict[str, List[FrameItem]] = {}
        if self.folders is None:
            keys = []
            for txt in self.labels_root.glob("*.txt"):
                time_raw = _parse_frame_time(txt.stem)
                if time_raw is None:
                    continue
                keys.append(FrameItem(txt.stem, float(time_raw) * self.label_time_unit))
            keys.sort(key=lambda item: item.time_s)
            frames[""] = keys
            return frames
        for folder in self.folders:
            labels_dir = self._labels_dir(folder)
            if not labels_dir.exists():
                continue
            items: List[FrameItem] = []
            for txt in labels_dir.glob("*.txt"):
                time_raw = _parse_frame_time(txt.stem)
                if time_raw is None:
                    continue
                items.append(FrameItem(txt.stem, float(time_raw) * self.label_time_unit))
            items.sort(key=lambda item: item.time_s)
            frames[folder] = items
        return frames

    def _select_track_subset(self) -> Optional[set[tuple[str, int]]]:
        if self.max_tracks is None:
            return None
        keys: List[tuple[str, int]] = []
        for folder in self.frames_by_folder.keys():
            tracks = _read_tracks(self._tracks_path(folder))
            for track_id in tracks.keys():
                keys.append((folder, track_id))
        if not keys:
            return set()
        rng = np.random.default_rng(self.seed)
        rng.shuffle(keys)
        subset = keys[: self.max_tracks]
        return set(subset)

    def _has_all_reps(self, folder: str, stem: str) -> bool:
        img_dir = self._images_dir(folder)
        for rep in self.representations:
            img = img_dir / f"{stem}_{rep}.png"
            if not img.exists():
                return False
        return True

    def _build_samples(
        self,
    ) -> List[Tuple[str, int, List[str], List[Tuple[float, float, float, float]]]]:
        samples: List[
            Tuple[str, int, List[str], List[Tuple[float, float, float, float]]]
        ] = []
        total = self.past_steps + self.future_steps

        print("Building samples...")
        for folder, frames in self.frames_by_folder.items():
            if not frames:
                continue
            labels_dir = self._labels_dir(folder)
            tracks = _read_tracks(self._tracks_path(folder))
            if not tracks:
                continue
            frame_w, frame_h = self.frame_size_override or _infer_frame_size(tracks)
            if frame_w <= 0 or frame_h <= 0:
                continue
            label_times = np.array([item.time_s for item in frames], dtype=np.float64)
            label_stems = [item.stem for item in frames]

            print(folder)
            for track_id, items in tracks.items():
                if self.allowed_tracks is not None and (folder, track_id) not in self.allowed_tracks:
                    continue
                times = np.array([t for t, *_ in items], dtype=np.float64) * self.track_time_unit
                xs = np.array([x for _, x, _, _, _ in items], dtype=np.float64)
                ys = np.array([y for _, _, y, _, _ in items], dtype=np.float64)
                ws = np.array([w for _, _, _, w, _ in items], dtype=np.float64)
                hs = np.array([h for _, _, _, _, h in items], dtype=np.float64)

                if times.size == 0:
                    continue
                if self.time_align == "start":
                    shift = label_times[0] - times[0]
                    times = times + shift
                elif self.time_align != "none":
                    raise ValueError(f"Unknown time_align: {self.time_align}")

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
                boxes = list(zip(cx, cy, bw, bh))

                stems = [label_stems[i] for i in idxs]
                for start in range(0, len(stems) - total * self.stride + 1):
                    window_idx = [start + i * self.stride for i in range(total)]
                    window_stems = [stems[i] for i in window_idx]
                    if not all(self._has_all_reps(folder, stem) for stem in window_stems):
                        continue
                    window_boxes = [boxes[i] for i in window_idx]
                    samples.append((folder, track_id, window_stems, window_boxes))

        return samples

    def _apply_sample_limit(self) -> None:
        if self.max_samples is None or self.max_samples <= 0:
            return
        if not self.samples:
            return
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(self.samples))[: self.max_samples]
        self.samples = [self.samples[i] for i in idx]

    def _cache_key(self) -> str:
        payload = {
            "labels_root": str(self.labels_root),
            "images_root": str(self.images_root),
            "folders": self.folders,
            "labels_subdir": self.labels_subdir,
            "tracks_file": self.tracks_file,
            "label_time_unit": self.label_time_unit,
            "track_time_unit": self.track_time_unit,
            "time_align": self.time_align,
            "frame_size": self.frame_size_override,
            "max_tracks": self.max_tracks,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "past_steps": self.past_steps,
            "future_steps": self.future_steps,
            "stride": self.stride,
            "representations": self.representations,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"track_samples_{self._cache_key()}.pkl"

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

    def __getitem__(self, idx: int) -> TrackSample:
        folder, track_id, stems, boxes = self.samples[idx]
        past_stems = stems[: self.past_steps]
        future_stems = stems[self.past_steps :]
        past_boxes = boxes[: self.past_steps]
        future_boxes = boxes[self.past_steps :]

        inputs: Dict[str, List[torch.Tensor]] = {r: [] for r in self.representations}
        img_dir = self._images_dir(folder)

        for stem in past_stems:
            for rep in self.representations:
                img_path = img_dir / f"{stem}_{rep}.png"
                inputs[rep].append(_load_image(img_path, self.image_size))

        inputs_t = {r: torch.stack(seq, dim=0) for r, seq in inputs.items()}
        frame_keys = [f"{folder}/{stem}" if folder else stem for stem in stems]
        return TrackSample(
            inputs=inputs_t,
            past_boxes=torch.tensor(past_boxes, dtype=torch.float32),
            future_boxes=torch.tensor(future_boxes, dtype=torch.float32),
            frame_keys=frame_keys,
            track_id=int(track_id),
        )
