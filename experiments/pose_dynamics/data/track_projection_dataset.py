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
class ProjectionSample:
    inputs: Dict[str, torch.Tensor]
    past_centers: torch.Tensor
    future_centers: torch.Tensor
    dt: torch.Tensor
    intrinsics: torch.Tensor
    camera_pose: torch.Tensor
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


def _load_image(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size[0], size[1]):
        img = img.resize(size, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class TrackProjectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_root: Path,
        labels_root: Path,
        representations: List[str],
        image_size: Tuple[int, int],
        history_steps: int,
        future_steps: int,
        stride: int,
        frame_size: Tuple[int, int],
        intrinsics: Tuple[float, float, float, float],
        camera_pose: Tuple[float, float, float, float, float, float],
        folders: Optional[List[str]] = None,
        labels_subdir: str = "Event_YOLO",
        tracks_file: str = "cleaned_tracks.txt",
        label_time_unit: float = 1e-6,
        track_time_unit: float = 1.0,
        time_align: str = "start",
        max_tracks: Optional[int] = None,
        max_samples: Optional[int] = None,
        seed: int = 123,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.representations = representations
        self.image_size = image_size
        self.history_steps = int(history_steps)
        self.future_steps = int(future_steps)
        self.stride = int(stride)
        self.frame_size = (float(frame_size[0]), float(frame_size[1]))
        self.intrinsics = tuple(float(v) for v in intrinsics)
        self.camera_pose = tuple(float(v) for v in camera_pose)
        self.folders = folders
        self.labels_subdir = labels_subdir
        self.tracks_file = tracks_file
        self.label_time_unit = float(label_time_unit)
        self.track_time_unit = float(track_time_unit)
        self.time_align = time_align
        self.max_tracks = max_tracks
        self.max_samples = max_samples
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.history_steps < 0:
            raise ValueError("history_steps must be >= 0")
        if self.future_steps <= 0:
            raise ValueError("future_steps must be > 0")

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

    def _build_samples(
        self,
    ) -> List[
        tuple[
            str,
            int,
            str,
            float,
            Dict[str, str],
            List[Tuple[float, float]],
            List[Tuple[float, float]],
            List[float],
        ]
    ]:
        samples: List[
            tuple[
                str,
                int,
                str,
                float,
                Dict[str, str],
                List[Tuple[float, float]],
                List[Tuple[float, float]],
                List[float],
            ]
        ] = []
        window = self.history_steps + self.future_steps + 1
        frame_w, frame_h = self.frame_size
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame_size must contain positive width and height.")

        print("Building pose+dynamics projection samples...")
        for folder, frames in self.frames_by_folder.items():
            if not frames:
                continue
            tracks = _read_tracks(self._tracks_path(folder))
            if not tracks:
                continue

            label_times = np.array([f.time_s for f in frames], dtype=np.float64)
            label_stems = [f.stem for f in frames]

            for track_id, items in tracks.items():
                if self.allowed_tracks is not None and (folder, int(track_id)) not in self.allowed_tracks:
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
                elif self.time_align == "auto":
                    shift = label_times[0] - times[0]
                    no_shift_count = int(
                        np.sum((label_times >= times[0]) & (label_times <= times[-1]))
                    )
                    shift_count = int(
                        np.sum((label_times >= times[0] + shift) & (label_times <= times[-1] + shift))
                    )
                    if shift_count > no_shift_count:
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
                centers = list(zip(cx.tolist(), cy.tolist()))
                stems = [label_stems[i] for i in idxs]
                stem_times = [float(label_times[i]) for i in idxs]

                for start in range(0, len(stems) - window * self.stride + 1):
                    idx_window = [start + i * self.stride for i in range(window)]
                    window_stems = [stems[i] for i in idx_window]
                    window_times = [stem_times[i] for i in idx_window]

                    anchor_idx = self.history_steps
                    anchor_stem = window_stems[anchor_idx]
                    if not self._has_all_reps(folder, anchor_stem):
                        continue

                    rep_paths = {
                        rep: str(self._images_dir(folder) / f"{anchor_stem}_{rep}.png")
                        for rep in self.representations
                    }
                    past_centers = [centers[i] for i in idx_window[: anchor_idx + 1]]
                    future_centers = [centers[i] for i in idx_window[anchor_idx + 1 :]]
                    dt = [
                        max(window_times[i + 1] - window_times[i], 1.0e-6)
                        for i in range(anchor_idx, len(window_times) - 1)
                    ]
                    samples.append(
                        (
                            folder,
                            int(track_id),
                            anchor_stem,
                            window_times[anchor_idx],
                            rep_paths,
                            past_centers,
                            future_centers,
                            dt,
                        )
                    )
        return samples

    def _apply_sample_limit(self) -> None:
        if self.max_samples is None or self.max_samples <= 0:
            return
        if not self.samples:
            return
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(self.samples))[: self.max_samples]
        self.samples = [self.samples[i] for i in perm]

    @staticmethod
    def _path_state(path: Path) -> Dict[str, object]:
        if not path.exists():
            return {"path": str(path), "exists": False}
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _dataset_state_signature(self) -> Dict[str, object]:
        folders = self.folders if self.folders is not None else [""]
        folder_states: List[Dict[str, object]] = []
        for folder in folders:
            frames = self.frames_by_folder.get(folder, [])
            labels_dir = self._labels_dir(folder)
            images_dir = self._images_dir(folder)
            tracks_path = self._tracks_path(folder)

            frame_entries: List[Dict[str, object]] = []
            for frame in frames:
                label_path = labels_dir / f"{frame.stem}.txt"
                rep_presence = {
                    rep: (images_dir / f"{frame.stem}_{rep}.png").exists()
                    for rep in self.representations
                }
                frame_entries.append(
                    {
                        "stem": frame.stem,
                        "label": self._path_state(label_path),
                        "rep_presence": rep_presence,
                    }
                )

            folder_states.append(
                {
                    "folder": folder,
                    "labels_dir": self._path_state(labels_dir),
                    "images_dir": self._path_state(images_dir),
                    "tracks": self._path_state(tracks_path),
                    "frames": frame_entries,
                }
            )
        return {"folders": folder_states}

    def _cache_key(self) -> str:
        payload = {
            "images_root": str(self.images_root),
            "labels_root": str(self.labels_root),
            "representations": self.representations,
            "image_size": self.image_size,
            "history_steps": self.history_steps,
            "future_steps": self.future_steps,
            "stride": self.stride,
            "frame_size": self.frame_size,
            "intrinsics": self.intrinsics,
            "camera_pose": self.camera_pose,
            "folders": self.folders,
            "labels_subdir": self.labels_subdir,
            "tracks_file": self.tracks_file,
            "label_time_unit": self.label_time_unit,
            "track_time_unit": self.track_time_unit,
            "time_align": self.time_align,
            "max_tracks": self.max_tracks,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "dataset_state": self._dataset_state_signature(),
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"pose_proj_samples_{self._cache_key()}.pkl"

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

    def __getitem__(self, idx: int) -> ProjectionSample:
        folder, track_id, stem, time_s, paths, past_centers, future_centers, dt = self.samples[idx]
        inputs = {
            rep: _load_image(Path(path), self.image_size) for rep, path in paths.items()
        }
        frame_key = f"{folder}/{stem}" if folder else stem
        return ProjectionSample(
            inputs=inputs,
            past_centers=torch.tensor(past_centers, dtype=torch.float32),
            future_centers=torch.tensor(future_centers, dtype=torch.float32),
            dt=torch.tensor(dt, dtype=torch.float32),
            intrinsics=torch.tensor(self.intrinsics, dtype=torch.float32),
            camera_pose=torch.tensor(self.camera_pose, dtype=torch.float32),
            frame_key=frame_key,
            frame_time_s=float(time_s),
            track_id=int(track_id),
        )
