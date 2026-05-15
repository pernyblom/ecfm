from __future__ import annotations

from collections.abc import Iterator
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class DetectionSample:
    inputs: Dict[str, torch.Tensor]
    gt_boxes_xywh: torch.Tensor
    gt_velocities_xy: torch.Tensor
    gt_velocity_mask: torch.Tensor
    heatmaps: Dict[str, torch.Tensor]
    frame_key: str
    frame_time_s: float
    input_paths: Dict[str, str]


_FRAME_RE = re.compile(r"_frame_(\d+)", re.IGNORECASE)
_TRAILING_TIME_RE = re.compile(r"_(\d+)$")
_RGB_TIME_RE = re.compile(r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)$")
_DATASET_RGB_REPS = {"rgb": "RGB", "padded_rgb": "PADDED_RGB"}


def _parse_frame_time_s(name: str, *, label_time_unit: float) -> Optional[float]:
    match = _FRAME_RE.search(name)
    if match:
        return float(match.group(1)) * label_time_unit
    match = _TRAILING_TIME_RE.search(name)
    if match:
        return float(match.group(1)) * label_time_unit
    match = _RGB_TIME_RE.search(name)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        fraction = float(f"0.{match.group(4)}")
        return float(hours * 3600 + minutes * 60 + seconds) + fraction
    return None


def _parse_frame_time(name: str) -> Optional[int]:
    match = _FRAME_RE.search(name)
    if match:
        return int(match.group(1))
    match = _TRAILING_TIME_RE.search(name)
    if match:
        return int(match.group(1))
    return None


def _is_dataset_rgb_rep(rep: str) -> bool:
    return rep.lower() in _DATASET_RGB_REPS


def _resolve_labels_subdir(labels_subdir: str, representations: List[str]) -> str:
    if str(labels_subdir).lower() != "auto":
        return str(labels_subdir)
    if all(_is_dataset_rgb_rep(rep) for rep in representations):
        return "RGB_YOLO"
    return "Event_YOLO"


def _load_image(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size[0], size[1]):
        img = img.resize(size, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _read_yolo_boxes(path: Path) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            out.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return out


def _progress_items(items: List[Tuple[str, Path]], *, desc: str, enabled: bool) -> Iterator[Tuple[str, Path]]:
    if not enabled:
        yield from items
        return
    try:
        from tqdm import tqdm
    except ImportError:
        yield from _simple_progress_items(items, desc=desc)
        return
    yield from tqdm(items, desc=desc, unit="label", dynamic_ncols=True)


def _simple_progress_items(items: List[Tuple[str, Path]], *, desc: str) -> Iterator[Tuple[str, Path]]:
    total = len(items)
    if total == 0:
        return
    width = 32
    start = time.monotonic()

    def show(done: int, current: str | None = None) -> None:
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else 0.0
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        suffix = f" | {current}" if current else ""
        print(
            f"\r{desc} [{bar}] {done}/{total} "
            f"elapsed {elapsed:5.1f}s eta {remaining:5.1f}s{suffix}",
            end="",
            flush=True,
        )

    show(0)
    for index, item in enumerate(items, start=1):
        folder, label_path = item
        current = f"{folder}/{label_path.name}" if folder else label_path.name
        yield item
        show(index, current)
    print()


def _box_xyxy_to_xywh_norm(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    frame_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    frame_w, frame_h = frame_size
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return (
        (x0 + w / 2.0) / frame_w,
        (y0 + h / 2.0) / frame_h,
        w / frame_w,
        h / frame_h,
    )


def _box_iou_xywh_norm(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0 = a[0] - a[2] / 2.0
    ay0 = a[1] - a[3] / 2.0
    ax1 = a[0] + a[2] / 2.0
    ay1 = a[1] + a[3] / 2.0
    bx0 = b[0] - b[2] / 2.0
    by0 = b[1] - b[3] / 2.0
    bx1 = b[0] + b[2] / 2.0
    by1 = b[1] + b[3] / 2.0
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FredDetectionDataset(torch.utils.data.Dataset):
    CACHE_VERSION = 8

    def __init__(
        self,
        *,
        images_root: Path,
        labels_root: Path,
        representations: List[str],
        image_sizes: Dict[str, Tuple[int, int]],
        frame_size: Tuple[int, int],
        heatmap_representations: Optional[List[str]] = None,
        folders: Optional[List[str]] = None,
        labels_subdir: str = "Event_YOLO",
        label_time_unit: float = 1e-6,
        image_window_ms: float = 33.333,
        image_window_mode: str = "trailing",
        verify_render_manifest: bool = True,
        render_manifest_name: str = "render_manifest.json",
        window_tolerance_ms: float = 2.0,
        require_boxes: bool = True,
        exclude_multiple_objects: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 123,
        cache_dir: Optional[Path] = None,
        velocity_tracks_file: Optional[str] = "cleaned_tracks.txt",
        velocity_match_iou: float = 0.3,
        filter_missing_representations: bool = True,
        show_build_progress: bool = False,
    ) -> None:
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.representations = list(representations)
        self.image_sizes = {
            str(rep): (int(size[0]), int(size[1])) for rep, size in dict(image_sizes).items()
        }
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.heatmap_representations = (
            ["xt_my", "yt_mx"] if heatmap_representations is None else list(heatmap_representations)
        )
        self.folders = folders
        self.labels_subdir = _resolve_labels_subdir(labels_subdir, self.representations)
        self.label_time_unit = float(label_time_unit)
        self.image_window_ms = float(image_window_ms)
        self.image_window_mode = str(image_window_mode)
        self.verify_render_manifest = bool(verify_render_manifest)
        self.render_manifest_name = str(render_manifest_name)
        self.window_tolerance_ms = float(window_tolerance_ms)
        self.require_boxes = bool(require_boxes)
        self.exclude_multiple_objects = bool(exclude_multiple_objects)
        self.max_samples = max_samples
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.velocity_tracks_file = velocity_tracks_file
        self.velocity_match_iou = float(velocity_match_iou)
        self.filter_missing_representations = bool(filter_missing_representations)
        self.show_build_progress = bool(show_build_progress)
        self._folder_manifests: Dict[str, Optional[dict]] = {}
        self._folder_manifest_entries: Dict[str, Optional[Dict[str, dict]]] = {}
        self._folder_available_stems: Dict[str, set[str]] = {}
        self._folder_tracks: Dict[str, Optional[dict]] = {}
        self._folder_rgb_indices: Dict[Tuple[str, str], List[Tuple[float, Path]]] = {}

        if not self.representations:
            raise ValueError("representations must not be empty.")
        missing_sizes = [rep for rep in self.representations if rep not in self.image_sizes]
        if missing_sizes:
            raise ValueError(f"Missing image sizes for representations: {missing_sizes}")
        missing_heatmap_reps = set(self.heatmap_representations) - set(self.representations)
        if missing_heatmap_reps:
            raise ValueError(f"Missing heatmap reps in representations: {sorted(missing_heatmap_reps)}")
        if self.image_window_mode not in {"trailing", "center", "leading"}:
            raise ValueError(f"Unknown image_window_mode: {self.image_window_mode}")

        cached = self._load_cache()
        if cached is not None:
            self.samples = cached
            self._filter_cached_missing_representations()
        else:
            self.samples = self._build_samples()
            self._apply_sample_limit()
            self._save_cache()

    def _labels_dir(self, folder: str) -> Path:
        return self.labels_root / folder / self.labels_subdir if self.folders is not None else self.labels_root

    def _images_dir(self, folder: str) -> Path:
        return self.images_root / folder if self.folders is not None else self.images_root

    def _dataset_rgb_dir(self, folder: str, rep: str) -> Path:
        dirname = _DATASET_RGB_REPS[rep.lower()]
        return self.labels_root / folder / dirname if self.folders is not None else self.labels_root / dirname

    def _parse_rgb_time(self, path: Path) -> Optional[float]:
        match = _RGB_TIME_RE.search(path.stem)
        if not match:
            return None
        hh, mm, ss, frac = match.groups()
        try:
            hours = int(hh)
            minutes = int(mm)
            seconds = int(ss)
            micros = int(frac.ljust(6, "0")[:6])
        except ValueError:
            return None
        return hours * 3600.0 + minutes * 60.0 + seconds + micros / 1_000_000.0

    def _build_rgb_index(self, folder: str, rep: str) -> List[Tuple[float, Path]]:
        key = (folder, rep.lower())
        if key in self._folder_rgb_indices:
            return self._folder_rgb_indices[key]
        rgb_dir = self._dataset_rgb_dir(folder, rep)
        if not rgb_dir.exists():
            self._folder_rgb_indices[key] = []
            return []
        files = [
            path
            for pattern in ("*.jpg", "*.png", "*.jpeg")
            for path in sorted(rgb_dir.glob(pattern))
            if not path.name.startswith(".") and not path.name.startswith("._")
        ]
        parsed = [self._parse_rgb_time(path) for path in files]
        out: List[Tuple[float, Path]] = []
        if any(t is not None for t in parsed):
            base = next(t for t in parsed if t is not None)
            for path, t in zip(files, parsed):
                if t is None:
                    continue
                rel_us = (t - base) * 1_000_000.0
                out.append((rel_us * self.label_time_unit, path))
        else:
            for idx, path in enumerate(files):
                out.append((float(idx), path))
        out.sort(key=lambda item: item[0])
        self._folder_rgb_indices[key] = out
        return out

    def _find_rgb_frame(self, folder: str, rep: str, label_time_s: float) -> Optional[Path]:
        rgb_index = self._build_rgb_index(folder, rep)
        if not rgb_index:
            return None
        times = [t for t, _ in rgb_index]
        idx = int(np.searchsorted(times, label_time_s, side="left"))
        if idx <= 0:
            return rgb_index[0][1]
        if idx >= len(rgb_index):
            return rgb_index[-1][1]
        before_t, before_path = rgb_index[idx - 1]
        after_t, after_path = rgb_index[idx]
        return before_path if abs(label_time_s - before_t) <= abs(after_t - label_time_s) else after_path

    def _resolve_input_path(self, folder: str, stem: str, rep: str) -> Optional[Path]:
        rendered_path = self._images_dir(folder) / f"{stem}_{rep}.png"
        if rendered_path.exists():
            return rendered_path
        if _is_dataset_rgb_rep(rep):
            rgb_dir = self._dataset_rgb_dir(folder, rep)
            for suffix in (".jpg", ".png", ".jpeg"):
                candidate = rgb_dir / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
            label_time_s = _parse_frame_time_s(stem, label_time_unit=self.label_time_unit)
            if label_time_s is not None:
                return self._find_rgb_frame(folder, rep, label_time_s)
        return None

    def _expected_input_path(self, folder: str, stem: str, rep: str) -> Path:
        if _is_dataset_rgb_rep(rep) and self.labels_subdir == "RGB_YOLO":
            return self._dataset_rgb_dir(folder, rep) / f"{stem}.jpg"
        return self._images_dir(folder) / f"{stem}_{rep}.png"

    def _manifest_path(self, folder: str) -> Path:
        return self._images_dir(folder) / self.render_manifest_name

    def _load_render_manifest(self, folder: str) -> Optional[dict]:
        if folder in self._folder_manifests:
            return self._folder_manifests[folder]
        path = self._manifest_path(folder)
        if not path.exists():
            self._folder_manifests[folder] = None
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            manifest = None
        self._folder_manifests[folder] = manifest
        return manifest

    def _validate_manifest_entry(self, folder: str, stem: str) -> None:
        if not self.verify_render_manifest:
            return
        manifest = self._load_render_manifest(folder)
        if manifest is None:
            raise FileNotFoundError(f"Missing render manifest for folder '{folder or '.'}'.")
        params = manifest.get("render_params") or {}
        if params.get("window_mode") != self.image_window_mode:
            raise ValueError(
                f"Window mode mismatch for '{folder}/{stem}': expected {self.image_window_mode}, "
                f"found {params.get('window_mode')}"
            )
        by_stem = self._folder_manifest_entries.get(folder)
        if by_stem is None:
            by_stem = {
                entry.get("label_stem"): entry
                for entry in manifest.get("files", [])
                if entry.get("label_stem") is not None
            }
            self._folder_manifest_entries[folder] = by_stem
        entry = by_stem.get(stem)
        if entry is None:
            raise ValueError(f"Manifest entry missing for '{folder}/{stem}'.")
        actual_s = float(entry.get("window_duration_render_units", -1.0)) * self.label_time_unit
        expected_s = self.image_window_ms / 1000.0
        tol_s = self.window_tolerance_ms / 1000.0
        if abs(actual_s - expected_s) > tol_s:
            raise ValueError(
                f"Window duration mismatch for '{folder}/{stem}': expected {expected_s:.6f}s, "
                f"found {actual_s:.6f}s."
            )
        rep_entries = entry.get("representations") or {}
        for rep in self.representations:
            rep_entry = rep_entries.get(rep)
            if rep_entry is None:
                if _is_dataset_rgb_rep(rep):
                    continue
                raise ValueError(f"Manifest representation '{rep}' missing for '{folder}/{stem}'.")
            expected_size = list(self.image_sizes[rep])
            actual_size = rep_entry.get("image_size")
            if actual_size is not None and list(actual_size) != expected_size:
                raise ValueError(
                    f"Image size mismatch for '{folder}/{stem}' rep '{rep}': "
                    f"expected {expected_size}, found {actual_size}."
                )

    def _has_all_reps(self, folder: str, stem: str) -> bool:
        return all(self._resolve_input_path(folder, stem, rep) is not None for rep in self.representations)

    def _missing_representation_paths(self, folder: str, stem: str) -> Dict[str, Path]:
        missing: Dict[str, Path] = {}
        for rep in self.representations:
            if self._resolve_input_path(folder, stem, rep) is None:
                missing[rep] = self._expected_input_path(folder, stem, rep)
        return missing

    def _warn_missing_representations(
        self,
        *,
        skipped: int,
        missing_by_rep: Counter,
        examples: List[str],
        source: str,
    ) -> None:
        if skipped <= 0:
            return
        rep_summary = ", ".join(f"{rep}={count}" for rep, count in sorted(missing_by_rep.items()))
        print(
            f"Warning: skipped {skipped} object detection samples with missing representation files "
            f"while {source}. Missing counts: {rep_summary or 'unknown'}."
        )
        if examples:
            print("Warning: missing representation examples:")
            for example in examples:
                print(f"  {example}")

    def _folders_to_scan(self) -> List[str]:
        return [""] if self.folders is None else list(self.folders)

    def _label_files_to_scan(self) -> List[Tuple[str, Path]]:
        items: List[Tuple[str, Path]] = []
        for folder in self._folders_to_scan():
            labels_dir = self._labels_dir(folder)
            if not labels_dir.exists():
                continue
            for label_path in sorted(labels_dir.glob("*.txt")):
                items.append((folder, label_path))
        return items

    def _build_samples(self) -> List[dict]:
        samples: List[dict] = []
        label_items = self._label_files_to_scan()
        progress_desc = "Building detection dataset"
        skipped_missing = 0
        missing_by_rep: Counter = Counter()
        missing_examples: List[str] = []
        for folder, label_path in _progress_items(
            label_items,
            desc=progress_desc,
            enabled=self.show_build_progress,
        ):
            time_s = _parse_frame_time_s(label_path.stem, label_time_unit=self.label_time_unit)
            if time_s is None:
                continue
            boxes = _read_yolo_boxes(label_path)
            if self.require_boxes and not boxes:
                continue
            if self.exclude_multiple_objects and len(boxes) > 1:
                continue
            if not self._has_all_reps(folder, label_path.stem):
                missing = self._missing_representation_paths(folder, label_path.stem)
                if self.filter_missing_representations:
                    skipped_missing += 1
                    missing_by_rep.update(missing.keys())
                    if len(missing_examples) < 5:
                        key = f"{folder}/{label_path.stem}" if folder else label_path.stem
                        missing_examples.append(f"{key}: {', '.join(sorted(missing))}")
                    continue
                missing_paths = ", ".join(str(path) for path in missing.values())
                raise FileNotFoundError(
                    f"Missing representation file(s) for '{folder}/{label_path.stem}': {missing_paths}"
                )
            self._validate_manifest_entry(folder, label_path.stem)
            samples.append(
                {
                    "folder": folder,
                    "stem": label_path.stem,
                    "time_s": time_s,
                    "boxes": boxes,
                    "input_paths": {
                        rep: str(self._resolve_input_path(folder, label_path.stem, rep))
                        for rep in self.representations
                    },
                }
            )
        self._warn_missing_representations(
            skipped=skipped_missing,
            missing_by_rep=missing_by_rep,
            examples=missing_examples,
            source="building the sample index",
        )
        return samples

    def _filter_cached_missing_representations(self) -> None:
        if not self.samples:
            return
        kept: List[dict] = []
        skipped_missing = 0
        missing_by_rep: Counter = Counter()
        missing_examples: List[str] = []
        for sample in self.samples:
            missing = {
                rep: Path(path)
                for rep, path in dict(sample.get("input_paths") or {}).items()
                if not Path(path).exists()
            }
            if not missing:
                kept.append(sample)
                continue
            if not self.filter_missing_representations:
                key = f"{sample.get('folder', '')}/{sample.get('stem', '')}".strip("/")
                missing_paths = ", ".join(str(path) for path in missing.values())
                raise FileNotFoundError(f"Missing cached representation file(s) for '{key}': {missing_paths}")
            skipped_missing += 1
            missing_by_rep.update(missing.keys())
            if len(missing_examples) < 5:
                key = f"{sample.get('folder', '')}/{sample.get('stem', '')}".strip("/")
                missing_examples.append(f"{key}: {', '.join(sorted(missing))}")
        self.samples = kept
        self._warn_missing_representations(
            skipped=skipped_missing,
            missing_by_rep=missing_by_rep,
            examples=missing_examples,
            source="loading the cached sample index",
        )

    def _load_tracks(self, folder: str) -> Optional[dict]:
        if folder in self._folder_tracks:
            return self._folder_tracks[folder]
        if not self.velocity_tracks_file:
            self._folder_tracks[folder] = None
            return None
        path = self.labels_root / folder / self.velocity_tracks_file if folder else self.labels_root / self.velocity_tracks_file
        if not path.exists():
            self._folder_tracks[folder] = None
            return None
        by_time: Dict[float, List[dict]] = {}
        by_id: Dict[int, List[dict]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                time_s = float(parts[0])
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError:
                continue
            box = _box_xyxy_to_xywh_norm(x, y, x + w, y + h, self.frame_size)
            row = {"time_s": time_s, "track_id": track_id, "box": box}
            by_time.setdefault(round(time_s, 6), []).append(row)
            by_id.setdefault(track_id, []).append(row)
        for rows in by_id.values():
            rows.sort(key=lambda item: item["time_s"])
        tracks = {"by_time": by_time, "by_id": by_id}
        self._folder_tracks[folder] = tracks
        return tracks

    def _estimate_velocity_for_track(self, tracks: dict, row: dict) -> Optional[Tuple[float, float]]:
        rows = tracks["by_id"].get(row["track_id"], [])
        idx = next((i for i, item in enumerate(rows) if item["time_s"] == row["time_s"]), None)
        if idx is None:
            return None
        prev_row = rows[idx - 1] if idx > 0 else None
        next_row = rows[idx + 1] if idx + 1 < len(rows) else None
        if prev_row is not None and next_row is not None:
            dt = next_row["time_s"] - prev_row["time_s"]
            dx = next_row["box"][0] - prev_row["box"][0]
            dy = next_row["box"][1] - prev_row["box"][1]
        elif prev_row is not None:
            dt = row["time_s"] - prev_row["time_s"]
            dx = row["box"][0] - prev_row["box"][0]
            dy = row["box"][1] - prev_row["box"][1]
        elif next_row is not None:
            dt = next_row["time_s"] - row["time_s"]
            dx = next_row["box"][0] - row["box"][0]
            dy = next_row["box"][1] - row["box"][1]
        else:
            return None
        if abs(dt) <= 1.0e-8:
            return None
        return dx / dt, dy / dt

    def _estimate_velocities(
        self,
        folder: str,
        time_s: float,
        boxes: List[Tuple[float, float, float, float]],
    ) -> List[Optional[Tuple[float, float]]]:
        tracks = self._load_tracks(folder)
        if tracks is None or not boxes:
            return [None for _ in boxes]
        rows = tracks["by_time"].get(round(time_s, 6), [])
        out: List[Optional[Tuple[float, float]]] = []
        used: set[int] = set()
        for box in boxes:
            best_idx = None
            best_iou = 0.0
            for idx, row in enumerate(rows):
                if idx in used:
                    continue
                iou = _box_iou_xywh_norm(box, row["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is None or best_iou < self.velocity_match_iou:
                out.append(None)
                continue
            used.add(best_idx)
            out.append(self._estimate_velocity_for_track(tracks, rows[best_idx]))
        return out

    def _apply_sample_limit(self) -> None:
        if self.max_samples is None or self.max_samples <= 0 or len(self.samples) <= self.max_samples:
            return
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(self.samples))[: self.max_samples]
        self.samples = [self.samples[i] for i in idx]

    def _cache_key(self) -> str:
        payload = {
            "images_root": str(self.images_root),
            "labels_root": str(self.labels_root),
            "representations": self.representations,
            "image_sizes": self.image_sizes,
            "frame_size": self.frame_size,
            "heatmap_representations": self.heatmap_representations,
            "folders": self.folders,
            "labels_subdir": self.labels_subdir,
            "label_time_unit": self.label_time_unit,
            "image_window_ms": self.image_window_ms,
            "image_window_mode": self.image_window_mode,
            "verify_render_manifest": self.verify_render_manifest,
            "render_manifest_name": self.render_manifest_name,
            "window_tolerance_ms": self.window_tolerance_ms,
            "require_boxes": self.require_boxes,
            "exclude_multiple_objects": self.exclude_multiple_objects,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "cache_version": self.CACHE_VERSION,
            "velocity_tracks_file": self.velocity_tracks_file,
            "velocity_match_iou": self.velocity_match_iou,
            "filter_missing_representations": self.filter_missing_representations,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"object_detection_samples_{self._cache_key()}.pkl"

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

    def _box_to_heatmap(self, rep: str, box: Tuple[float, float, float, float]) -> torch.Tensor:
        cx, cy, bw, bh = box
        width, height = self.image_sizes[rep]
        heat = torch.zeros((1, height, width), dtype=torch.float32)
        x0 = max(0, min(width - 1, int(np.floor((cx - bw / 2.0) * width))))
        x1 = max(x0 + 1, min(width, int(np.ceil((cx + bw / 2.0) * width))))
        y0 = max(0, min(height - 1, int(np.floor((cy - bh / 2.0) * height))))
        y1 = max(y0 + 1, min(height, int(np.ceil((cy + bh / 2.0) * height))))
        if rep == "xt_my":
            heat[0, :, x0:x1] = 1.0
            return heat
        if rep == "yt_mx":
            heat[0, y0:y1, :] = 1.0
            return heat
        raise ValueError(f"Unsupported heatmap representation: {rep}")

    def __getitem__(self, idx: int) -> DetectionSample:
        sample = self.samples[idx]
        inputs = {
            rep: _load_image(Path(path), self.image_sizes[rep])
            for rep, path in sample["input_paths"].items()
        }
        gt_boxes = (
            torch.tensor(sample["boxes"], dtype=torch.float32)
            if sample["boxes"]
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        velocities_raw = sample.get("velocities")
        if velocities_raw is None:
            velocities_raw = self._estimate_velocities(
                sample["folder"],
                float(sample["time_s"]),
                [tuple(float(v) for v in box.tolist()) for box in gt_boxes],
            )
        velocities: List[Tuple[float, float]] = []
        velocity_mask: List[bool] = []
        for velocity in velocities_raw:
            if velocity is None:
                velocities.append((0.0, 0.0))
                velocity_mask.append(False)
            else:
                velocities.append((float(velocity[0]), float(velocity[1])))
                velocity_mask.append(True)
        gt_velocities = (
            torch.tensor(velocities, dtype=torch.float32)
            if velocities
            else torch.zeros((0, 2), dtype=torch.float32)
        )
        gt_velocity_mask = (
            torch.tensor(velocity_mask, dtype=torch.bool)
            if velocity_mask
            else torch.zeros((0,), dtype=torch.bool)
        )
        heatmaps = {
            rep: torch.zeros((1, self.image_sizes[rep][1], self.image_sizes[rep][0]), dtype=torch.float32)
            for rep in self.heatmap_representations
        }
        for box in gt_boxes:
            for rep in self.heatmap_representations:
                heatmaps[rep] = torch.maximum(
                    heatmaps[rep],
                    self._box_to_heatmap(rep, tuple(float(v) for v in box.tolist())),
                )
        folder = sample["folder"]
        frame_key = f"{folder}/{sample['stem']}" if folder else sample["stem"]
        return DetectionSample(
            inputs=inputs,
            gt_boxes_xywh=gt_boxes,
            gt_velocities_xy=gt_velocities,
            gt_velocity_mask=gt_velocity_mask,
            heatmaps=heatmaps,
            frame_key=frame_key,
            frame_time_s=float(sample["time_s"]),
            input_paths=dict(sample["input_paths"]),
        )
