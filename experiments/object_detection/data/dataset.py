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
class DetectionSample:
    inputs: Dict[str, torch.Tensor]
    box_xywh: torch.Tensor
    heatmaps: Dict[str, torch.Tensor]
    frame_key: str
    frame_time_s: float
    selected_box_index: int
    all_boxes_xywh: torch.Tensor
    input_paths: Dict[str, str]


def _parse_frame_time(name: str) -> Optional[int]:
    parts = name.split("_frame_")
    if len(parts) < 2:
        return None
    digits = ""
    for ch in parts[1]:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


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


def _select_box_index(boxes: List[Tuple[float, float, float, float]], mode: str) -> int:
    if mode == "largest":
        return int(np.argmax([w * h for _, _, w, h in boxes]))
    if mode == "first":
        return 0
    if mode == "center":
        scores = [abs(cx - 0.5) + abs(cy - 0.5) for cx, cy, _, _ in boxes]
        return int(np.argmin(scores))
    raise ValueError(f"Unknown select_box mode: {mode}")


class FredDetectionDataset(torch.utils.data.Dataset):
    CACHE_VERSION = 1

    def __init__(
        self,
        *,
        images_root: Path,
        labels_root: Path,
        representations: List[str],
        image_size: Tuple[int, int],
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
        select_box: str = "largest",
        max_samples: Optional[int] = None,
        seed: int = 123,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.representations = list(representations)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.heatmap_representations = list(heatmap_representations or ["xt_my", "yt_mx"])
        self.folders = folders
        self.labels_subdir = labels_subdir
        self.label_time_unit = float(label_time_unit)
        self.image_window_ms = float(image_window_ms)
        self.image_window_mode = str(image_window_mode)
        self.verify_render_manifest = bool(verify_render_manifest)
        self.render_manifest_name = str(render_manifest_name)
        self.window_tolerance_ms = float(window_tolerance_ms)
        self.require_boxes = bool(require_boxes)
        self.select_box = str(select_box)
        self.max_samples = max_samples
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._folder_manifests: Dict[str, Optional[dict]] = {}

        if not self.representations:
            raise ValueError("representations must not be empty.")
        missing_heatmap_reps = set(self.heatmap_representations) - set(self.representations)
        if missing_heatmap_reps:
            raise ValueError(f"Missing heatmap reps in representations: {sorted(missing_heatmap_reps)}")
        if self.image_window_mode not in {"trailing", "center", "leading"}:
            raise ValueError(f"Unknown image_window_mode: {self.image_window_mode}")

        cached = self._load_cache()
        if cached is not None:
            self.samples = cached
        else:
            self.samples = self._build_samples()
            self._apply_sample_limit()
            self._save_cache()

    def _labels_dir(self, folder: str) -> Path:
        return self.labels_root / folder / self.labels_subdir if self.folders is not None else self.labels_root

    def _images_dir(self, folder: str) -> Path:
        return self.images_root / folder if self.folders is not None else self.images_root

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
        by_stem = {
            entry.get("label_stem"): entry
            for entry in manifest.get("files", [])
            if entry.get("label_stem") is not None
        }
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

    def _has_all_reps(self, folder: str, stem: str) -> bool:
        base = self._images_dir(folder)
        return all((base / f"{stem}_{rep}.png").exists() for rep in self.representations)

    def _folders_to_scan(self) -> List[str]:
        return [""] if self.folders is None else list(self.folders)

    def _build_samples(self) -> List[dict]:
        samples: List[dict] = []
        for folder in self._folders_to_scan():
            labels_dir = self._labels_dir(folder)
            if not labels_dir.exists():
                continue
            for label_path in sorted(labels_dir.glob("*.txt")):
                time_raw = _parse_frame_time(label_path.stem)
                if time_raw is None:
                    continue
                boxes = _read_yolo_boxes(label_path)
                if self.require_boxes and not boxes:
                    continue
                if not self._has_all_reps(folder, label_path.stem):
                    continue
                self._validate_manifest_entry(folder, label_path.stem)
                samples.append(
                    {
                        "folder": folder,
                        "stem": label_path.stem,
                        "time_s": float(time_raw) * self.label_time_unit,
                        "boxes": boxes,
                        "selected_box_index": _select_box_index(boxes, self.select_box) if boxes else -1,
                        "input_paths": {
                            rep: str(self._images_dir(folder) / f"{label_path.stem}_{rep}.png")
                            for rep in self.representations
                        },
                    }
                )
        return samples

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
            "image_size": self.image_size,
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
            "select_box": self.select_box,
            "max_samples": self.max_samples,
            "seed": self.seed,
            "cache_version": self.CACHE_VERSION,
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
        width = self.image_size[0]
        height = self.image_size[1]
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
        inputs = {rep: _load_image(Path(path), self.image_size) for rep, path in sample["input_paths"].items()}
        all_boxes = (
            torch.tensor(sample["boxes"], dtype=torch.float32)
            if sample["boxes"]
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        if sample["selected_box_index"] >= 0:
            box = all_boxes[sample["selected_box_index"]]
            heatmaps = {
                rep: self._box_to_heatmap(rep, tuple(float(v) for v in box.tolist()))
                for rep in self.heatmap_representations
            }
        else:
            box = torch.zeros((4,), dtype=torch.float32)
            heatmaps = {
                rep: torch.zeros((1, self.image_size[1], self.image_size[0]), dtype=torch.float32)
                for rep in self.heatmap_representations
            }
        folder = sample["folder"]
        frame_key = f"{folder}/{sample['stem']}" if folder else sample["stem"]
        return DetectionSample(
            inputs=inputs,
            box_xywh=box,
            heatmaps=heatmaps,
            frame_key=frame_key,
            frame_time_s=float(sample["time_s"]),
            selected_box_index=int(sample["selected_box_index"]),
            all_boxes_xywh=all_boxes,
            input_paths=dict(sample["input_paths"]),
        )
