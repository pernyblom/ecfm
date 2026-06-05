from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class KalmanForecastSample:
    inputs: Dict[str, torch.Tensor]
    past_boxes: torch.Tensor
    future_boxes: torch.Tensor
    past_times_s: torch.Tensor
    future_times_s: torch.Tensor
    frame_key: str
    frame_time_s: float
    track_id: int
    input_paths: Dict[str, str]


@dataclass
class FrameItem:
    stem: str
    time_s: float


_FRAME_RE = re.compile(r"_frame_(\d+)", re.IGNORECASE)
_TRAILING_TIME_RE = re.compile(r"_(\d+)$")
_RGB_TIME_RE = re.compile(r"_(\d{2})_(\d{2})_(\d{2})\.(\d+)$")
_DATASET_RGB_REPS = {"rgb": "RGB", "padded_rgb": "PADDED_RGB"}
_DATASET_EVENT_FRAME_REPS = {"event_frames", "event_frame"}


def _base_representation_name(rep: str) -> str:
    match = re.match(r"^(?P<base>.+)_\d+x\d+$", str(rep), flags=re.IGNORECASE)
    return match.group("base") if match is not None else str(rep)


def _sample_motion_features(
    *,
    cx_norm: float,
    cy_norm: float,
    vx_px_s: float,
    vy_px_s: float,
    feature_mode: str,
) -> list[float]:
    dx = float(cx_norm) - 0.5
    dy = float(cy_norm) - 0.5
    vx = float(vx_px_s)
    vy = float(vy_px_s)
    if feature_mode == "raw":
        return [float(cx_norm), float(cy_norm), vx, vy]
    if feature_mode == "centered":
        return [dx, dy, vx, vy]
    if feature_mode == "motion_priors":
        return [dx, dy, vx, vy, float(np.hypot(dx, dy)), float(np.hypot(vx, vy))]
    raise ValueError("feature_mode must be one of: raw, centered, motion_priors.")


def _fit_center_constant_acceleration(
    times_s: np.ndarray,
    centers_px: np.ndarray,
    *,
    anchor_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel_t = times_s.astype(np.float64) - float(anchor_time_s)
    design = np.stack([np.ones_like(rel_t), rel_t, 0.5 * rel_t * rel_t], axis=1)
    coeff, *_ = np.linalg.lstsq(design, centers_px.astype(np.float64), rcond=None)
    return coeff[0], coeff[1], coeff[2]


def _progress_iter(items: Iterable[int], *, desc: str, enabled: bool):
    if not enabled:
        yield from items
        return
    try:
        from tqdm import tqdm
    except ImportError:
        items = list(items)
        total = len(items)
        start = time.monotonic()
        for idx, item in enumerate(items, start=1):
            if idx == 1 or idx == total or idx % max(1, total // 20) == 0:
                elapsed = time.monotonic() - start
                print(f"{desc}: {idx}/{total} elapsed {elapsed:.1f}s")
            yield item
        return
    yield from tqdm(items, desc=desc, unit="removal", dynamic_ncols=True)


def _parse_frame_time_raw(name: str) -> Optional[int]:
    match = _FRAME_RE.search(name)
    if match:
        return int(match.group(1))
    match = _TRAILING_TIME_RE.search(name)
    if match:
        return int(match.group(1))
    return None


def _is_dataset_rgb_rep(rep: str) -> bool:
    return rep.lower() in _DATASET_RGB_REPS


def _is_dataset_event_frame_rep(rep: str) -> bool:
    return rep.lower() in _DATASET_EVENT_FRAME_REPS


def _is_dataset_native_rep(rep: str) -> bool:
    return _is_dataset_rgb_rep(rep) or _is_dataset_event_frame_rep(rep)


def _size_pair_from_config(value: Any, *, name: str) -> tuple[float, float]:
    if isinstance(value, dict):
        if "width" in value or "height" in value:
            return float(value.get("width", value.get("x", 0.0))), float(value.get("height", value.get("y", 0.0)))
        if "x" in value or "y" in value:
            return float(value.get("x", 0.0)), float(value.get("y", 0.0))
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a number or a two-item [width, height] value.")
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, scalar


def _spatial_cutout_mode(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("mode", "none")).lower()


def _fixed_spatial_cutout_size(cfg: Dict[str, Any]) -> tuple[int, int] | None:
    mode = _spatial_cutout_mode(cfg)
    if mode not in {"fixed", "fixed_pixels", "fixed_px"}:
        return None
    size_value = cfg.get("size_px", cfg.get("fixed_size_px", cfg.get("size", None)))
    if size_value is None:
        raise ValueError("data.spatial_cutout fixed mode requires size_px or fixed_size_px.")
    cut_w, cut_h = _size_pair_from_config(size_value, name="data.spatial_cutout.size_px")
    cut_w_i = max(1, int(round(cut_w)))
    cut_h_i = max(1, int(round(cut_h)))
    return cut_w_i, cut_h_i


def _crop_array_with_padding(
    arr: np.ndarray,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    fill: float,
) -> np.ndarray:
    width = max(1, int(width))
    height = max(1, int(height))
    out = np.full((height, width, arr.shape[2]), float(fill), dtype=arr.dtype)
    src_left = max(0, int(left))
    src_top = max(0, int(top))
    src_right = min(arr.shape[1], int(left) + width)
    src_bottom = min(arr.shape[0], int(top) + height)
    if src_right <= src_left or src_bottom <= src_top:
        return out
    dst_left = src_left - int(left)
    dst_top = src_top - int(top)
    out[
        dst_top : dst_top + (src_bottom - src_top),
        dst_left : dst_left + (src_right - src_left),
        :,
    ] = arr[src_top:src_bottom, src_left:src_right, :]
    return out


def _fixed_spatial_cutout_array(
    arr: np.ndarray,
    *,
    rep: str,
    anchor_box: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray | None:
    fixed_size = _fixed_spatial_cutout_size(cfg)
    if fixed_size is None:
        return None
    cut_w, cut_h = fixed_size
    img_h, img_w = int(arr.shape[0]), int(arr.shape[1])
    cx = float(anchor_box[0]) * img_w
    cy = float(anchor_box[1]) * img_h
    base_rep = _base_representation_name(rep).lower()
    if base_rep.startswith("xt"):
        width = cut_w
        height = img_h
        left = int(round(cx - width / 2.0))
        top = 0
    elif base_rep.startswith("yt"):
        width = img_w
        height = cut_h
        left = 0
        top = int(round(cy - height / 2.0))
    else:
        width = cut_w
        height = cut_h
        left = int(round(cx - width / 2.0))
        top = int(round(cy - height / 2.0))
    return _crop_array_with_padding(
        arr,
        left=left,
        top=top,
        width=width,
        height=height,
        fill=float(cfg.get("fill", cfg.get("fill_value", 0.0))),
    )


def _spatial_cutout_bounds(
    *,
    rep: str,
    image_size: Tuple[int, int],
    frame_size: Tuple[float, float],
    anchor_box: np.ndarray,
    cfg: Dict[str, Any],
) -> tuple[int, int, int, int] | None:
    mode = str(cfg.get("mode", "none")).lower()
    if mode in {"none", "disabled", "off", "false"}:
        return None
    if mode in {"box_fraction", "box_scale", "box"}:
        scale = float(cfg.get("scale", cfg.get("box_scale", cfg.get("fraction", 1.0))))
        if scale <= 0.0:
            raise ValueError("data.spatial_cutout scale/fraction must be > 0.")
        cut_w_px = float(anchor_box[2]) * float(frame_size[0]) * scale
        cut_h_px = float(anchor_box[3]) * float(frame_size[1]) * scale
    elif mode in {"fixed", "fixed_pixels", "fixed_px"}:
        size_value = cfg.get("size_px", cfg.get("fixed_size_px", cfg.get("size", None)))
        if size_value is None:
            raise ValueError("data.spatial_cutout fixed mode requires size_px or fixed_size_px.")
        cut_w_px, cut_h_px = _size_pair_from_config(size_value, name="data.spatial_cutout.size_px")
    else:
        raise ValueError(
            "data.spatial_cutout.mode must be one of: none, box_scale, box_fraction, fixed_pixels, fixed."
        )

    min_size_px = float(cfg.get("min_size_px", 1.0))
    cut_w_px = max(min_size_px, cut_w_px)
    cut_h_px = max(min_size_px, cut_h_px)
    frame_w, frame_h = float(frame_size[0]), float(frame_size[1])
    img_w, img_h = int(image_size[0]), int(image_size[1])
    cx_px = float(anchor_box[0]) * frame_w
    cy_px = float(anchor_box[1]) * frame_h
    base_rep = _base_representation_name(rep).lower()

    x0, x1 = 0, img_w
    y0, y1 = 0, img_h
    if base_rep.startswith("xt"):
        center = cx_px / frame_w * img_w
        half = cut_w_px / frame_w * img_w / 2.0
        x0, x1 = int(np.floor(center - half)), int(np.ceil(center + half))
    elif base_rep.startswith("yt"):
        center = cy_px / frame_h * img_h
        half = cut_h_px / frame_h * img_h / 2.0
        y0, y1 = int(np.floor(center - half)), int(np.ceil(center + half))
    else:
        center_x = cx_px / frame_w * img_w
        center_y = cy_px / frame_h * img_h
        half_w = cut_w_px / frame_w * img_w / 2.0
        half_h = cut_h_px / frame_h * img_h / 2.0
        x0, x1 = int(np.floor(center_x - half_w)), int(np.ceil(center_x + half_w))
        y0, y1 = int(np.floor(center_y - half_h)), int(np.ceil(center_y + half_h))

    x0 = max(0, min(img_w, x0))
    x1 = max(0, min(img_w, x1))
    y0 = max(0, min(img_h, y0))
    y1 = max(0, min(img_h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _load_image(
    path: Path,
    size: Tuple[int, int],
    *,
    source_size: Tuple[int, int],
    rep: str,
    frame_size: Tuple[float, float],
    anchor_box: np.ndarray,
    spatial_cutout: Dict[str, Any],
) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (source_size[0], source_size[1]):
        img = img.resize(source_size, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    fixed_cutout = _fixed_spatial_cutout_array(
        arr,
        rep=rep,
        anchor_box=anchor_box,
        cfg=spatial_cutout,
    )
    if fixed_cutout is not None:
        arr = fixed_cutout
        if (arr.shape[1], arr.shape[0]) != (size[0], size[1]):
            raise ValueError(
                f"Fixed spatial cutout for representation '{rep}' produced "
                f"{arr.shape[1]}x{arr.shape[0]}, but resolved image size is {size[0]}x{size[1]}."
            )
        return torch.from_numpy(arr).permute(2, 0, 1)
    if img.size != (size[0], size[1]):
        img = img.resize(size, resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    bounds = _spatial_cutout_bounds(
        rep=rep,
        image_size=size,
        frame_size=frame_size,
        anchor_box=anchor_box,
        cfg=spatial_cutout,
    )
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        fill = float(spatial_cutout.get("fill", spatial_cutout.get("fill_value", 0.0)))
        cut = np.full_like(arr, fill)
        cut[y0:y1, x0:x1, :] = arr[y0:y1, x0:x1, :]
        arr = cut
    return torch.from_numpy(arr).permute(2, 0, 1)


def _read_tracks(path: Path) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    tracks: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    if not path.exists():
        return tracks
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            t = float(parts[0])
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
        except ValueError:
            continue
        tracks.setdefault(track_id, []).append((t, x, y, w, h))
    for rows in tracks.values():
        rows.sort(key=lambda item: item[0])
    return tracks


class TrackKalmanForecastDataset(torch.utils.data.Dataset):
    CACHE_VERSION = 3

    def __init__(
        self,
        *,
        images_root: Path,
        labels_root: Path,
        frame_size: Tuple[int, int],
        representations: List[str],
        image_sizes: Dict[str, Tuple[int, int]],
        history_ms: float,
        forecast_ms: float,
        source_image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
        folders: Optional[List[str]] = None,
        labels_subdir: str = "Event_YOLO",
        tracks_file: str = "cleaned_tracks.txt",
        label_time_unit: float = 1e-6,
        track_time_unit: float = 1.0,
        time_align: str = "auto",
        image_window_ms: float = 400.0,
        image_window_mode: str = "trailing",
        verify_render_manifest: bool = True,
        render_manifest_name: str = "render_manifest.json",
        window_tolerance_ms: float = 5.0,
        label_period_s: Optional[float] = None,
        min_track_duration_ms: Optional[float] = None,
        max_tracks: Optional[int] = None,
        max_samples: Optional[int] = None,
        seed: int = 123,
        cache_dir: Optional[Path] = None,
        filter_missing_representations: bool = True,
        require_representations: bool = True,
        sample_decorrelation: Optional[Dict[str, Any]] = None,
        spatial_cutout: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.frame_size = (float(frame_size[0]), float(frame_size[1]))
        self.representations = list(representations)
        self.image_sizes = {str(k): (int(v[0]), int(v[1])) for k, v in image_sizes.items()}
        self.source_image_sizes = {
            str(k): (int(v[0]), int(v[1]))
            for k, v in dict(source_image_sizes or image_sizes).items()
        }
        self.history_ms = float(history_ms)
        self.forecast_ms = float(forecast_ms)
        self.folders = folders
        self.labels_subdir = labels_subdir
        self.tracks_file = tracks_file
        self.label_time_unit = float(label_time_unit)
        self.track_time_unit = float(track_time_unit)
        self.time_align = str(time_align)
        self.image_window_ms = float(image_window_ms)
        self.image_window_mode = str(image_window_mode)
        self.verify_render_manifest = bool(verify_render_manifest)
        self.render_manifest_name = str(render_manifest_name)
        self.window_tolerance_ms = float(window_tolerance_ms)
        self.label_period_s = None if label_period_s is None else float(label_period_s)
        self.min_track_duration_ms = None if min_track_duration_ms is None else float(min_track_duration_ms)
        self.max_tracks = max_tracks
        self.max_samples = max_samples
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.filter_missing_representations = bool(filter_missing_representations)
        self.require_representations = bool(require_representations)
        self.sample_decorrelation = dict(sample_decorrelation or {})
        self.spatial_cutout = dict(spatial_cutout or {})
        self._folder_manifests: Dict[str, Optional[dict]] = {}
        self._folder_manifest_entries: Dict[str, Optional[Dict[str, dict]]] = {}
        self._folder_rgb_indices: Dict[Tuple[str, str], List[Tuple[float, Path]]] = {}
        self._folder_event_frame_indices: Dict[str, List[Tuple[float, Path]]] = {}

        if self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise ValueError("frame_size must contain positive width and height.")
        if self.history_ms <= 0 or self.forecast_ms <= 0:
            raise ValueError("history_ms and forecast_ms must be > 0.")
        if self.min_track_duration_ms is not None and self.min_track_duration_ms <= 0:
            raise ValueError("min_track_duration_ms must be > 0 when set.")
        missing_sizes = [rep for rep in self.representations if rep not in self.image_sizes]
        if missing_sizes:
            raise ValueError(f"Missing image sizes for representations: {missing_sizes}")
        missing_source_sizes = [rep for rep in self.representations if rep not in self.source_image_sizes]
        if missing_source_sizes:
            raise ValueError(f"Missing source image sizes for representations: {missing_source_sizes}")
        if self.image_window_mode not in {"trailing", "center", "leading"}:
            raise ValueError(f"Unknown image_window_mode: {self.image_window_mode}")

        self.frames_by_folder = self._discover_frames()
        self.allowed_tracks = self._select_track_subset()
        cached = self._load_cache()
        if cached is not None:
            self.samples = cached
            self._filter_cached_missing_representations()
        else:
            self.samples = self._build_samples()
            self._apply_sample_limit()
            self._apply_sample_decorrelation()
            self._save_cache()

    def _labels_dir(self, folder: str) -> Path:
        return self.labels_root / folder / self.labels_subdir if self.folders is not None else self.labels_root

    def _images_dir(self, folder: str) -> Path:
        return self.images_root / folder if self.folders is not None else self.images_root

    def _tracks_path(self, folder: str) -> Path:
        return self.labels_root / folder / self.tracks_file if self.folders is not None else self.labels_root / self.tracks_file

    def _dataset_rgb_dir(self, folder: str, rep: str) -> Path:
        dirname = _DATASET_RGB_REPS[rep.lower()]
        return self.labels_root / folder / dirname if self.folders is not None else self.labels_root / dirname

    def _dataset_event_frames_dir(self, folder: str) -> Path:
        return self.labels_root / folder / "Event" / "Frames" if self.folders is not None else self.labels_root / "Event" / "Frames"

    def _manifest_path(self, folder: str) -> Path:
        return self._images_dir(folder) / self.render_manifest_name

    def _parse_rgb_time(self, path: Path) -> Optional[float]:
        match = _RGB_TIME_RE.search(path.stem)
        if not match:
            return None
        hh, mm, ss, frac = match.groups()
        try:
            micros = int(frac.ljust(6, "0")[:6])
            return int(hh) * 3600.0 + int(mm) * 60.0 + int(ss) + micros / 1_000_000.0
        except ValueError:
            return None

    def _build_rgb_index(self, folder: str, rep: str) -> List[Tuple[float, Path]]:
        key = (folder, rep.lower())
        if key in self._folder_rgb_indices:
            return self._folder_rgb_indices[key]
        rgb_dir = self._dataset_rgb_dir(folder, rep)
        files = []
        if rgb_dir.exists():
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
                if t is not None:
                    out.append(((t - base) * 1_000_000.0 * self.label_time_unit, path))
        else:
            out = [(float(idx), path) for idx, path in enumerate(files)]
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

    def _build_event_frame_index(self, folder: str) -> List[Tuple[float, Path]]:
        if folder in self._folder_event_frame_indices:
            return self._folder_event_frame_indices[folder]
        frames_dir = self._dataset_event_frames_dir(folder)
        files = []
        if frames_dir.exists():
            files = [
                path
                for pattern in ("*.png", "*.jpg", "*.jpeg")
                for path in sorted(frames_dir.glob(pattern))
                if not path.name.startswith(".") and not path.name.startswith("._")
            ]
        out = []
        for path in files:
            time_raw = _parse_frame_time_raw(path.stem)
            if time_raw is not None:
                out.append((float(time_raw) * self.label_time_unit, path))
        out.sort(key=lambda item: item[0])
        self._folder_event_frame_indices[folder] = out
        return out

    def _find_event_frame(self, folder: str, stem: str, label_time_s: float) -> Optional[Path]:
        frames_dir = self._dataset_event_frames_dir(folder)
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = frames_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        frame_index = self._build_event_frame_index(folder)
        if not frame_index:
            return None
        times = [t for t, _ in frame_index]
        idx = int(np.searchsorted(times, label_time_s, side="left"))
        if idx <= 0:
            return frame_index[0][1]
        if idx >= len(frame_index):
            return frame_index[-1][1]
        before_t, before_path = frame_index[idx - 1]
        after_t, after_path = frame_index[idx]
        return before_path if abs(label_time_s - before_t) <= abs(after_t - label_time_s) else after_path

    def _resolve_input_path(self, folder: str, stem: str, rep: str, label_time_s: float) -> Optional[Path]:
        rendered_path = self._images_dir(folder) / f"{stem}_{rep}.png"
        if rendered_path.exists():
            return rendered_path
        if _is_dataset_rgb_rep(rep):
            return self._find_rgb_frame(folder, rep, label_time_s)
        if _is_dataset_event_frame_rep(rep):
            return self._find_event_frame(folder, stem, label_time_s)
        return None

    def _has_all_reps(self, folder: str, stem: str, label_time_s: float) -> bool:
        return all(self._resolve_input_path(folder, stem, rep, label_time_s) is not None for rep in self.representations)

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
        if all(_is_dataset_native_rep(rep) for rep in self.representations):
            return
        manifest = self._load_render_manifest(folder)
        if manifest is None:
            raise FileNotFoundError(f"Missing render manifest for folder '{folder or '.'}'.")
        params = manifest.get("render_params") or {}
        if params.get("window_mode") != self.image_window_mode:
            raise ValueError(
                f"Render manifest mode mismatch for '{folder}/{stem}': expected "
                f"{self.image_window_mode}, found {params.get('window_mode')}."
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
            raise ValueError(f"Render manifest missing anchor stem '{folder}/{stem}'.")
        actual_s = float(entry.get("window_duration_render_units", -1.0)) * self.label_time_unit
        expected_s = self.image_window_ms / 1000.0
        tol_s = self.window_tolerance_ms / 1000.0
        if abs(actual_s - expected_s) > tol_s:
            raise ValueError(
                f"Render manifest duration mismatch for '{folder}/{stem}': expected "
                f"{expected_s:.6f}s, found {actual_s:.6f}s."
            )

    def _discover_frames(self) -> Dict[str, List[FrameItem]]:
        out: Dict[str, List[FrameItem]] = {}
        folders = [""] if self.folders is None else list(self.folders)
        for folder in folders:
            labels_dir = self._labels_dir(folder)
            if not labels_dir.exists():
                continue
            items: List[FrameItem] = []
            for txt in sorted(labels_dir.glob("*.txt")):
                time_raw = _parse_frame_time_raw(txt.stem)
                if time_raw is None:
                    continue
                time_s = float(time_raw) * self.label_time_unit
                if (
                    not self.require_representations
                    or self._has_all_reps(folder, txt.stem, time_s)
                    or not self.filter_missing_representations
                ):
                    items.append(FrameItem(stem=txt.stem, time_s=time_s))
            items.sort(key=lambda x: x.time_s)
            out[folder] = items
        return out

    def _select_track_subset(self) -> Optional[set[tuple[str, int]]]:
        if self.max_tracks is None:
            return None
        keys: List[tuple[str, int]] = []
        for folder in self.frames_by_folder:
            for track_id in _read_tracks(self._tracks_path(folder)):
                keys.append((folder, int(track_id)))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(keys)
        return set(keys[: self.max_tracks])

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

    def _eligibility_window(self, *, history_steps: int, future_steps: int, label_period_s: float) -> int:
        prediction_window = int(history_steps) + int(future_steps) + 1
        if self.min_track_duration_ms is None:
            return prediction_window
        min_duration_steps = max(1, int(round((self.min_track_duration_ms / 1000.0) / label_period_s)))
        return max(prediction_window, min_duration_steps + 1)

    def _align_track_times(self, track_times: np.ndarray, label_times: np.ndarray) -> np.ndarray:
        times = track_times.copy()
        if self.time_align == "start":
            return times + (label_times[0] - times[0])
        if self.time_align == "auto":
            shift = label_times[0] - times[0]
            no_shift = int(np.sum((label_times >= times[0]) & (label_times <= times[-1])))
            shifted = int(np.sum((label_times >= times[0] + shift) & (label_times <= times[-1] + shift)))
            return times + shift if shifted > no_shift else times
        if self.time_align == "none":
            return times
        raise ValueError(f"Unknown time_align: {self.time_align}")

    def _build_samples(self) -> list[dict]:
        samples: list[dict] = []
        skipped_missing = 0
        skipped_short_tracks = 0
        skipped_short_tracklets = 0
        missing_by_rep: Counter = Counter()
        frame_w, frame_h = self.frame_size
        for folder, frames in self.frames_by_folder.items():
            if not frames:
                continue
            tracks = _read_tracks(self._tracks_path(folder))
            if not tracks:
                continue
            label_period_s = self._resolve_label_period_s(frames)
            history_steps, future_steps = self._window_spec(label_period_s)
            window = history_steps + future_steps + 1
            eligibility_window = self._eligibility_window(
                history_steps=history_steps,
                future_steps=future_steps,
                label_period_s=label_period_s,
            )
            label_times = np.asarray([f.time_s for f in frames], dtype=np.float64)
            label_stems = [f.stem for f in frames]
            for track_id, rows in tracks.items():
                if self.allowed_tracks is not None and (folder, int(track_id)) not in self.allowed_tracks:
                    continue
                times = np.asarray([t for t, *_ in rows], dtype=np.float64) * self.track_time_unit
                if times.size == 0:
                    continue
                times = self._align_track_times(times, label_times)
                mask = (label_times >= times[0]) & (label_times <= times[-1])
                if not np.any(mask):
                    continue
                idxs = np.nonzero(mask)[0]
                query_times = label_times[idxs]
                if self.min_track_duration_ms is not None:
                    usable_duration_s = float(query_times[-1] - query_times[0]) if query_times.size >= 2 else 0.0
                    if usable_duration_s < self.min_track_duration_ms / 1000.0:
                        skipped_short_tracks += 1
                        continue
                xs = np.interp(query_times, times, np.asarray([x for _, x, _, _, _ in rows], dtype=np.float64))
                ys = np.interp(query_times, times, np.asarray([y for _, _, y, _, _ in rows], dtype=np.float64))
                ws = np.interp(query_times, times, np.asarray([w for _, _, _, w, _ in rows], dtype=np.float64))
                hs = np.interp(query_times, times, np.asarray([h for _, _, _, _, h in rows], dtype=np.float64))
                boxes = np.stack(
                    [(xs + ws / 2.0) / frame_w, (ys + hs / 2.0) / frame_h, ws / frame_w, hs / frame_h],
                    axis=-1,
                )
                stems = [label_stems[i] for i in idxs]
                prediction_candidates = max(0, len(stems) - window + 1)
                eligible_candidates = max(0, len(stems) - eligibility_window + 1)
                skipped_short_tracklets += max(0, prediction_candidates - eligible_candidates)
                for start in range(0, eligible_candidates):
                    end = start + window
                    anchor_idx = history_steps
                    anchor_stem = stems[start + anchor_idx]
                    anchor_time = float(query_times[start + anchor_idx])
                    input_paths = {}
                    if self.require_representations:
                        resolved_paths = {
                            rep: self._resolve_input_path(folder, anchor_stem, rep, anchor_time)
                            for rep in self.representations
                        }
                        missing = [rep for rep, path in resolved_paths.items() if path is None]
                        if missing:
                            if self.filter_missing_representations:
                                skipped_missing += 1
                                missing_by_rep.update(missing)
                                continue
                            raise FileNotFoundError(
                                f"Missing representation file(s) for '{folder}/{anchor_stem}': {missing}"
                            )
                        self._validate_manifest_entry(folder, anchor_stem)
                        input_paths = {rep: str(path) for rep, path in resolved_paths.items() if path is not None}
                    samples.append(
                        {
                            "folder": folder,
                            "track_id": int(track_id),
                            "anchor_stem": anchor_stem,
                            "anchor_time_s": anchor_time,
                            "input_paths": input_paths,
                            "past_boxes": boxes[start : start + anchor_idx + 1].astype(np.float32),
                            "future_boxes": boxes[start + anchor_idx + 1 : end].astype(np.float32),
                            "past_times_s": query_times[start : start + anchor_idx + 1].astype(np.float32),
                            "future_times_s": query_times[start + anchor_idx + 1 : end].astype(np.float32),
                        }
                    )
        if skipped_missing:
            rep_summary = ", ".join(f"{rep}={count}" for rep, count in sorted(missing_by_rep.items()))
            print(f"Warning: skipped {skipped_missing} Kalman ML samples with missing representations: {rep_summary}")
        if skipped_short_tracks:
            print(
                f"Skipped {skipped_short_tracks} Kalman ML tracks shorter than "
                f"data.min_track_duration_ms={self.min_track_duration_ms:.6g}."
            )
        if skipped_short_tracklets:
            print(
                f"Skipped {skipped_short_tracklets} Kalman ML tracklets shorter than "
                f"data.min_track_duration_ms={self.min_track_duration_ms:.6g}."
            )
        return samples

    def _filter_cached_missing_representations(self) -> None:
        if not self.samples:
            return
        kept = []
        for sample in self.samples:
            if all(Path(path).exists() for path in sample.get("input_paths", {}).values()):
                kept.append(sample)
            elif not self.filter_missing_representations:
                raise FileNotFoundError(f"Missing cached representation for {sample.get('anchor_stem')}")
        self.samples = kept

    def _sample_decorrelation_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_mode = str(self.sample_decorrelation.get("feature_mode", "motion_priors"))
        frame_w, frame_h = self.frame_size
        rows_x = []
        rows_y = []
        rows_vel = []
        for sample in self.samples:
            boxes = np.concatenate(
                [
                    np.asarray(sample["past_boxes"], dtype=np.float64),
                    np.asarray(sample["future_boxes"], dtype=np.float64),
                ],
                axis=0,
            )
            times = np.concatenate(
                [
                    np.asarray(sample["past_times_s"], dtype=np.float64),
                    np.asarray(sample["future_times_s"], dtype=np.float64),
                ],
                axis=0,
            )
            centers_px = np.stack([boxes[:, 0] * frame_w, boxes[:, 1] * frame_h], axis=1)
            pos, vel, accel = _fit_center_constant_acceleration(
                times,
                centers_px,
                anchor_time_s=float(sample["past_times_s"][-1]),
            )
            rows_x.append(
                _sample_motion_features(
                    cx_norm=float(pos[0] / frame_w),
                    cy_norm=float(pos[1] / frame_h),
                    vx_px_s=float(vel[0]),
                    vy_px_s=float(vel[1]),
                    feature_mode=feature_mode,
                )
            )
            rows_y.append([float(accel[0]), float(accel[1])])
            rows_vel.append([float(vel[0]), float(vel[1])])
        return (
            np.asarray(rows_x, dtype=np.float64),
            np.asarray(rows_y, dtype=np.float64),
            np.asarray(rows_vel, dtype=np.float64),
        )

    @staticmethod
    def _decorrelation_motion_summary(accel: np.ndarray, velocity: np.ndarray, selected: np.ndarray) -> dict[str, float]:
        accel = np.asarray(accel, dtype=np.float64)[selected]
        velocity = np.asarray(velocity, dtype=np.float64)[selected]
        if accel.size == 0:
            return {
                "abs_accel_mean": float("nan"),
                "abs_accel_median": float("nan"),
                "abs_accel_p90": float("nan"),
                "turning_accel_mean": float("nan"),
                "turning_accel_median": float("nan"),
                "turning_accel_p90": float("nan"),
                "turning_fraction_mean": float("nan"),
            }
        abs_accel = np.linalg.norm(accel, axis=1)
        speed = np.linalg.norm(velocity, axis=1)
        cross = velocity[:, 0] * accel[:, 1] - velocity[:, 1] * accel[:, 0]
        turning_accel = np.divide(
            np.abs(cross),
            speed,
            out=np.zeros_like(abs_accel),
            where=speed > 1.0e-9,
        )
        turning_fraction = np.divide(
            turning_accel,
            abs_accel,
            out=np.zeros_like(abs_accel),
            where=abs_accel > 1.0e-9,
        )
        return {
            "abs_accel_mean": float(np.mean(abs_accel)),
            "abs_accel_median": float(np.median(abs_accel)),
            "abs_accel_p90": float(np.percentile(abs_accel, 90.0)),
            "turning_accel_mean": float(np.mean(turning_accel)),
            "turning_accel_median": float(np.median(turning_accel)),
            "turning_accel_p90": float(np.percentile(turning_accel, 90.0)),
            "turning_fraction_mean": float(np.mean(turning_fraction)),
        }

    @staticmethod
    def _score_decorrelation_stats(
        n: float,
        sum_x: np.ndarray,
        sum_y: np.ndarray,
        sum_xx: np.ndarray,
        sum_xy: np.ndarray,
        sum_yy: np.ndarray,
        *,
        ridge_lambda: float,
        corr_weight: float,
        r2_weight: float,
        mean_accel_weight: float,
    ) -> dict[str, float]:
        if n < 4:
            return {
                "score": float("inf"),
                "samples": float(n),
                "mean_abs_corr": float("inf"),
                "mean_r2": float("inf"),
                "mean_accel_norm": float("inf"),
            }
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
            "samples": float(n),
            "mean_abs_corr": mean_abs_corr,
            "mean_r2": mean_r2,
            "mean_accel_norm": mean_accel_norm,
        }

    def _apply_sample_decorrelation(self) -> None:
        cfg = self.sample_decorrelation
        if not bool(cfg.get("enabled", False)):
            return
        if not self.samples:
            return
        keep_fraction = cfg.get("keep_fraction", 1.0)
        target_samples_raw = cfg.get("target_samples")
        if target_samples_raw is None:
            if not 0.0 < float(keep_fraction) <= 1.0:
                raise ValueError("data.decorrelation.keep_fraction must be in (0, 1].")
            target_samples = max(1, int(round(len(self.samples) * float(keep_fraction))))
        else:
            target_samples = int(target_samples_raw)
        min_samples = int(cfg.get("min_samples", 4))
        target_samples = max(min_samples, min(target_samples, len(self.samples)))
        if target_samples >= len(self.samples):
            return

        x, y, velocity = self._sample_decorrelation_arrays()
        n = float(x.shape[0])
        sum_x = x.sum(axis=0)
        sum_y = y.sum(axis=0)
        sum_xx = x.T @ x
        sum_xy = x.T @ y
        sum_yy = y.T @ y
        sample_xx = np.einsum("ni,nj->nij", x, x)
        sample_xy = np.einsum("ni,nj->nij", x, y)
        sample_yy = np.einsum("ni,nj->nij", y, y)

        seed_raw = cfg.get("seed")
        seed = self.seed if seed_raw is None else int(seed_raw)
        greedy_candidates = int(cfg.get("greedy_candidates", 64))
        ridge_lambda = float(cfg.get("ridge_lambda", 1.0e-3))
        corr_weight = float(cfg.get("corr_weight", 1.0))
        r2_weight = float(cfg.get("r2_weight", 1.0))
        mean_accel_weight = float(cfg.get("mean_accel_weight", 0.0))
        show_progress = bool(cfg.get("progress", True))
        rng = np.random.default_rng(seed)
        before_score = self._score_decorrelation_stats(
            n,
            sum_x,
            sum_y,
            sum_xx,
            sum_xy,
            sum_yy,
            ridge_lambda=ridge_lambda,
            corr_weight=corr_weight,
            r2_weight=r2_weight,
            mean_accel_weight=mean_accel_weight,
        )
        current_score = before_score
        selected = np.ones(len(self.samples), dtype=bool)
        before_motion = self._decorrelation_motion_summary(y, velocity, selected)
        removals = range(len(self.samples) - target_samples)
        for _ in _progress_iter(removals, desc="Decorrelating Kalman ML samples", enabled=show_progress):
            candidates = np.nonzero(selected)[0]
            if 0 < greedy_candidates < candidates.size:
                candidates = rng.choice(candidates, size=greedy_candidates, replace=False)
            best_idx = None
            best_score = None
            for idx in candidates:
                score = self._score_decorrelation_stats(
                    n - 1.0,
                    sum_x - x[idx],
                    sum_y - y[idx],
                    sum_xx - sample_xx[idx],
                    sum_xy - sample_xy[idx],
                    sum_yy - sample_yy[idx],
                    ridge_lambda=ridge_lambda,
                    corr_weight=corr_weight,
                    r2_weight=r2_weight,
                    mean_accel_weight=mean_accel_weight,
                )
                if best_score is None or score["score"] < best_score["score"]:
                    best_idx = int(idx)
                    best_score = score
            if best_idx is None:
                break
            selected[best_idx] = False
            n -= 1.0
            sum_x -= x[best_idx]
            sum_y -= y[best_idx]
            sum_xx -= sample_xx[best_idx]
            sum_xy -= sample_xy[best_idx]
            sum_yy -= sample_yy[best_idx]
            current_score = best_score
        before = len(self.samples)
        after_motion = self._decorrelation_motion_summary(y, velocity, selected)
        self.samples = [sample for sample, keep in zip(self.samples, selected) if bool(keep)]
        split_name = cfg.get("_split_name")
        split_text = f" for split '{split_name}'" if split_name else ""
        print(
            f"Applied Kalman ML sample decorrelation{split_text}: "
            f"kept {len(self.samples)}/{before} samples "
            f"(feature_mode={cfg.get('feature_mode', 'motion_priors')}, greedy_candidates={greedy_candidates}, "
            f"score={before_score['score']:.6g}->{current_score['score']:.6g}, "
            f"mean_accel_norm={before_score['mean_accel_norm']:.3g}->{current_score['mean_accel_norm']:.3g}, "
            f"|a| mean/median/p90={before_motion['abs_accel_mean']:.3g}/"
            f"{before_motion['abs_accel_median']:.3g}/{before_motion['abs_accel_p90']:.3g}->"
            f"{after_motion['abs_accel_mean']:.3g}/{after_motion['abs_accel_median']:.3g}/"
            f"{after_motion['abs_accel_p90']:.3g}, "
            f"turning |a_perp| mean/median/p90={before_motion['turning_accel_mean']:.3g}/"
            f"{before_motion['turning_accel_median']:.3g}/{before_motion['turning_accel_p90']:.3g}->"
            f"{after_motion['turning_accel_mean']:.3g}/{after_motion['turning_accel_median']:.3g}/"
            f"{after_motion['turning_accel_p90']:.3g}, "
            f"turning_fraction_mean={before_motion['turning_fraction_mean']:.3g}->"
            f"{after_motion['turning_fraction_mean']:.3g})."
        )

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
            "frame_size": self.frame_size,
            "representations": self.representations,
            "image_sizes": self.image_sizes,
            "source_image_sizes": self.source_image_sizes,
            "history_ms": self.history_ms,
            "forecast_ms": self.forecast_ms,
            "folders": self.folders,
            "labels_subdir": self.labels_subdir,
            "tracks_file": self.tracks_file,
            "label_time_unit": self.label_time_unit,
            "track_time_unit": self.track_time_unit,
            "time_align": self.time_align,
            "image_window_ms": self.image_window_ms,
            "image_window_mode": self.image_window_mode,
            "verify_render_manifest": self.verify_render_manifest,
            "render_manifest_name": self.render_manifest_name,
            "window_tolerance_ms": self.window_tolerance_ms,
            "label_period_s": self.label_period_s,
            "min_track_duration_ms": self.min_track_duration_ms,
            "max_tracks": self.max_tracks,
            "max_samples": self.max_samples,
            "require_representations": self.require_representations,
            "sample_decorrelation": self.sample_decorrelation,
            "seed": self.seed,
            "cache_version": self.CACHE_VERSION,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"kalman_ml_samples_{self._cache_key()}.pkl"

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

    def __getitem__(self, idx: int) -> KalmanForecastSample:
        sample = self.samples[idx]
        inputs = {
            rep: _load_image(
                Path(path),
                self.image_sizes[rep],
                source_size=self.source_image_sizes[rep],
                rep=rep,
                frame_size=self.frame_size,
                anchor_box=np.asarray(sample["past_boxes"], dtype=np.float32)[-1],
                spatial_cutout=self.spatial_cutout,
            )
            for rep, path in sample["input_paths"].items()
        }
        folder = sample["folder"]
        frame_key = f"{folder}/{sample['anchor_stem']}" if folder else sample["anchor_stem"]
        return KalmanForecastSample(
            inputs=inputs,
            past_boxes=torch.tensor(sample["past_boxes"], dtype=torch.float32),
            future_boxes=torch.tensor(sample["future_boxes"], dtype=torch.float32),
            past_times_s=torch.tensor(sample["past_times_s"], dtype=torch.float32),
            future_times_s=torch.tensor(sample["future_times_s"], dtype=torch.float32),
            frame_key=frame_key,
            frame_time_s=float(sample["anchor_time_s"]),
            track_id=int(sample["track_id"]),
            input_paths=dict(sample["input_paths"]),
        )
