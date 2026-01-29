from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class ForecastSample:
    inputs: Dict[str, torch.Tensor]
    past_boxes: torch.Tensor
    future_boxes: torch.Tensor
    frame_keys: List[str]


def _parse_frame_time(name: str) -> Optional[int]:
    # Example: Video_0_frame_100032333_xt.png -> 100032333
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
    # CHW
    return torch.from_numpy(arr).permute(2, 0, 1)


def _read_yolo_boxes(path: Path) -> List[Tuple[float, float, float, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    boxes: List[Tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = parts[:5]
        try:
            boxes.append((float(cx), float(cy), float(bw), float(bh)))
        except ValueError:
            continue
    return boxes


def _select_box(
    boxes: List[Tuple[float, float, float, float]], mode: str
) -> Optional[Tuple[float, float, float, float]]:
    if not boxes:
        return None
    if mode == "largest":
        return max(boxes, key=lambda b: b[2] * b[3])
    if mode == "first":
        return boxes[0]
    raise ValueError(f"Unknown select_box: {mode}")


FrameKey = Tuple[str, str]


class ForecastDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_root: Path,
        labels_root: Path,
        representations: List[str],
        past_steps: int,
        future_steps: int,
        stride: int,
        image_size: Tuple[int, int],
        select_box: str = "largest",
        drop_empty: bool = True,
        folders: Optional[List[str]] = None,
        labels_subdir: str = "Event_YOLO",
    ) -> None:
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.representations = representations
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.stride = stride
        self.image_size = image_size
        self.select_box = select_box
        self.drop_empty = drop_empty
        self.folders = folders
        self.labels_subdir = labels_subdir

        self.frames_by_folder = self._discover_frames()
        self.samples = self._build_samples()

    def _labels_dir(self, folder: str) -> Path:
        if self.folders is None:
            return self.labels_root
        return self.labels_root / folder / self.labels_subdir

    def _images_dir(self, folder: str) -> Path:
        if self.folders is None:
            return self.images_root
        return self.images_root / folder

    def _discover_frames(self) -> Dict[str, List[str]]:
        # Use label files as the source of frame keys per folder
        frames: Dict[str, List[str]] = {}
        if self.folders is None:
            keys = [p.stem for p in self.labels_root.glob("*.txt")]
            keys.sort(key=lambda k: (_parse_frame_time(k) is None, _parse_frame_time(k) or 0, k))
            frames[""] = keys
            return frames
        for folder in self.folders:
            labels_dir = self._labels_dir(folder)
            if not labels_dir.exists():
                continue
            keys = [p.stem for p in labels_dir.glob("*.txt")]
            keys.sort(key=lambda k: (_parse_frame_time(k) is None, _parse_frame_time(k) or 0, k))
            frames[folder] = keys
        return frames

    def _has_all_reps(self, key: FrameKey) -> bool:
        folder, stem = key
        img_dir = self._images_dir(folder)
        for rep in self.representations:
            img = img_dir / f"{stem}_{rep}.png"
            if not img.exists():
                return False
        return True

    def _build_samples(self) -> List[List[FrameKey]]:
        samples: List[List[FrameKey]] = []
        total = self.past_steps + self.future_steps
        for folder, keys in self.frames_by_folder.items():
            for idx in range(0, len(keys) - total * self.stride + 1):
                window = [
                    (folder, keys[idx + i * self.stride]) for i in range(total)
                ]
                if not all(self._has_all_reps(k) for k in window):
                    continue
                if self.drop_empty:
                    # Require at least one box in all past and future frames
                    ok = True
                    for f, stem in window:
                        labels_dir = self._labels_dir(f)
                        boxes = _read_yolo_boxes(labels_dir / f"{stem}.txt")
                        if not boxes:
                            ok = False
                            break
                    if not ok:
                        continue
                samples.append(window)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ForecastSample:
        window = self.samples[idx]
        past_keys = window[: self.past_steps]
        future_keys = window[self.past_steps :]

        inputs: Dict[str, List[torch.Tensor]] = {r: [] for r in self.representations}
        past_boxes = []
        future_boxes = []

        for key in past_keys:
            folder, stem = key
            img_dir = self._images_dir(folder)
            labels_dir = self._labels_dir(folder)
            for rep in self.representations:
                img_path = img_dir / f"{stem}_{rep}.png"
                inputs[rep].append(_load_image(img_path, self.image_size))
            boxes = _read_yolo_boxes(labels_dir / f"{stem}.txt")
            box = _select_box(boxes, self.select_box)
            if box is None:
                box = (0.0, 0.0, 0.0, 0.0)
            past_boxes.append(box)

        for key in future_keys:
            folder, stem = key
            labels_dir = self._labels_dir(folder)
            boxes = _read_yolo_boxes(labels_dir / f"{stem}.txt")
            box = _select_box(boxes, self.select_box)
            if box is None:
                box = (0.0, 0.0, 0.0, 0.0)
            future_boxes.append(box)

        inputs_t = {r: torch.stack(seq, dim=0) for r, seq in inputs.items()}
        return ForecastSample(
            inputs=inputs_t,
            past_boxes=torch.tensor(past_boxes, dtype=torch.float32),
            future_boxes=torch.tensor(future_boxes, dtype=torch.float32),
            frame_keys=[f"{f}/{stem}" if f else stem for f, stem in window],
        )


def split_dataset(
    dataset: ForecastDataset, train_split: float, seed: int
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    n = len(dataset)
    n_train = int(n * train_split)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n - n_train], generator=generator)
