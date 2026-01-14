from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import Region, build_patch


@dataclass
class EventSample:
    patches: torch.Tensor
    metadata: torch.Tensor
    event_counts: torch.Tensor
    plane_ids: torch.Tensor


class SyntheticEventDataset(Dataset):
    """Generates random events for fast iteration."""

    def __init__(
        self,
        num_samples: int,
        max_events: int,
        image_width: int,
        image_height: int,
        time_bins: int,
        region_scales: List[int],
        region_scales_x: List[int],
        region_scales_y: List[int],
        region_time_scales: List[float],
        region_sampling: str,
        grid_x: int,
        grid_y: int,
        grid_t: int,
        grid_plane_mode: str,
        plane_types: List[str],
        num_regions: int,
        patch_size: int,
        rng_seed: int = 0,
        fixed_region_seed: int = -1,
        patch_divider: float = 0.0,
        fixed_region_sizes: bool = False,
        fixed_region_positions_global: bool = False,
        fixed_single_region: bool = False,
        cache_token_max_samples: int = 0,
        cache_token_variants_per_sample: int = 0,
        cache_token_dir: str = "outputs/token_cache",
        cache_token_variant_mode: str = "random",
        cache_token_clear_on_start: bool = False,
        return_label: bool = False,
    ) -> None:
        self.num_samples = num_samples
        self.max_events = max_events
        self.image_width = image_width
        self.image_height = image_height
        self.time_bins = time_bins
        self.region_scales = region_scales
        self.region_scales_x = region_scales_x
        self.region_scales_y = region_scales_y
        self.region_time_scales = region_time_scales
        self.region_sampling = region_sampling
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_t = grid_t
        self.grid_plane_mode = grid_plane_mode
        self.plane_types = plane_types
        self.num_regions = num_regions
        self.patch_size = patch_size
        self.rng = np.random.default_rng(rng_seed)
        self.fixed_region_seed = fixed_region_seed
        self.patch_divider = patch_divider
        self.fixed_region_sizes = fixed_region_sizes
        self.fixed_region_positions_global = fixed_region_positions_global
        self.fixed_single_region = fixed_single_region
        self.cache_token_max_samples = cache_token_max_samples
        self._token_cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.cache_token_variants_per_sample = cache_token_variants_per_sample
        self.cache_token_dir = Path(cache_token_dir)
        self.cache_token_variant_mode = cache_token_variant_mode
        self.cache_token_clear_on_start = cache_token_clear_on_start
        if self.cache_token_clear_on_start and self.cache_token_dir.exists():
            for path in self.cache_token_dir.glob("synthetic_*.npz"):
                path.unlink()
        self._variant_cursor: Dict[int, int] = {}

    def __len__(self) -> int:
        return self.num_samples

    def _sample_events(self) -> np.ndarray:
        num_events = self.rng.integers(self.max_events // 2, self.max_events + 1)
        x = self.rng.integers(0, self.image_width, size=num_events)
        y = self.rng.integers(0, self.image_height, size=num_events)
        t = self.rng.random(num_events)
        p = self.rng.integers(0, 2, size=num_events)
        return np.stack([x, y, t, p], axis=1).astype(np.float32)

    def _sample_region(self, rng: np.random.Generator) -> Region:
        if self.fixed_region_sizes:
            dx = int(self.region_scales_x[0] if self.region_scales_x else self.region_scales[0])
            dy = int(self.region_scales_y[0] if self.region_scales_y else self.region_scales[0])
            dt = float(self.region_time_scales[0])
        else:
            if self.region_scales_x:
                dx = int(rng.choice(self.region_scales_x))
            else:
                dx = int(rng.choice(self.region_scales))
            if self.region_scales_y:
                dy = int(rng.choice(self.region_scales_y))
            else:
                dy = int(rng.choice(self.region_scales))
            dt = float(rng.choice(self.region_time_scales))
        dx = min(dx, self.image_width)
        dy = min(dy, self.image_height)
        x = int(rng.integers(0, max(1, self.image_width - dx)))
        y = int(rng.integers(0, max(1, self.image_height - dy)))
        t = float(rng.random() * max(1e-6, 1.0 - dt))
        plane = str(rng.choice(self.plane_types))
        return Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane)

    def _grid_regions(self) -> List[Region]:
        if self.grid_x <= 0 or self.grid_y <= 0 or self.grid_t <= 0:
            raise ValueError("grid_x, grid_y, grid_t must be > 0 for grid sampling")
        xs = np.linspace(0, self.image_width, self.grid_x + 1, dtype=int)
        ys = np.linspace(0, self.image_height, self.grid_y + 1, dtype=int)
        ts = np.linspace(0.0, 1.0, self.grid_t + 1, dtype=np.float32)
        regions: List[Region] = []
        for ti in range(self.grid_t):
            t = float(ts[ti])
            dt = float(ts[ti + 1] - ts[ti])
            for yi in range(self.grid_y):
                y = int(ys[yi])
                dy = int(max(1, ys[yi + 1] - ys[yi]))
                if y + dy > self.image_height:
                    dy = self.image_height - y
                for xi in range(self.grid_x):
                    x = int(xs[xi])
                    dx = int(max(1, xs[xi + 1] - xs[xi]))
                    if x + dx > self.image_width:
                        dx = self.image_width - x
                    if self.grid_plane_mode == "all":
                        for plane in self.plane_types:
                            regions.append(
                                Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane)
                            )
                    else:
                        plane = self.plane_types[len(regions) % len(self.plane_types)]
                        regions.append(Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane))
        return regions

    def _select_regions(
        self, rng: np.random.Generator, regions: List[Region]
    ) -> List[Region]:
        if self.num_regions <= 0 or self.num_regions == len(regions):
            return regions
        if self.num_regions < len(regions):
            idx = rng.choice(len(regions), size=self.num_regions, replace=False)
            return [regions[i] for i in idx]
        repeats = (self.num_regions + len(regions) - 1) // len(regions)
        tiled = (regions * repeats)[: self.num_regions]
        return tiled

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_token_variants_per_sample > 0:
            return self._get_token_variant(idx)

        if self.cache_token_max_samples > 0 and idx in self._token_cache:
            cached = self._token_cache.pop(idx)
            self._token_cache[idx] = cached
            return cached

        events = self._sample_events()
        seq_len_sec = 1.0
        if self.fixed_region_seed >= 0:
            seed = self.fixed_region_seed
            if not self.fixed_region_positions_global:
                seed += idx
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng
        patches, metadata, counts, plane_ids = self._build_sample(events, seq_len_sec, rng)
        sample = {
            "patches": patches,
            "metadata": metadata,
            "event_counts": counts,
            "plane_ids": plane_ids,
        }
        if self.return_label:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.cache_token_max_samples > 0:
            self._token_cache[idx] = sample
            if len(self._token_cache) > self.cache_token_max_samples:
                self._token_cache.popitem(last=False)
        return sample

    def _select_variant(self, idx: int) -> int:
        if self.cache_token_variant_mode == "round_robin":
            cursor = self._variant_cursor.get(idx, 0)
            variant = cursor % self.cache_token_variants_per_sample
            self._variant_cursor[idx] = cursor + 1
            return variant
        return int(self.rng.integers(0, self.cache_token_variants_per_sample))

    def _get_token_variant(self, idx: int) -> Dict[str, torch.Tensor]:
        variant = self._select_variant(idx)
        cache_path = self.cache_token_dir / f"synthetic_{idx:06d}_v{variant}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            return {
                "patches": torch.from_numpy(cached["patches"]),
                "metadata": torch.from_numpy(cached["metadata"]),
                "event_counts": torch.from_numpy(cached["event_counts"]),
                "plane_ids": torch.from_numpy(cached["plane_ids"]),
            }

        events = self._sample_events()
        seq_len_sec = 1.0
        if self.fixed_region_seed >= 0:
            seed = self.fixed_region_seed + idx * 1000 + variant
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        patches, metadata, counts, plane_ids = self._build_sample(events, seq_len_sec, rng)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            patches=patches.numpy(),
            metadata=metadata.numpy(),
            event_counts=counts.numpy(),
            plane_ids=plane_ids.numpy(),
        )
        return {
            "patches": patches,
            "metadata": metadata,
            "event_counts": counts,
            "plane_ids": plane_ids,
        }

    def _build_sample(
        self, events: np.ndarray, seq_len_sec: float, rng: np.random.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches: List[torch.Tensor] = []
        metadata: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        plane_ids: List[int] = []

        if self.fixed_single_region:
            region = self._sample_region(rng)
            regions = [region] * self.num_regions
        elif self.region_sampling == "grid":
            regions = self._select_regions(rng, self._grid_regions())
        else:
            regions = [self._sample_region(rng) for _ in range(self.num_regions)]

        for region in regions:
            patch, total_events = build_patch(
                events, region, self.patch_size, self.time_bins, patch_divider=self.patch_divider
            )
            patches.append(patch)
            counts.append(torch.tensor([total_events], dtype=torch.float32))
            plane_ids.append(self.plane_types.index(region.plane))

            meta = torch.tensor(
                [
                    region.x / self.image_width,
                    region.y / self.image_height,
                    region.dx / self.image_width,
                    region.dy / self.image_height,
                    region.t,
                    region.dt,
                    region.t * seq_len_sec,
                    region.dt * seq_len_sec,
                    seq_len_sec,
                ],
                dtype=torch.float32,
            )
            metadata.append(meta)

        return (
            torch.stack(patches, dim=0),
            torch.stack(metadata, dim=0),
            torch.stack(counts, dim=0),
            torch.tensor(plane_ids, dtype=torch.long),
        )


class THUEACTDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_width: int,
        image_height: int,
        time_unit: float,
        time_bins: int,
        region_scales: List[int],
        region_scales_x: List[int],
        region_scales_y: List[int],
        region_time_scales: List[float],
        region_sampling: str,
        grid_x: int,
        grid_y: int,
        grid_t: int,
        grid_plane_mode: str,
        plane_types: List[str],
        num_regions: int,
        patch_size: int,
        max_samples: int = 0,
        max_events: int = 0,
        subset_seed: int = 0,
        fixed_region_seed: int = -1,
        patch_divider: float = 0.0,
        fixed_region_sizes: bool = False,
        fixed_region_positions_global: bool = False,
        fixed_single_region: bool = False,
        cache_max_samples: int = 0,
        cache_token_max_samples: int = 0,
        cache_token_variants_per_sample: int = 0,
        cache_token_dir: str = "outputs/token_cache",
        cache_token_variant_mode: str = "random",
        cache_token_clear_on_start: bool = False,
        return_label: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_width = image_width
        self.image_height = image_height
        self.time_unit = time_unit
        self.time_bins = time_bins
        self.region_scales = region_scales
        self.region_scales_x = region_scales_x
        self.region_scales_y = region_scales_y
        self.region_time_scales = region_time_scales
        self.region_sampling = region_sampling
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_t = grid_t
        self.grid_plane_mode = grid_plane_mode
        self.plane_types = plane_types
        self.num_regions = num_regions
        self.patch_size = patch_size
        self.max_events = max_events
        self.fixed_region_seed = fixed_region_seed
        self.patch_divider = patch_divider
        self.fixed_region_sizes = fixed_region_sizes
        self.fixed_region_positions_global = fixed_region_positions_global
        self.fixed_single_region = fixed_single_region
        self.cache_max_samples = cache_max_samples
        self._event_cache: OrderedDict[int, Tuple[np.ndarray, float]] = OrderedDict()
        self.cache_token_max_samples = cache_token_max_samples
        self._token_cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.cache_token_variants_per_sample = cache_token_variants_per_sample
        self.cache_token_dir = Path(cache_token_dir)
        self.cache_token_variant_mode = cache_token_variant_mode
        self.cache_token_clear_on_start = cache_token_clear_on_start
        self.return_label = return_label
        if self.cache_token_clear_on_start and self.cache_token_dir.exists():
            for path in self.cache_token_dir.glob("thu_*.npz"):
                path.unlink()
        self._variant_cursor: Dict[int, int] = {}

        list_path = self.root / f"{split}.txt"
        if not list_path.exists():
            raise FileNotFoundError(f"Missing split list: {list_path}")
        entries = []
        labels = []
        for line in list_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            entries.append(parts[0])
            labels.append(int(parts[1]) if len(parts) > 1 else -1)
        self.files = []
        self.labels = []
        for idx, entry in enumerate(entries):
            name = Path(entry).name
            candidate = self.root / name
            if candidate.exists():
                self.files.append(candidate)
                self.labels.append(labels[idx])
        self.rng = np.random.default_rng(subset_seed)
        if max_samples > 0:
            if max_samples > len(self.files):
                max_samples = len(self.files)
            self.files = list(self.rng.choice(self.files, size=max_samples, replace=False))
        if not self.files:
            raise FileNotFoundError("No dataset files found for split list")
        if self.return_label and any(label < 0 for label in self.labels):
            raise ValueError("return_label requires labels in the split list")

    def __len__(self) -> int:
        return len(self.files)

    def _load_events(self, path: Path) -> Tuple[np.ndarray, float]:
        events = np.load(path).astype(np.float32)
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError(f"Unexpected event shape in {path}: {events.shape}")
        if self.max_events > 0 and events.shape[0] > self.max_events:
            idx = self.rng.choice(events.shape[0], size=self.max_events, replace=False)
            events = events[idx]
        t = events[:, 2]
        t_min = float(t.min())
        t_max = float(t.max())
        denom = max(1e-6, t_max - t_min)
        events[:, 2] = (t - t_min) / denom
        return events, denom * self.time_unit

    def _load_events_cached(self, idx: int) -> Tuple[np.ndarray, float]:
        if self.cache_max_samples > 0 and idx in self._event_cache:
            events, seq_len_sec = self._event_cache.pop(idx)
            self._event_cache[idx] = (events, seq_len_sec)
            return events, seq_len_sec

        events, seq_len_sec = self._load_events(self.files[idx])
        if self.cache_max_samples > 0:
            self._event_cache[idx] = (events, seq_len_sec)
            if len(self._event_cache) > self.cache_max_samples:
                self._event_cache.popitem(last=False)
        return events, seq_len_sec

    def _sample_region(self, rng: np.random.Generator) -> Region:
        if self.fixed_region_sizes:
            dx = int(self.region_scales_x[0] if self.region_scales_x else self.region_scales[0])
            dy = int(self.region_scales_y[0] if self.region_scales_y else self.region_scales[0])
            dt = float(self.region_time_scales[0])
        else:
            if self.region_scales_x:
                dx = int(rng.choice(self.region_scales_x))
            else:
                dx = int(rng.choice(self.region_scales))
            if self.region_scales_y:
                dy = int(rng.choice(self.region_scales_y))
            else:
                dy = int(rng.choice(self.region_scales))
            dt = float(rng.choice(self.region_time_scales))
        dx = min(dx, self.image_width)
        dy = min(dy, self.image_height)
        x = int(rng.integers(0, max(1, self.image_width - dx)))
        y = int(rng.integers(0, max(1, self.image_height - dy)))
        t = float(rng.random() * max(1e-6, 1.0 - dt))
        plane = str(rng.choice(self.plane_types))
        return Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane)

    def _grid_regions(self) -> List[Region]:
        if self.grid_x <= 0 or self.grid_y <= 0 or self.grid_t <= 0:
            raise ValueError("grid_x, grid_y, grid_t must be > 0 for grid sampling")
        xs = np.linspace(0, self.image_width, self.grid_x + 1, dtype=int)
        ys = np.linspace(0, self.image_height, self.grid_y + 1, dtype=int)
        ts = np.linspace(0.0, 1.0, self.grid_t + 1, dtype=np.float32)
        regions: List[Region] = []
        for ti in range(self.grid_t):
            t = float(ts[ti])
            dt = float(ts[ti + 1] - ts[ti])
            for yi in range(self.grid_y):
                y = int(ys[yi])
                dy = int(max(1, ys[yi + 1] - ys[yi]))
                if y + dy > self.image_height:
                    dy = self.image_height - y
                for xi in range(self.grid_x):
                    x = int(xs[xi])
                    dx = int(max(1, xs[xi + 1] - xs[xi]))
                    if x + dx > self.image_width:
                        dx = self.image_width - x
                    if self.grid_plane_mode == "all":
                        for plane in self.plane_types:
                            regions.append(
                                Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane)
                            )
                    else:
                        plane = self.plane_types[len(regions) % len(self.plane_types)]
                        regions.append(Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane))
        return regions

    def _select_regions(
        self, rng: np.random.Generator, regions: List[Region]
    ) -> List[Region]:
        if self.num_regions <= 0 or self.num_regions == len(regions):
            return regions
        if self.num_regions < len(regions):
            idx = rng.choice(len(regions), size=self.num_regions, replace=False)
            return [regions[i] for i in idx]
        repeats = (self.num_regions + len(regions) - 1) // len(regions)
        tiled = (regions * repeats)[: self.num_regions]
        return tiled

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_token_variants_per_sample > 0:
            return self._get_token_variant(idx)

        if self.cache_token_max_samples > 0 and idx in self._token_cache:
            cached = self._token_cache.pop(idx)
            self._token_cache[idx] = cached
            return cached

        events, seq_len_sec = self._load_events_cached(idx)
        if self.fixed_region_seed >= 0:
            seed = self.fixed_region_seed
            if not self.fixed_region_positions_global:
                seed += idx
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng
        patches, metadata, counts, plane_ids = self._build_sample(events, seq_len_sec, rng)

        sample = {
            "patches": patches,
            "metadata": metadata,
            "event_counts": counts,
            "plane_ids": plane_ids,
        }
        if self.cache_token_max_samples > 0:
            self._token_cache[idx] = sample
            if len(self._token_cache) > self.cache_token_max_samples:
                self._token_cache.popitem(last=False)
        return sample

    def _select_variant(self, idx: int) -> int:
        if self.cache_token_variant_mode == "round_robin":
            cursor = self._variant_cursor.get(idx, 0)
            variant = cursor % self.cache_token_variants_per_sample
            self._variant_cursor[idx] = cursor + 1
            return variant
        return int(self.rng.integers(0, self.cache_token_variants_per_sample))

    def _get_token_variant(self, idx: int) -> Dict[str, torch.Tensor]:
        variant = self._select_variant(idx)
        cache_path = self.cache_token_dir / f"thu_{idx:06d}_v{variant}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            sample = {
                "patches": torch.from_numpy(cached["patches"]),
                "metadata": torch.from_numpy(cached["metadata"]),
                "event_counts": torch.from_numpy(cached["event_counts"]),
                "plane_ids": torch.from_numpy(cached["plane_ids"]),
            }
            if self.return_label:
                if "label" in cached:
                    sample["label"] = torch.from_numpy(cached["label"])
                else:
                    sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return sample

        events, seq_len_sec = self._load_events_cached(idx)
        if self.fixed_region_seed >= 0:
            seed = self.fixed_region_seed + idx * 1000 + variant
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        patches, metadata, counts, plane_ids = self._build_sample(events, seq_len_sec, rng)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            patches=patches.numpy(),
            metadata=metadata.numpy(),
            event_counts=counts.numpy(),
            plane_ids=plane_ids.numpy(),
            label=np.array(self.labels[idx], dtype=np.int64),
        )
        sample = {
            "patches": patches,
            "metadata": metadata,
            "event_counts": counts,
            "plane_ids": plane_ids,
        }
        if self.return_label:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample

    def _build_sample(
        self, events: np.ndarray, seq_len_sec: float, rng: np.random.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches: List[torch.Tensor] = []
        metadata: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        plane_ids: List[int] = []

        if self.fixed_single_region:
            region = self._sample_region(rng)
            regions = [region] * self.num_regions
        elif self.region_sampling == "grid":
            regions = self._select_regions(rng, self._grid_regions())
        else:
            regions = [self._sample_region(rng) for _ in range(self.num_regions)]

        for region in regions:
            patch, total_events = build_patch(
                events, region, self.patch_size, self.time_bins, patch_divider=self.patch_divider
            )
            patches.append(patch)
            counts.append(torch.tensor([total_events], dtype=torch.float32))
            plane_ids.append(self.plane_types.index(region.plane))

            meta = torch.tensor(
                [
                    region.x / self.image_width,
                    region.y / self.image_height,
                    region.dx / self.image_width,
                    region.dy / self.image_height,
                    region.t,
                    region.dt,
                    region.t * seq_len_sec,
                    region.dt * seq_len_sec,
                    seq_len_sec,
                ],
                dtype=torch.float32,
            )
            metadata.append(meta)

        return (
            torch.stack(patches, dim=0),
            torch.stack(metadata, dim=0),
            torch.stack(counts, dim=0),
            torch.tensor(plane_ids, dtype=torch.long),
        )
