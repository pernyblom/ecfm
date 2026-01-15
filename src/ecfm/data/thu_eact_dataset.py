from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import Region, build_patch


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
        num_regions_choices: List[int],
        patch_size: int,
        max_samples: int = 0,
        max_events: int = 0,
        subset_seed: int = 0,
        fixed_region_seed: int = -1,
        patch_divider: float = 0.0,
        patch_norm: str = "region_max",
        patch_norm_eps: float = 1e-6,
        fixed_region_sizes: bool = False,
        fixed_region_positions_global: bool = False,
        fixed_single_region: bool = False,
        cache_max_samples: int = 0,
        cache_token_max_samples: int = 0,
        cache_token_variants_per_sample: int = 0,
        cache_token_dir: str = "outputs/token_cache",
        cache_token_variant_mode: str = "random",
        cache_token_clear_on_start: bool = False,
        cache_token_config_id: str = "",
        cache_token_drop_prob: float = 0.0,
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
        self.num_regions_choices = [int(v) for v in num_regions_choices if int(v) > 0]
        self.patch_size = patch_size
        self.max_events = max_events
        self.fixed_region_seed = fixed_region_seed
        self.patch_divider = patch_divider
        self.patch_norm = patch_norm
        self.patch_norm_eps = patch_norm_eps
        self.fixed_region_sizes = fixed_region_sizes
        self.fixed_region_positions_global = fixed_region_positions_global
        self.fixed_single_region = fixed_single_region
        grid_count = self._grid_region_count()
        self.max_regions = max(
            0,
            self.num_regions,
            max(self.num_regions_choices, default=0),
            grid_count if self.region_sampling == "grid" and self.num_regions <= 0 else 0,
        )
        if self.max_regions <= 0:
            raise ValueError("num_regions must be > 0 unless region_sampling=grid")
        if self.fixed_single_region and self.max_regions <= 0:
            raise ValueError("fixed_single_region requires num_regions > 0")
        self.cache_max_samples = cache_max_samples
        self._event_cache: OrderedDict[int, Tuple[np.ndarray, float]] = OrderedDict()
        self.cache_token_max_samples = cache_token_max_samples
        self._token_cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.cache_token_variants_per_sample = cache_token_variants_per_sample
        self.cache_token_dir = Path(cache_token_dir)
        self.cache_token_variant_mode = cache_token_variant_mode
        self.cache_token_clear_on_start = cache_token_clear_on_start
        self.cache_token_config_id = cache_token_config_id
        self.cache_token_drop_prob = float(cache_token_drop_prob)
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
            idx = self.rng.choice(len(self.files), size=max_samples, replace=False)
            self.files = [self.files[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]
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
        self, rng: np.random.Generator, regions: List[Region], desired_num: int
    ) -> List[Region]:
        if desired_num <= 0 or desired_num == len(regions):
            return regions
        if desired_num < len(regions):
            idx = rng.choice(len(regions), size=desired_num, replace=False)
            return [regions[i] for i in idx]
        repeats = (desired_num + len(regions) - 1) // len(regions)
        tiled = (regions * repeats)[:desired_num]
        return tiled

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_token_variants_per_sample > 0:
            return self._get_token_variant(idx)

        if self.cache_token_max_samples > 0 and idx in self._token_cache:
            if self.cache_token_drop_prob > 0 and self.rng.random() < self.cache_token_drop_prob:
                self._token_cache.pop(idx, None)
            else:
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
        patches, metadata, counts, plane_ids, valid_mask = self._build_sample(
            events, seq_len_sec, rng
        )

        sample = {
            "patches": patches,
            "metadata": metadata,
            "event_counts": counts,
            "plane_ids": plane_ids,
            "valid_mask": valid_mask,
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
        file_id = hashlib.sha1(str(self.files[idx]).encode("utf-8")).hexdigest()[:10]
        cfg_tag = f"_{self.cache_token_config_id}" if self.cache_token_config_id else ""
        cache_path = (
            self.cache_token_dir
            / f"thu_{self.split}_{idx:06d}_{file_id}{cfg_tag}_v{variant}.npz"
        )
        if cache_path.exists():
            if self.cache_token_drop_prob > 0 and self.rng.random() < self.cache_token_drop_prob:
                try:
                    cache_path.unlink()
                except OSError:
                    pass
            else:
                cached = np.load(cache_path)
                sample = {
                    "patches": torch.from_numpy(cached["patches"]),
                    "metadata": torch.from_numpy(cached["metadata"]),
                    "event_counts": torch.from_numpy(cached["event_counts"]),
                    "plane_ids": torch.from_numpy(cached["plane_ids"]),
                }
                if "valid_mask" in cached:
                    sample["valid_mask"] = torch.from_numpy(cached["valid_mask"])
                else:
                    sample["valid_mask"] = torch.ones(
                        sample["patches"].shape[0], dtype=torch.float32
                    )
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

        patches, metadata, counts, plane_ids, valid_mask = self._build_sample(
            events, seq_len_sec, rng
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            patches=patches.numpy(),
            metadata=metadata.numpy(),
            event_counts=counts.numpy(),
            plane_ids=plane_ids.numpy(),
            valid_mask=valid_mask.numpy(),
            label=np.array(self.labels[idx], dtype=np.int64),
        )
        sample = {
            "patches": patches,
            "metadata": metadata,
            "event_counts": counts,
            "plane_ids": plane_ids,
            "valid_mask": valid_mask,
        }
        if self.return_label:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample

    def _build_sample(
        self, events: np.ndarray, seq_len_sec: float, rng: np.random.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches: List[torch.Tensor] = []
        metadata: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        plane_ids: List[int] = []

        desired_num = self.num_regions
        if self.num_regions_choices:
            desired_num = int(rng.choice(self.num_regions_choices))
        if self.fixed_single_region:
            region = self._sample_region(rng)
            regions = [region] * desired_num
        elif self.region_sampling == "grid":
            regions = self._select_regions(rng, self._grid_regions(), desired_num)
        else:
            regions = [self._sample_region(rng) for _ in range(desired_num)]

        for region in regions:
            patch, total_events = build_patch(
                events,
                region,
                self.patch_size,
                self.time_bins,
                patch_divider=self.patch_divider,
                norm_mode=self.patch_norm,
                norm_eps=self.patch_norm_eps,
            )
            patches.append(patch)
            area = float(region.dx * region.dy)
            dt_sec = float(region.dt * seq_len_sec)
            denom = max(1e-6, area * dt_sec)
            rate = total_events / denom
            log_rate = float(np.log1p(rate))
            counts.append(torch.tensor([log_rate], dtype=torch.float32))
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

        valid_count = len(patches)
        if valid_count < self.max_regions:
            pad = self.max_regions - valid_count
            patches.extend(
                [torch.zeros_like(patches[0]) for _ in range(pad)]
                if patches
                else [torch.zeros((2, self.patch_size, self.patch_size), dtype=torch.float32)]
                * pad
            )
            metadata.extend([torch.zeros((9,), dtype=torch.float32) for _ in range(pad)])
            counts.extend([torch.tensor([0.0], dtype=torch.float32) for _ in range(pad)])
            plane_ids.extend([0 for _ in range(pad)])
        valid_mask = torch.zeros((self.max_regions,), dtype=torch.float32)
        valid_mask[:valid_count] = 1.0

        return (
            torch.stack(patches, dim=0),
            torch.stack(metadata, dim=0),
            torch.stack(counts, dim=0),
            torch.tensor(plane_ids, dtype=torch.long),
            valid_mask,
        )

    def _grid_region_count(self) -> int:
        if self.grid_x <= 0 or self.grid_y <= 0 or self.grid_t <= 0:
            return 0
        count = self.grid_x * self.grid_y * self.grid_t
        if self.grid_plane_mode == "all":
            count *= len(self.plane_types)
        return count
