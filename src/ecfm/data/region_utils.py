from __future__ import annotations

from typing import List

import numpy as np
import torch

from .augmentation import rotate_events
from .tokenizer import Region, build_patch


def sample_region(
    rng: np.random.Generator,
    image_width: int,
    image_height: int,
    region_scales: List[int],
    region_scales_x: List[int],
    region_scales_y: List[int],
    region_time_scales: List[float],
    plane_types_active: List[str],
    fixed_region_sizes: bool,
) -> Region:
    if fixed_region_sizes:
        dx = int(region_scales_x[0] if region_scales_x else region_scales[0])
        dy = int(region_scales_y[0] if region_scales_y else region_scales[0])
        dt = float(region_time_scales[0])
    else:
        if region_scales_x:
            dx = int(rng.choice(region_scales_x))
        else:
            dx = int(rng.choice(region_scales))
        if region_scales_y:
            dy = int(rng.choice(region_scales_y))
        else:
            dy = int(rng.choice(region_scales))
        dt = float(rng.choice(region_time_scales))
    dx = min(dx, image_width)
    dy = min(dy, image_height)
    x = int(rng.integers(0, max(1, image_width - dx)))
    y = int(rng.integers(0, max(1, image_height - dy)))
    t = float(rng.random() * max(1e-6, 1.0 - dt))
    plane = str(rng.choice(plane_types_active))
    return Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane)


def resolve_region_scales(
    region_scales: List[float],
    region_scales_x: List[float],
    region_scales_y: List[float],
    image_width: int,
    image_height: int,
    mode: str,
) -> tuple[List[int], List[int], List[int]]:
    if mode == "fraction" and region_scales and not region_scales_x and not region_scales_y:
        region_scales_x = list(region_scales)
        region_scales_y = list(region_scales)

    def _scale(values: List[float], dim: int) -> List[int]:
        if not values:
            return []
        if mode == "absolute":
            return [int(v) for v in values]
        if mode == "fraction":
            resolved = []
            for v in values:
                px = int(round(float(v) * dim))
                resolved.append(max(1, min(dim, px)))
            return resolved
        raise ValueError(f"Unknown region_scale_mode: {mode}")

    return (
        _scale(region_scales, image_width),
        _scale(region_scales_x, image_width),
        _scale(region_scales_y, image_height),
    )


def grid_regions(
    image_width: int,
    image_height: int,
    grid_x: int,
    grid_y: int,
    grid_t: int,
    grid_plane_mode: str,
    plane_types_active: List[str],
) -> List[Region]:
    if grid_x <= 0 or grid_y <= 0 or grid_t <= 0:
        raise ValueError("grid_x, grid_y, grid_t must be > 0 for grid sampling")
    xs = np.linspace(0, image_width, grid_x + 1, dtype=int)
    ys = np.linspace(0, image_height, grid_y + 1, dtype=int)
    ts = np.linspace(0.0, 1.0, grid_t + 1, dtype=np.float32)
    regions: List[Region] = []
    for ti in range(grid_t):
        t = float(ts[ti])
        dt = float(ts[ti + 1] - ts[ti])
        for yi in range(grid_y):
            y = int(ys[yi])
            dy = int(max(1, ys[yi + 1] - ys[yi]))
            if y + dy > image_height:
                dy = image_height - y
            for xi in range(grid_x):
                x = int(xs[xi])
                dx = int(max(1, xs[xi + 1] - xs[xi]))
                if x + dx > image_width:
                    dx = image_width - x
                if grid_plane_mode == "all":
                    for plane in plane_types_active:
                        regions.append(Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane))
                else:
                    plane = plane_types_active[len(regions) % len(plane_types_active)]
                    regions.append(Region(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, plane=plane))
    return regions


def select_regions(
    rng: np.random.Generator, regions: List[Region], desired_num: int
) -> List[Region]:
    if desired_num <= 0 or desired_num == len(regions):
        return regions
    if desired_num < len(regions):
        idx = rng.choice(len(regions), size=desired_num, replace=False)
        return [regions[i] for i in idx]
    repeats = (desired_num + len(regions) - 1) // len(regions)
    return (regions * repeats)[:desired_num]


def grid_region_count(
    grid_x: int,
    grid_y: int,
    grid_t: int,
    grid_plane_mode: str,
    plane_types_active: List[str],
) -> int:
    if grid_x <= 0 or grid_y <= 0 or grid_t <= 0:
        return 0
    count = grid_x * grid_y * grid_t
    if grid_plane_mode == "all":
        count *= len(plane_types_active)
    return count


def apply_augmentations(
    events: np.ndarray,
    rng: np.random.Generator,
    augmentations: List[str],
    rotation_max_deg: float,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    if not augmentations:
        return events
    if "rotate" in augmentations and rotation_max_deg > 0:
        angle = float(rng.uniform(-rotation_max_deg, rotation_max_deg))
        events = rotate_events(events, (image_height, image_width), angle)
    return events


def build_sample(
    events: np.ndarray,
    seq_len_sec: float,
    rng: np.random.Generator,
    image_width: int,
    image_height: int,
    time_bins: int,
    patch_size: int,
    patch_divider: float,
    patch_norm: str,
    patch_norm_eps: float,
    plane_types: List[str],
    plane_types_active: List[str],
    num_regions: int,
    num_regions_choices: List[int],
    fixed_single_region: bool,
    fixed_region_sizes: bool,
    region_sampling: str,
    grid_x: int,
    grid_y: int,
    grid_t: int,
    grid_plane_mode: str,
    region_scales: List[int],
    region_scales_x: List[int],
    region_scales_y: List[int],
    region_time_scales: List[float],
    max_regions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    patches: List[torch.Tensor] = []
    metadata: List[torch.Tensor] = []
    counts: List[torch.Tensor] = []
    plane_ids: List[int] = []

    desired_num = num_regions
    if num_regions_choices:
        desired_num = int(rng.choice(num_regions_choices))

    if fixed_single_region:
        region = sample_region(
            rng,
            image_width,
            image_height,
            region_scales,
            region_scales_x,
            region_scales_y,
            region_time_scales,
            plane_types_active,
            fixed_region_sizes,
        )
        regions = [region] * desired_num
    elif region_sampling == "grid":
        regions = select_regions(
            rng,
            grid_regions(
                image_width,
                image_height,
                grid_x,
                grid_y,
                grid_t,
                grid_plane_mode,
                plane_types_active,
            ),
            desired_num,
        )
    else:
        regions = [
            sample_region(
                rng,
                image_width,
                image_height,
                region_scales,
                region_scales_x,
                region_scales_y,
                region_time_scales,
                plane_types_active,
                fixed_region_sizes,
            )
            for _ in range(desired_num)
        ]

    for region in regions:
        patch, total_events = build_patch(
            events,
            region,
            patch_size,
            time_bins,
            patch_divider=patch_divider,
            norm_mode=patch_norm,
            norm_eps=patch_norm_eps,
        )
        patches.append(patch)
        area = float(region.dx * region.dy)
        dt_sec = float(region.dt * seq_len_sec)
        denom = max(1e-6, area * dt_sec)
        rate = total_events / denom
        log_rate = float(np.log1p(rate))
        counts.append(torch.tensor([log_rate], dtype=torch.float32))
        plane_ids.append(plane_types.index(region.plane))

        meta = torch.tensor(
            [
                region.x / image_width,
                region.y / image_height,
                region.dx / image_width,
                region.dy / image_height,
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
    if valid_count < max_regions:
        pad = max_regions - valid_count
        patches.extend(
            [torch.zeros_like(patches[0]) for _ in range(pad)]
            if patches
            else [torch.zeros((2, patch_size, patch_size), dtype=torch.float32)] * pad
        )
        metadata.extend([torch.zeros((9,), dtype=torch.float32) for _ in range(pad)])
        counts.extend([torch.tensor([0.0], dtype=torch.float32) for _ in range(pad)])
        plane_ids.extend([0 for _ in range(pad)])
    valid_mask = torch.zeros((max_regions,), dtype=torch.float32)
    valid_mask[:valid_count] = 1.0

    return (
        torch.stack(patches, dim=0),
        torch.stack(metadata, dim=0),
        torch.stack(counts, dim=0),
        torch.tensor(plane_ids, dtype=torch.long),
        valid_mask,
    )
