from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Region:
    x: int
    y: int
    t: float
    dx: int
    dy: int
    dt: float
    plane: str


def _histogram_xy(
    events: np.ndarray, region: Region, time_bins: int, polarity: int
) -> np.ndarray:
    mask = (
        (events[:, 0] >= region.x)
        & (events[:, 0] < region.x + region.dx)
        & (events[:, 1] >= region.y)
        & (events[:, 1] < region.y + region.dy)
        & (events[:, 2] >= region.t)
        & (events[:, 2] < region.t + region.dt)
        & (events[:, 3] == polarity)
    )
    sub = events[mask]
    if sub.shape[0] == 0:
        return np.zeros((region.dy, region.dx), dtype=np.float32)
    x = (sub[:, 0] - region.x).astype(np.int64)
    y = (sub[:, 1] - region.y).astype(np.int64)
    hist = np.zeros((region.dy, region.dx), dtype=np.float32)
    np.add.at(hist, (y, x), 1.0)
    return hist


def _histogram_xt(
    events: np.ndarray, region: Region, time_bins: int, polarity: int
) -> np.ndarray:
    mask = (
        (events[:, 0] >= region.x)
        & (events[:, 0] < region.x + region.dx)
        & (events[:, 1] >= region.y)
        & (events[:, 1] < region.y + region.dy)
        & (events[:, 2] >= region.t)
        & (events[:, 2] < region.t + region.dt)
        & (events[:, 3] == polarity)
    )
    sub = events[mask]
    if sub.shape[0] == 0:
        return np.zeros((time_bins, region.dx), dtype=np.float32)
    x = (sub[:, 0] - region.x).astype(np.int64)
    t_norm = (sub[:, 2] - region.t) / max(region.dt, 1e-6)
    t_idx = np.clip((t_norm * time_bins).astype(np.int64), 0, time_bins - 1)
    hist = np.zeros((time_bins, region.dx), dtype=np.float32)
    np.add.at(hist, (t_idx, x), 1.0)
    return hist


def _histogram_yt(
    events: np.ndarray, region: Region, time_bins: int, polarity: int
) -> np.ndarray:
    mask = (
        (events[:, 0] >= region.x)
        & (events[:, 0] < region.x + region.dx)
        & (events[:, 1] >= region.y)
        & (events[:, 1] < region.y + region.dy)
        & (events[:, 2] >= region.t)
        & (events[:, 2] < region.t + region.dt)
        & (events[:, 3] == polarity)
    )
    sub = events[mask]
    if sub.shape[0] == 0:
        return np.zeros((time_bins, region.dy), dtype=np.float32)
    y = (sub[:, 1] - region.y).astype(np.int64)
    t_norm = (sub[:, 2] - region.t) / max(region.dt, 1e-6)
    t_idx = np.clip((t_norm * time_bins).astype(np.int64), 0, time_bins - 1)
    hist = np.zeros((time_bins, region.dy), dtype=np.float32)
    np.add.at(hist, (t_idx, y), 1.0)
    return hist


def build_patch(
    events: np.ndarray,
    region: Region,
    patch_size: int,
    time_bins: int,
    patch_divider: float = 0.0,
    norm_mode: str = "region_max",
    norm_eps: float = 1e-6,
) -> Tuple[torch.Tensor, float]:
    if region.plane == "xy":
        hist0 = _histogram_xy(events, region, time_bins, polarity=0)
        hist1 = _histogram_xy(events, region, time_bins, polarity=1)
    elif region.plane == "xt":
        hist0 = _histogram_xt(events, region, time_bins, polarity=0)
        hist1 = _histogram_xt(events, region, time_bins, polarity=1)
    elif region.plane == "yt":
        hist0 = _histogram_yt(events, region, time_bins, polarity=0)
        hist1 = _histogram_yt(events, region, time_bins, polarity=1)
    else:
        raise ValueError(f"Unknown plane {region.plane}")

    total_events = float(hist0.sum() + hist1.sum())
    if patch_divider > 0:
        hist0 = hist0 / patch_divider
        hist1 = hist1 / patch_divider
    elif norm_mode == "region_max":
        max0 = float(hist0.max()) if hist0.size > 0 else 0.0
        max1 = float(hist1.max()) if hist1.size > 0 else 0.0
        if max0 > 0:
            hist0 = hist0 / max0
        if max1 > 0:
            hist1 = hist1 / max1
    elif norm_mode == "region_sum":
        sum0 = float(hist0.sum()) if hist0.size > 0 else 0.0
        sum1 = float(hist1.sum()) if hist1.size > 0 else 0.0
        if sum0 > 0:
            hist0 = hist0 / sum0
        if sum1 > 0:
            hist1 = hist1 / sum1
    elif norm_mode == "region_mean":
        mean0 = float(hist0.mean()) if hist0.size > 0 else 0.0
        mean1 = float(hist1.mean()) if hist1.size > 0 else 0.0
        if mean0 > norm_eps:
            hist0 = hist0 / mean0
        if mean1 > norm_eps:
            hist1 = hist1 / mean1
    elif norm_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown patch normalization mode: {norm_mode}")
    hist = np.stack([hist0, hist1], axis=0)
    hist_t = torch.from_numpy(hist).unsqueeze(0)
    patch = F.interpolate(hist_t, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
    return patch.squeeze(0), total_events
