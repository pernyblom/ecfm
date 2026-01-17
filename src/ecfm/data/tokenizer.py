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


def _histogram_xy_rot(
    events: np.ndarray, region: Region, time_bins: int, polarity: int, angle_deg: float
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
    y_norm = (sub[:, 1] - region.y) / max(region.dy, 1e-6)
    t_norm = (sub[:, 2] - region.t) / max(region.dt, 1e-6)
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    corners = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_rot_corners = corners[:, 0] * c + corners[:, 1] * s
    y_min = float(y_rot_corners.min())
    y_max = float(y_rot_corners.max())
    denom = max(1e-6, y_max - y_min)
    y_rot = (y_norm * c + t_norm * s - y_min) / denom
    y_idx = np.clip((y_rot * time_bins).astype(np.int64), 0, time_bins - 1)
    hist = np.zeros((time_bins, region.dx), dtype=np.float32)
    np.add.at(hist, (y_idx, x), 1.0)
    return hist


def _histogram_yt_rot(
    events: np.ndarray, region: Region, time_bins: int, polarity: int, angle_deg: float
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
    y_norm = (sub[:, 1] - region.y) / max(region.dy, 1e-6)
    x_norm = (sub[:, 0] - region.x) / max(region.dx, 1e-6)
    t_norm = (sub[:, 2] - region.t) / max(region.dt, 1e-6)
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    corners = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_rot_corners = corners[:, 0] * s + corners[:, 1] * c
    y_min = float(y_rot_corners.min())
    y_max = float(y_rot_corners.max())
    denom = max(1e-6, y_max - y_min)
    y_rot = (x_norm * s + y_norm * c - y_min) / denom
    y_idx = np.clip((y_rot * region.dy).astype(np.int64), 0, region.dy - 1)
    t_idx = np.clip((t_norm * time_bins).astype(np.int64), 0, time_bins - 1)
    hist = np.zeros((time_bins, region.dy), dtype=np.float32)
    np.add.at(hist, (t_idx, y_idx), 1.0)
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
    elif region.plane == "xy_p45":
        hist0 = _histogram_xy_rot(events, region, time_bins, polarity=0, angle_deg=45.0)
        hist1 = _histogram_xy_rot(events, region, time_bins, polarity=1, angle_deg=45.0)
    elif region.plane == "xy_m45":
        hist0 = _histogram_xy_rot(events, region, time_bins, polarity=0, angle_deg=-45.0)
        hist1 = _histogram_xy_rot(events, region, time_bins, polarity=1, angle_deg=-45.0)
    elif region.plane == "yt_p45":
        hist0 = _histogram_yt_rot(events, region, time_bins, polarity=0, angle_deg=45.0)
        hist1 = _histogram_yt_rot(events, region, time_bins, polarity=1, angle_deg=45.0)
    elif region.plane == "yt_m45":
        hist0 = _histogram_yt_rot(events, region, time_bins, polarity=0, angle_deg=-45.0)
        hist1 = _histogram_yt_rot(events, region, time_bins, polarity=1, angle_deg=-45.0)
    else:
        raise ValueError(f"Unknown plane {region.plane}")

    total_events = float(hist0.sum() + hist1.sum())
    hist = np.stack([hist0, hist1], axis=0)
    hist_t = torch.from_numpy(hist).unsqueeze(0)
    patch = F.interpolate(hist_t, size=(patch_size, patch_size), mode="bilinear", align_corners=False)
    patch = patch.squeeze(0)

    if patch_divider > 0:
        patch = patch / float(patch_divider)
    elif norm_mode == "region_max":
        for ch in range(patch.shape[0]):
            maxv = float(patch[ch].max().item())
            if maxv > 0:
                patch[ch] = patch[ch] / maxv
    elif norm_mode == "region_sum":
        for ch in range(patch.shape[0]):
            sumv = float(patch[ch].sum().item())
            if sumv > 0:
                patch[ch] = patch[ch] / sumv
    elif norm_mode == "region_mean":
        for ch in range(patch.shape[0]):
            meanv = float(patch[ch].mean().item())
            if meanv > norm_eps:
                patch[ch] = patch[ch] / meanv
    elif norm_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown patch normalization mode: {norm_mode}")

    return patch, total_events
