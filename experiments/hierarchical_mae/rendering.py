"""Partition events once per level and render local histograms without rescanning."""
import numpy as np
import torch
from torch.nn import functional as F

from ecfm.data.tokenizer import build_patch


def partition_level(events, splits, maximum, width, height):
    """Return voxels in t,y,x order, preserving event order within each voxel.

    Use the same rounded pixel and float32 temporal boundaries as the original
    half-open Region comparisons, including for non-power-of-two grids.
    """
    nx, ny, nt = splits
    mx, my, mt = maximum
    xs = np.arange(nx+1) * (mx//nx) * width // mx
    ys = np.arange(ny+1) * (my//ny) * height // my
    ts = (np.arange(nt+1) / nt).astype(np.float32)
    ix = np.searchsorted(xs, events[:, 0], side='right')-1
    iy = np.searchsorted(ys, events[:, 1], side='right')-1
    it = np.searchsorted(ts, events[:, 2], side='right')-1
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (it >= 0) & (it < nt)
    ids = (it[valid]*ny + iy[valid])*nx + ix[valid]
    order = np.argsort(ids, kind='stable')
    ordered = events[valid][order]
    offsets = np.r_[0, np.cumsum(np.bincount(ids, minlength=nx*ny*nt))]
    return [ordered[offsets[i]:offsets[i+1]] for i in range(nx*ny*nt)]


def render_histogram(events, region, size, time_bins, norm_mode):
    """Equivalent to build_patch for prefiltered xy/xt/yt voxels; fallback for rotations."""
    if region.plane not in ('xy', 'xt', 'yt'):
        return build_patch(events, region, size, time_bins, norm_mode=norm_mode)[0]
    x = (events[:, 0]-region.x).astype(np.int64)
    y = (events[:, 1]-region.y).astype(np.int64)
    polarity = events[:, 3].astype(np.int64)
    if region.plane == 'xy':
        h, w = region.dy, region.dx
        row, col = y, x
    else:
        h = time_bins
        row = np.clip(((events[:, 2]-region.t)/max(region.dt, 1e-6)*time_bins).astype(np.int64), 0, time_bins-1)
        w, col = (region.dx, x) if region.plane == 'xt' else (region.dy, y)
    flat = (polarity*h+row)*w+col
    histogram = np.bincount(flat, minlength=2*h*w).reshape(2, h, w).astype(np.float32)
    patch = F.interpolate(torch.from_numpy(histogram)[None], (size, size), mode='bilinear', align_corners=False)[0]
    if norm_mode == 'none':
        return patch
    if norm_mode == 'region_max':
        divisor = patch.amax((1, 2), keepdim=True)
    elif norm_mode == 'region_sum':
        divisor = patch.sum((1, 2), keepdim=True)
    elif norm_mode == 'region_mean':
        divisor = patch.mean((1, 2), keepdim=True)
    else:
        raise ValueError(f'Unknown patch normalization mode: {norm_mode}')
    threshold = 1e-6 if norm_mode == 'region_mean' else 0
    return patch / torch.where(divisor > threshold, divisor, torch.ones_like(divisor))
