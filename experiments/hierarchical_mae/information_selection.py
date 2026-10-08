"""Deterministic structure scores from unprojected, polarity-pooled event voxels."""
import math

import numpy as np
import torch

METRICS = ('support', 'entropy', 'autocorrelation')
STRATEGIES = ('information', 'activity_information')


def settings(cfg):
    """Opt in explicitly or automatically when either selector needs scores."""
    options = cfg['data'].get('information_selection')
    needed = any(s.get('strategy') in STRATEGIES for s in
                 (cfg.get('selection', {}), cfg.get('downstream', {}).get('selection', {})))
    return {'bins': [8, 8, 8], 'support_saturation': 3., **(options or {})} if options is not None or needed else None


def validate_settings(cfg):
    options = settings(cfg)
    if options is None:
        return
    if set(options) - {'bins', 'support_saturation'}:
        raise ValueError('Invalid data.information_selection settings')
    bins = options['bins']
    if (not isinstance(bins, (list, tuple)) or len(bins) != 3
            or any(type(v) is not int or v < 2 for v in bins)):
        raise ValueError('information_selection.bins must be three integers >=2 [x,y,t]')
    saturation = options['support_saturation']
    if type(saturation) not in (int, float) or not math.isfinite(saturation) or saturation <= 0:
        raise ValueError('information_selection.support_saturation must be finite and positive')


def voxel_histogram(events, region, bins):
    """Fixed local x/y/t resolution; no resizing of rendered projection patches."""
    shape = np.asarray(bins, dtype=np.int64)
    histogram = np.zeros(tuple(shape), dtype=np.float64)
    if len(events):
        origin = np.asarray([region.x, region.y, region.t])
        extent = np.asarray([region.dx, region.dy, region.dt])
        indices = np.floor((events[:, :3] - origin) / extent * shape).astype(np.int64)
        indices = np.clip(indices, 0, shape - 1)
        np.add.at(histogram, tuple(indices.T), 1)
    return histogram


def histogram_scores(volume, saturation=3.):
    """Return support, entropy deficit, and excess positive-axis autocorrelation.

    Support uses the 24 adjacent bins with a spatial displacement and |dt|<=1;
    same-pixel temporal repeats cannot support each other. Boundaries do not wrap.
    Autocorrelation subtracts the *exact* expected score under a uniform permutation
    of histogram bins (preserves count and histogram values, not an iid-event null).
    """
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3 or min(volume.shape) < 2 or not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError('Expected a finite nonnegative 3-D histogram with axes >=2')
    if not math.isfinite(saturation) or saturation <= 0:
        raise ValueError('saturation must be finite and positive')
    total = volume.sum()
    if total == 0:
        return np.zeros(len(METRICS), dtype=np.float32)
    padded = np.pad(volume, 1)
    neighbors = np.zeros_like(volume)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            for dt in (-1, 0, 1):
                neighbors += padded[1+dx:1+dx+volume.shape[0],
                                    1+dy:1+dy+volume.shape[1],
                                    1+dt:1+dt+volume.shape[2]]
    support = (volume * np.minimum(neighbors / saturation, 1)).sum() / total
    probabilities = volume[volume > 0] / total
    entropy = 1 + (probabilities * np.log(probabilities)).sum() / np.log(volume.size)
    centered = volume - volume.mean()
    variance = np.square(centered).sum()
    autocorrelation = 0.
    if variance > 0:
        for axis in range(3):
            left, right = [slice(None)] * 3, [slice(None)] * 3
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            pairs = centered[tuple(left)].size
            # E[z_i*z_j]/sum(z^2) = -1/(K*(K-1)) for distinct bins.
            autocorrelation += ((centered[tuple(left)] * centered[tuple(right)]).sum() / variance
                                + pairs / (volume.size * (volume.size - 1)))
    return np.asarray([support, np.clip(entropy, 0, 1), autocorrelation], dtype=np.float32)


def selection_scores(view, eligible, options):
    """Blend normalizes each signal over eligible tokens per recording only."""
    metric = options.get('metric', 'support')
    if metric not in METRICS:
        raise ValueError(f'Unknown information metric: {metric}')
    values = view.get('information_scores')
    if values is None:
        raise ValueError('Information selection requires raw-voxel information_scores from HierarchyDataset')
    if values.shape != (*eligible.shape, len(METRICS)) or not torch.isfinite(values).all():
        raise ValueError('information_scores must be finite [B,N,3]')
    information = values[..., METRICS.index(metric)]
    if options.get('strategy') == 'information':
        return information
    activity = view['log_counts']
    if not torch.isfinite(activity).all() or (activity < 0).any():
        raise ValueError('Activity must be finite and nonnegative')
    combination = options.get('combination', 'blend')
    if combination == 'product':
        return activity * information.clamp_min(0)
    if combination != 'blend':
        raise ValueError('Information combination must be blend or product')
    weight = options.get('activity_weight', .5)
    if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('activity_weight must be finite and in [0,1]')
    def normalize(signal):
        low = signal.masked_fill(~eligible, torch.inf).amin(1, keepdim=True)
        high = signal.masked_fill(~eligible, -torch.inf).amax(1, keepdim=True)
        # Also return finite scores for rows with no eligible tokens; plan validation
        # subsequently reports that there must be at least one visible token.
        low = torch.where(eligible.any(1, keepdim=True), low, torch.zeros_like(low))
        high = torch.where(eligible.any(1, keepdim=True), high, low)
        return (signal - low) / (high - low).clamp_min(torch.finfo(signal.dtype).eps)
    return weight * normalize(activity) + (1 - weight) * normalize(information)
