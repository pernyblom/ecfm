"""Display previews of event tensors; never used to construct training inputs."""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw

from .event_representations import _resolve_representation_alias, representation_options_for


def resolve_event_visualization(cfg, representation):
    """Merge visualization.event_images defaults with exact/base-name overrides."""
    settings = dict((cfg.get('visualization') or {}).get('event_images') or {})
    overrides = settings.pop('by_representation', {}) or {}
    settings.update(representation_options_for(representation, overrides))
    return settings


def event_tensor_preview(tensor, settings=None, *, signed=False):
    """Create a PIL RGB preview from [C,H,W]; channel selection uses zero-based indices.

    Auto preserves unsigned RGB, displays two channels as red/blue, and averages
    larger unsigned tensors into early/middle/late RGB groups. Signed inputs use
    positive red and negative blue. Grid mode shows every selected channel.
    """
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().cpu().numpy()
    values = np.asarray(tensor, dtype=np.float32)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('Preview expects finite nonempty [C,H,W] tensor')
    cfg = dict(settings or {})
    allowed = {'mode', 'channels', 'normalization', 'columns', 'labels'}
    if set(cfg) - allowed:
        raise ValueError(f'Unknown event visualization settings: {sorted(set(cfg)-allowed)}')
    indices = cfg.get('channels', list(range(values.shape[0])))
    if not isinstance(indices, (list, tuple)) or not indices or any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < values.shape[0] for i in indices
    ):
        raise ValueError('Preview channels must be a nonempty list of valid channel indices')
    values = values[list(indices)].copy()
    signed = signed or bool(np.any(values < 0))
    normalization = cfg.get('normalization', 'auto')
    if normalization not in {'auto', 'none', 'max', 'min_max', 'per_channel_min_max'}:
        raise ValueError('Unknown preview normalization')
    if normalization in {'min_max', 'per_channel_min_max'}:
        axes = (1, 2) if normalization == 'per_channel_min_max' else (0, 1, 2)
        low, high = values.min(axis=axes, keepdims=True), values.max(axis=axes, keepdims=True)
        values = np.divide(values-low, high-low, out=np.zeros_like(values), where=high > low)
        signed = False
    elif normalization == 'max' or normalization == 'auto':
        scale = float(np.abs(values).max())
        if scale > 0 and (normalization == 'max' or scale > 1):
            values /= scale
    mode = cfg.get('mode', 'auto')
    if mode == 'auto':
        mode = 'polarity' if signed else 'rgb' if len(values) == 3 else 'polarity' if len(values) == 2 else 'mean' if len(values) == 1 else 'temporal_rgb'
    if mode == 'grid':
        columns = cfg.get('columns', min(4, len(values)))
        if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
            raise ValueError('Preview columns must be a positive integer')
        labels = cfg.get('labels', True)
        h, w = values.shape[1:]
        label_h = 16 if labels else 0
        canvas = Image.new('RGB', (columns*w, math.ceil(len(values)/columns)*(h+label_h)), 'black')
        draw = ImageDraw.Draw(canvas)
        for i, channel in enumerate(values):
            gray = (channel+1)/2 if signed else channel
            tile = Image.fromarray(np.rint(np.clip(gray, 0, 1)*255).astype(np.uint8)).convert('RGB')
            x, y = (i % columns)*w, (i // columns)*(h+label_h)
            canvas.paste(tile, (x, y+label_h))
            if labels:
                draw.text((x, y), f'C{indices[i]}', fill='white')
        return canvas
    if mode == 'rgb':
        if len(values) != 3:
            raise ValueError('RGB preview requires exactly three selected channels')
        rgb = (values+1)/2 if signed else values
    elif mode == 'temporal_rgb':
        # Every channel contributes; repeat the nearest channel for groups with no entries.
        groups = np.array_split(np.arange(len(values)), 3)
        rgb = np.stack([values[g].mean(axis=0) if len(g) else values[-1] for g in groups])
        if signed:
            rgb = (rgb+1)/2
    elif mode == 'polarity':
        if signed:
            positive, negative = np.maximum(values, 0).mean(axis=0), np.maximum(-values, 0).mean(axis=0)
        elif len(values) == 2:
            positive, negative = values[0], values[1]
        else:
            half = max(1, len(values)//2)
            positive, negative = values[:half].mean(axis=0), values[half:].mean(axis=0) if len(values) > half else values[0]*0
        rgb = np.stack([positive, np.zeros_like(positive), negative])
    elif mode in {'mean', 'max'}:
        gray = values.mean(axis=0) if mode == 'mean' else values.max(axis=0)
        gray = (gray+1)/2 if signed else gray
        rgb = np.stack([gray]*3)
    else:
        raise ValueError(f'Unknown event preview mode: {mode}')
    return Image.fromarray(np.rint(np.clip(rgb.transpose(1, 2, 0), 0, 1)*255).astype(np.uint8))


def representation_preview(tensor, representation, *, options=None, settings=None):
    base, _, _ = _resolve_representation_alias(representation)
    signed = base == 'voxel_grid' and not (options or {}).get('split_polarity', False)
    settings = dict(settings or {})
    if (options or {}).get('split_polarity', False) and settings.get('mode', 'auto') == 'auto':
        settings['mode'] = 'polarity'
    return event_tensor_preview(tensor, settings, signed=signed)
