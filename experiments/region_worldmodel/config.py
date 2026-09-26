from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import yaml

from .actions import parse_actions
from .regions import downstream_layout


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop('extends', None)
    if parent:
        base = load_config(path.parent / parent)
        def merge(a, b):
            for key, value in b.items():
                if isinstance(value, dict) and isinstance(a.get(key), dict):
                    merge(a[key], value)
                else:
                    a[key] = deepcopy(value)
        merge(base, cfg)
        cfg = base
    validate(cfg)
    return cfg


def validate_downstream(cfg):
    downstream_layout(cfg)
    settings = cfg.get('downstream', {})
    for key in ('epochs', 'batch_size'):
        if key in settings and (type(settings[key]) is not int or settings[key] < 1):
            raise ValueError(f'downstream.{key} must be a positive integer')
    if type(settings.get('blank_diagnostic', True)) is not bool:
        raise ValueError('blank_diagnostic must be boolean')
    cache = settings.get('feature_cache', {})
    if not isinstance(cache, dict) or set(cache) - {'enabled', 'dir', 'batch_size', 'rebuild', 'storage'}:
        raise ValueError('Invalid downstream.feature_cache settings')
    if cache.get('storage', 'disk') not in ('disk', 'memory', 'temporary'):
        raise ValueError('feature_cache.storage must be disk, memory or temporary')
    for key in ('enabled', 'rebuild'):
        if key in cache and type(cache[key]) is not bool:
            raise ValueError(f'feature_cache.{key} must be boolean')
    if 'batch_size' in cache and (type(cache['batch_size']) is not int or cache['batch_size'] < 1):
        raise ValueError('feature_cache.batch_size must be a positive integer')
    if 'dir' in cache and (not isinstance(cache['dir'], str) or not cache['dir'].strip()):
        raise ValueError('feature_cache.dir must be a nonempty path')


def validate(cfg):
    parse_actions(cfg['actions'])
    validate_downstream(cfg)
    d, m, t, loss = (cfg[k] for k in ('data', 'model', 'train', 'loss'))
    if type(d.get('include_absolute_duration', True)) is not bool:
        raise ValueError('data.include_absolute_duration must be boolean')
    probe = t.get('linear_probe', {})
    allowed_probe = {'every', 'epochs', 'batch_size', 'lr', 'weight_decay', 'regions',
                     'feature_cache', 'blank_diagnostic', 'max_batches'}
    if not isinstance(probe, dict) or set(probe) - allowed_probe:
        raise ValueError('Invalid train.linear_probe settings')
    if type(probe.get('every', 0)) is not int or probe.get('every', 0) < 0:
        raise ValueError('train.linear_probe.every must be a nonnegative integer')
    probe_cfg = deepcopy(cfg)
    for key, value in probe.items():
        if key != 'every':
            if key == 'feature_cache':
                if not isinstance(value, dict):
                    raise ValueError('linear_probe.feature_cache must be a mapping')
                probe_cfg['downstream'].setdefault(key, {}).update(value)
            else:
                probe_cfg['downstream'][key] = value
    validate_downstream(probe_cfg)
    for key in ('image_width', 'image_height', 'time_bins'):
        if not isinstance(d[key], int) or d[key] < 1:
            raise ValueError(f'data.{key} must be a positive integer')
    counts = d['num_regions_choices']
    if not counts or any(not isinstance(v, int) or v < 1 for v in counts):
        raise ValueError('num_regions_choices must contain positive integers')
    if not d['plane_types'] or len(set(d['plane_types'])) != len(d['plane_types']):
        raise ValueError('plane_types must be nonempty and unique')
    allowed = {'xy', 'xt', 'yt', 'xy_m45', 'xy_p45', 'yt_m45', 'yt_p45'}
    if not set(d['plane_types']) <= allowed:
        raise ValueError('Unsupported projection plane')
    if d.get('region_scale_mode', 'fraction') not in ('fraction', 'absolute'):
        raise ValueError('region_scale_mode must be fraction or absolute')
    for key in ('region_scales', 'region_time_scales', 'region_scales_x', 'region_scales_y'):
        values = d.get(key, [])
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f'{key} must contain finite positive values')
    if not d['region_time_scales'] or not (d['region_scales'] or (d.get('region_scales_x') and d.get('region_scales_y'))):
        raise ValueError('Specify spatial and temporal scales')
    if d['time_unit'] <= 0 or d.get('max_events', 0) < 0 or d.get('max_samples', 0) < 0:
        raise ValueError('Invalid time_unit or dataset limits')
    if not 0 <= t['mask_ratio'] < 1:
        raise ValueError('mask_ratio must be in [0,1)')
    if t['batch_size'] < 2 or t['epochs'] < 1:
        raise ValueError('SSL needs batch_size >= 2 and epochs >= 1')
    if m['embed_dim'] % m['num_heads'] or m.get('decoder_embed_dim', m['embed_dim']) % m['num_heads']:
        raise ValueError('Encoder/decoder dimensions must be divisible by num_heads')
    if loss['regularizer_weight'] <= 0 or loss['projections'] < 1 or loss['integration_steps'] < 2:
        raise ValueError('Invalid anti-collapse settings')
    if loss.get('reconstruction_weight', 0) < 0:
        raise ValueError('reconstruction_weight must be nonnegative')
    if loss.get('reconstruction_weight', 0) > 0 and t['mask_ratio'] == 0:
        raise ValueError('Reconstruction requires a nonzero mask_ratio')
