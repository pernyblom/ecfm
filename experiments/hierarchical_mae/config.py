from copy import deepcopy
import math
from pathlib import Path

import yaml

REPRESENTATIONS = ('xy', 'xt', 'yt', 'xy_p45', 'xy_m45', 'yt_p45', 'yt_m45',
                   'cstr2', 'cstr3', 'cstr3_fixed')


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


def validate(cfg):
    d, h, m = (cfg[k] for k in ('data', 'hierarchy', 'model'))
    maximum = h['max_splits']
    if len(maximum) != 3 or any(type(v) is not int or v < 1 for v in maximum):
        raise ValueError('max_splits must be three positive cell counts [x,y,t]')
    if maximum[0] > d['image_width'] or maximum[1] > d['image_height']:
        raise ValueError('Smallest spatial cells must contain at least one pixel')
    spatial = d.get('spatial_crop_fraction', [1., 1.])
    if (not isinstance(spatial, (list, tuple)) or len(spatial) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in spatial)
            or not 0 < spatial[0] <= spatial[1] <= 1):
        raise ValueError('spatial_crop_fraction must be [min, max] with 0 < min <= max <= 1')
    if any(math.floor(d[key] * spatial[0]) < maximum[axis]
           for axis, key in enumerate(('image_width', 'image_height'))):
        raise ValueError('spatial_crop_fraction is too small for max_splits: each cell needs at least one pixel')
    previous = [1, 1, 1]
    if not h['levels']:
        raise ValueError('Need at least one hierarchy level')
    for level in h['levels']:
        splits = level['splits']
        if (len(splits) != 3 or any(type(v) is not int or v < 1 for v in splits)
                or any(limit % v or v % p for limit, v, p in zip(maximum, splits, previous))):
            raise ValueError('Levels must nest, and their splits must divide max_splits')
        previous = splits
        if type(level['patch_size']) is not int or level['patch_size'] < 1:
            raise ValueError('patch_size must be a positive integer')
        reps = level['representations']
        if not reps or len(set(reps)) != len(reps) or set(reps) - set(REPRESENTATIONS):
            raise ValueError(f'Invalid representations: {reps}')
        if 'cstr3_fixed' in reps and d.get('cstr_max_count', 0) <= 0:
            raise ValueError('cstr3_fixed requires data.cstr_max_count > 0')
    lo, hi = d['crop_fraction']
    if not 0 < lo <= hi <= 1 or not 0 < d.get('eval_fraction', 1) <= 1:
        raise ValueError('Crop fractions must be in (0,1]')
    if d['time_unit'] <= 0 or d['time_bins'] < 1:
        raise ValueError('time_unit and time_bins must be positive')
    cache = d.get('patch_cache', {})
    if not isinstance(cache, dict) or set(cache)-{'enabled', 'dir', 'train_views', 'rebuild'}:
        raise ValueError('Invalid data.patch_cache settings')
    if type(cache.get('train_views', 0)) is not int or cache.get('train_views', 0) < 0:
        raise ValueError('patch_cache.train_views must be a nonnegative integer')
    for key in ('enabled', 'rebuild'):
        if type(cache.get(key, False)) is not bool:
            raise ValueError(f'patch_cache.{key} must be boolean')
    if 'dir' in cache and (not isinstance(cache['dir'], str) or not cache['dir'].strip()):
        raise ValueError('patch_cache.dir must be a nonempty path')
    if type(cfg['train']['num_workers']) is not int or cfg['train']['num_workers'] < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    for width in ('embed_dim', 'decoder_dim'):
        if m[width] < 1 or m[width] % m['num_heads']:
            raise ValueError('Transformer widths must be positive multiples of num_heads')
    mask = cfg['masking']
    if mask['strategy'] not in ('token', 'voxel', 'subtree', 'mixed') or not 0 < mask['ratio'] < 1:
        raise ValueError('Mask strategy must be token/voxel/subtree/mixed with ratio in (0,1)')
    if not 0 <= mask.get('overlap_probability', .5) <= 1:
        raise ValueError('overlap_probability must be in [0,1]')
    level = mask.get('level', len(h['levels']) - 1)
    if not 0 <= level < len(h['levels']):
        raise ValueError('Invalid masking level')
    if mask['strategy'] != 'token':
        if math.prod(h['levels'][level]['splits']) < 2:
            raise ValueError('Voxel/subtree masking needs at least two cells at its level')
    for settings in (cfg['train'], cfg['downstream']):
        for key in ('epochs', 'batch_size'):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f'{key} must be positive')
    for selection in (cfg.get('selection', {}), cfg['downstream'].get('selection', {})):
        if selection.get('strategy', 'all') not in ('all', 'random', 'activity', 'activity_random', 'coarse'):
            raise ValueError('Selection strategy must be all/random/activity/activity_random/coarse')
        if type(selection.get('budget', 0)) is not int or selection.get('budget', 0) < 0:
            raise ValueError('Selection budget must be a nonnegative integer; 0 means unlimited')
        power = selection.get('activity_power', 1.)
        if type(power) not in (int, float) or not math.isfinite(power) or power < 0:
            raise ValueError('activity_power must be finite and nonnegative')
