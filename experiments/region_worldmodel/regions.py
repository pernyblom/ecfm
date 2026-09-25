"""Deterministic downstream layouts; counts include projection-plane tokens."""
from ecfm.data.region_utils import grid_regions


def downstream_layout(cfg):
    """Return a fixed layout, or None for the existing random sampler."""
    spec = cfg.get('downstream', {}).get('regions', {})
    mode = spec.get('mode', 'random')
    if mode not in ('random', 'grid', 'multiscale'):
        raise ValueError('downstream.regions.mode must be random, grid or multiscale')
    allowed = {'mode', 'num_regions'}
    if mode != 'random':
        allowed |= {'plane_mode', 'grid' if mode == 'grid' else 'levels'}
    if set(spec) - allowed:
        raise ValueError(f'Unsupported downstream region settings: {set(spec) - allowed}')
    count = spec.get('num_regions')
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError('downstream.regions.num_regions must be a positive integer')
    if mode == 'random':
        return None
    d = cfg['data']
    plane_mode = spec.get('plane_mode', 'all')
    if plane_mode not in ('all', 'cycle'):
        raise ValueError('downstream.regions.plane_mode must be all or cycle')
    levels = [spec.get('grid')] if mode == 'grid' else spec.get('levels')
    if not isinstance(levels, list) or not levels:
        raise ValueError('Specify a grid or nonempty multiscale levels')
    regions, seen = [], set()
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) != 3 or any(type(v) is not int or v < 1 for v in level):
            raise ValueError('Each grid must contain three positive integers [x, y, t]')
        if tuple(level) in seen:
            raise ValueError('Multiscale levels must be distinct')
        seen.add(tuple(level))
        x, y, t = level
        if x > d['image_width'] or y > d['image_height']:
            raise ValueError('Grid subdivisions cannot exceed spatial pixel dimensions')
        regions.extend(grid_regions(d['image_width'], d['image_height'],
                                    x, y, t, plane_mode, d['plane_types']))
    if count is not None and count != len(regions):
        raise ValueError(f'Layout produces {len(regions)} tokens, but num_regions={count}')
    return regions
