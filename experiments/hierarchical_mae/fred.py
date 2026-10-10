"""Unlabeled FRED windows for the existing hierarchical MAE training loop."""
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ecfm.data.fred_dataset import FREDDataset, read_fred_split
from .data import Layout
from .patch_cache import PatchCache


def fred_splits(cfg, include_test=False):
    d = cfg['data']
    root = Path(d['root'])
    train_path = Path(d.get('train_split_file', root/'dataset_splits/canonical/train_split.txt'))
    test_path = Path(d.get('test_split_file', root/'dataset_splits/canonical/test_split.txt'))
    train = read_fred_split(train_path)
    test = read_fred_split(test_path)
    if set(train) & set(test):
        raise ValueError('FRED train/test sequence overlap')
    if d.get('max_sequences', 0):
        train = train[:d['max_sequences']]
    validation_path = d.get('val_split_file')
    if validation_path:
        val = read_fred_split(validation_path)
        if set(train)&set(val) or set(test)&set(val):
            raise ValueError('FRED sequence splits must be disjoint')
    else:
        if len(train) < 2:
            raise ValueError('Need at least two FRED sequences for a validation holdout')
        np.random.default_rng(d.get('split_seed', 42)).shuffle(train)
        n = min(len(train)-1, max(1, round(len(train)*d.get('val_fraction', .15))))
        val, train = train[:n], train[n:]
    def entries(names):
        result = []
        for name in names:
            path = root/name/'Event/events.raw'
            if not path.is_file():
                raise FileNotFoundError(path)
            result.append((path, -1))
        return result
    return entries(train), entries(val), entries(test) if include_test else []


class FredHierarchyDataset(Dataset):
    paired = False

    def __init__(self, cfg, entries, training=False):
        self.cfg, self.entries, self.training, self.epoch = cfg, entries, training, 0
        d = cfg['data']
        self.layout = Layout(cfg)
        self.frames = FREDDataset(d['root'], sequences=[p.parent.parent.name for p, _ in entries],
                      frame_source='grid', frame_stride_us=d.get('frame_stride_us', 1000000),
                      modalities=(), max_samples=d.get('max_samples', 0),
                      subset_seed=d.get('region_seed', 123), event_cache_bytes=d.get('event_cache_bytes', 32*1024*1024))
        for info in self.frames.sequence_info.values():
            if (info['width'], info['height']) != (d['image_width'], d['image_height']):
                raise ValueError('FRED sensor geometry differs from configured image_width/image_height')
        options = d.get('patch_cache', {})
        self.train_views = options.get('train_views', 0)
        self.cache = PatchCache(cfg) if options.get('enabled', False) else None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        index, epoch = index if isinstance(index, tuple) else (index, self.epoch)
        d = self.cfg['data']
        crop_epoch = epoch % self.train_views if self.train_views else epoch
        rng = np.random.default_rng(np.random.SeedSequence([d.get('region_seed', 123), index,
                                                           crop_epoch if self.training else 0]))
        ref = self.frames.frame_ref(index)
        end = ref.time_us
        # Independent random windows remain bounded by a physical duration.
        if self.training:
            end = int(rng.integers(max(1, end-self.frames.frame_stride_us+1), end+1))
        durations = d.get('window_duration_s', [.033333, .4])
        duration = float(rng.uniform(*durations)) if self.training else d.get('eval_window_s', durations[-1])
        width, height = d['image_width'], d['image_height']
        x = y = 0
        if self.training:
            fractions = d.get('spatial_crop_fraction', [1., 1.])
            width, height = int(width*rng.uniform(*fractions)), int(height*rng.uniform(*fractions))
            x = int(rng.integers(0, d['image_width']-width+1))
            y = int(rng.integers(0, d['image_height']-height+1))
        use_cache = self.cache is not None and (not self.training or self.train_views > 0)
        if use_cache:
            path = self.frames.sequence_info[ref.sequence]['raw']
            stat = path.with_suffix('.raw.tmp_index').stat()
            cache_path, metadata = self.cache.entry(path, round(duration*1e6), end, [x, y, width, height],
                          extra=dict(window_us=[end-round(duration*1e6), end],
                                     index_stat=[stat.st_size, stat.st_mtime_ns]))
            view = self.cache.read(cache_path, metadata, self.layout)
            if view is not None:
                return dict(source=view, label=-1)
        view = self.frames.hierarchical_patches(ref.sequence, end, self.cfg, lookback_s=duration,
                                                spatial_crop=(x, y, width, height))
        if use_cache:
            self.cache.write(cache_path, metadata, view)
        return dict(source=view, label=-1)
