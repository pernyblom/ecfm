from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from ecfm.data.region_utils import sample_region, resolve_region_scales
from ecfm.data.tokenizer import build_patch
from .actions import parse_actions, transform_region
from .regions import downstream_layout


def read_entries(root: Path, split: str) -> list[tuple[Path, int]]:
    entries = []
    for line in (root / f'{split}.txt').read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        path = root / Path(parts[0].replace('\\', '/')).name
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append((path, int(parts[1]) if len(parts) > 1 else -1))
    if not entries:
        raise ValueError(f'Empty split: {split}')
    if len({p for p, _ in entries}) != len(entries):
        raise ValueError(f'Duplicate recordings in split: {split}')
    return entries


def partition_entries(cfg: dict):
    """Hold validation out of train; never select checkpoints on test."""
    d = cfg['data']
    root = Path(d['root'])
    entries = read_entries(root, d.get('train_split', 'train'))
    rng = np.random.default_rng(d.get('split_seed', 42))
    rng.shuffle(entries)
    if d.get('max_samples', 0):
        entries = entries[:d['max_samples']]
    fraction = d.get('val_fraction', 0.15)
    if not 0 < fraction < 1 or len(entries) < 4:
        raise ValueError('Need >=4 recordings and 0 < val_fraction < 1')
    count = min(len(entries) - 2, max(2, round(len(entries) * fraction)))
    return entries[count:], entries[:count]


def load_events(path: Path, time_unit: float):
    raw = np.load(path, allow_pickle=False)
    if raw.dtype.fields is not None:
        raw = np.column_stack([raw[k] for k in ('x', 'y', 't', 'p')])
    if raw.ndim != 2 or raw.shape[1] != 4 or len(raw) == 0:
        raise ValueError(f'Expected nonempty [N,4] events: {path}')
    # Subtract timestamps before converting to float32, preserving precision.
    raw = raw.astype(np.float64)
    if not np.isfinite(raw).all():
        raise ValueError(f'Nonfinite events: {path}')
    t = raw[:, 2]
    duration = max(float(t.max() - t.min()), 1e-6)
    raw[:, 2] = (t - t.min()) / duration
    raw[:, 3] = raw[:, 3] > 0
    return raw.astype(np.float32), duration * time_unit


class RegionActionDataset(Dataset):
    def __init__(self, cfg, entries, training=False, paired=True):
        self.cfg, self.entries = cfg, entries
        self.training, self.paired = training, paired
        self.epoch = 0
        self.actions = parse_actions(cfg['actions'])
        d = cfg['data']
        self.width, self.height = d['image_width'], d['image_height']
        self.scales = resolve_region_scales(
            d['region_scales'], d.get('region_scales_x', []),
            d.get('region_scales_y', []), self.width, self.height,
            d.get('region_scale_mode', 'fraction'))
        self.max_regions = max(d['num_regions_choices'])
        self.fixed_regions = downstream_layout(cfg) if not paired else None
        self.region_counts = d['num_regions_choices']
        if not paired:
            count = cfg.get('downstream', {}).get('regions', {}).get('num_regions')
            if self.fixed_regions is not None:
                self.max_regions = len(self.fixed_regions)
            elif count is not None:
                self.region_counts = [count]
                self.max_regions = count

    def __len__(self):
        # Validation enumerates every action on identical source regions.
        return len(self.entries) * (len(self.actions) if self.paired and not self.training else 1)

    def render(self, events, duration, regions):
        d, m = self.cfg['data'], self.cfg['model']
        patches = torch.zeros(self.max_regions, 2, m['patch_size'], m['patch_size'])
        metadata = torch.zeros(self.max_regions, 9)
        planes = torch.zeros(self.max_regions, dtype=torch.long)
        valid = torch.zeros(self.max_regions, dtype=torch.bool)
        for i, r in enumerate(regions):
            patches[i], _ = build_patch(events, r, m['patch_size'], d['time_bins'],
                                        norm_mode=d.get('patch_norm', 'region_max'))
            metadata[i] = torch.tensor([r.x / self.width, r.y / self.height,
                r.dx / self.width, r.dy / self.height, r.t, r.dt,
                r.t * duration, r.dt * duration, duration])
            if not d.get('include_absolute_duration', True):
                metadata[i, 6:] = 0
            planes[i] = d['plane_types'].index(r.plane)
            valid[i] = True
        return dict(patches=patches, metadata=metadata, plane_ids=planes, valid_mask=valid)

    def __getitem__(self, index):
        action_index = None
        if self.paired and not self.training:
            # Group by action so each validation batch spans recordings,
            # rather than estimating SIGReg on many views of one recording.
            action_index, index = divmod(index, len(self.entries))
        d = self.cfg['data']
        rng = np.random.default_rng(np.random.SeedSequence([
            d.get('region_seed', 123), index, self.epoch if self.training else 0]))
        path, label = self.entries[index]
        events, duration = load_events(path, d['time_unit'])
        limit = d.get('max_events', 0)
        if limit > 0 and len(events) > limit:
            events = events[rng.choice(len(events), limit, replace=False)]
        regions = self.fixed_regions
        if regions is None:
            count = int(rng.choice(self.region_counts))
            regions = [sample_region(rng, self.width, self.height, *self.scales,
                       d['region_time_scales'], d['plane_types'], False) for _ in range(count)]
        result = {'source': self.render(events, duration, regions), 'label': label}
        if self.paired:
            if action_index is None:
                action_index = int(rng.integers(len(self.actions)))
            action = self.actions[action_index]
            targets = [transform_region(r, action, self.width, self.height) for r in regions]
            result.update(target=self.render(events, duration, targets),
                          action=torch.from_numpy(action.vector()), action_id=action_index)
        return result
