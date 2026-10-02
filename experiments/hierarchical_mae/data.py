from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from ecfm.data.tokenizer import Region, build_patch
from experiments.region_worldmodel.data import load_events, partition_entries, read_entries


@dataclass
class Group:
    key: str
    level: int
    representation: str
    size: int
    channels: int
    start: int
    stop: int


class Layout:
    """Canonical order: level, representation, t, y, x; integer cell boundaries."""
    def __init__(self, cfg):
        self.groups, boxes, levels, planes = [], [], [], []
        self.maximum = cfg['hierarchy']['max_splits']
        self.representations = sorted({r for l in cfg['hierarchy']['levels'] for r in l['representations']})
        for lid, level in enumerate(cfg['hierarchy']['levels']):
            nx, ny, nt = level['splits']
            sx, sy, st = [m // n for m, n in zip(self.maximum, (nx, ny, nt))]
            for rep in level['representations']:
                start = len(boxes)
                for t in range(nt):
                    for y in range(ny):
                        for x in range(nx):
                            boxes.append([x*sx, y*sy, t*st, sx, sy, st])
                            levels.append(lid)
                            planes.append(self.representations.index(rep))
                self.groups.append(Group(f'l{lid}_{rep}', lid, rep, level['patch_size'],
                                         3 if rep.startswith('cstr') else 2, start, len(boxes)))
        self.boxes = torch.tensor(boxes, dtype=torch.long)
        self.level_ids = torch.tensor(levels)
        self.plane_ids = torch.tensor(planes)
        self.count = len(boxes)

    def descendants(self, token_ids):
        """[queries,N] mask of finer tokens inside selected coarse voxels, across planes.

        A coarse policy can use this to restrict its next selection stage without
        materializing a tree with ambiguous parents for alternate representations.
        """
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.boxes.device).reshape(-1)
        coarse = self.boxes[ids]
        contained = ((self.boxes[None, :, :3] >= coarse[:, None, :3]) &
                     (self.boxes[None, :, :3]+self.boxes[None, :, 3:] <= coarse[:, None, :3]+coarse[:, None, 3:])).all(-1)
        return contained & (self.level_ids[None] > self.level_ids[ids, None])


def render_cstr(events, region, size, max_count=None, include_count=True):
    """Float version of scripts.render_evt3_yolo_frames._cstr_patch (R=time+, G=count, B=time-)."""
    # Events are already cropped to this voxel; timestamps remain base-volume local.
    sums = np.zeros((2, region.dy, region.dx), dtype=np.float32)
    counts = np.zeros_like(sums)
    if len(events):
        x = (events[:, 0] - region.x).astype(np.int64)
        y = (events[:, 1] - region.y).astype(np.int64)
        p = (events[:, 3] > 0).astype(np.int64)
        time = np.clip((events[:, 2] - region.t) / region.dt, 0, 1)
        np.add.at(sums, (p, y, x), time)
        np.add.at(counts, (p, y, x), 1)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    total = counts.sum(0)
    count_channel = np.clip(total / max(float(total.max()) if max_count is None else max_count, 1e-6), 0, 1)
    patch = np.stack([means[1], count_channel if include_count else np.zeros_like(total), means[0]])
    return F.interpolate(torch.from_numpy(patch)[None], (size, size), mode='bilinear', align_corners=False)[0]


class HierarchyDataset(Dataset):
    paired = False

    def __init__(self, cfg, entries, training=False):
        self.cfg, self.entries, self.training = cfg, entries, training
        self.epoch = 0
        self.layout = Layout(cfg)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        d = self.cfg['data']
        rng = np.random.default_rng(np.random.SeedSequence([d.get('region_seed', 123), index,
                                                          self.epoch if self.training else 0]))
        path, label = self.entries[index]
        events, seconds = load_events(path, d['time_unit'])
        fraction = float(rng.uniform(*d['crop_fraction'])) if self.training else d.get('eval_fraction', 1.)
        start = float(rng.uniform(0, 1-fraction)) if self.training else (1-fraction)/2
        # Include the recording's last event, then use half-open voxel intervals.
        events = events[(events[:, 2] >= start) & (events[:, 2] <= start + fraction)].copy()
        events[:, 2] = np.minimum((events[:, 2] - start) / fraction, np.nextafter(np.float32(1), np.float32(0)))
        duration = seconds * fraction
        width, height = d['image_width'], d['image_height']
        mx, my, mt = self.layout.maximum
        patches, metadata, counts = {}, [], []
        for group in self.layout.groups:
            rendered = []
            for x, y, t, dx, dy, dt in self.layout.boxes[group.start:group.stop].tolist():
                x0, x1 = x * width // mx, (x+dx) * width // mx
                y0, y1 = y * height // my, (y+dy) * height // my
                r = Region(x0, y0, t/mt, x1-x0, y1-y0, dt/mt, group.representation)
                sub = events[(events[:, 0] >= x0) & (events[:, 0] < x1)
                             & (events[:, 1] >= y0) & (events[:, 1] < y1)
                             & (events[:, 2] >= r.t) & (events[:, 2] < r.t+r.dt)]
                counts.append(np.log1p(len(sub)))
                if group.representation.startswith('cstr'):
                    patch = render_cstr(sub, r, group.size,
                        d['cstr_max_count'] if group.representation == 'cstr3_fixed' else None,
                        group.representation != 'cstr2')
                else:
                    patch, _ = build_patch(sub, r, group.size, d['time_bins'], norm_mode=d['patch_norm'])
                rendered.append(patch)
                # Known geometry/time only. Event counts never enter masked queries.
                metadata.append([x0/width, y0/height, r.t, r.dx/width, r.dy/height, r.dt,
                                 np.log1p(r.t*duration), np.log1p(r.dt*duration), np.log1p(duration)])
            patches[group.key] = torch.stack(rendered)
        return dict(source=dict(patches=patches, metadata=torch.tensor(metadata, dtype=torch.float32),
                                log_counts=torch.tensor(counts, dtype=torch.float32),
                                valid_mask=torch.ones(self.layout.count, dtype=torch.bool)), label=label)


def splits(cfg, include_test=False):
    train, val = partition_entries(cfg)
    test = read_entries(Path(cfg['data']['root']), cfg['data'].get('test_split', 'test')) if include_test else []
    if {p.resolve() for p, _ in train+val} & {p.resolve() for p, _ in test}:
        raise ValueError('Train/validation overlap with test')
    if any(not 0 <= label < cfg['downstream']['num_classes'] for _, label in train+val+test):
        raise ValueError('Labels must be zero-based class indices')
    return train, val, test
