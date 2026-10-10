"""FRED frames, causal event windows and observed tracklets.

Event time is shifted integer microseconds. Boxes use normalized cx,cy,w,h;
track-file x,y,w,h pixels are converted with an explicit coordinate-frame size.
RGB names use wall-clock timestamps and are aligned to the first frame (the
existing FRED renderer's convention). RGB matching is causal by default.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from ecfm.utils.evt3_index import IndexedRawReader, read_index
from ecfm.utils.evt3 import read_raw_header
from ecfm.utils.event_representations import render_event_representation

_LABEL_TIME = re.compile(r'(?:_frame_|_)(\d+)$')
_RGB_TIME = re.compile(r'_(\d{2})_(\d{2})_(\d{2})\.(\d+)$')


@dataclass(frozen=True)
class FredFrameRef:
    sequence: str
    time_us: int


def read_fred_split(path):
    names = [line.strip().strip('/\\') for line in Path(path).read_text().splitlines() if line.strip()]
    if any(not n.isdigit() for n in names) or len(set(names)) != len(names):
        raise ValueError('FRED splits must contain unique numeric sequence folders')
    return names


def read_yolo(path):
    if path is None or not Path(path).is_file():
        return dict(boxes=torch.empty(0, 4), classes=torch.empty(0, dtype=torch.long), available=False, path=None)
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if (len(values) != 5 or not np.isfinite(values).all() or values[3] <= 0 or values[4] <= 0
                or values[0] < 0 or not values[0].is_integer()):
            raise ValueError(f'Invalid YOLO annotation in {path}: {line}')
        rows.append(values)
    return dict(boxes=torch.tensor([r[1:] for r in rows], dtype=torch.float32).reshape(-1, 4),
                classes=torch.tensor([int(r[0]) for r in rows], dtype=torch.long),
                available=True, path=str(path))


class FREDDataset(Dataset):
    def __init__(self, root, split=None, *, sequences=None, split_file=None,
                 frame_source='grid', frame_stride_us=33333, event_window_s=0.033333,
                 modalities=('events', 'event_boxes'), representations=(), image_sizes=None,
                 temporal_bins=224, cstr_max_count=None, timestamp_mode='integer',
                 tracks_file='tracks.txt', track_time_unit=1., track_frame_size=None,
                 require_original_tracks=True, match_tolerance_us=32,
                 rgb_match='previous', rgb_tolerance_us=50000,
                 reader_cache_size=2, event_cache_bytes=32*1024*1024,
                 max_samples=0, subset_seed=0, missing_raw='error'):
        self.root = Path(root)
        if sequences is not None and (split is not None or split_file is not None):
            raise ValueError('Specify sequences or a split, not both')
        if split_file is None and split is not None:
            split_file = self.root/'dataset_splits/canonical'/f'{split}_split.txt'
        names = ([str(n) for n in sequences] if sequences is not None else read_fred_split(split_file)
                 if split_file is not None else sorted((p.name for p in self.root.iterdir()
                         if p.is_dir() and p.name.isdigit() and int(p.name) < 900), key=int))
        if len(set(names)) != len(names) or any(not n.isdigit() for n in names):
            raise ValueError('Need unique numeric FRED sequence IDs')
        if frame_source not in {'grid', 'event_labels'} or frame_stride_us < 1 or event_window_s <= 0:
            raise ValueError('Invalid frame source, stride or event window')
        if reader_cache_size < 1 or event_cache_bytes < 0 or missing_raw not in {'error', 'skip'}:
            raise ValueError('Invalid reader cache, event cache or missing_raw policy')
        if rgb_match not in {'previous', 'nearest'} or rgb_tolerance_us < 0 or match_tolerance_us < 0:
            raise ValueError('Invalid timestamp matching settings')
        allowed = {'events', 'event_boxes', 'rgb', 'padded_rgb', 'rgb_boxes'}
        if set(modalities)-allowed:
            raise ValueError(f'Unknown modalities: {set(modalities)-allowed}')
        self.frame_source, self.frame_stride_us = frame_source, int(frame_stride_us)
        self.event_window_us = max(1, round(event_window_s*1e6))
        self.modalities, self.representations = tuple(modalities), tuple(representations)
        self.image_sizes, self.temporal_bins = image_sizes or {}, temporal_bins
        self.cstr_max_count, self.timestamp_mode = cstr_max_count, timestamp_mode
        self.tracks_file, self.track_time_unit = tracks_file, track_time_unit
        self.track_frame_size, self.require_original_tracks = track_frame_size, require_original_tracks
        self.match_tolerance_us = match_tolerance_us
        self.rgb_match, self.rgb_tolerance_us = rgb_match, rgb_tolerance_us
        self.reader_cache_size, self.event_cache_bytes = reader_cache_size, event_cache_bytes
        self._readers, self._events = OrderedDict(), OrderedDict()
        self._event_bytes = 0
        self._labels, self._rgb, self._tracks = {}, {}, {}
        self._label_times, self._rgb_times = {}, {}
        self._hierarchies = OrderedDict()
        self.sequence_info, self.excluded = {}, {}
        self._counts = []
        for name in names:
            folder = self.root/name
            raw = folder/'Event/events.raw'
            if not raw.is_file():
                if missing_raw == 'error':
                    raise FileNotFoundError(raw)
                self.excluded[name] = 'missing RAW'
                continue
            records, shift, _, _ = read_index(raw)
            meta = read_raw_header(raw)[2]
            end = int(records[-1]['t'])
            info = dict(raw=raw, width=int(meta['width']), height=int(meta['height']),
                        end_us=end, shift_us=shift,
                        event_labels=(folder/'Event_YOLO').is_dir(),
                        rgb_labels=(folder/'RGB_YOLO').is_dir(),
                        original_tracks=(folder/'tracks.txt').is_file(),
                        selected_tracks=(folder/tracks_file).is_file(),
                        generated_tracks=tracks_file != 'tracks.txt')
            self.sequence_info[name] = info
            count = end//self.frame_stride_us if frame_source == 'grid' else len(self._event_labels(name))
            self._counts.append(count)
        self.sequences = list(self.sequence_info)
        self._cumulative = np.cumsum(self._counts)
        total = int(self._cumulative[-1]) if len(self._cumulative) else 0
        if not total:
            raise ValueError('No eligible FRED frames; inspect excluded/annotation availability')
        self._subset = (np.sort(np.random.default_rng(subset_seed).choice(total, min(max_samples, total), replace=False))
                        if max_samples > 0 else None)

    def __len__(self):
        return len(self._subset) if self._subset is not None else int(self._cumulative[-1])

    def frame_ref(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        index = int(self._subset[index]) if self._subset is not None else index
        sequence_index = int(np.searchsorted(self._cumulative, index, side='right'))
        local = index-(int(self._cumulative[sequence_index-1]) if sequence_index else 0)
        name = self.sequences[sequence_index]
        time_us = (local+1)*self.frame_stride_us if self.frame_source == 'grid' else self._label_times[name][local]
        return FredFrameRef(name, int(time_us))

    def _event_labels(self, sequence):
        if sequence not in self._labels:
            labels = {}
            for path in (self.root/sequence/'Event_YOLO').glob('*.txt'):
                match = _LABEL_TIME.search(path.stem)
                if match:
                    t = int(match.group(1))
                    if t in labels:
                        raise ValueError(f'Duplicate event label timestamp: {sequence}/{t}')
                    labels[t] = path
            self._labels[sequence] = dict(sorted(labels.items()))
            self._label_times[sequence] = np.array(list(self._labels[sequence]), dtype=np.int64)
        return self._labels[sequence]

    def reader(self, sequence):
        sequence = str(sequence)
        if sequence in self._readers:
            reader = self._readers.pop(sequence)
        else:
            reader = IndexedRawReader(self.sequence_info[sequence]['raw'], timestamp_mode=self.timestamp_mode)
        self._readers[sequence] = reader
        while len(self._readers) > self.reader_cache_size:
            self._readers.popitem(last=False)
        return reader

    def events(self, sequence, start_us, end_us):
        key = (str(sequence), int(start_us), int(end_us))
        if key in self._events:
            value = self._events.pop(key)
            self._events[key] = value
            return value.copy()
        value = self.reader(key[0]).read_window(key[1], key[2])
        if value.nbytes <= self.event_cache_bytes and self.event_cache_bytes:
            self._events[key] = value
            self._event_bytes += value.nbytes
            while self._event_bytes > self.event_cache_bytes or len(self._events) > 128:
                _, old = self._events.popitem(last=False)
                self._event_bytes -= old.nbytes
        return value.copy()

    def representations_at(self, sequence, time_us, representations=None, *, lookback_s=None):
        duration = self.event_window_us if lookback_s is None else round(lookback_s*1e6)
        if duration < 1:
            raise ValueError('lookback_s must be positive')
        start, end = int(time_us)-duration, int(time_us)
        requested = tuple(self.representations if representations is None else representations)
        events = self.events(sequence, start, end) if any(rep not in {'rgb', 'padded_rgb'} for rep in requested) else None
        info = self.sequence_info[str(sequence)]
        tensors = {}
        for rep in requested:
            if rep in {'rgb', 'padded_rgb'}:
                match = self.rgb_at(sequence, time_us, rep)
                if match is None:
                    raise FileNotFoundError(f'No causal {rep} within timestamp tolerance: {sequence}/{time_us}')
                image = match['tensor']
                if rep in self.image_sizes:
                    w, h = self.image_sizes[rep]
                    image = torch.nn.functional.interpolate(image[None], (h, w), mode='bilinear', align_corners=False)[0]
                tensors[rep] = image
                continue
            pixels = render_event_representation(events, rep, width=info['width'], height=info['height'],
                        start_us=start, end_us=end, output_size=self.image_sizes.get(rep),
                        time_bins=self.temporal_bins, cstr_max_count=self.cstr_max_count)
            tensors[rep] = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float()/255
        return tensors

    def hierarchical_patches(self, sequence, time_us, cfg, *, lookback_s, spatial_crop=None):
        from experiments.hierarchical_mae.data import Layout, render_hierarchy
        from experiments.hierarchical_mae.information_selection import settings
        duration = round(lookback_s*1e6)
        if duration < 1:
            raise ValueError('lookback_s must be positive')
        end, start = int(time_us), int(time_us)-duration
        info = self.sequence_info[str(sequence)]
        width, height = info['width'], info['height']
        x, y, w, h = spatial_crop or (0, 0, width, height)
        if not 0 <= x < x+w <= width or not 0 <= y < y+h <= height:
            raise ValueError('Spatial crop must lie inside sensor geometry')
        if w < cfg['hierarchy']['max_splits'][0] or h < cfg['hierarchy']['max_splits'][1]:
            raise ValueError('Spatial crop too small for hierarchy')
        events = self.events(sequence, start, end)
        events = events[(events[:, 0]>=x)&(events[:, 0]<x+w)&(events[:, 1]>=y)&(events[:, 1]<y+h)]
        local = events.astype(np.float32)
        local[:, 0] -= x
        local[:, 1] -= y
        local[:, 2] = np.minimum((events[:, 2]-start).astype(np.float64)/duration,
                                np.nextafter(np.float32(1), np.float32(0)))
        key = json.dumps(cfg['hierarchy'], sort_keys=True)
        if key not in self._hierarchies:
            self._hierarchies[key] = Layout(cfg)
            while len(self._hierarchies) > 8:
                self._hierarchies.popitem(last=False)
        return render_hierarchy(cfg, self._hierarchies[key], local, duration/1e6, w, h, settings(cfg))

    def _rgb_index(self, sequence, kind):
        key = (sequence, kind)
        if key not in self._rgb:
            folder = self.root/sequence/('RGB' if kind == 'rgb' else 'PADDED_RGB')
            parsed = []
            for path in folder.iterdir() if folder.is_dir() else []:
                if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                    continue
                match = _RGB_TIME.search(path.stem)
                if match:
                    hh, mm, ss, frac = match.groups()
                    t = (int(hh)*3600+int(mm)*60+int(ss))*1000000+int(frac.ljust(6, '0')[:6])
                    parsed.append((t, path))
            parsed.sort()
            base = parsed[0][0] if parsed else 0
            self._rgb[key] = [(t-base, p) for t, p in parsed]
            self._rgb_times[key] = np.array([t-base for t, _ in parsed], dtype=np.int64)
        return self._rgb[key]

    def rgb_at(self, sequence, time_us, kind='rgb', *, load=True):
        if kind not in {'rgb', 'padded_rgb'}:
            raise ValueError('RGB kind must be rgb or padded_rgb')
        index = self._rgb_index(str(sequence), kind)
        if not index:
            return None
        times = self._rgb_times[(str(sequence), kind)]
        i = int(np.searchsorted(times, time_us, side='right'))-1
        if self.rgb_match == 'nearest':
            i = min(range(max(0, i), min(len(index), i+2)), key=lambda j: abs(times[j]-time_us))
        if i < 0 or abs(int(times[i])-time_us) > self.rgb_tolerance_us:
            return None
        t, path = index[i]
        result = dict(path=str(path), time_us=int(t), offset_us=int(t-time_us), tensor=None)
        if load:
            from PIL import Image
            with Image.open(path) as image:
                image = image.convert('RGB')
                result['size'] = image.size
                result['tensor'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255
        return result

    def get_frame(self, sequence, time_us, *, modalities=None):
        sequence, time_us = str(sequence), int(time_us)
        if time_us < 0:
            raise ValueError('Frame timestamp must be nonnegative')
        info = self.sequence_info[sequence]
        chosen = self.modalities if modalities is None else modalities
        result = dict(sequence=sequence, time_us=time_us, time_s=time_us/1e6,
                      sensor_size=(info['width'], info['height']), window_start_us=time_us-self.event_window_us,
                      window_end_us=time_us, inputs={})
        if 'events' in chosen:
            result['events'] = self.events(sequence, time_us-self.event_window_us, time_us)
        if 'event_boxes' in chosen:
            result['event_boxes'] = read_yolo(self._event_labels(sequence).get(time_us))
        for kind in ('rgb', 'padded_rgb'):
            if kind in chosen:
                result[kind] = self.rgb_at(sequence, time_us, kind)
        if 'rgb_boxes' in chosen:
            matched = result.get('rgb') or self.rgb_at(sequence, time_us, load=False)
            label = self.root/sequence/'RGB_YOLO'/(Path(matched['path']).stem+'.txt') if matched else None
            result['rgb_boxes'] = read_yolo(label)
            result['rgb_boxes']['coordinate_frame'] = 'RGB'
        if self.representations:
            result['inputs'] = self.representations_at(sequence, time_us)
        return result

    def __getitem__(self, index):
        ref = self.frame_ref(index)
        return self.get_frame(ref.sequence, ref.time_us)

    def tracks(self, sequence):
        sequence = str(sequence)
        if sequence not in self._tracks:
            info = self.sequence_info[sequence]
            if self.require_original_tracks and not info['original_tracks']:
                raise FileNotFoundError(f'Original tracks.txt required for supervised tracklets: {sequence}')
            path = self.root/sequence/self.tracks_file
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = {}
            width, height = self.track_frame_size or (info['width'], info['height'])
            if width <= 0 or height <= 0:
                raise ValueError('track_frame_size dimensions must be positive')
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                t, tid, x, y, w, h = map(float, line.split(',')[:6])
                if (not np.isfinite([t, tid, x, y, w, h]).all() or w <= 0 or h <= 0
                        or not tid.is_integer()):
                    raise ValueError(f'Invalid track annotation: {path}')
                rows.setdefault(int(tid), []).append((round(t*self.track_time_unit*1e6),
                                (x+w/2)/width, (y+h/2)/height, w/width, h/height))
            tracks = {}
            for tid, values in rows.items():
                values.sort()
                array = np.array(values, dtype=np.float64)
                if np.any(np.diff(array[:, 0]) <= 0):
                    raise ValueError(f'Duplicate/nonincreasing track time: {path}, ID {tid}')
                tracks[tid] = array
            self._tracks[sequence] = tracks
        return self._tracks[sequence]

    def get_tracklet(self, sequence, time_us, track_id, *, history_steps, future_steps,
                     step_us=None, representations=(), representation_window_s=None,
                     hierarchy_cfg=None, hierarchy_lookback_s=None):
        if (not isinstance(history_steps, (int, np.integer)) or not isinstance(future_steps, (int, np.integer))
                or history_steps < 0 or future_steps < 0):
            raise ValueError('Tracklet step counts must be nonnegative integers')
        step_us = self.frame_stride_us if step_us is None else int(step_us)
        if step_us < 1:
            raise ValueError('step_us must be positive')
        offsets = np.arange(-history_steps, future_steps+1, dtype=np.int64)*step_us
        query = int(time_us)+offsets
        rows = self.tracks(sequence)[int(track_id)]
        times = rows[:, 0].astype(np.int64)
        right = np.searchsorted(times, query)
        right = np.clip(right, 0, len(times)-1)
        left = np.maximum(right-1, 0)
        ids = np.where(np.abs(times[left]-query) <= np.abs(times[right]-query), left, right)
        if np.any(np.abs(times[ids]-query) > self.match_tolerance_us) or np.any(np.diff(ids) <= 0):
            raise ValueError('Incomplete tracklet: missing observation or track gap')
        boxes = torch.from_numpy(rows[ids, 1:].astype(np.float32))
        n = history_steps+1
        result = dict(sequence=str(sequence), track_id=int(track_id), anchor_time_us=int(time_us),
                      times_s=torch.from_numpy(offsets.astype(np.float32)/1e6),
                      observed_times_us=torch.from_numpy(times[ids].copy()), boxes=boxes,
                      past_boxes=boxes[:n], future_boxes=boxes[n:],
                      past_times_s=torch.from_numpy(offsets[:n].astype(np.float32)/1e6),
                      future_times_s=torch.from_numpy(offsets[n:].astype(np.float32)/1e6),
                      tracks_source=str(self.root/str(sequence)/self.tracks_file), inputs={})
        if representations:
            history = [self.representations_at(sequence, int(t), representations,
                       lookback_s=representation_window_s) for t in query[:n]]
            result['history_inputs'] = {rep: torch.stack([view[rep] for view in history]) for rep in representations}
        if hierarchy_cfg is not None:
            if hierarchy_lookback_s is None:
                raise ValueError('hierarchy_lookback_s is required')
            result['inputs']['hierarchy'] = self.hierarchical_patches(sequence, time_us, hierarchy_cfg,
                                                                     lookback_s=hierarchy_lookback_s)
        return result

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_readers'], state['_events'], state['_event_bytes'] = OrderedDict(), OrderedDict(), 0
        return state


def collate_fred_frames(frames):
    """Keep variable-length events, boxes and optional modalities as frame records."""
    return frames
