"""Shared RAW FRED inputs with existing detection/forecast trainer sample contracts."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ecfm.data.fred_dataset import FREDDataset, read_yolo
from .backbone import backbone_config


def task_source(cfg, folders):
    d = cfg['data']
    backbone = cfg['model'].get('backbone', {})
    if d.get('image_window_mode', 'trailing') != 'trailing':
        raise ValueError('RAW task inputs require causal trailing windows')
    if d.get('spatial_cutout', {}).get('mode', 'none') != 'none':
        raise ValueError('RAW adapter currently requires spatial_cutout.mode: none')
    if d.get('box_augmentation', {}).get('enabled') or d.get('decorrelation', {}).get('enabled'):
        raise ValueError('RAW adapter does not yet implement box augmentation/decorrelation')
    hierarchical = backbone.get('type') == 'hierarchical_mae'
    if hierarchical and d['representations'] != ['hierarchy']:
        raise ValueError('hierarchical_mae task backbone requires data.representations: [hierarchy]')
    hierarchy = backbone_config(backbone) if hierarchical else None
    source = FREDDataset(d.get('root', d.get('labels_root', 'datasets/FRED')),
                sequences=folders, frame_source='event_labels', modalities=(),
                frame_stride_us=d.get('frame_stride_us', 33333),
                event_window_s=float(d.get('image_window_ms', 33.333))/1000,
                image_sizes=d.get('image_sizes'), temporal_bins=d.get('temporal_bins', 224),
                cstr_max_count=d.get('cstr_max_count'), tracks_file=d.get('tracks_file', 'tracks.txt'),
                track_time_unit=d.get('track_time_unit', 1.), track_frame_size=d.get('track_frame_size'),
                require_original_tracks=True, match_tolerance_us=d.get('match_tolerance_us', 32),
                rgb_match='previous', rgb_tolerance_us=d.get('rgb_tolerance_us', 50000),
                event_cache_bytes=d.get('event_cache_bytes', 32*1024*1024), missing_raw='skip')
    expected_size = tuple(d.get('frame_size', (1280, 720)))
    if any((info['width'], info['height']) != expected_size for info in source.sequence_info.values()):
        raise ValueError('Task frame_size must match FRED sensor geometry')
    return source, hierarchy


def task_inputs(source, cfg, hierarchy, sequence, time_us):
    d = cfg['data']
    if hierarchy is not None:
        return {'hierarchy': source.hierarchical_patches(sequence, time_us, hierarchy,
                                     lookback_s=d.get('hierarchy_lookback_s', .4))}
    inputs = {}
    from experiments.kalman_ml_forecasting.utils.config import resolve_representation_sequences
    sequences = resolve_representation_sequences(d)
    for rep in d['representations']:
        spec = sequences.get(rep)
        anchors = ([time_us-(spec.get('length', 1)-1-i)*spec.get('stride', 1)*source.frame_stride_us
                    for i in range(spec.get('length', 1))] if spec else [time_us])
        values = []
        for anchor in anchors:
            if rep in {'rgb', 'padded_rgb'}:
                match = source.rgb_at(sequence, anchor, rep)
                if match is None:
                    raise FileNotFoundError(f'No causal RGB within tolerance: {sequence}/{anchor}/{rep}')
                image = match['tensor']
                if rep in source.image_sizes:
                    w, h = source.image_sizes[rep]
                    image = torch.nn.functional.interpolate(image[None], (h, w), mode='bilinear', align_corners=False)[0]
                values.append(image)
            else:
                values.append(source.representations_at(sequence, anchor, [rep])[rep])
        inputs[rep] = torch.stack(values) if spec else values[0]
    return inputs


def rgb_inputs_available(source, cfg, sequence, time_us):
    """Check optional RGB input eligibility without loading pixels or events."""
    if not any(rep in {'rgb', 'padded_rgb'} for rep in cfg['data']['representations']):
        return True
    from experiments.kalman_ml_forecasting.utils.config import resolve_representation_sequences
    sequences = resolve_representation_sequences(cfg['data'])
    for rep in cfg['data']['representations']:
        if rep not in {'rgb', 'padded_rgb'}:
            continue
        spec = sequences.get(rep, {'length': 1, 'stride': 1})
        for i in range(spec['length']):
            anchor = time_us-i*spec['stride']*source.frame_stride_us
            if source.rgb_at(sequence, anchor, rep, load=False) is None:
                return False
    return True


class RawFredDetectionDataset(Dataset):
    def __init__(self, cfg, folders, *, max_samples=None, seed=123):
        self.cfg = cfg
        self.source, self.hierarchy = task_source(cfg, folders)
        if cfg['model'].get('predict_velocity', False):
            raise ValueError('RAW CenterNet adapter currently requires predict_velocity: false')
        if cfg['data'].get('heatmap_representations'):
            raise ValueError('RAW detection adapter uses CenterNet XY supervision; set heatmap_representations: []')
        self.refs = []
        self.excluded = dict(self.source.excluded)
        d = cfg['data']
        if d.get('representation_sequences'):
            raise ValueError('CenterNet RAW adapter currently takes single contexts, not image sequences')
        for sequence in self.source.sequences:
            labels = self.source._event_labels(sequence)
            if not labels:
                self.excluded[sequence] = 'missing event annotations'
            for time_us, path in labels.items():
                if d.get('require_boxes', False) or d.get('exclude_multiple_objects', False):
                    boxes = read_yolo(path)
                    n = len(boxes['boxes'])
                    if d.get('require_boxes', False) and n == 0:
                        continue
                    if d.get('exclude_multiple_objects', False) and n > 1:
                        continue
                if d.get('filter_missing_representations', True) and not rgb_inputs_available(self.source, cfg, sequence, time_us):
                    continue
                if time_us <= self.source.sequence_info[sequence]['end_us']:
                    self.refs.append((sequence, time_us, path))
        if max_samples:
            ids = np.random.default_rng(seed).choice(len(self.refs), min(max_samples, len(self.refs)), replace=False)
            self.refs = [self.refs[i] for i in ids]
        if not self.refs:
            raise ValueError('No RAW detection samples with available event annotations')

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        from experiments.object_detection.data.dataset import DetectionSample
        sequence, time_us, path = self.refs[index]
        labels = read_yolo(path)
        if not labels['available']:
            raise FileNotFoundError(path)
        if torch.any(labels['classes'] != 0):
            raise ValueError('Existing CenterNet is single-class; expected UAV class ID 0')
        boxes = labels['boxes']
        return DetectionSample(inputs=task_inputs(self.source, self.cfg, self.hierarchy, sequence, time_us),
                    gt_boxes_xywh=boxes, gt_velocities_xy=torch.zeros(len(boxes), 2),
                    gt_velocity_mask=torch.zeros(len(boxes), dtype=torch.bool), heatmaps={},
                    frame_key=f'{sequence}/{path.stem}', frame_time_s=time_us/1e6,
                    input_paths={'raw': str(self.source.sequence_info[sequence]['raw'])})


class RawFredForecastDataset(Dataset):
    def __init__(self, cfg, folders, *, max_samples=None, seed=123):
        self.cfg = cfg
        self.source, self.hierarchy = task_source(cfg, folders)
        d = cfg['data']
        step = self.source.frame_stride_us
        self.history_steps = int(d.get('history_steps', round(d.get('history_ms', 400)*1000/step)))
        self.future_steps = int(d.get('future_steps', round(d.get('forecast_ms', 800)*1000/step)))
        if self.history_steps < 1 or self.future_steps < 1:
            raise ValueError('Forecasting needs positive history_steps and future_steps')
        self.refs, self.excluded = [], dict(self.source.excluded)
        for sequence, info in self.source.sequence_info.items():
            labels = self.source._event_labels(sequence)
            if not info['original_tracks'] or not info['selected_tracks'] or not labels:
                self.excluded[sequence] = 'requires original tracks, selected tracks, and event annotations'
                continue
            label_times = np.array(list(labels), dtype=np.int64)
            for tid, rows in self.source.tracks(sequence).items():
                times = rows[:, 0].astype(np.int64)
                total = self.history_steps+self.future_steps+1
                if len(times) < total:
                    continue
                # Metadata-only eligibility checks: no events or RGB are loaded.
                for i in range(self.history_steps, len(times)-self.future_steps):
                    j = int(np.searchsorted(label_times, times[i]))
                    candidates = [k for k in (j-1, j) if 0 <= k < len(label_times)]
                    if not candidates:
                        continue
                    anchor = int(label_times[min(candidates, key=lambda k: abs(label_times[k]-times[i]))])
                    query = anchor+np.arange(-self.history_steps, self.future_steps+1)*step
                    observed = times[i-self.history_steps:i+self.future_steps+1]
                    if (np.all(np.abs(observed-query) <= self.source.match_tolerance_us)
                            and query[0] >= 0 and query[-1] <= info['end_us']):
                        if d.get('filter_missing_representations', True) and not rgb_inputs_available(self.source, cfg, sequence, anchor):
                            continue
                        self.refs.append((sequence, anchor, tid))
        if max_samples:
            ids = np.random.default_rng(seed).choice(len(self.refs), min(max_samples, len(self.refs)), replace=False)
            self.refs = [self.refs[i] for i in ids]
        if not self.refs:
            raise ValueError(f'No complete observed FRED forecast tracklets. Excluded sequences: {self.excluded}')

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        from experiments.kalman_ml_forecasting.data.track_dataset import KalmanForecastSample
        sequence, time_us, tid = self.refs[index]
        sample = self.source.get_tracklet(sequence, time_us, tid,
                           history_steps=self.history_steps, future_steps=self.future_steps)
        return KalmanForecastSample(inputs=task_inputs(self.source, self.cfg, self.hierarchy, sequence, time_us),
                    past_boxes=sample['past_boxes'], future_boxes=sample['future_boxes'],
                    past_times_s=sample['past_times_s'], future_times_s=sample['future_times_s'],
                    frame_key=f'{sequence}/{time_us}', frame_time_s=time_us/1e6, track_id=tid,
                    input_paths={'raw': str(self.source.sequence_info[sequence]['raw']),
                                 'tracks': sample['tracks_source']})
