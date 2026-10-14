"""Export label-aligned float event tensors and display previews using training YAML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecfm.data.fred_dataset import FREDDataset
from ecfm.utils.event_representations import representation_options_for
from ecfm.utils.event_visualization import representation_preview, resolve_event_visualization
from experiments.kalman_ml_forecasting.utils.config import (
    load_config, read_split_file, resolve_representation_source_image_sizes,
)


def export_fred_tensors(cfg, output_dir, *, sequences=None, split='train', max_frames=None):
    """Write [C,H,W] NPZ tensors plus PNG previews; previews are never training data."""
    data = cfg['data']
    if data.get('image_window_mode', 'trailing') != 'trailing':
        raise ValueError('Tensor export requires causal trailing windows')
    if data.get('backend') == 'raw' and 'hierarchy' in data['representations']:
        raise ValueError('Tensor export renders image representations, not hierarchical patches')
    if max_frames is not None and max_frames < 1:
        raise ValueError('max_frames must be positive')
    if sequences is None:
        sequences = read_split_file(Path(data['split_files'][split]))
    reps = [rep for rep in data['representations'] if rep not in {'rgb', 'padded_rgb', 'event_frames', 'event_frame'}]
    if not reps:
        raise ValueError('No event tensor representations requested')
    source = FREDDataset(data.get('events_root', data.get('root', data['labels_root'])),
        sequences=sequences, frame_source='grid', frame_stride_us=1, modalities=(),
        event_window_s=data.get('image_window_ms', 400)/1000,
        image_sizes=resolve_representation_source_image_sizes(data),
        representations=reps, representation_options=data.get('representation_options'),
        temporal_bins=data.get('temporal_bins', 224), cstr_max_count=data.get('cstr_max_count'),
        time_surface_tau_ms=data.get('time_surface_tau_ms'),
        event_cache_bytes=data.get('event_cache_bytes', 32*1024*1024))
    written = 0
    for sequence in source.sequences:
        # Labels may live separately from the event storage root.
        labels = Path(data['labels_root']) / sequence / data.get('labels_subdir', 'Event_YOLO')
        from experiments.kalman_ml_forecasting.data.track_dataset import _parse_frame_time_raw
        entries = []
        folder = Path(output_dir) / sequence
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / data.get('render_manifest_name', 'render_manifest.json')
        if manifest_path.exists():
            raise FileExistsError(f'Tensor export destination already has a manifest: {manifest_path}')
        anchors = sorted(((_parse_frame_time_raw(p.stem), p) for p in labels.glob('*.txt')
                         if _parse_frame_time_raw(p.stem) is not None), key=lambda item: item[0])
        for timestamp, label in anchors:
            end = round(timestamp * data.get('label_time_unit', 1e-6) * 1e6)
            tensors = source.representations_at(sequence, end)
            entry = {'label_stem': label.stem, 'window_duration_render_units': source.event_window_us / (data.get('label_time_unit', 1e-6)*1e6),
                     'representations': {}}
            for rep, tensor in tensors.items():
                tensor_path = folder / f'{label.stem}_{rep}.npz'
                np.savez_compressed(tensor_path, tensor=tensor.numpy())
                # Distinct filename prevents accidental use of a lossy preview as input.
                preview_path = folder / f'{label.stem}_{rep}_preview.png'
                representation_preview(tensor, rep,
                    options=representation_options_for(rep, data.get('representation_options')),
                    settings=resolve_event_visualization(cfg, rep)).save(preview_path)
                entry['representations'][rep] = {'path': str(tensor_path), 'preview_path': str(preview_path),
                    'channels': tensor.shape[0], 'shape': list(tensor.shape), 'dtype': 'float32'}
            entries.append(entry)
            written += 1
            if max_frames is not None and written >= max_frames:
                break
        manifest_path.write_text(json.dumps({'render_params': {'window_mode': 'trailing',
            'representation_options': data.get('representation_options', {}), 'representation': reps,
            'time_surface_tau_ms': data.get('time_surface_tau_ms'), 'cstr3_max_count': data.get('cstr_max_count')},
            'files': entries}, indent=2), encoding='utf-8')
        if max_frames is not None and written >= max_frames:
            break
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--sequence', action='append', dest='sequences')
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    count = export_fred_tensors(load_config(args.config), args.output_dir,
        sequences=args.sequences, split=args.split, max_frames=args.max_frames)
    print(f'Exported {count} event frames to {args.output_dir}')


if __name__ == '__main__':
    main()
