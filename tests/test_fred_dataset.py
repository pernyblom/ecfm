from copy import deepcopy
from contextlib import closing
from pathlib import Path
import pickle
import sqlite3

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import default_collate

from ecfm.data.fred_dataset import FREDDataset
from ecfm.utils.evt3_index import INDEX_DTYPE, INDEX_MAGIC
from experiments.hierarchical_mae.config import load_config
from experiments.hierarchical_mae.data import make_dataset, splits
from experiments.hierarchical_mae.backbone import HierarchicalBackbone
from experiments.kalman_ml_forecasting.train import _build_dataset, _collate, _make_loader
from experiments.kalman_ml_forecasting.models.kalman_residual import KalmanResidualForecaster


@pytest.fixture
def fred(tmp_path):
    for sequence in range(4):
        folder = tmp_path/str(sequence)
        (folder/'Event').mkdir(parents=True)
        header = b'% format EVT3;height=8;width=8\n% end\n'
        words, offsets, times = [], [], []
        for i in range(1, 23):
            t = i*33333-1000
            offsets.append(len(header)+len(words)*2)
            times.append(t)
            words.extend([0x8000 | ((t>>12)&0xfff), 0x6000 | (t&0xfff), i%8, 0x2800 | (i%8)])
        raw = folder/'Event/events.raw'
        raw.write_bytes(header+np.asarray(words, dtype='<u2').tobytes())
        records = []
        for slot in range(times[-1]//2000+2):
            prior = np.flatnonzero(np.asarray(times) <= slot*2000)
            j = int(prior[-1]) if len(prior) else None
            records.append((-1, len(header), 0) if j is None else (times[j], offsets[j], 1))
        sidecar = (f'% format EVT3;height=8;width=8\n% index_version 2.0\n% size {raw.stat().st_size}\n'
                   '% ts_shift_us 0\n% bookmark_period_us 2000\n% end\n').encode()
        raw.with_suffix('.raw.tmp_index').write_bytes(sidecar+np.array(records, dtype=INDEX_DTYPE).tobytes()+INDEX_MAGIC)
        track = ''.join(f'{i*33333/1e6:.6f},1,1,1,2,2\n' for i in range(1, 23))
        (folder/'cleaned_tracks.txt').write_text(track)
        if sequence == 2:
            continue  # unannotated sequence with an untrusted generated track file
        (folder/'tracks.txt').write_text(track)
        for sub in ['Event_YOLO', 'RGB_YOLO', 'RGB', 'PADDED_RGB']:
            (folder/sub).mkdir()
        for i in range(1, 23):
            (folder/'Event_YOLO'/f'Video_{sequence}_frame_{i*33333}.txt').write_text(
                '' if i == 1 else '0 0.25 0.25 0.25 0.25\n')
        for i in range(4):
            stem = f'Video_{sequence}_12_00_00.{i*33333:06d}'
            Image.new('RGB', (8, 6), (i*50, 0, 0)).save(folder/'RGB'/f'{stem}.jpg')
            Image.new('RGB', (8, 8), (i*50, 0, 0)).save(folder/'PADDED_RGB'/f'{stem}.jpg')
            (folder/'RGB_YOLO'/f'{stem}.txt').write_text('0 0.5 0.5 0.2 0.2\n')
    split_dir = tmp_path/'dataset_splits/canonical'
    split_dir.mkdir(parents=True)
    (split_dir/'train_split.txt').write_text('0/\n1/\n2/\n')
    (split_dir/'test_split.txt').write_text('3/\n')
    cfg = load_config('experiments/hierarchical_mae/configs/fred_smoke.yaml')
    cfg['data'].update(root=str(tmp_path), image_width=8, image_height=8,
                       train_split_file=str(split_dir/'train_split.txt'),
                       test_split_file=str(split_dir/'test_split.txt'),
                       max_sequences=0, max_samples=0, frame_stride_us=33333)
    cfg['train']['output_dir'] = str(tmp_path/'out')
    return tmp_path, cfg


def test_frame_modalities_and_causal_rgb(fred):
    root, _ = fred
    ds = FREDDataset(root, sequences=['0'], frame_source='event_labels',
                     modalities=('events', 'event_boxes', 'rgb', 'padded_rgb', 'rgb_boxes'))
    frame = ds.get_frame('0', 50000)
    assert frame['events'].dtype == np.int64
    assert np.all(frame['events'][:, 2] < 50000)
    assert frame['rgb']['time_us'] == 33333  # later frame at 66666 would be closer
    assert frame['rgb']['tensor'].shape == (3, 6, 8)
    assert frame['padded_rgb']['tensor'].shape == (3, 8, 8)
    assert frame['rgb_boxes']['coordinate_frame'] == 'RGB'
    assert not frame['event_boxes']['available']  # absent differs from empty negative
    empty = ds.get_frame('0', 33333)['event_boxes']
    assert empty['available'] and empty['boxes'].shape == (0, 4)


def indexed_forecast_config(root):
    return {'data': {
        'labels_root': str(root), 'frame_size': [8, 8],
        'representations': ['time_surface', 'event_stack', 'padded_rgb'],
        'image_sizes': {'time_surface': [8, 8], 'event_stack': [8, 8], 'padded_rgb': [8, 8]},
        'event_image_source': 'indexed', 'image_window_ms': 30,
        'history_ms': 33.333, 'forecast_ms': 33.333, 'label_period_s': .033333,
        'time_align': 'none', 'verify_render_manifest': True,
        'representation_sequences': {'time_surface': {'length': 2, 'stride': 1}},
        'cache_dir': str(root / 'cache'),
    }}


def test_indexed_forecasting_training_sequences_crop_and_cache(fred, monkeypatch):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data']['spatial_cutout'] = {'mode': 'fixed_pixels', 'size_px': [4, 4]}
    ds = _build_dataset(cfg, 'train', folders_override=['0'])
    assert len(ds) > 0 and ds._indexed_source is None  # metadata-only sample building
    sample = ds[0]
    assert sample.inputs['time_surface'].shape == (2, 3, 4, 4)
    assert sample.inputs['event_stack'].shape == sample.inputs['padded_rgb'].shape == (3, 4, 4)
    refs = sample.input_paths['time_surface']
    assert [r['time_us'] for r in refs] == [33333, 66666]
    assert Path(refs[0]['raw']).is_file()
    reads = []
    original = ds._indexed_source.events
    def observe(sequence, start, end):
        reads.append((start, end))
        return original(sequence, start, end)
    monkeypatch.setattr(ds._indexed_source, 'events', observe)
    ds[0]
    assert reads == [(3333, 33333), (36666, 66666), (36666, 66666)]
    cached = _build_dataset(cfg, 'train', folders_override=['0'])
    assert len(cached) == len(ds)
    for rep in sample.inputs:
        torch.testing.assert_close(cached[0].inputs[rep], sample.inputs[rep])
    # The normal trainer contract reaches the encoders and supports backpropagation.
    batch = _collate([sample, cached[0]])
    model = KalmanResidualForecaster(
        representations=list(sample.inputs), image_sizes={rep: (4, 4) for rep in sample.inputs},
        backbone_cfg={'type': 'small_cnn', 'in_channels': 3, 'channels': [4], 'out_dim': 8},
        history_steps=1, fusion_hidden_dim=8, state_hidden_dim=8, residual_hidden_dim=8,
        representation_sequences=cfg['data']['representation_sequences'],
    )
    prediction = model(batch.inputs, batch.past_boxes, batch.past_times_s, batch.future_times_s)
    assert prediction.shape == batch.future_boxes.shape
    (prediction - batch.future_boxes).square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoders.parameters())
    restored = pickle.loads(pickle.dumps(cached))
    torch.testing.assert_close(restored[0].inputs['event_stack'], sample.inputs['event_stack'])


def test_indexed_forecasting_window_changes_inputs_and_cache_key(fred):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data']['representations'] = ['event_stack']
    cfg['data']['representation_sequences'] = {}
    short = _build_dataset(cfg, 'train', folders_override=['0'])
    cfg['data']['image_window_ms'] = 70
    long = _build_dataset(cfg, 'train', folders_override=['0'])
    assert short._cache_key() != long._cache_key()
    assert short.samples[0]['anchor_stem'] == long.samples[0]['anchor_stem']
    assert not torch.equal(short[0].inputs['event_stack'], long[0].inputs['event_stack'])


@pytest.mark.parametrize('cache_enabled', [False, True])
def test_indexed_forecasting_spawned_training_workers(fred, cache_enabled):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data'].update(representations=['event_stack'], representation_sequences={}, max_samples=4)
    cfg['data']['instance_cache'] = {'enabled': cache_enabled}
    dataset = _build_dataset(cfg, 'train', folders_override=['0', '1'])
    # Populate parent caches before spawning; workers must rebuild their own.
    expected = _collate([dataset[i] for i in range(len(dataset))])
    loader = _make_loader(dataset, batch_size=2, shuffle=False,
                          train_cfg={'num_workers': 2, 'persistent_workers': False})
    actual = list(loader)
    torch.testing.assert_close(torch.cat([batch.inputs['event_stack'] for batch in actual]),
                               expected.inputs['event_stack'])
    assert [key for batch in actual for key in batch.frame_keys] == expected.frame_keys


def _cutout_count(dataset):
    with closing(sqlite3.connect(dataset._input_cache.path)) as connection:
        return connection.execute('SELECT COUNT(*) FROM cutouts').fetchone()[0]


def test_indexed_input_cache_roundtrip_sequences_multichannel_and_rgb(fred, monkeypatch):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data']['representation_options'] = {
        'time_surface': {'include_activity': False},
        'event_stack': {'bins': 5, 'split_polarity': True},
    }
    cfg['data']['spatial_cutout'] = {
        'mode': 'fixed_pixels', 'size_px': [4, 4], 'event_count_normalization': 'joint_max',
    }
    reference = _build_dataset(cfg, 'train', folders_override=['0'])[0]
    cfg['data']['instance_cache'] = {'enabled': True}
    dataset = _build_dataset(cfg, 'train', folders_override=['0'])
    assert dataset._indexed_source is None and _cutout_count(dataset) == 0
    first = dataset[0]
    assert _cutout_count(dataset) == 3  # two time surfaces and one event stack; no RGB
    for rep in reference.inputs:
        torch.testing.assert_close(first.inputs[rep], reference.inputs[rep], rtol=0, atol=0)
    # Mutation of a fetched tensor must not change future cache hits.
    first.inputs['event_stack'].fill_(42)
    warm = _build_dataset(cfg, 'train', folders_override=['0'])
    original = warm._input_image
    native_reads = []

    def native_only(rep, ref):
        assert not warm._is_indexed_rep(rep), 'cache hit must bypass RAW decoding and rendering'
        native_reads.append(rep)
        return original(rep, ref)

    monkeypatch.setattr(warm, '_input_image', native_only)
    restored = warm[0]
    assert native_reads == ['padded_rgb']
    assert warm._indexed_source is None
    for rep in reference.inputs:
        torch.testing.assert_close(restored.inputs[rep], reference.inputs[rep], rtol=0, atol=0)
    for name in ('past_boxes', 'future_boxes', 'past_times_s', 'future_times_s'):
        torch.testing.assert_close(getattr(restored, name), getattr(reference, name), rtol=0, atol=0)
    assert restored.frame_key == reference.frame_key
    assert restored.input_paths == reference.input_paths
    dataset[1]
    assert _cutout_count(dataset) == 5  # overlapping sequence frame shares an entry


def test_indexed_input_cache_only_selected_samples_and_extends(fred, monkeypatch):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data'].update(representations=['event_stack'], representation_sequences={}, max_samples=2,
                       instance_cache={'enabled': True})
    small = _build_dataset(cfg, 'train', folders_override=['0', '1'])
    assert _cutout_count(small) == 0
    expected = [small[i] for i in range(len(small))]
    assert _cutout_count(small) == 2
    cfg['data']['max_samples'] = 4
    large = _build_dataset(cfg, 'train', folders_override=['0', '1'])
    assert _cutout_count(large) == 2
    original = large._input_image
    reads = []

    def observe(rep, ref):
        reads.append(ref)
        return original(rep, ref)

    monkeypatch.setattr(large, '_input_image', observe)
    for i, reference in enumerate(expected):
        assert large[i].frame_key == reference.frame_key
        torch.testing.assert_close(large[i].inputs['event_stack'], reference.inputs['event_stack'])
    assert reads == []
    large[2]
    large[3]
    assert len(reads) == 2 and _cutout_count(large) == 4
    cfg['data']['max_samples'] = 1
    smaller = _build_dataset(cfg, 'train', folders_override=['0', '1'])
    smaller[0]
    assert smaller._indexed_source is None and _cutout_count(smaller) == 4
    # A decorrelated subset also creates entries only for its final selection.
    cfg['data']['instance_cache']['dir'] = str(root / 'decorrelated_cache')
    cfg['data']['max_samples'] = 10
    cfg['data']['decorrelation'] = {'enabled': True, 'method': 'random', 'target_samples': 2,
                                  'min_samples': 1}
    selected = _build_dataset(cfg, 'train', folders_override=['0', '1'])
    assert len(selected) == 2 and _cutout_count(selected) == 0
    for i in range(len(selected)):
        selected[i]
    assert _cutout_count(selected) == 2


@pytest.mark.parametrize('change', [
    {'image_window_ms': 70},
    {'temporal_bins': 10},
    {'time_surface_tau_ms': 5},
    {'representation_options': {'time_surface': {'include_activity': False}}},
    {'image_sizes': {'time_surface': [6, 6]}},
    {'spatial_cutout': {'mode': 'fixed_pixels', 'size_px': [4, 4]}},
    {'spatial_cutout': {'event_count_normalization': 'joint_max'}},
])
def test_indexed_input_cache_invalidates_preprocessing(fred, change):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data'].update(representations=['time_surface'], representation_sequences={},
                       instance_cache={'enabled': True})
    original = _build_dataset(cfg, 'train', folders_override=['0'])
    original[0]
    cfg['data'].update(change)
    changed = _build_dataset(cfg, 'train', folders_override=['0'])
    changed[0]
    assert changed._indexed_source is not None and _cutout_count(changed) == 2


def test_indexed_input_cache_invalidates_source_and_anchor(fred):
    import os
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data'].update(representations=['event_stack'], representation_sequences={},
                       instance_cache={'enabled': True},
                       spatial_cutout={'mode': 'fixed_pixels', 'size_px': [4, 4]})
    original = _build_dataset(cfg, 'train', folders_override=['0'])
    original[0]
    for source in ('events.raw', 'events.raw.tmp_index'):
        path = root / '0/Event' / source
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        changed = _build_dataset(cfg, 'train', folders_override=['0'])
        changed[0]
        assert changed._indexed_source is not None
    changed.samples[0]['past_boxes'][-1, 0] += .1
    changed[0]
    assert _cutout_count(changed) == 4


@pytest.mark.parametrize('rep,cutout', [
    ('event_stack', {}),
    ('event_stack', {'mode': 'fixed_pixels', 'size_px': [4, 4]}),
    ('xt', {'mode': 'fixed_pixels', 'size_px': [4, 4]}),
    ('yt', {'mode': 'fixed_pixels', 'size_px': [4, 4]}),
    ('event_stack', {'mode': 'box_scale', 'scale': 1}),
])
def test_indexed_input_cache_deduplicates_equivalent_pixel_crops(fred, monkeypatch, rep, cutout):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data'].update(representations=[rep], representation_sequences={}, image_sizes={rep: [8, 8]},
                       instance_cache={'enabled': True}, spatial_cutout=cutout)
    dataset = _build_dataset(cfg, 'train', folders_override=['0'])
    dataset.samples[0]['past_boxes'][-1, :2] += .025  # away from integer mask boundaries
    first = dataset[0].inputs[rep].clone()
    box = dataset.samples[0]['past_boxes'][-1]
    # All changes keep the actual pixel crop unchanged.
    box[:2] += .001
    if cutout.get('mode') != 'box_scale':
        box[2:] *= 2
    if rep == 'xt':
        box[1] += .5
    if rep == 'yt':
        box[0] += .5

    def unexpected_render(*args):
        pytest.fail('identical pixel cutout should reuse the existing entry')

    monkeypatch.setattr(dataset, '_input_image', unexpected_render)
    torch.testing.assert_close(dataset[0].inputs[rep], first, rtol=0, atol=0)
    assert _cutout_count(dataset) == 1


def test_indexed_input_cache_concurrent_cold_workers(fred):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data'].update(representations=['time_surface', 'event_stack'], max_samples=6)
    reference = _build_dataset(cfg, 'train', folders_override=['0', '1'])
    expected = _collate([reference[i] for i in range(len(reference))])
    cfg['data']['instance_cache'] = {'enabled': True}
    dataset = _build_dataset(cfg, 'train', folders_override=['0', '1'])
    # Repeated indices deliberately send duplicate cache misses to both workers.
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=_collate,
                        sampler=[0, 1, 0, 1, 2, 3, 4, 5], multiprocessing_context='spawn')
    batches = list(loader)
    for rep in expected.inputs:
        actual = torch.cat([batch.inputs[rep] for batch in batches])
        torch.testing.assert_close(actual, expected.inputs[rep][[0, 1, 0, 1, 2, 3, 4, 5]], rtol=0, atol=0)
    assert _cutout_count(dataset) > 0
    # Warm main-process reads use worker-created entries without opening RAW.
    for i in range(len(dataset)):
        dataset[i]
    assert dataset._indexed_source is None
    restored = pickle.loads(pickle.dumps(dataset))
    torch.testing.assert_close(restored[0].inputs['event_stack'], expected.inputs['event_stack'][0])
    assert restored._indexed_source is None


@pytest.mark.parametrize('overrides, message', [
    ({'box_augmentation': {'enabled': True, 'live': True}}, 'box_augmentation'),
    ({'event_image_source': 'rendered'}, 'event_image_source'),
    ({'cache_dir': None}, 'requires instance_cache.dir'),
    ({'instance_cache': {'enabled': True, 'compression_level': 10}}, 'compression_level'),
])
def test_indexed_input_cache_validation(fred, overrides, message):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data']['instance_cache'] = {'enabled': True}
    cfg['data'].update(overrides)
    with pytest.raises(ValueError, match=message):
        _build_dataset(cfg, 'train', folders_override=['0'])


def test_indexed_forecasting_rejects_future_windows_and_missing_fixed_scale(fred):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data']['image_window_mode'] = 'center'
    with pytest.raises(ValueError, match='trailing'):
        _build_dataset(cfg, 'train', folders_override=['0'])
    cfg['data']['image_window_mode'] = 'trailing'
    cfg['data']['representations'] = ['cstr3_fixed']
    cfg['data']['representation_sequences'] = {}
    cfg['data']['image_sizes']['cstr3_fixed'] = [8, 8]
    with pytest.raises(ValueError, match='cstr_max_count'):
        _build_dataset(cfg, 'train', folders_override=['0'])


def test_indexed_forecasting_uses_label_sequence_not_assumed_cadence(fred):
    root, _ = fred
    # Removing one label makes consecutive label frames nonuniform in time.
    (root / '0/Event_YOLO/Video_0_frame_66666.txt').unlink()
    cfg = indexed_forecast_config(root)
    cfg['data']['representations'] = ['time_surface']
    ds = _build_dataset(cfg, 'train', folders_override=['0'])
    assert ds.samples[0]['anchor_time_s'] == pytest.approx(.099999)
    assert [r['time_us'] for r in ds[0].input_paths['time_surface']] == [33333, 99999]


def test_indexed_forecasting_cache_filters_missing_raw_index(fred):
    root, _ = fred
    cfg = indexed_forecast_config(root)
    original = _build_dataset(cfg, 'train', folders_override=['0'])
    assert len(original) > 0
    (root / '0/Event/events.raw.tmp_index').unlink()
    missing = _build_dataset(cfg, 'train', folders_override=['0'])
    assert len(missing) == 0
    cfg['data']['filter_missing_representations'] = False
    with pytest.raises(FileNotFoundError, match='cached representation'):
        _build_dataset(cfg, 'train', folders_override=['0'])


def test_multichannel_indexed_training_and_tensor_export_roundtrip(fred):
    from scripts.render_fred_tensors import export_fred_tensors
    from experiments.kalman_ml_forecasting.models.factory import build_model
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data']['representation_options'] = {'time_surface': {'include_activity': False},
        'event_stack': {'bins': 5, 'split_polarity': True}}
    cfg['data']['spatial_cutout'] = {'mode': 'fixed_pixels', 'size_px': [4, 4],
                                   'event_count_normalization': 'joint_max'}
    cfg['model'] = {'history_steps': 1, 'backbone': {'type': 'small_cnn', 'channels': [4], 'out_dim': 8},
                    'fusion_hidden_dim': 8, 'state_hidden_dim': 8, 'residual_hidden_dim': 8}
    indexed = _build_dataset(cfg, 'train', folders_override=['0'])
    sample = indexed[0]
    assert sample.inputs['time_surface'].shape == (2, 2, 4, 4)
    assert sample.inputs['event_stack'].shape == (10, 4, 4)
    batch = _collate([sample, indexed[1]])
    model = build_model(cfg, torch.device('cpu'))
    model(batch.inputs, batch.past_boxes, batch.past_times_s, batch.future_times_s).sum().backward()
    assert model.representation_channels['event_stack'] == 10
    assert export_fred_tensors(cfg, root/'tensor_export', sequences=['0'], max_frames=4) == 4
    cfg['data']['event_image_source'] = 'rendered'
    cfg['data']['images_root'] = str(root/'tensor_export')
    stored = _build_dataset(cfg, 'train', folders_override=['0'])
    assert stored[0].frame_key == sample.frame_key
    for rep in sample.inputs:
        torch.testing.assert_close(stored[0].inputs[rep], sample.inputs[rep])
    assert Path(stored[0].input_paths['event_stack']).suffix == '.npz'
    with pytest.raises(FileExistsError):
        export_fred_tensors(cfg, root/'tensor_export', sequences=['0'], max_frames=4)


def test_multichannel_cutout_inspector_respects_yaml_grid_preview(fred, monkeypatch):
    import sys
    import yaml
    from experiments.kalman_ml_forecasting.inspect_spatial_cutouts import main
    root, _ = fred
    cfg = indexed_forecast_config(root)
    cfg['data']['representations'] = ['event_stack']
    cfg['data']['representation_sequences'] = {}
    cfg['data']['representation_options'] = {'event_stack': {'bins': 5, 'split_polarity': True}}
    cfg['data']['spatial_cutout'] = {'mode': 'fixed_pixels', 'size_px': [4, 4]}
    cfg['visualization'] = {'event_images': {'mode': 'grid', 'columns': 5, 'labels': False}}
    config_file = root/'inspect.yaml'
    config_file.write_text(yaml.safe_dump(cfg))
    output = root/'preview_output'
    monkeypatch.setattr(sys, 'argv', ['inspect_spatial_cutouts.py', '--config', str(config_file),
        '--folder', '0', '--count', '1', '--include-original', '--output-dir', str(output)])
    main()
    with Image.open(next(output.rglob('event_stack_cutout.png'))) as image:
        assert image.size == (20, 8)
    with Image.open(next(output.rglob('event_stack_original.png'))) as image:
        assert image.size == (40, 16)


def test_unlabeled_sequence_and_supervision_gate(fred):
    root, cfg = fred
    ds = FREDDataset(root, sequences=['2'], tracks_file='cleaned_tracks.txt', modalities=())
    assert len(ds) > 0
    assert not ds.get_frame('2', 66666, modalities=('event_boxes',))['event_boxes']['available']
    with pytest.raises(FileNotFoundError, match='Original tracks'):
        ds.get_tracklet('2', 99999, 1, history_steps=1, future_steps=1)
    patches = ds.hierarchical_patches('2', 33333, cfg, lookback_s=.1)
    assert patches['valid_mask'].all()
    assert torch.allclose(patches['metadata'][:, -1], torch.full((18,), np.log1p(.1)))


def test_observed_tracklet_zero_times_and_no_future_inputs(fred, monkeypatch):
    root, cfg = fred
    ds = FREDDataset(root, sequences=['0'], modalities=(), image_sizes={'cstr3': (8, 8), 'xt_my': (8, 4)})
    reads = []
    original = ds.events
    def observed(sequence, start, end):
        reads.append((start, end))
        return original(sequence, start, end)
    monkeypatch.setattr(ds, 'events', observed)
    sample = ds.get_tracklet('0', 5*33333, 1, history_steps=2, future_steps=3,
                representations=['cstr3', 'xt_my'], hierarchy_cfg=cfg, hierarchy_lookback_s=.1)
    assert sample['past_boxes'].shape == (3, 4)
    assert sample['future_boxes'].shape == (3, 4)
    assert sample['past_times_s'][-1] == 0
    assert sample['future_times_s'][0] > 0
    assert sample['history_inputs']['xt_my'].shape == (3, 3, 4, 8)
    assert all(end <= 5*33333 for _, end in reads)
    np.testing.assert_allclose(sample['boxes'][0], [.25, .25, .25, .25])
    ds._tracks['0'][1] = np.delete(ds.tracks('0')[1], 3, axis=0)
    with pytest.raises(ValueError, match='Incomplete'):
        ds.get_tracklet('0', 5*33333, 1, history_steps=2, future_steps=3)


def test_rgb_history_does_not_decode_and_missing_rgb_is_explicit(fred, monkeypatch):
    root, _ = fred
    ds = FREDDataset(root, sequences=['0'], modalities=(), image_sizes={'rgb': (4, 3)})
    monkeypatch.setattr(ds, 'events', lambda *args: (_ for _ in ()).throw(AssertionError('RGB needs no events')))
    sample = ds.get_tracklet('0', 99999, 1, history_steps=2, future_steps=1, representations=['rgb', 'padded_rgb'])
    assert sample['history_inputs']['rgb'].shape == (3, 3, 3, 4)
    with pytest.raises(FileNotFoundError, match='causal rgb'):
        ds.representations_at('0', 500000, ['rgb'])


def test_bounded_cache_and_spawn_state(fred):
    root, _ = fred
    ds = FREDDataset(root, sequences=['0'], event_cache_bytes=32, modalities=())
    events = ds.events('0', 0, 33333)
    events[:, 0] = 99
    assert np.all(ds.events('0', 0, 33333)[:, 0] < 8)
    ds.events('0', 33333, 66666)
    assert ds._event_bytes <= 32
    restored = pickle.loads(pickle.dumps(ds))
    assert not restored._readers and not restored._events
    np.testing.assert_array_equal(ds.events('0', 33333, 66666), restored.events('0', 33333, 66666))


def test_fred_splits_pretraining_and_crop_cache(fred):
    _, cfg = fred
    train, val, test = splits(cfg, include_test=True)
    assert {p for p, _ in train}.isdisjoint({p for p, _ in val+test})
    from experiments.object_detection.train import _split_train_eval_folders
    task_cfg = dict(data=dict(backend='raw', split_files={'train': cfg['data']['train_split_file']},
                         train_eval_split=dict(enabled=True, eval_fraction=.15, seed=42)))
    task_train, task_val = _split_train_eval_folders(task_cfg)
    assert task_train == [p.parent.parent.name for p, _ in train]
    assert task_val == [p.parent.parent.name for p, _ in val]
    ds = make_dataset(cfg, train, True)
    sample = ds[(0, 0)]
    assert sample['label'] == -1
    assert sample['source']['metadata'].shape == (18, 9)
    cfg = deepcopy(cfg)
    cfg['data']['patch_cache'].update(enabled=True, dir=str(Path(cfg['data']['root'])/'patches'), train_views=2)
    cached = make_dataset(cfg, train, True)
    first = cached[(0, 0)]['source']
    cached.frames.hierarchical_patches = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('expected cache hit'))
    repeated = cached[(0, 2)]['source']
    for key in first['patches']:
        torch.testing.assert_close(first['patches'][key], repeated['patches'][key])
    Path(cfg['data']['test_split_file']).write_text('0/\n')
    with pytest.raises(ValueError, match='overlap'):
        splits(cfg)


def test_spatial_backbone_checkpoint_gradients_and_freeze(fred, tmp_path):
    root, cfg = fred
    source = FREDDataset(root, sequences=['0'], modalities=())
    view = source.hierarchical_patches('0', 99999, cfg, lookback_s=.1)
    batched = default_collate([view, view])
    backbone = HierarchicalBackbone(dict(mae_cfg=cfg, out_dim=8))
    output = backbone(batched)
    assert output.fmap.shape == (2, 16, 2, 2)
    assert output.pooled.shape == (2, 8)
    (output.fmap.square().mean()+output.pooled.square().mean()).backward()
    assert backbone.model.patch_encoders['l0_xt'][1].weight.grad is not None
    checkpoint = tmp_path/'pretrained.pt'
    torch.save(dict(config=cfg, model=backbone.model.state_dict()), checkpoint)
    frozen = HierarchicalBackbone(dict(mae_cfg=cfg, checkpoint=str(checkpoint), freeze=True, out_dim=8))
    frozen.train()
    assert not frozen.model.training
    out = frozen(batched)
    out.pooled.sum().backward()
    assert frozen.projection.weight.grad is not None
    assert all(p.grad is None for p in frozen.model.parameters())


def test_existing_task_collation_and_models_with_raw_inputs(fred):
    root, mae_cfg = fred
    from experiments.object_detection import train as detection
    from experiments.kalman_ml_forecasting import train as forecasting
    from experiments.object_detection.losses import compute_losses
    cfg = dict(data=dict(backend='raw', root=str(root), frame_size=[8, 8], representations=['hierarchy'],
                        image_sizes={'hierarchy': [16, 16]}, hierarchy_lookback_s=.1,
                        tracks_file='cleaned_tracks.txt', history_steps=2, future_steps=2),
               model=dict(detector='centernet', backbone=dict(type='hierarchical_mae', mae_cfg=mae_cfg, out_dim=8),
                          predict_velocity=False, history_steps=3, fusion_hidden_dim=8,
                          state_hidden_dim=8, residual_hidden_dim=8, centernet_hidden_dim=8, topk=4))
    det = detection._build_dataset(cfg, 'train', folders_override=['0'])
    batch = detection._collate([det[0], det[1]])
    model = detection.build_model(cfg, torch.device('cpu'))
    out = model(batch.inputs)
    loss, _, _ = compute_losses(out, batch.gt_boxes_xywh, batch.heatmaps)
    loss.backward()
    assert model.encoders['hierarchy'].model.encoder.layers[0].self_attn.in_proj_weight.grad is not None
    forecast = forecasting._build_dataset(cfg, 'train', folders_override=['0', '2'])
    assert '2' in forecast.excluded
    samples = forecasting._collate([forecast[0], forecast[1]])
    forecaster = forecasting.build_model(cfg, torch.device('cpu'))
    predicted = forecaster(samples.inputs, samples.past_boxes, samples.past_times_s, samples.future_times_s)
    assert predicted.shape == (2, 2, 4)
    torch.nn.functional.smooth_l1_loss(predicted, samples.future_boxes).backward()
    assert forecaster.encoders['hierarchy'].model.encoder.layers[0].self_attn.in_proj_weight.grad is not None
    with pytest.raises(ValueError, match='explicit split'):
        detection._build_dataset(cfg, 'test')
    det.refs[0][2].unlink()
    with pytest.raises(FileNotFoundError):
        det[0]


def test_raw_cnn_forecasting_with_causal_rgb_sequences(fred):
    root, _ = fred
    from experiments.kalman_ml_forecasting import train as forecasting
    cfg = dict(data=dict(backend='raw', root=str(root), frame_size=[8, 8], representations=['cstr3', 'rgb'],
                      image_sizes={'cstr3': [8, 8], 'rgb': [8, 8]}, tracks_file='tracks.txt',
                      history_steps=2, future_steps=2, representation_sequences={'cstr3': 3, 'rgb': {'length': 3, 'stride': 1}}),
               model=dict(backbone=dict(type='small_cnn', channels=[4, 8], out_dim=8),
                          history_steps=3, fusion_hidden_dim=8, state_hidden_dim=8, residual_hidden_dim=8))
    dataset = forecasting._build_dataset(cfg, 'train', folders_override=['0'])
    batch = forecasting._collate([dataset[0], dataset[1]])
    assert batch.inputs['rgb'].shape == (2, 3, 3, 8, 8)
    model = forecasting.build_model(cfg, torch.device('cpu'))
    assert model(batch.inputs, batch.past_boxes, batch.past_times_s, batch.future_times_s).shape == (2, 2, 4)
