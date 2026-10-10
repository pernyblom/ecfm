from copy import deepcopy
from pathlib import Path
import pickle

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
