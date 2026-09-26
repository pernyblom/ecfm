from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from ecfm.data.tokenizer import Region
from experiments.region_worldmodel.actions import parse_actions, transform_region
from experiments.region_worldmodel.config import load_config
from experiments.region_worldmodel.data import RegionActionDataset, load_events, partition_entries
from experiments.region_worldmodel.downstream import Classifier, run as downstream_run
from experiments.region_worldmodel.model import RegionWorldModel, objective
from experiments.region_worldmodel.train import make_loader, run
from experiments.region_worldmodel.regions import downstream_layout
from experiments.region_worldmodel.feature_cache import cached_features, cache_metadata, file_digest


def action(**kwargs):
    return parse_actions([dict(name='a', **kwargs)])[0]


@pytest.mark.parametrize('axis,expected', [
    ('x', [0.7, 0.4, 0.6]), ('y', [0.6, 0.6, 0.3]), ('t', [0.4, 0.7, 0.6])])
def test_rotation_centers_axes_and_inverse(axis, expected):
    region = Region(65, 55, .55, 10, 10, .1, 'xt')
    moved = transform_region(region, action(type='rotate', axis=axis, degrees=90), 100, 100)
    center = [moved.x / 100 + .05, moved.y / 100 + .05, moved.t + .05]
    np.testing.assert_allclose(center, expected, atol=1e-6)
    restored = transform_region(moved, action(type='rotate', axis=axis, degrees=-90), 100, 100)
    assert restored.x == region.x and restored.y == region.y
    assert restored.t == pytest.approx(region.t)
    assert moved.plane == region.plane and moved.dt == region.dt


def test_custom_axis_scaling_and_bounds():
    a = action(type='rotate', axis=[1, 1, 1], degrees=120)
    np.testing.assert_allclose(a.rotation @ [1, 0, 0], [0, 1, 0], atol=1e-6)
    region = Region(30, 40, .4, 20, 20, .2, 'xy')
    smaller = transform_region(region, action(type='scale', factor=.5), 100, 100)
    assert (smaller.x, smaller.y, smaller.dx, smaller.dy) == (35, 45, 10, 10)
    assert smaller.t == pytest.approx(.45) and smaller.dt == pytest.approx(.1)
    for a in [action(type='scale', factor=20), action(type='translate', offset=[2, -2, 3])]:
        r = transform_region(region, a, 100, 100)
        assert 0 <= r.x <= 100 - r.dx and 0 <= r.y <= 100 - r.dy
        assert 0 <= r.t <= 1 - r.dt + 1e-8
    assert np.all(action(type='identity').vector() == 0)


@pytest.mark.parametrize('spec', [dict(type='rotate', axis=[0, 0, 0], degrees=10),
    dict(type='scale', factor=-1), dict(type='scale', factor=float('nan')),
    dict(type='translate', offset=[1, 2]), dict(type='rotate', axis='z', degrees=10)])
def test_bad_actions_rejected(spec):
    with pytest.raises(ValueError):
        action(**spec)


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / 'experiments/region_worldmodel/configs/smoke.yaml')
    cfg['data'].update(root=str(tmp_path), image_width=20, image_height=20,
                       num_regions_choices=[2, 3], region_scales=[.3], region_time_scales=[.5])
    cfg['model'].update(embed_dim=16, decoder_embed_dim=16)
    cfg['train'].update(device='cpu', batch_size=2, max_batches=1, output_dir=str(tmp_path / 'output'))
    cfg['downstream'].update(num_classes=2, batch_size=2, max_batches=1)
    cfg['downstream']['feature_cache'] = dict(dir=str(tmp_path / 'features'))
    cfg['actions'] = [dict(name='identity', type='identity'),
                      dict(name='shift', type='translate', offset=[.3, 0, 0])]
    rng = np.random.default_rng(9)
    for i in range(10):
        events = np.column_stack([rng.integers(0, 20, 300), rng.integers(0, 20, 300),
                                  np.arange(300) + 1e9, rng.integers(0, 2, 300)])
        np.save(tmp_path / f'{i}.npy', events)
    (tmp_path / 'train.txt').write_text('\n'.join(f'{i}.npy {i % 2}' for i in range(8)))
    (tmp_path / 'test.txt').write_text('8.npy 0\n9.npy 1\n')
    return cfg


def test_timestamp_precision_and_split_isolation(cfg):
    events, duration = load_events(Path(cfg['data']['root']) / '0.npy', 1e-6)
    assert duration == pytest.approx(299e-6)
    assert len(np.unique(events[:, 2])) == 300
    train, val = partition_entries(cfg)
    assert not {p for p, _ in train} & {p for p, _ in val}
    assert (train, val) == partition_entries(cfg)


def test_targets_are_resensed_not_metadata_only(cfg):
    train, _ = partition_entries(cfg)
    ds = RegionActionDataset(cfg, train)
    identity, shifted = ds[0], ds[len(train)]
    for key in identity['source']:
        torch.testing.assert_close(identity['source'][key], identity['target'][key])
        torch.testing.assert_close(identity['source'][key], shifted['source'][key])
        torch.testing.assert_close(shifted['target'][key], ds[len(train)]['target'][key])
    assert not torch.equal(shifted['source']['patches'], shifted['target']['patches'])
    assert not torch.equal(shifted['source']['metadata'], shifted['target']['metadata'])
    ds.training = True
    first = ds[0]['source']['metadata']
    ds.epoch = 1
    assert not torch.equal(first, ds[0]['source']['metadata'])


def test_gradients_padding_and_action_conditioning(cfg):
    train, _ = partition_entries(cfg)
    ds = RegionActionDataset(cfg, train, training=True)
    batch = next(iter(make_loader(ds, 2)))
    model = RegionWorldModel(cfg).eval()
    # Ensure at least one padded slot regardless of sampled counts.
    for side in ('source', 'target'):
        batch[side]['valid_mask'][:, -1] = False
        batch[side]['metadata'][:, -1] = 0
    z = model.features(batch['source'])
    changed = deepcopy(batch['source'])
    changed['patches'][:, -1] = 100
    changed['metadata'][:, -1] = 10
    torch.testing.assert_close(z, model.features(changed))
    mask = batch['source']['valid_mask'].clone()
    loss, _, (_, target, pred) = objective(model, batch, cfg, mask)
    target.retain_grad()
    loss.backward()
    assert torch.isfinite(loss) and target.grad.abs().sum() > 0
    assert model.encoder.patch_encoder.net[0].weight.grad is not None
    assert model.predictor[0].weight.grad[:, -15:].abs().sum() > 0 or batch['action'].abs().sum() == 0
    source = model.encode(batch['source'])
    assert not torch.allclose(model.predict(source, torch.zeros(2, 15)),
                              model.predict(source, torch.ones(2, 15)))


@pytest.mark.parametrize('frozen', [True, False])
def test_probe_freezes_encoder_and_finetune_updates_it(cfg, frozen):
    train, _ = partition_entries(cfg)
    view = next(iter(make_loader(RegionActionDataset(cfg, train, paired=False), 2)))['source']
    model = Classifier(RegionWorldModel(cfg), 2, frozen).train()
    before = {k: p.clone() for k, p in model.backbone.state_dict().items()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(view).square().mean().backward()
    optimizer.step()
    changed = any(not torch.equal(before[k], p) for k, p in model.backbone.state_dict().items())
    assert changed == (not frozen)
    if frozen:
        assert not model.backbone.training


def test_end_to_end_checkpoints_and_resume(cfg):
    run(cfg)
    output = Path(cfg['train']['output_dir'])
    checkpoint = output / 'best.pt'
    for mode in ('linear_probe', 'finetune'):
        result = downstream_run(cfg, checkpoint, mode)
        assert result['test']['samples'] == 2
        assert (output / mode / 'results.json').is_file()
    cfg['train']['epochs'] = 2
    run(cfg, output / 'last.pt')
    state = torch.load(output / 'last.pt', weights_only=True)
    assert state['epoch'] == 1


@pytest.mark.parametrize('spec', [
    dict(mode='grid', grid=[3, 3, 1], num_regions=27),
    dict(mode='multiscale', levels=[[1, 1, 1], [2, 2, 2]], num_regions=27)])
def test_fixed_layout_coverage_and_epoch_independence(cfg, spec):
    cfg['downstream']['regions'] = spec
    regions = downstream_layout(cfg)
    assert len(regions) == 27
    for r in regions:
        assert 0 <= r.x < r.x + r.dx <= 20
        assert 0 <= r.y < r.y + r.dy <= 20
        assert 0 <= r.t < r.t + r.dt <= 1
    # Every level covers the volume exactly once per projection plane.
    groups = [regions] if spec['mode'] == 'grid' else [regions[:3], regions[3:]]
    for group in groups:
        for plane in cfg['data']['plane_types']:
            assert sum(r.dx * r.dy * r.dt for r in group if r.plane == plane) == pytest.approx(400)
    train, _ = partition_entries(cfg)
    ds = RegionActionDataset(cfg, train, training=True, paired=False)
    first = ds[0]['source']
    ds.epoch = 20
    for key, value in first.items():
        torch.testing.assert_close(value, ds[0]['source'][key])
    torch.testing.assert_close(first['metadata'][:, :6], ds[1]['source']['metadata'][:, :6])
    assert first['valid_mask'].all() and first['patches'].shape[0] == 27
    # Downstream settings must not affect pretraining's random region budget.
    assert RegionActionDataset(cfg, train, training=True).max_regions == 3


@pytest.mark.parametrize('spec', [
    dict(mode='grid', grid=[2, 2, 2], num_regions=27),
    dict(mode='multiscale', levels=[]),
    dict(mode='multiscale', levels=[[1, 1, 1], [1, 1, 1]]),
    dict(mode='grid', grid=[0, 2, 2]),
    dict(mode='grid', grid=[21, 2, 2]),
    dict(mode='grid', grid=[1, 1, 1], plane_mode='unknown'),
    dict(mode='random', num_regions=0)])
def test_reject_invalid_downstream_layouts(cfg, spec):
    cfg['downstream']['regions'] = spec
    with pytest.raises(ValueError):
        downstream_layout(cfg)


def test_fixed_random_count_and_cycle_planes(cfg):
    cfg['downstream']['regions'] = dict(mode='random', num_regions=7)
    train, _ = partition_entries(cfg)
    ds = RegionActionDataset(cfg, train, paired=False)
    assert ds[0]['source']['valid_mask'].sum() == 7
    cfg['downstream']['regions'] = dict(mode='grid', grid=[2, 2, 2], plane_mode='cycle', num_regions=8)
    regions = downstream_layout(cfg)
    assert len(regions) == 8
    assert [r.plane for r in regions[:3]] == cfg['data']['plane_types']


@pytest.mark.parametrize('mode', ['linear_probe', 'finetune'])
@pytest.mark.parametrize('layout', ['grid', 'multiscale'])
def test_different_downstream_count_from_pretrained_checkpoint(cfg, mode, layout):
    output = Path(cfg['train']['output_dir'])
    output.mkdir()
    backbone = RegionWorldModel(cfg)
    torch.save(dict(model=backbone.state_dict(), config=deepcopy(cfg)), output / 'pretrained.pt')
    spec = dict(mode=layout, num_regions=27)
    spec.update(grid=[3, 3, 1]) if layout == 'grid' else spec.update(levels=[[1, 1, 1], [2, 2, 2]])
    cfg['downstream']['regions'] = spec
    result = downstream_run(cfg, output / 'pretrained.pt', mode)
    assert result['max_region_tokens'] == 27 and result['test']['samples'] == 2
    state = torch.load(output / f'{mode}_{layout}_27' / 'best.pt', weights_only=True)
    restored = Classifier(RegionWorldModel(state['backbone_config']), 2, mode == 'linear_probe')
    restored.load_state_dict(state['model'])
    train, _ = partition_entries(cfg)
    view = next(iter(make_loader(RegionActionDataset(cfg, train, paired=False), 2)))['source']
    assert restored(view).shape == (2, 2)


def test_cached_probe_matches_online_training_and_reuses_features(cfg, monkeypatch):
    from experiments.region_worldmodel.downstream import classification_epoch
    train, _ = partition_entries(cfg)
    ds = RegionActionDataset(cfg, train, paired=False)
    online = Classifier(RegionWorldModel(cfg), 2, True)
    cached = deepcopy(online)
    settings = cfg['downstream']
    features = cached_features(cached.backbone, ds, 'checkpoint', settings, 'cpu', 0, 'train')
    assert features.features.shape == (len(ds), cfg['model']['embed_dim'])
    first_optimizer = torch.optim.SGD(online.head.parameters(), lr=.01)
    second_optimizer = torch.optim.SGD(cached.head.parameters(), lr=.01)
    first = classification_epoch(online, make_loader(ds, 2, shuffle=True, seed=3), 'cpu', first_optimizer)
    def fail(*args, **kwargs):
        pytest.fail('A cached probe must not load events or call the encoder')
    monkeypatch.setattr(RegionActionDataset, '__getitem__', fail)
    monkeypatch.setattr(cached.backbone, 'features', fail)
    reused = cached_features(cached.backbone, ds, 'checkpoint', settings, 'cpu', 0, 'train')
    torch.testing.assert_close(features.features, reused.features)
    second = classification_epoch(cached, make_loader(reused, 2, shuffle=True, seed=3), 'cpu', second_optimizer)
    assert first['loss'] == pytest.approx(second['loss'], abs=1e-6)
    torch.testing.assert_close(online.head.weight, cached.head.weight)
    torch.testing.assert_close(online.head.bias, cached.head.bias)


def test_feature_cache_invalidation_and_optimizer_independence(cfg):
    import os
    train, _ = partition_entries(cfg)
    ds = RegionActionDataset(cfg, train, paired=False)
    original = cache_metadata(ds, 'checkpoint')
    cfg['downstream']['lr'] *= 2
    cfg['downstream']['epochs'] += 1
    assert cache_metadata(ds, 'checkpoint') == original
    assert cache_metadata(ds, 'other checkpoint') != original
    cfg['data']['region_seed'] += 1
    assert cache_metadata(ds, 'checkpoint') != original
    cfg['data']['region_seed'] -= 1
    cfg['downstream']['regions'] = dict(mode='random', num_regions=7)
    assert cache_metadata(ds, 'checkpoint') != original
    del cfg['downstream']['regions']
    ds.entries = list(reversed(train))
    assert cache_metadata(ds, 'checkpoint') != original
    ds.entries = train
    path, _ = train[0]
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    assert cache_metadata(ds, 'checkpoint') != original
    before = file_digest(path)
    events = np.load(path)
    events[0, 0] += 1
    np.save(path, events)
    assert file_digest(path) != before


def test_feature_cache_rebuild_and_guard(cfg, monkeypatch):
    train, _ = partition_entries(cfg)
    ds = RegionActionDataset(cfg, train, paired=False)
    backbone = Classifier(RegionWorldModel(cfg), 2, True).backbone
    settings = cfg['downstream']
    cached_features(backbone, ds, 'checkpoint', settings, 'cpu', 0, 'train')
    calls = []
    original = backbone.features
    def record(view):
        calls.append(1)
        return original(view)
    monkeypatch.setattr(backbone, 'features', record)
    settings['feature_cache']['rebuild'] = True
    cached_features(backbone, ds, 'checkpoint', settings, 'cpu', 0, 'train')
    assert calls
    ds.training = True
    with pytest.raises(ValueError, match='fixed'):
        cached_features(backbone, ds, 'checkpoint', settings, 'cpu', 0, 'train')


def test_duration_option_preserves_patches_and_normalized_geometry(cfg):
    cfg['downstream']['regions'] = dict(mode='grid', grid=[1, 1, 1])
    train, _ = partition_entries(cfg)
    ds = RegionActionDataset(cfg, train, paired=False)
    events, duration = load_events(train[0][0], cfg['data']['time_unit'])
    original = ds.render(events, duration, ds.fixed_regions)
    cfg['data']['include_absolute_duration'] = False
    hidden = ds.render(events, duration, ds.fixed_regions)
    changed_duration = ds.render(events, duration * 20, ds.fixed_regions)
    torch.testing.assert_close(original['patches'], hidden['patches'])
    torch.testing.assert_close(original['metadata'][:, :6], hidden['metadata'][:, :6])
    assert original['metadata'][:, 6:].abs().sum() > 0
    assert hidden['metadata'][:, 6:].abs().sum() == 0
    torch.testing.assert_close(hidden['metadata'], changed_duration['metadata'])


def test_blank_features_cache_and_memory_storage(cfg, monkeypatch):
    from experiments.region_worldmodel.diagnostics import probe_blank_diagnostic
    train, val = partition_entries(cfg)
    cfg['data']['include_absolute_duration'] = False
    cfg['downstream']['regions'] = dict(mode='grid', grid=[1, 1, 1])
    cfg['downstream']['feature_cache']['storage'] = 'memory'
    ds = RegionActionDataset(cfg, val, paired=False)
    model = Classifier(RegionWorldModel(cfg), 2, True)
    features = cached_features(model.backbone, ds, '', cfg['downstream'], 'cpu', 0, 'validation')
    assert not Path(cfg['downstream']['feature_cache']['dir']).exists()
    torch.testing.assert_close(features.blank_features[0], features.blank_features[1])
    online = probe_blank_diagnostic(model, make_loader(ds, 2), 'cpu')
    cached = probe_blank_diagnostic(model, make_loader(features, 2), 'cpu')
    assert online == pytest.approx(cached, abs=1e-6)
    cfg['downstream']['feature_cache']['storage'] = 'disk'
    saved = cached_features(model.backbone, ds, 'checkpoint', cfg['downstream'], 'cpu', 0, 'validation')
    def fail(*args):
        pytest.fail('Cache hit should include blank features')
    monkeypatch.setattr(model.backbone, 'features', fail)
    reused = cached_features(model.backbone, ds, 'checkpoint', cfg['downstream'], 'cpu', 0, 'validation')
    torch.testing.assert_close(saved.blank_features, reused.blank_features)


def test_periodic_probe_isolated_and_validation_only(cfg):
    import json
    cfg['train']['epochs'] = 2
    cfg['data']['include_absolute_duration'] = False
    # Periodic probing must not even open the test split.
    (Path(cfg['data']['root']) / 'test.txt').unlink()
    control_cfg = deepcopy(cfg)
    control_cfg['train']['output_dir'] += '_control'
    control = run(control_cfg)
    cfg['train']['linear_probe'] = dict(every=1, epochs=2,
        regions=dict(mode='grid', grid=[1, 1, 1]), feature_cache=dict(storage='memory'))
    with_probe = run(cfg)
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, with_probe.state_dict()[key], rtol=0, atol=0)
    output = Path(cfg['train']['output_dir'])
    rows = [json.loads(line) for line in (output / 'probe_metrics.jsonl').read_text().splitlines()]
    assert [row['pretrained_epoch'] for row in rows] == [0, 1]
    assert all('blank_validation' in row and 'test' not in row for row in rows)
    assert not list((output / 'probes').rglob('*.pt'))
    assert not Path(cfg['downstream']['feature_cache']['dir']).exists()
    ssl_rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert 'mse_over_variance' in ssl_rows[0]['validation']['blank_features']


def test_temporary_probe_cache_cleanup(cfg, monkeypatch):
    from experiments.region_worldmodel import downstream
    from experiments.region_worldmodel.train import save_checkpoint
    checkpoint = Path(cfg['data']['root']) / 'pretrained.pt'
    save_checkpoint(checkpoint, RegionWorldModel(cfg), cfg, 3)
    cfg['downstream']['feature_cache']['storage'] = 'temporary'
    directories = []
    original = downstream.tempfile.TemporaryDirectory
    def temporary(*args, **kwargs):
        instance = original(*args, **kwargs)
        directories.append(Path(instance.name))
        return instance
    monkeypatch.setattr(downstream.tempfile, 'TemporaryDirectory', temporary)
    result = downstream_run(cfg, checkpoint, 'linear_probe')
    assert result['pretrained_epoch'] == 3
    assert 'blank_validation' in result and 'blank_test' in result
    assert directories and all(not path.exists() for path in directories)
    cfg['data']['include_absolute_duration'] = False
    with pytest.raises(ValueError, match='include_absolute_duration'):
        downstream_run(cfg, checkpoint, 'linear_probe')
    assert all(not path.exists() for path in directories)


def test_diagnostic_zero_variance_is_explicit():
    from experiments.region_worldmodel.diagnostics import feature_diagnostic
    result = feature_diagnostic(torch.ones(3, 4), torch.ones(3, 4))
    assert result['mse_over_variance'] is None
    assert result['feature_mse'] == 0


@pytest.mark.parametrize('overrides', [
    {'data': {'include_absolute_duration': 'false'}},
    {'downstream': {'feature_cache': {'storage': 'unknown'}}},
    {'train': {'linear_probe': {'every': -1}}},
    {'train': {'linear_probe': {'every': 2, 'epochs': 0}}}])
def test_invalid_probe_options(cfg, overrides):
    from experiments.region_worldmodel.config import validate
    for key, value in overrides.items():
        cfg[key].update(value)
    with pytest.raises(ValueError):
        validate(cfg)
