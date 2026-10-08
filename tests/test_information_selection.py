from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ecfm.data.tokenizer import Region
from experiments.hierarchical_mae.config import load_config, validate
from experiments.hierarchical_mae.data import HierarchyDataset, splits
from experiments.hierarchical_mae.information_selection import (
    METRICS, histogram_scores, selection_scores, voxel_histogram,
)
from experiments.hierarchical_mae.masking import make_plan
from experiments.hierarchical_mae.model import HierarchicalMAE
from experiments.hierarchical_mae.feature_cache import cache_metadata, cached_features
from experiments.hierarchical_mae.information_experiment import run_experiment, variants


@pytest.fixture
def cfg(tmp_path):
    config = load_config('experiments/hierarchical_mae/configs/smoke.yaml')
    config['data'].update(root=str(tmp_path), image_width=8, image_height=8, time_unit=1.,
                          crop_fraction=[1., 1.], information_selection={'bins': [4, 4, 4]})
    config['downstream'].update(num_classes=2, max_test_samples=0, max_val_batches=1)
    config['downstream']['feature_cache'].update(dir=str(tmp_path / 'features'), batch_size=2)
    rng = np.random.default_rng(2)
    for i in range(10):
        events = np.column_stack([rng.integers(0, 8, 32), rng.integers(0, 8, 32),
                                  np.linspace(0, 2, 32), rng.integers(0, 2, 32)])
        np.save(tmp_path / f'{i}.npy', events)
    (tmp_path / 'train.txt').write_text(''.join(f'{i}.npy {i % 2}\n' for i in range(8)))
    (tmp_path / 'test.txt').write_text('8.npy 0\n9.npy 1\n')
    return config


def test_equal_count_structure_beats_isolated_noise():
    edge, isolated = np.zeros((8, 8, 8)), np.zeros((8, 8, 8))
    edge[2:6, 3:5, 3] = 1
    isolated[::4, ::4, ::4] = 1
    assert edge.sum() == isolated.sum() == 8
    structured, noisy = histogram_scores(edge), histogram_scores(isolated)
    assert structured[0] > noisy[0] == 0
    assert structured[2] > noisy[2]
    # Entropy only measures concentration, not spatial arrangement.
    assert structured[1] == noisy[1]


def test_empty_uniform_hot_pixel_and_no_boundary_wrapping():
    np.testing.assert_array_equal(histogram_scores(np.zeros((4, 4, 4))), 0)
    scores = histogram_scores(np.ones((4, 4, 4)))
    assert scores[1] == scores[2] == 0
    hot = np.zeros((4, 4, 4))
    hot[1, 1, :] = 10
    assert histogram_scores(hot)[0] == 0  # Temporal repeats at one pixel do not support each other.
    boundary = np.zeros((4, 4, 4))
    boundary[0, 0, 0] = boundary[-1, 0, 0] = 1
    assert histogram_scores(boundary)[0] == 0
    # Entropy and correlation are invariant to multiplying every bin count.
    np.testing.assert_allclose(histogram_scores(hot)[1:], histogram_scores(hot * 10)[1:])


def test_autocorrelation_exact_permutation_null():
    # Enumerate every distinct shuffle of two ones among eight bins.
    scores = []
    for indices in combinations(range(8), 2):
        volume = np.zeros(8)
        volume[list(indices)] = 1
        scores.append(histogram_scores(volume.reshape(2, 2, 2))[2])
    assert abs(np.mean(scores)) < 1e-7


def test_voxel_coordinates_and_polarity_pooling():
    region = Region(10, 20, .5, 8, 8, .25, 'xy')
    events = np.array([[10, 20, .5, 0], [17, 27, .749, 1]])
    volume = voxel_histogram(events, region, [4, 4, 4])
    assert volume[0, 0, 0] == volume[3, 3, 3] == 1
    assert volume.sum() == 2


def test_blend_endpoints_eligible_normalization_product_and_ties():
    eligible = torch.tensor([[True, True, True, False]])
    view = dict(log_counts=torch.tensor([[1., 2., 3., 100.]]),
                information_scores=torch.tensor([[[1., 0., 0.], [.5, 0., 0.], [0., 0., 0.], [100., 0., 0.]]]),
                valid_mask=eligible)
    options = dict(strategy='activity_information', budget=1, activity_weight=0.)
    assert make_plan(view, None, selection=options).visible.tolist() == [[True, False, False, False]]
    options['activity_weight'] = 1.
    assert make_plan(view, None, selection=options).visible.tolist() == [[False, False, True, False]]
    options['activity_weight'] = .5
    assert make_plan(view, None, selection=options).visible.tolist() == [[True, False, False, False]]
    options['combination'] = 'product'
    torch.testing.assert_close(selection_scores(view, eligible, options), torch.tensor([[1., 1., 0., 10000.]]))
    with pytest.raises(ValueError, match='raw-voxel'):
        selection_scores({'log_counts': view['log_counts']}, eligible, options)


@pytest.mark.parametrize('strategy', ['information', 'activity_information'])
@pytest.mark.parametrize('metric', METRICS)
def test_scores_integrate_with_strict_masking_and_model(cfg, strategy, metric):
    cfg['selection'] = dict(strategy=strategy, metric=metric, budget=3)
    validate(cfg)
    train, _, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, train)
    view = next(iter(DataLoader(dataset, batch_size=2)))['source']
    assert view['information_scores'].shape == (2, dataset.layout.count, 3)
    # Identical voxel scores across the root's xt and cstr3 representations.
    torch.testing.assert_close(view['information_scores'][:, 0], view['information_scores'][:, 1])
    plan = make_plan(view, dataset.layout, dict(strategy='subtree', ratio=.5, level=1), cfg['selection'])
    assert plan.visible.sum(1).tolist() == [3, 3]
    assert not (plan.visible & plan.target).any()
    output = HierarchicalMAE(cfg)(view, plan=plan)
    assert torch.isfinite(output['loss'])


def test_patch_cache_preserves_scores_and_invalidates_settings(cfg, monkeypatch):
    from experiments.hierarchical_mae import data
    cfg['data']['patch_cache'] = dict(enabled=True, dir=str(Path(cfg['data']['root']) / 'patches'))
    train, _, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, train)
    original = dataset[0]['source']
    def fail(*args):
        raise AssertionError('Cache hit must not load events')
    with monkeypatch.context() as patch:
        patch.setattr(data, 'load_events', fail)
        cached = dataset[0]['source']
    torch.testing.assert_close(original['information_scores'], cached['information_scores'])
    changed = deepcopy(cfg)
    changed['data']['information_selection']['bins'] = [8, 8, 8]
    HierarchyDataset(changed, train)[0]
    assert len(list(Path(cfg['data']['patch_cache']['dir']).rglob('*.pt'))) == 2
    path, metadata = dataset.cache.entry(train[0][0], 1., 0., [0, 0, 8, 8])
    payload = torch.load(path, weights_only=True)
    payload['source']['information_scores'][0, 0] = float('nan')
    torch.save(payload, path)
    with pytest.raises(ValueError, match='Invalid patch cache'):
        dataset[0]


def test_feature_cache_single_record_slicing_and_batch_invariance(cfg, tmp_path):
    cfg['downstream']['selection'] = dict(strategy='activity_information', metric='support', budget=3)
    train, _, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, train)
    backbone = HierarchicalMAE(cfg).eval().requires_grad_(False)
    checkpoint = tmp_path / 'mae.pt'
    torch.save(dict(model=backbone.state_dict(), config=cfg), checkpoint)
    first = cached_features(backbone, dataset, checkpoint, 'cpu', 'train').tensors[0]
    before = deepcopy(cache_metadata(dataset, checkpoint))
    cfg['downstream']['feature_cache'].update(batch_size=1, rebuild=True)
    second = cached_features(backbone, dataset, checkpoint, 'cpu', 'train').tensors[0]
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    cfg['downstream']['selection']['metric'] = 'entropy'
    assert cache_metadata(dataset, checkpoint) != before


@pytest.mark.parametrize('field,value', [('bins', [1, 8, 8]), ('bins', [8, 8]),
                                          ('support_saturation', 0), ('support_saturation', float('nan'))])
def test_invalid_metric_settings(cfg, field, value):
    cfg['data']['information_selection'][field] = value
    with pytest.raises(ValueError, match='information_selection'):
        validate(cfg)


def test_automatic_opt_in_and_default_path(cfg):
    del cfg['data']['information_selection']
    train, _, _ = splits(cfg)
    assert 'information_scores' not in HierarchyDataset(cfg, train)[0]['source']
    cfg['downstream']['selection'] = dict(strategy='information', budget=3)
    assert 'information_scores' in HierarchyDataset(cfg, train)[0]['source']


@pytest.mark.parametrize('key,value', [('metric', 'missing'), ('combination', 'missing'),
                                      ('activity_weight', -1), ('activity_weight', float('nan'))])
def test_invalid_selection_options(cfg, key, value):
    cfg['downstream']['selection'] = dict(strategy='activity_information', budget=3, **{key: value})
    with pytest.raises(ValueError):
        validate(cfg)


@pytest.mark.parametrize('preset', ['information54', 'activity_information54'])
def test_thu_presets_and_dry_run(preset, tmp_path):
    cfg = load_config(f'experiments/hierarchical_mae/configs/thu_linear_probe_{preset}.yaml')
    plan = run_experiment(cfg, 'unread-checkpoint.pt', tmp_path / 'unused', dry_run=True)
    assert len(plan['selections']) == 18 and plan['seeds'] == [7, 17, 27]
    assert not (tmp_path / 'unused').exists()


def test_real_probe_suite_and_safe_continuation(cfg, tmp_path, monkeypatch):
    from experiments.hierarchical_mae import information_experiment
    checkpoint = tmp_path / 'mae.pt'
    torch.save(dict(model=HierarchicalMAE(cfg).state_dict(), config=cfg), checkpoint)
    output = tmp_path / 'experiment'
    result = run_experiment(cfg, checkpoint, output, seeds=[7], budget=3, metrics=['support'], weights=[.5])
    assert len(result['runs']) == 6
    assert set(result['summary']) == set(variants(3, ['support'], [.5]))
    assert all(r['result']['test']['samples'] == 2 for r in result['runs'])
    assert json.loads((output / 'results.json').read_text()) == result
    def fail(*args, **kwargs):
        raise AssertionError('Completed probes should be reused')
    with monkeypatch.context() as patch:
        patch.setattr(information_experiment, 'downstream_run', fail)
        assert run_experiment(cfg, checkpoint, output, [7], 3, ['support'], [.5]) == result
    # A checkpoint saved before final reporting should resume and finish evaluation.
    (output / 'support_seed7' / 'results.json').unlink()
    assert run_experiment(cfg, checkpoint, output, [7], 3, ['support'], [.5]) == result
    with pytest.raises(ValueError, match='differs'):
        run_experiment(cfg, checkpoint, output, [17], 3, ['support'], [.5])
    recording = tmp_path / '0.npy'
    events = np.load(recording)
    events[0, 0] = (events[0, 0] + 1) % 8
    np.save(recording, events)
    with pytest.raises(ValueError, match='differs'):
        run_experiment(cfg, checkpoint, output, [7], 3, ['support'], [.5])
