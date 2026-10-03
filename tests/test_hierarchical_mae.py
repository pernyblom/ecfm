from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from experiments.hierarchical_mae.config import load_config, validate
from experiments.hierarchical_mae.data import HierarchyDataset, Layout, render_cstr, splits
from experiments.hierarchical_mae.masking import TokenPlan, make_plan, overlaps, select
from experiments.hierarchical_mae.model import HierarchicalMAE
from experiments.hierarchical_mae.feature_cache import cache_metadata, cached_features
from experiments.hierarchical_mae.train import run
from experiments.hierarchical_mae.downstream import run as downstream_run
from experiments.hierarchical_mae.rendering import partition_level, render_histogram
from experiments.hierarchical_mae.loading import make_loader
from ecfm.data.tokenizer import Region, build_patch


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / 'experiments/hierarchical_mae/configs/smoke.yaml')
    cfg['data'].update(root=str(tmp_path), image_width=8, image_height=8, time_unit=1.,
                       crop_fraction=[1., 1.], eval_fraction=1.)
    cfg['train']['output_dir'] = str(tmp_path / 'out')
    cfg['downstream'].update(num_classes=2, max_test_samples=0)
    cfg['downstream']['feature_cache']['dir'] = str(tmp_path / 'features')
    rng = np.random.default_rng(2)
    for i in range(10):
        events = np.column_stack([rng.integers(0, 8, 32), rng.integers(0, 8, 32),
                                  np.linspace(0, 2, 32), rng.integers(0, 2, 32)])
        np.save(tmp_path / f'{i}.npy', events)
    (tmp_path / 'train.txt').write_text(''.join(f'{i}.npy {i%2}\n' for i in range(8)))
    (tmp_path / 'test.txt').write_text('8.npy 0\n9.npy 1\n')
    return cfg


def batch(cfg):
    train, _, _ = splits(cfg)
    return next(iter(DataLoader(HierarchyDataset(cfg, train), batch_size=2)))['source']


def test_counts_duration_endpoint_and_shapes(cfg):
    ds = HierarchyDataset(cfg, [(Path(cfg['data']['root'])/'0.npy', 0)])
    view = ds[0]['source']
    assert view['patches']['l0_xt'].shape == (1, 2, 8, 8)
    assert view['patches']['l0_cstr3'].shape == (1, 3, 8, 8)
    assert view['patches']['l1_xy'].shape == (8, 2, 4, 4)
    counts = view['log_counts'].expm1()
    assert counts[0].item() == pytest.approx(32)
    assert counts[2:10].sum().item() == pytest.approx(32)  # Includes t==last timestamp once.
    assert view['metadata'][0, 7].expm1().item() == pytest.approx(2)
    assert view['metadata'][2, 7].expm1().item() == pytest.approx(1)
    changed = deepcopy(cfg)
    changed['data']['time_unit'] = 10
    other = HierarchyDataset(changed, ds.entries)[0]['source']
    assert torch.equal(view['patches']['l0_xt'], other['patches']['l0_xt'])
    assert other['metadata'][0, 7].expm1().item() == pytest.approx(20)


def test_random_crop_is_epoch_specific_and_eval_fixed(cfg):
    cfg['data']['crop_fraction'] = [.25, .75]
    train, _, _ = splits(cfg)
    ds = HierarchyDataset(cfg, train, True)
    first = ds[0]['source']
    assert torch.equal(first['metadata'], ds[0]['source']['metadata'])
    ds.epoch = 1
    assert not torch.equal(first['metadata'], ds[0]['source']['metadata'])
    ds.training = False
    fixed = ds[0]['source']
    ds.epoch = 99
    assert torch.equal(fixed['metadata'], ds[0]['source']['metadata'])


def test_cstr_local_time_and_count_variants():
    events = np.array([[2, 3, .25, 1], [2, 3, .75, 0], [2, 3, .5, 1]])
    region = Region(2, 3, 0., 1, 1, 1., 'cstr3')
    torch.testing.assert_close(render_cstr(events, region, 1)[:, 0, 0], torch.tensor([.375, 1., .75]))
    assert render_cstr(events, region, 1, max_count=6)[1, 0, 0] == .5
    assert render_cstr(events, region, 1, include_count=False)[1].sum() == 0


@pytest.mark.parametrize('strategy', ['voxel', 'subtree'])
def test_strict_mask_excludes_all_overlapping_tokens(cfg, strategy):
    view, layout = batch(cfg), Layout(cfg)
    plan = make_plan(view, layout, dict(strategy=strategy, ratio=.5, level=1))
    for visible, target in zip(plan.visible, plan.target):
        assert not overlaps(layout.boxes)[visible][:, target].any()
    assert not plan.visible[:, :2].any()  # Both root representations excluded.
    assert plan.target.sum() > 0


def test_mixed_policy_retains_overlapping_hierarchy(cfg):
    view, layout = batch(cfg), Layout(cfg)
    torch.manual_seed(4)
    plan = make_plan(view, layout, dict(strategy='mixed', ratio=.5, level=1, overlap_probability=1.))
    assert any(overlaps(layout.boxes)[v][:, t].any() for v, t in zip(plan.visible, plan.target))
    strict = make_plan(view, layout, dict(strategy='mixed', ratio=.5, level=1, overlap_probability=0.))
    assert not strict.visible[:, :2].any()


def test_strict_subtrees_keep_visible_cross_level_examples(cfg):
    cfg['hierarchy']['levels'].append(dict(splits=[2, 2, 2], patch_size=2, representations=['xt']))
    layout, view = Layout(cfg), batch(cfg)
    plan = make_plan(view, layout, dict(strategy='subtree', ratio=.5, level=1))
    descendants = layout.descendants([0])
    assert descendants.shape == (1, layout.count)
    assert descendants[0, 2:].all() and not descendants[0, :2].any()
    for visible, target in zip(plan.visible, plan.target):
        assert not overlaps(layout.boxes)[visible][:, target].any()
        context_overlap = overlaps(layout.boxes)[visible][:, visible]
        context_overlap.fill_diagonal_(False)
        assert context_overlap.any()
        assert target[layout.level_ids == 2].sum() == 4


def test_masked_content_and_counts_cannot_reach_encoder_or_decoder(cfg):
    view = batch(cfg)
    model = HierarchicalMAE(cfg).eval()
    plan = make_plan(view, model.layout, dict(strategy='token', ratio=.5))
    changed = deepcopy(view)
    changed['log_counts'][~plan.visible] = 10000
    for g in model.layout.groups:
        changed['patches'][g.key][~plan.visible[:, g.start:g.stop]] = 10000
    first, second = model(view, plan), model(changed, plan)
    for key in first['predictions']:
        torch.testing.assert_close(first['predictions'][key], second['predictions'][key])
    torch.testing.assert_close(first['count_predictions'], second['count_predictions'])
    assert first['loss'] != second['loss']  # Targets still affect objective.


def test_sparse_attention_padding_and_differentiable_selection(cfg):
    view = batch(cfg)
    model = HierarchicalMAE(cfg).eval()
    scores = torch.randn_like(view['log_counts'], requires_grad=True)
    visible, gates = select(view['valid_mask'], model.layout, {'budget': 4}, view['log_counts'],
                            scores=scores, straight_through=True)
    visible[0, visible[0].nonzero()[0]] = False
    plan = TokenPlan(visible, torch.zeros_like(visible), gates)
    features = model.features(view, plan)
    encoded, _, padding = model.encode(view, plan)
    assert encoded.shape[1] == 4 and padding[0].sum() == 1
    features.square().sum().backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0
    single = {k: {g: p[:1] for g, p in v.items()} if isinstance(v, dict) else v[:1] for k, v in view.items()}
    torch.testing.assert_close(features[:1], model.features(single, TokenPlan(visible[:1], plan.target[:1], gates[:1])))


def test_activity_budget_and_empty_plan_rejected(cfg):
    view = batch(cfg)
    model = HierarchicalMAE(cfg)
    visible, _ = select(view['valid_mask'], model.layout, {'strategy': 'activity', 'budget': 2}, view['log_counts'])
    assert visible[:, :2].all()  # Full-volume counts are largest.
    assert visible.sum() == 4
    with pytest.raises(ValueError, match='visible'):
        model.features(view, TokenPlan(torch.zeros_like(visible), torch.zeros_like(visible)))


def test_invalid_layout(cfg):
    cfg['hierarchy']['levels'][1]['splits'] = [3, 2, 2]
    with pytest.raises(ValueError, match='nest'):
        validate(cfg)


def test_pretrain_resume_probe_finetune_cache_and_inspection(cfg):
    torch.set_num_threads(1)
    model = run(cfg)
    output = Path(cfg['train']['output_dir'])
    checkpoint = output / 'best.pt'
    assert (output / 'patches/epoch_0001.png').is_file()
    cfg['train']['epochs'] = 2
    run(cfg, output / 'last.pt')
    for mode in ('linear_probe', 'finetune'):
        result = downstream_run(cfg, checkpoint, mode)
        assert result['test']['samples'] == 2
    train, _, _ = splits(cfg)
    ds = HierarchyDataset(cfg, train)
    state = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(state['model'])
    model.requires_grad_(False)
    first = cached_features(model, ds, checkpoint, 'cpu', 'train')
    second = cached_features(model, ds, checkpoint, 'cpu', 'train')
    torch.testing.assert_close(first.tensors[0], second.tensors[0])
    before = cache_metadata(ds, checkpoint)
    cfg['downstream']['selection'] = dict(strategy='coarse', budget=2)
    assert before != cache_metadata(ds, checkpoint)
    changed = cached_features(model, ds, checkpoint, 'cpu', 'train')
    assert not torch.allclose(first.tensors[0], changed.tensors[0])
    assert len(list(Path(cfg['downstream']['feature_cache']['dir']).glob('train_*.pt'))) == 2


@pytest.mark.parametrize('norm', ['none', 'region_max', 'region_sum', 'region_mean'])
@pytest.mark.parametrize('plane', ['xy', 'xt', 'yt', 'xy_p45', 'yt_m45'])
def test_fast_histograms_match_existing_renderer(plane, norm):
    rng = np.random.default_rng(11)
    events = np.column_stack((rng.integers(2, 7, 160), rng.integers(3, 10, 160),
                              rng.uniform(.2, .6, 160), rng.integers(0, 2, 160))).astype(np.float32)
    region = Region(2, 3, .2, 5, 7, .4, plane)
    for sub in (events, events[:0]):
        expected, _ = build_patch(sub, region, 6, 9, norm_mode=norm)
        actual = render_histogram(sub, region, 6, 9, norm)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('grid', [[4, 4, 8], [3, 2, 3], [1, 1, 1]])
def test_partition_matches_half_open_regions_and_preserves_order(grid):
    width, height = 13, 9
    nx, ny, nt = grid
    rng = np.random.default_rng(3)
    events = np.column_stack((rng.integers(-1, width+1, 500), rng.integers(-1, height+1, 500),
                              rng.uniform(0, 1, 500), rng.integers(0, 2, 500))).astype(np.float32)
    # Include exact boundaries, endpoint and an invalid event.
    edges = np.array([[0, 0, t/nt, 1] for t in range(nt+1)], dtype=np.float32)
    events = np.concatenate((events, edges))
    voxels = partition_level(events, grid, grid, width, height)
    for t in range(nt):
        for y in range(ny):
            for x in range(nx):
                x0, x1 = x*width//nx, (x+1)*width//nx
                y0, y1 = y*height//ny, (y+1)*height//ny
                expected = events[(events[:, 0] >= x0) & (events[:, 0] < x1)
                    & (events[:, 1] >= y0) & (events[:, 1] < y1)
                    & (events[:, 2] >= t/nt) & (events[:, 2] < t/nt+1/nt)]
                np.testing.assert_array_equal(voxels[(t*ny+y)*nx+x], expected)


def enable_patch_cache(cfg, views=2):
    cfg['data']['patch_cache'] = dict(enabled=True, train_views=views,
                                     dir=str(Path(cfg['data']['root']) / 'patch_cache'))
    cfg['data']['crop_fraction'] = [.3, .9]


def assert_views_equal(first, second):
    for key in ('metadata', 'log_counts', 'valid_mask'):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for key in first['patches']:
        torch.testing.assert_close(first['patches'][key], second['patches'][key], rtol=0, atol=0)


def test_patch_cache_hit_skips_events_and_rendering_and_keeps_current_label(cfg, monkeypatch):
    enable_patch_cache(cfg)
    train, _, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, train, True)
    first = dataset[0]['source']
    from experiments.hierarchical_mae import data
    def fail(*args):
        raise AssertionError('Cache hit must not load raw events')
    monkeypatch.setattr(data, 'load_events', fail)
    dataset.entries[0] = (dataset.entries[0][0], 1)
    cached = dataset[0]
    assert cached['label'] == 1
    assert_views_equal(first, cached['source'])
    assert len(list(Path(cfg['data']['patch_cache']['dir']).rglob('*.pt'))) == 1


def test_crop_bank_repeats_but_uncached_random_crops_stay_fresh(cfg):
    enable_patch_cache(cfg)
    train, _, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, train, True)
    first, second = dataset[(0, 0)]['source'], dataset[(0, 1)]['source']
    assert not torch.equal(first['metadata'], second['metadata'])
    assert_views_equal(first, dataset[(0, 2)]['source'])
    assert len(list(Path(cfg['data']['patch_cache']['dir']).rglob('*.pt'))) == 2
    cfg['data']['patch_cache']['train_views'] = 0
    fresh = HierarchyDataset(cfg, train, True)
    assert_views_equal(first, fresh[(0, 0)]['source'])
    assert not torch.equal(first['metadata'], fresh[(0, 2)]['source']['metadata'])
    # Fresh crops do not fill the disk with one-use cache entries.
    assert len(list(Path(cfg['data']['patch_cache']['dir']).rglob('*.pt'))) == 2
    evaluation = HierarchyDataset(cfg, train)
    fixed = evaluation[0]['source']
    assert_views_equal(fixed, evaluation[(0, 100)]['source'])
    assert len(list(Path(cfg['data']['patch_cache']['dir']).rglob('*.pt'))) == 3


def test_patch_cache_invalidation_and_corruption(cfg):
    enable_patch_cache(cfg)
    train, _, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, train, True)
    original = dataset[0]['source']
    changed = deepcopy(cfg)
    changed['data']['time_unit'] = 3
    different = HierarchyDataset(changed, train, True)[0]['source']
    assert not torch.equal(original['metadata'], different['metadata'])
    entries = list(Path(cfg['data']['patch_cache']['dir']).rglob('*.pt'))
    assert len(entries) == 2
    source, _ = train[0]
    events = np.load(source)
    events[:, 3] = 1-events[:, 3]
    np.save(source, events)
    updated = dataset[0]['source']
    assert not torch.equal(original['patches']['l0_xt'], updated['patches']['l0_xt'])
    entries = list(Path(cfg['data']['patch_cache']['dir']).rglob('*.pt'))
    assert len(entries) == 3
    for entry in entries:
        entry.write_bytes(b'incomplete cache')
    with pytest.raises(ValueError, match='Invalid patch cache'):
        dataset[0]
    cfg['data']['patch_cache']['rebuild'] = True
    assert_views_equal(updated, HierarchyDataset(cfg, train, True)[0]['source'])


def test_persistent_spawn_workers_receive_current_epoch(cfg):
    cfg['data']['crop_fraction'] = [.3, .9]
    cfg['train']['persistent_workers'] = True
    train, _, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, train, True)
    loader = make_loader(dataset, 2, workers=1)
    first = next(iter(loader))['source']['metadata'][0]
    dataset.epoch = 1
    second = next(iter(loader))['source']['metadata'][0]
    assert not torch.equal(first, second)
    torch.testing.assert_close(second, dataset[0]['source']['metadata'])
    del loader


def test_enable_cache_when_resuming_old_checkpoint(cfg):
    run(cfg)
    checkpoint = Path(cfg['train']['output_dir']) / 'last.pt'
    enable_patch_cache(cfg)
    # Changing crop range remains a semantic mismatch; cache settings alone are allowed.
    cfg['data']['crop_fraction'] = [1., 1.]
    cfg['train']['epochs'] = 2
    run(cfg, checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    assert state['epoch'] == 1
    assert state['config']['data']['patch_cache']['train_views'] == 2


@pytest.mark.parametrize('mode', ['linear_probe', 'finetune'])
def test_downstream_resume_matches_uninterrupted_training(cfg, mode):
    torch.set_num_threads(1)
    run(cfg)
    root = Path(cfg['train']['output_dir'])
    pretrained = root / 'best.pt'
    cfg['downstream']['epochs'] = 3
    downstream_run(cfg, pretrained, mode, root / 'continuous')
    expected = torch.load(root / 'continuous/last.pt', weights_only=True)
    cfg['downstream']['epochs'] = 1
    downstream_run(cfg, pretrained, mode, root / 'interrupted')
    cfg['downstream']['epochs'] = 3
    # New-format downstream checkpoints are self-contained; probing still reuses its cache key.
    pretrained.rename(root / 'moved_pretrained.pt')
    downstream_run(cfg, output_dir=root / 'resumed', resume=root / 'interrupted/last.pt')
    actual = torch.load(root / 'resumed/last.pt', weights_only=True)
    assert actual['epoch'] == expected['epoch'] == 2
    assert actual['best_state']['epoch'] == expected['best_state']['epoch']
    assert actual['validation'] == expected['validation']
    for key in expected['model']:
        torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)
    assert actual['optimizer']['param_groups'] == expected['optimizer']['param_groups']
    for key, values in expected['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(actual['optimizer']['state'][key][name], value, rtol=0, atol=0)
    assert (root / 'resumed/best.pt').is_file()
    assert len(list(Path(cfg['downstream']['feature_cache']['dir']).glob('train_*.pt'))) == (1 if mode == 'linear_probe' else 0)


def test_downstream_legacy_resume_preserves_best_without_improvement(cfg, monkeypatch):
    run(cfg)
    root = Path(cfg['train']['output_dir'])
    downstream_run(cfg, root / 'best.pt', 'linear_probe')
    legacy = torch.load(root / 'linear_probe/best.pt', weights_only=True)
    for key in ('optimizer', 'loader_generator_state', 'pretrained_digest'):
        legacy.pop(key)
    path = root / 'legacy.pt'
    torch.save(legacy, path)
    cfg['downstream']['epochs'] = 2
    from experiments.hierarchical_mae import downstream
    monkeypatch.setattr(downstream, 'classification_epoch', lambda *args, **kwargs:
                        dict(loss=legacy['validation']['loss']+10, accuracy=0., samples=2))
    with pytest.warns(UserWarning, match='fresh optimizer'):
        result = downstream_run(cfg, resume=path, output_dir=root / 'legacy_resume')
    assert result['best_epoch'] == legacy['epoch']
    saved = torch.load(root / 'legacy_resume/best.pt', weights_only=True)
    for key in legacy['model']:
        torch.testing.assert_close(saved['model'][key], legacy['model'][key])
    assert torch.load(root / 'legacy_resume/last.pt', weights_only=True)['epoch'] == 1


def test_downstream_resume_rejects_changed_observations_and_mode(cfg):
    run(cfg)
    root = Path(cfg['train']['output_dir'])
    downstream_run(cfg, root / 'best.pt', 'linear_probe')
    path = root / 'linear_probe/last.pt'
    with pytest.raises(ValueError, match='mode differs'):
        downstream_run(cfg, mode='finetune', resume=path)
    cfg['downstream']['selection'] = dict(strategy='random', budget=2)
    with pytest.raises(ValueError, match='downstream differs'):
        downstream_run(cfg, resume=path)
