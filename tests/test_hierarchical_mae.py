from copy import deepcopy
from pathlib import Path
import os

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
from experiments.hierarchical_mae.downstream import save_checkpoint
from experiments.hierarchical_mae.rendering import partition_level, render_histogram
from experiments.hierarchical_mae.loading import make_loader
from ecfm.data.tokenizer import Region, build_patch
from experiments.hierarchical_mae.learned_selector import SelectionEncoder, relaxed_topk, schedule
from experiments.hierarchical_mae.downstream import Classifier


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


def test_set_policy_uses_only_root_content_and_one_transformer_pass(cfg):
    from experiments.hierarchical_mae.set_selector import SetPolicy, coarse_inputs, selected_features
    model = HierarchicalMAE(cfg).eval().requires_grad_(False)
    view = batch(cfg)
    roots, descriptors = coarse_inputs(model, view, 6, 2, 2)
    changed = deepcopy(view)
    for group in model.layout.groups:
        if group.level > 0:
            changed['patches'][group.key].fill_(1000.)
    roots2, descriptors2 = coarse_inputs(model, changed, 6, 2, 2)
    torch.testing.assert_close(roots, roots2)
    torch.testing.assert_close(descriptors, descriptors2)
    policy = SetPolicy(roots.shape[-1], descriptors.shape[-1], 2, 2, 2).eval()
    policy.fit_normalization(roots, descriptors)
    expected = model.features(view, selection=dict(strategy='activity', budget=6))
    calls = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
    actual, choices = selected_features(model, view, policy, 0., 6)
    hook.remove()
    assert choices.eq(0).all() and calls == [6]
    torch.testing.assert_close(actual, expected)
    # Supervised gains must give a nonzero gradient to the initialized scorer.
    loss = (policy(roots, descriptors)[:, 1:]-1).square().mean()
    loss.backward()
    assert policy.scorer[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('limit,sets', [(1, 25), (2, 115), (4, 210)])
def test_set_policy_hard_budget_and_noop(limit, sets):
    from experiments.hierarchical_mae.set_selector import SetPolicy
    policy = SetPolicy(12, 18, max_swaps=limit).eval()
    assert len(policy.states) == sets
    scores = torch.eye(sets)
    choices = policy.choose(scores, 0.)
    assert torch.equal(choices, torch.arange(sets))
    assert policy.depths[choices].max() == limit
    assert policy.choose(torch.zeros_like(scores)).eq(0).all()
    assert policy.choose(scores, 2.).eq(0).all()
    with pytest.raises(ValueError):
        policy.choose(scores, -.01)


def test_set_policy_checkpoint_loader_and_probe_identity(cfg, tmp_path):
    from experiments.hierarchical_mae.set_selector import SetPolicy, coarse_inputs, load_policy
    from experiments.hierarchical_mae.feature_cache import file_digest
    baseline = Classifier(HierarchicalMAE(cfg), cfg['downstream']['num_classes'], True).eval()
    view = batch(cfg)
    roots, descriptors = coarse_inputs(baseline.backbone, view, 6, 2, 2)
    policy = SetPolicy(roots.shape[-1], descriptors.shape[-1], 2, 2, 2).eval()
    baseline_path, policy_path, probe_path = [tmp_path/name for name in ('baseline.pt', 'policy.pt', 'probe.pt')]
    torch.save(dict(config=cfg, model=baseline.state_dict()), baseline_path)
    torch.save(dict(architecture=policy.settings, policy=policy.state_dict(), threshold=0., budget=6,
                    baseline=str(baseline_path), baseline_digest=file_digest(baseline_path)), policy_path)
    torch.save(dict(head=baseline.head.state_dict(), policy_digest=file_digest(policy_path)), probe_path)
    loaded = load_policy(policy_path, probe_path)
    expected = baseline.head(baseline.backbone.features(view, selection=dict(strategy='activity', budget=6)))
    torch.testing.assert_close(loaded(view), expected)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    torch.save(dict(head=baseline.head.state_dict(), policy_digest='different'), probe_path)
    with pytest.raises(ValueError, match='different policy'):
        load_policy(policy_path, probe_path)


def test_swap_proposals_preserve_activity_ties_budget_and_one_change():
    from experiments.hierarchical_mae.swap_selection import proposals
    count = torch.tensor([[9., 9., 8., 8., 7., 7., 6., 100.]])
    valid = torch.ones_like(count, dtype=torch.bool)
    valid[:, -1] = False
    ids, removed, inserted = proposals(count, valid, 4, 2, 3)
    assert ids.shape == (1, 7, 4)
    assert ids[0, 0].tolist() == [0, 1, 2, 3]
    base = set(ids[0, 0].tolist())
    for proposal in ids[0, 1:]:
        values = set(proposal.tolist())
        assert len(values) == 4 and len(values-base) == len(base-values) == 1
        assert 7 not in values
    with pytest.raises(ValueError):
        proposals(count, valid, 6, 2, 3)


def test_activity_coarse_overlap_uses_lowest_levels(cfg):
    from experiments.hierarchical_mae.swap_selection import activity_coarse_overlap
    layout = Layout(cfg)
    counts = (-layout.level_ids).float()[None]
    valid = torch.ones_like(counts, dtype=torch.bool)
    overlap, identical = activity_coarse_overlap(counts, valid, layout, 4)
    assert overlap.item() == 1 and identical.item() == 1
    counts[:, -4:] = 100
    overlap, identical = activity_coarse_overlap(counts, valid, layout, 4)
    assert overlap.item() == 0 and identical.item() == 0


def test_swap_policy_noop_and_deployment_match_enumerated_features(cfg):
    from experiments.hierarchical_mae.swap_selection import proposals, policy_inputs, SwapPolicy, selected_features
    model = HierarchicalMAE(cfg).eval().requires_grad_(False)
    view = batch(cfg)
    ids, removed, inserted = proposals(view['log_counts'], view['valid_mask'], 4, 2, 2)
    coarse, local = policy_inputs(model, view, removed, inserted)
    policy = SwapPolicy(local.shape[-1]).eval()
    policy.fit_normalization(local)
    actual, choices = selected_features(model, view, policy, 0., 4, 2, 2)
    assert choices.eq(0).all()
    expected = model.features(view, selection=dict(strategy='activity', budget=4))
    torch.testing.assert_close(actual, expected)
    # Constant positive prediction chooses first replacement; compare direct gather.
    with torch.no_grad():
        policy.net[-1].bias.fill_(1.)
    actual, choices = selected_features(model, view, policy, 0., 4, 2, 2)
    assert choices.eq(1).all()
    embedding = model.token_embeddings(view)
    packed = embedding.gather(1, ids[:, 1, :, None].expand(-1, -1, embedding.shape[-1]))
    expected = model.encoder(packed, src_key_padding_mask=torch.zeros(packed.shape[:2], dtype=torch.bool)).mean(1)
    torch.testing.assert_close(actual, expected)


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


def test_spatial_crops_filter_translate_tile_and_cache(cfg, monkeypatch):
    from experiments.hierarchical_mae import data
    enable_patch_cache(cfg)
    cfg['data'].update(crop_fraction=[1., 1.], spatial_crop_fraction=[.5, .5])
    validate(cfg)
    train, _, _ = splits(cfg)
    events = np.array([[x, y, t, (x+y) % 2] for t in (0., 1.)
                       for y in range(8) for x in range(8)], dtype=np.float32)
    monkeypatch.setattr(data, 'load_events', lambda *args: (events.copy(), 2.))
    dataset = HierarchyDataset(cfg, train, True)
    first = dataset[0]['source']
    counts = first['log_counts'].expm1()
    assert counts[0].item() == pytest.approx(32)
    assert counts[2:10].sum().item() == pytest.approx(32)
    entry = next(Path(cfg['data']['patch_cache']['dir']).rglob('*.pt'))
    payload = torch.load(entry, weights_only=True)
    x, y, width, height = payload['metadata']['spatial_crop']
    assert (width, height) == (4, 4)
    expected = events[(events[:, 0] >= x) & (events[:, 0] < x+width)
                      & (events[:, 1] >= y) & (events[:, 1] < y+height)].copy()
    expected[:, 0] -= x
    expected[:, 1] -= y
    local = deepcopy(cfg)
    local['data'].update(image_width=4, image_height=4, spatial_crop_fraction=[1., 1.],
                         patch_cache={'enabled': False})
    monkeypatch.setattr(data, 'load_events', lambda *args: (expected.copy(), 2.))
    assert_views_equal(first, HierarchyDataset(local, train)[0]['source'])
    def fail(*args):
        raise AssertionError('Spatial crop cache hit must skip event loading')
    monkeypatch.setattr(data, 'load_events', fail)
    assert_views_equal(first, dataset[(0, 2)]['source'])
    other_path, _ = dataset.cache.entry(train[0][0], 1., 0., [(x+1) % 5, y, width, height])
    assert other_path != entry
    monkeypatch.setattr(data, 'load_events', lambda *args: (events.copy(), 2.))
    evaluation = HierarchyDataset(cfg, train)[0]['source']
    assert evaluation['log_counts'][0].expm1().item() == pytest.approx(128)


def test_spatial_crop_defaults_and_epoch_seeds(cfg):
    cfg['data'].pop('spatial_crop_fraction', None)  # Test the omitted-setting default, not the training YAML.
    train, _, _ = splits(cfg)
    baseline = HierarchyDataset(cfg, train, True)[0]['source']
    cfg['data']['spatial_crop_fraction'] = [1., 1.]
    assert_views_equal(baseline, HierarchyDataset(cfg, train, True)[0]['source'])
    cfg['data']['spatial_crop_fraction'] = [.5, .9]
    ds = HierarchyDataset(cfg, train, True)
    first = ds[(0, 0)]['source']
    assert_views_equal(first, ds[(0, 0)]['source'])
    assert any(not torch.equal(first['patches']['l0_xt'], ds[(0, epoch)]['source']['patches']['l0_xt'])
               for epoch in range(1, 5))


@pytest.mark.parametrize('value', [[0., 1.], [.9, .5], [.5, 1.1], [.1, .2],
                                  [float('nan'), 1.], [.5], .5, [True, 1.]])
def test_invalid_spatial_crop_fraction(cfg, value):
    cfg['data']['spatial_crop_fraction'] = value
    with pytest.raises(ValueError, match='spatial_crop_fraction'):
        validate(cfg)


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


@pytest.mark.parametrize('power', [1., 2.])
def test_activity_random_follows_sampling_weights(power):
    torch.manual_seed(8)
    activity = torch.tensor([0., 1., 3.]).expand(20000, -1)
    selected, _ = select(torch.ones_like(activity, dtype=torch.bool), None,
                         dict(strategy='activity_random', budget=1, activity_power=power), activity)
    observed = selected.float().mean(0)
    expected = (1+activity[0]).pow(power)
    expected /= expected.sum()
    torch.testing.assert_close(observed, expected, atol=.015, rtol=0)
    assert observed[2] > observed[1] > observed[0] > 0


@pytest.mark.parametrize('budget', [0, 1, 3, 100])
def test_activity_random_budget_eligibility_and_reproducibility(budget):
    activity = torch.tensor([[0., 1., 2., 100.], [0., 0., 0., 0.]])
    eligible = torch.tensor([[True, True, True, False], [False, True, True, False]])
    options = dict(strategy='activity_random', budget=budget)
    torch.manual_seed(12)
    visible, _ = select(eligible, None, options, activity)
    torch.manual_seed(12)
    repeated, _ = select(eligible, None, options, activity)
    assert torch.equal(visible, repeated)
    assert not (visible & ~eligible).any()
    expected = eligible.sum(1).clamp_max(budget) if budget else eligible.sum(1)
    assert torch.equal(visible.sum(1), expected)


@pytest.mark.parametrize('empty', [False, True])
def test_activity_random_uniform_limits_match_random(empty):
    activity = torch.zeros(10, 20) if empty else torch.arange(20).float().expand(10, -1)
    eligible = torch.ones_like(activity, dtype=torch.bool)
    torch.manual_seed(4)
    weighted, _ = select(eligible, None, dict(strategy='activity_random', budget=5,
                                            activity_power=1. if empty else 0.), activity)
    torch.manual_seed(4)
    uniform, _ = select(eligible, None, dict(strategy='random', budget=5), activity)
    assert torch.equal(weighted, uniform)


@pytest.mark.parametrize('power', [-1, float('inf'), float('nan'), True, '1'])
def test_activity_random_invalid_power(cfg, power):
    cfg['downstream']['selection'] = dict(strategy='activity_random', budget=3, activity_power=power)
    with pytest.raises(ValueError, match='activity_power'):
        validate(cfg)


def test_activity_random_strict_masking_and_feature_cache(cfg):
    cfg['selection'] = dict(strategy='activity_random', budget=4)
    cfg['downstream']['selection'] = dict(strategy='activity_random', budget=4, activity_power=1.)
    validate(cfg)
    view = batch(cfg)
    model = HierarchicalMAE(cfg)
    plan = make_plan(view, model.layout, dict(strategy='subtree', ratio=.5, level=1), cfg['selection'])
    assert plan.visible.sum(1).eq(4).all()
    for visible, target in zip(plan.visible, plan.target):
        assert not overlaps(model.layout.boxes)[visible][:, target].any()
    model(view, plan)['loss'].backward()
    checkpoint = Path(cfg['data']['root']) / 'weighted.pt'
    torch.save(model.state_dict(), checkpoint)
    model.requires_grad_(False)
    entries, _, _ = splits(cfg)
    dataset = HierarchyDataset(cfg, entries)
    cfg['downstream']['feature_cache']['batch_size'] = 2
    first = cached_features(model, dataset, checkpoint, 'cpu', 'train')
    second = cached_features(model, dataset, checkpoint, 'cpu', 'train')
    torch.testing.assert_close(first.tensors[0], second.tensors[0], rtol=0, atol=0)
    cfg['downstream']['feature_cache'].update(batch_size=3, rebuild=True)
    rebuilt = cached_features(model, dataset, checkpoint, 'cpu', 'train')
    torch.testing.assert_close(first.tensors[0], rebuilt.tensors[0])
    before = deepcopy(cache_metadata(dataset, checkpoint))
    cfg['downstream']['selection']['activity_power'] = 2.
    assert before != cache_metadata(dataset, checkpoint)


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


def test_checkpoint_retries_replace_without_reserializing(tmp_path, monkeypatch):
    from experiments.hierarchical_mae import downstream
    path = tmp_path / 'last.pt'
    save_checkpoint(path, dict(epoch=1))
    replace, serialize = os.replace, torch.save
    attempts, writes, sleeps = [], [], []
    def transient(source, destination):
        attempts.append(source)
        assert torch.load(path, weights_only=True)['epoch'] == 1
        if len(attempts) < 3:
            raise PermissionError('Simulated Windows lock')
        replace(source, destination)
    def counted_save(*args, **kwargs):
        writes.append(1)
        return serialize(*args, **kwargs)
    monkeypatch.setattr(downstream.os, 'replace', transient)
    monkeypatch.setattr(downstream.torch, 'save', counted_save)
    monkeypatch.setattr(downstream.time, 'sleep', sleeps.append)
    with pytest.warns(RuntimeWarning, match='retrying'):
        save_checkpoint(path, dict(epoch=2))
    assert len(attempts) == 3 and len(set(attempts)) == 1 and len(writes) == 1
    assert sleeps == [.1, .2]
    assert torch.load(path, weights_only=True)['epoch'] == 2
    assert list(tmp_path.glob('*.pt')) == [path]


def test_checkpoint_persistent_lock_preserves_old_and_recovery(tmp_path, monkeypatch):
    from experiments.hierarchical_mae import downstream
    path = tmp_path / 'last.pt'
    save_checkpoint(path, dict(epoch=1))
    attempts, sleeps = [], []
    def locked(*args):
        attempts.append(1)
        raise PermissionError('Persistent lock')
    monkeypatch.setattr(downstream.os, 'replace', locked)
    monkeypatch.setattr(downstream.time, 'sleep', sleeps.append)
    with pytest.warns(RuntimeWarning), pytest.raises(PermissionError, match='preserved at'):
        save_checkpoint(path, dict(epoch=2, optimizer={'step': torch.tensor(42)}))
    assert len(attempts) == 8 and sum(sleeps) == pytest.approx(4.5)
    assert torch.load(path, weights_only=True)['epoch'] == 1
    recovery, = tmp_path.glob('last.recovery-*.pt')
    recovered = torch.load(recovery, weights_only=True)
    assert recovered['epoch'] == 2 and recovered['optimizer']['step'] == 42


def test_checkpoint_serialization_failure_keeps_previous_file(tmp_path, monkeypatch):
    from experiments.hierarchical_mae import downstream
    path = tmp_path / 'last.pt'
    save_checkpoint(path, dict(epoch=1))
    def fail(state, stream):
        stream.write(b'partial')
        raise OSError('Disk full')
    monkeypatch.setattr(downstream.torch, 'save', fail)
    with pytest.raises(OSError, match='Disk full'):
        save_checkpoint(path, dict(epoch=2))
    assert torch.load(path, weights_only=True)['epoch'] == 1
    assert list(tmp_path.glob('*.pt')) == [path]


@pytest.mark.skipif(os.name != 'nt', reason='Windows denies replacement of an open checkpoint')
def test_checkpoint_real_windows_file_lock(tmp_path, monkeypatch):
    from experiments.hierarchical_mae import downstream
    path = tmp_path / 'last.pt'
    save_checkpoint(path, dict(epoch=1))
    handle = path.open('rb')
    sleeps = []
    def release_lock(delay):
        sleeps.append(delay)
        handle.close()
    monkeypatch.setattr(downstream.time, 'sleep', release_lock)
    try:
        with pytest.warns(RuntimeWarning, match='retrying'):
            save_checkpoint(path, dict(epoch=2))
    finally:
        handle.close()
    assert sleeps == [.1]
    assert torch.load(path, weights_only=True)['epoch'] == 2


def with_selector(cfg, context='patch'):
    cfg['downstream']['learned_selector'] = dict(budget=6, context=context, hidden_dim=16,
        lr=.001, activity_prior_weight=1., temperature=[1., .3], noise_scale=[1., 0.],
        anneal_epochs=3, training_crops=False, grad_clip=1.)
    return cfg


def test_relaxed_topk_is_hard_forward_and_reaches_unselected_scores():
    scores = torch.tensor([[.1, .5, -.3, .2, .6]], requires_grad=True)
    eligible = torch.tensor([[True, True, False, True, True]])
    values = torch.tensor([[[1., 2.], [4., 0.], [99., 99.], [-2., 1.], [.5, -3.]]])
    matrix, ids = relaxed_topk(scores, eligible, 2, temperature=.8)
    assert ids.tolist() == [[4, 1]]
    assert matrix.sum().item() == 2 and (matrix.detach().sum(1) <= 1).all()
    torch.testing.assert_close(matrix @ values, values[:, [4, 1]], rtol=0, atol=0)
    (matrix @ values).square().sum().backward()
    assert scores.grad[0, 0].abs() > 0 and scores.grad[0, 3].abs() > 0
    assert scores.grad[0, 2] == 0 and torch.isfinite(scores.grad).all()


def test_relaxed_topk_noise_temperature_and_schedule():
    scores = torch.zeros(32, 8)
    valid = torch.ones_like(scores, dtype=torch.bool)
    torch.manual_seed(1)
    hot, hot_ids = relaxed_topk(scores, valid, 3, 1., 1.)
    torch.manual_seed(1)
    cold, cold_ids = relaxed_topk(scores, valid, 3, .1, 1.)
    assert torch.equal(hot_ids, cold_ids)  # Temperature cannot change hard ordering.
    assert hot_ids.unique(dim=0).shape[0] > 1
    assert torch.equal(hot.detach(), cold.detach())
    with pytest.raises(ValueError, match='eligible'):
        relaxed_topk(scores, valid, 9, 1.)
    options = dict(anneal_epochs=3, temperature=[1., .2], noise_scale=[1., 0.])
    assert schedule(options, 0) == (1., 1.)
    assert schedule(options, 2) == pytest.approx((.2, 0.))
    assert schedule(options, 100) == pytest.approx((.2, 0.))


@pytest.mark.parametrize('context', ['patch', 'transformer'])
def test_selector_activity_initialization_gradients_and_sparse_eval(cfg, context):
    with_selector(cfg, context)
    validate(cfg)
    view = batch(cfg)
    backbone = HierarchicalMAE(cfg)
    selected_encoder = SelectionEncoder(backbone, 'selector_train')
    model = Classifier(selected_encoder, 2, False)
    model.train()
    assert not backbone.training and all(not p.requires_grad for p in backbone.parameters())
    original = {k: v.clone() for k, v in backbone.state_dict().items()}
    scores = selected_encoder.selector(selected_encoder.root_features(view), view)
    torch.testing.assert_close(scores, view['log_counts'])
    before = selected_encoder.selector.scorer[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    nn_loss = torch.nn.functional.cross_entropy(model(view), torch.tensor([0, 1]))
    nn_loss.backward()
    assert selected_encoder.selector.scorer[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.parameters())
    optimizer.step()
    assert not torch.equal(before, selected_encoder.selector.scorer[-1].weight)
    for key in original:
        torch.testing.assert_close(backbone.state_dict()[key], original[key], rtol=0, atol=0)
    seen = []
    handle = backbone.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    model.eval()
    first, second = model(view), model(view)
    handle.remove()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert seen == ([6, 6] if context == 'patch' else [2, 6, 2, 6])
    assert selected_encoder.last_diagnostics['tokens_by_level'].sum() == 12
    assert selected_encoder.last_diagnostics['token_frequency'][:2].eq(2).all()


def test_selector_hard_training_matches_sparse_eval_without_noise(cfg):
    with_selector(cfg)
    cfg['downstream']['learned_selector']['noise_scale'] = [0., 0.]
    view = batch(cfg)
    model = Classifier(SelectionEncoder(HierarchicalMAE(cfg), 'selector_train'), 2, False)
    model.train()
    training = model(view)
    model.eval()
    evaluation = model(view)
    torch.testing.assert_close(training, evaluation, rtol=1e-5, atol=1e-6)


def test_selector_modes_resume_and_cached_probe(cfg):
    torch.set_num_threads(1)
    run(cfg)
    root = Path(cfg['train']['output_dir'])
    pretrained = root / 'best.pt'
    with_selector(cfg)
    cfg['downstream']['epochs'] = 2
    downstream_run(cfg, pretrained, 'selector_train', root / 'selector_full')
    expected = torch.load(root / 'selector_full/last.pt', weights_only=True)
    cfg['downstream']['epochs'] = 1
    downstream_run(cfg, pretrained, 'selector_train', root / 'selector_split')
    cfg['downstream']['epochs'] = 2
    downstream_run(cfg, resume=root / 'selector_split/last.pt')
    actual = torch.load(root / 'selector_split/last.pt', weights_only=True)
    for key in expected['model']:
        torch.testing.assert_close(actual['model'][key], expected['model'][key], rtol=0, atol=0)
    assert actual['validation'] == expected['validation']
    assert actual['validation']['selection']['tokens_by_level'][0] == 2
    cfg['downstream']['epochs'] = 1
    selector_checkpoint = root / 'selector_full/best.pt'
    for mode in ('selector_probe', 'selector_finetune'):
        result = downstream_run(cfg, selector_checkpoint, mode)
        assert result['test']['samples'] == 2 and result['selection']['budget'] == 6
    probe = torch.load(root / 'selector_probe/last.pt', weights_only=True)
    source = torch.load(selector_checkpoint, weights_only=True)
    for key, value in source['model'].items():
        if key.startswith('backbone.'):
            torch.testing.assert_close(probe['model'][key], value, rtol=0, atol=0)
    caches = list(Path(cfg['downstream']['feature_cache']['dir']).glob('train_*.pt'))
    assert len(caches) == 1
    cached = torch.load(caches[0], weights_only=True)
    assert cached['metadata']['learned_selector']['budget'] == 6
    assert cached['features'].shape == (6, cfg['model']['embed_dim'])
    cfg['downstream']['epochs'] = 2
    downstream_run(cfg, resume=root / 'selector_probe/last.pt')
    assert len(list(Path(cfg['downstream']['feature_cache']['dir']).glob('train_*.pt'))) == 1


@pytest.mark.parametrize('key,value', [('budget', 2), ('budget', 99), ('context', 'invalid'),
    ('temperature', [1., 0.]), ('noise_scale', [-1., 0.]), ('hidden_dim', 0), ('lr', float('nan'))])
def test_invalid_learned_selector_options(cfg, key, value):
    with_selector(cfg)
    cfg['downstream']['learned_selector'][key] = value
    with pytest.raises(ValueError, match='learned_selector'):
        validate(cfg)
