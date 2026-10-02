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
from ecfm.data.tokenizer import Region


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
