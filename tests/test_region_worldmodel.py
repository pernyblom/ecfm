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
