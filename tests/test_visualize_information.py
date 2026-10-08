"""Check rankings against training scores, cache invalidation and the HTTP API."""
from http.server import HTTPServer
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from experiments.hierarchical_mae.config import load_config
from experiments.hierarchical_mae.data import HierarchyDataset
from experiments.hierarchical_mae.information_selection import METRICS, selection_scores
from experiments.hierarchical_mae.visualize_information import InformationInspector, make_handler


@pytest.fixture
def inspector(tmp_path):
    cfg = load_config('experiments/hierarchical_mae/configs/smoke.yaml')
    cfg['data'].update(root=str(tmp_path), image_width=8, image_height=8, time_unit=1.,
                       crop_fraction=[.5, .9], spatial_crop_fraction=[1., 1.])
    cfg['data']['patch_cache'].update(enabled=True, train_views=2)
    rng = np.random.default_rng(9)
    for i in range(8):
        np.save(tmp_path / f'{i}.npy', np.column_stack((rng.integers(0, 8, (70, 2)),
                 np.linspace(0, 2, 70), rng.integers(0, 2, 70))))
    (tmp_path / 'train.txt').write_text(''.join(f'{i}.npy {i % 2}\n' for i in range(8)))
    return InformationInspector(cfg)


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('mode', ['information', 'blend', 'product'])
def test_ranking_matches_training_and_group_normalization(inspector, metric, mode):
    result = inspector.inspect(dict(metric=metric, mode=mode, group='l1_xy', activity_weight=.3))
    view = default_collate([HierarchyDataset(inspector.cfg, inspector.entries)[0]])['source']
    eligible = torch.zeros_like(view['valid_mask'])
    eligible[:, 2:10] = True
    scores = selection_scores(view, eligible, dict(metric=metric, combination=mode,
        strategy='information' if mode == 'information' else 'activity_information', activity_weight=.3))[0]
    expected = sorted(range(2, 10), key=lambda i: (-float(scores[i]), i))
    assert [t['id'] for t in result['tokens']] == expected
    assert [t['score'] for t in result['tokens']] == pytest.approx(scores[expected].tolist())
    assert all(t['image'].startswith('data:image/png;base64,') for t in result['tokens'])
    assert inspector.cfg['data']['patch_cache']['enabled'] is False


def test_cache_reuse_rebinning_and_training_crop_bank(inspector, monkeypatch):
    first = inspector.inspect({})
    cached = inspector.cached_view
    with monkeypatch.context() as patch:
        def fail(*args):
            raise AssertionError('Changing ranking should not load or render events')
        patch.setattr(HierarchyDataset, '__getitem__', fail)
        inspector.inspect(dict(metric='entropy', mode='blend', activity_weight=.8))
        inspector.inspect(dict(group='l1_xy'))
    assert inspector.cached_view is cached
    changed = inspector.inspect(dict(bins=[2, 3, 4], support_saturation=10))
    assert inspector.cached_view is not cached
    assert any(a['metrics'] != b['metrics'] for a, b in zip(
        sorted(first['tokens'], key=lambda t: t['id']), sorted(changed['tokens'], key=lambda t: t['id'])))
    a = inspector.inspect(dict(crop='training', epoch=0))
    b = inspector.inspect(dict(crop='training', epoch=2))
    assert a['tokens'] == b['tokens']
    assert inspector.inspect(dict(instance=1))['instance'] == 1


def test_ties_alternate_representations_and_saturation(inspector):
    a = {t['id']: t for t in inspector.inspect({})['tokens']}
    assert a[0]['metrics'] == a[1]['metrics']
    result = inspector.inspect(dict(support_saturation=10000))
    for t in result['tokens']:
        assert t['metrics']['support'] <= a[t['id']]['metrics']['support']
        assert t['metrics']['entropy'] == a[t['id']]['metrics']['entropy']
        assert t['metrics']['autocorrelation'] == a[t['id']]['metrics']['autocorrelation']
    for left, right in zip(result['tokens'], result['tokens'][1:]):
        assert left['score'] > right['score'] or left['id'] < right['id']


@pytest.mark.parametrize('payload', [dict(instance=-1), dict(instance=True), dict(bins=[1, 8, 8]),
    dict(bins=[65, 8, 8]), dict(bins=[8, 8]), dict(support_saturation=0),
    dict(support_saturation=float('nan')), dict(metric='other'), dict(mode='other'),
    dict(activity_weight=2), dict(group='missing'), dict(epoch=-1), dict(crop='other')])
def test_invalid_requests(inspector, payload):
    with pytest.raises(ValueError):
        inspector.inspect(payload)


def test_http_ui_and_errors(inspector):
    server = HTTPServer(('127.0.0.1', 0), make_handler(inspector))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url) as response:
            assert b'Patch information explorer' in response.read()
        with urlopen(url + '/api/info') as response:
            assert json.load(response)['tokens'] == 18
        with urlopen(Request(url + '/api/instance', data=b'{"metric":"entropy"}')) as response:
            assert len(json.load(response)['tokens']) == 18
        for payload in (b'{', b'[]', b'{"instance":999}', b'{"bins":[2,2,1000000]}'):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(url + '/api/instance', data=payload))
            assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(url + '/missing')
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
